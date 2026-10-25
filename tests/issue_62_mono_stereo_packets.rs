//! Issue #62 (item 3) regression: a packet whose channel count differs from
//! the decoder's must decode like libopus.
//!
//! `OpusDecoder::decode` rejected such packets with "Channel count mismatch
//! between packet and decoder". libopus plays a mono packet on both channels
//! of a stereo decoder and downmixes a stereo packet in a mono one, and its
//! stereo encoders send mono packets at low bitrates. Channel switches
//! mid-stream have their own rules (`silk/dec_API.c`): a stream turning stereo
//! starts the side channel and the right resampler afresh, and the first mono
//! frame after stereo still resamples the right channel with its own.
//!
//! These tests decode libopus stereo streams whose packets are mono, stereo,
//! or switch between the two, with opus-rs and libopus, and require agreement
//! at lag 0. Before the fix every test failed on the first mismatched packet.

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Decoder as CDec,
    Encoder as CEnc, FrameSize, Signal as CSignal,
};
use opus_rs::OpusDecoder;
use std::f32::consts::PI;
use std::ops::RangeInclusive;

/// Largest Opus frame per channel: 120 ms at 48 kHz.
const MAX_FRAME: usize = 5760;

/// The encoders' input rate.
const SR: usize = 48000;

/// Voiced-speech-like stereo test signal (interleaved): a 140 Hz harmonic
/// series with slow pitch drift and a syllable-rate envelope. The right
/// channel is a scaled, phase-shifted copy so the side signal is not zero.
fn voiced(sr: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * 2);
    let mut phase = 0.0f32;
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let f0 = 140.0 + 15.0 * (2.0 * PI * 0.7 * t).sin();
        phase += 2.0 * PI * f0 / sr as f32;
        let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin();
        let mut l = 0.0;
        let mut r = 0.0;
        for h in 1..=8 {
            let a = 0.25 / h as f32;
            l += a * (h as f32 * phase).sin();
            r += a * (h as f32 * phase + 0.3 * h as f32).sin();
        }
        out.push(l * env * 0.5);
        out.push(r * env * 0.4);
    }
    out
}

/// Broadband stereo test signal (interleaved): a 180 Hz harmonic series
/// under 100 ms noise bursts every 300 ms, so every CELT band carries energy.
/// The right channel is the left, scaled.
fn broadband(sr: usize, n: usize) -> Vec<f32> {
    let mut seed = 0x2468_ace1u32;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let tone: f32 = (1..=12)
            .map(|h| (2.0 * PI * 180.0 * h as f32 * t).sin() * 0.3 / h as f32)
            .sum();
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = ((seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * 2.0;
        let burst = if (i / (sr / 10)).is_multiple_of(3) {
            0.3
        } else {
            0.02
        };
        let l = tone + noise * burst;
        out.push(l);
        out.push(0.8 * l);
    }
    out
}

/// A libopus coding mode, pinned so that switching channels can't move it.
struct Mode {
    name: &'static str,
    app: CApp,
    bandwidth: CBw,
    bitrate: i32,
    ms: usize,
    /// The ToC configurations every packet must carry.
    configs: RangeInclusive<u8>,
    /// Worst 20 ms window SNR (dB) against libopus's decode.
    floor: f64,
    /// The stereo input, from the rate and the samples per channel.
    signal: fn(usize, usize) -> Vec<f32>,
}

/// SILK decodes bit-exactly with libopus, so its bar sits far above any real
/// mismatch.
const SILK_FLOOR: f64 = 100.0;

/// With matching channel counts, opus-rs's CELT decode agrees with libopus's
/// to 45.3-51.4 dB on these streams (stereo CELT sets the low end), and
/// Hybrid's at 48 kHz to 55.7-60.1 dB (bit-exact at 16 kHz, where only the
/// SILK layer is heard). The bars leave margin below that.
const CELT_FLOOR: f64 = 40.0;
const HYBRID_FLOOR: f64 = 45.0;

const SILK_WB: Mode = Mode {
    name: "SILK WB 20 ms",
    app: CApp::Voip,
    bandwidth: CBw::Wideband,
    bitrate: 24000,
    ms: 20,
    configs: 8..=11,
    floor: SILK_FLOOR,
    signal: voiced,
};

/// Three SILK frames per packet: a switch lands on the packet's first frame.
const SILK_WB_60: Mode = Mode {
    name: "SILK WB 60 ms",
    app: CApp::Voip,
    bandwidth: CBw::Wideband,
    bitrate: 24000,
    ms: 60,
    configs: 8..=11,
    floor: SILK_FLOOR,
    signal: voiced,
};

const HYBRID_FB: Mode = Mode {
    name: "Hybrid FB 20 ms",
    app: CApp::Voip,
    bandwidth: CBw::Fullband,
    bitrate: 40000,
    ms: 20,
    configs: 14..=15,
    floor: HYBRID_FLOOR,
    signal: voiced,
};

const CELT_FB: Mode = Mode {
    name: "CELT FB 20 ms",
    app: CApp::LowDelay,
    bandwidth: CBw::Fullband,
    bitrate: 64000,
    ms: 20,
    configs: 28..=31,
    floor: CELT_FLOOR,
    signal: voiced,
};

const MODES: [&Mode; 4] = [&SILK_WB, &SILK_WB_60, &HYBRID_FB, &CELT_FB];

/// CELT at whatever bandwidth the packet asks for. A broadband signal: the
/// voiced one barely reaches the upper bands, whose energies are at stake.
const CELT_ANY: Mode = Mode {
    name: "CELT, bandwidth steps",
    app: CApp::LowDelay,
    bandwidth: CBw::Fullband,
    bitrate: 64000,
    ms: 20,
    configs: 16..=31,
    floor: CELT_FLOOR,
    signal: broadband,
};

fn frame_duration(ms: usize) -> FrameSize {
    match ms {
        20 => FrameSize::Ms20,
        40 => FrameSize::Ms40,
        _ => FrameSize::Ms60,
    }
}

fn is_stereo(pkt: &[u8]) -> bool {
    pkt[0] & 0x04 != 0
}

/// 2 s of libopus stereo-encoded packets in `mode`. `plan(p)` gives packet
/// `p`'s forced channel count (`None` lets libopus choose), bitrate and
/// bandwidth.
fn libopus_packets(mode: &Mode, plan: impl Fn(usize) -> (Option<CCh>, i32, CBw)) -> Vec<Vec<u8>> {
    let frame = SR * mode.ms / 1000;
    let input = (mode.signal)(SR, 2 * SR);
    let mut enc = CEnc::new(SR as u32, CCh::Stereo, mode.app).unwrap();
    // A steady harmonic tone can read as music and pull libopus into CELT.
    enc.set_signal(CSignal::Voice).unwrap();
    enc.set_complexity(9).unwrap();
    enc.set_expert_frame_duration(frame_duration(mode.ms))
        .unwrap();
    let mut pkt = vec![0u8; 1500];
    input
        .chunks_exact(frame * 2)
        .enumerate()
        .map(|(p, f)| {
            let (channels, bitrate, bandwidth) = plan(p);
            enc.set_force_channels(channels).unwrap();
            enc.set_bitrate(CBitrate::Bits(bitrate)).unwrap();
            enc.set_bandwidth(bandwidth).unwrap();
            let n = enc.encode_float(f, &mut pkt).unwrap();
            let config = pkt[0] >> 3;
            assert!(
                mode.configs.contains(&config),
                "{}: packet {p} has ToC config {config}",
                mode.name
            );
            pkt[..n].to_vec()
        })
        .collect()
}

/// Number of times the ToC's channel count changes along the stream.
fn channel_switches(packets: &[Vec<u8>]) -> usize {
    packets
        .windows(2)
        .filter(|w| is_stereo(&w[0]) != is_stereo(&w[1]))
        .count()
}

/// opus-rs's `ch`-channel decode at `rate`; packets listed in `lost` are
/// concealed instead.
fn rs_decode(packets: &[Vec<u8>], ms: usize, ch: usize, rate: usize, lost: &[usize]) -> Vec<f32> {
    let frame = rate * ms / 1000;
    let mut dec = OpusDecoder::new(rate as i32, ch).unwrap();
    let mut buf = vec![0f32; MAX_FRAME * ch];
    let mut out = Vec::new();
    for (p, pkt) in packets.iter().enumerate() {
        let input: &[u8] = if lost.contains(&p) { &[] } else { pkt };
        let n = dec
            .decode(input, frame, &mut buf)
            .unwrap_or_else(|e| panic!("packet {p}: opus-rs decode failed: {e}"));
        assert_eq!(n, frame, "packet {p}: opus-rs decoded length");
        out.extend_from_slice(&buf[..n * ch]);
    }
    out
}

/// libopus's `ch`-channel decode at `rate`.
fn c_decode(packets: &[Vec<u8>], ms: usize, ch: usize, rate: usize) -> Vec<f32> {
    let channels = if ch == 2 { CCh::Stereo } else { CCh::Mono };
    let frame = rate * ms / 1000;
    let mut dec = CDec::new(rate as u32, channels).unwrap();
    let mut buf = vec![0f32; MAX_FRAME * ch];
    let mut out = Vec::new();
    for (p, pkt) in packets.iter().enumerate() {
        let n = dec.decode_float(pkt, &mut buf, false).unwrap();
        assert_eq!(n, frame, "packet {p}: libopus decoded length");
        out.extend_from_slice(&buf[..n * ch]);
    }
    out
}

/// Worst lag-0 SNR (dB) of `got` against `want` over 20 ms windows and
/// channels, after skipping the first `skip` samples per channel.
fn worst_window_snr(want: &[f32], got: &[f32], ch: usize, rate: usize, skip: usize) -> f64 {
    assert_eq!(want.len(), got.len(), "decoded lengths differ");
    let win = rate / 50;
    let mut worst = f64::INFINITY;
    for c in 0..ch {
        let mut start = skip;
        while start + win <= want.len() / ch {
            let (mut sig, mut err) = (0f64, 0f64);
            for i in start..start + win {
                let r = want[i * ch + c] as f64;
                let e = r - got[i * ch + c] as f64;
                sig += r * r;
                err += e * e;
            }
            let snr = if err == 0.0 {
                f64::INFINITY
            } else {
                10.0 * (sig / err).log10()
            };
            worst = worst.min(snr);
            start += win;
        }
    }
    worst
}

/// Decode `packets` with opus-rs and libopus, `ch` channels at `rate`, and
/// require the worst window to reach `mode.floor`.
fn assert_agrees_with_libopus(
    mode: &Mode,
    what: &str,
    packets: &[Vec<u8>],
    ch: usize,
    rate: usize,
) {
    let want = c_decode(packets, mode.ms, ch, rate);
    let got = rs_decode(packets, mode.ms, ch, rate, &[]);
    let worst = worst_window_snr(&want, &got, ch, rate, rate * mode.ms / 1000);
    eprintln!(
        "{}, {what}, {ch} ch at {rate} Hz: worst window {worst:.1} dB",
        mode.name
    );
    assert!(
        worst >= mode.floor,
        "{}, {what}, {ch} ch at {rate} Hz: worst window {worst:.1} dB < {} dB",
        mode.name,
        mode.floor
    );
}

/// A stereo decoder plays mono packets on both channels, sample for sample
/// what a mono decoder produces outside a concealed loss, and agrees with
/// libopus.
#[test]
fn stereo_decoder_plays_mono_packets_on_both_channels() {
    for mode in MODES {
        let packets = libopus_packets(mode, |_| (Some(CCh::Mono), mode.bitrate, mode.bandwidth));
        assert!(
            packets.iter().all(|p| !is_stereo(p)),
            "{}: stereo ToC",
            mode.name
        );
        for rate in [16000, 48000] {
            let lost = [packets.len() / 2];
            let stereo = rs_decode(&packets, mode.ms, 2, rate, &lost);
            let mono = rs_decode(&packets, mode.ms, 1, rate, &lost);
            // libopus 1.6's float CELT concealment is chaotic at octave
            // near-ties: the PLC pitch search of a stereo decoder (whose
            // whitened correlation input is the sum of both channels) and
            // of a mono decoder can make different period-vs-twice choices
            // for a few frames after a concealed loss, until the tiny
            // history differences the different choice leaves behind decay
            // with the signal. libopus 1.6.1's own x86 and aarch64 builds
            // agree only to ~74 dB on such frames. Equality of the two
            // decoders is asserted once that influence has decayed
            // (measured: within 20 frames of the loss).
            let frame = rate * mode.ms / 1000;
            let concealed = lost[0] * frame..(lost[0] + 20) * frame;
            for (i, m) in mono.iter().enumerate() {
                if concealed.contains(&i) {
                    continue;
                }
                for (c, s) in stereo[2 * i..2 * i + 2].iter().enumerate() {
                    assert_eq!(
                        s.to_bits(),
                        m.to_bits(),
                        "{} at {rate} Hz: channel {c} differs from mono at sample {i}",
                        mode.name
                    );
                }
            }
            assert_agrees_with_libopus(mode, "mono packets", &packets, 2, rate);
        }
    }
}

/// A mono decoder downmixes stereo packets as libopus does: SILK plays the
/// mid, CELT averages the channels' spectra.
#[test]
fn mono_decoder_downmixes_stereo_packets() {
    for mode in MODES {
        let packets = libopus_packets(mode, |_| (Some(CCh::Stereo), mode.bitrate, mode.bandwidth));
        assert!(
            packets.iter().all(|p| is_stereo(p)),
            "{}: mono ToC",
            mode.name
        );
        for rate in [16000, 48000] {
            assert_agrees_with_libopus(mode, "stereo packets", &packets, 1, rate);
        }
    }
}

/// Packets that switch between stereo and mono every 200 ms decode like
/// libopus in both decoders.
#[test]
fn channel_switches_decode_like_libopus() {
    for mode in MODES {
        let per_switch = 200 / mode.ms;
        let packets = libopus_packets(mode, |p| {
            let channels = if (p / per_switch) % 2 == 0 {
                CCh::Stereo
            } else {
                CCh::Mono
            };
            (Some(channels), mode.bitrate, mode.bandwidth)
        });
        let switches = channel_switches(&packets);
        assert!(switches >= 8, "{}: {switches} channel switches", mode.name);
        for rate in [16000, 48000] {
            for ch in [2, 1] {
                assert_agrees_with_libopus(mode, "switching packets", &packets, ch, rate);
            }
        }
    }
}

/// The issue's case: libopus left to choose, a stereo stream whose bitrate
/// drops to 12 kb/s turns mono and back.
#[test]
fn low_bitrate_stereo_stream_from_libopus() {
    let packets = libopus_packets(&SILK_WB, |p| {
        let bitrate = if (30..70).contains(&p) { 12000 } else { 32000 };
        (None, bitrate, SILK_WB.bandwidth)
    });
    let mono = packets.iter().filter(|p| !is_stereo(p)).count();
    assert!(mono > 0, "libopus sent no mono packet at 12 kb/s");
    assert!(channel_switches(&packets) >= 2, "no switch back to stereo");
    for rate in [16000, 48000] {
        for ch in [2, 1] {
            assert_agrees_with_libopus(&SILK_WB, "12 kb/s dip", &packets, ch, rate);
        }
    }
}

/// CELT whose bandwidth steps every 100 ms, so each frame codes a different
/// band range. A frame clears the energies of the bands it doesn't code, in
/// both halves as libopus does (`celt_decoder.c`, from PR #59). A mono stream
/// keeps the second half, and its next frame predicts from the larger of the
/// two. Clearing only the coded channel's half left stale energies there, and
/// the next wider frame decoded its new bands from them: noise, at -0.9 dB.
#[test]
fn celt_bandwidth_steps_decode_like_libopus() {
    // Each bandwidth and its first CELT ToC configuration.
    const STEPS: [(CBw, u8); 4] = [
        (CBw::Fullband, 28),
        (CBw::Wideband, 20),
        (CBw::Superwideband, 24),
        (CBw::Narrowband, 16),
    ];
    for (channels, what) in [(CCh::Mono, "mono stream"), (CCh::Stereo, "stereo stream")] {
        let packets = libopus_packets(&CELT_ANY, |p| {
            (Some(channels), CELT_ANY.bitrate, STEPS[(p / 5) % 4].0)
        });
        for (p, pkt) in packets.iter().enumerate() {
            let first = STEPS[(p / 5) % 4].1;
            let config = pkt[0] >> 3;
            assert!(
                (first..first + 4).contains(&config),
                "{what}: packet {p} has ToC config {config}"
            );
        }
        for rate in [16000, 48000] {
            for ch in [1, 2] {
                assert_agrees_with_libopus(&CELT_ANY, what, &packets, ch, rate);
            }
        }
    }
}
