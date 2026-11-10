//! Mode switches code 5 ms CELT redundancy like libopus.
//!
//! Entering or leaving CELT-only mode, libopus (`opus_encode_native`) adds a
//! 5 ms CELT frame of all bands: at the start of the first SILK or hybrid
//! frame after CELT, and, going to CELT, at the end of one more SILK or
//! hybrid frame, from which the CELT frames continue. The tests switch the
//! bitrate (and the loss rate) every 400 ms, encode the fixture with opus-rs
//! and with libopus, and decode both with libopus: the modes must follow
//! libopus frame by frame, and the audio around the switches must be as good.

#[path = "common/vbr.rs"]
mod vbr;

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Decoder as CDec,
    Encoder as CEnc, Signal as CSignal,
};
use opus_rs::range_coder::RangeCoder;
use opus_rs::silk::dec_api::SilkDecoder;
use opus_rs::{Application, OpusEncoder};

const PHASE_MS: usize = 400;

/// Rate (b/s) and loss (%) of the frame at `ms` milliseconds: the low phase
/// first, then the high one.
type Schedule = fn(usize) -> (i32, i32);

/// Hybrid (or SILK-only) at 20 kb/s, CELT-only at 100 kb/s.
fn hybrid_celt(ms: usize) -> (i32, i32) {
    if (ms / PHASE_MS).is_multiple_of(2) {
        (20000, 0)
    } else {
        (100000, 0)
    }
}

/// FEC at 30 % loss narrows 20 kb/s to SILK-only; 100 kb/s without loss is
/// CELT-only.
fn silk_celt(ms: usize) -> (i32, i32) {
    if (ms / PHASE_MS).is_multiple_of(2) {
        (20000, 30)
    } else {
        (100000, 0)
    }
}

/// Hybrid at 20 kb/s without loss; FEC at 30 % loss narrows it to
/// SILK-only.
fn hybrid_silk(ms: usize) -> (i32, i32) {
    if (ms / PHASE_MS).is_multiple_of(2) {
        (20000, 0)
    } else {
        (20000, 30)
    }
}

fn mode(packet: &[u8]) -> char {
    match packet[0] >> 3 {
        0..=11 => 'S',
        12..=15 => 'H',
        _ => 'C',
    }
}

struct Stream {
    packets: Vec<Vec<u8>>,
    decoded: Vec<f32>,
}

fn decode(packets: Vec<Vec<u8>>, sr: usize, channels: usize, frame: usize) -> Stream {
    let c_channels = if channels == 1 {
        CCh::Mono
    } else {
        CCh::Stereo
    };
    let mut decoder = CDec::new(sr as u32, c_channels).unwrap();
    let mut out = vec![0.0; frame * channels];
    let mut decoded = Vec::new();
    for (p, packet) in packets.iter().enumerate() {
        let n = decoder
            .decode_float(packet, &mut out, false)
            .unwrap_or_else(|e| panic!("packet {p}: libopus cannot decode it: {e:?}"));
        assert_eq!(n, frame, "packet {p}");
        decoded.extend_from_slice(&out);
    }
    Stream { packets, decoded }
}

fn encode(
    sr: usize,
    ms: usize,
    channels: usize,
    fec: bool,
    schedule: Schedule,
    input: &[f32],
) -> (Stream, Stream) {
    let frame = sr * ms / 1000;
    let frames = input.len() / (frame * channels);
    let mut rs = OpusEncoder::new(sr as i32, channels, Application::Voip).unwrap();
    rs.complexity = 9;
    rs.use_cbr = false;
    rs.use_inband_fec = fec;
    let c_channels = if channels == 1 {
        CCh::Mono
    } else {
        CCh::Stereo
    };
    let mut c = CEnc::new(sr as u32, c_channels, CApp::Voip).unwrap();
    let band = if sr == 24000 {
        CBw::Superwideband
    } else {
        CBw::Fullband
    };
    c.set_bandwidth(band).unwrap();
    c.set_complexity(9).unwrap();
    c.set_vbr(true).unwrap();
    c.set_signal(CSignal::Voice).unwrap();
    c.set_force_channels(Some(c_channels)).unwrap();
    c.set_inband_fec(fec).unwrap();
    let mut buffer = [0u8; 1500];
    let (mut ours, mut theirs) = (Vec::new(), Vec::new());
    for p in 0..frames {
        let (rate, loss) = schedule(p * ms);
        rs.bitrate_bps = rate;
        rs.packet_loss_perc = loss;
        c.set_bitrate(CBitrate::Bits(rate)).unwrap();
        c.set_packet_loss_perc(loss).unwrap();
        let pcm = &input[p * frame * channels..(p + 1) * frame * channels];
        let n = rs.encode(pcm, frame, &mut buffer).unwrap();
        ours.push(buffer[..n].to_vec());
        let n = c.encode_float(pcm, &mut buffer).unwrap();
        theirs.push(buffer[..n].to_vec());
    }
    (
        decode(ours, sr, channels, frame),
        decode(theirs, sr, channels, frame),
    )
}

/// SNR (dB) of `output` against `input`, delayed by `lag` samples, over the
/// samples from 20 ms before to 60 ms after each switch.
fn switch_snr(input: &[f32], output: &[f32], sr: usize, channels: usize, lag: usize) -> f64 {
    let phase = sr * PHASE_MS / 1000;
    let (mut signal, mut noise) = (0.0, 0.0);
    for switch in (phase..input.len() / channels - sr / 10).step_by(phase) {
        for i in switch - sr / 50..switch + 3 * sr / 50 {
            for c in 0..channels {
                let x = f64::from(input[i * channels + c]);
                let y = f64::from(output[(i + lag) * channels + c]);
                signal += x * x;
                noise += (x - y) * (x - y);
            }
        }
    }
    10.0 * (signal / noise).log10()
}

fn check(sr: usize, ms: usize, channels: usize, fec: bool, schedule: Schedule) {
    let frame = sr * ms / 1000;
    let input = vbr::speech(sr, channels, frame * (6000 / ms));
    let (ours, theirs) = encode(sr, ms, channels, fec, schedule, &input);
    let modes = |s: &Stream| s.packets.iter().map(|p| mode(p)).collect::<String>();
    assert_eq!(
        modes(&ours),
        modes(&theirs),
        "{sr} Hz {ms} ms {channels} ch"
    );
    assert!(modes(&ours).contains('C') && modes(&ours).contains(if fec { 'S' } else { 'H' }));
    // Both decode with the encoders' lookahead: 2.5 ms of CELT overlap and
    // 4 ms of delay compensation.
    let lag = sr / 400 + sr / 250;
    let (a, b) = (
        switch_snr(&input, &ours.decoded, sr, channels, lag),
        switch_snr(&input, &theirs.decoded, sr, channels, lag),
    );
    assert!(
        a > b - 0.5,
        "{sr} Hz {ms} ms {channels} ch: {a:.2} dB around the switches, libopus {b:.2} dB"
    );
}

#[test]
fn hybrid_celt_switches_follow_libopus() {
    check(24000, 20, 1, false, hybrid_celt);
    check(48000, 10, 2, false, hybrid_celt);
}

#[test]
fn multiframe_hybrid_celt_switches_follow_libopus() {
    // 40 ms CELT and hybrid packets are two 20 ms frames: going to CELT, the
    // second frame carries the redundancy.
    check(48000, 40, 1, false, hybrid_celt);
}

#[test]
fn silk_only_celt_switches_follow_libopus() {
    check(24000, 20, 1, true, silk_celt);
    check(48000, 10, 1, true, silk_celt);
}

fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}

/// The fixture at `sr` in 16 bits, the same input fixed-point libopus coded.
fn speech_i16(sr: usize, channels: usize) -> Vec<i16> {
    let source: Vec<f32> = vbr::speech_pcm()
        .iter()
        .map(|&x| x as f32 / 32768.0)
        .collect();
    (0..source.len() * sr / 16000)
        .flat_map(|i| {
            let pos = i * 16000;
            let at = pos / sr;
            let fraction = (pos % sr) as f32 / sr as f32;
            let x = source[at] + fraction * (source[(at + 1).min(source.len() - 1)] - source[at]);
            let x = (x * 32768.0).clamp(-32768.0, 32767.0).round_ties_even() as i16;
            std::iter::repeat_n(x, channels)
        })
        .collect()
}

/// A frame's flags after its SILK layer (RFC 6716 section 4.5.1):
/// redundancy, CELT-to-SILK, and for hybrid frames, the intra flag of the
/// CELT energy.
#[derive(Debug, Default, PartialEq)]
struct Flags {
    redundancy: bool,
    celt_to_silk: bool,
    intra: bool,
}

/// The frames of a packet without padding (RFC 6716 section 3.2).
fn frames(p: &[u8]) -> Vec<&[u8]> {
    let size = |d: &[u8]| {
        if d[0] < 252 {
            (usize::from(d[0]), 1)
        } else {
            (usize::from(d[0]) + 4 * usize::from(d[1]), 2)
        }
    };
    match p[0] & 3 {
        0 => vec![&p[1..]],
        1 => p[1..].chunks((p.len() - 1) / 2).collect(),
        2 => {
            let (n, k) = size(&p[1..]);
            vec![&p[1 + k..1 + k + n], &p[1 + k + n..]]
        }
        _ => {
            let count = usize::from(p[1] & 63);
            assert_eq!(p[1] & 0xc0, 0x80, "VBR without padding");
            let (mut lens, mut at) = (Vec::new(), 2);
            for _ in 0..count - 1 {
                let (n, k) = size(&p[at..]);
                lens.push(n);
                at += k;
            }
            let mut out = Vec::new();
            for n in lens {
                out.push(&p[at..at + n]);
                at += n;
            }
            out.push(&p[at..]);
            out
        }
    }
}

/// Decode the SILK layer of every SILK-only and hybrid frame, the SILK
/// decoder reset after CELT, as libopus decodes them: return a hash of the
/// bits each takes and the audio it decodes to, and per frame of every
/// packet its mode and the flags after its SILK layer.
fn silk_layers(packets: &[Vec<u8>], sr: i32, channels: i32) -> (u64, Vec<(char, Option<Flags>)>) {
    let mut silk = SilkDecoder::new();
    silk.init(sr.min(16000), channels);
    let (mut hash, mut after_celt, mut flags) = (0xcbf29ce484222325, false, Vec::new());
    let mut out = [0i16; 2 * 960];
    for p in packets {
        let config = usize::from(p[0] >> 3);
        for frame in frames(p) {
            if config >= 16 {
                after_celt = true;
                flags.push(('C', None));
                continue;
            }
            if after_celt {
                silk.reset();
                after_celt = false;
            }
            let hybrid = config >= 12;
            let (rate, ms) = if hybrid {
                (16000, [10, 20][config % 2])
            } else {
                (
                    [8000, 12000, 16000][config / 4],
                    [10, 20, 40, 60][config % 4],
                )
            };
            let coded = if p[0] & 4 != 0 { 2 } else { 1 };
            let mut rc = RangeCoder::new_decoder(frame);
            // SILK frames are 10 or 20 ms; a longer one holds two or three.
            let (mut decoded, total) = (0, (ms * rate / 1000) as usize);
            while decoded < total {
                let n =
                    silk.decode_with_channels(&mut rc, &mut out, 0, decoded == 0, ms, rate, coded);
                assert!(n > 0);
                hash = fnv(hash, &rc.tell().to_le_bytes());
                for s in &out[..n as usize * coded.min(channels) as usize] {
                    hash = fnv(hash, &s.to_le_bytes());
                }
                decoded += n as usize;
            }
            let mut f = Flags::default();
            let mut bits = 8 * frame.len() as i32;
            if rc.tell() + 17 + 20 * i32::from(hybrid) <= bits {
                f.redundancy = !hybrid || rc.decode_bit_logp(12);
                if f.redundancy {
                    f.celt_to_silk = rc.decode_bit_logp(1);
                    let size = if hybrid {
                        rc.dec_uint(256) as i32 + 2
                    } else {
                        frame.len() as i32 - ((rc.tell() + 7) >> 3)
                    };
                    bits -= 8 * size;
                    f.redundancy = size > 0;
                }
            }
            // Hybrid CELT starts at band 17: silence, transient (frames of 5
            // ms or more), then the intra flag.
            if hybrid && rc.tell() + 15 <= bits && !rc.decode_bit_logp(15) {
                rc.decode_bit_logp(3);
                f.intra = rc.tell() + 3 <= bits && rc.decode_bit_logp(3);
            }
            flags.push((if hybrid { 'H' } else { 'S' }, Some(f)));
        }
    }
    (hash, flags)
}

#[test]
fn silk_layer_follows_fixed_point_libopus_across_switches() {
    // Whole encodes of the fixture with 16-bit input: the SILK layer of
    // every packet equals fixed-point libopus 1.6.1's, through SILK's restart
    // after CELT (high-pass state and history kept in CELT frames, the second
    // control call after the prefill), its rate change into hybrid mode (the
    // history moved through the new resampler, the shaping state reset) and
    // FEC turning off (the previous packet's LBRR data still coded).
    let cases: [(Schedule, bool, usize, usize, usize, u64, usize); 5] = [
        (silk_celt, true, 24000, 20, 1, 0xb0223bd54b9a2d7e, 483),
        (silk_celt, true, 48000, 10, 1, 0xadd9c7509725d7f3, 943),
        (silk_celt, true, 24000, 20, 2, 0x0890ebabb7ad3e52, 483),
        (hybrid_silk, true, 24000, 20, 1, 0x55818a04b86bb9cf, 902),
        (hybrid_celt, false, 24000, 20, 1, 0x88324b53f7859128, 483),
    ];
    for (schedule, fec, sr, ms, channels, expected, layers) in cases {
        let input = speech_i16(sr, channels);
        let frame = sr * ms / 1000;
        let mut enc = OpusEncoder::new(sr as i32, channels, Application::Voip).unwrap();
        enc.complexity = 9;
        enc.use_cbr = false;
        enc.use_inband_fec = fec;
        let mut buffer = [0u8; 1500];
        let packets: Vec<Vec<u8>> = (0..input.len() / (frame * channels))
            .map(|p| {
                (enc.bitrate_bps, enc.packet_loss_perc) = schedule(p * ms);
                let pcm = &input[p * frame * channels..(p + 1) * frame * channels];
                let n = enc.encode_i16(pcm, frame, &mut buffer).unwrap();
                buffer[..n].to_vec()
            })
            .collect();
        let (hash, flags) = silk_layers(&packets, sr as i32, channels as i32);
        assert_eq!(
            (hash, flags.iter().filter(|(m, _)| *m != 'C').count()),
            (expected, layers),
            "{sr} Hz {ms} ms {channels} ch"
        );
    }
}

#[test]
fn switch_frames_signal_redundancy_and_restart_celt() {
    // Entering SILK or hybrid from CELT, the frame starts with CELT-to-SILK
    // redundancy; leaving for CELT, one more SILK or hybrid frame ends with
    // SILK-to-CELT redundancy, in a 40 ms packet the second frame. A hybrid
    // frame after CELT or SILK-only restarts CELT, whose energy is then coded
    // intra (no prediction).
    for (schedule, fec, ms) in [
        (hybrid_celt as Schedule, false, 20),
        (hybrid_celt, false, 40),
        (silk_celt, true, 20),
        (hybrid_silk, true, 20),
    ] {
        let sr = 24000;
        let frame = sr * ms / 1000;
        let packets = 4000 / ms;
        let input = vbr::speech(sr, 1, frame * packets);
        let mut enc = OpusEncoder::new(sr as i32, 1, Application::Voip).unwrap();
        enc.complexity = 9;
        enc.use_inband_fec = fec;
        let mut buffer = [0u8; 1500];
        let packets: Vec<Vec<u8>> = (0..packets)
            .map(|p| {
                (enc.bitrate_bps, enc.packet_loss_perc) = schedule(p * ms);
                let n = enc
                    .encode(&input[p * frame..(p + 1) * frame], frame, &mut buffer)
                    .unwrap();
                buffer[..n].to_vec()
            })
            .collect();
        let (_, flags) = silk_layers(&packets, sr as i32, 1);
        let modes: String = flags.iter().map(|(m, _)| m).collect();
        let mut switches = 0;
        for i in 1..flags.len() - 1 {
            let ((before, _), (this, Some(f)), (after, _)) =
                (&flags[i - 1], &flags[i], &flags[i + 1])
            else {
                continue;
            };
            let at = format!("{modes} frame {i}: {f:?}");
            if *before == 'C' {
                assert!(f.redundancy && f.celt_to_silk, "{at}");
                assert!(*this == 'S' || f.intra, "{at}");
                switches += 1;
            } else if *after == 'C' {
                assert!(f.redundancy && !f.celt_to_silk, "{at}");
                switches += 1;
            } else if *before == 'S' && *this == 'H' {
                assert!(f.intra && !f.redundancy, "{at}");
                switches += 1;
            } else if *before == 'H' && *this == 'H' {
                assert!(!f.redundancy, "{at}");
            }
        }
        assert!(switches >= 4, "{modes}");
    }
}
