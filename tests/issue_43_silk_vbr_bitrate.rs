//! Issue #43: SILK-only VBR must code at `bitrate_bps`, not at the size of
//! the output buffer.
//!
//! opus-rs used to derive the SILK target rate from the packet budget, which
//! under VBR is the caller's output buffer (capped at 1276 bytes): a 1500-byte
//! buffer asked SILK for ~510 kb/s, `silk_control_snr` saturated, and every
//! bitrate from 12 to 64 kb/s coded the same ~87-byte packets. libopus derives
//! the rate from the bitrate and only caps it by the buffer
//! (opus_encoder.c `bits_target` / `total_bitRate`).
//!
//! These tests put libopus (the `opus` dev-dependency) beside opus-rs at the
//! same settings and compare packet sizes. They stay mono and FEC-free:
//! opus-rs codes stereo as the mid only (#48), and its SILK encoder does not
//! yet subtract the LBRR bits from the frame target, so neither regime is
//! comparable on size.

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Encoder as CEnc,
    Signal as CSignal,
};
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

/// Packets per stream, and how many leading packets are warm-up.
const PACKETS: usize = 100;
const SKIP: usize = 10;

/// Voiced-speech-like test signal: a 140 Hz harmonic series with slow pitch
/// drift and a syllable-rate envelope (copied from
/// `issue_27_libopus_oracle.rs`, mono).
fn voiced(sr: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut phase = 0.0f32;
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let f0 = 140.0 + 15.0 * (2.0 * PI * 0.7 * t).sin();
        phase += 2.0 * PI * f0 / sr as f32;
        let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin();
        let mut l = 0.0;
        for h in 1..=8 {
            let a = 0.25 / h as f32;
            l += a * (h as f32 * phase).sin();
        }
        out.push(l * env * 0.5);
    }
    out
}

/// One encoding setup: sample rate, frame duration and bitrate.
#[derive(Clone, Copy)]
struct Setup {
    sr: usize,
    ms: usize,
    bitrate: i32,
}

impl Setup {
    fn new(sr: usize, ms: usize, bitrate: i32) -> Self {
        Self { sr, ms, bitrate }
    }

    fn frame(self) -> usize {
        self.sr * self.ms / 1000
    }

    /// The packet size the bitrate asks for, in bytes.
    fn target_bytes(self) -> f64 {
        self.bitrate as f64 * self.ms as f64 / 8000.0
    }

    /// Expected SILK-only TOC config: NB 0-3 or WB 8-11, by frame duration.
    fn silk_config(self) -> u8 {
        let base = if self.sr == 8000 { 0 } else { 8 };
        base + match self.ms {
            10 => 0,
            20 => 1,
            40 => 2,
            _ => 3,
        }
    }
}

fn assert_silk_toc(setup: Setup, pkt: &[u8], who: &str, p: usize) {
    assert_eq!(
        pkt[0] >> 3,
        setup.silk_config(),
        "{who} packet {p}: expected SILK-only config {} at {} Hz / {} ms",
        setup.silk_config(),
        setup.sr,
        setup.ms
    );
}

/// opus-rs packets for `setup`, encoded into an output buffer of `buf_len`.
fn rs_packets(setup: Setup, cbr: bool, buf_len: usize) -> Vec<Vec<u8>> {
    let frame = setup.frame();
    let input = voiced(setup.sr, frame * PACKETS);
    let mut enc = OpusEncoder::new(setup.sr as i32, 1, Application::Voip).unwrap();
    enc.bitrate_bps = setup.bitrate;
    enc.use_cbr = cbr;
    let mut out = vec![0u8; buf_len];
    (0..PACKETS)
        .map(|p| {
            let n = enc
                .encode(&input[p * frame..(p + 1) * frame], frame, &mut out)
                .unwrap_or_else(|e| panic!("packet {p}: opus-rs encode failed: {e}"));
            assert_silk_toc(setup, &out[..n], "opus-rs", p);
            out[..n].to_vec()
        })
        .collect()
}

/// libopus VBR packets for `setup`, pinned to opus-rs's settings.
fn c_packets(setup: Setup) -> Vec<Vec<u8>> {
    let frame = setup.frame();
    let input = voiced(setup.sr, frame * PACKETS);
    let mut enc = CEnc::new(setup.sr as u32, CCh::Mono, CApp::Voip).unwrap();
    // A steady harmonic tone can read as music and pull libopus into CELT.
    enc.set_signal(CSignal::Voice).unwrap();
    enc.set_bandwidth(if setup.sr == 8000 {
        CBw::Narrowband
    } else {
        CBw::Wideband
    })
    .unwrap();
    enc.set_complexity(9).unwrap();
    enc.set_vbr(true).unwrap();
    enc.set_bitrate(CBitrate::Bits(setup.bitrate)).unwrap();
    let mut out = vec![0u8; 1500];
    (0..PACKETS)
        .map(|p| {
            let n = enc
                .encode_float(&input[p * frame..(p + 1) * frame], &mut out)
                .unwrap();
            assert_silk_toc(setup, &out[..n], "libopus", p);
            out[..n].to_vec()
        })
        .collect()
}

/// Mean packet size after warm-up, in bytes.
fn mean_bytes(packets: &[Vec<u8>]) -> f64 {
    let tail = &packets[SKIP..];
    tail.iter().map(Vec::len).sum::<usize>() as f64 / tail.len() as f64
}

/// The band opus-rs's mean VBR packet must sit in, as a ratio to libopus's at
/// the same settings. Before the fix it was 2.3-3.2x at 12-24 kb/s.
///
/// Measured against libopus 1.5.2 (opus-rs / libopus):
/// - 16 kHz WB, 20 ms: 0.78 at 12 kb/s, 0.82 at 16k, 0.87 at 24k, 0.89 at
///   32k, 0.94 at 64k.
/// - 24 kb/s WB: 0.84 at 10 ms, 0.88 at 60 ms.
/// - 8 kHz NB, 20 ms, 12 kb/s: 1.01.
///
/// The band is asymmetric because the WB shortfall is not a rate error.
/// Both encoders get the same SILK target, and pick near-identical side
/// info: signal types, quantizer offsets, LTP scale and gains. libopus then
/// codes ~140 bits of excitation pulses a frame at 12 kb/s, where opus-rs
/// codes ~93. That is a separate SILK encoder divergence, and it grows with
/// the internal rate (NB 1.01, MB 0.96, WB 0.78). Raise `MIN_RATIO` once
/// it's fixed.
const MAX_RATIO: f64 = 1.10;
const MIN_RATIO: f64 = 0.70;

/// libopus 1.5.2 and older compute the 60 ms SILK rate with an integer frame
/// rate (`8*bytes_target*frame_rate`, frame_rate = 16, not 16.67), so their
/// target sits ~4% under libopus 1.6.1's `bits_to_bitrate`, which opus-rs
/// follows. 10/20/40 ms agree exactly at multiples of 400 b/s.
const MAX_RATIO_60MS_EXTRA: f64 = 0.04;

/// VBR may overshoot its target on a transient-heavy stretch, but not in the
/// mean: without the bit reservoir (libopus enc_API.c `nBitsExceeded`), this
/// is the check that would catch a drift.
const MAX_OVER_TARGET: f64 = 1.05;

/// Encode `setup` with both encoders and check opus-rs's VBR packet sizes
/// against libopus's and against the bitrate's own target. Returns
/// opus-rs's mean.
fn check_tracks_libopus(setup: Setup) -> f64 {
    let rs = rs_packets(setup, false, 1500);
    let c = c_packets(setup);
    let (rs_mean, c_mean) = (mean_bytes(&rs), mean_bytes(&c));
    let ratio = rs_mean / c_mean;
    let target = setup.target_bytes();
    println!(
        "{:>5} Hz {:>2} ms {:>5} b/s: opus-rs {rs_mean:6.1} B (max {:3}), \
         libopus {c_mean:6.1} B, ratio {ratio:.3}, target {target:5.1} B",
        setup.sr,
        setup.ms,
        setup.bitrate,
        rs[SKIP..].iter().map(Vec::len).max().unwrap()
    );
    let extra = if setup.ms == 60 {
        MAX_RATIO_60MS_EXTRA
    } else {
        0.0
    };
    let max_ratio = MAX_RATIO + extra;
    assert!(
        (MIN_RATIO..=max_ratio).contains(&ratio),
        "{} Hz {} ms {} b/s: opus-rs mean {rs_mean:.1} B is {ratio:.3}x libopus's \
         {c_mean:.1} B (allowed {MIN_RATIO}..={max_ratio})",
        setup.sr,
        setup.ms,
        setup.bitrate
    );
    assert!(
        rs_mean <= MAX_OVER_TARGET * target,
        "{} Hz {} ms {} b/s: opus-rs mean {rs_mean:.1} B overshoots the {target:.1} B \
         target",
        setup.sr,
        setup.ms,
        setup.bitrate
    );
    rs_mean
}

/// The issue's sweep: 16 kHz mono, 20 ms. Each bitrate must land near
/// libopus's packet size, and a higher bitrate must code larger packets.
#[test]
fn silk_vbr_size_tracks_bitrate_like_libopus() {
    let means: Vec<f64> = [12000, 16000, 24000, 32000, 64000]
        .into_iter()
        .map(|bitrate| check_tracks_libopus(Setup::new(16000, 20, bitrate)))
        .collect();
    assert!(
        means.windows(2).all(|w| w[0] < w[1]),
        "opus-rs VBR packet size does not rise with bitrate_bps: {means:.1?}"
    );
}

/// 10 and 60 ms frames, and narrowband at 8 kHz, take the same rate path.
#[test]
fn silk_vbr_size_tracks_libopus_across_frame_sizes_and_nb() {
    check_tracks_libopus(Setup::new(16000, 10, 24000));
    check_tracks_libopus(Setup::new(16000, 60, 24000));
    check_tracks_libopus(Setup::new(8000, 20, 12000));
}

/// The output buffer is a cap, not a rate: while it doesn't bind, the packets
/// must be byte-identical whatever its size. 200 bytes is ~80 kb/s at 20 ms,
/// which binds at 64 kb/s for libopus too, so that rate is checked against
/// the 400-byte buffer only.
#[test]
fn silk_vbr_packets_ignore_output_buffer_size() {
    for bitrate in [12000, 16000, 24000, 32000, 64000] {
        let setup = Setup::new(16000, 20, bitrate);
        let reference = rs_packets(setup, false, 1500);
        let mut buffers = vec![400];
        if bitrate < 64000 {
            buffers.push(200);
        }
        for buf_len in buffers {
            let got = rs_packets(setup, false, buf_len);
            let first_diff = (0..PACKETS).find(|&p| got[p] != reference[p]);
            assert_eq!(
                first_diff,
                None,
                "{bitrate} b/s: a {buf_len}-byte output buffer changed the packets \
                 (mean {:.1} B vs {:.1} B with 1500 bytes)",
                mean_bytes(&got),
                mean_bytes(&reference)
            );
        }
    }
}

/// CBR is unaffected: every packet is exactly the nominal size, at rates
/// whose per-frame bit budget does and does not divide into whole bytes.
#[test]
fn silk_cbr_packets_stay_nominal() {
    for ms in [10, 20] {
        for bitrate in [13000, 16000] {
            let setup = Setup::new(16000, ms, bitrate);
            let target_bits = bitrate as usize * setup.frame() / setup.sr;
            let cbr_bytes = (target_bits + 4) / 8;
            for (p, pkt) in rs_packets(setup, true, 1500).iter().enumerate() {
                assert_eq!(
                    pkt.len(),
                    cbr_bytes,
                    "{ms} ms {bitrate} b/s CBR packet {p}: not the nominal size"
                );
            }
        }
    }
}
