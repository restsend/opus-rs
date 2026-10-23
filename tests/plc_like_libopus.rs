//! Packet loss concealment checked against libopus.
//!
//! A synthetic vowel (harmonics of a gliding 140 Hz pitch, a little noise) is
//! encoded with opus-rs in SILK wideband, Hybrid and CELT-only mode. Packets
//! 20, 35, 36, 60 and 61 are dropped and the stream decoded by opus-rs and by
//! libopus. SILK conceals in fixed point, so it matches libopus bit for bit;
//! CELT conceals in floating point (`celt_decode_lost`), and Hybrid adds it
//! above SILK's band. Each lost frame, and the frame after it, must agree.

use opus::{Channels as CCh, Decoder as CDec};
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

const RATE: usize = 48_000;
const FRAME: usize = 960;
const LOST: [usize; 5] = [20, 35, 36, 60, 61];

/// Four seconds of the vowel at `sr`, silent in frames `silent`.
fn vowel(sr: usize, silent: std::ops::Range<usize>) -> Vec<f32> {
    let mut seed = 11u32;
    let mut phase = 0.0f32;
    (0..80 * sr / 50)
        .map(|i| {
            let t = i as f32 / sr as f32;
            phase += 2.0 * PI * (140.0 + 30.0 * (2.0 * PI * 0.7 * t).sin()) / sr as f32;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
            let mut s = 0.02 * noise;
            for h in 1..=40 {
                if 140.0 * h as f32 > sr as f32 / 2.0 - 2000.0 {
                    break;
                }
                s += 0.25 / h as f32 * (h as f32 * phase).sin();
            }
            if silent.contains(&(i * 50 / sr)) {
                0.0
            } else {
                s
            }
        })
        .collect()
}

/// Packets of the vowel encoded by opus-rs at `sr`, `bitrate`, as `app`.
fn encode(
    sr: usize,
    app: Application,
    bitrate: i32,
    hybrid: bool,
    silent: std::ops::Range<usize>,
) -> Vec<Vec<u8>> {
    let mut enc = OpusEncoder::new(sr as i32, 1, app).unwrap();
    enc.bitrate_bps = bitrate;
    if hybrid {
        enc.enable_hybrid_mode().unwrap();
    }
    let mut out = vec![0u8; 1500];
    vowel(sr, silent)
        .chunks_exact(sr / 50)
        .map(|frame| {
            let n = enc.encode(frame, sr / 50, &mut out).unwrap();
            out[..n].to_vec()
        })
        .collect()
}

/// opus-rs and libopus decodes of `packets` with `LOST` dropped.
fn decode_both(packets: &[Vec<u8>]) -> (Vec<f32>, Vec<f32>) {
    let mut rs = OpusDecoder::new(RATE as i32, 1).unwrap();
    let mut c = CDec::new(RATE as u32, CCh::Mono).unwrap();
    let (mut got, mut want) = (Vec::new(), Vec::new());
    let mut buf = vec![0f32; FRAME];
    for (k, packet) in packets.iter().enumerate() {
        let data: &[u8] = if LOST.contains(&k) { &[] } else { packet };
        let n = rs.decode(data, FRAME, &mut buf).unwrap();
        got.extend_from_slice(&buf[..n]);
        let n = c.decode_float(data, &mut buf, false).unwrap();
        want.extend_from_slice(&buf[..n]);
    }
    (got, want)
}

/// SNR of frame `k` of `got` against `want`, in dB.
fn snr(got: &[f32], want: &[f32], k: usize) -> f64 {
    let (mut sig, mut err) = (1e-20f64, 1e-20f64);
    for i in k * FRAME..(k + 1) * FRAME {
        sig += (want[i] as f64).powi(2);
        err += (got[i] as f64 - want[i] as f64).powi(2);
    }
    10.0 * (sig / err).log10()
}

/// Every lost frame and the frame after it reach `floor` dB against libopus.
fn assert_conceals_like_libopus(what: &str, packets: &[Vec<u8>], floor: f64) {
    let (got, want) = decode_both(packets);
    for k in LOST.iter().flat_map(|&k| [k, k + 1]) {
        let s = snr(&got, &want, k);
        assert!(s >= floor, "{what}: frame {k} at {s:.1} dB < {floor} dB");
    }
}

#[test]
fn silk_wideband_concealment_matches_libopus() {
    let packets = encode(16_000, Application::Voip, 20_000, false, 0..0);
    assert!(packets.iter().all(|p| p[0] >> 3 == 9), "SILK WB 20 ms");
    assert_conceals_like_libopus("SILK WB", &packets, 90.0);
}

#[test]
fn hybrid_concealment_matches_libopus() {
    let packets = encode(48_000, Application::Voip, 32_000, true, 0..0);
    assert!(
        packets.iter().all(|p| (12..=15).contains(&(p[0] >> 3))),
        "Hybrid"
    );
    assert_conceals_like_libopus("Hybrid", &packets, 40.0);
}

#[test]
fn celt_concealment_matches_libopus() {
    let packets = encode(48_000, Application::Audio, 64_000, false, 0..0);
    assert!(packets.iter().all(|p| p[0] >> 3 >= 16), "CELT");
    assert_conceals_like_libopus("CELT", &packets, 40.0);
}

#[test]
fn concealment_after_silence_stays_silent() {
    // Silent through frames 28..37, so the losses at 35 and 36 follow silence.
    for (what, sr, app, rate, hybrid) in [
        ("SILK WB", 16_000, Application::Voip, 20_000, false),
        ("Hybrid", 48_000, Application::Voip, 32_000, true),
    ] {
        let packets = encode(sr, app, rate, hybrid, 28..37);
        let (got, want) = decode_both(&packets);
        for k in [35, 36] {
            let peak = got[k * FRAME..(k + 1) * FRAME]
                .iter()
                .fold(0f32, |m, v| m.max(v.abs()));
            let want_peak = want[k * FRAME..(k + 1) * FRAME]
                .iter()
                .fold(0f32, |m, v| m.max(v.abs()));
            assert!(peak < 1e-3, "{what}: frame {k} peaks at {peak}");
            assert!(
                peak <= want_peak * 2.0 + 1e-6,
                "{what}: frame {k} {peak} vs libopus {want_peak}"
            );
        }
        assert!(
            snr(&got, &want, 37) >= 40.0,
            "{what}: first frame after silence"
        );
    }
}
