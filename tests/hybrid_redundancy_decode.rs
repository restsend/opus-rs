//! A Hybrid frame's mode-switch redundancy must decode like libopus.
//!
//! A Hybrid frame before a switch to CELT (or after one from CELT) may carry a
//! 5 ms redundant CELT frame at its end (RFC 6716 §4.5.1). The decoder read the
//! redundancy flag, then skipped the frame's whole CELT layer and never decoded
//! the redundant frame, so every Hybrid-to-CELT switch lost its high band for a
//! frame and glitched for 40-60 ms. libopus (`opus_decode_frame`) reads the
//! flag only when bits are left for it, decodes the main CELT layer from the
//! bytes before the redundant frame, decodes the redundant frame with all bands
//! (going SILK to CELT from a reset CELT state, which the next CELT frame
//! carries on from; going CELT to SILK from the old state, before CELT's reset)
//! and crossfades it in with the CELT window; only a SILK-to-CELT frame's
//! redundancy counts as `prev_redundancy`.
//!
//! The test switches a libopus encoder from Hybrid to CELT and decodes the
//! stream with both decoders. Before the fix the switch's window sat at 2 dB.

use opus::{
    Application as CApp, Bitrate as CBitrate, Channels as CCh, Decoder as CDec, Encoder as CEnc,
    Signal as CSignal,
};
use opus_rs::OpusDecoder;
use std::f32::consts::PI;

/// Largest Opus frame per channel: 120 ms at 48 kHz.
const MAX_FRAME: usize = 5760;
const RATE: usize = 48000;
/// 20 ms at 48 kHz: the encoders' frame and the comparison window.
const FRAME: usize = 960;

fn noise(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

/// A voiced harmonic series (160 Hz, drifting) with a little broadband
/// noise, under a syllable-rate envelope: content in every Opus band.
#[allow(dead_code)]
fn voiced(n: usize) -> Vec<f32> {
    let mut seed = 7u32;
    let mut phase = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            phase += 2.0 * PI * (160.0 + 20.0 * (2.0 * PI * 0.8 * t).sin()) / RATE as f32;
            let mut v = 0.0;
            for h in 1..=12 {
                v += 0.2 / h as f32 * (h as f32 * phase).sin();
            }
            (v + 0.05 * noise(&mut seed)) * (0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin())
        })
        .collect()
}

/// Percussive noise bursts every 100 ms, the first at t = 0.
#[allow(dead_code)]
fn bursts(n: usize) -> Vec<f32> {
    let mut seed = 11u32;
    (0..n)
        .map(|i| 0.6 * noise(&mut seed) * (-((i % 4800) as f32) / 300.0).exp())
        .collect()
}

/// libopus-encoded 20 ms mono packets of `input`; `setup` adjusts the
/// encoder before each frame (its index).
fn libopus_packets(input: &[f32], app: CApp, setup: impl Fn(usize, &mut CEnc)) -> Vec<Vec<u8>> {
    let mut enc = CEnc::new(RATE as u32, CCh::Mono, app).unwrap();
    let mut pkt = vec![0u8; 1500];
    input
        .chunks_exact(FRAME)
        .enumerate()
        .map(|(i, f)| {
            setup(i, &mut enc);
            let n = enc.encode_float(f, &mut pkt).unwrap();
            pkt[..n].to_vec()
        })
        .collect()
}

/// The packet's mode from its TOC: 'S' SILK-only, 'H' Hybrid, 'C' CELT-only.
#[allow(dead_code)]
fn mode(pkt: &[u8]) -> char {
    match pkt[0] >> 3 {
        0..=11 => 'S',
        12..=15 => 'H',
        _ => 'C',
    }
}

/// Decode `packets` with libopus and opus-rs at 48 kHz and return the worst
/// 20 ms window's error against libopus, in dB below the whole signal's
/// level (lag 0), over the windows from `from` on.
fn worst_window_db(packets: &[Vec<u8>], from: usize) -> (f64, usize) {
    let mut c_dec = CDec::new(RATE as u32, CCh::Mono).unwrap();
    let mut rs_dec = OpusDecoder::new(RATE as i32, 1).unwrap();
    let (mut c_pcm, mut rs_pcm) = (Vec::new(), Vec::new());
    let mut c_buf = vec![0f32; MAX_FRAME];
    let mut rs_buf = vec![0f32; MAX_FRAME];
    for (p, pkt) in packets.iter().enumerate() {
        let c_n = c_dec.decode_float(pkt, &mut c_buf, false).unwrap();
        let rs_n = rs_dec
            .decode(pkt, MAX_FRAME, &mut rs_buf)
            .unwrap_or_else(|e| panic!("packet {p}: opus-rs decode failed: {e}"));
        assert_eq!(rs_n, c_n, "packet {p}: decoded length");
        c_pcm.extend_from_slice(&c_buf[..c_n]);
        rs_pcm.extend_from_slice(&rs_buf[..rs_n]);
    }
    let level = c_pcm.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / c_pcm.len() as f64;
    let mut worst = (f64::INFINITY, 0);
    for (w, (c, r)) in c_pcm
        .chunks(FRAME)
        .zip(rs_pcm.chunks(FRAME))
        .enumerate()
        .skip(from)
    {
        let err = c
            .iter()
            .zip(r)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / FRAME as f64;
        let db = 10.0 * (level / err.max(1e-30)).log10();
        if db < worst.0 {
            worst = (db, w);
        }
    }
    worst
}

#[test]
fn hybrid_to_celt_switch_decodes_like_libopus() {
    let input = voiced(RATE * 2);
    // Hybrid at 24 kb/s voice; CELT from frame 50 at 128 kb/s music.
    let packets = libopus_packets(&input, CApp::Audio, |i, e| {
        let celt = i >= 50;
        e.set_bitrate(CBitrate::Bits(if celt { 128000 } else { 24000 }))
            .unwrap();
        e.set_signal(if celt { CSignal::Music } else { CSignal::Voice })
            .unwrap();
    });
    let modes: String = packets.iter().map(|p| mode(p)).collect();
    assert!(
        modes.contains("HC"),
        "the stream switches from Hybrid to CELT: {modes}"
    );
    let (db, at) = worst_window_db(&packets, 2);
    println!("Hybrid to CELT: worst window {db:.1} dB (window {at})");
    assert!(
        db > 40.0,
        "Hybrid to CELT: window {at} is {db:.1} dB from libopus"
    );
}
