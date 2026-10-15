//! A super-wideband Hybrid frame's CELT layer must end at band 19.
//!
//! libopus sets CELT's end band from the packet's bandwidth for every mode
//! (`opus_decode_frame`: narrowband 13, mediumband and wideband 17,
//! super-wideband 19, fullband 21). For Hybrid frames
//! `celt_end_band_from_toc` returned 21 at every bandwidth, so the CELT layer
//! of a super-wideband Hybrid frame read bands its encoder never coded.
//!
//! The test decodes libopus's super-wideband Hybrid. Before the fix it sat at
//! 10-15 dB from libopus.

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Decoder as CDec,
    Encoder as CEnc, Signal as CSignal,
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
fn swb_hybrid_decodes_like_libopus() {
    let input = voiced(RATE * 3);
    let packets = libopus_packets(&input, CApp::Voip, |_, e| {
        e.set_bandwidth(CBw::Superwideband).unwrap();
        e.set_signal(CSignal::Voice).unwrap();
        e.set_bitrate(CBitrate::Bits(24000)).unwrap();
    });
    // Hybrid at super-wideband: TOC configs 12 and 13.
    assert!(
        packets.iter().all(|p| p[0] >> 3 == 12 || p[0] >> 3 == 13),
        "SWB Hybrid packets"
    );
    let (db, at) = worst_window_db(&packets, 2);
    println!("SWB Hybrid: worst window {db:.1} dB (window {at})");
    assert!(
        db > 40.0,
        "SWB Hybrid: window {at} is {db:.1} dB from libopus"
    );
}
