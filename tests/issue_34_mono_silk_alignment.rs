//! Issue #34 regression: mono SILK output must be aligned with libopus.
//!
//! `SilkDecoder::decode` copied mono output from `w_silk_buf[0][2..]`, while
//! libopus (`silk/dec_API.c`, the resample call) reads mono and stereo alike
//! from `samplesOut1_tmp[n][1]` — the same one-sample-delayed position. The
//! opus-rs mono output was therefore one internal-rate sample early (3 samples
//! at a 48 kHz API rate, 1 at 16 kHz).
//!
//! These tests decode libopus-encoded mono WB SILK packets with both libopus
//! and opus-rs and require agreement at **lag 0** (no lag search). Before the
//! fix the best agreement at lag 0 is only ~19 dB at 48 kHz; the samples line
//! up only after shifting 3 samples (or 1 at 16 kHz).

use opus::{
    Application as CApp, Bandwidth as CBw, Channels as CCh, Decoder as CDec, Encoder as CEnc,
    Signal as CSignal,
};
use opus_rs::OpusDecoder;
use std::f32::consts::PI;

/// Largest Opus frame per channel: 120 ms at 48 kHz.
const MAX_FRAME: usize = 5760;

/// Voiced-speech-like mono harmonic series with slow pitch drift and a
/// syllable-rate envelope (same shape as the other libopus oracle tests).
fn voiced(sr: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut phase = 0.0f32;
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let f0 = 140.0 + 15.0 * (2.0 * PI * 0.7 * t).sin();
        phase += 2.0 * PI * f0 / sr as f32;
        let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin();
        let mut s = 0.0;
        for h in 1..=8 {
            s += (0.25 / h as f32) * (h as f32 * phase).sin();
        }
        out.push(s * env * 0.5);
    }
    out
}

/// SNR (dB) of `test` against `reference` over `reference[start..start + len]`
/// at zero lag only.
fn snr_at_lag0(reference: &[f32], test: &[f32], start: usize, len: usize) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for i in start..start + len {
        let r = reference[i] as f64;
        let t = test[i] as f64;
        sig += r * r;
        err += (r - t) * (r - t);
    }
    10.0 * (sig / err.max(1e-20)).log10()
}

/// Encode ~1 s of mono WB SILK at 16 kHz with libopus, decode each packet with
/// libopus and with opus-rs at `api_rate`, and return the worst 20 ms-window
/// SNR at lag 0 (after a one-packet warm-up).
fn worst_window_snr(api_rate: u32) -> f64 {
    let sr = 16000usize;
    let frame = 320usize; // 20 ms @ 16 kHz
    let packets = 50usize;
    let input = voiced(sr, frame * packets);

    let mut enc = CEnc::new(sr as u32, CCh::Mono, CApp::Voip).unwrap();
    enc.set_bitrate(opus::Bitrate::Bits(24000)).unwrap();
    enc.set_bandwidth(CBw::Wideband).unwrap();
    enc.set_signal(CSignal::Voice).unwrap();

    let out_per_pkt = api_rate as usize / 50; // 20 ms at the API rate
    let mut c_dec = CDec::new(api_rate, CCh::Mono).unwrap();
    let mut rs_dec = OpusDecoder::new(api_rate as i32, 1).unwrap();

    let (mut c_pcm, mut rs_pcm) = (Vec::new(), Vec::new());
    let mut pkt = vec![0u8; 1500];
    let mut c_buf = vec![0f32; MAX_FRAME];
    let mut rs_buf = vec![0f32; MAX_FRAME];
    for p in 0..packets {
        let n = enc
            .encode_float(&input[p * frame..(p + 1) * frame], &mut pkt)
            .unwrap();
        // The packet must be SILK-only (not CELT): config <= 11.
        assert!(
            pkt[0] >> 3 <= 11,
            "packet {p} is not SILK: toc {:#04x}",
            pkt[0]
        );
        let c_n = c_dec.decode_float(&pkt[..n], &mut c_buf, false).unwrap();
        let rs_n = rs_dec
            .decode(&pkt[..n], MAX_FRAME, &mut rs_buf)
            .unwrap_or_else(|e| panic!("packet {p}: opus-rs decode failed: {e}"));
        assert_eq!(c_n, out_per_pkt, "libopus decoded length");
        assert_eq!(rs_n, out_per_pkt, "opus-rs decoded length");
        c_pcm.extend_from_slice(&c_buf[..c_n]);
        rs_pcm.extend_from_slice(&rs_buf[..rs_n]);
    }

    // Skip the first packet (encoder/decoder warm-up); compare 20 ms windows.
    let mut worst = f64::INFINITY;
    let mut start = out_per_pkt;
    while start + out_per_pkt <= c_pcm.len() {
        worst = worst.min(snr_at_lag0(&c_pcm, &rs_pcm, start, out_per_pkt));
        start += out_per_pkt;
    }
    worst
}

/// Before the fix this is ~19 dB (the streams agree only after a 3-sample
/// shift); after it, it is essentially sample-exact.
const ALIGN_DB: f64 = 60.0;

#[test]
fn mono_silk_48k_matches_libopus_at_lag_zero() {
    let snr = worst_window_snr(48000);
    assert!(
        snr > ALIGN_DB,
        "mono SILK @48kHz runs ahead of libopus; worst window {snr:.1} dB at lag 0"
    );
}

#[test]
fn mono_silk_16k_matches_libopus_at_lag_zero() {
    let snr = worst_window_snr(16000);
    assert!(
        snr > ALIGN_DB,
        "mono SILK @16kHz runs ahead of libopus; worst window {snr:.1} dB at lag 0"
    );
}
