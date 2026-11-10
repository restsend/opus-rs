//! Intensity-coded stereo bands of two bins must keep their phase flip.
//!
//! In 2.5 ms (and 5 ms) CELT frames some stereo bands hold only two bins.
//! libopus `quant_band_stereo` codes them with a special case but still
//! negates the side channel afterwards when the band's intensity inversion
//! flag is set; only `stereo_merge` is skipped for them. The decoder returned
//! from that special case before the flip, so the right channel of such
//! bands came out with the wrong sign whenever the encoder used inversion
//! (test vector 07's 2.5 ms stereo frames fail `opus_compare`).

use opus::{
    Application as CApp, Bitrate as CBitrate, Channels as CCh, Decoder as CDec, Encoder as CEnc,
};
use opus_rs::OpusDecoder;
use std::f32::consts::PI;

const RATE: usize = 48000;
/// 2.5 ms frames.
const FRAME: usize = RATE / 400;

/// Two seconds of stereo with anti-phase tones in the two-bin bands at
/// 2.5 ms (2-3.2 kHz) over in-phase low tones.
fn anti_phase(n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(2 * n);
    for i in 0..n {
        let t = i as f32 / RATE as f32;
        let common = 0.2 * (2.0 * PI * 400.0 * t).sin() + 0.1 * (2.0 * PI * 900.0 * t).sin();
        let side = 0.15 * (2.0 * PI * 2100.0 * t).sin()
            + 0.12 * (2.0 * PI * 2500.0 * t).sin()
            + 0.1 * (2.0 * PI * 2900.0 * t).sin();
        out.push(common + side);
        out.push(common - side);
    }
    out
}

/// Worst per-channel SNR (dB) of opus-rs against libopus, both decoding the
/// stream libopus encodes with phase inversion on or off.
fn worst_snr(inversion: bool) -> f64 {
    let mut enc = CEnc::new(RATE as u32, CCh::Stereo, CApp::LowDelay).unwrap();
    enc.set_bitrate(CBitrate::Bits(64000)).unwrap();
    enc.set_phase_inversion_disabled(!inversion).unwrap();
    let mut c_dec = CDec::new(RATE as u32, CCh::Stereo).unwrap();
    let mut rs_dec = OpusDecoder::new(RATE as i32, 2).unwrap();
    let (mut c_pcm, mut rs_pcm) = (Vec::new(), Vec::new());
    let mut pkt = vec![0u8; 1500];
    let mut c_buf = vec![0f32; 2 * FRAME];
    let mut rs_buf = vec![0f32; 2 * FRAME];
    for frame in anti_phase(2 * RATE).as_chunks::<{ 2 * FRAME }>().0 {
        let n = enc.encode_float(frame, &mut pkt).unwrap();
        assert_eq!(pkt[0] >> 3 & 3, 0, "expected a 2.5 ms CELT frame");
        let c_n = c_dec.decode_float(&pkt[..n], &mut c_buf, false).unwrap();
        let rs_n = rs_dec.decode(&pkt[..n], FRAME, &mut rs_buf).unwrap();
        assert_eq!((c_n, rs_n), (FRAME, FRAME));
        c_pcm.extend_from_slice(&c_buf);
        rs_pcm.extend_from_slice(&rs_buf);
    }
    (0..2)
        .map(|c| {
            let (mut sig, mut err) = (0f64, 0f64);
            for (r, o) in c_pcm.iter().zip(&rs_pcm).skip(c).step_by(2) {
                sig += (*r as f64).powi(2);
                err += (*r as f64 - *o as f64).powi(2);
            }
            10.0 * (sig / err.max(1e-20)).log10()
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn two_bin_intensity_bands_decode_like_libopus() {
    let without = worst_snr(false);
    let with = worst_snr(true);
    println!("2.5 ms stereo at 64 kb/s: {without:.1} dB without inversion, {with:.1} dB with it");
    // A missing flip gives the right channel's two-bin bands the wrong sign,
    // which puts the inverted stream near 0 dB.
    assert!(without > 30.0, "without phase inversion: {without:.1} dB");
    assert!(with > 30.0, "with phase inversion: {with:.1} dB");
}
