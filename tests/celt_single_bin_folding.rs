//! Single-bin bands must feed band folding at the same scale as the others.
//!
//! A CELT band coded without pulses is filled by folding: a copy of lower,
//! already decoded bins taken from the `norm` buffer, renormalised. Every
//! band stores itself there as `sqrt(N) * X`. For single-bin bands (bands
//! 0-7 of a 2.5 ms frame) the decoder stored `X / 16` instead, the scaling of
//! libopus's fixed-point build (`SHR16(X[0], 4)`, a no-op in float), so any
//! band folding from the lowest bins got a differently shaped spectrum than
//! libopus decodes. Test vector 07 diverges this way from its first 2.5 ms
//! stereo frame on.

use opus::{
    Application as CApp, Bitrate as CBitrate, Channels as CCh, Decoder as CDec, Encoder as CEnc,
};
use opus_rs::OpusDecoder;

const RATE: usize = 48000;
/// 2.5 ms frames.
const FRAME: usize = RATE / 400;

/// Two seconds of white noise (a fixed LCG), the same on both channels with
/// a little independent noise on the right one.
fn noise(ch: usize, n: usize) -> Vec<f32> {
    let mut seed = 0x8251u32;
    let mut next = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 16) as f32 / 32768.0 - 1.0
    };
    let mut out = Vec::with_capacity(n * ch);
    for _ in 0..n {
        let common = 0.3 * next();
        out.push(common);
        if ch == 2 {
            out.push(common + 0.05 * next());
        }
    }
    out
}

/// Worst per-channel SNR (dB) of opus-rs against libopus, both decoding the
/// 2.5 ms CELT stream libopus encodes at `bitrate`.
fn worst_snr(ch: usize, bitrate: i32) -> f64 {
    let channels = if ch == 2 { CCh::Stereo } else { CCh::Mono };
    let mut enc = CEnc::new(RATE as u32, channels, CApp::LowDelay).unwrap();
    enc.set_bitrate(CBitrate::Bits(bitrate)).unwrap();
    let mut c_dec = CDec::new(RATE as u32, channels).unwrap();
    let mut rs_dec = OpusDecoder::new(RATE as i32, ch).unwrap();
    let (mut c_pcm, mut rs_pcm) = (Vec::new(), Vec::new());
    let mut pkt = vec![0u8; 1500];
    let mut c_buf = vec![0f32; ch * FRAME];
    let mut rs_buf = vec![0f32; ch * FRAME];
    for frame in noise(ch, 2 * RATE).chunks_exact(ch * FRAME) {
        let n = enc.encode_float(frame, &mut pkt).unwrap();
        assert_eq!(pkt[0] >> 3 & 3, 0, "expected a 2.5 ms CELT frame");
        let c_n = c_dec.decode_float(&pkt[..n], &mut c_buf, false).unwrap();
        let rs_n = rs_dec.decode(&pkt[..n], FRAME, &mut rs_buf).unwrap();
        assert_eq!((c_n, rs_n), (FRAME, FRAME));
        c_pcm.extend_from_slice(&c_buf);
        rs_pcm.extend_from_slice(&rs_buf);
    }
    (0..ch)
        .map(|c| {
            let (mut sig, mut err) = (0f64, 0f64);
            for (r, o) in c_pcm.iter().zip(&rs_pcm).skip(c).step_by(ch) {
                sig += (*r as f64).powi(2);
                err += (*r as f64 - *o as f64).powi(2);
            }
            10.0 * (sig / err.max(1e-20)).log10()
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn bands_folded_from_single_bin_bands_decode_like_libopus() {
    for ch in [1, 2] {
        for bitrate in [24000, 32000, 48000] {
            let snr = worst_snr(ch, bitrate);
            println!("2.5 ms noise, {ch} ch, {bitrate} b/s: {snr:.1} dB");
            assert!(snr > 40.0, "{ch} ch at {bitrate} b/s: {snr:.1} dB");
        }
    }
}
