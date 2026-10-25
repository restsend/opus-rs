//! 10 ms SILK frames must decode like libopus.
//!
//! A 10 ms SILK frame has two subframes and codes its pitch contour with the
//! 10 ms tables (`silk_pitch_contour_10_ms_iCDF`, and `_NB_` at 8 kHz). The
//! table is picked in `silk_decoder_set_fs` from the frame's subframe count,
//! which libopus sets before the call. opus-rs's `silk_decoder_set_fs` forced
//! four subframes first, so it always picked the 20 ms table: the first
//! voiced 10 ms frame read the wrong contour, the range decoder desynced, and
//! the rest of the stream decoded to noise.
//!
//! The test encodes a voiced signal with libopus in 10 ms SILK packets at
//! each SILK bandwidth and decodes them with both decoders.

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Decoder as CDec,
    Encoder as CEnc, Signal as CSignal,
};
use opus_rs::OpusDecoder;
use std::f32::consts::PI;

const RATE: usize = 48000;
/// 10 ms at 48 kHz: the packet and the comparison window.
const FRAME: usize = 480;

/// One second of a voiced harmonic series (150 Hz, drifting) under a
/// syllable-rate envelope; the right channel is a phase-shifted copy.
fn voiced(ch: usize) -> Vec<f32> {
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(RATE * ch);
    for i in 0..RATE {
        let t = i as f32 / RATE as f32;
        phase += 2.0 * PI * (150.0 + 20.0 * (2.0 * PI * 0.7 * t).sin()) / RATE as f32;
        let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin();
        let (mut l, mut r) = (0.0, 0.0);
        for h in 1..=10 {
            l += 0.2 / h as f32 * (h as f32 * phase).sin();
            r += 0.2 / h as f32 * (h as f32 * phase + 0.3 * h as f32).sin();
        }
        out.push(l * env * 0.5);
        if ch == 2 {
            out.push(r * env * 0.4);
        }
    }
    out
}

/// Decode libopus's 10 ms SILK packets at `bw` with libopus and opus-rs, and
/// return the worst 10 ms window's error against libopus, in dB below the
/// whole signal's level.
fn worst_window_db(ch: usize, bw: CBw) -> f64 {
    let cch = if ch == 2 { CCh::Stereo } else { CCh::Mono };
    let mut enc = CEnc::new(RATE as u32, cch, CApp::Voip).unwrap();
    enc.set_bitrate(CBitrate::Bits(16000 * ch as i32)).unwrap();
    enc.set_bandwidth(bw).unwrap();
    enc.set_signal(CSignal::Voice).unwrap();
    let mut c_dec = CDec::new(RATE as u32, cch).unwrap();
    let mut rs_dec = OpusDecoder::new(RATE as i32, ch).unwrap();
    let (mut c_pcm, mut rs_pcm) = (Vec::new(), Vec::new());
    let mut pkt = vec![0u8; 1500];
    let mut buf = vec![0f32; FRAME * ch];
    for (p, frame) in voiced(ch).chunks_exact(FRAME * ch).enumerate() {
        let n = enc.encode_float(frame, &mut pkt).unwrap();
        assert!(pkt[0] >> 3 <= 11, "packet {p}: expected SILK-only");
        let got = c_dec.decode_float(&pkt[..n], &mut buf, false).unwrap();
        c_pcm.extend_from_slice(&buf[..got * ch]);
        let got = rs_dec
            .decode(&pkt[..n], FRAME, &mut buf)
            .unwrap_or_else(|e| panic!("packet {p}: opus-rs decode failed: {e}"));
        rs_pcm.extend_from_slice(&buf[..got * ch]);
    }
    let ms = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64;
    let level = ms(&c_pcm);
    c_pcm
        .chunks(FRAME * ch)
        .zip(rs_pcm.chunks(FRAME * ch))
        .map(|(c, r)| {
            let err: Vec<f32> = c.iter().zip(r).map(|(a, b)| a - b).collect();
            10.0 * (level / ms(&err).max(1e-30)).log10()
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn silk_10ms_frames_decode_like_libopus() {
    for ch in [1, 2] {
        for (name, bw) in [
            ("NB", CBw::Narrowband),
            ("MB", CBw::Mediumband),
            ("WB", CBw::Wideband),
        ] {
            let db = worst_window_db(ch, bw);
            println!("SILK {name} {ch} ch 10 ms: worst window {db:.1} dB");
            assert!(
                db > 100.0,
                "SILK {name} {ch} ch 10 ms: {db:.1} dB from libopus"
            );
        }
    }
}
