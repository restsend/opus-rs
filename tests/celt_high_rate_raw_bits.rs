//! At very high rates a CELT frame's last raw bits must stay where decoders
//! read them.
//!
//! CELT codes some values as raw bits from the end of the frame, kept in a
//! window of up to 32 bits until they are flushed. When the rate leaves room
//! that no band can use (8 and 16 kHz at a few hundred kb/s), the encoder
//! padded the frame with zero bytes from the end before flushing the window,
//! which pushed the pending raw bits (fine energy and the last bands' pulse
//! data) past the padding: decoders read zeros there instead. libopus leaves
//! that room to the range coder's final flush, which fills it with zeros.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

fn signal(rate: usize, ch: usize, n: usize) -> Vec<f32> {
    let mut seed = 9u32;
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f32 / rate as f32;
        phase += 2.0 * PI * (170.0 + 40.0 * (2.0 * PI * 0.6 * t).sin()) / rate as f32;
        let mut v = 0.0;
        for h in 1..=20 {
            if 170.0 * h as f32 > rate as f32 / 2.0 {
                break;
            }
            v += 0.12 / h as f32 * (h as f32 * phase).sin();
        }
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = (v + 0.02 * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0))
            * (0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin());
        for c in 0..ch {
            out.push(if c == 0 { v } else { 0.7 * v });
        }
    }
    out
}

/// Median over frames of the SNR of libopus's decode against the input.
fn median_frame_snr(rate: usize, ch: usize, frame: usize) -> f64 {
    let pcm = signal(rate, ch, frame * 200);
    let mut enc = OpusEncoder::new(rate as i32, ch, Application::RestrictedLowDelay).unwrap();
    enc.bitrate_bps = 510000;
    enc.use_cbr = true;
    enc.complexity = 10;
    let lag = enc.lookahead();
    let channels = if ch == 2 {
        Channels::Stereo
    } else {
        Channels::Mono
    };
    let mut dec = CDec::new(rate as u32, channels).unwrap();
    let mut pkt = [0u8; 1500];
    let mut buf = vec![0f32; 5760 * 2];
    let mut out = Vec::new();
    for x in pcm.chunks_exact(frame * ch) {
        let n = enc.encode(x, frame, &mut pkt).unwrap();
        let k = dec.decode_float(&pkt[..n], &mut buf, false).unwrap();
        out.extend_from_slice(&buf[..k * ch]);
    }
    let mut snrs: Vec<f64> = (4..196)
        .map(|f| {
            let (mut s, mut e) = (0f64, 0f64);
            for i in f * frame * ch..(f + 1) * frame * ch {
                let (a, b) = (pcm[i] as f64, out[i + lag * ch] as f64);
                s += a * a;
                e += (a - b) * (a - b);
            }
            10.0 * (s / e.max(1e-20)).log10()
        })
        .collect();
    snrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    snrs[snrs.len() / 2]
}

#[test]
fn very_high_rate_celt_decodes_cleanly() {
    for (rate, ch, frame) in [
        (8000, 1, 80),
        (8000, 2, 40),
        (16000, 2, 80),
        (16000, 1, 160),
    ] {
        let snr = median_frame_snr(rate, ch, frame);
        println!("{rate} Hz, {ch} ch, {frame} samples: median frame SNR {snr:.1} dB");
        assert!(
            snr > 30.0,
            "{rate} Hz, {ch} ch, {frame} samples: median frame SNR {snr:.1} dB"
        );
    }
}
