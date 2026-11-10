//! When the dynalloc boosts leave no room for the allocation trim, the
//! encoder must allocate with the default trim of 5, the one the decoder
//! assumes. Band limited clicks at complexity 10 and 32-48 kb/s hit this:
//! the second MDCT makes dynalloc boost the top bands of their short-block
//! frames until the trim no longer fits. An encoder that allocated with
//! its own trim there decoded those clicks 13-16 dB low; libopus 1.6
//! loses at most 9 dB on the same input.
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f64::consts::PI;

const RATE: usize = 48000;
const FRAME: usize = RATE / 50;
const LOOKAHEAD: usize = 312;
const CLICKS: usize = 24;
const SPACING: usize = RATE / 5;

/// Click `i` sits `i * 40` samples into its 20 ms frame
fn positions() -> Vec<usize> {
    (0..CLICKS)
        .map(|i| RATE / 10 + i * SPACING + i * 40)
        .collect()
}

/// Clicks band limited to 11025 Hz, Hann windowed over +-256 samples,
/// peak 0.9: a single sample click at 22.05 kHz resampled to 48 kHz
fn input() -> Vec<f32> {
    let w = 2.0 * 11025.0 / RATE as f64;
    let half = 256isize;
    let mut s = vec![0.0f32; RATE / 10 + CLICKS * SPACING + RATE / 5];
    for p in positions() {
        for n in -half..=half {
            let x = n as f64;
            let sinc = if n == 0 {
                1.0
            } else {
                (PI * w * x).sin() / (PI * w * x)
            };
            let hann = 0.5 + 0.5 * (PI * x / (half as f64 + 1.0)).cos();
            s[(p as isize + n) as usize] += (0.9 * sinc * hann) as f32;
        }
    }
    s
}

/// The level of each decoded click in dB against the input peak
fn click_levels(input: &[f32], bitrate: i32, cbr: bool) -> Vec<f64> {
    let mut encoder = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
    encoder.bitrate_bps = bitrate;
    encoder.use_cbr = cbr;
    encoder.complexity = 10;
    let mut decoder = OpusDecoder::new(48000, 1).unwrap();
    let mut packet = vec![0u8; 1500];
    let mut pcm = vec![0.0f32; FRAME];
    let mut decoded = Vec::new();
    let frames = (input.len() + LOOKAHEAD).div_ceil(FRAME);
    for k in 0..frames {
        let mut frame = vec![0.0f32; FRAME];
        let start = (k * FRAME).min(input.len());
        let end = (start + FRAME).min(input.len());
        frame[..end - start].copy_from_slice(&input[start..end]);
        let len = encoder.encode(&frame, FRAME, &mut packet).unwrap();
        let n = decoder.decode(&packet[..len], FRAME, &mut pcm).unwrap();
        decoded.extend_from_slice(&pcm[..n]);
    }
    positions()
        .iter()
        .map(|&p| {
            let at = p + LOOKAHEAD;
            let peak = decoded[at - 240..at + 240]
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            20.0 * (f64::from(peak) / 0.9).log10()
        })
        .collect()
}

#[test]
fn clicks_survive_when_the_trim_does_not_fit() {
    let input = input();
    for cbr in [false, true] {
        for kbps in [32, 40, 48] {
            let levels = click_levels(&input, kbps * 1000, cbr);
            let worst = levels.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(
                worst > -11.0,
                "{} {kbps} kb/s: worst click {worst:.1} dB",
                if cbr { "CBR" } else { "VBR" }
            );
        }
    }
}
