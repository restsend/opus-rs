//! Issue #37: 24 kHz input decoded to garbage.
//!
//! The CELT encoder always coded all 21 bands, but the decoder derives its
//! `end` band from the TOC (SWB = 19 for 24 kHz). Coding even one band more
//! than the decoder reads desynchronises the range coder, so every 24 kHz
//! CELT packet came out as noise. A second, independent bug: at a non-48 kHz
//! API rate the CELT encoder must zero-stuff its input by `upsample` and
//! zero/scale the MDCT output (libopus `celt_preemphasis` / `compute_mdcts`);
//! it instead encoded the 480-sample frame as if it were 48 kHz.
//!
//! These tests encode the issue's chirp at 24 kHz, decode at 48 kHz, and
//! check each 100 ms window at its best lag against the analytic signal. On
//! the unfixed tree the 24 kHz CELT windows score ~0.04; now they track
//! libopus (which scores ~0.9 on the same signal).

use opus::{Application as CApp, Channels as CCh, Encoder as CEnc};
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::TAU;

const PRE_SKIP: usize = 312;

fn chirp(rate: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / rate as f32;
            (TAU * (200.0 * t + 600.0 * t * t)).sin() * 0.4
        })
        .collect()
}

fn c_channels(ch: usize) -> CCh {
    if ch == 2 { CCh::Stereo } else { CCh::Mono }
}

fn interleave_mono(x: &[f32], ch: usize) -> Vec<f32> {
    if ch == 1 {
        x.to_vec()
    } else {
        x.iter().flat_map(|&v| [v, v]).collect()
    }
}

/// Worst 100 ms-window correlation at the best lag (±40 samples) of a 48 kHz
/// decode against the analytic chirp, plus the RMS ratio (decoded / input)
/// after the first 100 ms.
fn rust_roundtrip(rate: usize, ch: usize, app: Application, br: i32) -> (u8, f64, f64) {
    let frame = rate / 50;
    let len = rate * 13 / 10;
    let mut input = interleave_mono(&chirp(rate, len), ch);
    // Pad for the pre-skip, rounded up to a whole number of frames.
    let pad = (PRE_SKIP * 48_000 / rate) * ch;
    input.resize(input.len() + pad.div_ceil(frame * ch) * frame * ch, 0.0);

    let mut enc = OpusEncoder::new(rate as i32, ch, app).unwrap();
    enc.bitrate_bps = br;
    let mut dec = OpusDecoder::new(48_000, ch).unwrap();
    let (mut out, mut pkt, mut pcm) = (Vec::new(), [0u8; 8000], vec![0f32; 5760 * ch]);
    let mut toc = 0;
    for f in input.chunks(frame * ch) {
        let n = enc.encode(f, frame, &mut pkt).unwrap();
        toc = pkt[0];
        let got = dec.decode(&pkt[..n], 5760, &mut pcm).unwrap();
        out.extend_from_slice(&pcm[..got * ch]);
    }

    let scale = 48_000 / rate;
    let out = &out[PRE_SKIP * ch..PRE_SKIP * ch + len * scale * ch];
    let reference = interleave_mono(&chirp(48_000, len * scale), ch);

    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f32>();
    let mut worst = 1.0f64;
    // Window hop 100 ms; compare the middle 4700 samples of each window.
    let win = 4800 * ch;
    let mut i = 0usize;
    while i + win + 100 <= out.len() && i + win + 100 <= reference.len() {
        let mut best = f64::NEG_INFINITY;
        for lag in -40isize..=40 {
            let mut sig = 0f64;
            let mut err = 0f64;
            let mut dotv = 0f64;
            let mut tt = 0f64;
            for k in (50 * ch..win - 100).step_by(ch) {
                let j = i as isize + k as isize + lag;
                if j < 0 || j as usize >= out.len() {
                    continue;
                }
                let r = reference[i + k] as f64;
                let t = out[j as usize] as f64;
                sig += r * r;
                err += (r - t) * (r - t);
                dotv += r * t;
                tt += t * t;
            }
            if sig == 0.0 {
                continue;
            }
            let corr = dotv / (sig * tt).sqrt().max(1e-20);
            if corr > best {
                best = corr;
            }
            let _ = err;
        }
        worst = worst.min(best);
        i += win;
    }

    // RMS ratio over the steady part.
    let ref_rms = dot(&reference, &reference) / reference.len() as f32;
    let out_rms = dot(out, out) / out.len() as f32;
    (toc, worst, (out_rms / ref_rms).sqrt() as f64)
}

fn libopus_enc_rust_dec(rate: usize, ch: usize, br: i32) -> f64 {
    let frame = rate / 50;
    let len = rate * 13 / 10;
    let mut input = interleave_mono(&chirp(rate, len), ch);
    let pad = (PRE_SKIP * 48_000 / rate) * ch;
    input.resize(input.len() + pad.div_ceil(frame * ch) * frame * ch, 0.0);
    let mut enc = CEnc::new(rate as u32, c_channels(ch), CApp::Audio).unwrap();
    enc.set_bitrate(opus::Bitrate::Bits(br)).unwrap();
    let mut dec = OpusDecoder::new(48_000, ch).unwrap();
    let (mut out, mut pkt, mut pcm) = (Vec::new(), vec![0u8; 8000], vec![0f32; 5760 * ch]);
    for f in input.chunks(frame * ch) {
        let n = enc.encode_float(f, &mut pkt).unwrap();
        let got = dec.decode(&pkt[..n], 5760, &mut pcm).unwrap();
        out.extend_from_slice(&pcm[..got * ch]);
    }
    let scale = 48_000 / rate;
    let out = &out[PRE_SKIP * ch..PRE_SKIP * ch + len * scale * ch];
    let reference = interleave_mono(&chirp(48_000, len * scale), ch);
    // Best lag over a coarse range (no windowing needed for the decode path).
    let mut best = f64::NEG_INFINITY;
    for lag in -600isize..=600 {
        let (mut sig, mut dotv, mut tt) = (0f64, 0f64, 0f64);
        for i in (2000..out.len()).step_by(ch) {
            let j = i as isize + lag;
            if j < 0 || j as usize >= out.len() {
                continue;
            }
            let r = reference[i] as f64;
            let t = out[j as usize] as f64;
            sig += r * r;
            dotv += r * t;
            tt += t * t;
        }
        if sig > 0.0 {
            best = best.max(dotv / (sig * tt).sqrt().max(1e-20));
        }
    }
    best
}

#[test]
fn audio_24k_mono_is_not_garbage() {
    let (toc, worst, rms) = rust_roundtrip(24_000, 1, Application::Audio, 64_000);
    assert_eq!(toc >> 3, 27, "24 kHz CELT SWB TOC expected, got {toc:#04x}");
    assert!(
        worst > 0.85,
        "24 kHz Audio mono diverges from the input: worst window {worst:.2}"
    );
    assert!(
        (0.7..1.4).contains(&rms),
        "24 kHz Audio mono level off: RMS ratio {rms:.2}"
    );
}

#[test]
fn audio_24k_stereo_is_not_garbage() {
    let (_toc, worst, _rms) = rust_roundtrip(24_000, 2, Application::Audio, 64_000);
    assert!(
        worst > 0.85,
        "24 kHz Audio stereo diverges from the input: worst window {worst:.2}"
    );
}

#[test]
fn restricted_low_delay_24k_is_not_garbage() {
    // RLD at 24 kHz is CELT too. It now matches the (pre-existing, imperfect)
    // 48 kHz RLD quality rather than decoding to noise.
    let (_toc, worst, _rms) = rust_roundtrip(24_000, 1, Application::RestrictedLowDelay, 64_000);
    assert!(
        worst > 0.25,
        "24 kHz RestrictedLowDelay diverges from the input: worst window {worst:.2}"
    );
}

#[test]
fn audio_48k_control() {
    let (toc, worst, rms) = rust_roundtrip(48_000, 1, Application::Audio, 64_000);
    assert_eq!(toc >> 3, 31, "48 kHz CELT FB TOC expected, got {toc:#04x}");
    assert!(
        worst > 0.85,
        "48 kHz Audio regressed: worst window {worst:.2}"
    );
    assert!(
        (0.7..1.4).contains(&rms),
        "48 kHz Audio level off: {rms:.2}"
    );
}

#[test]
fn libopus_24k_packets_still_decode() {
    let best = libopus_enc_rust_dec(24_000, 1, 64_000);
    assert!(
        best > 0.85,
        "opus-rs no longer decodes libopus 24 kHz packets: best corr {best:.2}"
    );
}
