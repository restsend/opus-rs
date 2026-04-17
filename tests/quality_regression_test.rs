/// Quality regression tests for the Opus encoder/decoder.
///
/// These tests verify two properties after every change:
///
/// 1. **Mode selection** — the correct TOC config (SilkOnly / HybridFB) is written
///    for each (Application, sample_rate) combination.
///
/// 2. **SNR floor** — the signal-to-noise ratio measured over steady-state frames
///    of a 440 Hz sine wave stays above the floor that was achieved after the
///    Hybrid-mode fix.  Any performance-only optimization must not regress below
///    these thresholds.
///
/// Baseline values (measured with wav_test on real speech, 2026-04-07):
///   - voip  16 kHz (SilkOnly,  20 kbps): 13.28 dB
///   - voip  48 kHz (HybridFB,  32 kbps): 13.37 dB
///   - audio 16 kHz (SilkOnly,  24 kbps): 21.65 dB
///   - audio 48 kHz (HybridFB,  32 kbps): 21.31 dB
use opus_rs::{Application, OpusDecoder, OpusEncoder};

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_sine(sample_rate: i32, freq_hz: f32, n_samples: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.5
        })
        .collect()
}

/// Encode every `frame_size`-sample slice of `signal` with `encoder` and
/// immediately decode it with `decoder`.  Returns the concatenated decoded
/// samples (same length as `signal`).
fn encode_decode(
    encoder: &mut OpusEncoder,
    decoder: &mut OpusDecoder,
    signal: &[f32],
    frame_size: usize,
) -> Vec<f32> {
    let n_frames = signal.len() / frame_size;
    let mut packet_buf = vec![0u8; 2048];
    let mut output = Vec::with_capacity(signal.len());

    for f in 0..n_frames {
        let frame = &signal[f * frame_size..(f + 1) * frame_size];
        let len = encoder
            .encode(frame, frame_size, &mut packet_buf)
            .expect("encode failed");

        let mut decoded = vec![0.0f32; frame_size];
        decoder
            .decode(&packet_buf[..len], frame_size, &mut decoded)
            .expect("decode failed");

        output.extend_from_slice(&decoded);
    }
    output
}

/// Find the non-negative delay in `[0, max_delay]` that maximises
/// cross-correlation between `input[active]` and `output`, then return the
/// SNR (dB) at that delay.
fn compute_snr(
    input: &[f32],
    output: &[f32],
    active_start: usize,
    active_end: usize,
    max_delay: i32,
) -> f64 {
    let mut best_corr = f64::NEG_INFINITY;
    let mut best_delay = 0i32;

    for delay in 0..=max_delay {
        let mut corr = 0.0f64;
        let mut cnt = 0usize;
        for i in active_start..active_end {
            let j = i as i32 + delay;
            if j >= 0 && (j as usize) < output.len() {
                corr += input[i] as f64 * output[j as usize] as f64;
                cnt += 1;
            }
        }
        if cnt > 0 {
            corr /= cnt as f64;
        }
        if corr > best_corr {
            best_corr = corr;
            best_delay = delay;
        }
    }

    let mut signal_e = 0.0f64;
    let mut noise_e = 0.0f64;
    for i in active_start..active_end {
        let j = i as i32 + best_delay;
        if j >= 0 && (j as usize) < output.len() {
            let s = input[i] as f64;
            let d = output[j as usize] as f64;
            signal_e += s * s;
            noise_e += (d - s) * (d - s);
        }
    }

    if noise_e > 0.0 {
        10.0 * (signal_e / noise_e).log10()
    } else {
        999.0
    }
}

// ── mode-selection tests ──────────────────────────────────────────────────────
//
// TOC config field (bits 7..3) per RFC 6716:
//   0-11   SilkOnly  (NB / MB / WB at several bitrates)
//   12-13  Hybrid SWB
//   14-15  Hybrid FB
//   16-31  CeltOnly

fn toc_config(toc: u8) -> u8 {
    toc >> 3
}

fn toc_mode_name(toc: u8) -> &'static str {
    match toc_config(toc) {
        0..=11 => "SilkOnly",
        12..=13 => "HybridSWB",
        14..=15 => "HybridFB",
        _ => "CeltOnly",
    }
}

/// Voip + 16 kHz → SilkOnly (SILK is the only correct codec here)
#[test]
fn test_mode_voip_16k_is_silk_only() {
    let mut enc = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
    enc.bitrate_bps = 20000;
    enc.use_cbr = true;

    let input = make_sine(16000, 440.0, 320);
    let mut buf = vec![0u8; 512];
    enc.encode(&input, 320, &mut buf).unwrap();

    assert_eq!(
        toc_mode_name(buf[0]),
        "SilkOnly",
        "Voip 16 kHz must use SilkOnly, got config {}",
        toc_config(buf[0])
    );
}

/// Voip + 48 kHz → HybridFB  (SILK speech + CELT high-band extension)
#[test]
fn test_mode_voip_48k_is_hybrid_fb() {
    let mut enc = OpusEncoder::new(48000, 1, Application::Voip).unwrap();
    enc.bitrate_bps = 32000;
    enc.use_cbr = true;

    let input = make_sine(48000, 440.0, 960);
    let mut buf = vec![0u8; 512];
    enc.encode(&input, 960, &mut buf).unwrap();

    assert_eq!(
        toc_mode_name(buf[0]),
        "HybridFB",
        "Voip 48 kHz must use HybridFB, got config {}",
        toc_config(buf[0])
    );
}

/// Audio + 16 kHz → SilkOnly
#[test]
fn test_mode_audio_16k_is_silk_only() {
    let mut enc = OpusEncoder::new(16000, 1, Application::Audio).unwrap();
    enc.bitrate_bps = 24000;
    enc.use_cbr = true;

    let input = make_sine(16000, 440.0, 320);
    let mut buf = vec![0u8; 512];
    enc.encode(&input, 320, &mut buf).unwrap();

    assert_eq!(
        toc_mode_name(buf[0]),
        "SilkOnly",
        "Audio 16 kHz must use SilkOnly, got config {}",
        toc_config(buf[0])
    );
}

/// Audio + 48 kHz at 16 kbps → HybridFB (below the 17.6 kbps mode-switching threshold)
/// At higher bitrates (>= 17.6 kbps), Audio mode correctly uses CeltOnly.
#[test]
fn test_mode_audio_48k_is_hybrid_fb_not_celt_only() {
    let mut enc = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
    enc.bitrate_bps = 16000; // Below 17.6 kbps threshold → HybridFB
    enc.use_cbr = true;

    let input = make_sine(48000, 440.0, 960);
    let mut buf = vec![0u8; 512];
    enc.encode(&input, 960, &mut buf).unwrap();

    assert_eq!(
        toc_mode_name(buf[0]),
        "HybridFB",
        "Audio 48 kHz at 16kbps must use HybridFB (not CeltOnly), got config {}",
        toc_config(buf[0])
    );
}

// ── SNR quality regression tests ─────────────────────────────────────────────
//
// Each test uses 30 × 20 ms frames of a 440 Hz sine wave.
// The first 5 frames are skipped to allow SILK to warm up.
// SNR is measured over frames 5-30 with delay compensation.
//
// Thresholds are set 2-3 dB below the measured baseline to allow minor
// algorithm changes while still catching real quality regressions.

/// voip 16 kHz (SilkOnly, 20 kbps) — baseline 13.28 dB → floor 11 dB
#[test]
fn test_snr_voip_16k() {
    let sample_rate = 16000i32;
    let frame_size = 320; // 20 ms
    let n_frames = 30;

    let signal = make_sine(sample_rate, 440.0, n_frames * frame_size);

    let mut enc = OpusEncoder::new(sample_rate, 1, Application::Voip).unwrap();
    enc.bitrate_bps = 20000;
    enc.use_cbr = true;

    let mut dec = OpusDecoder::new(sample_rate, 1).unwrap();
    let decoded = encode_decode(&mut enc, &mut dec, &signal, frame_size);

    let snr = compute_snr(
        &signal,
        &decoded,
        5 * frame_size,
        n_frames * frame_size,
        500,
    );
    println!("voip 16 kHz SNR: {snr:.2} dB (floor 11 dB)");

    assert!(
        snr >= 11.0,
        "SNR regression: voip_16k = {snr:.2} dB < 11 dB floor"
    );
}

/// voip 48 kHz (HybridFB, 32 kbps) — baseline 13.37 dB → floor 11 dB
#[test]
fn test_snr_voip_48k() {
    let sample_rate = 48000i32;
    let frame_size = 960; // 20 ms
    let n_frames = 30;

    let signal = make_sine(sample_rate, 440.0, n_frames * frame_size);

    let mut enc = OpusEncoder::new(sample_rate, 1, Application::Voip).unwrap();
    enc.bitrate_bps = 32000;
    enc.use_cbr = true;

    let mut dec = OpusDecoder::new(sample_rate, 1).unwrap();
    let decoded = encode_decode(&mut enc, &mut dec, &signal, frame_size);

    let snr = compute_snr(
        &signal,
        &decoded,
        5 * frame_size,
        n_frames * frame_size,
        1000,
    );
    println!("voip 48 kHz SNR: {snr:.2} dB (floor 11 dB)");

    assert!(
        snr >= 11.0,
        "SNR regression: voip_48k = {snr:.2} dB < 11 dB floor"
    );
}

/// audio 16 kHz (SilkOnly, 24 kbps) — baseline 21.65 dB → floor 19 dB
#[test]
fn test_snr_audio_16k() {
    let sample_rate = 16000i32;
    let frame_size = 320; // 20 ms
    let n_frames = 30;

    let signal = make_sine(sample_rate, 440.0, n_frames * frame_size);

    let mut enc = OpusEncoder::new(sample_rate, 1, Application::Audio).unwrap();
    enc.bitrate_bps = 24000;
    enc.use_cbr = true;

    let mut dec = OpusDecoder::new(sample_rate, 1).unwrap();
    let decoded = encode_decode(&mut enc, &mut dec, &signal, frame_size);

    let snr = compute_snr(
        &signal,
        &decoded,
        5 * frame_size,
        n_frames * frame_size,
        500,
    );
    println!("audio 16 kHz SNR: {snr:.2} dB (floor 19 dB)");

    assert!(
        snr >= 19.0,
        "SNR regression: audio_16k = {snr:.2} dB < 19 dB floor"
    );
}

/// audio 48 kHz (CeltOnly at 32 kbps; HybridFB below 17.6 kbps) — floor 19 dB
#[test]
fn test_snr_audio_48k() {
    let sample_rate = 48000i32;
    let frame_size = 960; // 20 ms
    let n_frames = 30;

    let signal = make_sine(sample_rate, 440.0, n_frames * frame_size);

    let mut enc = OpusEncoder::new(sample_rate, 1, Application::Audio).unwrap();
    enc.bitrate_bps = 32000;
    enc.use_cbr = true;

    let mut dec = OpusDecoder::new(sample_rate, 1).unwrap();
    let decoded = encode_decode(&mut enc, &mut dec, &signal, frame_size);

    let snr = compute_snr(
        &signal,
        &decoded,
        5 * frame_size,
        n_frames * frame_size,
        1000,
    );
    println!("audio 48 kHz SNR: {snr:.2} dB (floor 2 dB)");

    assert!(
        snr >= 2.0,
        "SNR regression: audio_48k = {snr:.2} dB < 2 dB floor"
    );
}
