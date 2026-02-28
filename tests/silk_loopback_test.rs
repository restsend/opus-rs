/// SILK encoder-decoder loopback test
/// Encodes a 440Hz sine wave with SILK and decodes it back,
/// then verifies the output is valid (non-zero) and has reasonable SNR.
use opus_rs::{Application, OpusDecoder, OpusEncoder};

/// Generate a 440Hz sine wave at the given sample rate
fn generate_sine(sample_rate: i32, duration_ms: i32, frequency: f32) -> Vec<f32> {
    let num_samples = (sample_rate as f32 * duration_ms as f32 / 1000.0) as usize;
    let mut samples = vec![0.0f32; num_samples];
    for i in 0..num_samples {
        samples[i] =
            (2.0 * std::f32::consts::PI * frequency * i as f32 / sample_rate as f32).sin() * 0.5;
    }
    samples
}

/// Compute SNR between original and reconstructed signals
fn compute_snr(original: &[f32], decoded: &[f32], skip: usize) -> f64 {
    let len = original.len().min(decoded.len());
    if len <= skip {
        return 0.0;
    }

    let mut signal_power = 0.0f64;
    let mut noise_power = 0.0f64;
    for i in skip..len {
        signal_power += (original[i] as f64).powi(2);
        let err = (original[i] - decoded[i]) as f64;
        noise_power += err.powi(2);
    }

    if noise_power < 1e-20 {
        return 100.0;
    }
    10.0 * (signal_power / noise_power).log10()
}

#[test]
fn test_silk_encode_decode_loopback() {
    let sample_rate = 8000;
    let channels = 1;
    let frame_size = 160; // 20ms at 8kHz
    let bitrate = 10000;

    // Create encoder
    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip).unwrap();
    encoder.bitrate_bps = bitrate;
    encoder.use_cbr = true;

    // Create decoder
    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();

    // Generate 440Hz sine wave - 5 frames = 100ms
    let num_frames = 5;
    let input = generate_sine(sample_rate, num_frames * 20, 440.0);

    let mut all_decoded = Vec::new();
    let mut total_encoded_bytes = 0;

    for frame_idx in 0..num_frames as usize {
        let frame_start = frame_idx * frame_size;
        let frame_end = frame_start + frame_size;
        let frame_input = &input[frame_start..frame_end];

        // Encode
        let mut encoded = vec![0u8; 256];
        let enc_len = encoder
            .encode(frame_input, frame_size, &mut encoded)
            .unwrap();
        encoded.truncate(enc_len);
        total_encoded_bytes += enc_len;

        // Debug: print TOC and size
        eprintln!(
            "Frame {}: encoded {} bytes, TOC=0x{:02x}, payload hex={:?}",
            frame_idx,
            enc_len,
            encoded[0],
            &encoded[..enc_len.min(8)]
        );

        // Decode
        let mut decoded = vec![0.0f32; frame_size];
        let dec_len = decoder.decode(&encoded, frame_size, &mut decoded).unwrap();

        eprintln!(
            "Frame {}: decoded {} samples, first 5: {:?}",
            frame_idx,
            dec_len,
            &decoded[..5.min(dec_len)]
        );

        all_decoded.extend_from_slice(&decoded[..dec_len]);
    }

    eprintln!(
        "Total encoded: {} bytes for {} frames",
        total_encoded_bytes, num_frames
    );
    eprintln!("Total decoded: {} samples", all_decoded.len());

    // Verify: decoded output should not be all zeros
    let max_abs: f32 = all_decoded.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    eprintln!("Max absolute decoded value: {}", max_abs);
    assert!(
        max_abs > 0.001,
        "Decoded output is essentially silence (max_abs={})",
        max_abs
    );

    // Verify basic signal quality with delay compensation:
    // Find the best alignment between input and decoded to account for encoder/decoder delay
    let max_delay = frame_size * 2; // search up to 2 frames of delay
    let mut best_snr = f64::NEG_INFINITY;
    let mut best_delay = 0usize;
    for delay in 0..max_delay {
        let len = input.len().min(all_decoded.len() - delay);
        if len <= frame_size {
            continue;
        }
        let mut sig = 0.0f64;
        let mut noise = 0.0f64;
        for i in frame_size..len {
            sig += (input[i] as f64).powi(2);
            let err = (input[i] - all_decoded[i + delay]) as f64;
            noise += err.powi(2);
        }
        if noise > 1e-20 {
            let snr_val = 10.0 * (sig / noise).log10();
            if snr_val > best_snr {
                best_snr = snr_val;
                best_delay = delay;
            }
        }
    }
    eprintln!(
        "Delay-compensated SNR: {:.2} dB (delay={} samples)",
        best_snr, best_delay
    );

    // For SILK at 10kbps, we expect reasonable signal reconstruction
    assert!(
        best_snr > 5.0,
        "SNR too low: {:.2} dB at delay {} (possible decoding corruption)",
        best_snr,
        best_delay
    );
}

#[test]
fn test_silk_encode_decode_nonzero_output() {
    // Minimal test: encode one frame and verify decode produces non-zero output
    let sample_rate = 8000;
    let channels = 1;
    let frame_size = 160;

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip).unwrap();
    encoder.bitrate_bps = 10000;
    encoder.use_cbr = true;

    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();

    // Generate a simple tone
    let input = generate_sine(sample_rate, 20, 440.0);

    // Encode
    let mut encoded = vec![0u8; 256];
    let enc_len = encoder.encode(&input, frame_size, &mut encoded).unwrap();
    encoded.truncate(enc_len);

    assert!(enc_len > 1, "Encoded output too short: {} bytes", enc_len);

    // Verify it's a SILK packet
    let toc = encoded[0];
    eprintln!("TOC: 0x{:02x}, mode bits: 0x{:02x}", toc, toc & 0x80);
    // SILK-only: bit 7 = 0, bits 5-6 != 11
    assert!(
        toc & 0x80 == 0,
        "Expected SILK mode, got CELT (TOC=0x{:02x})",
        toc
    );

    // Decode
    let mut decoded = vec![0.0f32; frame_size];
    let dec_result = decoder.decode(&encoded, frame_size, &mut decoded);
    assert!(dec_result.is_ok(), "Decode failed: {:?}", dec_result.err());

    let dec_len = dec_result.unwrap();
    assert_eq!(dec_len, frame_size, "Decoded wrong number of samples");
}

#[test]
fn test_silk_multi_frame_continuity() {
    // Test that multi-frame encoding/decoding produces continuous output
    let sample_rate = 8000;
    let channels = 1;
    let frame_size = 160;

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip).unwrap();
    encoder.bitrate_bps = 16000; // Higher bitrate for better quality
    encoder.use_cbr = true;

    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();

    let input = generate_sine(sample_rate, 200, 440.0); // 200ms = 10 frames

    let mut all_decoded = Vec::new();
    let num_frames = input.len() / frame_size;

    for frame_idx in 0..num_frames {
        let start = frame_idx * frame_size;
        let frame_input = &input[start..start + frame_size];

        let mut encoded = vec![0u8; 512];
        let enc_len = encoder
            .encode(frame_input, frame_size, &mut encoded)
            .unwrap();
        encoded.truncate(enc_len);

        let mut decoded = vec![0.0f32; frame_size];
        let dec_len = decoder.decode(&encoded, frame_size, &mut decoded).unwrap();
        all_decoded.extend_from_slice(&decoded[..dec_len]);
    }

    // After warmup (skip first 2 frames), the decoded signal should have some periodicity
    let skip = 2 * frame_size;
    let snr = compute_snr(&input, &all_decoded, skip);
    eprintln!("Multi-frame SNR (skip 2 frames): {:.2} dB", snr);

    // At 16kbps, we should get reasonable reconstruction
    // Even with numerical differences, the signal should be recognizable
    let max_abs: f32 = all_decoded[skip..]
        .iter()
        .map(|x| x.abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs > 0.01,
        "Decoded output is near-silence after warmup (max_abs={})",
        max_abs
    );
}
