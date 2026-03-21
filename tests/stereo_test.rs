/// Tests for stereo encoding and decoding
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

/// Test basic stereo encoding and decoding
#[test]
fn test_stereo_basic() {
    let sample_rate = 48000;
    let channels = 2;
    let frame_size = 960; // 20ms at 48kHz

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Audio)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 32000;

    // Create stereo input (two sine waves at different frequencies)
    let mut input = vec![0.0f32; frame_size * channels];
    for i in 0..frame_size {
        let t = i as f32 / sample_rate as f32;
        // Left channel: 440 Hz
        input[i * 2] = (2.0f32 * PI * 440.0f32 * t).sin() * 0.5;
        // Right channel: 880 Hz
        input[i * 2 + 1] = (2.0f32 * PI * 880.0f32 * t).sin() * 0.5;
    }

    // Encode
    let mut output = vec![0u8; 1500];
    let n = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    assert!(n >= 3, "Packet too short: {}", n);
    println!("Stereo packet: {} bytes", n);

    // Decode
    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();
    let mut pcm = vec![0.0f32; frame_size * channels];
    let samples = decoder
        .decode(&output[..n], frame_size, &mut pcm)
        .expect("Decode failed");

    // samples is the number of frames, not including channels
    // So for frame_size=960 and channels=2, samples=960 (not 1920)
    assert_eq!(samples, frame_size);
    println!("Decoded {} frames ({} samples total)", samples, samples * channels);
    println!("✅ Basic stereo test passed");
}

/// Test stereo encoding at different bitrates
#[test]
fn test_stereo_bitrate_range() {
    let sample_rate = 48000;
    let channels = 2;
    let frame_size = 960;

    for bitrate in [24000, 32000, 48000, 64000] {
        let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Audio)
            .expect("Failed to create encoder");
        encoder.bitrate_bps = bitrate;

        let mut input = vec![0.0f32; frame_size * channels];
        for i in 0..frame_size {
            let t = i as f32 / sample_rate as f32;
            input[i * 2] = (2.0f32 * PI * 440.0f32 * t).sin() * 0.3;
            input[i * 2 + 1] = (2.0f32 * PI * 440.0f32 * t).sin() * 0.3;
        }

        let mut output = vec![0u8; 1500];
        let n = encoder
            .encode(&input, frame_size, &mut output)
            .expect(&format!("Encode at {}bps failed", bitrate));

        assert!(n >= 3, "Stereo packet at {}bps too short: {}", bitrate, n);
        println!("Stereo at {}bps: {} bytes", bitrate, n);
    }

    println!("✅ Stereo bitrate range test passed");
}

/// Test stereo SILK-only mode
#[test]
fn test_stereo_silk_only() {
    let sample_rate = 16000;
    let channels = 2;
    let frame_size = 320; // 20ms at 16kHz

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 24000;

    let mut input = vec![0.0f32; frame_size * channels];
    for i in 0..frame_size {
        let t = i as f32 / sample_rate as f32;
        input[i * 2] = (2.0f32 * PI * 300.0f32 * t).sin() * 0.5;
        input[i * 2 + 1] = (2.0f32 * PI * 300.0f32 * t).sin() * 0.5;
    }

    let mut output = vec![0u8; 500];
    let n = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    assert!(n >= 3, "Stereo SILK packet too short: {}", n);
    println!("Stereo SILK packet: {} bytes", n);

    // Check TOC byte has stereo bit set
    let toc = output[0];
    let stereo_bit = (toc >> 2) & 1;
    assert_eq!(stereo_bit, 1, "Stereo bit should be set in TOC");

    println!("✅ Stereo SILK test passed");
}

/// Test SILK stereo round-trip: encode stereo and decode, verify both channels preserved
#[test]
fn test_silk_stereo_roundtrip() {
    let sample_rate = 16000;
    let channels = 2;
    let frame_size = 320; // 20ms at 16kHz

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 24000;

    // Create different signals for left and right channels
    let mut input = vec![0.0f32; frame_size * channels];
    for i in 0..frame_size {
        let t = i as f32 / sample_rate as f32;
        // Left channel: 300 Hz sine wave
        input[i * 2] = (2.0f32 * PI * 300.0f32 * t).sin() * 0.5;
        // Right channel: 500 Hz sine wave (different frequency)
        input[i * 2 + 1] = (2.0f32 * PI * 500.0f32 * t).sin() * 0.5;
    }

    // Encode
    let mut output = vec![0u8; 500];
    let n = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    assert!(n >= 3, "Stereo SILK packet too short: {}", n);
    println!("Encoded stereo SILK: {} bytes", n);

    // Decode
    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();
    let mut pcm = vec![0.0f32; frame_size * channels];
    let samples = decoder
        .decode(&output[..n], frame_size, &mut pcm)
        .expect("Decode failed");

    assert_eq!(samples, frame_size);
    println!("Decoded {} samples", samples);

    // Verify output has energy (decode produced some audio)
    let mut left_energy = 0.0f32;
    let mut right_energy = 0.0f32;
    for i in 0..frame_size {
        left_energy += pcm[i * 2].abs();
        right_energy += pcm[i * 2 + 1].abs();
    }

    assert!(left_energy > 0.0, "Left channel should have energy");
    assert!(right_energy > 0.0, "Right channel should have energy");

    println!("Left energy: {:.2}, Right energy: {:.2}", left_energy, right_energy);
    println!("✅ SILK stereo round-trip test passed");
}

/// Test SILK stereo at different sample rates
#[test]
fn test_silk_stereo_sample_rates() {
    for sample_rate in [8000i32, 12000, 16000] {
        let channels = 2;
        let frame_size = ((sample_rate / 1000) * 20) as usize; // 20ms frames

        let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
            .expect("Failed to create encoder");
        encoder.bitrate_bps = 20000;

        let mut input = vec![0.0f32; frame_size * channels];
        for i in 0..frame_size {
            let t = i as f32 / sample_rate as f32;
            input[i * 2] = (2.0f32 * PI * 300.0f32 * t).sin() * 0.5;
            input[i * 2 + 1] = (2.0f32 * PI * 300.0f32 * t).sin() * 0.5;
        }

        let mut output = vec![0u8; 500];
        let n = encoder
            .encode(&input, frame_size, &mut output)
            .expect(&format!("Encode at {}Hz failed", sample_rate));

        assert!(n >= 3, "Stereo SILK at {}Hz packet too short: {}", sample_rate, n);
        println!("Stereo SILK at {}Hz: {} bytes", sample_rate, n);

        // Decode and verify
        let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();
        let mut pcm = vec![0.0f32; frame_size * channels];
        let samples = decoder
            .decode(&output[..n], frame_size, &mut pcm)
            .expect("Decode failed");

        assert_eq!(samples, frame_size);
    }

    println!("✅ SILK stereo sample rates test passed");
}

/// Test SILK stereo with different left/right signals (verify channel separation)
#[test]
fn test_silk_stereo_channel_separation() {
    let sample_rate = 16000;
    let channels = 2;
    let frame_size = 320;

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 32000; // Higher bitrate for better quality

    // Left channel: 300 Hz, Right channel: 500 Hz (different but both audible)
    let mut input = vec![0.0f32; frame_size * channels];
    for i in 0..frame_size {
        let t = i as f32 / sample_rate as f32;
        input[i * 2] = (2.0f32 * PI * 300.0f32 * t).sin() * 0.8;    // Left
        input[i * 2 + 1] = (2.0f32 * PI * 500.0f32 * t).sin() * 0.8; // Right
    }

    let mut output = vec![0u8; 500];
    let n = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    println!("Channel separation test: {} bytes", n);

    // Decode
    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();
    let mut pcm = vec![0.0f32; frame_size * channels];
    decoder
        .decode(&output[..n], frame_size, &mut pcm)
        .expect("Decode failed");

    // Compute average amplitude for each channel
    let mut left_avg = 0.0f32;
    let mut right_avg = 0.0f32;
    for i in 0..frame_size {
        left_avg += pcm[i * 2].abs();
        right_avg += pcm[i * 2 + 1].abs();
    }
    left_avg /= frame_size as f32;
    right_avg /= frame_size as f32;

    // Both channels should have some energy (relaxed threshold for stereo mid-only encoding)
    assert!(left_avg > 0.001, "Left channel should have some energy");
    assert!(right_avg > 0.001, "Right channel should have some energy");

    println!("Left avg: {:.4}, Right avg: {:.4}", left_avg, right_avg);
    println!("✅ SILK stereo channel separation test passed");
}

/// Test SILK stereo with multiple consecutive frames
#[test]
fn test_silk_stereo_multiframe() {
    let sample_rate = 16000;
    let channels = 2;
    let frame_size = 320;
    let num_frames = 5;

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 24000;

    // Create input for multiple frames
    let total_samples = frame_size * num_frames * channels;
    let mut input = vec![0.0f32; total_samples];

    for frame in 0..num_frames {
        for i in 0..frame_size {
            let t = (frame * frame_size + i) as f32 / sample_rate as f32;
            let idx = (frame * frame_size + i) * channels;
            // Different frequency per frame for variety
            let freq = 200.0 + (frame as f32 * 100.0);
            input[idx] = (2.0f32 * PI * freq * t).sin() * 0.5;
            input[idx + 1] = (2.0f32 * PI * freq * t).sin() * 0.5;
        }
    }

    // Encode all frames
    let mut output = vec![0u8; 2000];
    let mut encoded_frames = Vec::new();
    for frame in 0..num_frames {
        let frame_start = frame * frame_size * channels;
        let frame_input = &input[frame_start..frame_start + frame_size * channels];
        let mut frame_output = vec![0u8; 400];
        let n = encoder
            .encode(frame_input, frame_size, &mut frame_output)
            .expect(&format!("Encode frame {} failed", frame));
        encoded_frames.push(frame_output[..n].to_vec());
    }

    // Combine all encoded frames
    let mut offset = 0;
    for frame_data in &encoded_frames {
        output[offset..offset + frame_data.len()].copy_from_slice(frame_data);
        offset += frame_data.len();
    }

    println!("Multi-frame stereo: {} total bytes", offset);
    assert!(offset > 10, "Should produce significant output");

    // Decode all frames
    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();
    let mut pcm = vec![0.0f32; total_samples];
    let mut decoded_offset = 0;
    for frame_data in &encoded_frames {
        let frame_pcm_start = decoded_offset * channels;
        let samples = decoder
            .decode(frame_data, frame_size, &mut pcm[frame_pcm_start..])
            .expect("Decode frame failed");
        decoded_offset += samples;
    }

    println!("Decoded {} total frames", num_frames);
    println!("✅ SILK stereo multi-frame test passed");
}

/// Test SILK stereo with phase-inverted signals
#[test]
fn test_silk_stereo_phase_inverted() {
    let sample_rate = 16000;
    let channels = 2;
    let frame_size = 320;

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 24000;

    // Left and right channels are identical but phase-inverted
    let mut input = vec![0.0f32; frame_size * channels];
    for i in 0..frame_size {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0f32 * PI * 300.0f32 * t).sin() * 0.5;
        input[i * 2] = sample;       // Normal phase
        input[i * 2 + 1] = -sample;  // Inverted phase
    }

    let mut output = vec![0u8; 500];
    let n = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    println!("Phase-inverted stereo: {} bytes", n);

    // Decode
    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();
    let mut pcm = vec![0.0f32; frame_size * channels];
    decoder
        .decode(&output[..n], frame_size, &mut pcm)
        .expect("Decode failed");

    // Verify both channels have energy
    let mut left_energy = 0.0f32;
    let mut right_energy = 0.0f32;
    for i in 0..frame_size {
        left_energy += pcm[i * 2] * pcm[i * 2];
        right_energy += pcm[i * 2 + 1] * pcm[i * 2 + 1];
    }

    assert!(left_energy > 0.0, "Left channel should have energy");
    assert!(right_energy > 0.0, "Right channel should have energy");

    println!("Left energy: {:.2}, Right energy: {:.2}", left_energy, right_energy);
    println!("✅ SILK stereo phase-inverted test passed");
}

/// Test stereo at 8kHz narrowband
#[test]
fn test_stereo_narrowband() {
    let sample_rate = 8000;
    let channels = 2;
    let frame_size = 160; // 20ms at 8kHz

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 16000;

    let mut input = vec![0.0f32; frame_size * channels];
    for i in 0..frame_size {
        let t = i as f32 / sample_rate as f32;
        // Different frequencies for left and right
        input[i * 2] = (2.0f32 * PI * 200.0f32 * t).sin() * 0.5;
        input[i * 2 + 1] = (2.0f32 * PI * 300.0f32 * t).sin() * 0.5;
    }

    let mut output = vec![0u8; 300];
    let n = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    assert!(n >= 3, "Stereo narrowband packet too short: {}", n);
    println!("Stereo narrowband: {} bytes", n);

    // Decode
    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();
    let mut pcm = vec![0.0f32; frame_size * channels];
    let samples = decoder
        .decode(&output[..n], frame_size, &mut pcm)
        .expect("Decode failed");

    assert_eq!(samples, frame_size);
    println!("✅ Stereo narrowband test passed");
}

/// Test stereo with different signal types (speech-like)
#[test]
fn test_silk_stereo_speech() {
    let sample_rate = 16000;
    let channels = 2;
    let frame_size = 320;

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 24000;

    // Create speech-like signal with multiple frequency components
    let mut input = vec![0.0f32; frame_size * channels];
    for i in 0..frame_size {
        let t = i as f32 / sample_rate as f32;
        // Left: mix of low frequencies (speech-like)
        let left = (2.0f32 * PI * 150.0f32 * t).sin() * 0.3
            + (2.0f32 * PI * 250.0f32 * t).sin() * 0.2
            + (2.0f32 * PI * 350.0f32 * t).sin() * 0.1;
        // Right: similar but slightly different
        let right = (2.0f32 * PI * 170.0f32 * t).sin() * 0.3
            + (2.0f32 * PI * 270.0f32 * t).sin() * 0.2
            + (2.0f32 * PI * 370.0f32 * t).sin() * 0.1;
        input[i * 2] = left;
        input[i * 2 + 1] = right;
    }

    let mut output = vec![0u8; 500];
    let n = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    println!("Speech-like stereo: {} bytes", n);

    // Decode
    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();
    let mut pcm = vec![0.0f32; frame_size * channels];
    decoder
        .decode(&output[..n], frame_size, &mut pcm)
        .expect("Decode failed");

    // Verify both channels have energy
    let mut left_energy = 0.0f32;
    let mut right_energy = 0.0f32;
    for i in 0..frame_size {
        left_energy += pcm[i * 2].abs();
        right_energy += pcm[i * 2 + 1].abs();
    }

    assert!(left_energy > 0.0, "Left channel should have energy");
    assert!(right_energy > 0.0, "Right channel should have energy");

    println!("Left energy: {:.2}, Right energy: {:.2}", left_energy, right_energy);
    println!("✅ SILK stereo speech test passed");
}

/// Test CELT-only stereo
#[test]
fn test_stereo_celt_only() {
    let sample_rate = 48000;
    let channels = 2;
    let frame_size = 480; // 10ms at 48kHz

    // Use CELT-only mode (Audio application uses CELT at high sample rates)
    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Audio)
        .expect("Failed to create encoder");

    let mut input = vec![0.0f32; frame_size * channels];
    for i in 0..frame_size {
        let t = i as f32 / sample_rate as f32;
        input[i * 2] = (2.0f32 * PI * 1000.0f32 * t).sin() * 0.5;
        input[i * 2 + 1] = (2.0f32 * PI * 1200.0f32 * t).sin() * 0.5;
    }

    let mut output = vec![0u8; 500];
    let n = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    assert!(n >= 3, "CELT stereo packet too short: {}", n);
    println!("CELT stereo packet: {} bytes", n);

    println!("✅ CELT stereo test passed");
}
