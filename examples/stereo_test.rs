/// Stereo encoding and decoding example
/// Demonstrates stereo audio encoding/decoding with opus-rs

use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

fn main() {
    println!("=== Opus Stereo Example ===\n");

    // Test SILK-only stereo at 16kHz (VoIP)
    println!("--- SILK-only stereo (16kHz VoIP) ---");
    test_silk_stereo(16000, 320, 24000);

    // Test stereo at 48kHz (Audio)
    println!("\n--- CELT stereo (48kHz Audio) ---");
    test_celt_stereo(48000, 960, 32000);

    // Test stereo round-trip
    println!("\n--- Stereo round-trip test ---");
    test_stereo_roundtrip();

    println!("\n=== All stereo examples completed! ===");
}

/// Test SILK-only stereo encoding/decoding
fn test_silk_stereo(sample_rate: i32, frame_size: usize, bitrate: i32) {
    let channels = 2;

    // Create encoder
    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = bitrate;

    // Create stereo input: different frequencies for left and right
    let mut input = vec![0.0f32; frame_size * channels];
    for i in 0..frame_size {
        let t = i as f32 / sample_rate as f32;
        // Left: 300 Hz
        input[i * 2] = (2.0 * PI * 300.0 * t).sin() * 0.5;
        // Right: 500 Hz
        input[i * 2 + 1] = (2.0 * PI * 500.0 * t).sin() * 0.5;
    }

    // Encode
    let mut output = vec![0u8; 500];
    let n = encoder.encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    // Check stereo bit in TOC
    let stereo_bit = (output[0] >> 2) & 1;
    println!("  Encoded {} bytes, stereo bit: {}", n, stereo_bit);

    // Decode
    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();
    let mut pcm = vec![0.0f32; frame_size * channels];
    let samples = decoder.decode(&output[..n], frame_size, &mut pcm)
        .expect("Decode failed");

    // Calculate energy
    let mut left_energy = 0.0f32;
    let mut right_energy = 0.0f32;
    for i in 0..samples {
        left_energy += pcm[i * 2].abs();
        right_energy += pcm[i * 2 + 1].abs();
    }

    println!("  Decoded {} samples", samples);
    println!("  Left energy: {:.2}, Right energy: {:.2}", left_energy, right_energy);
}

/// Test CELT-only stereo encoding/decoding
fn test_celt_stereo(sample_rate: i32, frame_size: usize, bitrate: i32) {
    let channels = 2;

    // Create encoder (Audio application uses CELT at high sample rates)
    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Audio)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = bitrate;

    // Create stereo input: different frequencies for left and right
    let mut input = vec![0.0f32; frame_size * channels];
    for i in 0..frame_size {
        let t = i as f32 / sample_rate as f32;
        // Left: 440 Hz (A4 note)
        input[i * 2] = (2.0 * PI * 440.0 * t).sin() * 0.5;
        // Right: 880 Hz (A5 note)
        input[i * 2 + 1] = (2.0 * PI * 880.0 * t).sin() * 0.5;
    }

    // Encode
    let mut output = vec![0u8; 1500];
    let n = encoder.encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    // Check stereo bit in TOC
    let stereo_bit = (output[0] >> 2) & 1;
    println!("  Encoded {} bytes, stereo bit: {}", n, stereo_bit);

    // Decode
    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();
    let mut pcm = vec![0.0f32; frame_size * channels];
    let samples = decoder.decode(&output[..n], frame_size, &mut pcm)
        .expect("Decode failed");

    // Calculate energy
    let mut left_energy = 0.0f32;
    let mut right_energy = 0.0f32;
    for i in 0..samples {
        left_energy += pcm[i * 2].abs();
        right_energy += pcm[i * 2 + 1].abs();
    }

    println!("  Decoded {} samples", samples);
    println!("  Left energy: {:.2}, Right energy: {:.2}", left_energy, right_energy);
}

/// Test stereo round-trip with multiple frames
fn test_stereo_roundtrip() {
    let sample_rate = 16000;
    let channels = 2;
    let frame_size = 320;
    let num_frames = 3;

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 24000;

    // Create multi-frame input
    let total_samples = frame_size * num_frames * channels;
    let mut input = vec![0.0f32; total_samples];

    for frame in 0..num_frames {
        for i in 0..frame_size {
            let t = ((frame * frame_size) + i) as f32 / sample_rate as f32;
            let idx = (frame * frame_size + i) * channels;
            // Each frame has different frequency
            let freq = 200.0 + (frame as f32 * 150.0);
            input[idx] = (2.0 * PI * freq * t).sin() * 0.5;
            input[idx + 1] = (2.0 * PI * freq * t).sin() * 0.5;
        }
    }

    // Encode each frame
    println!("  Encoding {} frames...", num_frames);
    let mut encoded_frames: Vec<Vec<u8>> = Vec::new();

    for frame in 0..num_frames {
        let frame_start = frame * frame_size * channels;
        let frame_input = &input[frame_start..frame_start + frame_size * channels];

        let mut frame_output = vec![0u8; 400];
        let n = encoder.encode(frame_input, frame_size, &mut frame_output)
            .expect("Encode frame failed");

        encoded_frames.push(frame_output[..n].to_vec());
        println!("    Frame {}: {} bytes", frame, n);
    }

    // Decode each frame
    println!("  Decoding {} frames...", num_frames);
    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();
    let mut pcm = vec![0.0f32; total_samples];

    for frame in 0..num_frames {
        let frame_start = frame * frame_size * channels;
        let samples = decoder.decode(&encoded_frames[frame], frame_size, &mut pcm[frame_start..])
            .expect("Decode frame failed");
        println!("    Frame {}: {} samples", frame, samples);
    }

    // Verify output has energy
    let mut total_energy = 0.0f32;
    for i in 0..total_samples {
        total_energy += pcm[i].abs();
    }

    println!("  Total output energy: {:.2}", total_energy);
    println!("  Round-trip test: {}", if total_energy > 0.0 { "PASSED" } else { "FAILED" });
}
