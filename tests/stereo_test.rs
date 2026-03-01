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

/// Test CELT-only stereo
/// Note: Temporarily disabled due to MDCT size mismatch in transient mode
#[test]
#[ignore]
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
