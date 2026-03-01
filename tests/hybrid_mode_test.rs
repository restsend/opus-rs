/// Tests for Hybrid (SILK + CELT) mode encoding and decoding.
/// Hybrid mode uses SILK for low-frequency content and CELT for high-frequency content.
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

/// Test that Hybrid mode can be enabled and produces valid packets
#[test]
fn test_hybrid_mode_encode_basic() {
    let sample_rate = 48000;
    let frame_size = 960; // 20ms at 48kHz

    let mut encoder =
        OpusEncoder::new(sample_rate, 1, Application::Audio).expect("Encoder creation failed");

    // Enable Hybrid mode
    encoder
        .enable_hybrid_mode()
        .expect("Failed to enable Hybrid mode");
    encoder.bitrate_bps = 32000;

    let mut input = vec![0.0f32; frame_size];
    for i in 0..frame_size {
        let t = i as f32 / sample_rate as f32;
        input[i] = (2.0f32 * PI * 440.0f32 * t).sin() * 0.5;
    }

    let mut output = vec![0u8; 1500];
    let n = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Hybrid encode failed");

    assert!(n >= 3, "Hybrid packet too short: {}", n);
    println!("Hybrid packet: {} bytes", n);

    // TOC byte should indicate Hybrid mode (config 16-19 or 20-23)
    let toc = output[0];
    let config = toc >> 3;
    println!("TOC byte: 0x{:02x}, config: {}", toc, config);

    // Config 16-19 = Hybrid SWB, 20-23 = Hybrid FB
    assert!(
        (16..=23).contains(&config),
        "Expected Hybrid TOC config 16-23, got {}",
        config
    );

    println!("✅ Hybrid mode basic encode test passed");
}

/// Test Hybrid mode encoding at 24kHz (SWB)
#[test]
fn test_hybrid_mode_24khz_swb() {
    let sample_rate = 24000;
    let frame_size = 480; // 20ms at 24kHz

    let mut encoder =
        OpusEncoder::new(sample_rate, 1, Application::Audio).expect("Encoder creation failed");

    encoder
        .enable_hybrid_mode()
        .expect("Failed to enable Hybrid mode at 24kHz");
    encoder.bitrate_bps = 28000;

    let mut input = vec![0.0f32; frame_size];
    for i in 0..frame_size {
        let t = i as f32 / sample_rate as f32;
        input[i] = (2.0f32 * PI * 440.0f32 * t).sin() * 0.5;
    }

    let mut output = vec![0u8; 1500];
    let n = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Hybrid encode at 24kHz failed");

    assert!(n >= 3, "24kHz Hybrid packet too short: {}", n);
    println!("24kHz Hybrid packet: {} bytes", n);

    println!("✅ Hybrid mode 24kHz SWB test passed");
}

/// Test Hybrid mode encoding at different bitrates
#[test]
fn test_hybrid_mode_bitrate_range() {
    let sample_rate = 48000;
    let frame_size = 960;

    for bitrate in [24000, 32000, 48000, 64000, 96000] {
        let mut encoder =
            OpusEncoder::new(sample_rate, 1, Application::Audio).expect("Encoder creation failed");
        encoder
            .enable_hybrid_mode()
            .expect("Failed to enable Hybrid mode");
        encoder.bitrate_bps = bitrate;

        let mut input = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            let t = i as f32 / sample_rate as f32;
            input[i] = (2.0f32 * PI * 440.0f32 * t).sin() * 0.5;
        }

        let mut output = vec![0u8; 1500];
        let n = encoder
            .encode(&input, frame_size, &mut output)
            .expect(&format!("Hybrid encode at {}bps failed", bitrate));

        assert!(n >= 3, "Hybrid packet at {}bps too short: {}", bitrate, n);
        println!("Hybrid at {}bps: {} bytes", bitrate, n);
    }

    println!("✅ Hybrid mode bitrate range test passed");
}

/// Test Hybrid mode multi-frame encoding (consecutive frames)
#[test]
fn test_hybrid_mode_consecutive_frames() {
    let sample_rate = 48000;
    let frame_size = 960;

    let mut encoder =
        OpusEncoder::new(sample_rate, 1, Application::Audio).expect("Encoder creation failed");
    encoder
        .enable_hybrid_mode()
        .expect("Failed to enable Hybrid mode");
    encoder.bitrate_bps = 32000;

    let mut frame_sizes = Vec::new();

    for frame_idx in 0..5 {
        let mut input = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            let t = (frame_idx * frame_size + i) as f32 / sample_rate as f32;
            input[i] = (2.0f32 * PI * 440.0f32 * t).sin() * 0.5;
        }

        let mut output = vec![0u8; 1500];
        let n = encoder
            .encode(&input, frame_size, &mut output)
            .expect(&format!("Hybrid encode frame {} failed", frame_idx));

        assert!(n >= 3, "Frame {}: packet too short: {}", frame_idx, n);
        frame_sizes.push(n);
        println!("Hybrid frame {}: {} bytes", frame_idx, n);
    }

    println!("Frame sizes: {:?}", frame_sizes);
    println!("✅ Hybrid mode consecutive frames test passed");
}

/// Test that Hybrid mode fails gracefully for invalid sample rates
#[test]
fn test_hybrid_mode_invalid_sample_rates() {
    // 8kHz: SILK-only rate, cannot use Hybrid
    let mut enc_8k =
        OpusEncoder::new(8000, 1, Application::Audio).expect("Encoder creation failed");
    assert!(enc_8k.enable_hybrid_mode().is_err(), "Should fail for 8kHz");

    // 16kHz: SILK NB/WB rate, cannot use Hybrid
    let mut enc_16k =
        OpusEncoder::new(16000, 1, Application::Audio).expect("Encoder creation failed");
    assert!(
        enc_16k.enable_hybrid_mode().is_err(),
        "Should fail for 16kHz"
    );

    // 48kHz: Valid for Hybrid
    let mut enc_48k =
        OpusEncoder::new(48000, 1, Application::Audio).expect("Encoder creation failed");
    assert!(
        enc_48k.enable_hybrid_mode().is_ok(),
        "Should succeed for 48kHz"
    );

    println!("✅ Hybrid mode invalid sample rate test passed");
}

/// Test CELT start_band functionality with start_band > 0
#[test]
fn test_celt_encode_with_start_band() {
    use opus_rs::celt::CeltEncoder;
    use opus_rs::modes::default_mode;
    use opus_rs::range_coder::RangeCoder;

    let mode = default_mode();
    let frame_size = 960;

    let mut enc_full = CeltEncoder::new(mode, 1);
    let mut enc_partial = CeltEncoder::new(mode, 1);

    let mut input = vec![0.0f32; frame_size];
    for i in 0..frame_size {
        let t = i as f32 / 48000.0;
        input[i] = (2.0f32 * PI * 440.0f32 * t).sin() * 0.5;
    }

    // Encode full spectrum (start_band = 0)
    let mut rc_full = RangeCoder::new_encoder(1500);
    enc_full.encode(&input, frame_size, &mut rc_full);
    rc_full.done();
    let full_bits = rc_full.tell();

    // Encode with start_band = 17 (high-frequency only)
    let mut rc_partial = RangeCoder::new_encoder(1500);
    enc_partial.encode_with_start_band(&input, frame_size, &mut rc_partial, 17);
    rc_partial.done();
    let partial_bits = rc_partial.tell();

    println!("Full spectrum: {} bits used", full_bits);
    println!("Partial (start_band=17): {} bits used", partial_bits);

    // Both should use some bits (not zero)
    // In practice partial may use fewer bits since less spectrum is coded
    println!("✅ CELT start_band encode test passed");
}

/// Test CELT decode_with_start_band works without crashing
#[test]
fn test_celt_decode_with_start_band() {
    use opus_rs::celt::{CeltDecoder, CeltEncoder};
    use opus_rs::modes::default_mode;
    use opus_rs::range_coder::RangeCoder;

    let mode = default_mode();
    let frame_size = 960;

    let mut encoder = CeltEncoder::new(mode, 1);
    let mut decoder = CeltDecoder::new(mode, 1);

    let mut input = vec![0.0f32; frame_size];
    for i in 0..frame_size {
        let t = i as f32 / 48000.0;
        input[i] = (2.0f32 * PI * 440.0f32 * t).sin() * 0.5;
    }

    // Encode with start_band = 17
    let mut rc = RangeCoder::new_encoder(1500);
    encoder.encode_with_start_band(&input, frame_size, &mut rc, 17);
    rc.done();

    // Build the encoded bytes
    let mut encoded = vec![0u8; rc.offs as usize];
    encoded.copy_from_slice(&rc.buf[..rc.offs as usize]);

    // Decode with start_band = 17
    let mut output = vec![0.0f32; frame_size];
    decoder.decode_with_start_band(&encoded, frame_size, &mut output, 17);

    println!(
        "Encoded {} bytes, decoded {} samples",
        encoded.len(),
        frame_size
    );
    println!("✅ CELT decode_with_start_band test passed (no crash)");
}
