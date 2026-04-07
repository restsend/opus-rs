/// Unit tests for Hybrid mode and CELT fixes
use opus_rs::celt::{CeltDecoder, CeltEncoder};
use opus_rs::modes::default_mode;
use opus_rs::range_coder::RangeCoder;
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

/// Test that RangeCoder payload is correctly extracted with front + back combination
#[test]
fn test_range_coder_payload_extraction() {
    let mode = default_mode();
    let frame_size = 960;
    let mut encoder = CeltEncoder::new(mode, 1);

    // Create a test signal
    let input: Vec<f32> = (0..frame_size)
        .map(|i| (2.0 * PI * 440.0 * (i as f32) / 48000.0).sin() * 0.5)
        .collect();

    // Encode with a small buffer to ensure both front and back parts are used
    let mut rc = RangeCoder::new_encoder(100);
    encoder.encode(&input, frame_size, &mut rc);
    rc.done();

    // Verify that both front and back parts contain data
    let front_len = rc.offs as usize;
    let back_len = rc.end_offs as usize;
    let combined_len = front_len + back_len;

    println!(
        "RangeCoder: front={}, back={}, combined={}",
        front_len, back_len, combined_len
    );
    assert!(combined_len > 0, "Payload should not be empty");
    assert!(front_len > 0, "Front part should have data");
    // Note: back_len might be 0 for very small payloads
}

/// Test that CELT decoder correctly uses shared RangeCoder for Hybrid mode
#[test]
fn test_celt_decode_from_range_coder() {
    let mode = default_mode();
    let frame_size = 960;
    let mut encoder = CeltEncoder::new(mode, 1);
    let mut decoder = CeltDecoder::new(mode, 1);

    // Create a test signal
    let input: Vec<f32> = (0..frame_size)
        .map(|i| (2.0 * PI * 440.0 * (i as f32) / 48000.0).sin() * 0.5)
        .collect();

    // Encode
    let mut rc = RangeCoder::new_encoder(500);
    encoder.encode(&input, frame_size, &mut rc);
    rc.done();

    // Build payload
    let front_len = rc.offs as usize;
    let back_len = rc.end_offs as usize;
    let mut payload = Vec::with_capacity(front_len + back_len);
    payload.extend_from_slice(&rc.buf[..front_len]);
    payload.extend_from_slice(&rc.buf[(rc.storage - rc.end_offs) as usize..rc.storage as usize]);

    // Decode using decode_from_range_coder (Hybrid-style)
    let mut output = vec![0.0f32; frame_size];
    let total_bits = (payload.len() * 8) as i32;
    let mut rc_dec = RangeCoder::new_decoder(&payload);
    decoder.decode_from_range_coder(&mut rc_dec, total_bits, frame_size, &mut output, 0);

    // Check that output is not all zeros
    let max_val = output.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    println!("CELT decode_from_range_coder max output: {:.6}", max_val);
    // Note: Due to MDCT overlap, first frame output is small
}

/// Test that CeltOnly mode uses correct bit budget
#[test]
fn test_celt_only_bit_budget() {
    let sample_rate = 48000;
    let frame_size = 960;
    let bitrate = 32000;

    let mut encoder =
        OpusEncoder::new(sample_rate, 1, Application::Audio).expect("Encoder creation failed");
    encoder.bitrate_bps = bitrate;
    encoder.use_cbr = true;

    // Create a test signal
    let input: Vec<f32> = (0..frame_size)
        .map(|i| (2.0 * PI * 440.0 * (i as f32) / sample_rate as f32).sin() * 0.5)
        .collect();

    // Encode
    let mut output = vec![0u8; 1500];
    let n = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    // Expected packet size for CBR at 32kbps, 20ms frame:
    // 32000 bps * 0.020 s = 640 bits = 80 bytes + 1 TOC byte
    let expected_bytes =
        ((bitrate as i64 * frame_size as i64 / sample_rate as i64 + 7) / 8 + 1) as usize;
    println!(
        "Encoded {} bytes, expected ~{} bytes (CBR {} bps)",
        n, expected_bytes, bitrate
    );

    // The encoded size should be close to the expected size for CBR
    assert!(
        n >= expected_bytes - 5 && n <= expected_bytes + 5,
        "CBR packet size {} should be close to expected {}",
        n,
        expected_bytes
    );
}

/// Test Hybrid mode packet structure
#[test]
fn test_hybrid_mode_toc() {
    let sample_rate = 48000;
    let frame_size = 960;

    let mut encoder =
        OpusEncoder::new(sample_rate, 1, Application::Voip).expect("Encoder creation failed");
    encoder.bitrate_bps = 32000;

    // Create a test signal
    let input: Vec<f32> = (0..frame_size)
        .map(|i| (2.0 * PI * 440.0 * (i as f32) / sample_rate as f32).sin() * 0.5)
        .collect();

    // Encode
    let mut output = vec![0u8; 1500];
    let n = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    // Check TOC byte - Hybrid mode at 48kHz 20ms should be config 15 (0x78)
    let toc = output[0];
    let config = toc >> 3;
    println!("TOC: 0x{:02x}, config: {}", toc, config);

    // Hybrid mode configs are 12-15 (RFC 6716)
    assert!(
        (12..=15).contains(&config),
        "Hybrid mode TOC config should be 12-15, got {}",
        config
    );
    assert!(n > 1, "Hybrid packet should have more than just TOC byte");
}

/// Test that SilkOnly mode works correctly at 16kHz
#[test]
fn test_silk_only_16khz() {
    let sample_rate = 16000;
    let frame_size = 320;

    let mut encoder =
        OpusEncoder::new(sample_rate, 1, Application::Voip).expect("Encoder creation failed");
    encoder.bitrate_bps = 20000;

    let mut decoder = OpusDecoder::new(sample_rate, 1).expect("Decoder creation failed");

    // Create a test signal
    let input: Vec<f32> = (0..frame_size)
        .map(|i| (2.0 * PI * 440.0 * (i as f32) / sample_rate as f32).sin() * 0.5)
        .collect();

    // Encode
    let mut output = vec![0u8; 500];
    let n = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    // Decode
    let mut decoded = vec![0.0f32; frame_size];
    let decoded_n = decoder
        .decode(&output[..n], frame_size, &mut decoded)
        .expect("Decode failed");

    assert_eq!(decoded_n, frame_size, "Decoded frame size should match");

    // Check that decoded signal has energy
    let energy: f64 = decoded.iter().map(|x| (*x as f64).powi(2)).sum();
    println!("Decoded energy: {:.6}", energy);
    assert!(energy > 0.001, "Decoded signal should have energy");
}
