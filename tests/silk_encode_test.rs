/// SILK end-to-end encode test
/// Tests that silk_encode_frame runs without crashing and produces valid output
use opus_rs::range_coder::RangeCoder;
use opus_rs::silk::control_codec::*;
use opus_rs::silk::define::*;
use opus_rs::silk::enc_api::silk_encode_frame;
use opus_rs::silk::init_encoder::silk_init_encoder;
use opus_rs::silk::structs::*;


/// Helper: generate a voiced-like sine wave at ~200 Hz for 16 kHz
fn generate_voiced_signal(len: usize) -> Vec<i16> {
    let mut signal = vec![0i16; len];
    let freq = 200.0f64;
    let fs = 16000.0f64;
    for i in 0..len {
        let t = i as f64 / fs;
        signal[i] = (10000.0 * (2.0 * std::f64::consts::PI * freq * t).sin()) as i16;
    }
    signal
}

/// Helper: create and configure a WB SILK encoder at complexity 1 (simplest non-trivial)
fn create_wb_encoder(complexity: i32) -> SilkEncoderState {
    let mut enc = SilkEncoderState::default();
    silk_init_encoder(&mut enc, 0);
    silk_control_encoder(&mut enc, 16, 20, 20000, complexity);
    // Set SNR target manually (normally done by silk_control_SNR)
    enc.s_cmn.snr_db_q7 = 25 * 128; // 25 dB in Q7
    enc
}

#[test]
fn test_silk_encode_frame_no_crash() {
    // Test that silk_encode_frame runs without panicking for a simple voiced signal
    let mut enc = create_wb_encoder(1);

    // Generate input signal
    let frame_length = enc.s_cmn.frame_length as usize;
    let input = generate_voiced_signal(frame_length);

    // Create range coder with enough buffer
    let mut rc = RangeCoder::new_encoder(1275); // max SILK packet size

    let mut n_bytes_out: i32 = 0;

    // Encode first frame (with CODE_INDEPENDENTLY)
    let ret = silk_encode_frame(
        &mut enc,
        &input,
        &mut rc,
        &mut n_bytes_out,
        CODE_INDEPENDENTLY,
        8000, // max bits
        0,    // use_cbr
    );

    assert_eq!(ret, 0, "silk_encode_frame should return 0 (no error)");
    assert!(n_bytes_out > 0, "Encoded frame should produce some bytes, got {}", n_bytes_out);

    // Verify pulses are not all zero (something was quantized)
    let pulse_sum: i32 = enc.pulses.iter().map(|&p| p.abs() as i32).sum();
    println!("Encoded {} bytes, pulse energy = {}", n_bytes_out, pulse_sum);
    assert!(pulse_sum > 0, "Pulses should be non-zero for voiced input");
}

#[test]
fn test_silk_encode_two_frames() {
    // Test encoding two consecutive frames (tests state continuity)
    let mut enc = create_wb_encoder(1);
    let frame_length = enc.s_cmn.frame_length as usize;

    // Frame 1
    let input1 = generate_voiced_signal(frame_length);
    let mut rc = RangeCoder::new_encoder(1275);
    let mut n_bytes = 0i32;
    let ret = silk_encode_frame(
        &mut enc,
        &input1,
        &mut rc,
        &mut n_bytes,
        CODE_INDEPENDENTLY,
        8000,
        0,
    );
    assert_eq!(ret, 0);
    let bytes1 = n_bytes;

    // Frame 2 (conditional coding)
    enc.s_cmn.n_frames_encoded = 1;
    let input2 = generate_voiced_signal(frame_length);
    let mut rc2 = RangeCoder::new_encoder(1275);
    let mut n_bytes2 = 0i32;
    let ret = silk_encode_frame(
        &mut enc,
        &input2,
        &mut rc2,
        &mut n_bytes2,
        CODE_CONDITIONALLY,
        8000,
        0,
    );
    assert_eq!(ret, 0);
    println!("Frame 1: {} bytes, Frame 2: {} bytes", bytes1, n_bytes2);
    assert!(n_bytes2 > 0, "Second frame should also produce bytes");
}

#[test]
fn test_silk_encode_silent_input() {
    // Test encoding silence (should produce small output)
    let mut enc = create_wb_encoder(1);
    let frame_length = enc.s_cmn.frame_length as usize;

    let input = vec![0i16; frame_length];
    let mut rc = RangeCoder::new_encoder(1275);
    let mut n_bytes = 0i32;
    let ret = silk_encode_frame(
        &mut enc,
        &input,
        &mut rc,
        &mut n_bytes,
        CODE_INDEPENDENTLY,
        8000,
        0,
    );
    assert_eq!(ret, 0);
    println!("Silent frame: {} bytes", n_bytes);
    assert!(n_bytes > 0, "Even silence should produce some bytes");
}
