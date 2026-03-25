/// Diagnostic test to compare Rust SILK encoder output with C reference
/// Decodes both bitstreams to compare the encoded parameters
use opus_rs::range_coder::RangeCoder;
use opus_rs::silk::decode_indices::silk_decode_indices;
use opus_rs::silk::decoder_structs::*;
use opus_rs::silk::define::*;
use std::f32::consts::PI;

use opus_rs::silk::decode_pulses::silk_decode_pulses;
use opus_rs::silk::init_decoder::silk_decoder_set_fs;

/// Manually decode SILK bitstream parameters for comparison
fn decode_silk_params(payload: &[u8], fs_khz: i32) {
    let mut rc = RangeCoder::new_decoder(payload);

    // 1. VAD/LBRR flags (C uses ec_dec_bit_logp(1) for each bit)
    let n_frames_per_packet = 1;
    let mut vad_flag = 0i32;
    for _ in 0..n_frames_per_packet {
        let v = rc.decode_bit_logp(1);
        vad_flag = if v { 1 } else { 0 };
    }
    let lbrr_flag = if rc.decode_bit_logp(1) { 1i32 } else { 0 };
    println!("  VAD/LBRR: vad={} lbrr={}", vad_flag, lbrr_flag);

    // 2. Decode indices using properly initialized decoder state
    let mut dec_state = SilkDecoderState::default();
    silk_decoder_set_fs(&mut dec_state, fs_khz, fs_khz * 1000);
    dec_state.first_frame_after_reset = 1;

    // Set vad_flags for frame 0 based on decoded flag
    dec_state.vad_flags[0] = vad_flag;

    silk_decode_indices(&mut dec_state, &mut rc, 0, 0, CODE_INDEPENDENTLY);

    let idx = &dec_state.indices;
    println!(
        "  signal_type={} quant_offset_type={}",
        idx.signal_type, idx.quant_offset_type
    );
    println!(
        "  gains_indices: {:?}",
        &idx.gains_indices[..dec_state.nb_subfr as usize]
    );
    println!(
        "  nlsf_indices: {:?}",
        &idx.nlsf_indices[..dec_state.lpc_order as usize + 1]
    );
    println!("  nlsf_interp_coef_q2={}", idx.nlsf_interp_coef_q2);
    if idx.signal_type == TYPE_VOICED as i8 {
        println!(
            "  lag_index={} contour_index={}",
            idx.lag_index, idx.contour_index
        );
        println!("  per_index={}", idx.per_index);
        println!(
            "  ltp_index: {:?}",
            &idx.ltp_index[..dec_state.nb_subfr as usize]
        );
        println!("  ltp_scale_index={}", idx.ltp_scale_index);
    }
    println!("  seed={}", idx.seed);

    let bits_used = rc.tell();
    println!("  bits_used_after_indices={}", bits_used);

    // 3. Decode pulses
    let frame_length = dec_state.frame_length;
    let mut pulses = vec![0i16; frame_length as usize];
    silk_decode_pulses(
        &mut rc,
        &mut pulses,
        idx.signal_type as i32,
        idx.quant_offset_type as i32,
        frame_length,
    );

    let bits_after_pulses = rc.tell();
    println!("  bits_used_after_pulses={}", bits_after_pulses);

    // Print first subframe pulses (first 40 samples for NB)
    let subfr_len = dec_state.subfr_length as usize;
    for sf in 0..dec_state.nb_subfr as usize {
        let start = sf * subfr_len;
        let p = &pulses[start..start + subfr_len.min(20)];
        let sum: i32 = pulses[start..start + subfr_len]
            .iter()
            .map(|&x| x.abs() as i32)
            .sum();
        println!("  subfr[{}] pulse_sum={} first20={:?}", sf, sum, p);
    }
}

#[test]
fn test_silk_bitstream_diagnostic() {
    // C reference output (from test_silk_bitstream_vs_c_reference)
    let c_opus = [
        0x0bu8, 0x01, 0x84, 0xc1, 0xc1, 0xc7, 0xb6, 0x6f, 0x5e, 0x06, 0xa4, 0xb7, 0x28, 0xc8, 0x1c,
        0x95, 0x61, 0x20, 0xe0, 0x78, 0x1c, 0x26, 0x4a, 0x17, 0x60,
    ];

    // Encode with Rust to get its output
    use opus_rs::{Application, OpusEncoder};
    let mut encoder =
        OpusEncoder::new(8000, 1, Application::Voip).expect("Failed to create encoder");
    encoder.complexity = 0;
    encoder.bitrate_bps = 10000;
    encoder.use_cbr = true;

    let mut input = vec![0.0f32; 160];
    for i in 0..160 {
        input[i] = (2.0f32 * PI * 440.0f32 * i as f32 / 8000.0f32).sin();
    }

    let mut rust_opus = vec![0u8; 25];
    let rust_bytes = encoder
        .encode(&input, 160, &mut rust_opus)
        .expect("Encode failed");

    println!("=== Rust output ({} bytes) ===", rust_bytes);
    print!("  Hex: ");
    for i in 0..rust_bytes {
        print!("{:02x}", rust_opus[i]);
    }
    println!();

    println!("\n=== C reference ({} bytes) ===", c_opus.len());
    print!("  Hex: ");
    for &b in &c_opus {
        print!("{:02x}", b);
    }
    println!();

    // Find first difference
    for i in 0..rust_bytes.min(c_opus.len()) {
        if rust_opus[i] != c_opus[i] {
            println!(
                "\nFirst difference at byte {}: Rust=0x{:02x} C=0x{:02x}",
                i, rust_opus[i], c_opus[i]
            );
            break;
        }
    }

    // Both use Code 3: TOC + count byte + SILK payload
    let rust_silk = &rust_opus[2..rust_bytes];
    let c_silk = &c_opus[2..];

    println!("\n=== Decoding Rust SILK parameters ===");
    decode_silk_params(rust_silk, 8);

    println!("\n=== Decoding C SILK parameters ===");
    decode_silk_params(c_silk, 8);
}

#[test]
fn test_decode_exact_compare_frame0() {
    // Decode the C reference frame 0 from bitstream_exact_compare test
    // (complexity=10, 12kbps, NB 8kHz)
    // Full C reference frame 0: 0b018455a4e3c2206bd13d16c0f1d332bbe6fca978f3eac09e538202d180
    // After skipping TOC (0x0b) and count byte (0x01), SILK payload starts at byte 2
    let c_silk_hex = "8455a4e3c2206bd13d16c0f1d332bbe6fca978f3eac09e538202d180";
    let c_silk: Vec<u8> = (0..c_silk_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&c_silk_hex[i..i + 2], 16).unwrap())
        .collect();

    println!("=== Decoding C exact_compare frame 0 (complexity=10, 12kbps) ===");
    decode_silk_params(&c_silk, 8);

    // Also decode corresponding Rust frame 0 for comparison
    use opus_rs::{Application, OpusEncoder};
    let mut encoder = OpusEncoder::new(8000, 1, Application::Voip).expect("Failed");
    encoder.bitrate_bps = 12000;
    encoder.use_cbr = true;
    encoder.complexity = 10;

    let mut input = vec![0.0f32; 160];
    for i in 0..160 {
        let val = (2.0f64 * std::f64::consts::PI * 440.0 * i as f64 / 8000.0).sin();
        let i16_val = (val * 16383.0) as i16;
        input[i] = i16_val as f32 / 32768.0;
    }

    let mut pkt_buf = vec![0u8; 200];
    let pkt_len = encoder
        .encode(&input, 160, &mut pkt_buf)
        .expect("Encode failed");
    let rust_silk = &pkt_buf[2..pkt_len]; // Skip TOC + count byte

    println!("\n=== Decoding Rust exact_compare frame 0 ===");
    decode_silk_params(rust_silk, 8);
}
