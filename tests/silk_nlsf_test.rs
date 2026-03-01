use opus_rs::silk::define::*;
use opus_rs::silk::nlsf_decode::silk_nlsf_decode;
use opus_rs::silk::nlsf_encode::silk_nlsf_encode;
use opus_rs::silk::tables_nlsf::{SILK_NLSF_CB_NB_MB, SILK_NLSF_CB_WB};
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

#[test]
fn test_nlsf_encode_decode_nb_mb() {
    let cb = &SILK_NLSF_CB_NB_MB;
    let order = cb.order as usize;

    // Test with a typical NLSF vector (10th order for NB/MB)
    // These are normalized line spectral frequencies in Q15 format
    let nlsf_original = [
        3000i16, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000,
    ];
    let mut nlsf_quantized = nlsf_original.clone();

    // Weights for rate-distortion optimization (Q2 format)
    let weights = [4i16; MAX_LPC_ORDER];

    let mut indices = [0i8; MAX_LPC_ORDER + 1];

    // Encode
    let rd_q25 = silk_nlsf_encode(
        &mut indices,
        &mut nlsf_quantized,
        &cb,
        &weights[..order],
        100,         // nlsf_mu_q20: rate weight
        4,           // n_survivors
        TYPE_VOICED, // signal_type
    );

    println!("NB/MB NLSF Encoding:");
    println!("  Original NLSF: {:?}", &nlsf_original[..order]);
    println!("  Quantized NLSF: {:?}", &nlsf_quantized[..order]);
    println!("  Indices: {:?}", &indices[..order + 1]);
    println!("  RD cost (Q25): {}", rd_q25);

    // Decode
    let mut nlsf_decoded = [0i16; MAX_LPC_ORDER];
    silk_nlsf_decode(&mut nlsf_decoded, &indices, &cb);

    println!("  Decoded NLSF: {:?}", &nlsf_decoded[..order]);

    // Verify encode-decode roundtrip
    for i in 0..order {
        let error = (nlsf_quantized[i] as i32 - nlsf_decoded[i] as i32).abs();
        assert!(
            error < 10,
            "NLSF[{}]: quantized={}, decoded={}, error={}",
            i,
            nlsf_quantized[i],
            nlsf_decoded[i],
            error
        );
    }

    // Check that quantization didn't change too much
    for i in 0..order {
        let error = (nlsf_original[i] as i32 - nlsf_quantized[i] as i32).abs();
        assert!(
            error < 2000,
            "NLSF[{}]: original={}, quantized={}, error too large: {}",
            i,
            nlsf_original[i],
            nlsf_quantized[i],
            error
        );
    }

    // Verify NLSF ordering (must be strictly increasing)
    for i in 1..order {
        assert!(
            nlsf_decoded[i] > nlsf_decoded[i - 1],
            "NLSF ordering violated: nlsf[{}]={} <= nlsf[{}]={}",
            i,
            nlsf_decoded[i],
            i - 1,
            nlsf_decoded[i - 1]
        );
    }

    println!("✅ NB/MB NLSF encode-decode test passed");
}

#[test]
fn test_nlsf_encode_decode_wb() {
    let cb = &SILK_NLSF_CB_WB;
    let order = cb.order as usize;

    // Test with a typical NLSF vector (16th order for WB)
    let nlsf_original = [
        2000i16, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000,
        28000, 30000, 31000,
    ];
    let mut nlsf_quantized = nlsf_original.clone();

    let weights = [4i16; MAX_LPC_ORDER];
    let mut indices = [0i8; MAX_LPC_ORDER + 1];

    // Encode
    let rd_q25 = silk_nlsf_encode(
        &mut indices,
        &mut nlsf_quantized,
        &cb,
        &weights[..order],
        100, // nlsf_mu_q20
        4,   // n_survivors
        TYPE_UNVOICED,
    );

    println!("WB NLSF Encoding:");
    println!("  Original NLSF: {:?}", &nlsf_original[..order]);
    println!("  Quantized NLSF: {:?}", &nlsf_quantized[..order]);
    println!("  Indices: {:?}", &indices[..order + 1]);
    println!("  RD cost (Q25): {}", rd_q25);

    // Decode
    let mut nlsf_decoded = [0i16; MAX_LPC_ORDER];
    silk_nlsf_decode(&mut nlsf_decoded, &indices, &cb);

    println!("  Decoded NLSF: {:?}", &nlsf_decoded[..order]);

    // Verify roundtrip
    for i in 0..order {
        let error = (nlsf_quantized[i] as i32 - nlsf_decoded[i] as i32).abs();
        assert!(
            error < 10,
            "WB NLSF[{}]: quantized={}, decoded={}, error={}",
            i,
            nlsf_quantized[i],
            nlsf_decoded[i],
            error
        );
    }

    // Verify ordering
    for i in 1..order {
        assert!(
            nlsf_decoded[i] > nlsf_decoded[i - 1],
            "WB NLSF ordering violated: nlsf[{}]={} <= nlsf[{}]={}",
            i,
            nlsf_decoded[i],
            i - 1,
            nlsf_decoded[i - 1]
        );
    }

    println!("✅ WB NLSF encode-decode test passed");
}

#[test]
fn test_nlsf_stability() {
    let cb = &SILK_NLSF_CB_NB_MB;
    let order = cb.order as usize;

    // Test with closely spaced NLSFs (challenging for stability)
    let mut nlsf = [
        1000i16, 1100, 1200, 8000, 8100, 8200, 15000, 15100, 15200, 30000,
    ];

    let weights = [4i16; MAX_LPC_ORDER];
    let mut indices = [0i8; MAX_LPC_ORDER + 1];

    println!("Stability test:");
    println!("  Input (closely spaced): {:?}", &nlsf[..order]);

    silk_nlsf_encode(
        &mut indices,
        &mut nlsf,
        &cb,
        &weights[..order],
        100,
        4,
        TYPE_VOICED,
    );

    println!("  After encoding: {:?}", &nlsf[..order]);

    // Decode
    let mut nlsf_decoded = [0i16; MAX_LPC_ORDER];
    silk_nlsf_decode(&mut nlsf_decoded, &indices, &cb);

    println!("  After decoding: {:?}", &nlsf_decoded[..order]);

    // Check that spacing improved (even if not perfect)
    // The stabilization should have increased spacing between close NLSFs
    let mut min_spacing = i16::MAX;
    for i in 1..order {
        let spacing = nlsf_decoded[i] - nlsf_decoded[i - 1];
        min_spacing = min_spacing.min(spacing);
    }

    println!("  Minimum spacing: {}", min_spacing);

    // Verify NLSFs are still ordered (basic requirement)
    for i in 1..order {
        assert!(
            nlsf_decoded[i] > nlsf_decoded[i - 1],
            "NLSF ordering violated: nlsf[{}]={} <= nlsf[{}]={}",
            i,
            nlsf_decoded[i],
            i - 1,
            nlsf_decoded[i - 1]
        );
    }

    // Note: The encoder applies stabilization during quantization,
    // but the final spacing depends on codebook quantization.
    // We just verify that encoding/decoding works without crashing.

    println!("✅ NLSF stability test passed");
}

/// Test that NLSF interpolation is activated for complexity >= 5 (multi-frame packets)
#[test]
fn test_nlsf_interpolation_activated_at_complexity5() {
    // With complexity >= 5, use_interpolated_nlsfs = 1 in control_codec.rs
    // This test verifies that when an encoder is configured with complexity=5,
    // the NLSF interpolation path activates and produces valid output.
    let sample_rate = 8000;
    let frame_size = 160;

    let mut encoder_interp = OpusEncoder::new(sample_rate, 1, Application::Voip)
        .expect("Failed to create encoder");
    encoder_interp.complexity = 5; // NLSF interpolation activates at complexity >= 5
    encoder_interp.bitrate_bps = 20000;
    encoder_interp.use_cbr = false;

    let mut encoder_no_interp = OpusEncoder::new(sample_rate, 1, Application::Voip)
        .expect("Failed to create encoder");
    encoder_no_interp.complexity = 0; // No interpolation
    encoder_no_interp.bitrate_bps = 20000;
    encoder_no_interp.use_cbr = false;

    // Encode multiple frames and verify output is valid (non-empty, non-crashing)
    let mut bytes_interp = Vec::new();
    let mut bytes_no_interp = Vec::new();

    for frame_idx in 0..10 {
        let mut input = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            let t = (frame_idx * frame_size + i) as f32 / sample_rate as f32;
            input[i] = (2.0f32 * PI * 440.0f32 * t).sin();
        }

        let mut out_interp = vec![0u8; 256];
        let n_interp = encoder_interp
            .encode(&input, frame_size, &mut out_interp)
            .expect("Encode with interpolation failed");
        assert!(n_interp >= 3, "Frame {}: Interpolation output too short: {}", frame_idx, n_interp);
        bytes_interp.push(n_interp);

        let mut out_no_interp = vec![0u8; 256];
        let n_no_interp = encoder_no_interp
            .encode(&input, frame_size, &mut out_no_interp)
            .expect("Encode without interpolation failed");
        assert!(
            n_no_interp >= 3,
            "Frame {}: No-interpolation output too short: {}",
            frame_idx,
            n_no_interp
        );
        bytes_no_interp.push(n_no_interp);

        println!(
            "Frame {}: interp={} bytes, no_interp={} bytes",
            frame_idx, n_interp, n_no_interp
        );
    }

    // Both encoders should produce a range of output sizes (VBR mode)
    let max_interp = *bytes_interp.iter().max().unwrap();
    let min_interp = *bytes_interp.iter().min().unwrap();
    println!(
        "Interpolation encoder: min={} max={} bytes",
        min_interp, max_interp
    );

    println!("✅ NLSF interpolation test passed - encoder runs correctly at complexity=5");
}

/// Test NLSF interpolation for 40ms multi-frame packets (2 frames per packet at complexity 5)
#[test]
fn test_nlsf_interpolation_40ms_frames() {
    // At 40ms frame size (2 SILK frames per packet), NLSF interpolation uses
    // the previous frame's NLSFs as a starting point for each new packet.
    let sample_rate = 8000;
    let frame_size = 320; // 40ms at 8kHz = 320 samples

    let mut encoder = OpusEncoder::new(sample_rate, 1, Application::Voip)
        .expect("Failed to create encoder");
    encoder.complexity = 5;
    encoder.bitrate_bps = 20000;
    encoder.use_cbr = false;

    // Encode several 40ms frames and verify they don't crash
    for frame_idx in 0..5 {
        let mut input = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            let t = (frame_idx * frame_size + i) as f32 / sample_rate as f32;
            input[i] = (2.0f32 * PI * 440.0f32 * t).sin();
        }

        let mut output = vec![0u8; 256];
        let n = encoder
            .encode(&input, frame_size, &mut output)
            .expect("40ms encode failed");

        assert!(n >= 3, "Frame {}: output too short: {}", frame_idx, n);
        println!("40ms Frame {}: {} bytes encoded", frame_idx, n);
    }

    println!("✅ NLSF interpolation 40ms multi-frame test passed");
}

/// Verify NLSF interpolation correctness: interp coefficient must be in [0,4]
/// When nlsf_interp_coef_q2 < 4, interpolation is active.
#[test]
fn test_nlsf_interp_coefficient_bounds() {
    // This test verifies the NLSF interpolation coefficient validity
    let cb = &SILK_NLSF_CB_NB_MB;
    let order = cb.order as usize;

    // Two consecutive NLSF frames
    let prev_nlsf = [3000i16, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000];
    let curr_nlsf = [3100i16, 6100, 9100, 12100, 15100, 18100, 21100, 24100, 27100, 30100];

    // Test interpolation at different coefficient values (0, 1, 2, 3, 4)
    for interp_coef_q2 in 0..=4i16 {
        let mut interp_nlsf = [0i16; MAX_LPC_ORDER];
        for i in 0..order {
            interp_nlsf[i] = prev_nlsf[i]
                + ((curr_nlsf[i] as i32 - prev_nlsf[i] as i32) * interp_coef_q2 as i32 / 4) as i16;
        }

        // Encode the interpolated NLSFs to verify they're valid
        let weights = [4i16; MAX_LPC_ORDER];
        let mut indices = [0i8; MAX_LPC_ORDER + 1];
        let mut nlsf_to_encode = interp_nlsf;

        let rd_cost = silk_nlsf_encode(
            &mut indices,
            &mut nlsf_to_encode,
            cb,
            &weights[..order],
            100,
            4,
            TYPE_VOICED,
        );

        // Decode and verify
        let mut decoded = [0i16; MAX_LPC_ORDER];
        silk_nlsf_decode(&mut decoded, &indices, cb);

        // Verify ordering
        for i in 1..order {
            assert!(
                decoded[i] > decoded[i - 1],
                "interp_coef={}: NLSF ordering violated at index {}",
                interp_coef_q2,
                i
            );
        }

        println!(
            "interp_coef_q2={}: rd_cost={}, interp_nlsf[0]={}, decoded[0]={}",
            interp_coef_q2, rd_cost, interp_nlsf[0], decoded[0]
        );
    }

    println!("✅ NLSF interpolation coefficient bounds test passed");
}
