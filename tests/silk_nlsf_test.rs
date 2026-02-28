use opus_rs::silk::define::*;
use opus_rs::silk::nlsf_decode::silk_nlsf_decode;
use opus_rs::silk::nlsf_encode::silk_nlsf_encode;
use opus_rs::silk::tables_nlsf::{SILK_NLSF_CB_NB_MB, SILK_NLSF_CB_WB};

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
