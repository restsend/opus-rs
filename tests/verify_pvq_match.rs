use opus_rs::pvq::{alg_quant, alg_unquant};
use opus_rs::range_coder::RangeCoder;

#[test]
fn test_pvq_match() {
    let n = 8;
    let k = 10;
    let spread = 0;

    // Input vector X from C
    // float X[N]; for(int i=0; i<N; i++) X[i] = sin(i * 0.5f);
    let mut x_in = vec![0.0f32; n];
    for i in 0..n {
        x_in[i] = (i as f32 * 0.5f32).sin();
    }

    // Normalize input to match C input assumption?
    // In C, I passed X to alg_quant. C alg_quant might normalize internally?
    // C source: alg_quant calls celt_renormalise_vector(X, N, gain, arch) if gain is passed?
    // Wait, alg_quant in C modifies X to be the QUANTIZED vector. It does NOT normalize input initially?
    // Actually, alg_quant typically expects normalized input, or at least input that resembles the band energy.
    // The coefficients passed to alg_quant are usually already normalized by band energy.
    // But `pvq_search` projects X onto the K-pyramid.
    // In strict C implementation, `alg_quant` assumes X is already somewhat normalized or energy doesn't matter for specific pulse search if using projection.

    // Let's assume the Rust implementation expects raw coefficients as passed.
    // The C example passed unnormalized sin wave.

    let mut rc = RangeCoder::new_encoder(1024);
    let mut x_quant = x_in.clone();
    alg_quant(&mut x_quant, n, k, spread, 1, &mut rc, 1.0, true);

    rc.done();
    let nbytes_start = rc.offs as usize;
    let mut nbytes_end = rc.end_offs as usize;
    if rc.nend_bits > 0 {
        nbytes_end += 1;
    }

    let mut buffer = Vec::new();
    buffer.extend_from_slice(&rc.buf[0..nbytes_start]);

    if nbytes_end > 0 {
        let start_idx = rc.storage as usize - nbytes_end;
        buffer.extend_from_slice(&rc.buf[start_idx..]);
    }

    println!("Final buffer: {:?}", buffer);
    // ...

    // Expected quantized X from C:
    // 0.000000 0.235702 0.471405 0.471405 0.471405 0.471405 0.000000 -0.235702
    let expected_x = vec![
        0.000000, 0.235702, 0.471405, 0.471405, 0.471405, 0.471405, 0.000000, -0.235702,
    ];

    println!("Quantized X: {:?}", x_quant);

    for i in 0..n {
        assert!(
            (x_quant[i] - expected_x[i]).abs() < 1e-4,
            "Quantized value mismatch at index {}",
            i
        );
    }

    // Test Unquant
    let mut rc_dec = RangeCoder::new_decoder(buffer.clone());
    let mut x_unquant = vec![0.0f32; n];
    alg_unquant(&mut x_unquant, n, k, spread, 1, &mut rc_dec, 1.0);

    println!("Unquantized X: {:?}", x_unquant);

    for i in 0..n {
        assert!(
            (x_unquant[i] - expected_x[i]).abs() < 1e-4,
            "Unquantized value mismatch at index {}",
            i
        );
    }
}

#[test]
fn test_pvq_complex_match() {
    use opus_rs::pvq::alg_quant;
    use opus_rs::range_coder::RangeCoder;

    let n = 16;
    let k = 10;
    let spread = 2; // NORMAL

    let mut x_in = vec![0.0f32; n];
    for i in 0..n {
        x_in[i] = (i as f32 * 0.4f32).sin();
    }

    let mut rc = RangeCoder::new_encoder(1024);
    let mut x_quant = x_in.clone();
    alg_quant(&mut x_quant, n, k, spread, 1, &mut rc, 1.0, true);

    rc.done();

    // Check bitstream bytes
    // Reference from C (SMALL_FOOTPRINT): 62 80
    println!("rc.offs: {}", rc.offs);
    println!("rc.buf[0]: {:02X}", rc.buf[0]);
    assert_eq!(rc.offs, 2);
    assert_eq!(rc.buf[0], 0x62);
    assert_eq!(rc.buf[1], 0x80);

    // Check end bits
    // Reference from C (SMALL_FOOTPRINT, last 4 bytes of 1024 buffer): 00 08 87 C3
    println!(
        "End bits: {:02X} {:02X} {:02X} {:02X}",
        rc.buf[1020], rc.buf[1021], rc.buf[1022], rc.buf[1023]
    );
    assert_eq!(rc.buf[1020], 0x00);
    assert_eq!(rc.buf[1021], 0x08);
    assert_eq!(rc.buf[1022], 0x87);
    assert_eq!(rc.buf[1023], 0xC3);

    // Test Unquant
    let mut buffer = Vec::new();
    buffer.extend_from_slice(&rc.buf[0..rc.offs as usize]);
    if rc.end_offs < 1024 {
        buffer.extend_from_slice(&rc.buf[rc.end_offs as usize..1024]);
    }

    let mut rc_dec = RangeCoder::new_decoder(buffer);
    let mut x_unquant = vec![0.0f32; n];
    alg_unquant(&mut x_unquant, n, k, spread, 1, &mut rc_dec, 1.0);

    println!("Unquantized X: {:?}", x_unquant);
    // x_quant was normalized at the end of alg_quant.
    for i in 0..n {
        assert!(
            (x_unquant[i] - x_quant[i]).abs() < 1e-5,
            "Unquant mismatch at index {}",
            i
        );
    }
}
