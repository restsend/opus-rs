use opus_rs::modes::default_mode;

#[test]
fn test_mdct_loopback() {
    let mode = default_mode();
    let mdct = &mode.mdct;
    let n = mdct.n;
    let n2 = n / 2;
    let overlap = mode.overlap;

    let mut input = vec![0.0f32; n2 + overlap];
    for i in 0..input.len() {
        input[i] = (i as f32 * 0.1).sin();
    }

    let mut freq = vec![0.0f32; n2];
    mdct.forward(&input, &mut freq, &mode.window, overlap, 0, 1);

    let mut output = vec![0.0f32; n2 + overlap];
    mdct.backward(&freq, &mut output, &mode.window, overlap, 0, 1);

    // Check loopback SNR in the valid range
    let mut sig_nrg = 0.0;
    let mut err_nrg = 0.0;

    // Range where loopback should be perfect (middle of the windowed part)
    // For a single frame, TDAC is not complete, but the windowing is applied twice.
    // However, the MDCT logic itself should be consistent.
    for i in overlap..n2 {
        let expected = input[i];
        let actual = output[i];
        if i < overlap + 5 {
            println!("Index {}: expected={}, actual={}", i, expected, actual);
        }
        sig_nrg += expected * expected;
        err_nrg += (expected - actual) * (expected - actual);
    }

    let snr = 10.0 * (sig_nrg / err_nrg).log10();
    println!("MDCT Loopback SNR: {:.2} dB", snr);
    assert!(snr > 30.0, "SNR too low: {:.2} dB", snr);
}
