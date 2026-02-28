use opus_rs::mdct::MdctLookup;

#[test]
fn test_mdct_pure_loopback() {
    let n = 960;
    let overlap = 120;
    let n2 = n / 2;
    let mdct = MdctLookup::new(n, 0);

    let mut window = vec![0.0; overlap];
    for i in 0..overlap {
        let val = (0.5 * std::f32::consts::PI * (i as f32 + 0.5) / overlap as f32).sin();
        let val2 = (0.5 * std::f32::consts::PI * val * val).sin();
        window[i] = val2;
    }

    // Two frames to test TDAC
    // Frame 1: samples 0 to N
    // Frame 2: samples N2 to N2+N
    // The overlap is between samples N2 and N.
    
    let total_samples = n + n2;
    let mut input = vec![0.0f32; total_samples];
    for i in 0..total_samples {
        input[i] = (i as f32 * 0.1).sin();
    }

    let mut freq1 = vec![0.0f32; n2];
    let mut freq2 = vec![0.0f32; n2];
    
    // In CELT, forward expects input buffer of size N.
    // Frame 1 uses input[0..N]
    // Frame 2 uses input[N2..N2+N]
    
    mdct.forward(&input[0..n], &mut freq1, &window, overlap, 0, 1);
    mdct.forward(&input[n2..n2+n], &mut freq2, &window, overlap, 0, 1);

    let mut final_out = vec![0.0f32; total_samples + overlap + 100]; // Extra space
    
    mdct.backward(&freq1, &mut final_out[0..], &window, overlap, 0, 1);
    mdct.backward(&freq2, &mut final_out[n2..], &window, overlap, 0, 1);

    // Now check SNR in the middle where TDAC works
    // Frame 1 covers [0, n]
    // Frame 2 covers [n2, n+n2]
    // Overlap is [n2, n]
    
    let mut best_snr = -100.0;
    let mut best_delay = 0;

    for delay in 0..n {
        let mut sig_nrg = 0.0;
        let mut err_nrg = 0.0;
        
        let start = n2.max(delay);
        let end = n;
        if start >= end { continue; }

        for i in start..end {
            let expected = input[i - delay];
            let actual = final_out[i];
            sig_nrg += expected * expected;
            err_nrg += (expected - actual) * (expected - actual);
        }

        let snr = 10.0 * (sig_nrg / err_nrg).log10();
        if snr > best_snr {
            best_snr = snr;
            best_delay = delay;
        }
    }

    println!("Pure MDCT Loopback Best SNR: {:.2} dB at delay {}", best_snr, best_delay);
    assert!(best_snr > 100.0, "SNR too low: {:.2} dB at delay {}", best_snr, best_delay);
}
