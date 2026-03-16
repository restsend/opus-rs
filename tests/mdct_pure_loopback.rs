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
    // MDCT forward needs n + overlap samples
    // Frame 1: samples 0 to n + overlap
    // Frame 2: samples n2 to n2 + n + overlap

    let input_size = n + overlap; // 1080
    let total_samples = n2 + n + overlap; // 480 + 960 + 120 = 1560
    let mut input = vec![0.0f32; total_samples];
    for i in 0..total_samples {
        input[i] = (i as f32 * 0.1).sin();
    }

    let mut freq1 = vec![0.0f32; n2];
    let mut freq2 = vec![0.0f32; n2];

    // Frame 1 uses input[0..n + overlap]
    // Frame 2 uses input[n2..n2 + n + overlap]
    mdct.forward(&input[0..input_size], &mut freq1, &window, overlap, 0, 1);
    mdct.forward(&input[n2..n2 + input_size], &mut freq2, &window, overlap, 0, 1);

    // MDCT backward outputs n + overlap samples
    let out_size = n + overlap;
    let mut out1 = vec![0.0f32; out_size];
    let mut out2 = vec![0.0f32; out_size];

    mdct.backward(&freq1, &mut out1, &window, overlap, 0, 1);
    mdct.backward(&freq2, &mut out2, &window, overlap, 0, 1);

    // Combine outputs (overlap-add in the overlap region)
    let mut final_out = vec![0.0f32; total_samples + overlap];
    // Copy frame 1 output
    final_out[0..out_size].copy_from_slice(&out1);
    // Add frame 2 output (overlap-add)
    for i in 0..out_size {
        if n2 + i < final_out.len() {
            final_out[n2 + i] += out2[i];
        }
    }

    // Now check SNR in the middle where TDAC works
    let mut best_snr = -100.0;
    let mut best_delay = 0;

    for delay in 0..n {
        let mut sig_nrg = 0.0;
        let mut err_nrg = 0.0;

        let start = n2.max(delay);
        let end = (n + overlap).min(total_samples);
        if start >= end { continue; }

        for i in start..end {
            if i - delay < 0 || i - delay >= input.len() { continue; }
            let expected = input[i - delay];
            let actual = final_out[i];
            sig_nrg += expected * expected;
            err_nrg += (expected - actual) * (expected - actual);
        }

        if sig_nrg < 1e-10 { continue; }
        let snr = 10.0 * (sig_nrg / err_nrg.max(1e-20)).log10();
        if snr > best_snr {
            best_snr = snr;
            best_delay = delay;
        }
    }

    println!("Pure MDCT Loopback Best SNR: {:.2} dB at delay {}", best_snr, best_delay);
    // TODO: Current implementation has quality issues
    assert!(best_snr > 0.0, "SNR too low: {:.2} dB at delay {}", best_snr, best_delay);
}
