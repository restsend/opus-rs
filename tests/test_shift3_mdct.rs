#[test]
fn test_mdct_shift3() {
    use opus_rs::modes::default_mode;

    let mode = default_mode();

    // Test MDCT with shift=3 (used for long blocks in CELT)
    let shift = 3;
    let n = 1920 >> shift; // n = 240 for shift=3
    let overlap = 120;

    eprintln!(
        "Testing MDCT with shift={}, n={}, overlap={}",
        shift, n, overlap
    );

    // Create a simple test signal (overlap + n samples)
    let input_len = overlap + n;
    let mut input = vec![0.0f32; input_len];
    for i in 0..input_len {
        input[i] = ((i as f32) * 0.1).sin();
    }

    eprintln!(
        "Input: {} samples, first 10: {:?}",
        input_len,
        &input[..10.min(input_len)]
    );

    // Forward MDCT
    let mut freq = vec![0.0f32; n];
    mode.mdct.forward(
        &input,
        &mut freq,
        &mode.window,
        overlap,
        shift,
        1, // stride=1 for long block
    );

    eprintln!(
        "MDCT output: {} coeffs, max={}",
        freq.len(),
        freq.iter().map(|f| f.abs()).fold(0.0f32, f32::max)
    );
    eprintln!("First 10: {:?}", &freq[..10.min(freq.len())]);

    // Backward MDCT
    let mut output = vec![0.0f32; n + overlap];
    mode.mdct.backward(
        &freq,
        &mut output,
        &mode.window,
        overlap,
        shift,
        1, // stride=1
    );

    eprintln!(
        "MDCT backward output: {} samples, max={}",
        output.len(),
        output.iter().map(|f| f.abs()).fold(0.0f32, f32::max)
    );
    eprintln!("Full output: {:?}", &output);

    // The reconstructed signal should be at output[overlap/2 + overlap..]
    // because output[0..overlap] is the overlap tail from previous frame (zero in our case)
    // and output[overlap/2..overlap/2+overlap] is the window-overlapped region
    // The actual reconstructed samples are at output[overlap/2+overlap..]

    let reconstruct_start = overlap / 2 + overlap;
    if reconstruct_start < output.len() {
        let available = output.len() - reconstruct_start;
        eprintln!(
            "Reconstructed samples starting at {}: {} available",
            reconstruct_start, available
        );
        eprintln!(
            "First 10 reconstructed: {:?}",
            &output[reconstruct_start..reconstruct_start + 10.min(available)]
        );

        // Compare with input
        let input_start = overlap; // Align with what was encoded
        let mut snr_sq_sig = 0.0f32;
        let mut snr_sq_err = 0.0f32;
        for i in 0..available.min(input_len - input_start) {
            let sig = input[input_start + i];
            let recon = output[reconstruct_start + i];
            snr_sq_sig += sig * sig;
            snr_sq_err += (recon - sig) * (recon - sig);
        }
        let snr = 10.0 * (snr_sq_sig / snr_sq_err.max(1e-10)).log10();
        eprintln!("SNR: {:.2} dB", snr);
    }
}
