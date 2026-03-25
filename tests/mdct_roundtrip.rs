use opus_rs::modes::default_mode;

#[test]
fn test_mdct_simple_roundtrip() {
    let mode = default_mode();
    let overlap = mode.overlap; // 120
    let max_lm = mode.max_lm; // 4

    // Test with shift=1 (frame_size = 960)
    let shift = 1;
    let n = mode.mdct.n >> shift; // 960
    let n2 = n / 2; // 480
    let frame_size = n;

    // Create a simple test signal - need n + overlap input samples
    let input_size = n + overlap;
    let mut input = vec![0.0f32; input_size];
    for i in 0..input_size {
        input[i] = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin() * 0.4;
    }

    // Forward MDCT - output is n2 elements
    let mut freq = vec![0.0f32; n2];
    mode.mdct.forward(
        &input,
        &mut freq,
        mode.window,
        overlap,
        shift,
        1, // stride
    );

    // Backward MDCT - output is n + overlap elements (with overlap region at start for TDAC)
    let mut output = vec![0.0f32; n + overlap];
    mode.mdct.backward(
        &freq,
        &mut output,
        mode.window,
        overlap,
        shift,
        1, // stride
    );

    // Check the overlap region (after TDAC)
    let overlap2 = overlap / 2;

    // Print some values
    eprintln!(
        "Input (overlap region): {:?}",
        &input[overlap2..overlap2 + 4]
    );
    eprintln!(
        "Output (overlap region): {:?}",
        &output[overlap2..overlap2 + 4]
    );
    eprintln!("Freq coefficients: {:?}", &freq[0..4]);

    // Calculate SNR in the non-overlap region (after TDAC)
    // The output[0..overlap] contains the TDAC result
    // The output[overlap..n] should match input[overlap..n]
    let mut signal_power = 0.0f64;
    let mut noise_power = 0.0f64;
    for i in overlap..n {
        let s = input[i] as f64;
        let d = output[i] as f64;
        signal_power += s * s;
        noise_power += (s - d) * (s - d);
    }
    let snr = 10.0 * (signal_power / (noise_power + 1e-12)).log10();
    eprintln!("SNR in non-overlap region: {:.2} dB", snr);

    // Check scaling
    let in_max = input.iter().cloned().fold(0.0f32, f32::max).abs();
    let out_max = output.iter().cloned().fold(0.0f32, f32::max).abs();
    eprintln!(
        "Input max: {:.6}, Output max: {:.6}, Ratio: {:.2}",
        in_max,
        out_max,
        out_max / in_max
    );

    // The roundtrip should preserve signal reasonably well
    // Due to windowing, we expect some loss but not 100x
    assert!(snr > 0.0, "MDCT roundtrip SNR too low: {:.2} dB", snr);
}
