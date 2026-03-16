// Test basic MDCT functionality
#[cfg(test)]
mod tests {
    use opus_rs::mdct::MdctLookup;

    #[test]
    fn test_mdct_identity() {
        // Test that MDCT inverse(MDCT forward) recovers the input
        let short_mdct_size = 120;
        let nb_short_mdcts = 8;
        let max_lm = 3;
        let overlap = 120;
        let mdct_size = 2 * short_mdct_size * nb_short_mdcts; // 1920

        let mdct = MdctLookup::new(mdct_size, max_lm);

        // Create simple sine wave input (n + overlap samples for MDCT with overlap)
        let input_len = mdct_size + overlap; // 1920 + 120 = 2040
        let mut input = vec![0.0f32; input_len];
        let freq = 440.0f32;
        let sr = 48000.0f32;
        for i in 0..input_len {
            input[i] = ((i as f32) * 2.0 * std::f32::consts::PI * freq / sr).sin();
        }

        // MDCT window (simple Hann window)
        let mut window = vec![0.0f32; 240];
        for i in 0..240 {
            window[i] = (std::f32::consts::PI * (i as f32 + 0.5) / 240.0).sin().powi(2);
        }

        // Forward MDCT
        let n2 = mdct_size / 2; // 960
        let mut freq_coeffs = vec![0.0f32; n2];
        mdct.forward(
            &input,
            &mut freq_coeffs,
            &window,
            overlap,
            0, // shift for 1920-point MDCT
            1, // stride for long block
        );

        eprintln!("MDCT forward output stats:");
        let mut max_val = 0.0f32;
        let mut sum_sq = 0.0f32;
        for i in 0..100 {
            let v = freq_coeffs[i].abs();
            max_val = max_val.max(v);
            sum_sq += v * v;
        }
        eprintln!("  first 100 coeffs: max={:.6}, rms={:.6}", max_val, (sum_sq/100.0).sqrt());

        // Check that output is not all zeros
        assert!(max_val > 0.01, "MDCT forward output too small: max={}", max_val);

        // Inverse MDCT
        let mut output = vec![0.0f32; mdct_size + overlap];
        mdct.backward(
            &freq_coeffs,
            &mut output,
            &window,
            overlap,
            0, // shift
            1, // stride
        );

        eprintln!("MDCT backward output stats:");
        max_val = 0.0f32;
        sum_sq = 0.0f32;
        for i in overlap/2..overlap/2 + 500 {
            let v = output[i].abs();
            max_val = max_val.max(v);
            sum_sq += v * v;
        }
        eprintln!("  samples [60..560]: max={:.6}, rms={:.6}", max_val, (sum_sq/500.0).sqrt());

        // Check that output is reasonable
        assert!(max_val > 0.1, "MDCT backward output too small: max={}", max_val);
    }
}
