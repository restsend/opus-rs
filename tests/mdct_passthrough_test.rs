#[test]
fn test_celt_mdct_passthrough() {
    use opus_rs::modes::default_mode;

    let mode = default_mode();
    let frame_size = 960;
    let overlap = mode.overlap;
    // For long blocks: shift = mode.max_lm - lm. At full frame_size, lm = max_lm, so shift = 0.
    let shift = 0usize;
    let b = 1usize;
    let syn_mem_size = 2048 + overlap;
    let mdct_base = syn_mem_size - frame_size - overlap;

    // Create a continuous sine wave input (3 frames worth)
    let freq_hz = 440.0 / 48000.0 * 2.0 * std::f32::consts::PI;
    let total_samples = 3 * frame_size;
    let mut all_in = vec![0.0f32; total_samples];
    for i in 0..total_samples {
        all_in[i] = (freq_hz * i as f32).sin();
    }

    // Simulate encoder syn_mem + decoder decode_mem
    let mut syn_mem = vec![0.0f32; syn_mem_size];
    let decode_buffer_size = 2048;
    let mut decode_mem = vec![0.0f32; decode_buffer_size + overlap];

    let mut all_out = Vec::new();

    // Encode and decode 3 frames
    for frame_idx in 0..3 {
        let frame_start = frame_idx * frame_size;

        // Shift encoder history left by frame_size, then insert new samples at end
        for i in 0..syn_mem_size - frame_size {
            syn_mem[i] = syn_mem[i + frame_size];
        }
        for i in 0..frame_size {
            syn_mem[syn_mem_size - frame_size + i] = all_in[frame_start + i];
        }

        // MDCT forward
        let mut freq_buf = vec![0.0f32; frame_size];
        mode.mdct.forward(
            &syn_mem[mdct_base..],
            &mut freq_buf,
            mode.window,
            overlap,
            shift,
            b,
        );

        // Shift decoder history left by frame_size
        for i in 0..decode_buffer_size - frame_size + overlap {
            decode_mem[i] = decode_mem[i + frame_size];
        }
        for i in decode_buffer_size - frame_size + overlap..decode_mem.len() {
            decode_mem[i] = 0.0;
        }

        // MDCT backward (overlap-add with previous frame's tail)
        let out_syn_idx = decode_buffer_size - frame_size;
        mode.mdct.backward(
            &freq_buf,
            &mut decode_mem[out_syn_idx..],
            mode.window,
            overlap,
            shift,
            b,
        );

        // Extract frame output
        let mut frame_out = vec![0.0f32; frame_size];
        frame_out.copy_from_slice(&decode_mem[out_syn_idx..out_syn_idx + frame_size]);
        all_out.extend_from_slice(&frame_out);
    }

    // Compare frame 2 output against input, searching for the best delay alignment.
    // Frame 0 is startup (OLA with zeros), so we compare frame 2 which should be clean.
    let compare_frame = 2;
    let mut best_snr = -100.0f64;
    let mut best_delay = 0usize;

    for delay in 0..2 * frame_size {
        let mut signal_power = 0.0f64;
        let mut error_power = 0.0f64;
        let mut count = 0;
        for i in 0..frame_size {
            let out_idx = compare_frame * frame_size + i;
            if delay > compare_frame * frame_size + i {
                continue;
            }
            let in_idx = compare_frame * frame_size + i - delay;
            if in_idx < total_samples && out_idx < all_out.len() {
                let sig = all_in[in_idx] as f64;
                let out = all_out[out_idx] as f64;
                signal_power += sig * sig;
                error_power += (out - sig) * (out - sig);
                count += 1;
            }
        }
        if count > frame_size / 2 && signal_power > 1e-10 {
            let snr = 10.0 * (signal_power / error_power.max(1e-20)).log10();
            if snr > best_snr {
                best_snr = snr;
                best_delay = delay;
            }
        }
    }

    eprintln!(
        "CELT MDCT passthrough: best SNR = {:.2} dB at delay = {}",
        best_snr, best_delay
    );

    // After 2 frames of warmup, TDAC should give near-perfect reconstruction
    assert!(
        best_snr > 60.0,
        "MDCT passthrough SNR should be >60 dB, got {:.2} dB at delay {}",
        best_snr,
        best_delay
    );
}
