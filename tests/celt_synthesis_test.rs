use opus_rs::modes::default_mode;

#[test]
fn celt_synthesis_chain_bypass() {
    let mode = default_mode();
    let frame_size = 960;
    let overlap = mode.overlap; // 120
    let num_frames = 10;

    let freq = 440.0;
    let mut all_in = vec![0.0f32; frame_size * num_frames];
    let mut all_out = vec![0.0f32; frame_size * num_frames];

    for i in 0..(frame_size * num_frames) {
        let t = i as f32 / 48000.0;
        all_in[i] = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.4;
    }

    // Simulate encoder's pre-emphasis + MDCT forward
    let syn_mem_size = 2048 + overlap;
    let mut syn_mem = vec![0.0f32; syn_mem_size];
    let mut enc_preemph_mem = 0.0f32;
    let coef = mode.preemph[0]; // 0.85

    // Simulate decoder's iMDCT backward + de-emphasis
    let decode_buffer_size = 2048;
    let mut decode_mem = vec![0.0f32; decode_buffer_size + overlap];
    let mut dec_preemph_mem = 0.0f32;

    let mut lm = 0;
    while (mode.short_mdct_size << lm) != frame_size {
        lm += 1;
    }
    let shift = mode.max_lm - lm; // 3 - 3 = 0

    for f in 0..num_frames {
        let pcm_in = &all_in[f * frame_size..(f + 1) * frame_size];

        // === ENCODER SIDE: pre-emphasis + forward MDCT ===

        // Shift encoder history
        for i in 0..syn_mem_size - frame_size {
            syn_mem[i] = syn_mem[i + frame_size];
        }

        // Pre-emphasis
        let mut m = enc_preemph_mem;
        for i in 0..frame_size {
            let x = pcm_in[i];
            let val = x - m;
            syn_mem[syn_mem_size - frame_size + i] = val;
            m = x * coef;
        }
        enc_preemph_mem = m;

        // Forward MDCT
        let mdct_base = syn_mem_size - frame_size - overlap; // 1088
        let mut freq_coeffs = vec![0.0f32; frame_size];
        mode.mdct.forward(
            &syn_mem[mdct_base..],
            &mut freq_coeffs,
            mode.window,
            overlap,
            shift,
            1, // stride=1 for b=1
        );

        // === DECODER SIDE: backward MDCT + de-emphasis ===
        // (Skip PVQ/energy quantization - pass freq_coeffs directly)

        // Shift decoder memory
        for i in 0..decode_buffer_size - frame_size + overlap {
            decode_mem[i] = decode_mem[i + frame_size];
        }

        let out_syn_idx = decode_buffer_size - frame_size; // 1088
        mode.mdct.backward(
            &freq_coeffs,
            &mut decode_mem[out_syn_idx..],
            mode.window,
            overlap,
            shift,
            1, // stride=1
        );

        // Read output (before de-emphasis)
        let mut pcm_frame = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            pcm_frame[i] = decode_mem[out_syn_idx + i];
        }

        // De-emphasis
        let mut m = dec_preemph_mem;
        for i in 0..frame_size {
            let x = pcm_frame[i];
            let val = x + m;
            all_out[f * frame_size + i] = val;
            m = val * coef;
        }
        dec_preemph_mem = m;

        if f < 3 {
            eprintln!(
                "Frame {}: in[0..4]={:?} out[0..4]={:?}",
                f,
                &pcm_in[0..4],
                &all_out[f * frame_size..f * frame_size + 4]
            );
            eprintln!("  freq[0..4]={:?}", &freq_coeffs[0..4]);
        }
    }

    // Check SNR with various delays
    let start_idx = 2 * frame_size;
    let end_idx = 9 * frame_size;
    let mut best_snr: f32 = -100.0;
    let mut best_delay = 0;
    for delay in 0..(frame_size * 2) {
        let mut s_e = 0.0f64;
        let mut n_e = 0.0f64;
        let mut count = 0;
        for i in start_idx..end_idx {
            if i + delay >= all_out.len() {
                break;
            }
            let s = all_in[i] as f64;
            let d = all_out[i + delay] as f64;
            s_e += s * s;
            n_e += (s - d) * (s - d);
            count += 1;
        }
        if count < frame_size {
            continue;
        }
        let snr = 10.0 * (s_e / (n_e + 1e-12)).log10() as f32;
        if snr > best_snr {
            best_snr = snr;
            best_delay = delay;
        }
    }

    eprintln!(
        "Synthesis chain bypass: Best SNR = {:.2} dB at delay {}",
        best_snr, best_delay
    );

    // With no quantization, this should be near-perfect reconstruction
    // MDCT TDAC should give very high SNR (>80 dB)
    assert!(
        best_snr > 60.0,
        "Synthesis chain SNR too low: {:.2} dB — MDCT or emphasis bug",
        best_snr
    );
}

/// Test the full chain INCLUDING energy quantization (coarse+fine) but
/// skipping PVQ (just pass the normalized coefficients directly).
/// This isolates whether the energy quantization alone causes the issue.
#[test]
fn celt_energy_roundtrip_only() {
    use opus_rs::bands::{amp2log2, compute_band_energies, denormalise_bands, normalise_bands};
    use opus_rs::quant_bands::{quant_coarse_energy, quant_energy_finalise, quant_fine_energy};
    use opus_rs::range_coder::RangeCoder;
    use opus_rs::rate::clt_compute_allocation;

    let mode = default_mode();
    let channels = 1;
    let frame_size = 960;
    let overlap = mode.overlap;
    let nb_ebands = mode.nb_ebands;
    let num_frames = 10;
    let n_bytes = 160;

    let freq_hz = 440.0;
    let mut all_in = vec![0.0f32; frame_size * num_frames];
    let mut all_out = vec![0.0f32; frame_size * num_frames];

    for i in 0..(frame_size * num_frames) {
        let t = i as f32 / 48000.0;
        all_in[i] = (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.4;
    }

    let syn_mem_size = 2048 + overlap;
    let mut syn_mem = vec![0.0f32; syn_mem_size];
    let mut enc_preemph_mem = 0.0f32;
    let coef = mode.preemph[0];

    let decode_buffer_size = 2048;
    let mut decode_mem = vec![0.0f32; decode_buffer_size + overlap];
    let mut dec_preemph_mem = 0.0f32;

    let mut lm = 0;
    while (mode.short_mdct_size << lm) != frame_size {
        lm += 1;
    }
    let shift = mode.max_lm - lm;

    let mut enc_old_band_e = vec![-28.0f32; nb_ebands * channels];

    for f in 0..num_frames {
        let pcm_in = &all_in[f * frame_size..(f + 1) * frame_size];

        // === ENCODER: pre-emphasis + forward MDCT ===
        for i in 0..syn_mem_size - frame_size {
            syn_mem[i] = syn_mem[i + frame_size];
        }
        let mut m = enc_preemph_mem;
        for i in 0..frame_size {
            let x = pcm_in[i];
            syn_mem[syn_mem_size - frame_size + i] = x - m;
            m = x * coef;
        }
        enc_preemph_mem = m;

        let mdct_base = syn_mem_size - frame_size - overlap;
        let mut freq_coeffs = vec![0.0f32; frame_size];
        mode.mdct.forward(
            &syn_mem[mdct_base..],
            &mut freq_coeffs,
            mode.window,
            overlap,
            shift,
            1,
        );

        // Compute band energies
        let mut band_e = vec![0.0f32; nb_ebands * channels];
        compute_band_energies(mode, &freq_coeffs, &mut band_e, nb_ebands, channels, lm);

        // Normalize
        let mut x = vec![0.0f32; frame_size * channels];
        normalise_bands(
            mode,
            &freq_coeffs,
            &mut x,
            &band_e,
            nb_ebands,
            channels,
            1 << lm,
        );

        // Convert to log domain
        let mut band_log_e = vec![0.0f32; nb_ebands * channels];
        amp2log2(
            mode,
            nb_ebands,
            nb_ebands,
            &band_e,
            &mut band_log_e,
            channels,
        );

        // Encode coarse + fine energy
        let total_bits = (n_bytes * 8) as i32;
        let mut error = vec![0.0f32; nb_ebands * channels];
        let mut rc = RangeCoder::new_encoder(n_bytes as u32);

        // Need to encode same control bits as CELT: pf_on=false, transient=false
        rc.encode_bit_logp(false, 1); // pf_on
        rc.encode_bit_logp(false, 3); // transient

        quant_coarse_energy(
            mode,
            0,
            nb_ebands,
            &mut band_log_e,
            &mut enc_old_band_e,
            (total_bits << 3) as u32,
            &mut error,
            &mut rc,
            channels,
            lm,
            false,
        );

        // Compute allocation to get ebits
        let mut tf_res = vec![0i32; nb_ebands];
        let offsets = vec![0i32; nb_ebands];
        let mut cap = vec![0i32; nb_ebands];
        for i in 0..nb_ebands {
            cap[i] = (mode.cache.caps[nb_ebands * (2 * lm + channels - 1) + i] as i32 + 64)
                * channels as i32
                * 2;
        }
        let alloc_trim = 6; // default
        rc.encode_icdf(alloc_trim, &opus_rs::modes::TRIM_ICDF, 7);

        // TF
        for i in 0..nb_ebands {
            tf_res[i] = 0;
        }
        // Skip tf_encode for simplicity

        let mut intensity = 0i32;
        let mut dual_stereo = 0i32;
        let mut balance = 0;
        let mut pulses = vec![0i32; nb_ebands];
        let mut ebits = vec![0i32; nb_ebands];
        let mut fine_priority = vec![0i32; nb_ebands];

        clt_compute_allocation(
            mode,
            0,
            nb_ebands,
            &offsets,
            &cap,
            alloc_trim,
            &mut intensity,
            &mut dual_stereo,
            total_bits << 3,
            &mut balance,
            &mut pulses,
            &mut ebits,
            &mut fine_priority,
            channels as i32,
            lm as i32,
            &mut rc,
            true,
            0,
            nb_ebands as i32 - 1,
        );

        quant_fine_energy(
            mode,
            0,
            nb_ebands,
            &mut enc_old_band_e,
            &mut error,
            &ebits,
            &mut rc,
            channels,
        );

        // Skip PVQ encoding — just pass normalized coefficients directly

        quant_energy_finalise(
            mode,
            0,
            nb_ebands,
            &mut enc_old_band_e,
            &mut error,
            &ebits,
            &fine_priority,
            (total_bits - rc.tell() as i32) << 3,
            &mut rc,
            channels,
        );

        // === DECODER: denormalize with quantized energy, then iMDCT ===
        // Use enc_old_band_e (quantized energy) + x (exact normalized coefficients)

        let mut recon_freq = vec![0.0f32; frame_size * channels];
        denormalise_bands(
            mode,
            &x,
            &mut recon_freq,
            &enc_old_band_e,
            0,
            nb_ebands,
            channels,
            1 << lm,
        );

        // Shift decoder memory
        for i in 0..decode_buffer_size - frame_size + overlap {
            decode_mem[i] = decode_mem[i + frame_size];
        }
        let out_syn_idx = decode_buffer_size - frame_size;
        mode.mdct.backward(
            &recon_freq,
            &mut decode_mem[out_syn_idx..],
            mode.window,
            overlap,
            shift,
            1,
        );

        let mut pcm_frame = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            pcm_frame[i] = decode_mem[out_syn_idx + i];
        }

        let mut m = dec_preemph_mem;
        for i in 0..frame_size {
            let val = pcm_frame[i] + m;
            all_out[f * frame_size + i] = val;
            m = val * coef;
        }
        dec_preemph_mem = m;
    }

    // Check SNR
    let start_idx = 2 * frame_size;
    let end_idx = 9 * frame_size;
    let mut best_snr: f32 = -100.0;
    let mut best_delay = 0;
    for delay in 0..(frame_size * 2) {
        let mut s_e = 0.0f64;
        let mut n_e = 0.0f64;
        let mut count = 0;
        for i in start_idx..end_idx {
            if i + delay >= all_out.len() {
                break;
            }
            let s = all_in[i] as f64;
            let d = all_out[i + delay] as f64;
            s_e += s * s;
            n_e += (s - d) * (s - d);
            count += 1;
        }
        if count < frame_size {
            continue;
        }
        let snr = 10.0 * (s_e / (n_e + 1e-12)).log10() as f32;
        if snr > best_snr {
            best_snr = snr;
            best_delay = delay;
        }
    }

    eprintln!(
        "Energy roundtrip only: Best SNR = {:.2} dB at delay {}",
        best_snr, best_delay
    );

    // Energy quantization adds noise but should still be >10 dB
    assert!(
        best_snr > 10.0,
        "Energy-only roundtrip SNR too low: {:.2} dB",
        best_snr
    );
}
