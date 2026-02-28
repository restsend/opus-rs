use opus_rs::mdct::MdctLookup;

#[test]
fn test_mdct_roundtrip() {
    let frame_size = 960;
    let overlap = 120;
    let n = frame_size * 2;
    let mdct = MdctLookup::new(n, 0);

    let mut window = vec![0.0f32; overlap];
    for i in 0..overlap {
        let x = (i as f32 + 0.5) / (overlap as f32);
        window[i] =
            (std::f32::consts::PI * 0.5 * (std::f32::consts::PI * 0.5 * x).sin().powi(2)).sin();
    }

    let num_frames = 10;
    let mut pcm_in = vec![0.0f32; frame_size * (num_frames + 2)];
    for i in 0..pcm_in.len() {
        pcm_in[i] = (i as f32 * 0.01).sin();
    }

    let mut history_enc = vec![0.0f32; overlap / 2];
    let mut history_dec = vec![0.0f32; overlap / 2];
    let mut pcm_out = vec![0.0f32; frame_size * num_frames];

    for f in 0..num_frames {
        let mut in_buf = vec![0.0f32; n];
        in_buf[..60].copy_from_slice(&history_enc);
        in_buf[60..60 + 960].copy_from_slice(&pcm_in[f * frame_size..(f + 1) * frame_size]);
        in_buf[1020..1080]
            .copy_from_slice(&pcm_in[(f + 1) * frame_size..(f + 1) * frame_size + 60]);

        let mut spectrum = vec![0.0f32; frame_size];
        mdct.forward(&in_buf, &mut spectrum, &window, overlap, 0, 1);

        history_enc.copy_from_slice(&pcm_in[(f + 1) * frame_size - 60..(f + 1) * frame_size]);

        let mut out_buf = vec![0.0f32; n];
        out_buf[..60].copy_from_slice(&history_dec);
        mdct.backward(&spectrum, &mut out_buf, &window, overlap, 0, 1);

        pcm_out[f * frame_size..(f + 1) * frame_size].copy_from_slice(&out_buf[..960]);

        history_dec.copy_from_slice(&out_buf[960..1020]);
    }

    let mut best_snr = -100.0;
    let mut best_offset = 0;

    for offset in -150..=150 {
        let mut signal_nrg = 0.0;
        let mut noise_nrg = 0.0;
        let start = (frame_size * 2) as i32;
        let end = (pcm_out.len() - 200) as i32;
        for i in start..end {
            let target = pcm_in[i as usize];
            let actual_idx = i + offset;
            if actual_idx >= 0 && actual_idx < pcm_out.len() as i32 {
                let actual = pcm_out[actual_idx as usize];
                let err = target - actual;
                signal_nrg += target * target;
                noise_nrg += err * err;
            }
        }
        let snr = 10.0 * (signal_nrg / noise_nrg.max(1e-20)).log10();
        if snr > best_snr {
            best_snr = snr;
            best_offset = offset;
        }
    }
    println!("Best Offset: {}, SNR: {:.2} dB", best_offset, best_snr);
}
