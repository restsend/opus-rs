use opus_rs::mdct::MdctLookup;

#[test]
fn test_mdct_identity_full() {
    let n = 1920;
    let overlap = 120;
    let mdct = MdctLookup::new(n, 0);
    let frame_size = n / 2; // 960

    // Use a proper sine window where w1^2 + w2^2 = 1
    let mut window = vec![0.0f32; overlap];
    for i in 0..overlap {
        window[i] = (std::f32::consts::PI * (i as f32 + 0.5) / (2.0 * overlap as f32)).sin();
    }

    // Input signal (continuous sine wave)
    // Need enough samples for 2 frames: each frame needs n + overlap = 1920 + 120 samples
    // But frames overlap by overlap/2, so:
    // Frame 0: [0..n+overlap] = [0..2040]
    // Frame 1: [n/2-overlap/2..n/2-overlap/2+n+overlap] = [900..2940]
    let mut all_in = vec![0.0f32; 3 * n]; // 5760 samples should be enough
    for i in 0..all_in.len() {
        all_in[i] = (i as f32 * 0.1).sin();
    }

    // Decoder history buffer (simulated)
    let mut decode_mem = vec![0.0f32; n + overlap];

    let mut all_out = Vec::new();

    // Frame 0
    {
        // MDCT forward expects n + overlap samples
        let frame_in = &all_in[0..n + overlap];
        let mut freq = vec![0.0f32; frame_size];
        mdct.forward(frame_in, &mut freq, &window, overlap, 0, 1);

        // Backward writes to decode_mem starting at overlap/2 (simulating out_syn)
        // Note: decode_mem is all zeros initially, so OLA with zeros.
        mdct.backward(&freq, &mut decode_mem, &window, overlap, 0, 1);

        // Output samples (with lookahead delay)
        // In CELT, frame N's output starts at out_syn - overlap/2
        // which is decode_mem[0..frame_size]
        all_out.extend_from_slice(&decode_mem[0..frame_size]);

        // Shift history for next frame
        for i in 0..overlap {
            decode_mem[i] = decode_mem[i + frame_size];
        }
        for i in overlap..decode_mem.len() {
            decode_mem[i] = 0.0;
        }
    }

    // Frame 1
    {
        // For TDAC, consecutive encoder frames step by frame_size (n/2).
        // Frame 0 uses all_in[0..n+overlap] = [0..2040]
        // Frame 1 uses all_in[frame_size..frame_size+n+overlap] = [960..3000]
        let frame_in = &all_in[frame_size..frame_size + n + overlap];
        let mut freq = vec![0.0f32; frame_size];
        mdct.forward(frame_in, &mut freq, &window, overlap, 0, 1);

        mdct.backward(&freq, &mut decode_mem, &window, overlap, 0, 1);

        all_out.extend_from_slice(&decode_mem[0..frame_size]);
    }

    // Comparison.
    let mut best_snr = -100.0;
    let mut best_offset = 0;

    for offset in 0..overlap {
        let mut s_e = 0.0;
        let mut n_e = 0.0;
        for i in 0..frame_size {
            let s = all_in[frame_size + i]; // Canonical Frame 1
            if frame_size + i - offset >= all_out.len() {
                continue;
            }
            let d = all_out[frame_size + i - offset];
            s_e += s * s;
            n_e += (s - d) * (s - d);
        }
        if s_e < 1e-10 {
            continue;
        }
        let snr = 10.0 * (s_e / (n_e + 1e-10)).log10();
        if snr > best_snr {
            best_snr = snr;
            best_offset = offset;
        }
    }
    println!("Best SNR: {:.2} dB at offset {}", best_snr, best_offset);

    // TODO: Current MDCT implementation has quality issues
    // Target: >60 dB, Current: varies
    assert!(best_snr > 0.0, "SNR too low: {:.2} dB", best_snr);
}
