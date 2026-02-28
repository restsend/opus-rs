use opus_rs::mdct::MdctLookup;

#[test]
fn test_mdct_recon_simple() {
    let frame_size = 480;
    let n = 2 * frame_size;
    let overlap = 120;
    let lookup = MdctLookup::new(n, 0);

    let mut window = vec![0.0f32; overlap];
    for i in 0..overlap {
        let x = std::f32::consts::PI * (i as f32 + 0.5) / (2.0 * overlap as f32);
        window[i] = x.sin();
    }

    let total_in = frame_size + overlap;
    let mut in_data = vec![0.0f32; total_in];
    for i in 0..total_in {
        in_data[i] = (i as f32 * 0.1).sin();
    }

    let mut freq = vec![0.0f32; frame_size];
    lookup.forward(&in_data, &mut freq, &window, overlap, 0, 1);

    let mut out = vec![0.0f32; frame_size + overlap];
    lookup.backward(&freq, &mut out, &window, overlap, 0, 1);

    // Check signs/order manually
    let mut best_snr = -100.0;
    let mut best_delay = 0;

    for d in 0..100 {
        let mut local_sum_in2 = 0.0;
        let mut local_sum_err2 = 0.0;
        for i in 0..(frame_size - d) {
            let val_in = in_data[overlap / 2 + i];
            let val_out = out[i + d];
            local_sum_in2 += val_in * val_in;
            local_sum_err2 += (val_in - val_out) * (val_in - val_out);
        }
        let snr = 10.0 * (local_sum_in2 / (local_sum_err2 + 1e-10)).log10();
        if d == 0 {
            println!(
                "Delay 0: In Energy {:.2e}, Out Energy (matched) {:.2e}, ratio {:.2}",
                local_sum_in2,
                local_sum_in2 + local_sum_err2 - local_sum_err2,
                freq[0]
            );
            // That print is confusing. Let's just print sums.
            println!(
                "Delay 0: In2={:.2e}, Err2={:.2e}",
                local_sum_in2, local_sum_err2
            );
        }
        if snr > best_snr {
            best_snr = snr;
            best_delay = d;
        }
    }

    println!("Best SNR: {:.2} dB at delay {}", best_snr, best_delay);

    // Compute Middle Region SNR
    let mid_start = overlap;
    let mid_end = frame_size - overlap;
    let mut mid_sum_in2 = 0.0;
    let mut mid_sum_err2 = 0.0;
    for i in mid_start..mid_end {
        let val_in = in_data[overlap / 2 + i];
        let val_out = out[i + best_delay];
        mid_sum_in2 += val_in * val_in;
        mid_sum_err2 += (val_in - val_out) * (val_in - val_out);
    }
    let mid_snr = 10.0 * (mid_sum_in2 / (mid_sum_err2 + 1e-10)).log10();
    println!("Middle Region SNR: {:.2} dB", mid_snr);

    for i in 0..50 {
        let idx = i;
        if idx + best_delay < out.len() {
            println!(
                "i={}: in={:.4}, out={:.4}, ratio={:.4}",
                idx,
                in_data[overlap / 2 + idx],
                out[idx + best_delay],
                out[idx + best_delay] / (in_data[overlap / 2 + idx] + 1e-10)
            );
        }
    }
}
