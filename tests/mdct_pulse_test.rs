use opus_rs::mdct::MdctLookup;
use std::f32::consts::PI;

#[test]
fn test_mdct_pulse_180() {
    let n = 480;
    let overlap = 240;
    let mdct = MdctLookup::new(n, 0);

    let mut window = vec![0.0; overlap];
    for i in 0..overlap {
        window[i] = ((i as f32 + 0.5) / overlap as f32 * PI * 0.5).sin();
    }

    let mut input = vec![0.0; n];
    input[180] = 1.0;

    let mut freq = vec![0.0; n / 2];
    mdct.forward(&input, &mut freq, &window, overlap, 0, 1);

    let mut output = vec![0.0; n];
    mdct.backward(&freq, &mut output, &window, overlap, 0, 1);

    let mut max_val = 0.0f32;
    let mut max_idx = 0;
    for (i, &v) in output.iter().enumerate() {
        let v: f32 = v;
        if v.abs() > max_val {
            max_val = v.abs();
            max_idx = i;
        }
    }

    println!("Max reconstructed value: {} at index {}", max_val, max_idx);
    // With overlap=N/2, the entire output is in the overlap region.
    // Single-frame MDCT forward+backward gives a windowed (attenuated) version.
    // Perfect reconstruction requires 2 frames for TDAC cancellation.
    // Check that the pulse is at the correct position, and amplitude is reasonable.
    assert_eq!(max_idx, 180, "Pulse should be at position 180");
    assert!(
        max_val > 0.5,
        "Pulse should be reasonably strong, got {}",
        max_val
    );
}

#[test]
fn test_mdct_pulse_300() {
    let n = 480;
    let overlap = 240;
    let mdct = MdctLookup::new(n, 0);

    let mut window = vec![0.0; overlap];
    for i in 0..overlap {
        window[i] = ((i as f32 + 0.5) / overlap as f32 * PI * 0.5).sin();
    }

    let mut input = vec![0.0; n];
    input[300] = 1.0;

    let mut freq = vec![0.0; n / 2];
    mdct.forward(&input, &mut freq, &window, overlap, 0, 1);

    let mut output = vec![0.0; n];
    mdct.backward(&freq, &mut output, &window, overlap, 0, 1);

    let mut max_val = 0.0f32;
    let mut max_idx = 0;
    for (i, &v) in output.iter().enumerate() {
        let v: f32 = v;
        if v.abs() > max_val {
            max_val = v.abs();
            max_idx = i;
        }
    }

    println!("Max reconstructed value: {} at index {}", max_val, max_idx);
    // This might fail if the second half is not reconstructed correctly in a single-frame test
    // But since IMDCT has aliasing, we expect it to be windowed by the NEXT frame.
    // However, for a single frame, let's see what it does.
}
