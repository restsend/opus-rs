use opus_rs::modes::default_mode;

#[test]
fn test_preemphasis_roundtrip() {
    let mode = default_mode();
    let coef = mode.preemph[0];

    // Test data
    let frame_size = 120;
    let mut input = vec![0.0f32; frame_size];
    for i in 0..frame_size {
        input[i] = (i as f32 * 0.1).sin();
    }

    // Pre-emphasis
    let mut preemphasized = vec![0.0f32; frame_size];
    let mut mem_pre = 0.0f32;
    for i in 0..frame_size {
        let x = input[i];
        preemphasized[i] = x - mem_pre;
        mem_pre = x * coef;
    }

    // De-preemphasis (IIR filter)
    let mut deemphasized = vec![0.0f32; frame_size];
    let mut mem_de = 0.0f32;
    for i in 0..frame_size {
        let x = preemphasized[i];
        let val = x + mem_de;
        deemphasized[i] = val;
        mem_de = val * coef;
    }

    // Check roundtrip
    let mut max_error = 0.0f32;
    for i in 0..frame_size {
        let err = (input[i] - deemphasized[i]).abs();
        max_error = max_error.max(err);
    }

    println!("Pre-emphasis roundtrip max error: {:.6e}", max_error);
    println!("Input[0..5]: {:?}", &input[0..5]);
    println!("Preemph[0..5]: {:?}", &preemphasized[0..5]);
    println!("Deemph[0..5]: {:?}", &deemphasized[0..5]);

    assert!(
        max_error < 1e-5,
        "Pre-emphasis roundtrip error too large: {}",
        max_error
    );
}
