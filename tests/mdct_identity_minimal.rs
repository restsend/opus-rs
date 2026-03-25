use opus_rs::modes::default_mode;

#[test]
fn test_mdct_identity_minimal() {
    let mode = default_mode();

    // For shift=0, MDCT size is 1920, so input needs 1920 + overlap samples
    let mdct_size = 1920;
    let overlap = mode.overlap;
    let input_size = mdct_size + overlap;

    // Create a simple impulse-like signal
    let mut input = vec![0.0f32; input_size];
    input[mdct_size / 2] = 1.0; // Impulse at the center

    // Forward MDCT
    let mut freq = vec![0.0f32; mdct_size / 2];
    mode.mdct
        .forward(&input, &mut freq, mode.window, overlap, 0, 1);

    eprintln!("MDCT forward output (first 10): {:?}", &freq[..10]);
    eprintln!(
        "MDCT output max: {}",
        freq.iter().map(|f| f.abs()).fold(0.0f32, f32::max)
    );

    // Backward MDCT
    let mut output = vec![0.0f32; mdct_size + overlap];
    mode.mdct
        .backward(&freq, &mut output, mode.window, overlap, 0, 1);

    eprintln!("MDCT backward output (first 10): {:?}", &output[..10]);
    eprintln!(
        "MDCT backward output max: {}",
        output.iter().map(|f| f.abs()).fold(0.0f32, f32::max)
    );

    // Check overlap-add reconstruction
    // The overlap region should reconstruct the input
    let n2 = mdct_size / 2;
    eprintln!(
        "Checking overlap region: [{}..{}]",
        overlap / 2,
        overlap / 2 + 20
    );
    for i in overlap / 2..overlap / 2 + 20.min(n2 - overlap / 2) {
        let reconstructed = output[i];
        let original = input[i];
        eprintln!(
            "  i={}: original={:.6}, reconstructed={:.6}, diff={:.6}",
            i,
            original,
            reconstructed,
            (original - reconstructed).abs()
        );
        if (original - reconstructed).abs() > 0.01 {
            eprintln!("    ERROR: large difference!");
        }
    }
}
