use opus_rs::bands::compute_band_energies;
use opus_rs::bands::normalise_bands;
use opus_rs::modes::default_mode;

#[test]
fn test_bands_match() {
    let mode = default_mode();
    let nb_ebands = mode.nb_ebands;
    let lm: usize = 0;
    // mode.shortMdctSize ? In Rust mode struct checks.
    // In modes.rs: pub short_mdct_size: usize
    let n = mode.short_mdct_size << lm;

    // Input generation
    let mut x = vec![0.0f32; n];
    for i in 0..n {
        x[i] = (i as f32 * 0.1).sin() * (1.0 + 0.5 * (i as f32 * 0.01).cos());
    }

    let mut band_e = vec![0.0f32; nb_ebands];
    // compute_band_energies(m, X, bandE, end, C, LM, arch=0)
    // Rust signature: compute_band_energies(m: &CeltMode, X: &[f32], bandE: &mut [f32], end: usize, C: usize, LM: i32, arch: i32)
    compute_band_energies(mode, &x, &mut band_e, nb_ebands, 1, lm);

    // Reference values from verify_bands output
    let ref_energies = [
        0.000000, 0.149748, 0.297984, 0.443214, 0.583972, 0.718839, 0.846456, 0.965538, 1.591308,
        1.835139, 2.005610, 2.095897, 2.919261, 2.482801, 1.667935, 0.692719, 1.801952, 3.756825,
        3.890290, 3.614069, 4.083487,
    ];

    // Check first 21 bands (some modes have more or less, verify_bands output printed 21)
    for i in 0..ref_energies.len() {
        let diff = (band_e[i] - ref_energies[i]).abs();
        assert!(
            diff < 1e-5,
            "Band {} energy mismatch: expected {}, got {}",
            i,
            ref_energies[i],
            band_e[i]
        );
    }

    // Normalization
    let mut x_norm = vec![0.0f32; n];
    // normalise_bands(m, X, X_norm, bandE, end, C, M)
    // Rust signature: normalise_bands(m: &CeltMode, freq: &[f32], norm: &mut [f32], band_e: &[f32], end: usize, C: usize, M: i32)
    // M = 1 << LM = 1.
    normalise_bands(mode, &x, &mut x_norm, &band_e, nb_ebands, 1, 1 << lm);

    let ref_norm_head = [
        0.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.675474,
        0.737384,
    ];

    for i in 0..10 {
        let diff = (x_norm[i] - ref_norm_head[i]).abs();
        assert!(
            diff < 1e-5,
            "Norm index {} mismatch: expected {}, got {}",
            i,
            ref_norm_head[i],
            x_norm[i]
        );
    }
}
