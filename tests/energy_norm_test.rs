use opus_rs::bands::{compute_band_energies, denormalise_bands, normalise_bands};
use opus_rs::modes::default_mode;

#[test]
fn test_energy_norm_denorm() {
    let mode = default_mode();
    let channels = 1;
    let frame_size = 960;
    let lm = 3;
    let nb_ebands = mode.nb_ebands;

    // Create some frequency domain data
    let mut freq = vec![0.0f32; frame_size];
    for i in 0..frame_size {
        freq[i] = (i as f32 * 0.01).sin() * 0.1;
    }

    // Compute energies
    let mut band_e = vec![0.0f32; nb_ebands * channels];
    compute_band_energies(mode, &freq, &mut band_e, nb_ebands, channels, lm);

    let mut band_log_e = vec![0.0f32; nb_ebands * channels];
    opus_rs::bands::amp2log2(
        mode,
        nb_ebands,
        nb_ebands,
        &band_e,
        &mut band_log_e,
        channels,
    );

    println!("Original band_e[0..5]: {:?}", &band_e[0..5]);
    println!("Original freq[0..10]: {:?}", &freq[0..10]);

    // Normalize
    let mut x = vec![0.0f32; frame_size];
    normalise_bands(mode, &freq, &mut x, &band_e, nb_ebands, channels, 1 << lm);

    println!("Normalized x[0..10]: {:?}", &x[0..10]);

    // Denormalize
    let mut freq_restored = vec![0.0f32; frame_size];
    denormalise_bands(
        mode,
        &x,
        &mut freq_restored,
        &band_log_e,
        0,
        nb_ebands,
        channels,
        1 << lm,
    );

    println!("Restored freq[0..10]: {:?}", &freq_restored[0..10]);

    // Check error only within band coverage (samples covered by e_bands)
    let lm = 3;
    let band_end = (mode.e_bands[mode.nb_ebands] as usize) << lm;
    let mut max_error = 0.0f32;
    for i in 0..band_end {
        let err = (freq[i] - freq_restored[i]).abs();
        max_error = max_error.max(err);
    }

    println!(
        "Max error (within bands, 0..{}): {:.6e}",
        band_end, max_error
    );
    assert!(
        max_error < 1e-4,
        "Energy norm/denorm error too large: {}",
        max_error
    );
}
