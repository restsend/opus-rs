//! libopus `_celt_lpc` clears all `p` coefficients before the recursion
//! and runs it only when `ac[0] > 1e-10` (celt_lpc.c:52-91).
use opus_rs::celt_lpc::lpc;

/// After the 30 dB early stop, the remaining coefficients are zero,
/// whatever the buffer held.
#[test]
fn coefficients_after_the_early_stop_are_zero() {
    let mut ac = [0.0f32; 5];
    ac[0] = 1.0;
    ac[1] = -0.9995;
    let mut l = [7.0f32; 4];
    lpc(&mut l, &ac, 4);
    assert!(l[0] > 0.99, "{l:?}");
    assert_eq!(l[1..], [0.0; 3], "{l:?}");
}

/// A NaN `ac[0]` fails the `ac[0] > 1e-10` test: every coefficient is zero.
#[test]
fn nan_energy_gives_zero_coefficients() {
    let ac = [f32::NAN, 0.5, 0.25, 0.125, 0.0625];
    let mut l = [7.0f32; 4];
    lpc(&mut l, &ac, 4);
    assert_eq!(l, [0.0; 4]);
}
