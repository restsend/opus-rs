/// Interpolate two vectors.
/// Port of silk/interpolate.c
use crate::silk::define::MAX_LPC_ORDER;
use crate::silk::macros::*;

/// Interpolate two vectors.
///
/// # Arguments
/// * `x0` - first vector
/// * `x1` - second vector
/// * `ifact_q2` - interpolation factor (0..4), weight on second vector
/// * `d` - number of parameters to interpolate
///
/// Returns interpolated vector.
pub fn silk_interpolate(x0: &[i16], x1: &[i16], ifact_q2: i32, d: usize) -> [i16; MAX_LPC_ORDER] {
    debug_assert!(ifact_q2 >= 0 && ifact_q2 <= 4);
    let mut xi = [0i16; MAX_LPC_ORDER];
    for i in 0..d {
        // xi[i] = x0[i] + ((x1[i] - x0[i]) * ifact_Q2 + 2) >> 2
        xi[i] = (x0[i] as i32 + ((x1[i] as i32 - x0[i] as i32) * ifact_q2 + 2) / 4) as i16;
    }
    xi
}

/// In-place version: write result to output slice.
pub fn silk_interpolate_inplace(xi: &mut [i16], x0: &[i16], x1: &[i16], ifact_q2: i32, d: usize) {
    debug_assert!(ifact_q2 >= 0 && ifact_q2 <= 4);
    for i in 0..d {
        xi[i] = silk_add_rshift(
            x0[i] as i32,
            silk_smulbb(x1[i] as i32 - x0[i] as i32, ifact_q2),
            2,
        ) as i16;
    }
}

fn silk_add_rshift(a: i32, b: i32, shift: i32) -> i32 {
    a + (b >> shift)
}
