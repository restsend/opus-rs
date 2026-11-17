//! silk_lpc_inverse_pred_gain against libopus's silk_LPC_inverse_pred_gain_c
//! (silk/LPC_inv_pred_gain.c) on random filters and on silk_NLSF2A outputs
//! of nearly coincident NLSFs, the filters on the edge of stability that
//! NLSF2A's stabilisation loop feeds it.
use opusref::*;
use std::os::raw::c_int;

unsafe extern "C" {
    fn silk_LPC_inverse_pred_gain_c(a: *const i16, order: c_int) -> i32;
    fn silk_NLSF2A(a: *mut i16, nlsf: *const i16, d: c_int, arch: c_int);
}

fn check(a: &[i16]) {
    let c = unsafe { silk_LPC_inverse_pred_gain_c(a.as_ptr(), a.len() as c_int) };
    let r = opus_rs::silk::lpc_analysis::silk_lpc_inverse_pred_gain(a, a.len());
    assert_eq!(r, c, "filter {a:?}");
}

#[test]
fn inverse_prediction_gain_matches_libopus() {
    let _ = unsafe { silk_lin2log(1) };
    let mut seed = 1u32;
    let mut rand = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed >> 8
    };
    for t in 0..200_000u32 {
        let order = if t % 2 == 0 { 16 } else { 10 };
        // Random filters of several sizes.
        let scale = 1 + (t % 7) as i32 * 2000;
        let a: Vec<i16> = (0..order)
            .map(|_| ((rand() as i32 % (2 * scale + 1)) - scale) as i16)
            .collect();
        check(&a);
        // NLSF2A of a sorted NLSF set with a cluster of close values.
        let mut nlsf: Vec<i16> = (0..order).map(|_| (rand() % 32000) as i16 + 300).collect();
        nlsf.sort();
        let at = (rand() as usize) % (order - 2);
        nlsf[at + 1] = nlsf[at] + (rand() % 4) as i16;
        nlsf[at + 2] = nlsf[at + 1] + (rand() % 4) as i16;
        let mut a = vec![0i16; order];
        unsafe { silk_NLSF2A(a.as_mut_ptr(), nlsf.as_ptr(), order as c_int, 0) };
        check(&a);
    }
}
