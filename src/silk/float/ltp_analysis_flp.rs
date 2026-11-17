//! libopus `silk/float/find_LTP_FLP.c`, `corrMatrix_FLP.c` and
//! `LTP_analysis_filter_FLP.c`: the LTP correlations of a voiced frame and
//! its LTP residual.

use crate::silk::define::*;
use crate::silk::float::sigproc_flp::*;
use crate::silk::tuning_parameters::LTP_CORR_INV_MAX;

/// libopus `silk_corrVector_FLP`: `xt[lag]` is the inner product of the
/// target `t[..l]` with `x` delayed by `lag`, where `x` holds `l + order - 1`
/// samples, the latest column first.
pub fn silk_corr_vector_flp(x: &[f32], t: &[f32], l: usize, order: usize, xt: &mut [f32], arch: crate::Arch) {
    for (lag, v) in xt[..order].iter_mut().enumerate() {
        let start = order - 1 - lag;
        *v = silk_inner_product_flp(&x[start..start + l], t, arch) as f32;
    }
}

/// libopus `silk_corrMatrix_FLP`: the `order` x `order` correlation matrix
/// `xx` of the delayed copies of `x` (`l + order - 1` samples).
pub fn silk_corr_matrix_flp(x: &[f32], l: usize, order: usize, xx: &mut [f32], arch: crate::Arch) {
    let p1 = order - 1; // First sample of column 0.
    let mut energy = silk_energy_flp(&x[p1..p1 + l]);
    xx[0] = energy as f32;
    for j in 1..order {
        // The products in f32, their difference added in f64.
        energy += f64::from(x[p1 - j] * x[p1 - j] - x[p1 + l - j] * x[p1 + l - j]);
        xx[j * order + j] = energy as f32;
    }
    for lag in 1..order {
        let p2 = order - 1 - lag; // First sample of column lag.
        let mut energy = silk_inner_product_flp(&x[p1..p1 + l], &x[p2..p2 + l], arch);
        xx[lag * order] = energy as f32;
        xx[lag] = energy as f32;
        for j in 1..order - lag {
            energy += f64::from(x[p1 - j] * x[p2 - j] - x[p1 + l - j] * x[p2 + l - j]);
            xx[(lag + j) * order + j] = energy as f32;
            xx[j * order + lag + j] = energy as f32;
        }
    }
}

/// libopus `silk_find_LTP_FLP`: the normalized LTP correlation matrices
/// `xx` and vectors `x_x` of the subframes of the residual `r`, where
/// `r[r_start..]` is the frame and `lag` the pitch lags.
pub fn silk_find_ltp_flp(
    xx: &mut [f32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER],
    x_x: &mut [f32; MAX_NB_SUBFR * LTP_ORDER],
    r: &[f32],
    r_start: usize,
    lag: &[i32; MAX_NB_SUBFR],
    subfr_length: usize,
    nb_subfr: usize,
    arch: crate::Arch,
) {
    let mut r_ptr = r_start;
    for k in 0..nb_subfr {
        let lag_ptr = r_ptr - (lag[k] as usize + LTP_ORDER / 2);
        let xx_k = &mut xx[k * LTP_ORDER * LTP_ORDER..(k + 1) * LTP_ORDER * LTP_ORDER];
        let x_x_k = &mut x_x[k * LTP_ORDER..(k + 1) * LTP_ORDER];
        silk_corr_matrix_flp(&r[lag_ptr..], subfr_length, LTP_ORDER, xx_k, arch);
        silk_corr_vector_flp(&r[lag_ptr..], &r[r_ptr..r_ptr + subfr_length], subfr_length, LTP_ORDER, x_x_k, arch);
        let e = silk_energy_flp(&r[r_ptr..r_ptr + subfr_length + LTP_ORDER]) as f32;
        let temp = 1.0 / e.max(LTP_CORR_INV_MAX * 0.5 * (xx_k[0] + xx_k[24]) + 1.0);
        silk_scale_vector_flp(xx_k, temp);
        silk_scale_vector_flp(x_x_k, temp);
        r_ptr += subfr_length;
    }
}

/// libopus `silk_LTP_analysis_filter_FLP`: the LTP residual of each
/// subframe of `x` (from `x[x_start..]`, with `pre_length` preceding
/// samples each), scaled by the inverse gains.
#[allow(clippy::too_many_arguments)]
pub fn silk_ltp_analysis_filter_flp(
    ltp_res: &mut [f32],
    x: &[f32],
    x_start: usize,
    b: &[f32; LTP_ORDER * MAX_NB_SUBFR],
    pitch_l: &[i32; MAX_NB_SUBFR],
    inv_gains: &[f32; MAX_NB_SUBFR],
    subfr_length: usize,
    nb_subfr: usize,
    pre_length: usize,
) {
    let mut x_ptr = x_start;
    let mut res_ptr = 0;
    for k in 0..nb_subfr {
        let mut x_lag = x_ptr - pitch_l[k] as usize;
        let inv_gain = inv_gains[k];
        let btmp = &b[k * LTP_ORDER..(k + 1) * LTP_ORDER];
        for i in 0..subfr_length + pre_length {
            let mut v = x[x_ptr + i];
            for (j, &bj) in btmp.iter().enumerate() {
                v -= bj * x[x_lag + LTP_ORDER / 2 - j];
            }
            ltp_res[res_ptr + i] = v * inv_gain;
            x_lag += 1;
        }
        res_ptr += subfr_length + pre_length;
        x_ptr += subfr_length;
    }
}
