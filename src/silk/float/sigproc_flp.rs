//! The floating-point signal processing functions of libopus's float SILK
//! encoder (`silk/float/SigProc_FLP.h` and the files it declares), in the
//! operation order of the C code: products of two `silk_float`s are rounded
//! to `f32`, and the `double` accumulations stay `f64`.

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::silk::define::MAX_PREDICTION_POWER_GAIN;
use crate::silk::tuning_parameters::FIND_LPC_COND_FAC;

/// `SILK_MAX_ORDER_LPC` (SigProc_FIX.h).
pub const SILK_MAX_ORDER_LPC: usize = 24;

/// `PI` of SigProc_FLP.h, `3.1415926536f`: the same f32.
const PI: f32 = core::f32::consts::PI;

/// libopus `silk_sigmoid`.
#[inline]
pub fn silk_sigmoid(x: f32) -> f32 {
    (1.0 / (1.0 + (-f64::from(x)).exp())) as f32
}

/// libopus `silk_float2int`: round to the nearest integer, ties to even
/// (`lrintf`).
#[inline]
pub fn silk_float2int(x: f32) -> i32 {
    // Below 2^51 in magnitude, x + 1.5 * 2^52 lands where the f64 spacing
    // is 1, so the sum rounds x to the nearest integer, ties to even.
    const MAGIC: f64 = 6_755_399_441_055_744.0;
    ((f64::from(x) + MAGIC) - MAGIC) as i32
}

/// libopus `silk_float2short_array`: rounded and saturated to 16 bits.
pub fn silk_float2short_array(out: &mut [i16], input: &[f32]) {
    for (o, &x) in out.iter_mut().zip(input) {
        *o = silk_float2int(x).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

/// libopus `silk_short2float_array`.
pub fn silk_short2float_array(out: &mut [f32], input: &[i16]) {
    for (o, &x) in out.iter_mut().zip(input) {
        *o = f32::from(x);
    }
}

/// libopus `silk_log2` (SigProc_FLP.h): `3.32192809488736 * log10(x)`,
/// with libopus's constant.
#[inline]
#[allow(clippy::approx_constant)]
pub fn silk_log2(x: f64) -> f32 {
    (3.321_928_094_887_36 * x.log10()) as f32
}

/// libopus `silk_energy_FLP`: the sum of squares in `f64`, four at a time.
pub fn silk_energy_flp(data: &[f32]) -> f64 {
    let mut result = 0.0f64;
    let (quads, rest) = data.as_chunks::<4>();
    for d in quads {
        result += f64::from(d[0]) * f64::from(d[0])
            + f64::from(d[1]) * f64::from(d[1])
            + f64::from(d[2]) * f64::from(d[2])
            + f64::from(d[3]) * f64::from(d[3]);
    }
    for &d in rest {
        result += f64::from(d) * f64::from(d);
    }
    result
}

/// libopus `silk_inner_product_FLP_c`: the inner product of `data1` and
/// `data2[..data1.len()]` in `f64`, four at a time.
pub fn silk_inner_product_flp(data1: &[f32], data2: &[f32]) -> f64 {
    let data2 = &data2[..data1.len()];
    let mut result = 0.0f64;
    let (a, a_rest) = data1.as_chunks::<4>();
    let (b, b_rest) = data2.as_chunks::<4>();
    for (x, y) in a.iter().zip(b) {
        result += f64::from(x[0]) * f64::from(y[0])
            + f64::from(x[1]) * f64::from(y[1])
            + f64::from(x[2]) * f64::from(y[2])
            + f64::from(x[3]) * f64::from(y[3]);
    }
    for (&x, &y) in a_rest.iter().zip(b_rest) {
        result += f64::from(x) * f64::from(y);
    }
    result
}

/// libopus `silk_autocorrelation_FLP`: `results[i]` for the lags
/// `0..results.len()` (at most `input.len()`).
pub fn silk_autocorrelation_flp(results: &mut [f32], input: &[f32]) {
    let n = input.len();
    for (i, r) in results.iter_mut().take(n).enumerate() {
        *r = silk_inner_product_flp(&input[..n - i], &input[i..]) as f32;
    }
}

/// libopus `silk_bwexpander_FLP`: chirp of the AR filter `ar`.
pub fn silk_bwexpander_flp(ar: &mut [f32], chirp: f32) {
    let d = ar.len();
    let mut cfac = chirp;
    for a in &mut ar[..d - 1] {
        *a *= cfac;
        cfac *= chirp;
    }
    ar[d - 1] *= cfac;
}

/// libopus `silk_k2a_FLP`: prediction coefficients `a` from the reflection
/// coefficients `rc`.
pub fn silk_k2a_flp(a: &mut [f32], rc: &[f32]) {
    for (k, &rck) in rc.iter().enumerate() {
        for n in 0..(k + 1) >> 1 {
            let tmp1 = a[n];
            let tmp2 = a[k - n - 1];
            a[n] = tmp1 + tmp2 * rck;
            a[k - n - 1] = tmp2 + tmp1 * rck;
        }
        a[k] = -rck;
    }
}

/// libopus `silk_schur_FLP`: reflection coefficients from the
/// autocorrelation `auto_corr[..=order]`; returns the residual energy.
pub fn silk_schur_flp(refl_coef: &mut [f32], auto_corr: &[f32], order: usize) -> f32 {
    let mut c = [[0.0f64; 2]; SILK_MAX_ORDER_LPC + 1];
    for k in 0..=order {
        c[k] = [f64::from(auto_corr[k]); 2];
    }
    for k in 0..order {
        let rc_tmp = -c[k + 1][0] / c[0][1].max(f64::from(1e-9f32));
        refl_coef[k] = rc_tmp as f32;
        for n in 0..order - k {
            let ctmp1 = c[n + k + 1][0];
            let ctmp2 = c[n][1];
            c[n + k + 1][0] = ctmp1 + ctmp2 * rc_tmp;
            c[n][1] = ctmp2 + ctmp1 * rc_tmp;
        }
    }
    c[0][1] as f32
}

/// libopus `silk_LPC_inverse_pred_gain_FLP`: the inverse prediction gain of
/// `a`, or 0 if the filter is unstable or its gain exceeds
/// `MAX_PREDICTION_POWER_GAIN`.
pub fn silk_lpc_inverse_pred_gain_flp(a: &[f32]) -> f32 {
    let order = a.len();
    let mut atmp = [0.0f32; SILK_MAX_ORDER_LPC];
    atmp[..order].copy_from_slice(a);
    let max_gain = f64::from(MAX_PREDICTION_POWER_GAIN);
    let mut inv_gain = 1.0f64;
    for k in (1..order).rev() {
        let rc = -f64::from(atmp[k]);
        let rc_mult1 = 1.0 - rc * rc;
        inv_gain *= rc_mult1;
        if inv_gain * max_gain < 1.0 {
            return 0.0;
        }
        let rc_mult2 = 1.0 / rc_mult1;
        for n in 0..(k + 1) >> 1 {
            let tmp1 = f64::from(atmp[n]);
            let tmp2 = f64::from(atmp[k - n - 1]);
            atmp[n] = ((tmp1 - tmp2 * rc) * rc_mult2) as f32;
            atmp[k - n - 1] = ((tmp2 - tmp1 * rc) * rc_mult2) as f32;
        }
    }
    let rc = -f64::from(atmp[0]);
    let rc_mult1 = 1.0 - rc * rc;
    inv_gain *= rc_mult1;
    if inv_gain * max_gain < 1.0 {
        return 0.0;
    }
    inv_gain as f32
}

/// libopus `silk_LPC_analysis_filter_FLP` for the orders 6, 8, 10, 12 and
/// 16: `r_lpc[..length]` is the residual of `s` with zero history, its
/// first `pred_coef.len()` samples zero.
pub fn silk_lpc_analysis_filter_flp(r_lpc: &mut [f32], pred_coef: &[f32], s: &[f32], length: usize) {
    match pred_coef.len() {
        6 => lpc_filter::<6>(r_lpc, pred_coef, s, length),
        8 => lpc_filter::<8>(r_lpc, pred_coef, s, length),
        10 => lpc_filter::<10>(r_lpc, pred_coef, s, length),
        12 => lpc_filter::<12>(r_lpc, pred_coef, s, length),
        16 => lpc_filter::<16>(r_lpc, pred_coef, s, length),
        order => panic!("LPC order {order}"),
    }
}

#[inline(always)]
fn lpc_filter<const D: usize>(r_lpc: &mut [f32], pred_coef: &[f32], s: &[f32], length: usize) {
    let c: &[f32; D] = pred_coef.try_into().unwrap();
    for ix in D..length {
        // The products summed from the latest sample back.
        let mut pred = s[ix - 1] * c[0];
        for j in 1..D {
            pred += s[ix - 1 - j] * c[j];
        }
        r_lpc[ix] = s[ix] - pred;
    }
    r_lpc[..D].fill(0.0);
}

/// libopus `silk_burg_modified_FLP`: prediction coefficients `a[..d]` of the
/// `nb_subfr` subframes of `x`, each `subfr_length` long including its `d`
/// preceding samples, with the prediction gain capped at `1 /
/// min_inv_gain`; returns the residual energy.
pub fn silk_burg_modified_flp(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) -> f32 {
    let mut c_first_row = [0.0f64; SILK_MAX_ORDER_LPC];
    let mut caf = [0.0f64; SILK_MAX_ORDER_LPC + 1];
    let mut cab = [0.0f64; SILK_MAX_ORDER_LPC + 1];
    let mut af = [0.0f64; SILK_MAX_ORDER_LPC];
    let min_inv_gain = f64::from(min_inv_gain);
    let cond = f64::from(FIND_LPC_COND_FAC);

    let mut c0 = silk_energy_flp(&x[..nb_subfr * subfr_length]);
    for s in 0..nb_subfr {
        let x_ptr = &x[s * subfr_length..(s + 1) * subfr_length];
        for n in 1..=d {
            c_first_row[n - 1] += silk_inner_product_flp(&x_ptr[..subfr_length - n], &x_ptr[n..]);
        }
    }
    let mut c_last_row = c_first_row;

    caf[0] = c0 + cond * c0 + 1e-1;
    cab[0] = caf[0];
    let mut inv_gain = 1.0f64;
    let mut reached_max_gain = false;
    for n in 0..d {
        for s in 0..nb_subfr {
            let x_ptr = &x[s * subfr_length..(s + 1) * subfr_length];
            let xn = x_ptr[n];
            let xl = x_ptr[subfr_length - n - 1];
            let mut tmp1 = f64::from(xn);
            let mut tmp2 = f64::from(xl);
            for k in 0..n {
                c_first_row[k] -= f64::from(xn * x_ptr[n - k - 1]);
                c_last_row[k] -= f64::from(xl * x_ptr[subfr_length - n + k]);
                let atmp = af[k];
                tmp1 += f64::from(x_ptr[n - k - 1]) * atmp;
                tmp2 += f64::from(x_ptr[subfr_length - n + k]) * atmp;
            }
            for k in 0..=n {
                caf[k] -= tmp1 * f64::from(x_ptr[n - k]);
                cab[k] -= tmp2 * f64::from(x_ptr[subfr_length - n + k - 1]);
            }
        }
        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        for k in 0..n {
            let atmp = af[k];
            tmp1 += c_last_row[n - k - 1] * atmp;
            tmp2 += c_first_row[n - k - 1] * atmp;
        }
        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;

        let mut num = cab[n + 1];
        let mut nrg_b = cab[0];
        let mut nrg_f = caf[0];
        for k in 0..n {
            let atmp = af[k];
            num += cab[n - k] * atmp;
            nrg_b += cab[k + 1] * atmp;
            nrg_f += caf[k + 1] * atmp;
        }

        let mut rc = -2.0 * num / (nrg_f + nrg_b);
        let tmp1 = inv_gain * (1.0 - rc * rc);
        if tmp1 <= min_inv_gain {
            // Hit the maximum prediction gain exactly, with rc's sign.
            rc = (1.0 - min_inv_gain / inv_gain).sqrt();
            if num > 0.0 {
                rc = -rc;
            }
            inv_gain = min_inv_gain;
            reached_max_gain = true;
        } else {
            inv_gain = tmp1;
        }

        for k in 0..(n + 1) >> 1 {
            let tmp1 = af[k];
            let tmp2 = af[n - k - 1];
            af[k] = tmp1 + rc * tmp2;
            af[n - k - 1] = tmp2 + rc * tmp1;
        }
        af[n] = rc;

        if reached_max_gain {
            af[n + 1..d].fill(0.0);
            break;
        }

        for k in 0..=n + 1 {
            let tmp1 = caf[k];
            caf[k] += rc * cab[n + 1 - k];
            cab[n + 1 - k] += rc * tmp1;
        }
    }

    let nrg_f = if reached_max_gain {
        for k in 0..d {
            a[k] = (-af[k]) as f32;
        }
        for s in 0..nb_subfr {
            c0 -= silk_energy_flp(&x[s * subfr_length..s * subfr_length + d]);
        }
        c0 * inv_gain
    } else {
        let mut nrg_f = caf[0];
        let mut tmp1 = 1.0f64;
        for k in 0..d {
            let atmp = af[k];
            nrg_f += caf[k + 1] * atmp;
            tmp1 += atmp * atmp;
            a[k] = (-atmp) as f32;
        }
        nrg_f - cond * c0 * tmp1
    };
    nrg_f as f32
}

/// libopus `silk_scale_copy_vector_FLP`.
pub fn silk_scale_copy_vector_flp(data_out: &mut [f32], data_in: &[f32], gain: f32) {
    for (o, &x) in data_out.iter_mut().zip(data_in) {
        *o = gain * x;
    }
}

/// libopus `silk_scale_vector_FLP`.
pub fn silk_scale_vector_flp(data: &mut [f32], gain: f32) {
    for x in data {
        *x *= gain;
    }
}

/// libopus `silk_insertion_sort_decreasing_FLP`: the `idx.len()` largest
/// values of `a` sorted to its front, decreasing, with their indices.
pub fn silk_insertion_sort_decreasing_flp(a: &mut [f32], idx: &mut [usize]) {
    let k = idx.len();
    for (i, v) in idx.iter_mut().enumerate() {
        *v = i;
    }
    for i in 1..k {
        let value = a[i];
        let mut j = i;
        while j > 0 && value > a[j - 1] {
            a[j] = a[j - 1];
            idx[j] = idx[j - 1];
            j -= 1;
        }
        a[j] = value;
        idx[j] = i;
    }
    for i in k..a.len() {
        let value = a[i];
        if value > a[k - 1] {
            let mut j = k - 1;
            while j > 0 && value > a[j - 1] {
                a[j] = a[j - 1];
                idx[j] = idx[j - 1];
                j -= 1;
            }
            a[j] = value;
            idx[j] = i;
        }
    }
}

/// libopus `silk_apply_sine_window_FLP`: `px_win` is `px[..length]` times a
/// sine window from 0 to pi/2 (`win_type` 1) or from pi/2 to pi (2);
/// `length` is a multiple of 4.
pub fn silk_apply_sine_window_flp(px_win: &mut [f32], px: &[f32], win_type: i32, length: usize) {
    let freq = PI / (length + 1) as f32;
    // Approximation of 2 * cos(f).
    let c = 2.0f32 - freq * freq;
    let (mut s0, mut s1) = if win_type < 2 {
        (0.0f32, freq)
    } else {
        (1.0f32, 0.5f32 * c)
    };
    // sin(n*f) = 2 * cos(f) * sin((n-1)*f) - sin((n-2)*f), four at a time.
    for k in (0..length).step_by(4) {
        px_win[k] = px[k] * 0.5 * (s0 + s1);
        px_win[k + 1] = px[k + 1] * s1;
        s0 = c * s1 - s0;
        px_win[k + 2] = px[k + 2] * 0.5 * (s1 + s0);
        px_win[k + 3] = px[k + 3] * s0;
        s1 = c * s0 - s1;
    }
}

/// libopus `silk_warped_autocorrelation_FLP`: `corr[..=order]` of `input`
/// on a frequency axis warped by `warping`; `order` is even.
pub fn silk_warped_autocorrelation_flp(corr: &mut [f32], input: &[f32], warping: f32, order: usize) {
    const N: usize = crate::silk::define::MAX_SHAPE_LPC_ORDER + 1;
    let mut state = [0.0f64; N];
    let mut c = [0.0f64; N];
    let warping = f64::from(warping);
    for &x in input {
        let mut tmp1 = f64::from(x);
        for i in (0..order).step_by(2) {
            // Two products, as in libopus, for a shorter dependency chain.
            let tmp2 = state[i] + warping * state[i + 1] - warping * tmp1;
            state[i] = tmp1;
            c[i] += state[0] * tmp1;
            tmp1 = state[i + 1] + warping * state[i + 2] - warping * tmp2;
            state[i + 1] = tmp2;
            c[i + 1] += state[0] * tmp2;
        }
        state[order] = tmp1;
        c[order] += state[0] * tmp1;
    }
    for (o, &v) in corr[..=order].iter_mut().zip(&c) {
        *o = v as f32;
    }
}
