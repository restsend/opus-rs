//! Burg's method in double precision (libopus `dnn/burg.c`, a copy of
//! SILK's `silk_burg_modified_FLP`). Products of two floats that C computes
//! in float stay in float here.

#[cfg(not(feature = "std"))]
use crate::compat::Math;

const MAX_FRAME_SIZE: usize = 384;
const SILK_MAX_ORDER_LPC: usize = 16;
const FIND_LPC_COND_FAC: f32 = 1e-5;

/// `silk_energy_FLP`: products in double, four per group.
fn energy(data: &[f32]) -> f64 {
    let n = data.len();
    let mut result = 0.0f64;
    let mut i = 0;
    while i + 3 < n {
        result += data[i] as f64 * data[i] as f64
            + data[i + 1] as f64 * data[i + 1] as f64
            + data[i + 2] as f64 * data[i + 2] as f64
            + data[i + 3] as f64 * data[i + 3] as f64;
        i += 4;
    }
    while i < n {
        result += data[i] as f64 * data[i] as f64;
        i += 1;
    }
    result
}

/// `silk_inner_product_FLP`.
fn inner_product(a: &[f32], b: &[f32], n: usize) -> f64 {
    let mut result = 0.0f64;
    let mut i = 0;
    while i + 3 < n {
        result += a[i] as f64 * b[i] as f64
            + a[i + 1] as f64 * b[i + 1] as f64
            + a[i + 2] as f64 * b[i + 2] as f64
            + a[i + 3] as f64 * b[i + 3] as f64;
        i += 4;
    }
    while i < n {
        result += a[i] as f64 * b[i] as f64;
        i += 1;
    }
    result
}

/// `silk_burg_analysis`: `d` prediction coefficients of `x`
/// (`nb_subfr` subframes of `subfr_length` samples) into `a`; returns the
/// residual energy.
pub fn silk_burg_analysis(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) -> f32 {
    assert!(subfr_length * nb_subfr <= MAX_FRAME_SIZE && d <= SILK_MAX_ORDER_LPC);
    let mut c_first_row = [0.0f64; SILK_MAX_ORDER_LPC];
    let mut caf = [0.0f64; SILK_MAX_ORDER_LPC + 1];
    let mut cab = [0.0f64; SILK_MAX_ORDER_LPC + 1];
    let mut af = [0.0f64; SILK_MAX_ORDER_LPC];
    let mut reached_max_gain = false;

    let mut c0 = energy(&x[..nb_subfr * subfr_length]);
    for s in 0..nb_subfr {
        let xp = &x[s * subfr_length..];
        for n in 1..d + 1 {
            c_first_row[n - 1] += inner_product(xp, &xp[n..], subfr_length - n);
        }
    }
    let mut c_last_row = c_first_row;

    let init = c0 + FIND_LPC_COND_FAC as f64 * c0 + 1e-9f32 as f64;
    caf[0] = init;
    cab[0] = init;
    let mut inv_gain = 1.0f64;
    let mig = min_inv_gain as f64;
    for n in 0..d {
        for s in 0..nb_subfr {
            let xp = &x[s * subfr_length..];
            let mut tmp1 = xp[n] as f64;
            let mut tmp2 = xp[subfr_length - n - 1] as f64;
            for k in 0..n {
                c_first_row[k] -= (xp[n] * xp[n - k - 1]) as f64;
                c_last_row[k] -= (xp[subfr_length - n - 1] * xp[subfr_length - n + k]) as f64;
                let atmp = af[k];
                tmp1 += xp[n - k - 1] as f64 * atmp;
                tmp2 += xp[subfr_length - n + k] as f64 * atmp;
            }
            for k in 0..=n {
                caf[k] -= tmp1 * xp[n - k] as f64;
                cab[k] -= tmp2 * xp[subfr_length - n + k - 1] as f64;
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
        let t = inv_gain * (1.0 - rc * rc);
        if t <= mig {
            rc = (1.0 - mig / inv_gain).sqrt();
            if num > 0.0 {
                rc = -rc;
            }
            inv_gain = mig;
            reached_max_gain = true;
        } else {
            inv_gain = t;
        }

        for k in 0..(n + 1) >> 1 {
            let t1 = af[k];
            let t2 = af[n - k - 1];
            af[k] = t1 + rc * t2;
            af[n - k - 1] = t2 + rc * t1;
        }
        af[n] = rc;

        if reached_max_gain {
            for v in af[n + 1..d].iter_mut() {
                *v = 0.0;
            }
            break;
        }

        for k in 0..=n + 1 {
            let t1 = caf[k];
            caf[k] += rc * cab[n - k + 1];
            cab[n - k + 1] += rc * t1;
        }
    }

    let nrg_f = if reached_max_gain {
        for k in 0..d {
            a[k] = (-af[k]) as f32;
        }
        for s in 0..nb_subfr {
            c0 -= energy(&x[s * subfr_length..s * subfr_length + d]);
        }
        c0 * inv_gain
    } else {
        let mut nf = caf[0];
        let mut tmp1 = 1.0f64;
        for k in 0..d {
            let atmp = af[k];
            nf += caf[k + 1] * atmp;
            tmp1 += atmp * atmp;
            a[k] = (-atmp) as f32;
        }
        nf -= FIND_LPC_COND_FAC as f64 * c0 * tmp1;
        nf
    };
    let r = nrg_f as f32;
    if 0.0 > r { 0.0 } else { r }
}
