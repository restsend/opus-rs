use crate::celt_lpc::{autocorr, lpc};

/// Inner product with NEON optimization on aarch64
pub fn inner_prod(x: &[f32], y: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return inner_prod_neon(x, y, n);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += x[i] * y[i];
        }
        sum
    }
}

/// Dual inner product with NEON optimization on aarch64
pub fn dual_inner_prod(x: &[f32], y1: &[f32], y2: &[f32], n: usize) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return dual_inner_prod_neon(x, y1, y2, n);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut xy1 = 0.0f32;
        let mut xy2 = 0.0f32;
        for i in 0..n {
            xy1 += x[i] * y1[i];
            xy2 += x[i] * y2[i];
        }
        (xy1, xy2)
    }
}

/// Pitch cross-correlation with NEON optimization on aarch64
pub fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        // For large max_pitch, use xcorr_kernel_neon which computes 4 correlations
        // simultaneously with shared y-vector loads. For small max_pitch (e.g. autocorr
        // with lag ≤ 24), the kernel setup overhead exceeds the data-sharing benefit,
        // so individual inner_prod_neon calls are faster on Apple Silicon.
        if max_pitch >= 32 {
            unsafe {
                return pitch_xcorr_neon(x, y, xcorr, len, max_pitch);
            }
        }
        for i in 0..max_pitch {
            xcorr[i] = unsafe { inner_prod_neon(x, &y[i..], len) };
        }
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..max_pitch {
            xcorr[i] = inner_prod(x, &y[i..], len);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn inner_prod_neon(x: &[f32], y: &[f32], n: usize) -> f32 {
    use std::arch::aarch64::*;

    let mut xy = vdupq_n_f32(0.0);
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let y0 = vld1q_f32(y.as_ptr().add(i));
        xy = vfmaq_f32(xy, x0, y0);

        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        let y1 = vld1q_f32(y.as_ptr().add(i + 4));
        xy = vfmaq_f32(xy, x1, y1);

        i += 8;
    }

    // Process 4 more elements
    if i + 4 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let y0 = vld1q_f32(y.as_ptr().add(i));
        xy = vfmaq_f32(xy, x0, y0);
        i += 4;
    }

    // Horizontal sum
    let mut sum = vaddvq_f32(xy);

    // Scalar tail
    for j in i..n {
        sum += x[j] * y[j];
    }

    sum
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dual_inner_prod_neon(x: &[f32], y1: &[f32], y2: &[f32], n: usize) -> (f32, f32) {
    use std::arch::aarch64::*;

    let mut xy1 = vdupq_n_f32(0.0);
    let mut xy2 = vdupq_n_f32(0.0);
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x4 = vld1q_f32(x.as_ptr().add(i + 4));

        let y1_0 = vld1q_f32(y1.as_ptr().add(i));
        let y1_4 = vld1q_f32(y1.as_ptr().add(i + 4));
        let y2_0 = vld1q_f32(y2.as_ptr().add(i));
        let y2_4 = vld1q_f32(y2.as_ptr().add(i + 4));

        xy1 = vfmaq_f32(xy1, x0, y1_0);
        xy2 = vfmaq_f32(xy2, x0, y2_0);
        xy1 = vfmaq_f32(xy1, x4, y1_4);
        xy2 = vfmaq_f32(xy2, x4, y2_4);

        i += 8;
    }

    // Process 4 more elements
    if i + 4 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let y1_0 = vld1q_f32(y1.as_ptr().add(i));
        let y2_0 = vld1q_f32(y2.as_ptr().add(i));
        xy1 = vfmaq_f32(xy1, x0, y1_0);
        xy2 = vfmaq_f32(xy2, x0, y2_0);
        i += 4;
    }

    let sum1 = vaddvq_f32(xy1);
    let sum2 = vaddvq_f32(xy2);

    let mut s1 = sum1;
    let mut s2 = sum2;
    for j in i..n {
        s1 += x[j] * y1[j];
        s2 += x[j] * y2[j];
    }

    (s1, s2)
}

/// Compute 4 cross-correlation values using NEON lane multiplies
/// sum[k] = Σ(x[i+k] * y[i]) for k=0..3
/// Matches C implementation: celt_neon_intr.c xcorr_kernel_neon_float
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xcorr_kernel_neon(x: &[f32], y: &[f32], sum: &mut [f32; 4], mut len: usize) {
    use std::arch::aarch64::*;

    debug_assert!(x.len() >= len, "xcorr_kernel_neon: x too short");
    debug_assert!(y.len() >= len + 3, "xcorr_kernel_neon: y too short (need len+3 for vextq)");

    let mut summ = vdupq_n_f32(0.0);
    let mut xi = x.as_ptr();
    let mut yi = y.as_ptr();

    // Load initial y[0..3]
    let mut yy = vld1q_f32(yi);

    // Process 8 elements at a time
    // Note: loop condition is len > 8 (not >=) to avoid reading past array bounds
    while len > 8 {
        yi = yi.add(4);
        let yy1 = vld1q_f32(yi);
        yi = yi.add(4);
        let yy2 = vld1q_f32(yi);

        let xx0 = vld1q_f32(xi);
        xi = xi.add(4);
        let xx1 = vld1q_f32(xi);
        xi = xi.add(4);

        // Compute 4 correlations using lane multiplies with rotated y vectors
        summ = vfmaq_lane_f32(summ, yy, vget_low_f32(xx0), 0);
        let yext = vextq_f32(yy, yy1, 1);
        summ = vfmaq_lane_f32(summ, yext, vget_low_f32(xx0), 1);
        let yext = vextq_f32(yy, yy1, 2);
        summ = vfmaq_lane_f32(summ, yext, vget_high_f32(xx0), 0);
        let yext = vextq_f32(yy, yy1, 3);
        summ = vfmaq_lane_f32(summ, yext, vget_high_f32(xx0), 1);

        summ = vfmaq_lane_f32(summ, yy1, vget_low_f32(xx1), 0);
        let yext = vextq_f32(yy1, yy2, 1);
        summ = vfmaq_lane_f32(summ, yext, vget_low_f32(xx1), 1);
        let yext = vextq_f32(yy1, yy2, 2);
        summ = vfmaq_lane_f32(summ, yext, vget_high_f32(xx1), 0);
        let yext = vextq_f32(yy1, yy2, 3);
        summ = vfmaq_lane_f32(summ, yext, vget_high_f32(xx1), 1);

        yy = yy2;
        len -= 8;
    }

    // Process 4 more elements
    // Note: condition is len > 4 (not >=) to avoid reading past array bounds
    if len > 4 {
        yi = yi.add(4);
        let yy1 = vld1q_f32(yi);

        let xx0 = vld1q_f32(xi);
        xi = xi.add(4);

        summ = vfmaq_lane_f32(summ, yy, vget_low_f32(xx0), 0);
        let yext = vextq_f32(yy, yy1, 1);
        summ = vfmaq_lane_f32(summ, yext, vget_low_f32(xx0), 1);
        let yext = vextq_f32(yy, yy1, 2);
        summ = vfmaq_lane_f32(summ, yext, vget_high_f32(xx0), 0);
        let yext = vextq_f32(yy, yy1, 3);
        summ = vfmaq_lane_f32(summ, yext, vget_high_f32(xx0), 1);

        yy = yy1;
        len -= 4;
    }

    // Process remaining elements one at a time
    // Note: --len > 0 means loop while len-1 > 0, i.e., len > 1
    while len > 1 {
        let xx = vld1_dup_f32(xi);
        xi = xi.add(1);
        summ = vfmaq_lane_f32(summ, yy, xx, 0);
        yi = yi.add(1);
        yy = vld1q_f32(yi);
        len -= 1;
    }

    // Final element
    let xx = vld1_dup_f32(xi);
    summ = vfmaq_lane_f32(summ, yy, xx, 0);

    vst1q_f32(sum.as_mut_ptr(), summ);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn pitch_xcorr_neon(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
    debug_assert!(x.len() >= len, "pitch_xcorr_neon: x too short");
    debug_assert!(y.len() >= max_pitch + len - 1, "pitch_xcorr_neon: y too short");
    debug_assert!(xcorr.len() >= max_pitch, "pitch_xcorr_neon: xcorr too short");

    let mut i = 0;

    // Process 4 pitch values at a time using kernel
    while i + 4 <= max_pitch {
        let mut sum = [0.0f32; 4];
        unsafe { xcorr_kernel_neon(x, &y[i..], &mut sum, len) };
        xcorr[i] = sum[0];
        xcorr[i + 1] = sum[1];
        xcorr[i + 2] = sum[2];
        xcorr[i + 3] = sum[3];
        i += 4;
    }

    // Scalar tail
    for j in i..max_pitch {
        xcorr[j] = unsafe { inner_prod_neon(x, &y[j..], len) };
    }
}

fn celt_fir5(x: &mut [f32], num: &[f32], n: usize) {
    let mut mem = [0.0f32; 5];

    let num0 = num[0];
    let num1 = num[1];
    let num2 = num[2];
    let num3 = num[3];
    let num4 = num[4];

    for i in 0..n {
        let mut sum = x[i];
        sum += num0 * mem[0];
        sum += num1 * mem[1];
        sum += num2 * mem[2];
        sum += num3 * mem[3];
        sum += num4 * mem[4];

        mem[4] = mem[3];
        mem[3] = mem[2];
        mem[2] = mem[1];
        mem[1] = mem[0];
        mem[0] = x[i];

        x[i] = sum;
    }
}

pub fn pitch_downsample(
    x: &[&[f32]],
    x_lp: &mut [f32],
    len: usize,
    c: usize,
    factor: usize,
) {

    let offset = factor / 2;

    if x_lp.len() < len {
        return;
    }

    for i in 1..len {
        let mut val = 0.0f32;
        for k in 0..c {

            let x_k = x[k];

            let idx_m = factor * i - offset;
            let idx_p = factor * i + offset;
            let idx_c = factor * i;

            if idx_p < x_k.len() {
                val += 0.25 * x_k[idx_m] + 0.25 * x_k[idx_p] + 0.5 * x_k[idx_c];
            }
        }
        x_lp[i] = val;
    }

    {
        let mut val = 0.0f32;
        for k in 0..c {
            let x_k = x[k];

            let idx_offset = offset;
            let idx_0 = 0;
            if idx_offset < x_k.len() {
                val += 0.25 * x_k[idx_offset] + 0.5 * x_k[idx_0];
            }
        }
        x_lp[0] = val;
    }

    let mut ac = [0.0f32; 5];
    autocorr(&x_lp[0..len], &mut ac, None, 0, 4, len);

    ac[0] *= 1.0001;

    for i in 1..=4 {
        let f = 0.008 * (i as f32);
        ac[i] -= ac[i] * f * f;
    }

    let mut lpc_coeffs = [0.0f32; 4];
    lpc(&mut lpc_coeffs, &ac, 4);

    let mut tmp = 1.0f32;
    for i in 0..4 {
        tmp *= 0.9;
        lpc_coeffs[i] *= tmp;
    }

    let c1 = 0.8f32;
    let mut lpc2 = [0.0f32; 5];
    lpc2[0] = lpc_coeffs[0] + c1;
    lpc2[1] = lpc_coeffs[1] + c1 * lpc_coeffs[0];
    lpc2[2] = lpc_coeffs[2] + c1 * lpc_coeffs[1];
    lpc2[3] = lpc_coeffs[3] + c1 * lpc_coeffs[2];
    lpc2[4] = c1 * lpc_coeffs[3];

    celt_fir5(x_lp, &lpc2, len);
}

fn find_best_pitch(
    xcorr: &[f32],
    y: &[f32],
    len: usize,
    max_pitch: usize,
    best_pitch: &mut [usize; 2],
) {
    let mut best_num = [-1.0f32, -1.0f32];
    let mut best_den = [0.0f32, 0.0f32];

    best_pitch[0] = 0;
    best_pitch[1] = 1;

    let mut syy = 1.0f32;
    for j in 0..len {
        syy += y[j] * y[j];
    }

    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            let num = xcorr[i] * xcorr[i];
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i;
                }
            }
        }
        syy += y[i + len] * y[i + len] - y[i] * y[i];
        if syy < 1.0 {
            syy = 1.0;
        }
    }
}

pub fn pitch_search(x_lp: &[f32], y: &[f32], mut len: usize, mut max_pitch: usize) -> usize {
    let mut best_pitch = [0, 0];

    max_pitch >>= 1;
    len >>= 1;
    let lag = len + max_pitch;

    let mut x_lp4 = vec![0.0f32; len >> 1];
    let mut y_lp4 = vec![0.0f32; lag >> 1];
    let mut xcorr = vec![0.0f32; max_pitch];

    for j in 0..(len >> 1) {
        x_lp4[j] = x_lp[2 * j];
    }
    for j in 0..(lag >> 1) {
        y_lp4[j] = y[2 * j];
    }

    pitch_xcorr(&x_lp4, &y_lp4, &mut xcorr, len >> 1, max_pitch >> 1);

    find_best_pitch(&xcorr, &y_lp4, len >> 1, max_pitch >> 1, &mut best_pitch);

    for i in 0..max_pitch {
        xcorr[i] = -1.0;
        if (i as i32 - 2 * best_pitch[0] as i32).abs() > 2
            && (i as i32 - 2 * best_pitch[1] as i32).abs() > 2
        {
            continue;
        }
        xcorr[i] = inner_prod(x_lp, &y[i..], len);
        if xcorr[i] < -1.0 {
            xcorr[i] = -1.0;
        }
    }

    find_best_pitch(&xcorr, y, len, max_pitch, &mut best_pitch);

    let mut offset = 0;
    if best_pitch[0] > 0 && best_pitch[0] < max_pitch - 1 {
        let a = xcorr[best_pitch[0] - 1];
        let b = xcorr[best_pitch[0]];
        let c = xcorr[best_pitch[0] + 1];
        if (c - a) > 0.7 * (b - a) {
            offset = 1;
        } else if (a - c) > 0.7 * (b - c) {
            offset = -1;
        }
    }

    ((2 * best_pitch[0]) as isize).wrapping_sub(offset as isize) as usize
}

fn compute_pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    if xy <= 0.0 || xx <= 0.0 || yy <= 0.0 {
        return 0.0;
    }
    xy / (1.0 + xx * yy).sqrt()
}

static SECOND_CHECK: [usize; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

pub fn remove_doubling(
    x: &[f32],
    mut max_period: usize,
    mut min_period: usize,
    mut n: usize,
    t0_ptr: &mut usize,
    mut prev_period: usize,
    prev_gain: f32,
) -> f32 {
    let min_period0 = min_period;
    max_period /= 2;
    min_period /= 2;
    *t0_ptr /= 2;
    prev_period /= 2;
    n /= 2;

    let x_target = &x[max_period..];

    if *t0_ptr >= max_period {
        *t0_ptr = max_period - 1;
    }

    let mut t = *t0_ptr;
    let t0 = *t0_ptr;

    let mut yy_lookup = vec![0.0f32; max_period + 1];
    let (xx, xy) = {
        let mut sum_xx = 0.0f32;
        let mut sum_xy = 0.0f32;
        for i in 0..n {
            sum_xx += x_target[i] * x_target[i];
            sum_xy += x_target[i] * x[max_period - t0 + i];
        }
        (sum_xx, sum_xy)
    };

    yy_lookup[0] = xx;
    let mut yy_curr = xx;
    for i in 1..=max_period {
        yy_curr = yy_curr + x[max_period - i] * x[max_period - i]
            - x[max_period + n - i] * x[max_period + n - i];
        if yy_curr < 0.0 {
            yy_curr = 0.0;
        }
        yy_lookup[i] = yy_curr;
    }

    let mut best_xy = xy;
    let mut best_yy = yy_lookup[t0];
    let mut g = compute_pitch_gain(best_xy, xx, best_yy);
    let g0 = g;

    for k in 2..=15 {
        let t1 = (2 * t0 + k) / (2 * k);
        if t1 < min_period {
            break;
        }

        let t1b;
        if k == 2 {
            if t1 + t0 > max_period {
                t1b = t0;
            } else {
                t1b = t0 + t1;
            }
        } else {
            t1b = (2 * SECOND_CHECK[k] * t0 + k) / (2 * k);
        }

        let (xy_a, xy_b) =
            dual_inner_prod(x_target, &x[max_period - t1..], &x[max_period - t1b..], n);
        let xy_new = 0.5 * (xy_a + xy_b);
        let yy_new = 0.5 * (yy_lookup[t1] + yy_lookup[t1b]);
        let g1 = compute_pitch_gain(xy_new, xx, yy_new);

        let mut cont = 0.0f32;
        if (t1 as i32 - prev_period as i32).abs() <= 1 {
            cont = prev_gain;
        } else if (t1 as i32 - prev_period as i32).abs() <= 2 && 5 * k * k < t0 {
            cont = 0.5 * prev_gain;
        }

        let mut thresh = (0.7 * g0 - cont).max(0.3);
        if t1 < 3 * min_period {
            thresh = (0.85 * g0 - cont).max(0.4);
        } else if t1 < 2 * min_period {
            thresh = (0.9 * g0 - cont).max(0.5);
        }

        if g1 > thresh {
            best_xy = xy_new;
            best_yy = yy_new;
            t = t1;
            g = g1;
        }
    }

    best_xy = best_xy.max(0.0);
    let pg = if best_yy <= best_xy {
        1.0f32
    } else {
        best_xy / (best_yy + 1.0)
    };

    let mut xcorr_res = [0.0f32; 3];
    for k_idx in 0..3 {
        let lag = (t as i32 + k_idx as i32 - 1) as usize;
        xcorr_res[k_idx] = inner_prod(x_target, &x[max_period - lag..], n);
    }

    let mut offset = 0;
    if (xcorr_res[2] - xcorr_res[0]) > 0.7 * (xcorr_res[1] - xcorr_res[0]) {
        offset = 1;
    } else if (xcorr_res[0] - xcorr_res[2]) > 0.7 * (xcorr_res[1] - xcorr_res[2]) {
        offset = -1;
    }

    let pg = pg.min(g);
    *t0_ptr = (2 * t as i32 + offset) as usize;
    if *t0_ptr < min_period0 {
        *t0_ptr = min_period0;
    }

    pg
}
