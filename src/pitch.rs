use crate::celt_lpc::{autocorr, lpc};

pub fn inner_prod(x: &[f32], y: &[f32], n: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += x[i] * y[i];
    }
    sum
}

pub fn dual_inner_prod(x: &[f32], y1: &[f32], y2: &[f32], n: usize) -> (f32, f32) {
    let mut xy1 = 0.0f32;
    let mut xy2 = 0.0f32;
    for i in 0..n {
        xy1 += x[i] * y1[i];
        xy2 += x[i] * y2[i];
    }
    (xy1, xy2)
}

pub fn pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
    for i in 0..max_pitch {
        xcorr[i] = inner_prod(x, &y[i..], len);
    }
}

fn celt_fir5(x: &mut [f32], num: &[f32], n: usize) {
    let mut mem = [0.0f32; 5];
    // num is lpc2, size 5.
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
    x: &[&[f32]], // Multichannel input
    x_lp: &mut [f32],
    len: usize, // Length of x_lp
    c: usize,   // Number of channels
    factor: usize,
) {
    // Corresponds to pitch_downsample in celt/pitch.c
    let offset = factor / 2;
    // Opus `pitch_downsample` signature has `len`.
    // In `celt_pitch_xcorr` it calls `pitch_downsample(x, x_lp, len, C, 4, arch)`.
    // In `pitch_downsample` C impl:
    // for (i=1; i<len; i++) ...
    // The `len` param passed into `pitch_downsample` seems to be the OUTPUT length?
    // Wait, let's check `celt/celt_encoder.c`.
    // `pitch_downsample(in, pitch_buf, NB_PITCH_EXTRACT, C, 4, arch);`
    // NB_PITCH_EXTRACT = PITCH_MAX + PITCH_BUF_SIZE.
    // It seems `len` is the number of samples to produce in `x_lp`?
    // In `pitch.c`:
    // for (i=1; i<len; i++) x_lp[i] = ...
    // So yes, `len` is the length of `x_lp`.

    // Safety
    if x_lp.len() < len {
        return;
    }

    // 1. Downsampling
    for i in 1..len {
        let mut val = 0.0f32;
        for k in 0..c {
            // formula: .25*x[factors*i - offset] + .25*x[factors*i+offset] + .5*x[factors*i]
            let x_k = x[k];
            // Check bounds
            let idx_m = factor * i - offset; // 4*i - 2
            let idx_p = factor * i + offset; // 4*i + 2
            let idx_c = factor * i; // 4*i

            // Assume x is padded/long enough?
            // In C, `celt_pitch_xcorr` calls `pitch_downsample` with `x` being the input.
            // We need to be careful with indices.
            if idx_p < x_k.len() {
                val += 0.25 * x_k[idx_m] + 0.25 * x_k[idx_p] + 0.5 * x_k[idx_c];
            }
        }
        x_lp[i] = val;
    }
    // i=0 case
    {
        let mut val = 0.0f32;
        for k in 0..c {
            let x_k = x[k];
            // x_lp[0] = .25f*x[0][offset] + .5f*x[0][0];
            // (first term x[0][-offset] is out of bounds so assumed 0 or handled? No, C starts at offset)
            // C code: x_lp[0] = .25f*x[0][offset] + .5f*x[0][0];
            let idx_offset = offset;
            let idx_0 = 0;
            if idx_offset < x_k.len() {
                val += 0.25 * x_k[idx_offset] + 0.5 * x_k[idx_0];
            }
        }
        x_lp[0] = val;
    }

    // 2. Autocorrelation on x_lp
    let mut ac = [0.0f32; 5];
    autocorr(&x_lp[0..len], &mut ac, None, 0, 4, len);

    // 3. Noise floor
    ac[0] *= 1.0001;

    // 4. Lag windowing
    // ac[i] -= ac[i]*(.008f*i)*(.008f*i);
    for i in 1..=4 {
        let f = 0.008 * (i as f32);
        ac[i] -= ac[i] * f * f;
    }

    // 5. LPC
    let mut lpc_coeffs = [0.0f32; 4];
    lpc(&mut lpc_coeffs, &ac, 4);

    // 6. Bandwidth expansion
    let mut tmp = 1.0f32;
    for i in 0..4 {
        tmp *= 0.9; // QCONST16(.9f, 15) is 0.9.
        lpc_coeffs[i] *= tmp;
    }

    // 7. Add a zero (pre-emphasis-ish)
    // lpc2[0] = lpc[0] + 0.8
    // lpc2[1] = lpc[1] + 0.8*lpc[0]
    // ...
    // lpc2[4] = 0.8*lpc[3]
    let c1 = 0.8f32;
    let mut lpc2 = [0.0f32; 5];
    lpc2[0] = lpc_coeffs[0] + c1;
    lpc2[1] = lpc_coeffs[1] + c1 * lpc_coeffs[0];
    lpc2[2] = lpc_coeffs[2] + c1 * lpc_coeffs[1];
    lpc2[3] = lpc_coeffs[3] + c1 * lpc_coeffs[2];
    lpc2[4] = c1 * lpc_coeffs[3];

    // 8. FIR Filter
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

    // Downsample by 2 again (it was already downsampled by 4 in pitch_downsample)
    // So this is 8x downsampling relative to original.
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

    /* Coarse search with 4x decimation (of the input to this function, so 8x total) */
    pitch_xcorr(&x_lp4, &y_lp4, &mut xcorr, len >> 1, max_pitch >> 1);

    find_best_pitch(&xcorr, &y_lp4, len >> 1, max_pitch >> 1, &mut best_pitch);

    /* Finer search with 2x decimation (4x total) */
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

    /* Refine by pseudo-interpolation */
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

    // x += max_period
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
