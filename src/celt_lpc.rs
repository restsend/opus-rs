use crate::fixedvec::FixedVec;
use crate::pitch::pitch_xcorr;

pub fn lpc(lpc: &mut [f32], ac: &[f32], p: usize) {
    let mut error = ac[0];
    if error <= 1e-10 {
        for x in lpc.iter_mut() {
            *x = 0.0;
        }
        return;
    }

    for i in 0..p {
        let mut rr = 0.0f32;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;

        lpc[i] = r;
        for j in 0..i.div_ceil(2) {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }

        error = error - r * r * error;

        if error <= 0.001 * ac[0] {
            break;
        }
    }
}

pub fn autocorr(
    x: &[f32],
    ac: &mut [f32],
    window: Option<&[f32]>,
    overlap: usize,
    lag: usize,
    n: usize,
) {
    let xx_vec: FixedVec<f32, 2880>;
    let xx: &[f32] = if let Some(win) = window {
        if x.len() < n {
            return;
        }
        xx_vec = {
            let mut v: FixedVec<f32, 2880> = FixedVec::from_slice(&x[0..n]);
            for i in 0..overlap {
                v[i] *= win[i];
                v[n - 1 - i] *= win[i];
            }
            v
        };
        &xx_vec
    } else {
        &x[0..n]
    };

    let fast_n = n - lag;

    pitch_xcorr(xx, xx, ac, fast_n, lag + 1);

    for k in 0..=lag {
        let mut d = 0.0f32;
        for i in (k + fast_n)..n {
            d += xx[i] * xx[i - k];
        }
        ac[k] += d;
    }
}

pub fn celt_fir(x: &[f32], num: &[f32], y: &mut [f32], n: usize, ord: usize) {
    for i in 0..n {
        let mut sum = x[i];
        for j in 0..ord {
            if i > j {
                sum += num[j] * x[i - j - 1];
            }
        }
        y[i] = sum;
    }
}

pub fn celt_iir(x: &[f32], den: &[f32], y: &mut [f32], n: usize, ord: usize, mem: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        celt_iir_neon(x, den, y, n, ord, mem);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            let mut sum = x[i];
            for j in 0..ord {
                sum -= den[j] * mem[j];
            }
            for j in (1..ord).rev() {
                mem[j] = mem[j - 1];
            }
            mem[0] = sum;
            y[i] = sum;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn celt_iir_neon(
    x: &[f32],
    den: &[f32],
    y: &mut [f32],
    n: usize,
    ord: usize,
    mem: &mut [f32],
) {
    use core::arch::aarch64::*;

    if ord < 4 {
        for i in 0..n {
            let mut sum = x[i];
            for j in 0..ord {
                sum -= den[j] * mem[j];
            }
            for j in (1..ord).rev() {
                mem[j] = mem[j - 1];
            }
            mem[0] = sum;
            y[i] = sum;
        }
        return;
    }

    for i in 0..n {
        let mut feedback = vdupq_n_f32(0.0);

        let mut j = 0;
        while j + 4 <= ord {
            let coeff = vld1q_f32(den.as_ptr().add(j));
            let mem_vals = vld1q_f32(mem.as_ptr().add(j));
            feedback = vfmaq_f32(feedback, coeff, mem_vals);
            j += 4;
        }

        let fb_low = vget_low_f32(feedback);
        let fb_high = vget_high_f32(feedback);
        let fb_pair = vadd_f32(fb_low, fb_high);
        let mut fb_sum = vget_lane_f32(fb_pair, 0) + vget_lane_f32(fb_pair, 1);

        while j < ord {
            fb_sum += den[j] * mem[j];
            j += 1;
        }

        let sum = x[i] - fb_sum;

        for j in (1..ord).rev() {
            mem[j] = mem[j - 1];
        }
        mem[0] = sum;
        y[i] = sum;
    }
}
