use crate::kiss_fft::{KissCpx, KissFftState, opus_fft_impl};
use std::f32::consts::PI;
use std::mem::MaybeUninit;

const MAX_N2: usize = 960;
const MAX_N4: usize = 480;

pub struct MdctLookup {
    pub n: usize,
    pub max_lm: usize,
    kfft: Vec<Option<KissFftState>>,
    trig: Vec<f32>,
}

impl MdctLookup {
    pub fn new(n: usize, max_lm: usize) -> Self {
        let mut kfft = Vec::new();
        let mut trig = Vec::new();
        let mut curr_n = n;

        for shift in 0..=max_lm {
            let n4 = curr_n / 4;

            if shift == 0 {
                kfft.push(KissFftState::new(n4));
            } else if let Some(base) = kfft.first().unwrap().as_ref() {
                kfft.push(KissFftState::new_sub(base, n4));
            } else {
                kfft.push(None);
            }

            let n2 = curr_n / 2;
            for i in 0..n2 {
                let angle = 2.0 * PI * (i as f32 + 0.125) / curr_n as f32;
                trig.push(angle.cos());
            }

            curr_n >>= 1;
        }

        Self {
            n,
            max_lm,
            kfft,
            trig,
        }
    }

    fn get_trig(&self, shift: usize) -> (&[f32], usize) {
        let mut offset = 0;
        let mut curr_n = self.n;
        for _ in 0..shift {
            offset += curr_n / 2;
            curr_n >>= 1;
        }
        (&self.trig[offset..offset + curr_n / 2], curr_n / 4)
    }

    pub fn get_trig_debug(&self, shift: usize) -> &[f32] {
        let (trig, _) = self.get_trig(shift);
        trig
    }

    #[inline]
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let st = self.kfft[shift]
            .as_ref()
            .expect("FFT state not initialized");
        let n = self.n >> shift;
        let n2 = n / 2;
        let n4 = n / 4;
        let scale = st.scale();

        let (trig, _) = self.get_trig(shift);
        let overlap2 = overlap / 2;

        let mut f_buf = [MaybeUninit::<f32>::uninit(); MAX_N2];
        let mut f2_buf = [MaybeUninit::<KissCpx>::uninit(); MAX_N4];

        let f = unsafe { std::slice::from_raw_parts_mut(f_buf.as_mut_ptr() as *mut f32, n2) };
        let f2 = unsafe { std::slice::from_raw_parts_mut(f2_buf.as_mut_ptr() as *mut KissCpx, n4) };

        assert!(input.len() >= n2 + overlap2);
        assert!(window.len() >= overlap);
        assert!(
            output.len() >= n2,
            "MDCT forward: output buffer too small (need {}, have {})",
            n2,
            output.len()
        );

        {
            let mut yp = 0usize;
            let mut xp1 = overlap2;
            let mut xp2 = n2 - 1 + overlap2;
            let mut wp1 = overlap2;

            let mut wp2 = overlap2.saturating_sub(1);

            let limit = overlap.div_ceil(4);
            let mid = n4.saturating_sub(limit);

            let loop1_iters = limit.min(n4);
            for _ in 0..loop1_iters {
                let w1 = window[wp1];
                let w2 = window[wp2];

                f[yp] = input[xp1 + n2] * w2 + input[xp2] * w1;
                yp += 1;

                f[yp] = input[xp1] * w1 - input[xp2 - n2] * w2;
                yp += 1;

                xp1 += 2;
                xp2 -= 2;
                wp1 += 2;
                wp2 = wp2.saturating_sub(2);
            }

            for _ in limit..mid {
                f[yp] = input[xp2];
                yp += 1;

                f[yp] = input[xp1];
                yp += 1;
                xp1 += 2;
                xp2 -= 2;
            }

            let loop3_iters = if mid > limit { n4 - mid } else { 0 };
            let mut wp1_l3 = 0usize;
            let mut wp2_l3 = overlap.saturating_sub(1);
            for _ in 0..loop3_iters {
                let w1 = window[wp1_l3];
                let w2 = window[wp2_l3];

                f[yp] = -input[xp1 - n2] * w1 + input[xp2] * w2;
                yp += 1;

                f[yp] = input[xp1] * w2 + input[xp2 + n2] * w1;
                yp += 1;

                xp1 += 2;
                xp2 -= 2;
                wp1_l3 += 2;
                wp2_l3 -= 2;
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if std::arch::is_x86_feature_detected!("avx") {
                mdct_pre_rotation_avx(f, f2, trig, &st.bitrev[..n4], n4, scale);
            } else {
                for i in 0..n4 {
                    let re = f[2 * i];
                    let im = f[2 * i + 1];
                    let t0 = trig[i];
                    let t1 = trig[n4 + i];

                    let yr = re * t0 - im * t1;
                    let yi = im * t0 + re * t1;

                    f2[st.bitrev[i] as usize] = KissCpx::new(yr * scale, yi * scale);
                }
            }
        }
        #[cfg(all(
            not(any(target_arch = "x86", target_arch = "x86_64")),
            target_arch = "aarch64"
        ))]
        {
            mdct_pre_rotation_neon(f, f2, trig, &st.bitrev[..n4], n4, scale);
        }
        #[cfg(all(
            not(any(target_arch = "x86", target_arch = "x86_64")),
            not(target_arch = "aarch64")
        ))]
        for i in 0..n4 {
            let re = f[2 * i];
            let im = f[2 * i + 1];
            let t0 = trig[i];
            let t1 = trig[n4 + i];

            let yr = re * t0 - im * t1;
            let yi = im * t0 + re * t1;

            f2[st.bitrev[i] as usize] = KissCpx::new(yr * scale, yi * scale);
        }

        opus_fft_impl(st, f2);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if std::arch::is_x86_feature_detected!("avx") {
                mdct_post_rotation_avx(f2, trig, output, n4, n2, stride);
            } else {
                for i in 0..n4 {
                    let fp = &f2[i];
                    let t0 = trig[i];
                    let t1 = trig[n4 + i];

                    let yr = fp.i * t1 - fp.r * t0;
                    let yi = fp.r * t1 + fp.i * t0;

                    output[i * 2 * stride] = yr;
                    output[stride * (n2 - 1 - 2 * i)] = yi;
                }
            }
        }
        #[cfg(all(
            not(any(target_arch = "x86", target_arch = "x86_64")),
            target_arch = "aarch64"
        ))]
        {
            mdct_post_rotation_neon(f2, trig, output, n4, n2, stride);
        }
        #[cfg(all(
            not(any(target_arch = "x86", target_arch = "x86_64")),
            not(target_arch = "aarch64")
        ))]
        for i in 0..n4 {
            let fp = &f2[i];
            let t0 = trig[i];
            let t1 = trig[n4 + i];

            let yr = fp.i * t1 - fp.r * t0;
            let yi = fp.r * t1 + fp.i * t0;

            output[i * 2 * stride] = yr;
            output[stride * (n2 - 1 - 2 * i)] = yi;
        }
    }

    #[inline]
    pub fn backward(
        &self,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let st = self.kfft[shift]
            .as_ref()
            .expect("FFT state not initialized");
        let n = self.n >> shift;
        let n2 = n / 2;
        let n4 = n / 4;
        let overlap2 = overlap / 2;

        let (trig, _) = self.get_trig(shift);

        let mut f2_buf = [MaybeUninit::<KissCpx>::uninit(); MAX_N4];

        let f2 = unsafe { std::slice::from_raw_parts_mut(f2_buf.as_mut_ptr() as *mut KissCpx, n4) };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if std::arch::is_x86_feature_detected!("avx") {
                mdct_backward_pre_rotation_avx(input, f2, trig, &st.bitrev[..n4], n4, n2, stride);
            } else {
                for i in 0..n4 {
                    let rev = st.bitrev[i] as usize;
                    let x1 = input[2 * i * stride];
                    let x2 = input[stride * (n2 - 1 - 2 * i)];
                    let t0 = trig[i];
                    let t1 = trig[n4 + i];

                    let yr = x2 * t0 + x1 * t1;
                    let yi = x1 * t0 - x2 * t1;

                    f2[rev] = KissCpx::new(yi, yr);
                }
            }
        }
        #[cfg(all(
            not(any(target_arch = "x86", target_arch = "x86_64")),
            target_arch = "aarch64"
        ))]
        {
            mdct_backward_pre_rotation_neon(input, f2, trig, &st.bitrev[..n4], n4, n2, stride);
        }
        #[cfg(all(
            not(any(target_arch = "x86", target_arch = "x86_64")),
            not(target_arch = "aarch64")
        ))]
        for i in 0..n4 {
            let rev = st.bitrev[i] as usize;
            let x1 = input[2 * i * stride];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            let t0 = trig[i];
            let t1 = trig[n4 + i];

            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;

            f2[rev] = KissCpx::new(yi, yr);
        }

        opus_fft_impl(st, f2);

        assert!(output.len() >= overlap2 + n2);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if std::arch::is_x86_feature_detected!("avx") {
                mdct_backward_post_rotation_avx(f2, trig, output, n4, n2, overlap2);
            } else {
                for i in 0..((n4 + 1) >> 1) {
                    let im0 = f2[i].r;
                    let re0 = f2[i].i;
                    let t0_0 = trig[i];
                    let t1_0 = trig[n4 + i];

                    let yr0 = re0 * t0_0 + im0 * t1_0;
                    let yi0 = re0 * t1_0 - im0 * t0_0;

                    let j = n4 - 1 - i;
                    let im1 = f2[j].r;
                    let re1 = f2[j].i;
                    let t0_1 = trig[j];
                    let t1_1 = trig[n4 + j];

                    let yr1 = re1 * t0_1 + im1 * t1_1;
                    let yi1 = re1 * t1_1 - im1 * t0_1;

                    output[overlap2 + 2 * i] = yr0;
                    output[overlap2 + n2 - 1 - 2 * i] = yi0;
                    output[overlap2 + n2 - 2 - 2 * i] = yr1;
                    output[overlap2 + 2 * i + 1] = yi1;
                }
            }
        }
        #[cfg(all(
            not(any(target_arch = "x86", target_arch = "x86_64")),
            target_arch = "aarch64"
        ))]
        {
            mdct_backward_post_rotation_neon(f2, trig, output, n4, n2, overlap2);
        }
        #[cfg(all(
            not(any(target_arch = "x86", target_arch = "x86_64")),
            not(target_arch = "aarch64")
        ))]
        for i in 0..((n4 + 1) >> 1) {
            let im0 = f2[i].r;
            let re0 = f2[i].i;
            let t0_0 = trig[i];
            let t1_0 = trig[n4 + i];

            let yr0 = re0 * t0_0 + im0 * t1_0;
            let yi0 = re0 * t1_0 - im0 * t0_0;

            let j = n4 - 1 - i;
            let im1 = f2[j].r;
            let re1 = f2[j].i;
            let t0_1 = trig[j];
            let t1_1 = trig[n4 + j];

            let yr1 = re1 * t0_1 + im1 * t1_1;
            let yi1 = re1 * t1_1 - im1 * t0_1;

            output[overlap2 + 2 * i] = yr0;
            output[overlap2 + n2 - 1 - 2 * i] = yi0;
            output[overlap2 + n2 - 2 - 2 * i] = yr1;
            output[overlap2 + 2 * i + 1] = yi1;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if std::arch::is_x86_feature_detected!("avx") {
                mdct_tdac_avx(output, window, overlap);
            } else {
                for i in 0..overlap2 {
                    let x1 = output[overlap - 1 - i];
                    let x2 = output[i];
                    let wp1 = window[i];
                    let wp2 = window[overlap - 1 - i];

                    output[i] = x2 * wp2 - x1 * wp1;
                    output[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
                }
            }
        }
        #[cfg(all(
            not(any(target_arch = "x86", target_arch = "x86_64")),
            target_arch = "aarch64"
        ))]
        {
            mdct_tdac_neon(output, window, overlap);
        }
        #[cfg(all(
            not(any(target_arch = "x86", target_arch = "x86_64")),
            not(target_arch = "aarch64")
        ))]
        for i in 0..overlap2 {
            let x1 = output[overlap - 1 - i];
            let x2 = output[i];
            let wp1 = window[i];
            let wp2 = window[overlap - 1 - i];

            output[i] = x2 * wp2 - x1 * wp1;
            output[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn mdct_pre_rotation_avx(
    f: &[f32],
    f2: &mut [KissCpx],
    trig: &[f32],
    bitrev: &[i16],
    n4: usize,
    scale: f32,
) {
    for i in 0..n4 {
        let re = f[2 * i];
        let im = f[2 * i + 1];
        let t0 = trig[i];
        let t1 = trig[n4 + i];

        let yr = re * t0 - im * t1;
        let yi = im * t0 + re * t1;

        f2[bitrev[i] as usize] = KissCpx::new(yr * scale, yi * scale);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn mdct_post_rotation_avx(
    f2: &[KissCpx],
    trig: &[f32],
    output: &mut [f32],
    n4: usize,
    n2: usize,
    stride: usize,
) {
    for i in 0..n4 {
        let fp = &f2[i];
        let t0 = trig[i];
        let t1 = trig[n4 + i];

        let yr = fp.i * t1 - fp.r * t0;
        let yi = fp.r * t1 + fp.i * t0;

        output[i * 2 * stride] = yr;
        output[stride * (n2 - 1 - 2 * i)] = yi;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn mdct_backward_pre_rotation_avx(
    input: &[f32],
    f2: &mut [KissCpx],
    trig: &[f32],
    bitrev: &[i16],
    n4: usize,
    n2: usize,
    stride: usize,
) {
    for i in 0..n4 {
        let rev = bitrev[i] as usize;
        let x1 = input[2 * i * stride];
        let x2 = input[stride * (n2 - 1 - 2 * i)];
        let t0 = trig[i];
        let t1 = trig[n4 + i];

        let yr = x2 * t0 + x1 * t1;
        let yi = x1 * t0 - x2 * t1;

        f2[rev] = KissCpx::new(yi, yr);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn mdct_backward_post_rotation_avx(
    f2: &[KissCpx],
    trig: &[f32],
    output: &mut [f32],
    n4: usize,
    n2: usize,
    overlap2: usize,
) {
    for i in 0..((n4 + 1) >> 1) {
        let im0 = f2[i].r;
        let re0 = f2[i].i;
        let t0_0 = trig[i];
        let t1_0 = trig[n4 + i];

        let yr0 = re0 * t0_0 + im0 * t1_0;
        let yi0 = re0 * t1_0 - im0 * t0_0;

        let j = n4 - 1 - i;
        let im1 = f2[j].r;
        let re1 = f2[j].i;
        let t0_1 = trig[j];
        let t1_1 = trig[n4 + j];

        let yr1 = re1 * t0_1 + im1 * t1_1;
        let yi1 = re1 * t1_1 - im1 * t0_1;

        output[overlap2 + 2 * i] = yr0;
        output[overlap2 + n2 - 1 - 2 * i] = yi0;
        output[overlap2 + n2 - 2 - 2 * i] = yr1;
        output[overlap2 + 2 * i + 1] = yi1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn mdct_tdac_avx(output: &mut [f32], window: &[f32], overlap: usize) {
    use std::arch::x86_64::*;

    let overlap2 = overlap / 2;
    let mut i = 0usize;

    while i + 8 <= overlap2 {
        let x2 = _mm256_loadu_ps(output.as_ptr().add(i));

        let mut x1_tmp = [0f32; 8];
        let mut w2_tmp = [0f32; 8];
        for j in 0..8 {
            x1_tmp[j] = output[overlap - 1 - (i + j)];
            w2_tmp[j] = window[overlap - 1 - (i + j)];
        }
        let x1 = _mm256_loadu_ps(x1_tmp.as_ptr());

        let w1 = _mm256_loadu_ps(window.as_ptr().add(i));
        let w2 = _mm256_loadu_ps(w2_tmp.as_ptr());

        let out_fwd = _mm256_sub_ps(_mm256_mul_ps(x2, w2), _mm256_mul_ps(x1, w1));
        let out_rev = _mm256_add_ps(_mm256_mul_ps(x2, w1), _mm256_mul_ps(x1, w2));

        _mm256_storeu_ps(output.as_mut_ptr().add(i), out_fwd);

        let mut out_rev_tmp = [0f32; 8];
        _mm256_storeu_ps(out_rev_tmp.as_mut_ptr(), out_rev);
        for j in 0..8 {
            output[overlap - 1 - (i + j)] = out_rev_tmp[j];
        }

        i += 8;
    }

    for i in i..overlap2 {
        let x1 = output[overlap - 1 - i];
        let x2 = output[i];
        let wp1 = window[i];
        let wp2 = window[overlap - 1 - i];
        output[i] = x2 * wp2 - x1 * wp1;
        output[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn mdct_pre_rotation_neon(
    f: &[f32],
    f2: &mut [KissCpx],
    trig: &[f32],
    bitrev: &[i16],
    n4: usize,
    scale: f32,
) {
    use std::arch::aarch64::*;

    unsafe {
        let vscale = vdupq_n_f32(scale);
        let f_ptr = f.as_ptr();
        let trig_ptr = trig.as_ptr();
        let bitrev_ptr = bitrev.as_ptr();
        let f2_ptr = f2.as_mut_ptr() as *mut f32;

        let n4_vec = n4 & !3;
        let mut i = 0;

        while i < n4_vec {
            let t0 = vld1q_f32(trig_ptr.add(i));
            let t1 = vld1q_f32(trig_ptr.add(n4 + i));

            let f0 = vld1q_f32(f_ptr.add(2 * i));
            let f1 = vld1q_f32(f_ptr.add(2 * i + 4));

            let even_odd = vuzpq_f32(f0, f1);
            let re_v = even_odd.0;
            let im_v = even_odd.1;

            let yr = vsubq_f32(vmulq_f32(re_v, t0), vmulq_f32(im_v, t1));
            let yi = vaddq_f32(vmulq_f32(im_v, t0), vmulq_f32(re_v, t1));

            let yr = vmulq_f32(yr, vscale);
            let yi = vmulq_f32(yi, vscale);

            let yr_arr: [f32; 4] = std::mem::transmute(yr);
            let yi_arr: [f32; 4] = std::mem::transmute(yi);

            for j in 0..4 {
                let rev = *bitrev_ptr.add(i + j) as usize;
                *f2_ptr.add(2 * rev) = yr_arr[j];
                *f2_ptr.add(2 * rev + 1) = yi_arr[j];
            }

            i += 4;
        }

        for i in n4_vec..n4 {
            let re = *f_ptr.add(2 * i);
            let im = *f_ptr.add(2 * i + 1);
            let t0 = *trig_ptr.add(i);
            let t1 = *trig_ptr.add(n4 + i);
            let yr = re * t0 - im * t1;
            let yi = im * t0 + re * t1;
            let rev = *bitrev_ptr.add(i) as usize;
            *f2_ptr.add(2 * rev) = yr * scale;
            *f2_ptr.add(2 * rev + 1) = yi * scale;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn mdct_post_rotation_neon(
    f2: &[KissCpx],
    trig: &[f32],
    output: &mut [f32],
    n4: usize,
    n2: usize,
    stride: usize,
) {
    use std::arch::aarch64::*;

    if stride > 1 {
        for i in 0..n4 {
            let fp = &f2[i];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = fp.i * t1 - fp.r * t0;
            let yi = fp.r * t1 + fp.i * t0;
            output[i * 2 * stride] = yr;
            output[stride * (n2 - 1 - 2 * i)] = yi;
        }
        return;
    }

    unsafe {
        let f2_ptr = f2.as_ptr() as *const f32;
        let trig_ptr = trig.as_ptr();
        let out_ptr = output.as_mut_ptr();

        let n4_vec = n4 & !3;
        let mut i = 0;

        while i < n4_vec {
            let c0 = vld1q_f32(f2_ptr.add(2 * i));
            let c1 = vld1q_f32(f2_ptr.add(2 * i + 4));

            let t0 = vld1q_f32(trig_ptr.add(i));
            let t1 = vld1q_f32(trig_ptr.add(n4 + i));

            let ri = vuzpq_f32(c0, c1);
            let r_v = ri.0;
            let i_v = ri.1;

            let yr = vsubq_f32(vmulq_f32(i_v, t1), vmulq_f32(r_v, t0));

            let yi = vaddq_f32(vmulq_f32(r_v, t1), vmulq_f32(i_v, t0));

            let yr_arr: [f32; 4] = std::mem::transmute(yr);
            let yi_arr: [f32; 4] = std::mem::transmute(yi);

            for j in 0..4 {
                *out_ptr.add((i + j) * 2) = yr_arr[j];
                *out_ptr.add(n2 - 1 - 2 * (i + j)) = yi_arr[j];
            }

            i += 4;
        }

        for i in n4_vec..n4 {
            let fp = &f2[i];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = fp.i * t1 - fp.r * t0;
            let yi = fp.r * t1 + fp.i * t0;
            output[i * 2] = yr;
            output[n2 - 1 - 2 * i] = yi;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn mdct_backward_pre_rotation_neon(
    input: &[f32],
    f2: &mut [KissCpx],
    trig: &[f32],
    bitrev: &[i16],
    n4: usize,
    n2: usize,
    stride: usize,
) {
    use std::arch::aarch64::*;

    if stride != 1 {
        for i in 0..n4 {
            let rev = bitrev[i] as usize;
            let x1 = input[2 * i * stride];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            f2[rev] = KissCpx::new(yi, yr);
        }
        return;
    }

    unsafe {
        let in_ptr = input.as_ptr();
        let trig_ptr = trig.as_ptr();
        let bitrev_ptr = bitrev.as_ptr();
        let f2_ptr = f2.as_mut_ptr() as *mut f32;

        let n4_vec = n4 & !3;
        let mut i = 0;

        while i < n4_vec {
            let f0 = vld1q_f32(in_ptr.add(2 * i));
            let f1 = vld1q_f32(in_ptr.add(2 * i + 4));
            let deint_x1 = vuzpq_f32(f0, f1);
            let x1_v = deint_x1.0;

            let g0 = vld1q_f32(in_ptr.add(n2 - 7 - 2 * i));
            let g1 = vld1q_f32(in_ptr.add(n2 - 3 - 2 * i));
            let deint_x2 = vuzpq_f32(g0, g1);

            let x2_raw = deint_x2.0;
            let x2_v = vrev64q_f32(x2_raw);
            let x2_v = vextq_f32(x2_v, x2_v, 2);

            let t0 = vld1q_f32(trig_ptr.add(i));
            let t1 = vld1q_f32(trig_ptr.add(n4 + i));

            let yr = vaddq_f32(vmulq_f32(x2_v, t0), vmulq_f32(x1_v, t1));
            let yi = vsubq_f32(vmulq_f32(x1_v, t0), vmulq_f32(x2_v, t1));

            let yr_arr: [f32; 4] = std::mem::transmute(yr);
            let yi_arr: [f32; 4] = std::mem::transmute(yi);

            for j in 0..4 {
                let rev = *bitrev_ptr.add(i + j) as usize;
                *f2_ptr.add(2 * rev) = yi_arr[j];
                *f2_ptr.add(2 * rev + 1) = yr_arr[j];
            }

            i += 4;
        }

        for i in n4_vec..n4 {
            let rev = *bitrev_ptr.add(i) as usize;
            let x1 = *in_ptr.add(2 * i);
            let x2 = *in_ptr.add(n2 - 1 - 2 * i);
            let t0 = *trig_ptr.add(i);
            let t1 = *trig_ptr.add(n4 + i);
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            *f2_ptr.add(2 * rev) = yi;
            *f2_ptr.add(2 * rev + 1) = yr;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn mdct_backward_post_rotation_neon(
    f2: &[KissCpx],
    trig: &[f32],
    output: &mut [f32],
    n4: usize,
    n2: usize,
    overlap2: usize,
) {
    unsafe {
        let trig_ptr = trig.as_ptr();
        let out_base = output.as_mut_ptr().add(overlap2);

        let half = (n4 + 1) >> 1;

        let mut i = 0;
        while i + 1 < half {
            let j0 = n4 - 1 - i;
            let j1 = n4 - 1 - (i + 1);

            let re0 = f2[i].i;
            let im0 = f2[i].r;
            let t0_0 = *trig_ptr.add(i);
            let t1_0 = *trig_ptr.add(n4 + i);
            let yr0 = re0 * t0_0 + im0 * t1_0;
            let yi0 = re0 * t1_0 - im0 * t0_0;

            let im1 = f2[j0].r;
            let re1 = f2[j0].i;
            let t0_1 = *trig_ptr.add(j0);
            let t1_1 = *trig_ptr.add(n4 + j0);
            let yr1 = re1 * t0_1 + im1 * t1_1;
            let yi1 = re1 * t1_1 - im1 * t0_1;

            *out_base.add(2 * i) = yr0;
            *out_base.add(n2 - 1 - 2 * i) = yi0;
            *out_base.add(n2 - 2 - 2 * i) = yr1;
            *out_base.add(2 * i + 1) = yi1;

            let re0b = f2[i + 1].i;
            let im0b = f2[i + 1].r;
            let t0_0b = *trig_ptr.add(i + 1);
            let t1_0b = *trig_ptr.add(n4 + i + 1);
            let yr0b = re0b * t0_0b + im0b * t1_0b;
            let yi0b = re0b * t1_0b - im0b * t0_0b;

            let im1b = f2[j1].r;
            let re1b = f2[j1].i;
            let t0_1b = *trig_ptr.add(j1);
            let t1_1b = *trig_ptr.add(n4 + j1);
            let yr1b = re1b * t0_1b + im1b * t1_1b;
            let yi1b = re1b * t1_1b - im1b * t0_1b;

            *out_base.add(2 * (i + 1)) = yr0b;
            *out_base.add(n2 - 1 - 2 * (i + 1)) = yi0b;
            *out_base.add(n2 - 2 - 2 * (i + 1)) = yr1b;
            *out_base.add(2 * (i + 1) + 1) = yi1b;

            i += 2;
        }

        if i < half {
            let j = n4 - 1 - i;
            let im0 = f2[i].r;
            let re0 = f2[i].i;
            let t0_0 = *trig_ptr.add(i);
            let t1_0 = *trig_ptr.add(n4 + i);
            let yr0 = re0 * t0_0 + im0 * t1_0;
            let yi0 = re0 * t1_0 - im0 * t0_0;

            let im1 = f2[j].r;
            let re1 = f2[j].i;
            let t0_1 = *trig_ptr.add(j);
            let t1_1 = *trig_ptr.add(n4 + j);
            let yr1 = re1 * t0_1 + im1 * t1_1;
            let yi1 = re1 * t1_1 - im1 * t0_1;

            *out_base.add(2 * i) = yr0;
            *out_base.add(n2 - 1 - 2 * i) = yi0;
            *out_base.add(n2 - 2 - 2 * i) = yr1;
            *out_base.add(2 * i + 1) = yi1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn mdct_tdac_neon(output: &mut [f32], window: &[f32], overlap: usize) {
    use std::arch::aarch64::*;

    let overlap2 = overlap / 2;
    if overlap2 < 4 {
        for i in 0..overlap2 {
            let x1 = output[overlap - 1 - i];
            let x2 = output[i];
            let wp1 = window[i];
            let wp2 = window[overlap - 1 - i];
            output[i] = x2 * wp2 - x1 * wp1;
            output[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
        }
        return;
    }

    unsafe {
        let out_ptr = output.as_mut_ptr();
        let win_ptr = window.as_ptr();
        let n4 = overlap2 & !3;
        let mut i = 0;

        while i < n4 {
            let x2_fwd = vld1q_f32(out_ptr.add(i));
            let x1_rev = vld1q_f32(out_ptr.add(overlap - 4 - i));

            let x1 = vrev64q_f32(x1_rev);
            let x1 = vextq_f32(x1, x1, 2);

            let wp1_fwd = vld1q_f32(win_ptr.add(i));
            let wp2_rev = vld1q_f32(win_ptr.add(overlap - 4 - i));
            let wp2 = vrev64q_f32(wp2_rev);
            let wp2 = vextq_f32(wp2, wp2, 2);
            let wp1 = wp1_fwd;

            let out_fwd = vsubq_f32(vmulq_f32(x2_fwd, wp2), vmulq_f32(x1, wp1));

            let out_rev = vaddq_f32(vmulq_f32(x2_fwd, wp1), vmulq_f32(x1, wp2));

            let out_rev = vrev64q_f32(out_rev);
            let out_rev = vextq_f32(out_rev, out_rev, 2);

            vst1q_f32(out_ptr.add(i), out_fwd);
            vst1q_f32(out_ptr.add(overlap - 4 - i), out_rev);

            i += 4;
        }

        for i in n4..overlap2 {
            let x1 = output[overlap - 1 - i];
            let x2 = output[i];
            output[i] = x2 * window[overlap - 1 - i] - x1 * window[i];
            output[overlap - 1 - i] = x2 * window[i] + x1 * window[overlap - 1 - i];
        }
    }
}

#[cfg(test)]
mod mdct_tests {
    #[test]
    fn test_mdct_backward_transient_no_blowup() {
        let mode = crate::modes::default_mode();
        let shift = 3;
        let n = mode.mdct.n >> shift; // 120
        let overlap = mode.overlap; // 120
        let stride = 8;

        let frame_size = 960usize;
        let mut freq = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            freq[i] = ((i as f32) * 0.01).sin() * 10.0;
        }

        let out_len = n + overlap; // 240
        let mut output0 = vec![0.0f32; out_len];
        let mut output1 = vec![0.0f32; out_len];

        mode.mdct.backward(
            &freq[0..],
            &mut output0,
            mode.window,
            overlap,
            shift,
            stride,
        );
        mode.mdct.backward(
            &freq[1..],
            &mut output1,
            mode.window,
            overlap,
            shift,
            stride,
        );

        let max0 = output0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max1 = output1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("sub0 max={} sub1 max={}", max0, max1);
        eprintln!("sub0[60..70]={:?}", &output0[60..70]);
        eprintln!("sub1[60..70]={:?}", &output1[60..70]);

        assert!(max0.abs() < 500.0, "sub0 blowup: {}", max0);
        assert!(max1.abs() < 500.0, "sub1 blowup: {}", max1);
    }

    #[test]
    fn test_mdct_backward_stride1_neon_matches_scalar() {
        let mode = crate::modes::default_mode();
        let shift = 0; // non-transient full-size MDCT
        let n = mode.mdct.n >> shift; // 1920
        let n2 = n / 2; // 960
        let n4 = n / 4; // 480
        let overlap = mode.overlap; // 120
        let overlap2 = overlap / 2; // 60
        let stride = 1;

        let freq_len = n2;
        let mut freq = vec![0.0f32; freq_len + 4];
        for i in 0..freq_len {
            freq[i] = ((i as f32) * 0.01).sin() * 4577.0;
        }

        let out_len = overlap2 + n2; // 60 + 960 = 1020
        let mut output_hw = vec![0.0f32; out_len + 100];
        mode.mdct.backward(
            &freq[..],
            &mut output_hw,
            mode.window,
            overlap,
            shift,
            stride,
        );

        let st = mode.mdct.kfft[shift].as_ref().unwrap();
        let (trig, _) = mode.mdct.get_trig(shift);

        use crate::kiss_fft::KissCpx;
        let mut f2 = vec![KissCpx::new(0.0, 0.0); n4];
        for i in 0..n4 {
            let rev = st.bitrev[i] as usize;
            let x1 = freq[2 * i * stride];
            let x2 = freq[stride * (n2 - 1 - 2 * i)];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            f2[rev] = KissCpx::new(yi, yr);
        }
        crate::kiss_fft::opus_fft_impl(st, &mut f2);

        let mut output_scalar = vec![0.0f32; out_len + 100];
        for i in 0..((n4 + 1) >> 1) {
            let im0 = f2[i].r;
            let re0 = f2[i].i;
            let t0_0 = trig[i];
            let t1_0 = trig[n4 + i];
            let yr0 = re0 * t0_0 + im0 * t1_0;
            let yi0 = re0 * t1_0 - im0 * t0_0;
            let j = n4 - 1 - i;
            let im1 = f2[j].r;
            let re1 = f2[j].i;
            let t0_1 = trig[j];
            let t1_1 = trig[n4 + j];
            let yr1 = re1 * t0_1 + im1 * t1_1;
            let yi1 = re1 * t1_1 - im1 * t0_1;
            output_scalar[overlap2 + 2 * i] = yr0;
            output_scalar[overlap2 + n2 - 1 - 2 * i] = yi0;
            output_scalar[overlap2 + n2 - 2 - 2 * i] = yr1;
            output_scalar[overlap2 + 2 * i + 1] = yi1;
        }
        // TDAC
        for i in 0..overlap2 {
            let x1 = output_scalar[overlap - 1 - i];
            let x2 = output_scalar[i];
            let wp1 = mode.window[i];
            let wp2 = mode.window[overlap - 1 - i];
            output_scalar[i] = x2 * wp2 - x1 * wp1;
            output_scalar[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
        }

        let max_diff = output_hw[..out_len]
            .iter()
            .zip(output_scalar[..out_len].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.5,
            "stride=1 NEON vs scalar mismatch: max_diff={}",
            max_diff
        );
    }

    #[test]
    fn test_mdct_backward_neon_matches_scalar() {
        let mode = crate::modes::default_mode();
        let shift = 3;
        let n = mode.mdct.n >> shift; // 240
        let n2 = n / 2; // 120
        let n4 = n / 4; // 60
        let overlap = mode.overlap; // 120
        let overlap2 = overlap / 2; // 60
        let stride = 8;

        // Build a realistic freq vector (sine wave @ 440Hz)
        let frame_size = 960usize;
        let mut freq = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            freq[i] = ((i as f32) * 0.01).sin() * 200.0;
        }

        let out_len = n + overlap; // 360
        let mut output_hw = vec![0.0f32; out_len];
        mode.mdct.backward(
            &freq[0..],
            &mut output_hw,
            mode.window,
            overlap,
            shift,
            stride,
        );

        // Scalar reference
        let st = mode.mdct.kfft[shift].as_ref().unwrap();
        let (trig, _) = mode.mdct.get_trig(shift);

        use crate::kiss_fft::KissCpx;
        let mut f2 = vec![KissCpx::new(0.0, 0.0); n4];
        for i in 0..n4 {
            let rev = st.bitrev[i] as usize;
            let x1 = freq[2 * i * stride];
            let x2 = freq[stride * (n2 - 1 - 2 * i)];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            f2[rev] = KissCpx::new(yi, yr);
        }
        crate::kiss_fft::opus_fft_impl(st, &mut f2);

        let mut output_scalar = vec![0.0f32; out_len];
        for i in 0..((n4 + 1) >> 1) {
            let im0 = f2[i].r;
            let re0 = f2[i].i;
            let t0_0 = trig[i];
            let t1_0 = trig[n4 + i];
            let yr0 = re0 * t0_0 + im0 * t1_0;
            let yi0 = re0 * t1_0 - im0 * t0_0;
            let j = n4 - 1 - i;
            let im1 = f2[j].r;
            let re1 = f2[j].i;
            let t0_1 = trig[j];
            let t1_1 = trig[n4 + j];
            let yr1 = re1 * t0_1 + im1 * t1_1;
            let yi1 = re1 * t1_1 - im1 * t0_1;
            output_scalar[overlap2 + 2 * i] = yr0;
            output_scalar[overlap2 + n2 - 1 - 2 * i] = yi0;
            output_scalar[overlap2 + n2 - 2 - 2 * i] = yr1;
            output_scalar[overlap2 + 2 * i + 1] = yi1;
        }
        // TDAC
        for i in 0..overlap2 {
            let x1 = output_scalar[overlap - 1 - i];
            let x2 = output_scalar[i];
            let wp1 = mode.window[i];
            let wp2 = mode.window[overlap - 1 - i];
            output_scalar[i] = x2 * wp2 - x1 * wp1;
            output_scalar[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
        }

        for i in 0..out_len {
            let diff = (output_hw[i] - output_scalar[i]).abs();
            if diff > 1e-3 {
                eprintln!(
                    "Mismatch at output[{}]: hw={} scalar={} diff={}",
                    i, output_hw[i], output_scalar[i], diff
                );
            }
        }
        let max_diff = output_hw
            .iter()
            .zip(output_scalar.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.1,
            "NEON/HW vs scalar mismatch: max_diff={}",
            max_diff
        );
    }
}
