use crate::modes::CeltMode;
use crate::pvq::*;
use crate::range_coder::RangeCoder;
use crate::rate::{BITRES, bits2pulses, get_pulses, pulses2bits};
use crate::tell_frac_inline;

const MIN_STEREO_ENERGY: f32 = 1e-10;

pub struct BandCtx<'a> {
    pub encode: bool,
    pub m: &'a CeltMode,
    pub i: usize,
    pub band_e: &'a [f32],
    pub rc: &'a mut RangeCoder,
    pub spread: i32,
    pub remaining_bits: i32,
    pub resynth: bool,
    pub tf_change: i32,
    pub intensity: usize,
    pub theta_round: i32,
    pub avoid_split_noise: bool,
    pub arch: i32,
    pub disable_inv: bool,
    pub seed: u32,
}

#[inline]
fn bitexact_cos(x: i16) -> i16 {
    #[inline(always)]
    fn frac_mul16(a: i16, b: i16) -> i16 {
        ((16384i32 + (a as i32) * (b as i32)) >> 15) as i16
    }

    let tmp = (4096i32 + (x as i32) * (x as i32)) >> 13;
    let x2 = tmp as i16;
    let x2 = (32767 - x2 as i32
        + frac_mul16(x2, -7651 + frac_mul16(x2, 8277 + frac_mul16(-626, x2))) as i32)
        as i16;
    1 + x2
}

#[inline]
pub fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let ec_ilog = |x: u32| -> i32 {
        if x == 0 {
            0
        } else {
            32 - x.leading_zeros() as i32
        }
    };
    let lc = ec_ilog(icos.max(0) as u32);
    let ls = ec_ilog(isin.max(0) as u32);
    let icos_shifted = if lc > 0 {
        icos.max(0) << (15 - lc).max(0)
    } else {
        0
    };
    let isin_shifted = if ls > 0 {
        isin.max(0) << (15 - ls).max(0)
    } else {
        0
    };
    let fract_mul = |a: i32, b: i32| -> i32 { (a * b + 16384) >> 15 };
    (ls - lc) * (1 << 11) + fract_mul(isin_shifted, fract_mul(isin_shifted, -2597) + 7932)
        - fract_mul(icos_shifted, fract_mul(icos_shifted, -2597) + 7932)
}

#[inline(always)]
fn celt_sudiv(n: i32, d: i32) -> i32 {
    n / d
}

#[inline]
fn isqrt32(mut val: u32) -> u32 {
    let mut g = 0u32;
    let mut bshift = ((32 - val.leading_zeros()) as i32 - 1) >> 1;
    let mut b = 1u32 << bshift;
    while bshift >= 0 {
        let t = (((g << 1) + b) as u64) << bshift;
        if t <= val as u64 {
            g += b;
            val -= t as u32;
        }
        b >>= 1;
        bshift -= 1;
    }
    g
}

pub const SPREAD_NONE: i32 = 0;
pub const SPREAD_LIGHT: i32 = 1;
pub const SPREAD_NORMAL: i32 = 2;
pub const SPREAD_AGGRESSIVE: i32 = 3;

#[allow(clippy::too_many_arguments)]
pub fn spreading_decision(
    m: &CeltMode,
    x_buf: &[f32],
    average: &mut i32,
    last_decision: i32,
    hf_average: &mut i32,
    tapset_decision: &mut i32,
    update_hf: bool,
    end: usize,
    channels: usize,
    m_val: usize,
    spread_weight: &[i32],
) -> i32 {
    let mut sum = 0;
    let mut nb_bands = 0;
    let n0 = m_val * m.short_mdct_size;
    let mut hf_sum = 0;

    if m_val * (m.e_bands[end] as usize - m.e_bands[end - 1] as usize) <= 8 {
        return SPREAD_NONE;
    }

    for c in 0..channels {
        for (i, &sw) in spread_weight[..end].iter().enumerate() {
            let n = m_val * (m.e_bands[i + 1] as usize - m.e_bands[i] as usize);
            if n <= 8 {
                continue;
            }

            let mut tcount = [0; 3];
            let offset = m_val * m.e_bands[i] as usize + c * n0;
            let x = &x_buf[offset..offset + n];

            for xv in x.iter().copied() {
                let x2n = xv * xv * (n as f32);
                if x2n < 0.25 {
                    tcount[0] += 1;
                }
                if x2n < 0.0625 {
                    tcount[1] += 1;
                }
                if x2n < 0.015625 {
                    tcount[2] += 1;
                }
            }

            if i > m.nb_ebands - 4 {
                hf_sum += 32 * (tcount[1] + tcount[0]) / (n as i32);
            }

            let tmp = (if 2 * tcount[2] >= (n as i32) { 1 } else { 0 })
                + (if 2 * tcount[1] >= (n as i32) { 1 } else { 0 })
                + (if 2 * tcount[0] >= (n as i32) { 1 } else { 0 });
            sum += tmp * sw;
            nb_bands += sw;
        }
    }

    if update_hf {
        if hf_sum > 0 {
            hf_sum /= (channels as i32) * (4 - m.nb_ebands as i32 + end as i32);
        }
        *hf_average = (*hf_average + hf_sum) >> 1;
        hf_sum = *hf_average;

        if *tapset_decision == 2 {
            hf_sum += 4;
        } else if *tapset_decision == 0 {
            hf_sum -= 4;
        }

        if hf_sum > 22 {
            *tapset_decision = 2;
        } else if hf_sum > 18 {
            *tapset_decision = 1;
        } else {
            *tapset_decision = 0;
        }
    }

    if nb_bands == 0 {
        return SPREAD_NORMAL;
    }

    let mut sum_scaled = (sum << 8) / nb_bands;
    sum_scaled = (sum_scaled + *average) >> 1;
    *average = sum_scaled;

    let sum_final = (3 * sum_scaled + (((3 - last_decision) << 7) + 64) + 2) >> 2;

    if sum_final < 80 {
        SPREAD_AGGRESSIVE
    } else if sum_final < 256 {
        SPREAD_NORMAL
    } else if sum_final < 384 {
        SPREAD_LIGHT
    } else {
        SPREAD_NONE
    }
}

pub fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        haar1_neon(x, n0, stride);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        if stride == 1 && n0 >= 16 && is_x86_feature_detected!("avx") {
            haar1_avx(x, n0);
            return;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    haar1_scalar(x, n0, stride);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn haar1_avx(x: &mut [f32], n0: usize) {
    use std::arch::x86_64::*;
    let n = n0 >> 1;
    let scale = _mm256_set1_ps(std::f32::consts::FRAC_1_SQRT_2);
    let mut j = 0;
    while j + 8 <= n {
        let ptr = x.as_mut_ptr().add(2 * j);
        let a = _mm256_loadu_ps(ptr);
        let b = _mm256_loadu_ps(ptr.add(4));

        let t0 = _mm256_unpacklo_ps(a, b);
        let t1 = _mm256_unpackhi_ps(a, b);

        let even = _mm256_unpacklo_ps(t0, t1);
        let odd = _mm256_unpackhi_ps(t0, t1);

        let sum = _mm256_mul_ps(_mm256_add_ps(even, odd), scale);
        let diff = _mm256_mul_ps(_mm256_sub_ps(even, odd), scale);

        let r0 = _mm256_unpacklo_ps(sum, diff);
        let r1 = _mm256_unpackhi_ps(sum, diff);

        let out0 = _mm256_permute2f128_ps(r0, r1, 0x20);
        let out1 = _mm256_permute2f128_ps(r0, r1, 0x31);

        _mm256_storeu_ps(ptr, out0);
        _mm256_storeu_ps(ptr.add(8), out1);
        j += 8;
    }

    let scale = std::f32::consts::FRAC_1_SQRT_2;
    while j < n {
        let idx1 = 2 * j;
        let idx2 = 2 * j + 1;
        let tmp1 = scale * x[idx1];
        let tmp2 = scale * x[idx2];
        x[idx1] = tmp1 + tmp2;
        x[idx2] = tmp1 - tmp2;
        j += 1;
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn haar1_scalar(x: &mut [f32], n0: usize, stride: usize) {
    let n = n0 >> 1;
    let scale = std::f32::consts::FRAC_1_SQRT_2;
    for i in 0..stride {
        for j in 0..n {
            let idx1 = stride * 2 * j + i;
            let idx2 = stride * (2 * j + 1) + i;
            let tmp1 = scale * x[idx1];
            let tmp2 = scale * x[idx2];
            x[idx1] = tmp1 + tmp2;
            x[idx2] = tmp1 - tmp2;
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn haar1_neon(x: &mut [f32], n0: usize, stride: usize) {
    use std::arch::aarch64::*;

    let n = n0 >> 1;
    let scale = std::f32::consts::FRAC_1_SQRT_2;

    unsafe {
        let vscale = vdupq_n_f32(scale);

        for i in 0..stride {
            let mut j = 0;
            while j + 4 <= n {
                let idx_even = stride * 2 * j + i;
                let idx_odd = stride * (2 * j + 1) + i;

                let ve0 = vld1q_f32(x.as_ptr().add(idx_even));
                let ve1 = vld1q_f32(x.as_ptr().add(idx_even + stride * 2));
                let ve2 = vld1q_f32(x.as_ptr().add(idx_even + stride * 4));
                let ve3 = vld1q_f32(x.as_ptr().add(idx_even + stride * 6));

                let vo0 = vld1q_f32(x.as_ptr().add(idx_odd));
                let vo1 = vld1q_f32(x.as_ptr().add(idx_odd + stride * 2));
                let vo2 = vld1q_f32(x.as_ptr().add(idx_odd + stride * 4));
                let vo3 = vld1q_f32(x.as_ptr().add(idx_odd + stride * 6));

                let te0 = vmulq_f32(ve0, vscale);
                let te1 = vmulq_f32(ve1, vscale);
                let te2 = vmulq_f32(ve2, vscale);
                let te3 = vmulq_f32(ve3, vscale);

                let to0 = vmulq_f32(vo0, vscale);
                let to1 = vmulq_f32(vo1, vscale);
                let to2 = vmulq_f32(vo2, vscale);
                let to3 = vmulq_f32(vo3, vscale);

                vst1q_f32(x.as_mut_ptr().add(idx_even), vaddq_f32(te0, to0));
                vst1q_f32(
                    x.as_mut_ptr().add(idx_even + stride * 2),
                    vaddq_f32(te1, to1),
                );
                vst1q_f32(
                    x.as_mut_ptr().add(idx_even + stride * 4),
                    vaddq_f32(te2, to2),
                );
                vst1q_f32(
                    x.as_mut_ptr().add(idx_even + stride * 6),
                    vaddq_f32(te3, to3),
                );

                vst1q_f32(x.as_mut_ptr().add(idx_odd), vsubq_f32(te0, to0));
                vst1q_f32(
                    x.as_mut_ptr().add(idx_odd + stride * 2),
                    vsubq_f32(te1, to1),
                );
                vst1q_f32(
                    x.as_mut_ptr().add(idx_odd + stride * 4),
                    vsubq_f32(te2, to2),
                );
                vst1q_f32(
                    x.as_mut_ptr().add(idx_odd + stride * 6),
                    vsubq_f32(te3, to3),
                );

                j += 4;
            }

            while j < n {
                let idx1 = stride * 2 * j + i;
                let idx2 = stride * (2 * j + 1) + i;
                let tmp1 = scale * x[idx1];
                let tmp2 = scale * x[idx2];
                x[idx1] = tmp1 + tmp2;
                x[idx2] = tmp1 - tmp2;
                j += 1;
            }
        }
    }
}

#[inline(always)]
pub fn compute_qn(n: usize, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    static EXP2_TABLE8: [i16; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut n2 = (2 * n as i32) - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let mut qb = celt_sudiv(b + n2 * offset, n2);
    qb = qb.min(b - pulse_cap - (4 << BITRES));
    qb = qb.min(8 << BITRES);
    if qb < (1i32 << BITRES >> 1) {
        1
    } else {
        let val = EXP2_TABLE8[(qb & 0x7) as usize] as i32;
        let shift = 14 - (qb >> BITRES);
        let raw = if (0..32).contains(&shift) {
            val >> shift
        } else {
            0
        };
        let qn = (raw + 1) >> 1 << 1;
        qn.min(256)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn stereo_itheta_neon(x: &[f32], y: &[f32], stereo: bool, n: usize) -> i32 {
    use std::arch::aarch64::*;

    let mut emid = 1e-15f32;
    let mut eside = 1e-15f32;

    if stereo {
        let mut sum_mid = vdupq_n_f32(0.0);
        let mut sum_side = vdupq_n_f32(0.0);
        let mut i = 0;

        while i + 16 <= n {
            let x0 = vld1q_f32(x.as_ptr().add(i));
            let x1 = vld1q_f32(x.as_ptr().add(i + 4));
            let x2 = vld1q_f32(x.as_ptr().add(i + 8));
            let x3 = vld1q_f32(x.as_ptr().add(i + 12));
            let y0 = vld1q_f32(y.as_ptr().add(i));
            let y1 = vld1q_f32(y.as_ptr().add(i + 4));
            let y2 = vld1q_f32(y.as_ptr().add(i + 8));
            let y3 = vld1q_f32(y.as_ptr().add(i + 12));

            let m0 = vaddq_f32(x0, y0);
            let m1 = vaddq_f32(x1, y1);
            let m2 = vaddq_f32(x2, y2);
            let m3 = vaddq_f32(x3, y3);
            let s0 = vsubq_f32(x0, y0);
            let s1 = vsubq_f32(x1, y1);
            let s2 = vsubq_f32(x2, y2);
            let s3 = vsubq_f32(x3, y3);

            sum_mid = vfmaq_f32(sum_mid, m0, m0);
            sum_mid = vfmaq_f32(sum_mid, m1, m1);
            sum_mid = vfmaq_f32(sum_mid, m2, m2);
            sum_mid = vfmaq_f32(sum_mid, m3, m3);
            sum_side = vfmaq_f32(sum_side, s0, s0);
            sum_side = vfmaq_f32(sum_side, s1, s1);
            sum_side = vfmaq_f32(sum_side, s2, s2);
            sum_side = vfmaq_f32(sum_side, s3, s3);

            i += 16;
        }

        while i + 8 <= n {
            let x0 = vld1q_f32(x.as_ptr().add(i));
            let x1 = vld1q_f32(x.as_ptr().add(i + 4));
            let y0 = vld1q_f32(y.as_ptr().add(i));
            let y1 = vld1q_f32(y.as_ptr().add(i + 4));

            let m0 = vaddq_f32(x0, y0);
            let m1 = vaddq_f32(x1, y1);
            let s0 = vsubq_f32(x0, y0);
            let s1 = vsubq_f32(x1, y1);

            sum_mid = vfmaq_f32(sum_mid, m0, m0);
            sum_mid = vfmaq_f32(sum_mid, m1, m1);
            sum_side = vfmaq_f32(sum_side, s0, s0);
            sum_side = vfmaq_f32(sum_side, s1, s1);

            i += 8;
        }

        while i + 4 <= n {
            let x0 = vld1q_f32(x.as_ptr().add(i));
            let y0 = vld1q_f32(y.as_ptr().add(i));
            let m0 = vaddq_f32(x0, y0);
            let s0 = vsubq_f32(x0, y0);
            sum_mid = vfmaq_f32(sum_mid, m0, m0);
            sum_side = vfmaq_f32(sum_side, s0, s0);
            i += 4;
        }

        emid += vaddvq_f32(sum_mid);
        eside += vaddvq_f32(sum_side);

        for j in i..n {
            let m = x[j] + y[j];
            let s = x[j] - y[j];
            emid += m * m;
            eside += s * s;
        }
    } else {
        let mut sum_mid = vdupq_n_f32(0.0);
        let mut sum_side = vdupq_n_f32(0.0);
        let mut i = 0;

        while i + 16 <= n {
            let x0 = vld1q_f32(x.as_ptr().add(i));
            let x1 = vld1q_f32(x.as_ptr().add(i + 4));
            let x2 = vld1q_f32(x.as_ptr().add(i + 8));
            let x3 = vld1q_f32(x.as_ptr().add(i + 12));
            let y0 = vld1q_f32(y.as_ptr().add(i));
            let y1 = vld1q_f32(y.as_ptr().add(i + 4));
            let y2 = vld1q_f32(y.as_ptr().add(i + 8));
            let y3 = vld1q_f32(y.as_ptr().add(i + 12));

            sum_mid = vfmaq_f32(sum_mid, x0, x0);
            sum_mid = vfmaq_f32(sum_mid, x1, x1);
            sum_mid = vfmaq_f32(sum_mid, x2, x2);
            sum_mid = vfmaq_f32(sum_mid, x3, x3);
            sum_side = vfmaq_f32(sum_side, y0, y0);
            sum_side = vfmaq_f32(sum_side, y1, y1);
            sum_side = vfmaq_f32(sum_side, y2, y2);
            sum_side = vfmaq_f32(sum_side, y3, y3);

            i += 16;
        }

        while i + 8 <= n {
            let x0 = vld1q_f32(x.as_ptr().add(i));
            let x1 = vld1q_f32(x.as_ptr().add(i + 4));
            let y0 = vld1q_f32(y.as_ptr().add(i));
            let y1 = vld1q_f32(y.as_ptr().add(i + 4));

            sum_mid = vfmaq_f32(sum_mid, x0, x0);
            sum_mid = vfmaq_f32(sum_mid, x1, x1);
            sum_side = vfmaq_f32(sum_side, y0, y0);
            sum_side = vfmaq_f32(sum_side, y1, y1);

            i += 8;
        }

        while i + 4 <= n {
            let x0 = vld1q_f32(x.as_ptr().add(i));
            let y0 = vld1q_f32(y.as_ptr().add(i));
            sum_mid = vfmaq_f32(sum_mid, x0, x0);
            sum_side = vfmaq_f32(sum_side, y0, y0);
            i += 4;
        }

        emid += vaddvq_f32(sum_mid);
        eside += vaddvq_f32(sum_side);

        for j in i..n {
            emid += x[j] * x[j];
            eside += y[j] * y[j];
        }
    }

    let mid = emid.sqrt();
    let side = eside.sqrt();
    let theta_norm = celt_atan2p_norm(side, mid);
    (0.5 + 16384.0 * theta_norm) as i32
}

#[inline(always)]
#[cfg(target_arch = "aarch64")]
pub fn stereo_itheta(x: &[f32], y: &[f32], stereo: bool, n: usize) -> i32 {
    unsafe { stereo_itheta_neon(x, y, stereo, n) }
}

#[inline(always)]
#[cfg(not(target_arch = "aarch64"))]
pub fn stereo_itheta(x: &[f32], y: &[f32], stereo: bool, n: usize) -> i32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return stereo_itheta_neon(x, y, stereo, n);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut emid = 1e-15f32;
        let mut eside = 1e-15f32;
        if stereo {
            for i in 0..n {
                let m = x[i] + y[i];
                let s = x[i] - y[i];
                emid += m * m;
                eside += s * s;
            }
        } else {
            for i in 0..n {
                emid += x[i] * x[i];
                eside += y[i] * y[i];
            }
        }
        let mid = emid.sqrt();
        let side = eside.sqrt();
        let theta_norm = celt_atan2p_norm(side, mid);
        (0.5 + 16384.0 * theta_norm) as i32
    }
}

#[inline(always)]
fn celt_atan2p_norm(y: f32, x: f32) -> f32 {
    #[inline(always)]
    fn atan_norm(x: f32) -> f32 {
        const ATAN2_2_OVER_PI: f32 = std::f32::consts::FRAC_2_PI;
        const A03: f32 = -3.333_166e-1_f32;
        const A05: f32 = 1.996_270_4e-1_f32;
        const A07: f32 = -1.397_658_3e-1_f32;
        const A09: f32 = 9.794_234_e-2_f32;
        const A11: f32 = -5.777_359_e-2_f32;
        const A13: f32 = 2.304_014e-2_f32;
        const A15: f32 = -4.355_406e-3_f32;
        let x2 = x * x;
        ATAN2_2_OVER_PI
            * x
            * (1.0
                + x2 * (A03
                    + x2 * (A05 + x2 * (A07 + x2 * (A09 + x2 * (A11 + x2 * (A13 + x2 * A15)))))))
    }
    if x * x + y * y < 1e-18 {
        return 0.0;
    }
    if y < x {
        atan_norm(y / x)
    } else {
        1.0 - atan_norm(x / y)
    }
}

pub struct SplitCtx {
    pub inv: bool,
    pub imid: i32,
    pub iside: i32,
    pub delta: i32,
    pub itheta: i32,
    pub qalloc: i32,
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn compute_theta_encode(
    ctx: &mut BandCtx,
    sctx: &mut SplitCtx,
    x: &[f32],
    y: &[f32],
    n: usize,
    b: &mut i32,
    b_blocks: i32,
    b0: i32,
    lm: i32,
    stereo: bool,
    fill: &mut u32,
) {
    let pulse_cap = ctx.m.log_n[ctx.i] as i32 + (lm << BITRES);
    let offset = (pulse_cap >> 1) - if stereo && n == 2 { 16 } else { 4 };
    let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);

    if stereo && ctx.i >= ctx.intensity {
        qn = 1;
    }

    if qn == 1 && !(stereo && ctx.i >= ctx.intensity) {
        sctx.itheta = 8192;
        sctx.qalloc = 0;
        let imid = bitexact_cos(8192i16);
        sctx.imid = imid as i32;
        let iside = bitexact_cos(8192i16);
        sctx.iside = iside as i32;
        sctx.delta =
            (((n as i32 - 1) << 7) * bitexact_log2tan(sctx.iside, sctx.imid) + 16384) >> 15;
        return;
    }

    let mut itheta = stereo_itheta(x, y, stereo, n);

    let tell_start = tell_frac_inline!(ctx.rc);

    if qn != 1 {
        if !stereo || ctx.theta_round == 0 {
            itheta = (itheta * qn + 8192) >> 14;
            if !stereo && ctx.avoid_split_noise && itheta > 0 && itheta < qn {
                let unquantized = (itheta * 16384) / qn;
                let imid = bitexact_cos(unquantized as i16) as i32;
                let iside = bitexact_cos((16384 - unquantized) as i16) as i32;
                let delta = (((n as i32 - 1) << 7) * bitexact_log2tan(iside, imid) + 16384) >> 15;
                if delta > *b {
                    itheta = qn;
                } else if delta < -*b {
                    itheta = 0;
                }
            }
        } else {
            let bias = if itheta > 8192 {
                32767 / qn
            } else {
                -32767 / qn
            };
            let down = (itheta * qn + bias) >> 14;
            let down = down.clamp(0, qn - 1);
            if ctx.theta_round < 0 {
                itheta = down;
            } else {
                itheta = down + 1;
            }
        }

        if stereo && n > 2 {
            let p0 = 3;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            let fl = if itheta <= x0 {
                p0 * itheta
            } else {
                (itheta - 1 - x0) + (x0 + 1) * p0
            };
            let fh = if itheta <= x0 {
                p0 * (itheta + 1)
            } else {
                (itheta - x0) + (x0 + 1) * p0
            };
            ctx.rc.encode(fl as u32, fh as u32, ft as u32);
        } else if b0 > 1 || stereo {
            ctx.rc.enc_uint(itheta as u32, (qn + 1) as u32);
        } else {
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            let fs = if itheta <= (qn >> 1) {
                itheta + 1
            } else {
                qn + 1 - itheta
            };
            let fl = if itheta <= (qn >> 1) {
                (itheta * (itheta + 1)) >> 1
            } else {
                ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1)
            };
            ctx.rc.encode(fl as u32, (fl + fs) as u32, ft as u32);
        }
        itheta = (itheta as u32 * 16384 / qn as u32) as i32;
    } else if stereo && ctx.i >= ctx.intensity {
        let mut emid = 1e-15f32;
        let mut eside = 1e-15f32;
        for i in 0..n {
            let m = x[i] + y[i];
            let s = x[i] - y[i];
            emid += m * m;
            eside += s * s;
        }
        let inv = eside > emid;
        ctx.rc.encode_bit_logp(inv, 1);
        itheta = 0;
        sctx.inv = inv;
    } else {
        itheta = 8192;
    }

    sctx.itheta = itheta;

    sctx.qalloc = if qn == 1 && !(stereo && ctx.i >= ctx.intensity) {
        0
    } else {
        tell_frac_inline!(ctx.rc) - tell_start
    };
    *b -= sctx.qalloc; // matches C: *b -= qalloc

    if itheta == 0 {
        sctx.imid = 32767;
        sctx.iside = 0;
        sctx.delta = -16384;
        *fill &= (1 << b_blocks) - 1;
    } else if itheta == 16384 {
        sctx.imid = 0;
        sctx.iside = 32767;
        sctx.delta = 16384;
        *fill &= !((1 << b_blocks) - 1);
    } else {
        let imid = bitexact_cos(itheta as i16);
        sctx.imid = imid as i32;
        let iside = bitexact_cos((16384 - itheta) as i16);
        sctx.iside = iside as i32;
        sctx.delta =
            (((n as i32 - 1) << 7) * bitexact_log2tan(sctx.iside, sctx.imid) + 16384) >> 15;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn compute_theta(
    ctx: &mut BandCtx,
    sctx: &mut SplitCtx,
    x: &[f32],
    y: &[f32],
    n: usize,
    b: &mut i32,
    b_blocks: i32,
    b0: i32,
    lm: i32,
    stereo: bool,
    fill: &mut u32,
) {
    let pulse_cap = ctx.m.log_n[ctx.i] as i32 + (lm << BITRES);
    let offset = (pulse_cap >> 1) - if stereo && n == 2 { 16 } else { 4 };
    let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);

    if stereo && ctx.i >= ctx.intensity {
        qn = 1;
    }

    let mut itheta = 0;
    if ctx.encode {
        itheta = stereo_itheta(x, y, stereo, n);
    }

    let tell_start = tell_frac_inline!(ctx.rc);

    if qn != 1 {
        if ctx.encode {
            if !stereo || ctx.theta_round == 0 {
                itheta = (itheta * qn + 8192) >> 14;
                if !stereo && ctx.avoid_split_noise && itheta > 0 && itheta < qn {
                    let unquantized = (itheta * 16384) / qn;
                    let imid = bitexact_cos(unquantized as i16) as i32;
                    let iside = bitexact_cos((16384 - unquantized) as i16) as i32;
                    let delta =
                        (((n as i32 - 1) << 7) * bitexact_log2tan(iside, imid) + 16384) >> 15;
                    if delta > *b {
                        itheta = qn;
                    } else if delta < -*b {
                        itheta = 0;
                    }
                }
            } else {
                let bias = if itheta > 8192 {
                    32767 / qn
                } else {
                    -32767 / qn
                };
                let down = (itheta * qn + bias) >> 14;
                let down = down.clamp(0, qn - 1);
                if ctx.theta_round < 0 {
                    itheta = down;
                } else {
                    itheta = down + 1;
                }
            }
        }

        if stereo && n > 2 {
            let p0 = 3;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            if ctx.encode {
                let fl = if itheta <= x0 {
                    p0 * itheta
                } else {
                    (itheta - 1 - x0) + (x0 + 1) * p0
                };
                let fh = if itheta <= x0 {
                    p0 * (itheta + 1)
                } else {
                    (itheta - x0) + (x0 + 1) * p0
                };
                ctx.rc.encode(fl as u32, fh as u32, ft as u32);
            } else {
                let fs = ctx.rc.decode(ft as u32);
                if fs < (x0 + 1) as u32 * p0 as u32 {
                    itheta = fs as i32 / p0;
                } else {
                    itheta = (x0 + 1) + (fs as i32 - (x0 + 1) * p0);
                }
                let fl = if itheta <= x0 {
                    p0 * itheta
                } else {
                    (itheta - 1 - x0) + (x0 + 1) * p0
                };
                let fh = if itheta <= x0 {
                    p0 * (itheta + 1)
                } else {
                    (itheta - x0) + (x0 + 1) * p0
                };
                ctx.rc.update(fl as u32, fh as u32, ft as u32);
            }
        } else if b0 > 1 || stereo {
            if ctx.encode {
                ctx.rc.enc_uint(itheta as u32, (qn + 1) as u32);
            } else {
                itheta = ctx.rc.dec_uint((qn + 1) as u32) as i32;
            }
        } else {
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            if ctx.encode {
                let fs = if itheta <= (qn >> 1) {
                    itheta + 1
                } else {
                    qn + 1 - itheta
                };
                let fl = if itheta <= (qn >> 1) {
                    (itheta * (itheta + 1)) >> 1
                } else {
                    ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1)
                };
                ctx.rc.encode(fl as u32, (fl + fs) as u32, ft as u32);
            } else {
                let fm = ctx.rc.decode(ft as u32) as i32;
                if fm < (((qn >> 1) * ((qn >> 1) + 1)) >> 1) {
                    itheta = (isqrt32((8 * fm + 1) as u32) as i32 - 1) >> 1;
                    let fl = (itheta * (itheta + 1)) >> 1;
                    let fs = itheta + 1;
                    ctx.rc.update(fl as u32, (fl + fs) as u32, ft as u32);
                } else {
                    itheta = (2 * (qn + 1) - isqrt32((8 * (ft - fm - 1) + 1) as u32) as i32) >> 1;
                    let fs = qn + 1 - itheta;
                    let fl = ft - (((qn + 1 - itheta) * (qn + 2 - itheta)) >> 1);
                    ctx.rc.update(fl as u32, (fl + fs) as u32, ft as u32);
                }
            }
        }
        itheta = (itheta as u32 * 16384 / qn as u32) as i32;
    } else if stereo && ctx.i >= ctx.intensity {
        if ctx.encode {
            let mut emid = 1e-15f32;
            let mut eside = 1e-15f32;
            for i in 0..n {
                let m = x[i] + y[i];
                let s = x[i] - y[i];
                emid += m * m;
                eside += s * s;
            }
            let inv = eside > emid;
            ctx.rc.encode_bit_logp(inv, 1);
            itheta = 0;
            sctx.inv = inv;
        } else {
            sctx.inv = ctx.rc.decode_bit_logp(1);
            itheta = 0;
        }
    } else {
        itheta = 8192;
    }

    sctx.itheta = itheta;

    sctx.qalloc = if qn == 1 && !(stereo && ctx.i >= ctx.intensity) {
        0
    } else {
        tell_frac_inline!(ctx.rc) - tell_start
    };
    *b -= sctx.qalloc; // matches C: *b -= qalloc

    if itheta == 0 {
        sctx.imid = 32767;
        sctx.iside = 0;
        sctx.delta = -16384;
        *fill &= (1 << b_blocks) - 1;
    } else if itheta == 16384 {
        sctx.imid = 0;
        sctx.iside = 32767;
        sctx.delta = 16384;
        *fill &= !((1 << b_blocks) - 1);
    } else {
        let imid = bitexact_cos(itheta as i16);
        sctx.imid = imid as i32;
        let iside = bitexact_cos((16384 - itheta) as i16);
        sctx.iside = iside as i32;
        sctx.delta =
            (((n as i32 - 1) << 7) * bitexact_log2tan(sctx.iside, sctx.imid) + 16384) >> 15;
    }
}

#[inline(always)]
fn quant_partition_n2_encode(
    ctx: &mut BandCtx,
    x: &mut [f32],
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    gain: f32,
    fill: u32,
) -> u32 {
    let mut q = bits2pulses(ctx.m, ctx.i, lm, b);
    let mut curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
    ctx.remaining_bits -= curr_bits;

    while ctx.remaining_bits < 0 && q > 0 {
        ctx.remaining_bits += curr_bits;
        q -= 1;
        curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
        ctx.remaining_bits -= curr_bits;
    }

    if q != 0 {
        let k = get_pulses(q);
        alg_quant(x, 2, k, ctx.spread, b_blocks as usize, ctx.rc, gain, false)
    } else {
        let has_lowband = lowband.is_some();
        if has_lowband {
            fill
        } else {
            (1u32 << b_blocks) - 1
        }
    }
}

#[inline(always)]
fn quant_partition_n4_encode(
    ctx: &mut BandCtx,
    x: &mut [f32],
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    gain: f32,
    fill: u32,
) -> u32 {
    let mut q = bits2pulses(ctx.m, ctx.i, lm, b);
    let mut curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
    ctx.remaining_bits -= curr_bits;

    while ctx.remaining_bits < 0 && q > 0 {
        ctx.remaining_bits += curr_bits;
        q -= 1;
        curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
        ctx.remaining_bits -= curr_bits;
    }

    if q != 0 {
        let k = get_pulses(q);
        alg_quant(x, 4, k, ctx.spread, b_blocks as usize, ctx.rc, gain, false)
    } else {
        let has_lowband = lowband.is_some();
        if has_lowband {
            fill
        } else {
            (1u32 << b_blocks) - 1
        }
    }
}

#[inline(always)]
fn quant_partition_n8_encode(
    ctx: &mut BandCtx,
    x: &mut [f32],
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    gain: f32,
    fill: u32,
) -> u32 {
    let mut q = bits2pulses(ctx.m, ctx.i, lm, b);
    let mut curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
    ctx.remaining_bits -= curr_bits;

    while ctx.remaining_bits < 0 && q > 0 {
        ctx.remaining_bits += curr_bits;
        q -= 1;
        curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
        ctx.remaining_bits -= curr_bits;
    }

    if q != 0 {
        let k = get_pulses(q);
        alg_quant(x, 8, k, ctx.spread, b_blocks as usize, ctx.rc, gain, false)
    } else {
        let has_lowband = lowband.is_some();
        if has_lowband {
            fill
        } else {
            (1u32 << b_blocks) - 1
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn quant_partition_direct_encode(
    ctx: &mut BandCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    gain: f32,
    fill: u32,
) -> u32 {
    let mut q = bits2pulses(ctx.m, ctx.i, lm, b);
    let mut curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
    ctx.remaining_bits -= curr_bits;

    while ctx.remaining_bits < 0 && q > 0 {
        ctx.remaining_bits += curr_bits;
        q -= 1;
        curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
        ctx.remaining_bits -= curr_bits;
    }

    if q != 0 {
        let k = get_pulses(q);
        alg_quant(x, n, k, ctx.spread, b_blocks as usize, ctx.rc, gain, false)
    } else {
        let has_lowband = lowband.is_some();
        if has_lowband {
            fill
        } else {
            (1u32 << b_blocks) - 1
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn quant_partition_encode(
    ctx: &mut BandCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    gain: f32,
    fill: u32,
) -> u32 {
    // N==2 can never split (should_split requires n>2), dispatch immediately
    if n == 2 {
        return quant_partition_n2_encode(ctx, x, b, b_blocks, lowband, lm, gain, fill);
    }

    // Check split condition FIRST (matching C's quant_partition which checks this before dispatch)
    let should_split = if lm >= 0 && n > 2 {
        let cache_idx = (lm + 1) as usize * ctx.m.nb_ebands + ctx.i;
        let cache_base = unsafe { *ctx.m.cache.index.get_unchecked(cache_idx) } as usize;
        if cache_base > 0 {
            let cache_ptr = ctx.m.cache.bits.as_ptr().wrapping_add(cache_base);
            let max_q = unsafe { *cache_ptr } as usize;
            b > (unsafe { *cache_ptr.add(max_q) } as i32) + 12
        } else {
            false
        }
    } else {
        false
    };

    if should_split {
        let mut sctx = SplitCtx {
            inv: false,
            imid: 0,
            iside: 0,
            delta: 0,
            itheta: 0,
            qalloc: 0,
        };
        let mut b_mut = b;
        let mut fill_mut = fill;
        let mid = n / 2;
        let lm = lm - 1;
        let b0 = b_blocks;
        if b_blocks == 1 {
            fill_mut = (fill_mut & 1) | (fill_mut << 1);
        }
        let b_blocks = (b_blocks + 1) >> 1;
        let (x_mid, x_side) = x.split_at_mut(mid);

        compute_theta_encode(
            ctx,
            &mut sctx,
            x_mid,
            x_side,
            mid,
            &mut b_mut,
            b_blocks,
            b0,
            lm,
            false,
            &mut fill_mut,
        );

        ctx.remaining_bits -= sctx.qalloc;
        let mut delta = sctx.delta;
        /* Give more bits to low-energy MDCTs than they would otherwise deserve */
        if b0 > 1 && (sctx.itheta & 0x3fff) != 0 {
            if sctx.itheta > 8192 {
                delta -= delta >> (4 - lm);
            } else {
                delta = 0.min(delta + ((mid as i32) << BITRES >> (5 - lm)));
            }
        }
        let mbits = (0).max((b_mut - delta) / 2).min(b_mut);
        let mut sbits = b_mut - mbits;
        let mut mbits = mbits;

        let mut rebalance = ctx.remaining_bits;
        let mut cm;

        if mbits >= sbits {
            cm = quant_partition_encode(
                ctx, x_mid, mid, mbits, b_blocks, lowband, lm, gain, fill_mut,
            );
            rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > (3 << 3) && sctx.itheta != 0 {
                sbits += rebalance - (3 << 3);
            }
            cm |= quant_partition_encode(
                ctx,
                x_side,
                mid,
                sbits,
                b_blocks,
                None,
                lm,
                gain,
                fill_mut >> b_blocks,
            ) << (b0 >> 1);
        } else {
            cm = quant_partition_encode(
                ctx,
                x_side,
                mid,
                sbits,
                b_blocks,
                None,
                lm,
                gain,
                fill_mut >> b_blocks,
            ) << (b0 >> 1);
            rebalance = sbits - (rebalance - ctx.remaining_bits);
            if rebalance > (3 << 3) && sctx.itheta != 16384 {
                mbits += rebalance - (3 << 3);
            }
            cm |= quant_partition_encode(
                ctx, x_mid, mid, mbits, b_blocks, lowband, lm, gain, fill_mut,
            );
        }
        cm
    } else {
        // No split — dispatch to small-N specialized encoders or direct path
        if n == 4 {
            return quant_partition_n4_encode(ctx, x, b, b_blocks, lowband, lm, gain, fill);
        }
        if n == 8 {
            return quant_partition_n8_encode(ctx, x, b, b_blocks, lowband, lm, gain, fill);
        }
        if n == 16 {
            return quant_partition_direct_encode(ctx, x, n, b, b_blocks, lowband, lm, gain, fill);
        }
        let mut q = bits2pulses(ctx.m, ctx.i, lm, b);
        let mut curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
        ctx.remaining_bits -= curr_bits;

        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
            ctx.remaining_bits -= curr_bits;
        }

        if q != 0 {
            let k = get_pulses(q);
            alg_quant(x, n, k, ctx.spread, b_blocks as usize, ctx.rc, gain, false)
        } else if lowband.is_some() {
            fill
        } else {
            (1 << b_blocks) - 1
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn quant_partition(
    ctx: &mut BandCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    gain: f32,
    fill: u32,
) -> u32 {
    /* Check split condition FIRST, before dispatching to specialized handlers.
    This matches the C code which checks this at the top of quant_partition. */
    let should_split = if lm >= 0 && n > 2 {
        let cache_idx = (lm + 1) as usize * ctx.m.nb_ebands + ctx.i;
        let cache_base = unsafe { *ctx.m.cache.index.get_unchecked(cache_idx) } as usize;
        if cache_base > 0 {
            let cache_ptr = ctx.m.cache.bits.as_ptr().wrapping_add(cache_base);
            let max_q = unsafe { *cache_ptr } as usize;
            b > (unsafe { *cache_ptr.add(max_q) } as i32) + 12
        } else {
            false
        }
    } else {
        false
    };
    if should_split {
        let mut sctx = SplitCtx {
            inv: false,
            imid: 0,
            iside: 0,
            delta: 0,
            itheta: 0,
            qalloc: 0,
        };
        let mut b_mut = b;
        let mut fill_mut = fill;
        let mid = n / 2;
        let lm = lm - 1;
        let b0 = b_blocks; // Save original B0
        if b_blocks == 1 {
            fill_mut = (fill_mut & 1) | (fill_mut << 1);
        }
        let b_blocks = (b_blocks + 1) >> 1;
        let (x_mid, x_side) = x.split_at_mut(mid);

        compute_theta(
            ctx,
            &mut sctx,
            x_mid,
            x_side,
            mid,
            &mut b_mut,
            b_blocks,
            b0,
            lm,
            false,
            &mut fill_mut,
        );

        ctx.remaining_bits -= sctx.qalloc;
        let mut delta = sctx.delta;
        /* Give more bits to low-energy MDCTs than they would otherwise deserve
        (matches C quant_partition's B0>1 adjustment) */
        if b0 > 1 && (sctx.itheta & 0x3fff) != 0 {
            if sctx.itheta > 8192 {
                delta -= delta >> (4 - lm);
            } else {
                delta = 0.min(delta + ((mid as i32) << BITRES >> (5 - lm)));
            }
        }
        let mbits = (0).max((b_mut - delta) / 2).min(b_mut);
        let mut sbits = b_mut - mbits;
        let mut mbits = mbits;

        let mut rebalance = ctx.remaining_bits;
        let mut cm;

        if mbits >= sbits {
            cm = quant_partition(
                ctx,
                x_mid,
                mid,
                mbits,
                b_blocks,
                lowband,
                lm,
                gain * (sctx.imid as f32 / 32768.0),
                fill_mut,
            );
            rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > (3 << 3) && sctx.itheta != 0 {
                sbits += rebalance - (3 << 3);
            }
            cm |= quant_partition(
                ctx,
                x_side,
                mid,
                sbits,
                b_blocks,
                None,
                lm,
                gain * (sctx.iside as f32 / 32768.0),
                fill_mut >> b_blocks,
            ) << (b0 >> 1);
        } else {
            cm = quant_partition(
                ctx,
                x_side,
                mid,
                sbits,
                b_blocks,
                None,
                lm,
                gain * (sctx.iside as f32 / 32768.0),
                fill_mut >> b_blocks,
            ) << (b0 >> 1);
            rebalance = sbits - (rebalance - ctx.remaining_bits);
            if rebalance > (3 << 3) && sctx.itheta != 16384 {
                mbits += rebalance - (3 << 3);
            }
            cm |= quant_partition(
                ctx,
                x_mid,
                mid,
                mbits,
                b_blocks,
                lowband,
                lm,
                gain * (sctx.imid as f32 / 32768.0),
                fill_mut,
            );
        }
        cm
    } else {
        let mut q = bits2pulses(ctx.m, ctx.i, lm, b);
        let mut curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
        ctx.remaining_bits -= curr_bits;

        while ctx.remaining_bits < 0 && q > 0 {
            ctx.remaining_bits += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
            ctx.remaining_bits -= curr_bits;
        }

        if q != 0 {
            let k = get_pulses(q);
            if ctx.encode {
                alg_quant(
                    x,
                    n,
                    k,
                    ctx.spread,
                    b_blocks as usize,
                    ctx.rc,
                    gain,
                    ctx.resynth,
                )
            } else {
                alg_unquant(x, n, k, ctx.spread, b_blocks as usize, ctx.rc, gain)
            }
        } else {
            let has_lowband = lowband.is_some();
            if ctx.resynth {
                let cm_mask = (1u32 << b_blocks) - 1;
                let fill_masked = fill & cm_mask;
                if fill_masked == 0 {
                    x[..n].fill(0.0);
                } else if has_lowband {
                    let lb = lowband.unwrap();
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        use std::arch::aarch64::*;
                        let n8 = n & !7;
                        let mut i = 0;
                        while i < n8 {
                            let mut vals = [0.0f32; 8];
                            for j in 0..8 {
                                ctx.seed = celt_lcg_rand(ctx.seed);
                                vals[j] = if ctx.seed & 0x8000 != 0 {
                                    1.0 / 256.0
                                } else {
                                    -1.0 / 256.0
                                };
                            }
                            let vnoise = vld1q_f32(vals.as_ptr());
                            let vnoise1 = vld1q_f32(vals.as_ptr().add(4));
                            let vlb = vld1q_f32(lb.as_ptr().add(i));
                            let vlb1 = vld1q_f32(lb.as_ptr().add(i + 4));
                            let vres = vaddq_f32(vlb, vnoise);
                            let vres1 = vaddq_f32(vlb1, vnoise1);
                            vst1q_f32(x.as_mut_ptr().add(i), vres);
                            vst1q_f32(x.as_mut_ptr().add(i + 4), vres1);
                            i += 8;
                        }
                        for j in i..n {
                            ctx.seed = celt_lcg_rand(ctx.seed);
                            x[j] = lb[j]
                                + if ctx.seed & 0x8000 != 0 {
                                    1.0 / 256.0
                                } else {
                                    -1.0 / 256.0
                                };
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        for j in 0..n {
                            ctx.seed = celt_lcg_rand(ctx.seed);
                            x[j] = lb[j]
                                + if ctx.seed & 0x8000 != 0 {
                                    1.0 / 256.0
                                } else {
                                    -1.0 / 256.0
                                };
                        }
                    }
                    renormalise_vector(x, n, gain);
                } else {
                    for xv in x[..n].iter_mut() {
                        ctx.seed = celt_lcg_rand(ctx.seed);
                        *xv = ((ctx.seed as i32 >> 20) as f32) / 16384.0;
                    }
                    renormalise_vector(x, n, gain);
                }
            }
            if has_lowband {
                fill
            } else {
                (1 << b_blocks) - 1
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn deinterleave_hadamard_neon(x: &mut [f32], n0: usize, stride: usize) {
    let n = n0 * stride;
    let mut tmp_buf = [std::mem::MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    let tmp = std::slice::from_raw_parts_mut(tmp_buf.as_mut_ptr() as *mut f32, n);

    for i in 0..stride {
        let src_offset = i;
        let dst_offset = i * n0;
        for j in 0..n0 {
            tmp[dst_offset + j] = x[j * stride + src_offset];
        }
    }

    x[..n].copy_from_slice(tmp);
}

pub fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;

    let mut tmp_buf = [std::mem::MaybeUninit::<f32>::uninit(); MAX_PVQ_N];

    let tmp = unsafe { std::slice::from_raw_parts_mut(tmp_buf.as_mut_ptr() as *mut f32, n) };
    if hadamard {
        let offset = match stride {
            2 => 0,
            4 => 2,
            8 => 6,
            16 => 14,
            _ => 0,
        };
        let ordery = &ORDERY_TABLE[offset..offset + stride];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[ordery[i] as usize * n0 + j] = x[j * stride + i];
            }
        }
    } else {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            if n0 >= 4 {
                deinterleave_hadamard_neon(x, n0, stride);
                return;
            }
        }
        for i in 0..stride {
            for j in 0..n0 {
                tmp[i * n0 + j] = x[j * stride + i];
            }
        }
    }
    x[..n].copy_from_slice(tmp);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn interleave_hadamard_neon(x: &mut [f32], n0: usize, stride: usize) {
    let n = n0 * stride;
    let mut tmp_buf = [std::mem::MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    let tmp = std::slice::from_raw_parts_mut(tmp_buf.as_mut_ptr() as *mut f32, n);

    for i in 0..stride {
        let src_offset = i * n0;
        let dst_offset = i;
        for j in 0..n0 {
            tmp[j * stride + dst_offset] = x[src_offset + j];
        }
    }

    x[..n].copy_from_slice(tmp);
}

pub fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp_buf = [std::mem::MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    let tmp = unsafe { std::slice::from_raw_parts_mut(tmp_buf.as_mut_ptr() as *mut f32, n) };
    if hadamard {
        let offset = match stride {
            2 => 0,
            4 => 2,
            8 => 6,
            16 => 14,
            _ => 0,
        };
        let ordery = &ORDERY_TABLE[offset..offset + stride];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[ordery[i] as usize * n0 + j];
            }
        }
    } else {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            if n0 >= 4 {
                interleave_hadamard_neon(x, n0, stride);
                return;
            }
        }
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[i * n0 + j];
            }
        }
    }
    x[..n].copy_from_slice(tmp);
}

const ORDERY_TABLE: [i32; 30] = [
    1, 0, 3, 0, 2, 1, 7, 0, 4, 3, 6, 1, 5, 2, 15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

fn quant_band_n1(
    ctx: &mut BandCtx,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    lowband_out: Option<&mut [f32]>,
) -> u32 {
    let mut sign = 0;
    if ctx.remaining_bits >= 1 << BITRES {
        if ctx.encode {
            sign = if x[0] < 0.0 { 1 } else { 0 };
            ctx.rc.enc_bits(sign as u32, 1);
        } else {
            sign = ctx.rc.dec_bits(1) as i32;
        }
        ctx.remaining_bits -= 1 << BITRES;
    }
    if ctx.resynth {
        x[0] = if sign != 0 { -1.0 } else { 1.0 };
    }
    if let Some(y_val) = y {
        let mut y_sign = 0;
        if ctx.remaining_bits >= 1 << BITRES {
            if ctx.encode {
                y_sign = if y_val[0] < 0.0 { 1 } else { 0 };
                ctx.rc.enc_bits(y_sign as u32, 1);
            } else {
                y_sign = ctx.rc.dec_bits(1) as i32;
            }
            ctx.remaining_bits -= 1 << BITRES;
        }
        if ctx.resynth {
            y_val[0] = if y_sign != 0 { -1.0 } else { 1.0 };
        }
    }
    if let Some(l_out) = lowband_out {
        l_out[0] = x[0] / 16.0;
    }
    1
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn quant_band(
    ctx: &mut BandCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    gain: f32,
    fill: u32,
) -> u32 {
    let n0 = n;
    let b0 = b_blocks;
    let long_blocks = b0 == 1;

    if n == 1 {
        return quant_band_n1(ctx, x, None, lowband_out);
    }

    let mut b_blocks = b_blocks;
    let mut n_b = n / b_blocks as usize;
    let mut time_divide = 0;
    let mut recombine = 0;
    let mut tf_change_local = ctx.tf_change;
    let mut fill = fill;

    if tf_change_local > 0 {
        recombine = tf_change_local;
    }

    let mut lowband_buf = lowband;

    static BIT_INTERLEAVE_TABLE: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];

    for k in 0..recombine {
        if ctx.encode {
            haar1(x, n >> k, 1 << k);
        }
        if let Some(ref mut lb) = lowband_buf {
            haar1(lb, n >> k, 1 << k);
        }
        fill = (BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize] as u32)
            | ((BIT_INTERLEAVE_TABLE[(fill >> 4) as usize] as u32) << 2);
    }
    b_blocks >>= recombine;
    n_b <<= recombine;

    while n_b & 1 == 0 && tf_change_local < 0 {
        if ctx.encode {
            haar1(x, n_b, b_blocks as usize);
        }
        if let Some(ref mut lb) = lowband_buf {
            haar1(lb, n_b, b_blocks as usize);
        }
        fill |= fill << b_blocks;
        b_blocks <<= 1;
        n_b >>= 1;
        time_divide += 1;
        tf_change_local += 1;
    }

    let b0_after = b_blocks;
    let n_b0 = n_b;

    if b_blocks > 1 {
        if ctx.encode {
            deinterleave_hadamard(
                x,
                n_b >> recombine as usize,
                (b_blocks << recombine) as usize,
                long_blocks,
            );
        }
        if let Some(ref mut lb) = lowband_buf {
            deinterleave_hadamard(
                lb,
                n_b >> recombine as usize,
                (b_blocks << recombine) as usize,
                long_blocks,
            );
        }
    }

    let cm = if ctx.encode {
        quant_partition_encode(ctx, x, n, b, b_blocks, lowband_buf, lm, gain, fill)
    } else {
        quant_partition(ctx, x, n, b, b_blocks, lowband_buf, lm, gain, fill)
    };

    if ctx.resynth {
        let mut cm = cm;

        if b_blocks > 1 {
            interleave_hadamard(
                x,
                n_b >> recombine as usize,
                (b0_after << recombine) as usize,
                long_blocks,
            );
        }

        let mut n_b_undo = n_b0;
        let mut b_undo = b0_after;
        for _ in 0..time_divide {
            b_undo >>= 1;
            n_b_undo <<= 1;
            cm |= cm >> b_undo;
            haar1(x, n_b_undo, b_undo as usize);
        }

        static BIT_DEINTERLEAVE_TABLE: [u8; 16] = [
            0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3,
            0xFC, 0xFF,
        ];
        for k in 0..recombine {
            cm = BIT_DEINTERLEAVE_TABLE[cm as usize & 0xF] as u32;
            haar1(x, n0 >> k, 1 << k);
        }
        let mut b_final = b0_after;
        b_final <<= recombine;

        if let Some(lb_out) = lowband_out {
            let scale = (n0 as f32).sqrt();
            for j in 0..n0 {
                lb_out[j] = scale * x[j];
            }
        }
        cm &= (1u32 << b_final) - 1;
        return cm;
    }

    cm
}

pub fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, side: f32, n: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        stereo_merge_neon(x, y, mid, side, n);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { stereo_merge_avx2(x, y, mid, side, n) };
        return;
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { stereo_merge_avx2(x, y, mid, side, n) };
            return;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    stereo_merge_scalar(x, y, mid, side, n);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn stereo_merge_avx2(x: &mut [f32], y: &mut [f32], mid: f32, side: f32, n: usize) {
    use std::arch::x86_64::*;

    let mut i = 0;

    let v_mid = _mm256_set1_ps(mid);
    let v_side = _mm256_set1_ps(side);

    while i + 15 < n {
        let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let x1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        let y0 = _mm256_loadu_ps(y.as_ptr().add(i));
        let y1 = _mm256_loadu_ps(y.as_ptr().add(i + 8));

        let x_val0 = _mm256_mul_ps(x0, v_mid);
        let x_val1 = _mm256_mul_ps(x1, v_mid);
        let y_val0 = _mm256_mul_ps(y0, v_side);
        let y_val1 = _mm256_mul_ps(y1, v_side);

        let new_x0 = _mm256_sub_ps(x_val0, y_val0);
        let new_x1 = _mm256_sub_ps(x_val1, y_val1);
        let new_y0 = _mm256_add_ps(x_val0, y_val0);
        let new_y1 = _mm256_add_ps(x_val1, y_val1);

        _mm256_storeu_ps(x.as_mut_ptr().add(i), new_x0);
        _mm256_storeu_ps(x.as_mut_ptr().add(i + 8), new_x1);
        _mm256_storeu_ps(y.as_mut_ptr().add(i), new_y0);
        _mm256_storeu_ps(y.as_mut_ptr().add(i + 8), new_y1);

        i += 16;
    }

    while i + 7 < n {
        let x0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let y0 = _mm256_loadu_ps(y.as_ptr().add(i));

        let x_val = _mm256_mul_ps(x0, v_mid);
        let y_val = _mm256_mul_ps(y0, v_side);

        let new_x = _mm256_sub_ps(x_val, y_val);
        let new_y = _mm256_add_ps(x_val, y_val);

        _mm256_storeu_ps(x.as_mut_ptr().add(i), new_x);
        _mm256_storeu_ps(y.as_mut_ptr().add(i), new_y);

        i += 8;
    }

    for j in i..n {
        let x_val = x[j] * mid;
        let y_val = y[j] * side;
        x[j] = x_val - y_val;
        y[j] = x_val + y_val;
    }
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn stereo_merge_scalar(x: &mut [f32], y: &mut [f32], mid: f32, side: f32, n: usize) {
    for i in 0..n {
        let x_val = x[i] * mid;
        let y_val = y[i] * side;
        x[i] = x_val - y_val;
        y[i] = x_val + y_val;
    }
}

#[cfg(target_arch = "aarch64")]
fn stereo_merge_neon(x: &mut [f32], y: &mut [f32], mid: f32, side: f32, n: usize) {
    use std::arch::aarch64::*;

    unsafe {
        let vmid = vdupq_n_f32(mid);
        let vside = vdupq_n_f32(side);

        let n16 = n & !15;
        for i in (0..n16).step_by(16) {
            let x0 = vld1q_f32(x.as_ptr().add(i));
            let x1 = vld1q_f32(x.as_ptr().add(i + 4));
            let x2 = vld1q_f32(x.as_ptr().add(i + 8));
            let x3 = vld1q_f32(x.as_ptr().add(i + 12));

            let y0 = vld1q_f32(y.as_ptr().add(i));
            let y1 = vld1q_f32(y.as_ptr().add(i + 4));
            let y2 = vld1q_f32(y.as_ptr().add(i + 8));
            let y3 = vld1q_f32(y.as_ptr().add(i + 12));

            let xv0 = vmulq_f32(x0, vmid);
            let xv1 = vmulq_f32(x1, vmid);
            let xv2 = vmulq_f32(x2, vmid);
            let xv3 = vmulq_f32(x3, vmid);

            let yv0 = vmulq_f32(y0, vside);
            let yv1 = vmulq_f32(y1, vside);
            let yv2 = vmulq_f32(y2, vside);
            let yv3 = vmulq_f32(y3, vside);

            vst1q_f32(x.as_mut_ptr().add(i), vsubq_f32(xv0, yv0));
            vst1q_f32(x.as_mut_ptr().add(i + 4), vsubq_f32(xv1, yv1));
            vst1q_f32(x.as_mut_ptr().add(i + 8), vsubq_f32(xv2, yv2));
            vst1q_f32(x.as_mut_ptr().add(i + 12), vsubq_f32(xv3, yv3));

            vst1q_f32(y.as_mut_ptr().add(i), vaddq_f32(xv0, yv0));
            vst1q_f32(y.as_mut_ptr().add(i + 4), vaddq_f32(xv1, yv1));
            vst1q_f32(y.as_mut_ptr().add(i + 8), vaddq_f32(xv2, yv2));
            vst1q_f32(y.as_mut_ptr().add(i + 12), vaddq_f32(xv3, yv3));
        }

        let n4 = (n & !3) - n16;
        for i in (n16..n16 + n4).step_by(4) {
            let xv = vld1q_f32(x.as_ptr().add(i));
            let yv = vld1q_f32(y.as_ptr().add(i));

            let x_val = vmulq_f32(xv, vmid);
            let y_val = vmulq_f32(yv, vside);

            vst1q_f32(x.as_mut_ptr().add(i), vsubq_f32(x_val, y_val));
            vst1q_f32(y.as_mut_ptr().add(i), vaddq_f32(x_val, y_val));
        }

        for i in (n16 + n4)..n {
            let x_val = x[i] * mid;
            let y_val = y[i] * side;
            x[i] = x_val - y_val;
            y[i] = x_val + y_val;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub fn quant_band_stereo(
    ctx: &mut BandCtx,
    x: &mut [f32],
    y: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    _gain: f32,
    fill: u32,
) -> u32 {
    if n == 1 {
        return quant_band_n1(ctx, x, Some(y), lowband_out);
    }

    if ctx.encode
        && (ctx.band_e[ctx.i] < MIN_STEREO_ENERGY
            || ctx.band_e[ctx.m.nb_ebands + ctx.i] < MIN_STEREO_ENERGY)
    {
        if ctx.band_e[ctx.i] > ctx.band_e[ctx.m.nb_ebands + ctx.i] {
            y.copy_from_slice(x);
        } else {
            x.copy_from_slice(y);
        }
    }

    let mut sctx = SplitCtx {
        inv: false,
        imid: 0,
        iside: 0,
        delta: 0,
        itheta: 0,
        qalloc: 0,
    };
    let mut b_mut = b;
    let mut fill_mut = fill;
    if ctx.encode {
        compute_theta_encode(
            ctx,
            &mut sctx,
            x,
            y,
            n,
            &mut b_mut,
            b_blocks,
            b_blocks,
            lm,
            true,
            &mut fill_mut,
        );
    } else {
        compute_theta(
            ctx,
            &mut sctx,
            x,
            y,
            n,
            &mut b_mut,
            b_blocks,
            b_blocks,
            lm,
            true,
            &mut fill_mut,
        );
    };

    let mid_gain = sctx.imid as f32 / 32768.0;
    let side_gain = sctx.iside as f32 / 32768.0;

    if n == 2 {
        let mut mbits = b_mut;
        let mut sbits = 0;
        if sctx.itheta != 0 && sctx.itheta != 16384 {
            sbits = 1 << BITRES;
        }
        mbits -= sbits;
        let c = sctx.itheta > 8192;
        ctx.remaining_bits -= sctx.qalloc + sbits;

        let mut sign = 0;
        if sbits != 0 {
            if ctx.encode {
                sign = if c {
                    if (y[0] * x[1] - y[1] * x[0]) < 0.0 {
                        1
                    } else {
                        0
                    }
                } else if (x[0] * y[1] - x[1] * y[0]) < 0.0 {
                    1
                } else {
                    0
                };
                ctx.rc.enc_bits(sign as u32, 1);
            } else {
                sign = ctx.rc.dec_bits(1) as i32;
            }
        }
        let sign_val = (1 - 2 * sign) as f32;
        let cm = if c {
            let cm = quant_band(
                ctx,
                y,
                n,
                mbits,
                b_blocks,
                lowband,
                lm,
                lowband_out,
                1.0,
                fill,
            );
            x[0] = -sign_val * y[1];
            x[1] = sign_val * y[0];
            cm
        } else {
            let cm = quant_band(
                ctx,
                x,
                n,
                mbits,
                b_blocks,
                lowband,
                lm,
                lowband_out,
                1.0,
                fill,
            );
            y[0] = -sign_val * x[1];
            y[1] = sign_val * x[0];
            cm
        };

        if ctx.resynth {
            let x0 = x[0];
            let x1 = x[1];
            let y0 = y[0];
            let y1 = y[1];
            x[0] = mid_gain * x0 - side_gain * y0;
            x[1] = mid_gain * x1 - side_gain * y1;
            y[0] = mid_gain * x0 + side_gain * y0;
            y[1] = mid_gain * x1 + side_gain * y1;
        }
        return cm;
    }

    ctx.remaining_bits -= sctx.qalloc;
    let mut mbits = (0).max((b_mut - sctx.delta) / 2).min(b_mut);
    let mut sbits = b_mut - mbits;

    let mut rebalance = ctx.remaining_bits;
    let mut cm;

    if mbits >= sbits {
        cm = quant_band(
            ctx,
            x,
            n,
            mbits,
            b_blocks,
            lowband,
            lm,
            lowband_out,
            1.0,
            fill_mut,
        );
        rebalance = mbits - (rebalance - ctx.remaining_bits);
        if rebalance > (3 << 3) && sctx.itheta != 0 {
            sbits += rebalance - (3 << 3);
        }
        cm |= quant_band(
            ctx,
            y,
            n,
            sbits,
            b_blocks,
            None,
            lm,
            None,
            side_gain,
            fill_mut >> b_blocks,
        ) << (b_blocks >> 1);
    } else {
        cm = quant_band(
            ctx,
            y,
            n,
            sbits,
            b_blocks,
            None,
            lm,
            None,
            side_gain,
            fill_mut >> b_blocks,
        ) << (b_blocks >> 1);
        rebalance = sbits - (rebalance - ctx.remaining_bits);
        if rebalance > (3 << 3) && sctx.itheta != 16384 {
            mbits += rebalance - (3 << 3);
        }
        cm |= quant_band(
            ctx,
            x,
            n,
            mbits,
            b_blocks,
            lowband,
            lm,
            lowband_out,
            1.0,
            fill_mut,
        );
    }

    if ctx.resynth {
        stereo_merge(x, y, mid_gain, side_gain, n);
        if sctx.inv {
            for yv in y[..n].iter_mut() {
                *yv = -*yv;
            }
        }
    }
    cm
}

#[allow(clippy::too_many_arguments)]
pub fn quant_all_bands(
    encode: bool,
    m: &CeltMode,
    start: usize,
    end: usize,
    x: &mut [f32],
    mut y: Option<&mut [f32]>,
    collapse_masks: &mut [u32],
    band_e: &[f32],
    pulses: &[i32],
    short_blocks: bool,
    spread: i32,
    dual_stereo: &mut bool,
    intensity: usize,
    tf_res: &[i32],
    total_bits: i32,
    balance: &mut i32,
    rc: &mut RangeCoder,
    lm: i32,
    coded_bands: i32,
    resynth: bool,
    seed: &mut u32,
) {
    let mut balance_val = *balance;
    let b_blocks = if short_blocks { 1 << lm } else { 1 };
    let c_channels = if y.is_some() { 2 } else { 1 };
    let m_val = 1usize << lm as usize;

    let norm_offset = m_val * (m.e_bands[start] as usize);
    let norm_size = m_val * (m.e_bands[m.nb_ebands - 1] as usize) - norm_offset;

    const MAX_NORM_SIZE: usize = 800;
    debug_assert!(norm_size <= MAX_NORM_SIZE);

    let mut norm_buf = [std::mem::MaybeUninit::<f32>::uninit(); MAX_NORM_SIZE];
    let norm =
        unsafe { std::slice::from_raw_parts_mut(norm_buf.as_mut_ptr() as *mut f32, norm_size) };

    let mut lowband_scratch_buf = [std::mem::MaybeUninit::<f32>::uninit(); MAX_PVQ_N];
    let lowband_scratch_ptr = lowband_scratch_buf.as_mut_ptr() as *mut f32;

    let lowband_offset: usize = 0;
    let mut avoid_split_noise = b_blocks > 1;

    let e_bands = &m.e_bands;
    let mut ctx_seed = *seed;

    for i in start..end {
        let e_band_i = e_bands[i] as usize;
        let e_band_i1 = e_bands[i + 1] as usize;
        let offset = m_val * e_band_i;
        let n = m_val * (e_band_i1 - e_band_i);
        let last = i == end - 1;

        let tell = tell_frac_inline!(rc);
        if i != start {
            balance_val -= tell;
        }
        let remaining_bits = total_bits - tell - 1;

        let mut b = 0i32;
        if i < coded_bands as usize {
            let curr_balance = celt_sudiv(balance_val, 3i32.min(coded_bands - i as i32));
            b = 0i32.max(16383i32.min((remaining_bits + 1).min(pulses[i] + curr_balance)));
        }

        let norm_pos = m_val * e_band_i - norm_offset;
        let tf_change = tf_res[i];

        let mut effective_lowband: i32 = -1;
        let mut x_cm: u32;
        let mut y_cm: u32;

        if lowband_offset != 0 && (spread != SPREAD_AGGRESSIVE || b_blocks > 1 || tf_change < 0) {
            effective_lowband = 0i32.max(
                (m_val * e_bands[lowband_offset] as usize) as i32 - norm_offset as i32 - n as i32,
            );
            let el_abs = effective_lowband as usize + norm_offset;

            let mut fold_start = lowband_offset;
            loop {
                if fold_start == 0 {
                    break;
                }
                fold_start -= 1;
                if m_val * (e_bands[fold_start] as usize) <= el_abs {
                    break;
                }
            }
            let mut fold_end = lowband_offset.saturating_sub(1);
            while fold_end + 1 < i && m_val * (e_bands[fold_end + 1] as usize) < el_abs + n {
                fold_end += 1;
            }

            x_cm = 0;
            y_cm = 0;
            for fi in fold_start..fold_end {
                x_cm |= collapse_masks[fi * c_channels];
                y_cm |= collapse_masks[fi * c_channels + c_channels - 1];
            }
        } else {
            x_cm = (1u32 << b_blocks) - 1;
            y_cm = (1u32 << b_blocks) - 1;
        }

        let mut ctx = BandCtx {
            encode,
            m,
            i,
            band_e,
            rc,
            spread,
            remaining_bits,
            resynth,
            tf_change,
            intensity,
            theta_round: 0,
            avoid_split_noise,
            arch: 0,
            disable_inv: false,
            seed: ctx_seed,
        };

        if *dual_stereo && i == intensity {
            *dual_stereo = false;
        }

        let mut lowband_scratch: Option<&mut [f32]> = if effective_lowband >= 0 {
            let lb_start = effective_lowband as usize;
            let lb_end = lb_start + n;
            if lb_end <= norm.len() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        norm.as_ptr().add(lb_start),
                        lowband_scratch_ptr,
                        n,
                    )
                };
                Some(unsafe { std::slice::from_raw_parts_mut(lowband_scratch_ptr, n) })
            } else {
                None
            }
        } else {
            None
        };

        let x_slice = &mut x[offset..offset + n];
        if *dual_stereo {
            let y_slice = &mut y.as_mut().unwrap()[offset..offset + n];
            let lb_x = lowband_scratch.as_deref_mut();
            let lb_out_x = if !last && norm_pos + n <= norm.len() {
                Some(&mut norm[norm_pos..norm_pos + n])
            } else {
                None
            };
            x_cm = quant_band(
                &mut ctx,
                x_slice,
                n,
                b / 2,
                b_blocks,
                lb_x,
                lm,
                lb_out_x,
                1.0,
                x_cm,
            );
            y_cm = quant_band(
                &mut ctx,
                y_slice,
                n,
                b / 2,
                b_blocks,
                None,
                lm,
                None,
                1.0,
                y_cm,
            );
        } else if let Some(y_all) = y.as_mut() {
            let y_slice = &mut y_all[offset..offset + n];
            let lb = lowband_scratch.as_deref_mut();
            let lb_out = if !last && norm_pos + n <= norm.len() {
                Some(&mut norm[norm_pos..norm_pos + n])
            } else {
                None
            };
            x_cm = quant_band_stereo(
                &mut ctx,
                x_slice,
                y_slice,
                n,
                b,
                b_blocks,
                lb,
                lm,
                lb_out,
                1.0,
                x_cm | y_cm,
            );
            y_cm = x_cm;
        } else {
            let lb = lowband_scratch;
            let lb_out = if !last && norm_pos + n <= norm.len() {
                Some(&mut norm[norm_pos..norm_pos + n])
            } else {
                None
            };
            x_cm = quant_band(&mut ctx, x_slice, n, b, b_blocks, lb, lm, lb_out, 1.0, x_cm);
            y_cm = x_cm;
        }

        collapse_masks[i * c_channels] = (x_cm & 0xFF) as u8 as u32;
        if c_channels == 2 {
            collapse_masks[i * c_channels + 1] = (y_cm & 0xFF) as u8 as u32;
        }

        balance_val += pulses[i] + tell;
        ctx_seed = ctx.seed;

        avoid_split_noise = false;
    }
    *balance = balance_val;
    *seed = ctx_seed;
}

#[cfg(target_arch = "aarch64")]
fn compute_band_energy_neon(band: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let n = band.len();
    let mut sum = 1e-27f32;

    unsafe {
        let n16 = n & !15;
        if n16 > 0 {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            for i in (0..n16).step_by(16) {
                let v0 = vld1q_f32(band.as_ptr().add(i));
                let v1 = vld1q_f32(band.as_ptr().add(i + 4));
                let v2 = vld1q_f32(band.as_ptr().add(i + 8));
                let v3 = vld1q_f32(band.as_ptr().add(i + 12));

                acc0 = vfmaq_f32(acc0, v0, v0);
                acc1 = vfmaq_f32(acc1, v1, v1);
                acc2 = vfmaq_f32(acc2, v2, v2);
                acc3 = vfmaq_f32(acc3, v3, v3);
            }

            acc0 = vaddq_f32(acc0, acc1);
            acc2 = vaddq_f32(acc2, acc3);
            acc0 = vaddq_f32(acc0, acc2);
            sum += vaddvq_f32(acc0);
        }

        let n4 = (n & !3) - n16;
        if n4 > 0 {
            let mut acc = vdupq_n_f32(0.0);
            for i in (n16..n16 + n4).step_by(4) {
                let v = vld1q_f32(band.as_ptr().add(i));
                acc = vfmaq_f32(acc, v, v);
            }
            sum += vaddvq_f32(acc);
        }

        for i in (n16 + n4)..n {
            let v = band[i];
            sum += v * v;
        }
    }

    sum.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_band_energy_avx2(band: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = band.len();
    let mut i = 0usize;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while i + 16 <= n {
        let v0 = _mm256_loadu_ps(band.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(band.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        i += 16;
    }

    if i + 8 <= n {
        let v0 = _mm256_loadu_ps(band.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        i += 8;
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s4 = _mm_add_ps(lo, hi);
    let t1 = _mm_movehl_ps(s4, s4);
    let s2 = _mm_add_ps(s4, t1);
    let t2 = _mm_shuffle_ps(s2, s2, 0x55);
    let mut sum = 1e-27f32 + _mm_cvtss_f32(_mm_add_ss(s2, t2));

    for &v in &band[i..] {
        sum += v * v;
    }

    sum.sqrt()
}

pub fn compute_band_energies(
    m: &CeltMode,
    x: &[f32],
    band_e: &mut [f32],
    end: usize,
    channels: usize,
    lm: usize,
) {
    let frame_size = m.short_mdct_size << lm;

    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::arch::is_x86_feature_detected!("avx2");

    for c in 0..channels {
        let ch = &x[c * frame_size..(c + 1) * frame_size];
        for i in 0..end {
            let offset = (m.e_bands[i] as usize) << lm;
            let n = ((m.e_bands[i + 1] - m.e_bands[i]) as usize) << lm;
            let band = &ch[offset..offset + n];

            #[cfg(target_arch = "aarch64")]
            {
                band_e[c * m.nb_ebands + i] = compute_band_energy_neon(band);
            }
            #[cfg(target_arch = "x86_64")]
            {
                if n >= 8 && use_avx2 {
                    band_e[c * m.nb_ebands + i] = unsafe { compute_band_energy_avx2(band) };
                } else {
                    let sum = band.iter().fold(1e-27f32, |acc, &v| acc + v * v);
                    band_e[c * m.nb_ebands + i] = sum.sqrt();
                }
            }
            #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
            {
                let sum = band.iter().fold(1e-27f32, |acc, &v| acc + v * v);
                band_e[c * m.nb_ebands + i] = sum.sqrt();
            }
        }
    }
}

pub fn amp2log2(
    m: &CeltMode,
    start: usize,
    end: usize,
    band_e: &[f32],
    band_log_e: &mut [f32],
    channels: usize,
) {
    for c in 0..channels {
        for i in 0..start {
            band_log_e[c * m.nb_ebands + i] = -14.0;
        }
        for i in start..end {
            let val = band_e[c * m.nb_ebands + i].max(1e-10);
            band_log_e[c * m.nb_ebands + i] = val.log2() - m.e_means[i];
        }
    }
}

pub fn log2amp(m: &CeltMode, end: usize, band_e: &mut [f32], band_log_e: &[f32], channels: usize) {
    for c in 0..channels {
        for i in 0..end {
            band_e[c * m.nb_ebands + i] = band_log_e[c * m.nb_ebands + i] + m.e_means[i];
        }
    }
}

pub fn normalise_bands(
    m: &CeltMode,
    freq: &[f32],
    x: &mut [f32],
    band_e: &[f32],
    end: usize,
    channels: usize,
    m_val: usize,
) {
    let lm = m_val.trailing_zeros() as usize;
    let frame_size = m.short_mdct_size << lm;
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::arch::is_x86_feature_detected!("avx2");
    for c in 0..channels {
        for i in 0..end {
            let base = c * frame_size + ((m.e_bands[i] as usize) << lm);
            let n = ((m.e_bands[i + 1] - m.e_bands[i]) as usize) << lm;
            let norm = 1.0 / (1e-27 + band_e[c * m.nb_ebands + i]);
            let src = &freq[base..base + n];
            let dst = &mut x[base..base + n];
            #[cfg(target_arch = "x86_64")]
            if n >= 8 && use_avx2 {
                unsafe { scale_slice_avx2(src, dst, norm, n) };
                continue;
            }
            #[cfg(target_arch = "aarch64")]
            if n >= 8 {
                unsafe { scale_slice_neon(src, dst, norm, n) };
                continue;
            }
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s * norm;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_slice_avx2(src: &[f32], dst: &mut [f32], scale: f32, n: usize) {
    use std::arch::x86_64::*;
    let vscale = _mm256_set1_ps(scale);
    let mut i = 0;

    while i + 16 <= n {
        let s0 = _mm256_loadu_ps(src.as_ptr().add(i));
        let s1 = _mm256_loadu_ps(src.as_ptr().add(i + 8));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_mul_ps(s0, vscale));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i + 8), _mm256_mul_ps(s1, vscale));
        i += 16;
    }
    while i + 8 <= n {
        let sv = _mm256_loadu_ps(src.as_ptr().add(i));
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_mul_ps(sv, vscale));
        i += 8;
    }
    for j in i..n {
        dst[j] = src[j] * scale;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn scale_slice_neon(src: &[f32], dst: &mut [f32], scale: f32, n: usize) {
    use std::arch::aarch64::*;
    let vscale = vdupq_n_f32(scale);
    let mut i = 0;

    while i + 16 <= n {
        let s0 = vld1q_f32(src.as_ptr().add(i));
        let s1 = vld1q_f32(src.as_ptr().add(i + 4));
        let s2 = vld1q_f32(src.as_ptr().add(i + 8));
        let s3 = vld1q_f32(src.as_ptr().add(i + 12));
        vst1q_f32(dst.as_mut_ptr().add(i), vmulq_f32(s0, vscale));
        vst1q_f32(dst.as_mut_ptr().add(i + 4), vmulq_f32(s1, vscale));
        vst1q_f32(dst.as_mut_ptr().add(i + 8), vmulq_f32(s2, vscale));
        vst1q_f32(dst.as_mut_ptr().add(i + 12), vmulq_f32(s3, vscale));
        i += 16;
    }
    while i + 8 <= n {
        let s0 = vld1q_f32(src.as_ptr().add(i));
        let s1 = vld1q_f32(src.as_ptr().add(i + 4));
        vst1q_f32(dst.as_mut_ptr().add(i), vmulq_f32(s0, vscale));
        vst1q_f32(dst.as_mut_ptr().add(i + 4), vmulq_f32(s1, vscale));
        i += 8;
    }
    while i + 4 <= n {
        let s0 = vld1q_f32(src.as_ptr().add(i));
        vst1q_f32(dst.as_mut_ptr().add(i), vmulq_f32(s0, vscale));
        i += 4;
    }
    for j in i..n {
        dst[j] = src[j] * scale;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn denormalise_bands(
    m: &CeltMode,
    x: &[f32],
    freq: &mut [f32],
    band_e: &[f32],
    start: usize,
    end: usize,
    channels: usize,
    m_val: usize,
) {
    let lm = m_val.trailing_zeros() as usize;
    let frame_size = m.short_mdct_size << lm;
    #[cfg(target_arch = "x86_64")]
    let use_avx2 = std::arch::is_x86_feature_detected!("avx2");

    for c in 0..channels {
        for i in start..end {
            let base = c * frame_size + ((m.e_bands[i] as usize) << lm);
            let n = ((m.e_bands[i + 1] - m.e_bands[i]) as usize) << lm;
            let band_log = band_e[c * m.nb_ebands + i];

            // Match C: celt_exp2_db(MIN32(32.f, lg)) — cap gain to prevent overflow
            let g = (2.0f32).powf(band_log.min(32.0));
            let src = &x[base..base + n];
            let dst = &mut freq[base..base + n];
            #[cfg(target_arch = "x86_64")]
            if n >= 8 && use_avx2 {
                unsafe { scale_slice_avx2(src, dst, g, n) };
                continue;
            }
            #[cfg(target_arch = "aarch64")]
            if n >= 8 {
                unsafe { scale_slice_neon(src, dst, g, n) };
                continue;
            }
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = s * g;
            }
        }
    }
}

pub fn celt_lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1664525).wrapping_add(1013904223)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn renormalise_vector_neon(x: &mut [f32], n: usize, gain: f32) {
    use std::arch::aarch64::*;

    let mut sum_vec = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 16 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        let x2 = vld1q_f32(x.as_ptr().add(i + 8));
        let x3 = vld1q_f32(x.as_ptr().add(i + 12));
        sum_vec = vfmaq_f32(sum_vec, x0, x0);
        sum_vec = vfmaq_f32(sum_vec, x1, x1);
        sum_vec = vfmaq_f32(sum_vec, x2, x2);
        sum_vec = vfmaq_f32(sum_vec, x3, x3);
        i += 16;
    }

    while i + 8 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        sum_vec = vfmaq_f32(sum_vec, x0, x0);
        sum_vec = vfmaq_f32(sum_vec, x1, x1);
        i += 8;
    }

    while i + 4 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        sum_vec = vfmaq_f32(sum_vec, x0, x0);
        i += 4;
    }

    let mut e = 1e-15f32 + vaddvq_f32(sum_vec);

    for j in i..n {
        e += x[j] * x[j];
    }

    let norm = gain / e.sqrt();
    let vnorm = vdupq_n_f32(norm);

    i = 0;
    while i + 16 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        let x2 = vld1q_f32(x.as_ptr().add(i + 8));
        let x3 = vld1q_f32(x.as_ptr().add(i + 12));
        vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(x0, vnorm));
        vst1q_f32(x.as_mut_ptr().add(i + 4), vmulq_f32(x1, vnorm));
        vst1q_f32(x.as_mut_ptr().add(i + 8), vmulq_f32(x2, vnorm));
        vst1q_f32(x.as_mut_ptr().add(i + 12), vmulq_f32(x3, vnorm));
        i += 16;
    }

    while i + 8 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(x0, vnorm));
        vst1q_f32(x.as_mut_ptr().add(i + 4), vmulq_f32(x1, vnorm));
        i += 8;
    }

    while i + 4 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        vst1q_f32(x.as_mut_ptr().add(i), vmulq_f32(x0, vnorm));
        i += 4;
    }

    for j in i..n {
        x[j] *= norm;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn renormalise_vector_avx2(x: &mut [f32], n: usize, gain: f32) {
    use std::arch::x86_64::*;

    let mut i = 0usize;

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    while i + 16 <= n {
        let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        acc1 = _mm256_fmadd_ps(v1, v1, acc1);
        i += 16;
    }

    if i + 8 <= n {
        let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(v0, v0, acc0);
        i += 8;
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let s4 = _mm_add_ps(lo, hi);
    let t1 = _mm_movehl_ps(s4, s4);
    let s2 = _mm_add_ps(s4, t1);
    let t2 = _mm_shuffle_ps(s2, s2, 0x55);
    let mut e = 1e-15f32 + _mm_cvtss_f32(_mm_add_ss(s2, t2));

    for &v in &x[i..n] {
        e += v * v;
    }

    let norm = gain / e.sqrt();
    let vnorm = _mm256_set1_ps(norm);

    i = 0;
    while i + 16 <= n {
        let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(v0, vnorm));
        _mm256_storeu_ps(x.as_mut_ptr().add(i + 8), _mm256_mul_ps(v1, vnorm));
        i += 16;
    }
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(v, vnorm));
        i += 8;
    }
    for v in &mut x[i..n] {
        *v *= norm;
    }
}

pub fn renormalise_vector(x: &mut [f32], n: usize, gain: f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        renormalise_vector_neon(x, n, gain);
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if n >= 16 && std::arch::is_x86_feature_detected!("avx2") {
            renormalise_vector_avx2(x, n, gain);
            return;
        }
    }
    #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
    {
        let mut e = 1e-15f32;
        for &xv in x[..n].iter() {
            e += xv * xv;
        }
        let norm = gain / e.sqrt();
        for xv in x[..n].iter_mut() {
            *xv *= norm;
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        let mut e = 1e-15f32;
        for &xv in x[..n].iter() {
            e += xv * xv;
        }
        let norm = gain / e.sqrt();
        for xv in x[..n].iter_mut() {
            *xv *= norm;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn anti_collapse(
    m: &CeltMode,
    x_buf: &mut [f32],
    collapse_masks: &[u32],
    lm: i32,
    channels: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[f32],
    prev1_log_e: &[f32],
    prev2_log_e: &[f32],
    pulses: &[i32],
    mut seed: u32,
) -> u32 {
    for i in start..end {
        let n0 = (m.e_bands[i + 1] - m.e_bands[i]) as usize;
        let depth = if n0 > 0 {
            ((1 + pulses[i]) / n0 as i32) >> lm
        } else {
            0
        };

        let thresh = 0.5 * (-(0.125 * depth as f32)).exp2();
        let sqrt_1 = 1.0 / ((n0 << lm) as f32).sqrt();

        for c in 0..channels {
            let p1 = prev1_log_e[c * m.nb_ebands + i];
            let p2 = prev2_log_e[c * m.nb_ebands + i];

            let (p1_adj, p2_adj) = if channels == 1 && prev1_log_e.len() >= 2 * m.nb_ebands {
                (
                    p1.max(prev1_log_e[m.nb_ebands + i]),
                    p2.max(prev2_log_e[m.nb_ebands + i]),
                )
            } else {
                (p1, p2)
            };

            let e_diff = log_e[c * m.nb_ebands + i] - p1_adj.min(p2_adj);
            let e_diff = e_diff.max(0.0);

            let mut r = 2.0 * (-e_diff).exp2();
            if lm == 3 {
                r *= std::f32::consts::SQRT_2;
            }
            r = r.min(thresh);
            r *= sqrt_1;

            let x_offset = c * size + ((m.e_bands[i] as usize) << lm);
            let mut renormalize = false;
            for k in 0..(1 << lm) {
                if (collapse_masks[i * channels + c] & (1 << k)) == 0 {
                    for j in 0..n0 {
                        seed = celt_lcg_rand(seed);
                        x_buf[x_offset + (j << lm) + k] = if (seed & 0x8000) != 0 { r } else { -r };
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                renormalise_vector(&mut x_buf[x_offset..x_offset + (n0 << lm)], n0 << lm, 1.0);
            }
        }
    }
    seed
}
