//! The x86 kernels of libopus `dnn/vec_avx.h`, at the levels where its
//! RTCD selects them (`nnet_sse2.c`, `nnet_sse4_1.c`, `nnet_avx2.c`).
//!
//! - `Sse2` and `Sse4_1`: vec_avx.h built without AVX. The tanh, sigmoid
//!   and exp approximations run four lanes at a time, `_mm_fmadd_ps(a, b,
//!   c)` being `_mm_add_ps(_mm_mul_ps(a, b), c)`; the exp floors with
//!   `_mm_cvtps_epi32(x - 0.5)` (vec_avx.h tests `__SSE_4_1__`, which no
//!   compiler defines) and clears the sign bit of its result.
//! - `Avx2`: eight lanes with fused multiply-adds, `_mm256_floor_ps` and
//!   no sign mask.
//!
//! The reciprocals come from `_mm_rcp_ps` and `_mm256_rcp_ps`,
//! approximations whose results the CPU defines, so they are the same
//! instructions here and the results follow the CPU as libopus's do. A
//! single value (`tanh_approx`, `sigmoid_approx`, `lpcnet_exp`) is lane 0
//! of the vector function, so every function is element-wise.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use fearless_simd::{Avx2, Sse2, f32x4, f32x8, prelude::*};

use super::vec::Elem;
use crate::Arch;

const TANH_N: [f32; 3] = [952.52801514, 96.39235687, 0.60863042];
const TANH_D: [f32; 3] = [952.72399902, 413.36801147, 11.88600922];
const SIGMOID_N: [f32; 3] = [238.13200378, 6.02452230, 0.00950985];
const SIGMOID_D: [f32; 3] = [952.72399902, 103.34200287, 0.74287558];
const EXP_K: [f32; 4] = [0.99992522, 0.69583354, 0.22606716, 0.078024523];
const LOG2_E: f32 = 1.44269504;

fearless_simd::kernel!(
    /// `tanh4_approx`, `sigmoid4_approx`, `exp4_approx` and the swish of
    /// four values.
    #[inline]
    fn elem4(simd: Sse2, f: Elem, x: __m128) -> __m128 {
        let s = |v: f32| _mm_set1_ps(v);
        let madd = |a: __m128, b: __m128, c: __m128| _mm_add_ps(_mm_mul_ps(a, b), c);
        let ratio = |n: [f32; 3], d: [f32; 3], x: __m128| {
            let x2 = _mm_mul_ps(x, x);
            let num = madd(madd(s(n[2]), x2, s(n[1])), x2, s(n[0]));
            let den = madd(madd(s(d[2]), x2, s(d[1])), x2, s(d[0]));
            (_mm_mul_ps(num, x), _mm_rcp_ps(den))
        };
        let sigmoid = |x: __m128| {
            let (num, den) = ratio(SIGMOID_N, SIGMOID_D, x);
            _mm_max_ps(s(0.0), _mm_min_ps(s(1.0), madd(num, den, s(0.5))))
        };
        match f {
            Elem::Tanh => {
                let (num, den) = ratio(TANH_N, TANH_D, x);
                _mm_max_ps(s(-1.0), _mm_min_ps(s(1.0), _mm_mul_ps(num, den)))
            }
            Elem::Sigmoid => sigmoid(x),
            Elem::Swish => _mm_mul_ps(x, sigmoid(x)),
            Elem::Exp => {
                let x = _mm_mul_ps(x, s(LOG2_E));
                let x = _mm_max_ps(s(-50.0), _mm_min_ps(s(50.0), x));
                let xf = _mm_cvtepi32_ps(_mm_cvtps_epi32(_mm_sub_ps(x, s(0.5))));
                let i = _mm_slli_epi32::<23>(_mm_cvtps_epi32(xf));
                let x = _mm_sub_ps(x, xf);
                let y = madd(
                    madd(madd(s(EXP_K[3]), x, s(EXP_K[2])), x, s(EXP_K[1])),
                    x,
                    s(EXP_K[0]),
                );
                let y = _mm_add_epi32(i, _mm_castps_si128(y));
                _mm_castsi128_ps(_mm_and_si128(_mm_set1_epi32(0x7fff_ffff), y))
            }
        }
    }
);

fearless_simd::kernel!(
    /// `tanh8_approx`, `sigmoid8_approx`, `exp8_approx` and the swish of
    /// eight values.
    #[inline]
    fn elem8(simd: Avx2, f: Elem, x: __m256) -> __m256 {
        let s = |v: f32| _mm256_set1_ps(v);
        let ratio = |n: [f32; 3], d: [f32; 3], x: __m256| {
            let x2 = _mm256_mul_ps(x, x);
            let num = _mm256_fmadd_ps(_mm256_fmadd_ps(s(n[2]), x2, s(n[1])), x2, s(n[0]));
            let den = _mm256_fmadd_ps(_mm256_fmadd_ps(s(d[2]), x2, s(d[1])), x2, s(d[0]));
            (_mm256_mul_ps(num, x), _mm256_rcp_ps(den))
        };
        let sigmoid = |x: __m256| {
            let (num, den) = ratio(SIGMOID_N, SIGMOID_D, x);
            let y = _mm256_fmadd_ps(num, den, s(0.5));
            _mm256_max_ps(s(0.0), _mm256_min_ps(s(1.0), y))
        };
        match f {
            Elem::Tanh => {
                let (num, den) = ratio(TANH_N, TANH_D, x);
                _mm256_max_ps(s(-1.0), _mm256_min_ps(s(1.0), _mm256_mul_ps(num, den)))
            }
            Elem::Sigmoid => sigmoid(x),
            Elem::Swish => _mm256_mul_ps(x, sigmoid(x)),
            Elem::Exp => {
                let x = _mm256_mul_ps(x, s(LOG2_E));
                let x = _mm256_max_ps(s(-50.0), _mm256_min_ps(s(50.0), x));
                let xf = _mm256_floor_ps(x);
                let i = _mm256_slli_epi32::<23>(_mm256_cvtps_epi32(xf));
                let x = _mm256_sub_ps(x, xf);
                let k = |c: usize| s(EXP_K[c]);
                let y = _mm256_fmadd_ps(
                    _mm256_fmadd_ps(_mm256_fmadd_ps(k(3), x, k(2)), x, k(1)),
                    x,
                    k(0),
                );
                _mm256_castsi256_ps(_mm256_add_epi32(i, _mm256_castps_si256(y)))
            }
        }
    }
);

fearless_simd::kernel!(
    /// [`elem4`] on every value of `x`.
    fn apply4(simd: Sse2, f: Elem, x: &mut [f32]) {
        let (chunks, rest) = x.as_chunks_mut::<4>();
        for c in chunks {
            let y = elem4(simd, f, f32x4::from_slice(simd, c).into());
            let y: f32x4<Sse2> = y.simd_into(simd);
            y.store_slice(c);
        }
        if !rest.is_empty() {
            let mut b = [0.0f32; 4];
            b[..rest.len()].copy_from_slice(rest);
            let y = elem4(simd, f, f32x4::from_slice(simd, &b).into());
            let y: f32x4<Sse2> = y.simd_into(simd);
            y.store_slice(&mut b);
            rest.copy_from_slice(&b[..rest.len()]);
        }
    }
);

fearless_simd::kernel!(
    /// [`elem8`] on every value of `x`.
    fn apply8(simd: Avx2, f: Elem, x: &mut [f32]) {
        let (chunks, rest) = x.as_chunks_mut::<8>();
        for c in chunks {
            let y = elem8(simd, f, f32x8::from_slice(simd, c).into());
            let y: f32x8<Avx2> = y.simd_into(simd);
            y.store_slice(c);
        }
        if !rest.is_empty() {
            let mut b = [0.0f32; 8];
            b[..rest.len()].copy_from_slice(rest);
            let y = elem8(simd, f, f32x8::from_slice(simd, &b).into());
            let y: f32x8<Avx2> = y.simd_into(simd);
            y.store_slice(&mut b);
            rest.copy_from_slice(&b[..rest.len()]);
        }
    }
);

/// `f` on every value of `x` at `arch`, one of the x86 levels.
pub(crate) fn apply(f: Elem, x: &mut [f32], arch: Arch) {
    match arch.avx2() {
        Some(avx2) => apply8(avx2, f, x),
        None => apply4(sse2(), f, x),
    }
}

/// The SSE2 token (SSE2 is in the x86-64 baseline).
pub(crate) fn sse2() -> Sse2 {
    fearless_simd::Level::baseline().as_sse2().unwrap()
}
