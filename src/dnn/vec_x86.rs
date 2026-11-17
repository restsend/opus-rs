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

use fearless_simd::{Avx2, Sse2, f32x4, f32x8, prelude::*, u8x32};

use super::nnet::diag_by;
use super::vec::{Elem, MAX_INPUTS, c_floor, sgemv_by, sparse_sgemv8x4_by};
use super::weights::{F32s, I32s};
use crate::Arch;
#[cfg(not(feature = "std"))]
use crate::compat::Math;

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

/// `vector_ps_to_epi8` without AVX2: the unsigned inputs of the int8
/// layers, `127 + (int)floor(.5 + 127*x)` kept to its low 8 bits.
fn quantize4(q: &mut [u8], x: &[f32]) {
    for (q, &v) in q.iter_mut().zip(x) {
        *q = c_floor(0.5f64 + (127.0f32 * v) as f64).wrapping_add(127) as u8;
    }
}

fearless_simd::kernel!(
    /// `vector_ps_to_epi8` with AVX2: `_mm256_cvtps_epi32(fma(x, 127,
    /// 127))`, saturated to 0..65535 by `_mm256_packus_epi32`, then as
    /// signed 16-bit values to 0..255 by `_mm256_packus_epi16`.
    fn quantize8(simd: Avx2, q: &mut [u8], x: &[f32]) {
        let c127 = _mm256_set1_ps(127.0);
        let zero = _mm256_setzero_si256();
        for (q, x) in q.chunks_mut(8).zip(x.chunks(8)) {
            let mut b = [0.0f32; 8];
            b[..x.len()].copy_from_slice(x);
            let v = _mm256_fmadd_ps(f32x8::from_slice(simd, &b).into(), c127, c127);
            let v = _mm256_packus_epi32(_mm256_cvtps_epi32(v), zero);
            let v = _mm256_packus_epi16(_mm256_permute4x64_epi64::<0xD8>(v), zero);
            let mut bytes = [0u8; 32];
            let v: u8x32<Avx2> = v.simd_into(simd);
            v.store_slice(&mut bytes);
            q.copy_from_slice(&bytes[..q.len()]);
        }
    }
);

/// The four input bytes of the block at column `pos` as a 32-bit lane.
#[inline(always)]
fn word(x: &[u8], pos: usize) -> i32 {
    i32::from_le_bytes([x[pos], x[pos + 1], x[pos + 2], x[pos + 3]])
}

/// The column blocks of the rows `i..i + 8` (`cgemv8x4`: every fourth
/// column; `sparse_cgemv8x4`: the positions `idx` lists from `*p`).
#[inline(always)]
fn columns<'a>(
    idx: Option<I32s<'a>>,
    p: &mut usize,
    cols: usize,
) -> impl Iterator<Item = usize> + 'a {
    let (start, n) = match idx {
        Some(idx) => {
            let n = idx.at(*p) as usize;
            *p += 1 + n;
            (*p - n, n)
        }
        None => (0, cols / 4),
    };
    (0..n).map(move |k| match idx {
        Some(idx) => idx.at(start + k) as usize,
        None => 4 * k,
    })
}

/// The 8x4 int8 blocks at the SSE levels, `opus_mm256_dpbusds_epi32`
/// lane by lane: SSE2 adds the four products of a row exactly, SSSE3's
/// `_mm_maddubs_epi16` (the SSE4.1 build) adds them in pairs saturated to
/// 16 bits.
fn blocks4(
    out: &mut [f32],
    w: &[u8],
    idx: Option<I32s<'_>>,
    scale: F32s<'_>,
    rows: usize,
    cols: usize,
    x: &[u8],
    saturate: bool,
) {
    let sat = |v: i32| if saturate { v.clamp(-32768, 32767) } else { v };
    let (mut p, mut wp) = (0, 0);
    for i in (0..rows).step_by(8) {
        let mut acc = [0i32; 8];
        for pos in columns(idx, &mut p, cols) {
            let xb = &x[pos..pos + 4];
            let blk = &w[wp..wp + 32];
            for (k, a) in acc.iter_mut().enumerate() {
                let m = |c: usize| xb[c] as i32 * blk[4 * k + c] as i8 as i32;
                *a = a.wrapping_add(sat(m(0) + m(1)) + sat(m(2) + m(3)));
            }
            wp += 32;
        }
        for (k, &a) in acc.iter().enumerate() {
            out[i + k] = a as f32 * scale.at(i + k);
        }
    }
}

fearless_simd::kernel!(
    /// The 8x4 int8 blocks with AVX2: `_mm256_maddubs_epi16` (pair sums
    /// saturated to 16 bits), `_mm256_madd_epi16` with ones, the 32-bit
    /// sums converted to float and scaled.
    #[allow(clippy::too_many_arguments)]
    fn blocks8(
        simd: Avx2,
        out: &mut [f32],
        w: &[u8],
        idx: Option<I32s<'_>>,
        scale: F32s<'_>,
        rows: usize,
        cols: usize,
        x: &[u8],
    ) {
        let ones = _mm256_set1_epi16(1);
        let (mut p, mut wp) = (0, 0);
        for i in (0..rows).step_by(8) {
            let mut acc = _mm256_setzero_si256();
            for pos in columns(idx, &mut p, cols) {
                let vx = _mm256_set1_epi32(word(x, pos));
                let vw = u8x32::from_slice(simd, &w[wp..wp + 32]).into();
                let t = _mm256_madd_epi16(_mm256_maddubs_epi16(vx, vw), ones);
                acc = _mm256_add_epi32(acc, t);
                wp += 32;
            }
            let s = f32x8::from_slice(simd, &scale.array::<8>(i)).into();
            let y: f32x8<Avx2> = _mm256_mul_ps(_mm256_cvtepi32_ps(acc), s).simd_into(simd);
            y.store_slice(&mut out[i..i + 8]);
        }
    }
);

/// `cgemv8x4` (`idx` `None`) and `sparse_cgemv8x4` of vec_avx.h at `arch`
/// (an x86 level): the inputs as unsigned bytes, integer sums of the
/// products of each 8x4 block, converted to float and scaled. The layer
/// then adds its `subias` (`USE_SU_BIAS`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cgemv8x4(
    out: &mut [f32],
    w: &[u8],
    idx: Option<I32s<'_>>,
    scale: F32s<'_>,
    rows: usize,
    cols: usize,
    x_in: &[f32],
    arch: Arch,
) {
    let mut x = [0u8; MAX_INPUTS];
    let (x, x_in) = (&mut x[..cols], &x_in[..cols]);
    match arch.avx2() {
        Some(avx2) => {
            quantize8(avx2, x, x_in);
            blocks8(avx2, out, w, idx, scale, rows, cols, x);
        }
        None => {
            quantize4(x, x_in);
            blocks4(out, w, idx, scale, rows, cols, x, arch == Arch::Sse4_1);
        }
    }
}

fearless_simd::kernel!(
    /// `sgemv` with AVX2: the outputs in libopus's blocks of 16, 8 and 4
    /// rows sum fused multiply-adds (`_mm256_fmadd_ps`, `_mm_fmadd_ps`);
    /// the last `rows % 4` take libopus's C loop, `out[i] +=
    /// weights[...]*x[j]`, which GCC compiles to a multiply and an add
    /// (`vmulss`, `vaddss` in the reference's `compute_linear_avx2`).
    pub(crate) fn sgemv(
        simd: Avx2,
        out: &mut [f32],
        w: F32s<'_>,
        rows: usize,
        cols: usize,
        col_stride: usize,
        x: &[f32],
    ) {
        let r4 = rows & !3;
        sgemv_by(out, w, r4, cols, col_stride, x, |w, x, y| w.mul_add(x, y));
        let rest = rows - r4;
        sgemv_by(
            &mut out[r4..],
            w.from(r4),
            rest,
            cols,
            col_stride,
            x,
            |w, x, y| y + w * x,
        );
    }
);

fearless_simd::kernel!(
    /// `sparse_sgemv8x4` with AVX2: fused multiply-adds
    /// (`_mm256_fmadd_ps`).
    pub(crate) fn sparse_sgemv8x4(
        simd: Avx2,
        out: &mut [f32],
        w: F32s<'_>,
        idx: I32s<'_>,
        rows: usize,
        x: &[f32],
    ) {
        sparse_sgemv8x4_by(out, w, idx, rows, x, |w, x, y| w.mul_add(x, y));
    }
);

fearless_simd::kernel!(
    /// The diagonal term of `compute_linear_avx2`: GCC contracts each
    /// `out[i] += linear->diag[i]*in[i]` of nnet_arch.h to a fused
    /// multiply-add (the three `vfmadd132ss` of the reference object;
    /// none in a `-ffp-contract=off` build of it).
    pub(crate) fn diag(simd: Avx2, out: &mut [f32], diag: F32s<'_>, input: &[f32], m: usize) {
        diag_by(out, diag, input, m, |d, x, y| d.mul_add(x, y));
    }
);
