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

use super::nnet::{Conv2dLayer, conv2d_by, diag_by, with_zeroed};
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
        let quantize = |x: &[f32; 8]| -> [u8; 8] {
            let v = _mm256_fmadd_ps(f32x8::from_slice(simd, x).into(), c127, c127);
            let v = _mm256_packus_epi32(_mm256_cvtps_epi32(v), zero);
            let v = _mm256_packus_epi16(_mm256_permute4x64_epi64::<0xD8>(v), zero);
            let v: u8x32<Avx2> = v.simd_into(simd);
            v.as_array()[..8].try_into().unwrap()
        };
        let (qs, q_rest) = q.as_chunks_mut::<8>();
        let (xs, x_rest) = x.as_chunks::<8>();
        for (q, x) in qs.iter_mut().zip(xs) {
            *q = quantize(x);
        }
        if !x_rest.is_empty() {
            let mut b = [0.0f32; 8];
            b[..x_rest.len()].copy_from_slice(x_rest);
            q_rest.copy_from_slice(&quantize(&b)[..q_rest.len()]);
        }
    }
);

/// The 8x4 int8 blocks of `cgemv8x4` (`idx` `None`: every fourth input
/// column of every 8 rows) or `sparse_cgemv8x4` (the columns `idx` lists):
/// `finish(i, sum)` for the rows `i..i + 8`, with `sum` the `add` of the
/// `prod(inputs, weights)` of their blocks (integer sums, so the grouping
/// in fours does not change them).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn blocks<A: Copy>(
    rows: usize,
    cols: usize,
    idx: Option<I32s<'_>>,
    w: &[u8],
    x: &[u8],
    zero: A,
    prod: impl Fn(&[u8; 4], &[u8; 32]) -> A,
    add: impl Fn(A, A) -> A,
    mut finish: impl FnMut(usize, A),
) {
    let wb = w.as_chunks::<32>().0;
    let xb = x[..cols].as_chunks::<4>().0;
    let p = |(x, w): (&[u8; 4], &[u8; 32])| prod(x, w);
    match idx {
        None => {
            let n = xb.len();
            let (x4, x_rest) = xb.as_chunks::<4>();
            for i in (0..rows).step_by(8) {
                let row = &wb[i / 8 * n..i / 8 * n + n];
                let (w4, w_rest) = row.as_chunks::<4>();
                let mut s = zero;
                for (x, w) in x4.iter().zip(w4) {
                    let a = add(p((&x[0], &w[0])), p((&x[1], &w[1])));
                    let b = add(p((&x[2], &w[2])), p((&x[3], &w[3])));
                    s = add(s, add(a, b));
                }
                for b in x_rest.iter().zip(w_rest) {
                    s = add(s, p(b));
                }
                finish(i, s);
            }
        }
        Some(idx) => {
            let (mut p, mut wp) = (0, 0);
            let block = |pos: &[u8; 4]| -> &[u8; 4] {
                let pos = i32::from_le_bytes(*pos) as usize;
                x[pos..pos + 4].try_into().unwrap()
            };
            for i in (0..rows).step_by(8) {
                let n = idx.at(p) as usize;
                let (p4, p_rest) = idx.bytes(p + 1, n).as_chunks::<4>();
                let (w4, w_rest) = wb[wp..wp + n].as_chunks::<4>();
                let mut s = zero;
                for (pos, w) in p4.iter().zip(w4) {
                    let q = |k: usize| prod(block(&pos[k]), &w[k]);
                    s = add(s, add(add(q(0), q(1)), add(q(2), q(3))));
                }
                for (pos, w) in p_rest.iter().zip(w_rest) {
                    s = add(s, prod(block(pos), w));
                }
                (p, wp) = (p + 1 + n, wp + n);
                finish(i, s);
            }
        }
    }
}

/// The 8x4 int8 blocks at the SSE levels, `opus_mm256_dpbusds_epi32`
/// lane by lane: SSE2 adds the four products of a row exactly, SSSE3's
/// `_mm_maddubs_epi16` (the SSE4.1 build) adds them in pairs saturated to
/// 16 bits.
#[allow(clippy::too_many_arguments)]
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
    let prod = |x: &[u8; 4], w: &[u8; 32]| {
        core::array::from_fn(|k| {
            let m = |c: usize| x[c] as i32 * w[4 * k + c] as i8 as i32;
            sat(m(0) + m(1)) + sat(m(2) + m(3))
        })
    };
    let add = |a: [i32; 8], b: [i32; 8]| core::array::from_fn(|k| a[k].wrapping_add(b[k]));
    blocks(rows, cols, idx, w, x, [0i32; 8], prod, add, |i, acc| {
        for (k, &a) in acc.iter().enumerate() {
            out[i + k] = a as f32 * scale.at(i + k);
        }
    });
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
        let prod = |x: &[u8; 4], w: &[u8; 32]| {
            let vx = _mm256_set1_epi32(i32::from_le_bytes(*x));
            let vw = u8x32::from_slice(simd, w).into();
            _mm256_madd_epi16(_mm256_maddubs_epi16(vx, vw), ones)
        };
        let add = |a, b| _mm256_add_epi32(a, b);
        let zero = _mm256_setzero_si256();
        blocks(rows, cols, idx, w, x, zero, prod, add, |i, acc| {
            let s = f32x8::from_slice(simd, &scale.array::<8>(i)).into();
            let y: f32x8<Avx2> = _mm256_mul_ps(_mm256_cvtepi32_ps(acc), s).simd_into(simd);
            y.store_slice(&mut out[i..i + 8]);
        });
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
    assert!(cols <= MAX_INPUTS);
    let x_in = &x_in[..cols];
    with_zeroed(cols, |x: &mut [u8]| match arch.avx2() {
        Some(avx2) => {
            quantize8(avx2, x, x_in);
            blocks8(avx2, out, w, idx, scale, rows, cols, x);
        }
        None => {
            quantize4(x, x_in);
            blocks4(out, w, idx, scale, rows, cols, x, arch == Arch::Sse4_1);
        }
    })
}

fearless_simd::kernel!(
    /// `sgemv` with AVX2: the outputs in libopus's blocks of 16, 8 and 4
    /// rows sum fused multiply-adds (`_mm256_fmadd_ps`, `_mm_fmadd_ps`) in
    /// registers over the columns; the last `rows % 4` take libopus's C
    /// loop, `out[i] += weights[...]*x[j]`, which GCC compiles to a
    /// multiply and an add (`vmulss`, `vaddss` in the reference's
    /// `compute_linear_avx2`).
    pub(crate) fn sgemv(
        simd: Avx2,
        out: &mut [f32],
        w: F32s<'_>,
        rows: usize,
        cols: usize,
        col_stride: usize,
        x: &[f32],
    ) {
        let x = &x[..cols];
        let mut i = 0;
        while i + 16 <= rows {
            let (mut y0, mut y8) = (f32x8::splat(simd, 0.0), f32x8::splat(simd, 0.0));
            for (j, &xj) in x.iter().enumerate() {
                let xj = f32x8::splat(simd, xj);
                let wj = w.array::<16>(j * col_stride + i);
                y0 = f32x8::from_slice(simd, &wj[..8]).mul_add(xj, y0);
                y8 = f32x8::from_slice(simd, &wj[8..]).mul_add(xj, y8);
            }
            y0.store_slice(&mut out[i..i + 8]);
            y8.store_slice(&mut out[i + 8..i + 16]);
            i += 16;
        }
        if i + 8 <= rows {
            let mut y = f32x8::splat(simd, 0.0);
            for (j, &xj) in x.iter().enumerate() {
                let wj = f32x8::from_slice(simd, &w.array::<8>(j * col_stride + i));
                y = wj.mul_add(f32x8::splat(simd, xj), y);
            }
            y.store_slice(&mut out[i..i + 8]);
            i += 8;
        }
        if i + 4 <= rows {
            let mut y = f32x4::splat(simd, 0.0);
            for (j, &xj) in x.iter().enumerate() {
                let wj = f32x4::from_slice(simd, &w.array::<4>(j * col_stride + i));
                y = wj.mul_add(f32x4::splat(simd, xj), y);
            }
            y.store_slice(&mut out[i..i + 4]);
            i += 4;
        }
        let rest = rows - i;
        sgemv_by(
            &mut out[i..],
            w.from(i),
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

fearless_simd::kernel!(
    /// The convolution of `compute_conv2d_avx2`. GCC contracts nnet_arch.h's
    /// `out[...] += w0*x0 + w1*x1 + ... + w8*x8` (conv2d_3x3_float) to
    /// fma(w0, x0, w1*x1), then fma(wk, xk, sum) for k = 2..8, then adds
    /// the sum to the output (one `vmulss`, eight `vfmadd*ss` and one
    /// `vaddss` in the reference object), and conv2d_float's
    /// `out[...] += w*in[...]` to fma(w, in, out) (one `vfmadd132ss`).
    pub(crate) fn conv2d(
        simd: Avx2,
        conv: &Conv2dLayer<'_>,
        out: &mut [f32],
        input: &[f32],
        height: usize,
        hstride: usize,
    ) {
        conv2d_by(conv, out, input, height, hstride, |w, x, y| w.mul_add(x, y));
    }
);
