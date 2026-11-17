//! The generic C kernels of libopus `dnn/vec.h` (its no-SIMD branch,
//! without `USE_SU_BIAS`), with the same operation order: every output
//! sums its products in the order of the C loops, starting from +0.
//!
//! libopus selects these kernels when no SIMD header applies. The
//! reference build of `fuzz/opusref` (feature `dnn`) compiles `dnn/` with
//! `-U__SSE2__` so that x86-64 uses them too.

#[cfg(not(feature = "std"))]
use crate::compat::Math;

use super::weights::{F32s, I32s};

/// `MAX_INPUTS`: the largest input count of an int8 layer.
pub const MAX_INPUTS: usize = 2048;

/// C's `(int)x` for a double on x86-64 (`cvttsd2si`): truncation, and
/// `INT_MIN` for NaN and values out of range.
#[inline(always)]
pub fn c_int(x: f64) -> i32 {
    if x > -2147483649.0 && x < 2147483648.0 { x as i32 } else { i32::MIN }
}

/// C's `(int)floor(x)` for a double.
#[inline(always)]
pub fn c_floor(x: f64) -> i32 {
    c_int(x.floor())
}

/// The int8 input of `cgemv8x4`: `(int)floor(.5+127*x)` stored in an
/// `opus_int8` (gcc keeps the low 8 bits).
#[inline(always)]
fn quantize(x: f32) -> i8 {
    c_int((0.5f64 + (127.0f32 * x) as f64).floor()) as i8
}

/// `sgemv` (all three variants sum each output in column order):
/// `out[i] = sum_j weights[j*col_stride + i] * x[j]`.
pub fn sgemv(out: &mut [f32], weights: F32s<'_>, rows: usize, cols: usize, col_stride: usize, x: &[f32]) {
    let out = &mut out[..rows];
    out.fill(0.0);
    for (j, &xj) in x[..cols].iter().enumerate() {
        let w = weights.from(j * col_stride);
        for (i, y) in out.iter_mut().enumerate() {
            *y += w.at(i) * xj;
        }
    }
}

/// `sparse_sgemv8x4`: float weights in 8x4 blocks listed by `idx`.
pub fn sparse_sgemv8x4(out: &mut [f32], w: F32s<'_>, idx: I32s<'_>, rows: usize, x: &[f32]) {
    out[..rows].fill(0.0);
    let mut p = 0;
    let mut wp = 0;
    for i in (0..rows).step_by(8) {
        let cols = idx.at(p);
        p += 1;
        for _ in 0..cols {
            let pos = idx.at(p) as usize;
            p += 1;
            let xj = [x[pos], x[pos + 1], x[pos + 2], x[pos + 3]];
            let y = &mut out[i..i + 8];
            for (c, &xc) in xj.iter().enumerate() {
                for (k, yk) in y.iter_mut().enumerate() {
                    *yk += w.at(wp + 8 * c + k) * xc;
                }
            }
            wp += 32;
        }
    }
}

/// `cgemv8x4`: int8 weights in 8x4 blocks, inputs quantized to int8 and
/// converted back to float; the four products of a block row are summed
/// in float before they are added to the output.
pub fn cgemv8x4(out: &mut [f32], w: &[u8], scale: F32s<'_>, rows: usize, cols: usize, x_in: &[f32]) {
    let mut x = [0i8; MAX_INPUTS];
    out[..rows].fill(0.0);
    for (q, &v) in x.iter_mut().zip(&x_in[..cols]) {
        *q = quantize(v);
    }
    let mut wp = 0;
    for i in (0..rows).step_by(8) {
        for j in (0..cols).step_by(4) {
            let xj = [x[j] as f32, x[j + 1] as f32, x[j + 2] as f32, x[j + 3] as f32];
            let blk = &w[wp..wp + 32];
            for k in 0..8 {
                let b = &blk[4 * k..4 * k + 4];
                out[i + k] += (b[0] as i8 as f32) * xj[0]
                    + (b[1] as i8 as f32) * xj[1]
                    + (b[2] as i8 as f32) * xj[2]
                    + (b[3] as i8 as f32) * xj[3];
            }
            wp += 32;
        }
    }
    for (i, y) in out[..rows].iter_mut().enumerate() {
        *y *= scale.at(i);
    }
}

/// `sparse_cgemv8x4`: like `cgemv8x4` over the blocks listed by `idx`, but
/// the four products of a block row are summed as integers.
pub fn sparse_cgemv8x4(
    out: &mut [f32],
    w: &[u8],
    idx: I32s<'_>,
    scale: F32s<'_>,
    rows: usize,
    cols: usize,
    x_in: &[f32],
) {
    let mut x = [0i8; MAX_INPUTS];
    out[..rows].fill(0.0);
    for (q, &v) in x.iter_mut().zip(&x_in[..cols]) {
        *q = quantize(v);
    }
    let mut p = 0;
    let mut wp = 0;
    for i in (0..rows).step_by(8) {
        let colblocks = idx.at(p);
        p += 1;
        for _ in 0..colblocks {
            let pos = idx.at(p) as usize;
            p += 1;
            let xj = [x[pos] as i32, x[pos + 1] as i32, x[pos + 2] as i32, x[pos + 3] as i32];
            let blk = &w[wp..wp + 32];
            for k in 0..8 {
                let b = &blk[4 * k..4 * k + 4];
                let s = (b[0] as i8 as i32) * xj[0]
                    + (b[1] as i8 as i32) * xj[1]
                    + (b[2] as i8 as i32) * xj[2]
                    + (b[3] as i8 as i32) * xj[3];
                out[i + k] += s as f32;
            }
            wp += 32;
        }
    }
    for (i, y) in out[..rows].iter_mut().enumerate() {
        *y *= scale.at(i);
    }
}

/// `lpcnet_exp2`: 2^x from a cubic polynomial of the fraction, with the
/// integer part added to the exponent bits.
#[inline]
pub fn lpcnet_exp2(x: f32) -> f32 {
    let integer = c_int((x as f64).floor());
    if integer < -50 {
        return 0.0;
    }
    let frac = x - integer as f32;
    let f = 0.99992522f32 + frac * (0.69583354f32 + frac * (0.22606716f32 + 0.078024523f32 * frac));
    let bits = f.to_bits().wrapping_add((integer << 23) as u32) & 0x7fff_ffff;
    f32::from_bits(bits)
}

/// `lpcnet_exp`.
#[inline]
pub fn lpcnet_exp(x: f32) -> f32 {
    lpcnet_exp2(x * 1.44269504f32)
}

/// libopus `MIN32`/`MAX32` for floats: `a < b ? a : b`.
#[inline(always)]
pub(crate) fn min32(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `MAX32`: `a > b ? a : b`.
#[inline(always)]
pub(crate) fn max32(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// `tanh_approx`: a rational approximation, clamped to [-1, 1].
#[inline]
pub fn tanh_approx(x: f32) -> f32 {
    const N0: f32 = 952.52801514;
    const N1: f32 = 96.39235687;
    const N2: f32 = 0.60863042;
    const D0: f32 = 952.72399902;
    const D1: f32 = 413.36801147;
    const D2: f32 = 11.88600922;
    let x2 = x * x;
    let num = (N2 * x2 + N1) * x2 + N0;
    let den = (D2 * x2 + D1) * x2 + D0;
    let num = num * x / den;
    max32(-1.0, min32(1.0, num))
}

/// `sigmoid_approx`.
#[inline]
pub fn sigmoid_approx(x: f32) -> f32 {
    0.5 + 0.5 * tanh_approx(0.5 * x)
}

/// `softmax` (which only exponentiates).
pub fn softmax(y: &mut [f32], x: &[f32]) {
    for (o, &v) in y.iter_mut().zip(x) {
        *o = lpcnet_exp(v);
    }
}

/// `vec_tanh`.
pub fn vec_tanh(y: &mut [f32], x: &[f32]) {
    for (o, &v) in y.iter_mut().zip(x) {
        *o = tanh_approx(v);
    }
}

/// `vec_sigmoid`.
pub fn vec_sigmoid(y: &mut [f32], x: &[f32]) {
    for (o, &v) in y.iter_mut().zip(x) {
        *o = sigmoid_approx(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp2_matches_known_values() {
        assert_eq!(lpcnet_exp2(-60.0), 0.0);
        assert!((lpcnet_exp2(3.0) - 8.0).abs() < 1e-3);
        assert!((lpcnet_exp(1.0) - core::f32::consts::E).abs() < 1e-3);
    }

    #[test]
    fn quantize_wraps_like_gcc() {
        assert_eq!(quantize(1.0), 127);
        assert_eq!(quantize(-1.0), -127);
        // 127*2 = 254 -> (int8)254 = -2
        assert_eq!(quantize(2.0), -2);
        assert_eq!(quantize(f32::NAN), 0);
    }
}
