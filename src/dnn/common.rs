//! CELT helpers that the neural network code calls, with the operation
//! order of libopus's float build (no FLOAT_APPROX): `celt_fir` and the
//! `mathops.h` functions. `celt_inner_prod` and `celt_pitch_xcorr` are
//! [`crate::pitch::inner_prod`] and [`crate::pitch::pitch_xcorr`].

#[cfg(not(feature = "std"))]
use crate::compat::Math;

use crate::Arch;
use crate::pitch::xcorr_kernel;

/// The largest order of [`celt_fir`].
const MAX_FIR_ORDER: usize = 24;

/// `celt_fir_c` at `arch`: `y[i] = x[i] + sum_j num[ord-1-j] * x[i+j-ord]`
/// with the history in `x[..ord]` and the input in `x[ord..ord + n]`. The
/// outputs in whole groups of four are `xcorr_kernel` at `arch` (libopus's
/// SSE kernel at the x86 levels), the rest sums in order.
pub fn celt_fir(x: &[f32], num: &[f32], y: &mut [f32], n: usize, ord: usize, arch: Arch) {
    assert!(ord <= MAX_FIR_ORDER);
    let mut rnum = [0.0f32; MAX_FIR_ORDER];
    for (r, &v) in rnum[..ord].iter_mut().zip(num[..ord].iter().rev()) {
        *r = v;
    }
    let rnum = &rnum[..ord];
    let groups = n & !3;
    for i in (0..groups).step_by(4) {
        let sum: [f32; 4] = x[ord + i..ord + i + 4].try_into().unwrap();
        let out = match arch {
            Arch::Scalar => xcorr_kernel::<4, false>(rnum, &x[i..], sum),
            _ => xcorr_kernel::<4, true>(rnum, &x[i..], sum),
        };
        y[i..i + 4].copy_from_slice(&out);
    }
    for i in groups..n {
        let mut sum = x[ord + i];
        for (j, &r) in rnum.iter().enumerate() {
            sum += r * x[i + j];
        }
        y[i] = sum;
    }
}

/// `celt_log2` (float build): `(float)(1.442695040888963387*log(x))`.
#[inline]
pub fn celt_log2(x: f32) -> f32 {
    (1.442695040888963387f64 * (x as f64).ln()) as f32
}

/// `celt_log`.
#[inline]
pub fn celt_log(x: f32) -> f32 {
    celt_log2(x) * 0.6931471805599453f32
}

/// `celt_exp2` (float build): `(float)exp(0.6931471805599453094*(x))`.
#[inline]
pub fn celt_exp2(x: f32) -> f32 {
    (0.6931471805599453094f64 * x as f64).exp() as f32
}

/// `celt_exp`.
#[inline]
pub fn celt_exp(x: f32) -> f32 {
    celt_exp2(x * 1.4426950408889634f32)
}

/// `celt_cos_norm2`: cos(pi/2 x) from an even polynomial.
#[inline]
pub fn celt_cos_norm2(x: f32) -> f32 {
    let mut x = (x as f64 - 4.0 * (0.25 * (x + 1.0) as f64).floor()) as f32;
    let gt = (x > 1.0) as i32;
    let output_sign = 1 - 2 * gt;
    x -= (2 * gt) as f32;
    let x2 = x * x;
    output_sign as f32
        * (9.999999403953552246093750000000e-01f32
            + x2 * (-1.233698248863220214843750000000000f32
                + x2 * (2.536507546901702880859375000000e-01f32
                    + x2 * (-2.08106283098459243774414062500e-02f32
                        + x2 * 8.581906440667808055877685546875e-04f32))))
}

/// `celt_sin`: `celt_cos_norm2((2.f/PI)*(x) - 1.0f)` with PI a double.
#[inline]
pub fn celt_sin(x: f32) -> f32 {
    celt_cos_norm2(((2.0f32 as f64 / 3.1415926535897931f64) * x as f64 - 1.0) as f32)
}

/// `log2_approx` (`dnn/common.h`).
#[inline]
pub fn log2_approx(x: f32) -> f32 {
    let mut bits = x.to_bits() as i32;
    let integer = (bits >> 23) - 127;
    bits = bits.wrapping_sub(integer << 23);
    let mut frac = f32::from_bits(bits as u32) - 1.5;
    frac = -0.41445418f32 + frac * (0.95909232f32 + frac * (-0.33951290f32 + frac * 0.16541097f32));
    1.0 + integer as f32 + frac
}
