//! The float build's `celt_log2` and `celt_exp2` (libopus mathops.h:346-347,
//! without FLOAT_APPROX): the natural logarithm and exponential in double
//! precision, rounded to float.

#[cfg(not(feature = "std"))]
use crate::compat::Math;

#[allow(clippy::excessive_precision)]
pub(crate) fn celt_log2(x: f32) -> f32 {
    (1.442695040888963387 * f64::from(x).ln()) as f32
}

#[allow(clippy::excessive_precision)]
pub(crate) fn celt_exp2(x: f32) -> f32 {
    (0.6931471805599453094 * f64::from(x)).exp() as f32
}
