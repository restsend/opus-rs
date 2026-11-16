//! no_std / std compatibility shims — zero external dependencies.
//!
//! Provides:
//! * [`x86_has_avx`] / [`x86_has_avx2`] / [`x86_has_avx_fma`] / [`x86_has_avx2_fma`]
//!   — x86 SIMD availability probes. Under the
//!   `std` feature these use the OS-backed `is_x86_feature_detected!`; without `std`
//!   they fall back to compile-time `cfg!(target_feature = ...)`.

#[cfg(feature = "fuzzing")]
use core::sync::atomic::Ordering;

// ---------------------------------------------------------------------------
// No-std slice sorting (`alloc` provides `[T]::sort*` as inherent methods; since
// this crate does not depend on `alloc`, we provide small insertion sorts here).
// ---------------------------------------------------------------------------

/// Sort a slice by a comparator (insertion sort). Intended for the small slices
/// used in this crate (energy bands, NLSF coefficients, median-of-3/5); O(n²) is
/// fine because `n` is tiny.
pub fn sort_by<T>(slice: &mut [T], mut cmp: impl FnMut(&T, &T) -> core::cmp::Ordering) {
    for i in 1..slice.len() {
        let mut j = i;
        while j > 0 && cmp(&slice[j - 1], &slice[j]) == core::cmp::Ordering::Greater {
            slice.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// Sort a slice in ascending order (`Ord` version). See [`sort_by`].
pub fn sort<T: Ord>(slice: &mut [T]) {
    sort_by(slice, |a, b| a.cmp(b));
}

// ---------------------------------------------------------------------------
// Zeroed scratch buffers
// ---------------------------------------------------------------------------

/// Runs `f` on a zeroed stack buffer of `n <= 512` elements, zeroing only the
/// next power of two from 32, so short buffers cost little to initialize.
#[inline(always)]
pub fn with_scratch<T: Copy + Default, R>(n: usize, f: impl FnOnce(&mut [T]) -> R) -> R {
    let z = T::default();
    match n {
        0..=32 => f(&mut [z; 32][..n]),
        33..=64 => f(&mut [z; 64][..n]),
        65..=128 => f(&mut [z; 128][..n]),
        129..=256 => f(&mut [z; 256][..n]),
        _ => f(&mut [z; 512][..n]),
    }
}

// ---------------------------------------------------------------------------
// x86 SIMD feature detection
// ---------------------------------------------------------------------------

/// Runtime switch for the x86 AVX/FMA kernels, so the `fuzzing` harnesses can
/// compare them with the scalar paths in one process.
#[cfg(feature = "fuzzing")]
pub static SIMD_ENABLED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);

#[inline]
#[allow(dead_code)]
fn simd_enabled() -> bool {
    #[cfg(feature = "fuzzing")]
    return SIMD_ENABLED.load(Ordering::Relaxed);
    #[cfg(not(feature = "fuzzing"))]
    true
}

/// Whether the `avx` x86 feature is usable.
///
/// * With the `std` feature: OS-backed runtime detection via CPUID.
/// * Without `std`: compile-time detection (`cfg!(target_feature = "avx")`); build
///   with e.g. `RUSTFLAGS="-C target-feature=+avx"` to enable it.
#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub fn x86_has_avx() -> bool {
    simd_enabled() && std::arch::is_x86_feature_detected!("avx")
}
#[cfg(not(all(feature = "std", target_arch = "x86_64")))]
#[allow(dead_code)]
pub fn x86_has_avx() -> bool {
    cfg!(target_feature = "avx")
}

/// Whether the `avx2` x86 feature is usable. See [`x86_has_avx`].
#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub fn x86_has_avx2() -> bool {
    simd_enabled() && std::arch::is_x86_feature_detected!("avx2")
}
#[cfg(not(all(feature = "std", target_arch = "x86_64")))]
#[allow(dead_code)]
pub fn x86_has_avx2() -> bool {
    cfg!(target_feature = "avx2")
}

/// Whether both `avx` and `fma` are usable. Gates kernels compiled with
/// `#[target_feature(enable = "avx,fma")]`: AVX alone is not enough, since
/// Sandy Bridge / Ivy Bridge (and some VMs) have AVX without FMA and would fault
/// with SIGILL on the first `vfmadd`. See [`x86_has_avx`].
#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub fn x86_has_avx_fma() -> bool {
    simd_enabled()
        && std::arch::is_x86_feature_detected!("avx")
        && std::arch::is_x86_feature_detected!("fma")
}
#[cfg(not(all(feature = "std", target_arch = "x86_64")))]
#[allow(dead_code)]
pub fn x86_has_avx_fma() -> bool {
    cfg!(all(target_feature = "avx", target_feature = "fma"))
}

/// Whether both `avx2` and `fma` are usable. Gates kernels compiled with
/// `#[target_feature(enable = "avx2,fma")]`; AVX2 does not architecturally imply
/// FMA. See [`x86_has_avx_fma`].
#[cfg(all(feature = "std", target_arch = "x86_64"))]
pub fn x86_has_avx2_fma() -> bool {
    simd_enabled()
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
}
#[cfg(not(all(feature = "std", target_arch = "x86_64")))]
#[allow(dead_code)]
pub fn x86_has_avx2_fma() -> bool {
    cfg!(all(target_feature = "avx2", target_feature = "fma"))
}

// ---------------------------------------------------------------------------
// Float math (no_std needs libm for transcendentals / rounding)
// ---------------------------------------------------------------------------

#[cfg(not(feature = "std"))]
#[cfg(not(feature = "libm"))]
compile_error!(
    "opus-rs no_std builds require the `libm` feature (or the `std` feature). \
     Set features = [\"libm\"] (or re-enable default-features)."
);

/// Extension trait giving `f32`/`f64` the usual math methods (`sqrt`, `sin`, ...)
/// under `no_std` by forwarding to `libm`. Under the `std` feature the inherent
/// methods take priority and this trait is simply unused.
///
/// Bring it into scope with `use crate::compat::Math;` in any module that calls
/// these methods so the same `x.sqrt()` syntax works in both build modes.
#[cfg(all(not(feature = "std"), feature = "libm"))]
#[allow(dead_code)]
pub trait Math {
    fn sqrt(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn abs(self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn log(self, base: Self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn atan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn cbrt(self) -> Self;
    fn hypot(self, other: Self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
impl Math for f32 {
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    fn floor(self) -> Self {
        libm::floorf(self)
    }
    fn ceil(self) -> Self {
        libm::ceilf(self)
    }
    fn round(self) -> Self {
        libm::roundf(self)
    }
    fn trunc(self) -> Self {
        libm::truncf(self)
    }
    fn fract(self) -> Self {
        self - libm::truncf(self)
    }
    fn abs(self) -> Self {
        libm::fabsf(self)
    }
    fn ln(self) -> Self {
        libm::logf(self)
    }
    fn log2(self) -> Self {
        libm::log2f(self)
    }
    fn log10(self) -> Self {
        libm::log10f(self)
    }
    fn log(self, base: Self) -> Self {
        libm::logf(self) / libm::logf(base)
    }
    fn exp(self) -> Self {
        libm::expf(self)
    }
    fn exp2(self) -> Self {
        libm::exp2f(self)
    }
    fn powf(self, n: Self) -> Self {
        libm::powf(self, n)
    }
    fn powi(self, n: i32) -> Self {
        libm::powf(self, n as Self)
    }
    fn sin(self) -> Self {
        libm::sinf(self)
    }
    fn cos(self) -> Self {
        libm::cosf(self)
    }
    fn tan(self) -> Self {
        libm::tanf(self)
    }
    fn atan2(self, other: Self) -> Self {
        libm::atan2f(self, other)
    }
    fn atan(self) -> Self {
        libm::atanf(self)
    }
    fn asin(self) -> Self {
        libm::asinf(self)
    }
    fn acos(self) -> Self {
        libm::acosf(self)
    }
    fn sinh(self) -> Self {
        libm::sinhf(self)
    }
    fn cosh(self) -> Self {
        libm::coshf(self)
    }
    fn tanh(self) -> Self {
        libm::tanhf(self)
    }
    fn cbrt(self) -> Self {
        libm::cbrtf(self)
    }
    fn hypot(self, other: Self) -> Self {
        libm::hypotf(self, other)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fmaf(self, a, b)
    }
}

#[cfg(all(not(feature = "std"), feature = "libm"))]
impl Math for f64 {
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    fn floor(self) -> Self {
        libm::floor(self)
    }
    fn ceil(self) -> Self {
        libm::ceil(self)
    }
    fn round(self) -> Self {
        libm::round(self)
    }
    fn trunc(self) -> Self {
        libm::trunc(self)
    }
    fn fract(self) -> Self {
        self - libm::trunc(self)
    }
    fn abs(self) -> Self {
        libm::fabs(self)
    }
    fn ln(self) -> Self {
        libm::log(self)
    }
    fn log2(self) -> Self {
        libm::log2(self)
    }
    fn log10(self) -> Self {
        libm::log10(self)
    }
    fn log(self, base: Self) -> Self {
        libm::log(self) / libm::log(base)
    }
    fn exp(self) -> Self {
        libm::exp(self)
    }
    fn exp2(self) -> Self {
        libm::exp2(self)
    }
    fn powf(self, n: Self) -> Self {
        libm::pow(self, n)
    }
    fn powi(self, n: i32) -> Self {
        libm::pow(self, n as Self)
    }
    fn sin(self) -> Self {
        libm::sin(self)
    }
    fn cos(self) -> Self {
        libm::cos(self)
    }
    fn tan(self) -> Self {
        libm::tan(self)
    }
    fn atan2(self, other: Self) -> Self {
        libm::atan2(self, other)
    }
    fn atan(self) -> Self {
        libm::atan(self)
    }
    fn asin(self) -> Self {
        libm::asin(self)
    }
    fn acos(self) -> Self {
        libm::acos(self)
    }
    fn sinh(self) -> Self {
        libm::sinh(self)
    }
    fn cosh(self) -> Self {
        libm::cosh(self)
    }
    fn tanh(self) -> Self {
        libm::tanh(self)
    }
    fn cbrt(self) -> Self {
        libm::cbrt(self)
    }
    fn hypot(self, other: Self) -> Self {
        libm::hypot(self, other)
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fma(self, a, b)
    }
}

#[cfg(all(test, feature = "std", target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn fma_probes_require_both_features() {
        let fma = std::arch::is_x86_feature_detected!("fma");
        assert_eq!(x86_has_avx_fma(), x86_has_avx() && fma);
        assert_eq!(x86_has_avx2_fma(), x86_has_avx2() && fma);
    }
}
