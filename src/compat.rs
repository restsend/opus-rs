//! no_std / std compatibility shims — zero external dependencies.

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
// Fuzzing switch
// ---------------------------------------------------------------------------

/// Switch the `fuzzing` harnesses flip to compare SIMD and scalar paths.
/// opus-rs has no target-specific kernels, so it has no effect.
#[cfg(feature = "fuzzing")]
pub static SIMD_ENABLED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(true);

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
