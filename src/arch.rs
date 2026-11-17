//! The architecture level of a codec state (libopus `arch`), and the CPU
//! features of the kernels that give the scalar results at every level.

/// The kernels whose arithmetic a codec state reproduces, chosen when the
/// state is created as libopus's `opus_select_arch()` chooses them.
/// `Scalar` is libopus built without intrinsics. On x86-64, where libopus
/// presumes SSE and SSE2, the other levels are libopus's: `Sse2` (arch 2),
/// `Sse4_1` (3) and `Avx2` (4, AVX2 and FMA). Elsewhere the level is
/// `Scalar`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Arch {
    #[default]
    Scalar,
    Sse2,
    Sse4_1,
    Avx2,
}

impl Arch {
    /// The level of this CPU, or the one [`crate::fuzzing::force_arch`]
    /// set when this CPU has it.
    pub fn select() -> Arch {
        let detected = detect();
        #[cfg(feature = "fuzzing")]
        if let Some(forced) = forced() {
            return forced.min(detected);
        }
        detected
    }

    /// The AVX2 token at the `Avx2` level.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[inline]
    pub fn avx2(self) -> Option<fearless_simd::Avx2> {
        match self {
            Arch::Avx2 => cpu_avx2(),
            _ => None,
        }
    }
}

/// libopus's `opus_select_arch()` on x86-64 (SSE4.1 from fearless_simd's
/// SSE4.2 level; without `std` no detection runs and SSE2 is presumed).
fn detect() -> Arch {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        match fearless_simd::Level::try_detect() {
            Some(level) if level.as_avx2().is_some() => Arch::Avx2,
            Some(level) if level.as_sse4_2().is_some() => Arch::Sse4_1,
            _ => Arch::Sse2,
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        Arch::Scalar
    }
}

/// The AVX2 token when the CPU has AVX2 (runtime detection needs `std`),
/// for kernels that give the scalar results.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) fn cpu_avx2() -> Option<fearless_simd::Avx2> {
    fearless_simd::Level::try_detect().and_then(|level| level.as_avx2())
}

/// The NEON token (NEON is in the aarch64 baseline).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn cpu_neon() -> Option<fearless_simd::Neon> {
    fearless_simd::Level::baseline().as_neon()
}

/// The simd128 token, in builds with the `simd128` target feature.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
pub(crate) fn cpu_simd128() -> Option<fearless_simd::WasmSimd128> {
    fearless_simd::Level::baseline().as_wasm_simd128()
}

#[cfg(feature = "fuzzing")]
static FORCED: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

/// The level codec states created afterwards take (`None`: the CPU's).
#[cfg(feature = "fuzzing")]
pub fn force_arch(arch: Option<Arch>) {
    let v = arch.map_or(0, |a| a as u8 + 1);
    FORCED.store(v, core::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "fuzzing")]
fn forced() -> Option<Arch> {
    const ALL: [Arch; 4] = [Arch::Scalar, Arch::Sse2, Arch::Sse4_1, Arch::Avx2];
    match FORCED.load(core::sync::atomic::Ordering::Relaxed) {
        0 => None,
        v => Some(ALL[v as usize - 1]),
    }
}
