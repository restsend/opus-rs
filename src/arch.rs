//! The architecture level of a codec state (libopus `arch`), and the CPU
//! features of the kernels that give the scalar results at every level.

/// The kernels whose arithmetic a codec state reproduces, chosen when the
/// state is created as libopus's `opus_select_arch()` chooses them.
/// `Scalar` is libopus built without intrinsics (arch 0). On x86 and x86-64
/// the other levels are libopus's: `Sse2` (arch 2), `Sse4_1` (3) and `Avx2`
/// (4). On other targets the level is `Scalar`. Kernel functions take a
/// level at most [`Arch::native`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Arch {
    #[default]
    Scalar,
    Sse2,
    Sse4_1,
    Avx2,
}

impl Arch {
    /// The level codec states take: this CPU's ([`Arch::native`]). With the
    /// `fuzzing` feature it is the one [`crate::fuzzing::force_arch`] set,
    /// else the one the environment variable `OPUS_RS_ARCH` names
    /// (`scalar`, `sse2`, `sse4_1`, `avx2` or `native`; unset: `scalar`, the
    /// level of the scalar reference), at most this CPU's.
    pub fn select() -> Arch {
        #[cfg(feature = "fuzzing")]
        return forced().min(Self::native());
        #[cfg(not(feature = "fuzzing"))]
        Self::native()
    }

    /// This CPU's level.
    pub fn native() -> Arch {
        detect()
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

/// libopus's `opus_select_arch()` on x86 and x86-64, from the CPU level
/// fearless_simd detects (with `std`) or the target's baseline (without):
/// - `Sse2` needs fearless_simd's SSE2 token (SSE2 and FXSR) and a target
///   with SSE2 in its baseline (x86-64, i686). Elsewhere the level is
///   `Scalar`: libopus takes arch 0 without SSE, arch 1 with SSE (the SSE
///   float kernels with the C PVQ search and C neural kernels, a level
///   opus-rs does not have) and detects arch 2 to 4 in an i586 build.
/// - `Sse4_1` needs SSE4.1, as in libopus. Without `std` it needs the
///   `sse4.1` target feature.
/// - `Avx2` needs fearless_simd's AVX2 token, the x86-64-v3 set: AVX2 and
///   FMA, which libopus checks, and BMI1, BMI2, CMPXCHG16B, F16C, FXSR,
///   LZCNT, MOVBE, POPCNT and XSAVE, which it does not. A CPU with AVX, AVX2
///   and FMA that lacks one of these (a masked virtual CPU) is `Sse4_1`
///   where libopus takes arch 4.
/// - Without `std` the level is the target's baseline: on a CPU above it,
///   libopus detects a higher level.
fn detect() -> Arch {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let level = cpu_level();
        if level.as_avx2().is_some() {
            Arch::Avx2
        } else if level.as_sse2().is_none() {
            Arch::Scalar
        } else if has_sse4_1() {
            Arch::Sse4_1
        } else {
            Arch::Sse2
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        Arch::Scalar
    }
}

/// The CPU's level: detected at runtime with `std`, the target's baseline
/// without. A 32-bit x86 target without SSE2 in its baseline (i586) has
/// none: there fearless_simd's SSE2 kernels give wrong results in release
/// builds (`op_pvq_search_sse2` reads past its blocks), and the scalar code
/// computes on the x87 FPU.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn cpu_level() -> fearless_simd::Level {
    if cfg!(target_feature = "sse2") {
        fearless_simd::Level::try_detect().unwrap_or(fearless_simd::Level::baseline())
    } else {
        fearless_simd::Level::baseline()
    }
}

/// Whether the CPU has SSE4.1 (CPUID.1:ECX bit 19, libopus's arch 3 test).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn has_sse4_1() -> bool {
    #[cfg(feature = "std")]
    return std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(not(feature = "std"))]
    cfg!(target_feature = "sse4.1")
}

/// The AVX2 token when the CPU has fearless_simd's AVX2 level, for kernels
/// that give the scalar results.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) fn cpu_avx2() -> Option<fearless_simd::Avx2> {
    cpu_level().as_avx2()
}

/// The SSE2 token; every level above `Scalar` has it.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) fn cpu_sse2() -> Option<fearless_simd::Sse2> {
    cpu_level().as_sse2()
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

/// The level codec states created afterwards take (`None`: the one
/// `OPUS_RS_ARCH` names).
#[cfg(feature = "fuzzing")]
pub fn force_arch(arch: Option<Arch>) {
    let v = arch.map_or(0, |a| a as u8 + 1);
    FORCED.store(v, core::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "fuzzing")]
fn forced() -> Arch {
    const ALL: [Arch; 4] = [Arch::Scalar, Arch::Sse2, Arch::Sse4_1, Arch::Avx2];
    match FORCED.load(core::sync::atomic::Ordering::Relaxed) {
        0 => match std::env::var("OPUS_RS_ARCH").as_deref() {
            Ok("native") => Arch::native(),
            Ok("sse2") => Arch::Sse2,
            Ok("sse4_1") => Arch::Sse4_1,
            Ok("avx2") => Arch::Avx2,
            _ => Arch::Scalar,
        },
        v => ALL[v as usize - 1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every level above `Scalar` has the token its SSE2 kernels take.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn levels_above_scalar_have_sse2() {
        if Arch::native() > Arch::Scalar {
            assert!(cpu_sse2().is_some());
        }
        if Arch::native() == Arch::Avx2 {
            assert!(cpu_avx2().is_some());
        }
    }

    /// The level follows libopus's CPUID tests up to `Sse4_1`: SSE2 for
    /// arch 2, SSE4.1 for arch 3.
    #[cfg(all(feature = "std", target_feature = "sse2"))]
    #[test]
    fn sse_levels_follow_cpuid() {
        let libopus = if !std::arch::is_x86_feature_detected!("sse2") {
            Arch::Scalar
        } else if std::arch::is_x86_feature_detected!("sse4.1") {
            Arch::Sse4_1
        } else {
            Arch::Sse2
        };
        assert_eq!(Arch::native().min(Arch::Sse4_1), libopus);
    }
}
