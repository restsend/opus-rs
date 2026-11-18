//! Differential kernel fuzzer: opus-rs DSP functions against the same
//! functions in the scalar libopus reference, on arbitrary inputs
//! within each function's documented domain.
//!
//! Each kernel runs twice in opus-rs, with the fuzzing SIMD switch off
//! (`scalar`) and on (`simd`); the switch has no effect yet
//! (fuzz/DIFFERENTIAL.md).
//!
//! Oracles per kernel:
//! - exact: integer kernels and element-wise float kernels must match bit for bit.
//! - tol: float kernels whose summation order differs from libopus must
//!   match within `OFZ_KTOL` (relative, default 1e-5).
//! - pvq: the pulse vector must have K pulses, and at the reference's level
//!   it must equal libopus's (op_pvq_search_sse2 follows the CPU's rcp/rsqrt,
//!   so other levels may pick other pulses).

use crate::common::*;
use arbitrary::{Arbitrary, Unstructured};
use opusref as c;

macro_rules! fail_k {
    ($sig:expr, $($arg:tt)*) => {
        return Ok(Err(Finding { oracle: "kernel", sig: $sig, msg: format!($($arg)*) }))
    };
}

/// The libFuzzer entry point's body.
pub fn fuzz(data: &[u8]) {
    let mut u = Unstructured::new(data);
    let Ok(which) = u8::arbitrary(&mut u) else {
        return;
    };
    let r = guarded(|| run(which, &mut u).unwrap_or(Ok(())));
    report("diff_kernels", data, r);
}

type R = Result<Result<(), Finding>, arbitrary::Error>;

fn floats(u: &mut Unstructured, n: usize, scale: f32) -> arbitrary::Result<Vec<f32>> {
    (0..n)
        .map(|_| Ok(i16::arbitrary(u)? as f32 / 32768.0 * scale))
        .collect()
}

fn both<T>(mut f: impl FnMut() -> T) -> (T, T) {
    set_simd(false);
    let a = f();
    set_simd(true);
    let b = f();
    (a, b)
}

fn exact(name: &str, rs: &[f32], c: &[f32]) -> Result<(), Finding> {
    if let Some(i) = (0..rs.len()).find(|&i| rs[i].to_bits() != c[i].to_bits()) {
        fail!(
            "kernel",
            format!("{name} exact"),
            "{name}: [{i}] opus-rs {:e} libopus {:e}",
            rs[i],
            c[i]
        );
    }
    Ok(())
}

fn tol(name: &str, rs: &[f32], c: &[f32]) -> Result<(), Finding> {
    let t = env_f64("OFZ_KTOL", 1e-5) as f32;
    let (d, peak, i) = max_diff(rs, c);
    if !(d <= t * peak) {
        fail!(
            "kernel",
            format!("{name} tol"),
            "{name}: [{i}] opus-rs {:e} libopus {:e} (diff {d:e}, peak {peak:e})",
            rs[i],
            c[i]
        );
    }
    Ok(())
}

macro_rules! check2 {
    ($name:expr, $how:ident, $pair:expr, $c:expr) => {{
        let (sc, si) = $pair;
        if let Err(f) = $how(concat!($name, "/scalar"), &sc, &$c) {
            return Ok(Err(f));
        }
        if let Err(f) = $how(concat!($name, "/simd"), &si, &$c) {
            return Ok(Err(f));
        }
    }};
}

fn objective(x: &[f32], y: &[i32]) -> f64 {
    let rxy: f64 = x
        .iter()
        .zip(y)
        .map(|(a, &b)| a.abs() as f64 * b.unsigned_abs() as f64)
        .sum();
    let ryy: f64 = y.iter().map(|&b| (b as f64) * (b as f64)).sum();
    if ryy == 0.0 { 0.0 } else { rxy * rxy / ryy }
}

fn run(which: u8, u: &mut Unstructured) -> R {
    let mode = opus_rs::modes::default_mode();
    match which % 10 {
        // haar1: element-wise, so bit-exact.
        0 => {
            let stride = 1usize << (u8::arbitrary(u)? % 4);
            let n0 = 2 * (1 + u8::arbitrary(u)? as usize % 128);
            let x = floats(u, n0 * stride, 1.0)?;
            let mut cx = x.clone();
            unsafe { c::haar1(cx.as_mut_ptr(), n0 as i32, stride as i32) };
            ijon_set(0x100 | (n0 >= 16 && stride == 1) as u64);
            check2!(
                "haar1",
                exact,
                both(|| {
                    let mut v = x.clone();
                    opus_rs::bands::haar1(&mut v, n0, stride);
                    v
                }),
                cx
            );
        }
        // exp_rotation: element-wise rotations with computed coefficients.
        1 => {
            // Callers pass N = B blocks of N/B coefficients each.
            let stride = 1 + u8::arbitrary(u)? as usize % 8;
            let len = stride * (1 + u8::arbitrary(u)? as usize % (176 / stride));
            let k = 1 + u8::arbitrary(u)? as i32 % 64;
            let spread = 1 + u8::arbitrary(u)? as i32 % 3;
            let dir = if bool::arbitrary(u)? { 1 } else { -1 };
            if 2 * k >= len as i32 {
                return Ok(Ok(()));
            }
            let x = floats(u, len, 1.0)?;
            let mut cx = x.clone();
            unsafe { c::exp_rotation(cx.as_mut_ptr(), len as i32, dir, stride as i32, k, spread) };
            check2!(
                "exp_rotation",
                exact,
                both(|| {
                    let mut v = x.clone();
                    opus_rs::pvq::exp_rotation(&mut v, len, dir, stride, k, spread);
                    v
                }),
                cx
            );
        }
        // PVQ search: the search must find a valid vector at least as good as libopus's.
        2 => {
            let n = 2 + u8::arbitrary(u)? as usize % 175;
            let k = 1 + u8::arbitrary(u)? as i32 % 128;
            let x = floats(u, n, 1.0)?;
            if x.iter().all(|&v| v == 0.0) {
                return Ok(Ok(()));
            }
            // libopus's alg_quant allocates N+3 entries: op_pvq_search_sse2
            // clears and writes them in groups of four.
            let mut cx = x.clone();
            cx.resize(n + 3, 0.0);
            let mut cy = vec![0i32; n + 3];
            // The fixed-point reference has no float search wrapper.
            #[cfg(not(feature = "fixed"))]
            unsafe {
                c::ref_op_pvq_search(cx.as_mut_ptr(), cy.as_mut_ptr(), k, n as i32, c::ref_arch())
            };
            #[cfg(feature = "fixed")]
            unsafe {
                c::op_pvq_search_c(cx.as_mut_ptr(), cy.as_mut_ptr(), k, n as i32, 0)
            };
            cy.truncate(n);
            // A run at the reference's level gives libopus's pulses (libopus
            // arch: 0 C, 1 SSE, 2 SSE2, 3 SSE4.1, 4 AVX2).
            // The fixed-point reference searches in fixed point: no exact check.
            #[cfg(not(feature = "fixed"))]
            let lref = Some(match unsafe { c::ref_arch() } {
                0 => opus_rs::Arch::Scalar,
                1 | 2 => opus_rs::Arch::Sse2,
                3 => opus_rs::Arch::Sse4_1,
                _ => opus_rs::Arch::Avx2,
            });
            #[cfg(feature = "fixed")]
            let lref = None;
            let (lc, ln) = both(opus_rs::Arch::select);
            let (sc, si) = both(|| {
                let mut y = vec![0i32; n + 8];
                opus_rs::pvq::pvq_search(&x, &mut y, k, n, opus_rs::Arch::select());
                y.truncate(n);
                y
            });
            ijon_set(0x200 | (n.min(64) as u64) << 1 | (si != cy) as u64);
            for (tag, y, level) in [("scalar", &sc, lc), ("simd", &si, ln)] {
                let pulses: i32 = y.iter().map(|v| v.abs()).sum();
                if pulses != k {
                    fail_k!(
                        format!("pvq_search/{tag} pulses"),
                        "pvq_search/{tag}: {pulses} pulses, K={k}, N={n}"
                    );
                }
                if Some(level) == lref && *y != cy {
                    fail_k!(
                        format!("pvq_search/{tag} exact"),
                        "pvq_search/{tag}: pulses differ from libopus at the same level (N={n} K={k}, objective {} vs {})",
                        objective(&x, y),
                        objective(&x, &cy)
                    );
                }
            }
        }
        // celt_fir: same filter, different summation order.
        3 => {
            let ord = 1 + u8::arbitrary(u)? as usize % 24;
            let n = 1 + u16::arbitrary(u)? as usize % 960;
            let num = floats(u, ord, 1.0)?;
            let x = floats(u, n, 1.0)?;
            // libopus reads `ord` samples of history before x; opus-rs treats it as zero.
            let mut cx = vec![0f32; ord];
            cx.extend_from_slice(&x);
            let mut cy = vec![0f32; n];
            unsafe {
                c::celt_fir_c(
                    cx.as_ptr().add(ord),
                    num.as_ptr(),
                    cy.as_mut_ptr(),
                    n as i32,
                    ord as i32,
                    0,
                )
            };
            check2!(
                "celt_fir",
                tol,
                both(|| {
                    let mut y = vec![0f32; n];
                    opus_rs::celt_lpc::celt_fir(&x, &num, &mut y, n, ord);
                    y
                }),
                cy
            );
        }
        // silk_sum_sqr_shift: fixed point, bit-exact.
        4 => {
            let len = u16::arbitrary(u)? as usize % 1024;
            let x: Vec<i16> = (0..len)
                .map(|_| i16::arbitrary(u))
                .collect::<Result<_, _>>()?;
            let (mut ce, mut cs) = (0i32, 0i32);
            unsafe { c::silk_sum_sqr_shift(&mut ce, &mut cs, x.as_ptr(), len as i32) };
            let (a, b) = both(|| {
                let (mut e, mut s) = (0i32, 0i32);
                opus_rs::silk::sigproc_fix::silk_sum_sqr_shift(&mut e, &mut s, &x, len);
                (e, s)
            });
            ijon_max(0x400, len as u64);
            for (tag, v) in [("scalar", a), ("simd", b)] {
                if v != (ce, cs) {
                    fail_k!(
                        format!("silk_sum_sqr_shift/{tag}"),
                        "silk_sum_sqr_shift/{tag}: (energy, shift) {v:?}, libopus {:?}, len {len}",
                        (ce, cs)
                    );
                }
            }
        }
        // comb_filter (CELT pre/post-filter): element-wise products and sums
        // in libopus order, so bit-exact with SIMD off.
        5 => {
            let t0 = 15 + u16::arbitrary(u)? as usize % 1010;
            let t1 = 15 + u16::arbitrary(u)? as usize % 1010;
            let n = 1 + u16::arbitrary(u)? as usize % 960;
            let g0 = u8::arbitrary(u)? as f32 / 256.0 * 0.75;
            let g1 = u8::arbitrary(u)? as f32 / 256.0 * 0.75;
            let tap0 = u8::arbitrary(u)? as i32 % 3;
            let tap1 = u8::arbitrary(u)? as i32 % 3;
            // libopus requires N >= overlap.
            let overlap = if bool::arbitrary(u)? && n >= 120 {
                120
            } else {
                0
            };
            let hist = 1026;
            let x = floats(u, hist + n, 1.0)?;
            let mut cy = vec![0f32; hist + n];
            let mut cx = x.clone();
            unsafe {
                c::comb_filter(
                    cy.as_mut_ptr().add(hist),
                    cx.as_mut_ptr().add(hist),
                    t0 as i32,
                    t1 as i32,
                    n as i32,
                    g0,
                    g1,
                    tap0,
                    tap1,
                    mode.window.as_ptr(),
                    overlap as i32,
                    c::ref_arch(),
                )
            };
            ijon_set(0x500 | (tap0 * 3 + tap1) as u64);
            let (sc, si) = both(|| {
                let mut y = vec![0f32; hist + n];
                opus_rs::fuzzing::comb_filter(
                    &mut y,
                    &x,
                    hist,
                    hist,
                    t0,
                    t1,
                    n,
                    g0,
                    g1,
                    tap0,
                    tap1,
                    mode.window,
                    overlap,
                );
                y
            });
            if let Err(mut f) = exact("comb_filter/scalar", &sc[hist..], &cy[hist..]) {
                f.msg += &format!(
                    " (t0 {t0} t1 {t1} n {n} g0 {g0} g1 {g1} taps {tap0}/{tap1} overlap {overlap})"
                );
                return Ok(Err(f));
            }
            if let Err(f) = tol("comb_filter/simd", &si[hist..], &cy[hist..]) {
                return Ok(Err(f));
            }
        }
        // SILK fixed-point macros: bit-exact over their domains.
        6 => {
            let op = u8::arbitrary(u)? % 22;
            let a = i32::arbitrary(u)?;
            let b = i32::arbitrary(u)?;
            let cc = i32::arbitrary(u)?;
            let sh = b.rem_euclid(31);
            use opus_rs::silk::macros as m;
            let (name, rs, rf): (&str, i32, i32) = unsafe {
                match op {
                    0 => (
                        "silk_lshift_sat32",
                        m::silk_lshift_sat32(a, sh),
                        c::ref_silk_lshift_sat32(a, sh),
                    ),
                    1 => (
                        "silk_add_sat32",
                        m::silk_add_sat32(a, b),
                        c::ref_silk_add_sat32(a, b),
                    ),
                    2 => (
                        "silk_sub_sat32",
                        m::silk_sub_sat32(a, b),
                        c::ref_silk_sub_sat32(a, b),
                    ),
                    3 => (
                        "silk_add_pos_sat32",
                        m::silk_add_pos_sat32(a & i32::MAX, b & i32::MAX),
                        c::ref_silk_add_pos_sat32(a & i32::MAX, b & i32::MAX),
                    ),
                    4 => (
                        "silk_rshift_round",
                        m::silk_rshift_round(a, 1 + sh),
                        c::ref_silk_rshift_round(a, 1 + sh),
                    ),
                    5 => (
                        "silk_smulwb",
                        m::silk_smulwb(a, b),
                        c::ref_silk_smulwb(a, b),
                    ),
                    6 => (
                        "silk_smulwt",
                        m::silk_smulwt(a, b),
                        c::ref_silk_smulwt(a, b),
                    ),
                    7 => (
                        "silk_smulww",
                        m::silk_smulww(a, b),
                        c::ref_silk_smulww(a, b),
                    ),
                    8 => (
                        "silk_smulbb",
                        m::silk_smulbb(a, b),
                        c::ref_silk_smulbb(a, b),
                    ),
                    9 => ("silk_smmul", m::silk_smmul(a, b), c::ref_silk_smmul(a, b)),
                    10 => (
                        "silk_smlawb",
                        m::silk_smlawb(a, b, cc),
                        c::ref_silk_smlawb(a, b, cc),
                    ),
                    11 => (
                        "silk_smlawt",
                        m::silk_smlawt(a, b, cc),
                        c::ref_silk_smlawt(a, b, cc),
                    ),
                    12 => (
                        "silk_smlaww",
                        m::silk_smlaww(a, b, cc),
                        c::ref_silk_smlaww(a, b, cc),
                    ),
                    13 => (
                        "silk_smlabb",
                        m::silk_smlabb(a, b, cc),
                        c::ref_silk_smlabb(a, b, cc),
                    ),
                    14 => (
                        "silk_limit_32",
                        m::silk_limit_32(a, b, cc),
                        c::ref_silk_limit_32(a, b, cc),
                    ),
                    15 => ("silk_sat16", m::silk_sat16(a), c::ref_silk_sat16(a)),
                    16 if a != 0 => ("silk_clz32", m::silk_clz32(a), c::ref_silk_clz32(a)),
                    17 if a > 0 => (
                        "silk_sqrt_approx",
                        m::silk_sqrt_approx(a),
                        c::ref_silk_sqrt_approx(a),
                    ),
                    18 if b != 0 && b != i32::MIN && a != i32::MIN => {
                        // Shifts of 32 and more are undefined in C (silk_LSHIFT_SAT32).
                        let q = cc.rem_euclid(29);
                        (
                            "silk_div32_varq",
                            m::silk_div32_varq(a, b, q),
                            c::ref_silk_div32_varq(a, b, q),
                        )
                    }
                    19 if b != 0 && b != i32::MIN => {
                        let q = 1 + cc.rem_euclid(31);
                        (
                            "silk_inverse32_varq",
                            m::silk_inverse32_varq(b, q),
                            c::ref_silk_inverse32_varq(b, q),
                        )
                    }
                    20 => (
                        "silk_lin2log",
                        opus_rs::silk::lin2log::silk_lin2log(a),
                        c::silk_lin2log(a),
                    ),
                    21 if (0..3967).contains(&(a & 0xfff)) => {
                        let v = a & 0xfff;
                        (
                            "silk_log2lin",
                            opus_rs::silk::log2lin::silk_log2lin(v),
                            c::silk_log2lin(v),
                        )
                    }
                    _ => return Ok(Ok(())),
                }
            };
            ijon_set(0x600 | op as u64);
            if rs != rf {
                fail_k!(
                    name.to_string(),
                    "{name}({a}, {b}, {cc}): opus-rs {rs}, libopus {rf}"
                );
            }
        }
        // LPC from autocorrelation.
        7 => {
            let p = 1 + u8::arbitrary(u)? as usize % 24;
            let x = floats(u, 256, 1.0)?;
            let mut ac = vec![0f32; p + 1];
            unsafe {
                c::_celt_autocorr(
                    x.as_ptr(),
                    ac.as_mut_ptr(),
                    core::ptr::null(),
                    0,
                    p as i32,
                    256,
                    c::ref_arch(),
                )
            };
            ac[0] *= 1.0001;
            let mut cl = vec![0f32; p];
            unsafe { c::_celt_lpc(cl.as_mut_ptr(), ac.as_ptr(), p as i32) };
            check2!(
                "celt_lpc",
                exact,
                both(|| {
                    let mut l = vec![0f32; p];
                    opus_rs::celt_lpc::lpc(&mut l, &ac, p);
                    l
                }),
                cl
            );
        }
        // Autocorrelation, optionally windowed.
        8 => {
            let lag = 1 + u8::arbitrary(u)? as usize % 24;
            let n = 240 + u16::arbitrary(u)? as usize % 720;
            let overlap = if bool::arbitrary(u)? { 120 } else { 0 };
            let x = floats(u, n, 1.0)?;
            let mut ca = vec![0f32; lag + 1];
            let w = if overlap > 0 {
                mode.window.as_ptr()
            } else {
                core::ptr::null()
            };
            unsafe {
                c::_celt_autocorr(
                    x.as_ptr(),
                    ca.as_mut_ptr(),
                    w,
                    overlap as i32,
                    lag as i32,
                    n as i32,
                    c::ref_arch(),
                )
            };
            check2!(
                "autocorr",
                tol,
                both(|| {
                    let mut a = vec![0f32; lag + 1];
                    opus_rs::celt_lpc::autocorr(
                        &x,
                        &mut a,
                        (overlap > 0).then_some(mode.window),
                        overlap,
                        lag,
                        n,
                        opus_rs::Arch::select(),
                    );
                    a
                }),
                ca
            );
        }
        _ => {}
    }
    Ok(Ok(()))
}
