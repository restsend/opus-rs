#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::fixedvec::FixedVec;

/// Capacity for pitch-search working buffers (x_lp4 / y_lp4 / xcorr). Bounded by
/// `COMBFILTER_MAXPERIOD = 1024` and the maximum pitch lag.
const PITCH_BUF_MAX: usize = 1024;

use crate::celt_lpc::{autocorr, lpc};

/// libopus `celt_inner_prod` at `arch`: [`inner_prods`] of one vector.
#[inline]
pub fn inner_prod(x: &[f32], y: &[f32], n: usize, arch: crate::Arch) -> f32 {
    inner_prods(x, [y], n, arch)[0]
}

/// [`inner_prod`] of the products `product(i)`, i < n.
pub(crate) fn inner_prod_by(n: usize, arch: crate::Arch, product: impl Fn(usize) -> f32) -> f32 {
    let mut j = 0;
    let mut acc = 0.0f32;
    if arch != crate::Arch::Scalar {
        let mut s = [0.0f32; 4];
        while j + 4 <= n {
            for (l, s) in s.iter_mut().enumerate() {
                *s += product(j + l);
            }
            j += 4;
        }
        acc = (s[0] + s[2]) + (s[1] + s[3]);
    }
    for j in j..n {
        acc += product(j);
    }
    acc
}

/// libopus `dual_inner_prod` at `arch`: [`inner_prods`] of two vectors.
pub fn dual_inner_prod(
    x: &[f32],
    y1: &[f32],
    y2: &[f32],
    n: usize,
    arch: crate::Arch,
) -> (f32, f32) {
    let [xy1, xy2] = inner_prods(x, [y1, y2], n, arch);
    (xy1, xy2)
}

/// libopus `celt_pitch_xcorr`: `xcorr[i]` is the inner product of `x[..len]`
/// and `y[i..i + len]`.
pub fn pitch_xcorr(
    x: &[f32],
    y: &[f32],
    xcorr: &mut [f32],
    len: usize,
    max_pitch: usize,
    arch: crate::Arch,
) {
    // Every lag window reads y[i..i+len], so the contract is
    // y.len() >= len + max_pitch - 1; clamp so that direct callers with a
    // short `y` degrade gracefully (issue #27 deep scan). In-repo callers
    // always satisfy the contract and are unaffected.
    let max_pitch = if len > 0 {
        max_pitch.min(y.len() + 1 - len)
    } else {
        0
    };
    xcorr_lags(&x[..len], y, &mut xcorr[..max_pitch], arch);
}

/// `out[i]` = the inner product of `x` and `y[i..i + x.len()]`
/// (`celt_pitch_xcorr` at `arch`). At the Avx2 level it is
/// [`xcorr_lags_avx2`]. Otherwise the kernels give each lag a vector lane,
/// up to 64 lags at a time (with AVX2 on 8-lane vectors); at the Sse2 and
/// Sse4_1 levels the lags in whole groups of four take [`xcorr_kernel`]'s
/// SSE arithmetic and the last `max_pitch % 4` libopus's
/// `celt_inner_prod_sse`.
fn xcorr_lags(x: &[f32], y: &[f32], out: &mut [f32], arch: crate::Arch) {
    let sse = arch != crate::Arch::Scalar;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(avx2) = arch.avx2() {
        use fearless_simd::Simd;
        return avx2.vectorize(|| xcorr_lags_avx2(avx2, x, y, out));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(avx2) = crate::arch::cpu_avx2() {
        use fearless_simd::Simd;
        return match sse {
            true => avx2.vectorize(|| xcorr_lags_blocks::<64, true>(x, y, out, arch)),
            false => avx2.vectorize(|| xcorr_lags_blocks::<64, false>(x, y, out, arch)),
        };
    }
    match sse {
        true => xcorr_lags_blocks::<32, true>(x, y, out, arch),
        false => xcorr_lags_blocks::<32, false>(x, y, out, arch),
    }
}

/// libopus `celt_pitch_xcorr_avx2`: eight lags at a time
/// ([`xcorr_kernel_avx`]); the last `max_pitch % 8` lags are
/// `celt_inner_prod_sse`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
fn xcorr_lags_avx2(simd: fearless_simd::Avx2, x: &[f32], y: &[f32], out: &mut [f32]) {
    let groups = out.len() & !7;
    for g in (0..groups).step_by(8) {
        out[g..g + 8].copy_from_slice(&xcorr_kernel_avx(simd, x, &y[g..]));
    }
    for (k, o) in out.iter_mut().enumerate().skip(groups) {
        *o = inner_prod(x, &y[k..], x.len(), crate::Arch::Avx2);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    /// libopus `xcorr_kernel_avx`: the inner products of `x` with
    /// `y[k..k + x.len()]` for eight lags `k`, each from eight fused
    /// multiply-add sums of every eighth product (the last partial block
    /// with zeros in the missing lanes), added as
    /// ((s0 + s4) + (s1 + s5)) + ((s2 + s6) + (s3 + s7)).
    fn xcorr_kernel_avx(simd: Avx2, x: &[f32], y: &[f32]) -> [f32; 8] {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        use fearless_simd::{Avx2, f32x8, prelude::*};
        let len = x.len();
        let y = &y[..len + 7];
        let load = |v: &[f32]| -> __m256 { f32x8::from_slice(simd, &v[..8]).into() };
        let mut acc = [_mm256_setzero_ps(); 8];
        let mut i = 0;
        while i + 8 <= len {
            let x0 = load(&x[i..]);
            let yw = &y[i..][..15];
            for (k, acc) in acc.iter_mut().enumerate() {
                *acc = _mm256_fmadd_ps(x0, load(&yw[k..]), *acc);
            }
            i += 8;
        }
        if i != len {
            let r = len - i;
            let mut xb = [0.0f32; 8];
            xb[..r].copy_from_slice(&x[i..]);
            let x0 = load(&xb);
            for (k, acc) in acc.iter_mut().enumerate() {
                let mut yb = [0.0f32; 8];
                yb[..r].copy_from_slice(&y[i + k..i + k + r]);
                *acc = _mm256_fmadd_ps(x0, load(&yb), *acc);
            }
        }
        // [s0+s4 s1+s5 s2+s6 s3+s7] of lag k in the low half, of lag k + 4
        // in the high half; two horizontal adds finish each sum.
        let fold = |a: __m256, b: __m256| {
            _mm256_add_ps(
                _mm256_permute2f128_ps::<0x20>(a, b),
                _mm256_permute2f128_ps::<0x31>(a, b),
            )
        };
        let s = _mm256_hadd_ps(
            _mm256_hadd_ps(fold(acc[0], acc[4]), fold(acc[1], acc[5])),
            _mm256_hadd_ps(fold(acc[2], acc[6]), fold(acc[3], acc[7])),
        );
        let s: f32x8<Avx2> = s.simd_into(simd);
        s.to_array()
    }
);

/// [`xcorr_lags`] in blocks of `B` lags, then the rest in one block each of
/// 32, 16, 8 and 4 lags and the last few. Its helpers take no closures so
/// they inline into the caller's target features.
#[inline(always)]
fn xcorr_lags_blocks<const B: usize, const SSE: bool>(
    x: &[f32],
    y: &[f32],
    out: &mut [f32],
    arch: crate::Arch,
) {
    let mut blocks = out.chunks_exact_mut(B);
    let mut y = y;
    for o in &mut blocks {
        y = xcorr_block::<B, SSE>(x, y, o);
    }
    let mut rest = blocks.into_remainder();
    for size in [32, 16, 8, 4] {
        if size < B && rest.len() >= size {
            let (o, r) = rest.split_at_mut(size);
            y = xcorr_block::<B, SSE>(x, y, o);
            rest = r;
        }
    }
    if SSE {
        let n = x.len();
        let ys = |k: usize| &y[k..];
        match rest.len() {
            1 => rest.copy_from_slice(&inner_prods(x, [ys(0)], n, arch)),
            2 => rest.copy_from_slice(&inner_prods(x, [ys(0), ys(1)], n, arch)),
            3 => rest.copy_from_slice(&inner_prods(x, [ys(0), ys(1), ys(2)], n, arch)),
            _ => {}
        }
    } else if !rest.is_empty() {
        xcorr_block::<B, SSE>(x, y, rest);
    }
}

/// The lags of `out` from the start of `y`; returns `y` from the next lag.
#[inline(always)]
fn xcorr_block<'a, const B: usize, const SSE: bool>(
    x: &[f32],
    y: &'a [f32],
    out: &mut [f32],
) -> &'a [f32] {
    let w = &y[..x.len() + out.len() - 1];
    let o = out;
    match o.len() {
        1 => o.copy_from_slice(&xcorr_kernel::<1, SSE>(x, w, [0.0; 1])),
        2 => o.copy_from_slice(&xcorr_kernel::<2, SSE>(x, w, [0.0; 2])),
        3 => o.copy_from_slice(&xcorr_kernel::<3, SSE>(x, w, [0.0; 3])),
        4 => o.copy_from_slice(&xcorr_kernel::<4, SSE>(x, w, [0.0; 4])),
        8 => o.copy_from_slice(&xcorr_kernel::<8, SSE>(x, w, [0.0; 8])),
        16 => o.copy_from_slice(&xcorr_kernel::<16, SSE>(x, w, [0.0; 16])),
        32 => o.copy_from_slice(&xcorr_kernel::<32, SSE>(x, w, [0.0; 32])),
        _ => o.copy_from_slice(&xcorr_kernel::<B, SSE>(x, w, [0.0; B])),
    }
    &y[o.len()..]
}

/// libopus `xcorr_kernel`: `sum[k]` plus the products `x[j] * y[j + k]`.
/// The scalar kernel adds them in order. The SSE kernel
/// (`xcorr_kernel_sse`) adds the even-`j` products in order to `sum[k]`
/// and the odd-`j` ones in order to a second sum from zero, then adds the
/// two; lanes do not interact, so any number of lags give the four-lag
/// kernel's values.
#[inline(always)]
pub(crate) fn xcorr_kernel<const L: usize, const SSE: bool>(
    x: &[f32],
    y: &[f32],
    sum: [f32; L],
) -> [f32; L] {
    let y = &y[..x.len() + L - 1];
    let mut even = sum;
    if !SSE {
        for (&xj, w) in x.iter().zip(y.windows(L)) {
            let w: &[f32; L] = w.try_into().unwrap();
            for k in 0..L {
                even[k] += xj * w[k];
            }
        }
        return even;
    }
    let mut odd = [0.0f32; L];
    let mut pairs = x.chunks_exact(2);
    for (p, xp) in (&mut pairs).enumerate() {
        let w: &[f32; L] = (&y[2 * p..2 * p + L]).try_into().unwrap();
        let v: &[f32; L] = (&y[2 * p + 1..2 * p + 1 + L]).try_into().unwrap();
        for k in 0..L {
            even[k] += xp[0] * w[k];
            odd[k] += xp[1] * v[k];
        }
    }
    if let [xj] = pairs.remainder() {
        let j = x.len() - 1;
        let w: &[f32; L] = (&y[j..j + L]).try_into().unwrap();
        for k in 0..L {
            even[k] += xj * w[k];
        }
    }
    for k in 0..L {
        even[k] += odd[k];
    }
    even
}

/// `K` inner products of `x[..n]` with the `ys`, interleaved so they run in
/// parallel. At the scalar level each sums its products in order
/// (`celt_inner_prod_c`). At the x86 levels each is libopus's
/// `celt_inner_prod_sse` (`dual_inner_prod_sse` is two of them): four
/// running sums of every fourth product, added as (s0 + s2) + (s1 + s3),
/// then the last `n % 4` products in order.
#[inline(always)]
fn inner_prods<const K: usize>(
    x: &[f32],
    ys: [&[f32]; K],
    n: usize,
    arch: crate::Arch,
) -> [f32; K] {
    let x = &x[..n];
    let ys = ys.map(|y| &y[..n]);
    let mut acc = [0.0f32; K];
    let mut j = 0;
    if arch != crate::Arch::Scalar {
        let mut lanes = [[0.0f32; 4]; K];
        j = n & !3;
        for (b, xb) in x[..j].chunks_exact(4).enumerate() {
            for k in 0..K {
                let yb = &ys[k][4 * b..4 * b + 4];
                for l in 0..4 {
                    lanes[k][l] += xb[l] * yb[l];
                }
            }
        }
        for k in 0..K {
            let s = lanes[k];
            acc[k] = (s[0] + s[2]) + (s[1] + s[3]);
        }
    }
    for j in j..n {
        for k in 0..K {
            acc[k] += x[j] * ys[k][j];
        }
    }
    acc
}

fn celt_fir5(x: &mut [f32], num: &[f32], n: usize) {
    let mut mem = [0.0f32; 5];

    let num0 = num[0];
    let num1 = num[1];
    let num2 = num[2];
    let num3 = num[3];
    let num4 = num[4];

    for i in 0..n {
        let mut sum = x[i];
        sum += num0 * mem[0];
        sum += num1 * mem[1];
        sum += num2 * mem[2];
        sum += num3 * mem[3];
        sum += num4 * mem[4];

        mem[4] = mem[3];
        mem[3] = mem[2];
        mem[2] = mem[1];
        mem[1] = mem[0];
        mem[0] = x[i];

        x[i] = sum;
    }
}

pub fn pitch_downsample(
    x: &[&[f32]],
    x_lp: &mut [f32],
    len: usize,
    c: usize,
    factor: usize,
    arch: crate::Arch,
) {
    let offset = factor / 2;

    if x_lp.len() < len {
        return;
    }

    // x_lp[i] sums, channel by channel, the x_k[factor * i - offset..=
    // factor * i + offset] that lie inside x_k.
    let lp = &mut x_lp[1..len];
    lp.fill(0.0);
    for x_k in &x[..c] {
        let windows = x_k[factor - offset..]
            .windows(2 * offset + 1)
            .step_by(factor);
        for (o, w) in lp.iter_mut().zip(windows) {
            *o += 0.25 * w[0] + 0.25 * w[2 * offset] + 0.5 * w[offset];
        }
    }

    pitch_downsample_boundary(x, x_lp, c, offset);
    whiten_downsampled(x_lp, len, arch);
}

/// Whitening of the decimated signal, shared by every `pitch_downsample`
/// path (the tail of libopus `pitch_downsample`, pitch.c:192-229): a 4-tap
/// LPC from the decimated autocorrelation, lag-windowed, with a zero added,
/// applied as a 5-tap FIR so the pitch search sees a flat spectrum with a
/// sharp correlation peak.
fn whiten_downsampled(x_lp: &mut [f32], len: usize, arch: crate::Arch) {
    let mut ac = [0.0f32; 5];
    autocorr(&x_lp[0..len], &mut ac, None, 0, 4, len, arch);

    // Noise floor -40 dB.
    ac[0] *= 1.0001;

    // Lag windowing.
    for i in 1..=4 {
        let f = 0.008 * (i as f32);
        ac[i] -= ac[i] * f * f;
    }

    let mut lpc_coeffs = [0.0f32; 4];
    lpc(&mut lpc_coeffs, &ac, 4);

    let mut tmp = 1.0f32;
    for coeff in lpc_coeffs.iter_mut() {
        tmp *= 0.9;
        *coeff *= tmp;
    }

    // Add a zero.
    let c1 = 0.8f32;
    let lpc2 = [
        lpc_coeffs[0] + c1,
        lpc_coeffs[1] + c1 * lpc_coeffs[0],
        lpc_coeffs[2] + c1 * lpc_coeffs[1],
        lpc_coeffs[3] + c1 * lpc_coeffs[2],
        c1 * lpc_coeffs[3],
    ];

    celt_fir5(x_lp, &lpc2, len);
}

#[inline]
fn pitch_downsample_boundary(x: &[&[f32]], x_lp: &mut [f32], c: usize, offset: usize) {
    {
        let mut val = 0.0f32;
        for k in 0..c {
            let x_k = x[k];

            let idx_offset = offset;
            let idx_0 = 0;
            if idx_offset < x_k.len() {
                val += 0.25 * x_k[idx_offset] + 0.5 * x_k[idx_0];
            }
        }
        x_lp[0] = val;
    }
}

#[inline(always)]
fn find_best_pitch(
    xcorr: &[f32],
    y: &[f32],
    len: usize,
    max_pitch: usize,
    best_pitch: &mut [usize; 2],
) {
    let mut best_num = [-1.0f32, -1.0f32];
    let mut best_den = [0.0f32, 0.0f32];

    best_pitch[0] = 0;
    best_pitch[1] = 1;

    // libopus accumulates `Syy` sequentially in scalar float
    // (pitch.c find_best_pitch); SIMD tree reductions reorder the adds
    // and change the last bits, which can flip a near-tied pitch choice.
    let mut syy = 1.0f32;
    for yv in &y[..len] {
        syy += yv * yv;
    }

    for i in 0..max_pitch {
        if xcorr[i] > 0.0 {
            // The 1e-12 scaling is libopus's (pitch.c find_best_pitch): it
            // keeps `num` in a safe squaring range, and because it is not a
            // power of two the rounding of every comparison below depends
            // on it being applied first.
            let mut xcorr16 = xcorr[i];
            xcorr16 *= 1e-12;
            let num = xcorr16 * xcorr16;
            if num * best_den[1] > best_num[1] * syy {
                if num * best_den[0] > best_num[0] * syy {
                    best_num[1] = best_num[0];
                    best_den[1] = best_den[0];
                    best_pitch[1] = best_pitch[0];
                    best_num[0] = num;
                    best_den[0] = syy;
                    best_pitch[0] = i;
                } else {
                    best_num[1] = num;
                    best_den[1] = syy;
                    best_pitch[1] = i;
                }
            }
        }
        syy += y[i + len] * y[i + len] - y[i] * y[i];
        if syy < 1.0 {
            syy = 1.0;
        }
    }
}

pub fn pitch_search(
    x_lp: &[f32],
    y: &[f32],
    mut len: usize,
    mut max_pitch: usize,
    arch: crate::Arch,
) -> usize {
    let mut best_pitch = [0, 0];

    max_pitch >>= 1;
    len >>= 1;
    let lag = len + max_pitch;
    let len4 = len >> 1;
    let lag4 = lag >> 1;

    const MAX_LEN4: usize = 512;
    const MAX_LAG4: usize = 1024;
    const MAX_PITCH: usize = 512;

    let mut x_lp4_stack = [0.0f32; MAX_LEN4];
    let mut y_lp4_stack = [0.0f32; MAX_LAG4];
    let mut xcorr_stack = [0.0f32; MAX_PITCH];

    let x_lp4: &mut [f32];
    let y_lp4: &mut [f32];
    let xcorr: &mut [f32];

    if len4 <= MAX_LEN4 && lag4 <= MAX_LAG4 && max_pitch <= MAX_PITCH {
        x_lp4 = &mut x_lp4_stack[..len4];
        y_lp4 = &mut y_lp4_stack[..lag4];
        xcorr = &mut xcorr_stack[..max_pitch];
    } else {
        return pitch_search_heap(x_lp, y, len << 1, max_pitch << 1, arch);
    }

    for j in 0..len4 {
        x_lp4[j] = x_lp[2 * j];
    }
    for j in 0..lag4 {
        y_lp4[j] = y[2 * j];
    }

    pitch_xcorr(x_lp4, y_lp4, xcorr, len >> 1, max_pitch >> 1, arch);

    find_best_pitch(xcorr, y_lp4, len >> 1, max_pitch >> 1, &mut best_pitch);

    fine_xcorr(x_lp, y, &mut xcorr[..max_pitch], len, best_pitch, arch);

    find_best_pitch(xcorr, y, len, max_pitch, &mut best_pitch);

    let mut offset = 0;
    if best_pitch[0] > 0 && best_pitch[0] < max_pitch - 1 {
        let a = xcorr[best_pitch[0] - 1];
        let b = xcorr[best_pitch[0]];
        let c = xcorr[best_pitch[0] + 1];
        if (c - a) > 0.7 * (b - a) {
            offset = 1;
        } else if (a - c) > 0.7 * (b - c) {
            offset = -1;
        }
    }

    ((2 * best_pitch[0]) as isize).wrapping_sub(offset as isize) as usize
}

fn pitch_search_heap(
    x_lp: &[f32],
    y: &[f32],
    mut len: usize,
    mut max_pitch: usize,
    arch: crate::Arch,
) -> usize {
    let mut best_pitch = [0, 0];

    max_pitch >>= 1;
    len >>= 1;
    let lag = len + max_pitch;

    let mut x_lp4: FixedVec<f32, PITCH_BUF_MAX> = FixedVec::from_value(0.0f32, len >> 1);
    let mut y_lp4: FixedVec<f32, PITCH_BUF_MAX> = FixedVec::from_value(0.0f32, lag >> 1);
    let mut xcorr: FixedVec<f32, PITCH_BUF_MAX> = FixedVec::from_value(0.0f32, max_pitch);

    for j in 0..(len >> 1) {
        x_lp4[j] = x_lp[2 * j];
    }
    for j in 0..(lag >> 1) {
        y_lp4[j] = y[2 * j];
    }

    pitch_xcorr(&x_lp4, &y_lp4, &mut xcorr, len >> 1, max_pitch >> 1, arch);

    find_best_pitch(&xcorr, &y_lp4, len >> 1, max_pitch >> 1, &mut best_pitch);

    fine_xcorr(x_lp, y, &mut xcorr[..max_pitch], len, best_pitch, arch);

    find_best_pitch(&xcorr, y, len, max_pitch, &mut best_pitch);

    let mut offset = 0;
    if best_pitch[0] > 0 && best_pitch[0] < max_pitch - 1 {
        let a = xcorr[best_pitch[0] - 1];
        let b = xcorr[best_pitch[0]];
        let c = xcorr[best_pitch[0] + 1];
        if (c - a) > 0.7 * (b - a) {
            offset = 1;
        } else if (a - c) > 0.7 * (b - c) {
            offset = -1;
        }
    }

    ((2 * best_pitch[0]) as isize).wrapping_sub(offset as isize) as usize
}

/// The full-rate correlations within 2 lags of the two coarse candidates,
/// floored at -1, and 0 elsewhere (pitch.c:373-386).
fn fine_xcorr(
    x_lp: &[f32],
    y: &[f32],
    xcorr: &mut [f32],
    len: usize,
    best_pitch: [usize; 2],
    arch: crate::Arch,
) {
    xcorr.fill(0.0);
    for b in best_pitch {
        let lo = (2 * b).saturating_sub(2).min(xcorr.len());
        let hi = (2 * b + 3).min(xcorr.len());
        for i in lo..hi {
            xcorr[i] = inner_prod(x_lp, &y[i..], len, arch);
        }
    }
    for v in xcorr.iter_mut() {
        if *v < -1.0 {
            *v = -1.0;
        }
    }
}

/// pitch.c:447-450 (float build).
fn compute_pitch_gain(xy: f32, xx: f32, yy: f32) -> f32 {
    xy / (1.0 + xx * yy).sqrt()
}

static SECOND_CHECK: [usize; 16] = [0, 0, 3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2];

pub fn remove_doubling(
    x: &[f32],
    mut max_period: usize,
    mut min_period: usize,
    mut n: usize,
    t0_ptr: &mut usize,
    mut prev_period: usize,
    prev_gain: f32,
    arch: crate::Arch,
) -> f32 {
    let min_period0 = min_period;
    max_period /= 2;
    min_period /= 2;
    *t0_ptr /= 2;
    prev_period /= 2;
    n /= 2;

    let x_target = &x[max_period..];

    if *t0_ptr >= max_period {
        *t0_ptr = max_period - 1;
    }

    let mut t = *t0_ptr;
    let t0 = *t0_ptr;

    const MAX_YY_SIZE: usize = 1024;
    let mut yy_lookup_buf = [0.0f32; MAX_YY_SIZE];
    let yy_lookup = &mut yy_lookup_buf[..=max_period];

    let [xx, xy] = inner_prods(x_target, [x_target, &x[max_period - t0..]], n, arch);

    yy_lookup[0] = xx;
    let mut yy_curr = xx;
    for i in 1..=max_period {
        yy_curr = yy_curr + x[max_period - i] * x[max_period - i]
            - x[max_period + n - i] * x[max_period + n - i];
        // pitch.c:486: the running sum stays unclamped.
        yy_lookup[i] = yy_curr.max(0.0);
    }

    let mut best_xy = xy;
    let mut best_yy = yy_lookup[t0];
    let mut g = compute_pitch_gain(best_xy, xx, best_yy);
    let g0 = g;

    // The candidate periods do not depend on the decisions, so their
    // correlations are computed first, four candidates at a time.
    let mut periods = [(0usize, 0usize); 14];
    let mut count = 0;
    for k in 2..=15 {
        let t1 = (2 * t0 + k) / (2 * k);
        if t1 < min_period {
            break;
        }
        let t1b = if k == 2 {
            if t1 + t0 > max_period { t0 } else { t0 + t1 }
        } else {
            (2 * SECOND_CHECK[k] * t0 + k) / (2 * k)
        };
        periods[count] = (t1, t1b);
        count += 1;
    }
    let mut xys = [(0.0f32, 0.0f32); 14];
    for start in (0..count).step_by(4) {
        let lag = |c: usize| {
            let (t1, t1b) = periods[(start + c / 2).min(count - 1)];
            &x[max_period - if c % 2 == 0 { t1 } else { t1b }..]
        };
        let r = inner_prods(x_target, core::array::from_fn::<_, 8, _>(lag), n, arch);
        for c in 0..4.min(count - start) {
            xys[start + c] = (r[2 * c], r[2 * c + 1]);
        }
    }

    for (i, (&(t1, t1b), &(xy_a, xy_b))) in periods[..count].iter().zip(&xys).enumerate() {
        let k = i + 2;
        let xy_new = 0.5 * (xy_a + xy_b);
        let yy_new = 0.5 * (yy_lookup[t1] + yy_lookup[t1b]);
        let g1 = compute_pitch_gain(xy_new, xx, yy_new);

        let mut cont = 0.0f32;
        if (t1 as i32 - prev_period as i32).abs() <= 1 {
            cont = prev_gain;
        } else if (t1 as i32 - prev_period as i32).abs() <= 2 && 5 * k * k < t0 {
            cont = 0.5 * prev_gain;
        }

        let mut thresh = (0.7 * g0 - cont).max(0.3);
        if t1 < 3 * min_period {
            thresh = (0.85 * g0 - cont).max(0.4);
        } else if t1 < 2 * min_period {
            thresh = (0.9 * g0 - cont).max(0.5);
        }

        if g1 > thresh {
            best_xy = xy_new;
            best_yy = yy_new;
            t = t1;
            g = g1;
        }
    }

    // A short period with as strong a correlation at 5/8 or 6/8 of it is
    // not a pitch (pitch.c:538-547).
    if t < 2 * min_period {
        let (t1, t2) = (t * 5 / 8, t * 6 / 8);
        let [xy1, xy2] = inner_prods(
            x_target,
            [&x[max_period - t1..], &x[max_period - t2..]],
            n,
            arch,
        );
        let g1 = compute_pitch_gain(xy1, xx, yy_lookup[t1]);
        let g2 = compute_pitch_gain(xy2, xx, yy_lookup[t2]);
        if g1 >= g || g2 >= g {
            g = 0.0;
        }
    }

    best_xy = best_xy.max(0.0);
    let pg = if best_yy <= best_xy {
        1.0f32
    } else {
        best_xy / (best_yy + 1.0)
    };

    let xcorr_res = inner_prods(
        x_target,
        core::array::from_fn::<_, 3, _>(|k| &x[max_period + 1 - t - k..]),
        n,
        arch,
    );

    let mut offset = 0;
    if (xcorr_res[2] - xcorr_res[0]) > 0.7 * (xcorr_res[1] - xcorr_res[0]) {
        offset = 1;
    } else if (xcorr_res[0] - xcorr_res[2]) > 0.7 * (xcorr_res[1] - xcorr_res[2]) {
        offset = -1;
    }

    let pg = pg.min(g);
    *t0_ptr = (2 * t as i32 + offset).max(min_period0 as i32) as usize;

    pg
}
