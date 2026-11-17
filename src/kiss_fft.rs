use core::f64::consts::PI;

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::fixedvec::FixedVec;

pub const MAXFACTORS: usize = 8;

/// Maximum FFT size (`nfft`) used by the MDCT plans. Bounded by `MAX_N4 = 480`.
pub const KISS_MAX_N: usize = 480;

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct KissCpx {
    pub r: f32,
    pub i: f32,
}

impl crate::fixedvec::ConstDefault for KissCpx {
    const DEFAULT: Self = Self::new(0.0, 0.0);
}

impl KissCpx {
    #[inline(always)]
    pub const fn new(r: f32, i: f32) -> Self {
        Self { r, i }
    }
}

#[inline(always)]
fn c_mul(a: &KissCpx, b: &KissCpx) -> KissCpx {
    KissCpx {
        r: a.r * b.r - a.i * b.i,
        i: a.r * b.i + a.i * b.r,
    }
}

#[inline(always)]
fn c_sub(a: &KissCpx, b: &KissCpx) -> KissCpx {
    KissCpx {
        r: a.r - b.r,
        i: a.i - b.i,
    }
}

#[inline(always)]
fn c_add(a: &KissCpx, b: &KissCpx) -> KissCpx {
    KissCpx {
        r: a.r + b.r,
        i: a.i + b.i,
    }
}

pub struct KissFftState {
    nfft: usize,
    scale: f32,
    shift: i32,
    factors: [i16; 2 * MAXFACTORS],
    pub bitrev: FixedVec<i16, KISS_MAX_N>,
    twiddles: FixedVec<KissCpx, KISS_MAX_N>,
}

fn kf_factor(n_orig: usize, factors: &mut [i16; 2 * MAXFACTORS]) -> bool {
    // The twiddle table (KissCpx, KISS_MAX_N entries) and the bit-reversal
    // table are sized for KISS_MAX_N; anything larger would overflow them on
    // push (fail-fast assert) — reject up front so `KissFftState::new`
    // returns None gracefully (issue #27 deep scan).
    if n_orig > KISS_MAX_N {
        return false;
    }
    let mut n = n_orig;
    let mut p: i32 = 4;
    let mut stages = 0;

    loop {
        while !n.is_multiple_of(p as usize) {
            p = match p {
                4 => 2,
                2 => 3,
                _ => p + 2,
            };
            if p > 32000 || (p as i64) * (p as i64) > n as i64 {
                p = n as i32;
            }
        }
        n /= p as usize;

        if p > 5 {
            return false;
        }

        factors[2 * stages] = p as i16;

        if p == 2 && stages > 1 {
            factors[2 * stages] = 4;
            factors[2] = 2;
        }
        stages += 1;

        if n <= 1 {
            break;
        }
    }

    for i in 0..(stages / 2) {
        factors.swap(2 * i, 2 * (stages - i - 1));
    }

    n = n_orig;
    for i in 0..stages {
        n /= factors[2 * i] as usize;
        factors[2 * i + 1] = n as i16;
    }

    true
}

fn compute_bitrev_table(
    fout: i32,
    f: &mut [i16],
    fstride: usize,
    in_stride: usize,
    factors: &[i16],
) {
    let p = factors[0] as i32;
    let m = factors[1] as i32;

    if m == 1 {
        for j in 0..p {
            let idx = (j as usize) * fstride * in_stride;
            f[idx] = (fout + j) as i16;
        }
    } else {
        let mut fout = fout;
        let mut f_offset = 0usize;
        for _ in 0..p {
            compute_bitrev_table(
                fout,
                &mut f[f_offset..],
                fstride * (p as usize),
                in_stride,
                &factors[2..],
            );
            f_offset += fstride * in_stride;
            fout += m;
        }
    }
}

/// libopus compute_twiddles (kiss_fft.c, float build): the phase and its
/// cosine and sine in double precision.
fn compute_twiddles(nfft: usize) -> FixedVec<KissCpx, KISS_MAX_N> {
    let mut out: FixedVec<KissCpx, KISS_MAX_N> = FixedVec::new();
    for i in 0..nfft {
        let phase = (-2.0 * PI / nfft as f64) * i as f64;
        out.push(KissCpx::new(phase.cos() as f32, phase.sin() as f32));
    }
    out
}

impl KissFftState {
    pub fn new(nfft: usize) -> Option<Self> {
        let mut factors = [0i16; 2 * MAXFACTORS];
        if !kf_factor(nfft, &mut factors) {
            return None;
        }

        let scale = 1.0 / nfft as f32;
        let twiddles = compute_twiddles(nfft);

        let mut bitrev = FixedVec::from_value(0i16, nfft);
        compute_bitrev_table(0, &mut bitrev, 1, 1, &factors);

        Some(Self {
            nfft,
            scale,
            shift: -1,
            factors,
            bitrev,
            twiddles,
        })
    }

    pub fn new_sub(base: &KissFftState, nfft: usize) -> Option<Self> {
        let mut factors = [0i16; 2 * MAXFACTORS];
        if !kf_factor(nfft, &mut factors) {
            return None;
        }

        let mut shift = 0i32;
        while shift < 32 && (nfft << shift) != base.nfft {
            shift += 1;
        }
        if shift >= 32 {
            return None;
        }

        let mut bitrev = FixedVec::from_value(0i16, nfft);
        compute_bitrev_table(0, &mut bitrev, 1, 1, &factors);

        Some(Self {
            nfft,
            scale: 1.0 / nfft as f32,
            shift,
            factors,
            bitrev,
            twiddles: base.twiddles.clone(),
        })
    }

    /// A plan of the static mode, from libopus's tables (see `static_modes`).
    pub(crate) const fn from_tables(
        nfft: usize,
        shift: i32,
        factors: [i16; 2 * MAXFACTORS],
        bitrev: [i16; KISS_MAX_N],
        twiddles: [KissCpx; KISS_MAX_N],
    ) -> Self {
        Self {
            nfft,
            scale: 1.0 / nfft as f32,
            shift,
            factors,
            bitrev: FixedVec::from_array(bitrev, nfft),
            twiddles: FixedVec::from_array(twiddles, KISS_MAX_N),
        }
    }

    /// Every field but the twiddles, with floats as bits, for comparing
    /// plans in tests.
    #[cfg(all(test, feature = "std"))]
    pub(crate) fn fingerprint(&self) -> impl PartialEq + core::fmt::Debug + use<> {
        (
            (self.nfft, self.scale.to_bits(), self.shift, self.factors),
            self.bitrev.iter().copied().collect::<std::vec::Vec<_>>(),
        )
    }

    /// The twiddles as (re, im) pairs, for comparing plans in tests.
    #[cfg(all(test, feature = "std"))]
    pub(crate) fn twiddle_values(&self) -> std::vec::Vec<f32> {
        self.twiddles.iter().flat_map(|c| [c.r, c.i]).collect()
    }

    #[inline]
    pub fn nfft(&self) -> usize {
        self.nfft
    }

    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

/// The `n` butterfly groups of `p` quarters of `m` values each, the `g`-th
/// starting at `g * mm`.
#[inline(always)]
fn groups(
    fout: &mut [KissCpx],
    p: usize,
    m: usize,
    n: usize,
    mm: usize,
) -> impl Iterator<Item = &mut [KissCpx]> {
    let len = p * m;
    // Only the last stage, with a single group, has mm < len.
    let step = if n > 1 { mm } else { len };
    fout.chunks_mut(step).take(n).map(move |g| &mut g[..len])
}

#[inline(always)]
fn kf_bfly2(fout: &mut [KissCpx], m: usize, n: usize) {
    if m == 1 {
        for f in fout.chunks_exact_mut(2).take(n) {
            let t = f[1];
            f[1] = c_sub(&f[0], &t);
            f[0] = c_add(&f[0], &t);
        }
    } else {
        let tw: f32 = core::f32::consts::FRAC_1_SQRT_2;
        for f in fout.chunks_exact_mut(8).take(n) {
            let t = f[4];
            f[4] = c_sub(&f[0], &t);
            f[0] = c_add(&f[0], &t);

            let t = KissCpx::new((f[5].r + f[5].i) * tw, (f[5].i - f[5].r) * tw);
            f[5] = c_sub(&f[1], &t);
            f[1] = c_add(&f[1], &t);

            let t = KissCpx::new(f[6].i, -f[6].r);
            f[6] = c_sub(&f[2], &t);
            f[2] = c_add(&f[2], &t);

            let t = KissCpx::new((f[7].i - f[7].r) * tw, -(f[7].i + f[7].r) * tw);
            f[7] = c_sub(&f[3], &t);
            f[3] = c_add(&f[3], &t);
        }
    }
}

#[inline(always)]
fn kf_bfly4(
    fout: &mut [KissCpx],
    fstride: usize,
    twiddles: &[KissCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    if m == 1 {
        for f in fout.chunks_exact_mut(4).take(n) {
            let scratch0 = c_sub(&f[0], &f[2]);
            let sum02 = c_add(&f[0], &f[2]);
            let scratch1 = c_add(&f[1], &f[3]);
            let diff13 = c_sub(&f[1], &f[3]);

            f[0] = c_add(&sum02, &scratch1);
            f[2] = c_sub(&sum02, &scratch1);
            f[1] = KissCpx::new(scratch0.r + diff13.i, scratch0.i - diff13.r);
            f[3] = KissCpx::new(scratch0.r - diff13.i, scratch0.i + diff13.r);
        }
        return;
    }
    for g in groups(fout, 4, m, n, mm) {
        let (f0, rest) = g.split_at_mut(m);
        let (f1, rest) = rest.split_at_mut(m);
        let (f2, f3) = rest.split_at_mut(m);
        let tw1 = twiddles.iter().step_by(fstride);
        let tw2 = twiddles.iter().step_by(2 * fstride);
        let tw3 = twiddles.iter().step_by(3 * fstride);
        let quarters = f0.iter_mut().zip(f1).zip(f2).zip(f3);
        for ((((a, b), c), d), ((t1, t2), t3)) in quarters.zip(tw1.zip(tw2).zip(tw3)) {
            let scratch0 = c_mul(b, t1);
            let scratch1 = c_mul(c, t2);
            let scratch2 = c_mul(d, t3);

            let scratch5 = c_sub(a, &scratch1);
            *a = c_add(a, &scratch1);

            let scratch3 = c_add(&scratch0, &scratch2);
            let scratch4 = c_sub(&scratch0, &scratch2);

            *c = c_sub(a, &scratch3);
            *a = c_add(a, &scratch3);

            *b = KissCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r);
            *d = KissCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r);
        }
    }
}

#[inline(always)]
fn kf_bfly3(
    fout: &mut [KissCpx],
    fstride: usize,
    twiddles: &[KissCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    let epi3_i = twiddles[fstride * m].i;
    for g in groups(fout, 3, m, n, mm) {
        let (f0, rest) = g.split_at_mut(m);
        let (f1, f2) = rest.split_at_mut(m);
        let tw1 = twiddles.iter().step_by(fstride);
        let tw2 = twiddles.iter().step_by(2 * fstride);
        for (((a, b), c), (t1, t2)) in f0.iter_mut().zip(f1).zip(f2).zip(tw1.zip(tw2)) {
            let scratch1 = c_mul(b, t1);
            let scratch2 = c_mul(c, t2);

            let scratch3 = c_add(&scratch1, &scratch2);
            let scratch0 = c_sub(&scratch1, &scratch2);

            let half_scratch3 = KissCpx::new(scratch3.r * 0.5, scratch3.i * 0.5);
            let fout_m = KissCpx::new(a.r - half_scratch3.r, a.i - half_scratch3.i);
            let scratch0_scaled = KissCpx::new(scratch0.r * epi3_i, scratch0.i * epi3_i);

            *a = c_add(a, &scratch3);
            *b = KissCpx::new(fout_m.r - scratch0_scaled.i, fout_m.i + scratch0_scaled.r);
            *c = KissCpx::new(fout_m.r + scratch0_scaled.i, fout_m.i - scratch0_scaled.r);
        }
    }
}

#[inline(always)]
fn kf_bfly5(
    fout: &mut [KissCpx],
    fstride: usize,
    twiddles: &[KissCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    let ya = twiddles[fstride * m];
    let yb = twiddles[2 * fstride * m];
    for g in groups(fout, 5, m, n, mm) {
        let (f0, rest) = g.split_at_mut(m);
        let (f1, rest) = rest.split_at_mut(m);
        let (f2, rest) = rest.split_at_mut(m);
        let (f3, f4) = rest.split_at_mut(m);
        let tw1 = twiddles.iter().step_by(fstride);
        let tw2 = twiddles.iter().step_by(2 * fstride);
        let tw3 = twiddles.iter().step_by(3 * fstride);
        let tw4 = twiddles.iter().step_by(4 * fstride);
        let fifths = f0.iter_mut().zip(f1).zip(f2).zip(f3).zip(f4);
        let tws = tw1.zip(tw2).zip(tw3).zip(tw4);
        for (((((f0, f1), f2), f3), f4), (((t1, t2), t3), t4)) in fifths.zip(tws) {
            let scratch0 = *f0;

            let scratch1 = c_mul(f1, t1);
            let scratch2 = c_mul(f2, t2);
            let scratch3 = c_mul(f3, t3);
            let scratch4 = c_mul(f4, t4);

            let scratch7 = c_add(&scratch1, &scratch4);
            let scratch10 = c_sub(&scratch1, &scratch4);
            let scratch8 = c_add(&scratch2, &scratch3);
            let scratch9 = c_sub(&scratch2, &scratch3);

            *f0 = KissCpx::new(
                scratch0.r + (scratch7.r + scratch8.r),
                scratch0.i + (scratch7.i + scratch8.i),
            );

            let scratch5 = KissCpx::new(
                scratch0.r + (scratch7.r * ya.r + scratch8.r * yb.r),
                scratch0.i + (scratch7.i * ya.r + scratch8.i * yb.r),
            );

            let scratch6 = KissCpx::new(
                scratch10.i * ya.i + scratch9.i * yb.i,
                -(scratch10.r * ya.i + scratch9.r * yb.i),
            );

            *f1 = c_sub(&scratch5, &scratch6);
            *f4 = c_add(&scratch5, &scratch6);

            let scratch11 = KissCpx::new(
                scratch0.r + (scratch7.r * yb.r + scratch8.r * ya.r),
                scratch0.i + (scratch7.i * yb.r + scratch8.i * ya.r),
            );

            let scratch12 = KissCpx::new(
                scratch9.i * ya.i - scratch10.i * yb.i,
                scratch10.r * yb.i - scratch9.r * ya.i,
            );

            *f2 = c_add(&scratch11, &scratch12);
            *f3 = c_sub(&scratch11, &scratch12);
        }
    }
}

/// The FFT stages in place; with AVX2 the butterflies compile for it.
pub fn opus_fft_impl(st: &KissFftState, fout: &mut [KissCpx]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(avx2) = crate::compat::avx2() {
        use fearless_simd::Simd;
        return avx2.vectorize(|| fft_stages(st, fout));
    }
    fft_stages(st, fout);
}

#[inline(always)]
fn fft_stages(st: &KissFftState, fout: &mut [KissCpx]) {
    // The butterfly kernels index up to nfft elements; the SIMD paths read
    // through raw pointers, so an undersized slice would be out-of-bounds
    // memory access rather than a panic — check up front (issue #27 deep
    // scan).
    assert!(
        fout.len() >= st.nfft,
        "opus_fft_impl: fout.len() ({}) < nfft ({})",
        fout.len(),
        st.nfft
    );
    let factors = &st.factors;
    let twiddles = &st.twiddles;

    let mut fstride = [0usize; MAXFACTORS + 1];
    fstride[0] = 1;
    let mut l = 0;
    let mut m;

    loop {
        let p = factors[2 * l] as usize;
        m = factors[2 * l + 1] as usize;
        fstride[l + 1] = fstride[l] * p;
        l += 1;
        if m == 1 {
            break;
        }
    }

    let shift = if st.shift > 0 { st.shift as usize } else { 0 };

    m = factors[2 * l - 1] as usize;
    for i in (0..l).rev() {
        let p = factors[2 * i] as usize;
        let fstride_i = fstride[i];
        let fstride_adjusted = fstride_i << shift;

        let m2 = if i > 0 {
            factors[2 * i - 1] as usize
        } else {
            1
        };

        match p {
            2 => kf_bfly2(fout, m, fstride_i),
            4 => kf_bfly4(fout, fstride_adjusted, twiddles, m, fstride_i, m2),
            3 => kf_bfly3(fout, fstride_adjusted, twiddles, m, fstride_i, m2),
            5 => kf_bfly5(fout, fstride_adjusted, twiddles, m, fstride_i, m2),
            _ => {}
        }

        m = m2;
    }
}

pub fn opus_fft(st: &KissFftState, fin: &[KissCpx], fout: &mut [KissCpx]) {
    let scale = st.scale;
    let nfft = st.nfft;

    for i in 0..nfft {
        let x = &fin[i];
        let rev = st.bitrev[i] as usize;
        fout[rev] = KissCpx::new(x.r * scale, x.i * scale);
    }

    opus_fft_impl(st, fout);
}

pub fn opus_ifft(st: &KissFftState, fin: &[KissCpx], fout: &mut [KissCpx]) {
    let nfft = st.nfft;

    for i in 0..nfft {
        let rev = st.bitrev[i] as usize;
        fout[rev] = KissCpx::new(fin[i].r, -fin[i].i);
    }

    opus_fft_impl(st, fout);

    for i in 0..nfft {
        fout[i].i = -fout[i].i;
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    fn almost_equal(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() < tolerance
    }

    fn cpx_almost_equal(a: &KissCpx, b: &KissCpx, tolerance: f32) -> bool {
        almost_equal(a.r, b.r, tolerance) && almost_equal(a.i, b.i, tolerance)
    }

    #[test]
    fn test_kf_factor_60() {
        let st = KissFftState::new(60).unwrap();
        assert!(!st.factors.iter().all(|&f| f == 0));
        assert_eq!(st.bitrev.len(), 60);
    }

    #[test]
    fn test_kf_factor_240() {
        let st = KissFftState::new(240).unwrap();
        assert!(!st.factors.iter().all(|&f| f == 0));
        assert_eq!(st.bitrev.len(), 240);
    }

    #[test]
    fn test_bitrev_permutation() {
        for &nfft in &[60, 120, 240, 480] {
            let st = KissFftState::new(nfft).unwrap();
            let mut sorted: Vec<i16> = st.bitrev.as_slice().to_vec();
            sorted.sort();
            let expected: Vec<i16> = (0..nfft).map(|x| x as i16).collect();
            assert_eq!(
                sorted,
                expected,
                "bitrev for nfft={} should be a permutation of 0..{}",
                nfft,
                nfft - 1
            );
        }
    }

    #[test]
    fn test_sub_fft() {
        let nfft = 480;
        let base = KissFftState::new(nfft).unwrap();

        for &n in &[60, 120, 240] {
            let sub = KissFftState::new_sub(&base, n).unwrap();
            assert_eq!(sub.nfft(), n);
            assert_eq!(sub.twiddles.len(), nfft);
        }
    }

    #[test]
    fn test_fft_roundtrip_60() {
        let nfft = 60;
        let st = KissFftState::new(nfft).unwrap();

        let mut fin = vec![KissCpx::default(); nfft];
        let mut fout = vec![KissCpx::default(); nfft];
        let mut finv = vec![KissCpx::default(); nfft];

        fin[0] = KissCpx::new(1.0, 0.0);

        opus_fft(&st, &fin, &mut fout);
        opus_ifft(&st, &fout, &mut finv);

        for i in 0..nfft {
            let expected = if i == 0 { 1.0 } else { 0.0 };
            assert!(
                cpx_almost_equal(&finv[i], &KissCpx::new(expected, 0.0), 1e-5),
                "Roundtrip failed at index {}: got ({}, {}), expected ({}, 0)",
                i,
                finv[i].r,
                finv[i].i,
                expected
            );
        }
    }

    #[test]
    fn test_fft_roundtrip_120() {
        let nfft = 120;
        let st = KissFftState::new(nfft).unwrap();

        let mut fin = vec![KissCpx::default(); nfft];
        let mut fout = vec![KissCpx::default(); nfft];
        let mut finv = vec![KissCpx::default(); nfft];

        for i in 0..nfft {
            fin[i] = KissCpx::new((2.0 * PI * 3.0 * i as f32 / nfft as f32).sin(), 0.0);
        }

        opus_fft(&st, &fin, &mut fout);
        opus_ifft(&st, &fout, &mut finv);

        for i in 0..nfft {
            assert!(
                cpx_almost_equal(&finv[i], &fin[i], 1e-4),
                "Roundtrip failed at index {}: got ({}, {}), expected ({}, {})",
                i,
                finv[i].r,
                finv[i].i,
                fin[i].r,
                fin[i].i
            );
        }
    }

    #[test]
    fn test_fft_roundtrip_480() {
        let nfft = 480;
        let st = KissFftState::new(nfft).unwrap();

        let mut fin = vec![KissCpx::default(); nfft];
        let mut fout = vec![KissCpx::default(); nfft];
        let mut finv = vec![KissCpx::default(); nfft];

        for i in 0..nfft {
            fin[i] = KissCpx::new(
                (2.0 * PI * 7.0 * i as f32 / nfft as f32).sin(),
                (2.0 * PI * 11.0 * i as f32 / nfft as f32).cos(),
            );
        }

        opus_fft(&st, &fin, &mut fout);
        opus_ifft(&st, &fout, &mut finv);

        for i in 0..nfft {
            assert!(
                cpx_almost_equal(&finv[i], &fin[i], 1e-4),
                "Roundtrip failed at index {}: got ({}, {}), expected ({}, {})",
                i,
                finv[i].r,
                finv[i].i,
                fin[i].r,
                fin[i].i
            );
        }
    }

    #[test]
    fn test_fft_dc_component() {
        let nfft = 120;
        let st = KissFftState::new(nfft).unwrap();

        let mut fin = vec![KissCpx::default(); nfft];
        let mut fout = vec![KissCpx::default(); nfft];

        for i in 0..nfft {
            fin[i] = KissCpx::new(1.0, 0.0);
        }

        opus_fft(&st, &fin, &mut fout);

        assert!(almost_equal(fout[0].r, 1.0, 1e-5));
        assert!(almost_equal(fout[0].i, 0.0, 1e-5));

        for i in 1..nfft {
            assert!(
                cpx_almost_equal(&fout[i], &KissCpx::new(0.0, 0.0), 1e-5),
                "Non-DC component at index {} is ({}, {}), expected (0, 0)",
                i,
                fout[i].r,
                fout[i].i
            );
        }
    }

    #[test]
    fn test_fft_performance() {
        use std::time::Instant;

        let nfft = 480;
        let st = KissFftState::new(nfft).unwrap();

        let mut fin = vec![KissCpx::default(); nfft];
        let mut fout = vec![KissCpx::default(); nfft];
        let mut finv = vec![KissCpx::default(); nfft];

        for i in 0..nfft {
            fin[i] = KissCpx::new(
                (2.0 * PI * 7.0 * i as f32 / nfft as f32).sin(),
                (2.0 * PI * 11.0 * i as f32 / nfft as f32).cos(),
            );
        }

        for _ in 0..100 {
            opus_fft(&st, &fin, &mut fout);
            opus_ifft(&st, &fout, &mut finv);
        }

        let iterations = 10000;
        let start = Instant::now();

        for _ in 0..iterations {
            opus_fft(&st, &fin, &mut fout);
            opus_ifft(&st, &fout, &mut finv);
        }

        let elapsed = start.elapsed();
        let ns_per_iter = elapsed.as_nanos() as f64 / (iterations as f64 * 2.0);

        println!("\nFFT/IFFT performance (nfft={}):", nfft);
        println!(
            "  Total time for {} FFT+IFFT pairs: {:?}",
            iterations, elapsed
        );
        println!("  Time per FFT or IFFT: {:.2} ns", ns_per_iter);
        println!(
            "  Throughput: {:.2} M points/sec",
            (nfft as f64) / ns_per_iter * 1000.0
        );
    }
}
