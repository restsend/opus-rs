use std::f32::consts::PI;

pub const MAXFACTORS: usize = 8;

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct KissCpx {
    pub r: f32,
    pub i: f32,
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
    pub bitrev: Vec<i16>,
    twiddles: Vec<KissCpx>,
}

fn kf_factor(n_orig: usize, factors: &mut [i16; 2 * MAXFACTORS]) -> bool {
    let mut n = n_orig;
    let mut p: i32 = 4;
    let mut stages = 0;

    loop {
        while n % (p as usize) != 0 {
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
        let tmp = factors[2 * i];
        factors[2 * i] = factors[2 * (stages - i - 1)];
        factors[2 * (stages - i - 1)] = tmp;
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
    let p = factors[0] as i32; // the radix
    let m = factors[1] as i32; // stage's fft length / p

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

fn compute_twiddles(nfft: usize) -> Vec<KissCpx> {
    let two_pi_over_n = -2.0 * PI / nfft as f32;
    (0..nfft)
        .map(|i| {
            let phase = two_pi_over_n * i as f32;
            KissCpx::new(phase.cos(), phase.sin())
        })
        .collect()
}

impl KissFftState {
    pub fn new(nfft: usize) -> Option<Self> {
        let mut factors = [0i16; 2 * MAXFACTORS];
        if !kf_factor(nfft, &mut factors) {
            return None;
        }

        let scale = 1.0 / nfft as f32;
        let twiddles = compute_twiddles(nfft);

        let mut bitrev = vec![0i16; nfft];
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

        let mut bitrev = vec![0i16; nfft];
        compute_bitrev_table(0, &mut bitrev, 1, 1, &factors);

        Some(Self {
            nfft,
            scale: base.scale,
            shift,
            factors,
            bitrev,
            twiddles: base.twiddles.clone(),
        })
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

#[inline(always)]
fn kf_bfly2(fout: &mut [KissCpx], m: usize, n: usize) {
    if m == 1 {
        for i in 0..n {
            let idx = i * 2;
            let t = fout[idx + 1];
            fout[idx + 1] = c_sub(&fout[idx], &t);
            fout[idx] = c_add(&fout[idx], &t);
        }
    } else {
        let tw: f32 = 0.7071067812;
        for i in 0..n {
            let base = i * 8;

            let t = fout[base + 4];
            fout[base + 4] = c_sub(&fout[base], &t);
            fout[base] = c_add(&fout[base], &t);

            let t = KissCpx::new(
                (fout[base + 5].r + fout[base + 5].i) * tw,
                (fout[base + 5].i - fout[base + 5].r) * tw,
            );
            fout[base + 5] = c_sub(&fout[base + 1], &t);
            fout[base + 1] = c_add(&fout[base + 1], &t);

            let t = KissCpx::new(fout[base + 6].i, -fout[base + 6].r);
            fout[base + 6] = c_sub(&fout[base + 2], &t);
            fout[base + 2] = c_add(&fout[base + 2], &t);

            let t = KissCpx::new(
                (fout[base + 7].i - fout[base + 7].r) * tw,
                -(fout[base + 7].i + fout[base + 7].r) * tw,
            );
            fout[base + 7] = c_sub(&fout[base + 3], &t);
            fout[base + 3] = c_add(&fout[base + 3], &t);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn kf_bfly4(
    fout: &mut [KissCpx],
    fstride: usize,
    twiddles: &[KissCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    let m2 = 2 * m;
    let m3 = 3 * m;

    if m == 1 {
        for i in 0..n {
            let base = i * 4;

            let scratch0 = c_sub(&fout[base], &fout[base + 2]);
            let sum02 = c_add(&fout[base], &fout[base + 2]);
            let scratch1 = c_add(&fout[base + 1], &fout[base + 3]);
            let diff13 = c_sub(&fout[base + 1], &fout[base + 3]);

            fout[base] = c_add(&sum02, &scratch1);
            fout[base + 2] = c_sub(&sum02, &scratch1);
            fout[base + 1] = KissCpx::new(scratch0.r + diff13.i, scratch0.i - diff13.r);
            fout[base + 3] = KissCpx::new(scratch0.r - diff13.i, scratch0.i + diff13.r);
        }
    } else {
        for i in 0..n {
            let base = i * mm;
            let stride2 = fstride * 2;
            let stride3 = fstride * 3;

            let m4 = m / 4;

            for j in 0..m4 {
                let j4 = j * 4;

                for jj in 0..4 {
                    let jv = j4 + jj;
                    let idx = base + jv;

                    let scratch0 = c_mul(&fout[idx + m], &twiddles[jv * fstride]);
                    let scratch1 = c_mul(&fout[idx + m2], &twiddles[jv * stride2]);
                    let scratch2 = c_mul(&fout[idx + m3], &twiddles[jv * stride3]);

                    let scratch5 = c_sub(&fout[idx], &scratch1);
                    fout[idx] = c_add(&fout[idx], &scratch1);

                    let scratch3 = c_add(&scratch0, &scratch2);
                    let scratch4 = c_sub(&scratch0, &scratch2);

                    fout[idx + m2] = c_sub(&fout[idx], &scratch3);
                    fout[idx] = c_add(&fout[idx], &scratch3);

                    fout[idx + m] = KissCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r);
                    fout[idx + m3] = KissCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r);
                }
            }

            for j in (m4 * 4)..m {
                let idx = base + j;

                let scratch0 = c_mul(&fout[idx + m], &twiddles[j * fstride]);
                let scratch1 = c_mul(&fout[idx + m2], &twiddles[j * stride2]);
                let scratch2 = c_mul(&fout[idx + m3], &twiddles[j * stride3]);

                let scratch5 = c_sub(&fout[idx], &scratch1);
                fout[idx] = c_add(&fout[idx], &scratch1);

                let scratch3 = c_add(&scratch0, &scratch2);
                let scratch4 = c_sub(&scratch0, &scratch2);

                fout[idx + m2] = c_sub(&fout[idx], &scratch3);
                fout[idx] = c_add(&fout[idx], &scratch3);

                fout[idx + m] = KissCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r);
                fout[idx + m3] = KissCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r);
            }
        }
    }
}

/// Radix-4 butterfly (matches C kf_bfly4) - scalar version for non-NEON
#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn kf_bfly4(
    fout: &mut [KissCpx],
    fstride: usize,
    twiddles: &[KissCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    let m2 = 2 * m;
    let m3 = 3 * m;

    if m == 1 {
        // Degenerate case where all twiddles are 1
        for i in 0..n {
            let base = i * 4;

            let scratch0 = c_sub(&fout[base], &fout[base + 2]);
            let sum02 = c_add(&fout[base], &fout[base + 2]);
            let scratch1 = c_add(&fout[base + 1], &fout[base + 3]);
            let diff13 = c_sub(&fout[base + 1], &fout[base + 3]);

            fout[base] = c_add(&sum02, &scratch1);
            fout[base + 2] = c_sub(&sum02, &scratch1);
            fout[base + 1] = KissCpx::new(scratch0.r + diff13.i, scratch0.i - diff13.r);
            fout[base + 3] = KissCpx::new(scratch0.r - diff13.i, scratch0.i + diff13.r);
        }
    } else {
        for i in 0..n {
            let base = i * mm;
            let stride2 = fstride * 2;
            let stride3 = fstride * 3;

            // m is guaranteed to be a multiple of 4
            for j in 0..m {
                let idx = base + j;

                let scratch0 = c_mul(&fout[idx + m], &twiddles[j * fstride]);
                let scratch1 = c_mul(&fout[idx + m2], &twiddles[j * stride2]);
                let scratch2 = c_mul(&fout[idx + m3], &twiddles[j * stride3]);

                let scratch5 = c_sub(&fout[idx], &scratch1);
                fout[idx] = c_add(&fout[idx], &scratch1);

                let scratch3 = c_add(&scratch0, &scratch2);
                let scratch4 = c_sub(&scratch0, &scratch2);

                fout[idx + m2] = c_sub(&fout[idx], &scratch3);
                fout[idx] = c_add(&fout[idx], &scratch3);

                fout[idx + m] = KissCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r);
                fout[idx + m3] = KissCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r);
            }
        }
    }
}

/// Radix-3 butterfly (matches C kf_bfly3)
#[inline(always)]
fn kf_bfly3(
    fout: &mut [KissCpx],
    fstride: usize,
    twiddles: &[KissCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    let m2 = 2 * m;

    // epi3 = exp(-2*pi*i/3) = -0.5 - 0.86602540i
    let epi3_i: f32 = -0.86602540;

    for i in 0..n {
        let base = i * mm;
        let stride2 = fstride * 2;

        for j in 0..m {
            let idx = base + j;

            let scratch1 = c_mul(&fout[idx + m], &twiddles[j * fstride]);
            let scratch2 = c_mul(&fout[idx + m2], &twiddles[j * stride2]);

            let scratch3 = c_add(&scratch1, &scratch2);
            let scratch0 = c_sub(&scratch1, &scratch2);

            // HALF_OF(scratch3)
            let half_scratch3 = KissCpx::new(scratch3.r * 0.5, scratch3.i * 0.5);

            let fout_m = KissCpx::new(fout[idx].r - half_scratch3.r, fout[idx].i - half_scratch3.i);

            // C_MULBYSCALAR(scratch0, epi3.i)
            let scratch0_scaled = KissCpx::new(scratch0.r * epi3_i, scratch0.i * epi3_i);

            fout[idx] = c_add(&fout[idx], &scratch3);

            fout[idx + m] = KissCpx::new(fout_m.r - scratch0_scaled.i, fout_m.i + scratch0_scaled.r);
            fout[idx + m2] = KissCpx::new(fout_m.r + scratch0_scaled.i, fout_m.i - scratch0_scaled.r);
        }
    }
}

/// Radix-5 butterfly (matches C kf_bfly5)
#[inline(always)]
fn kf_bfly5(
    fout: &mut [KissCpx],
    fstride: usize,
    twiddles: &[KissCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    // ya = exp(-2*pi*i/5), yb = exp(-4*pi*i/5)
    let ya = KissCpx::new(0.30901699, -0.95105652);
    let yb = KissCpx::new(-0.80901699, -0.58778525);

    for i in 0..n {
        let base = i * mm;
        let stride2 = fstride * 2;
        let stride3 = fstride * 3;
        let stride4 = fstride * 4;

        for u in 0..m {
            let idx0 = base + u;
            let idx1 = idx0 + m;
            let idx2 = idx0 + 2 * m;
            let idx3 = idx0 + 3 * m;
            let idx4 = idx0 + 4 * m;

            // Save original value (scratch[0] in C)
            let scratch0 = fout[idx0];

            let scratch1 = c_mul(&fout[idx1], &twiddles[u * fstride]);
            let scratch2 = c_mul(&fout[idx2], &twiddles[u * stride2]);
            let scratch3 = c_mul(&fout[idx3], &twiddles[u * stride3]);
            let scratch4 = c_mul(&fout[idx4], &twiddles[u * stride4]);

            let scratch7 = c_add(&scratch1, &scratch4);
            let scratch10 = c_sub(&scratch1, &scratch4);
            let scratch8 = c_add(&scratch2, &scratch3);
            let scratch9 = c_sub(&scratch2, &scratch3);

            // Update fout[idx0] first (this is Fout0 in C)
            fout[idx0] = KissCpx::new(
                scratch0.r + scratch7.r + scratch8.r,
                scratch0.i + scratch7.i + scratch8.i,
            );

            // Use scratch0 (original value) for scratch5 and scratch11 (matches C)
            let scratch5 = KissCpx::new(
                scratch0.r + scratch7.r * ya.r + scratch8.r * yb.r,
                scratch0.i + scratch7.i * ya.r + scratch8.i * yb.r,
            );

            let scratch6 = KissCpx::new(
                scratch10.i * ya.i + scratch9.i * yb.i,
                -(scratch10.r * ya.i + scratch9.r * yb.i),
            );

            fout[idx1] = c_sub(&scratch5, &scratch6);
            fout[idx4] = c_add(&scratch5, &scratch6);

            let scratch11 = KissCpx::new(
                scratch0.r + scratch7.r * yb.r + scratch8.r * ya.r,
                scratch0.i + scratch7.i * yb.r + scratch8.i * ya.r,
            );

            let scratch12 = KissCpx::new(
                scratch9.i * ya.i - scratch10.i * yb.i,
                scratch10.r * yb.i - scratch9.r * ya.i,
            );

            fout[idx2] = c_add(&scratch11, &scratch12);
            fout[idx3] = c_sub(&scratch11, &scratch12);
        }
    }
}

pub fn opus_fft_impl(st: &KissFftState, fout: &mut [KissCpx]) {
    let factors = &st.factors;
    let twiddles = &st.twiddles;

    // Compute strides
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
        fout[rev] = fin[i];
    }

    for i in 0..nfft {
        fout[i].i = -fout[i].i;
    }

    opus_fft_impl(st, fout);

    for i in 0..nfft {
        fout[i].i = -fout[i].i;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Verify bitrev is a valid permutation for various sizes
        for &nfft in &[60, 120, 240, 480] {
            let st = KissFftState::new(nfft).unwrap();
            let mut sorted: Vec<i16> = st.bitrev.clone();
            sorted.sort();
            let expected: Vec<i16> = (0..nfft).map(|x| x as i16).collect();
            assert_eq!(sorted, expected, "bitrev for nfft={} should be a permutation of 0..{}", nfft, nfft - 1);
        }
    }

    #[test]
    fn test_sub_fft() {
        let nfft = 480;
        let base = KissFftState::new(nfft).unwrap();

        for &n in &[60, 120, 240] {
            let sub = KissFftState::new_sub(&base, n).unwrap();
            assert_eq!(sub.nfft(), n);
            assert_eq!(sub.twiddles.len(), nfft); // Shares twiddles with base
        }
    }

    #[test]
    fn test_fft_roundtrip_60() {
        let nfft = 60;
        let st = KissFftState::new(nfft).unwrap();

        // Create impulse input
        let mut fin = vec![KissCpx::default(); nfft];
        let mut fout = vec![KissCpx::default(); nfft];
        let mut finv = vec![KissCpx::default(); nfft];

        fin[0] = KissCpx::new(1.0, 0.0);

        opus_fft(&st, &fin, &mut fout);
        opus_ifft(&st, &fout, &mut finv);

        // Check roundtrip
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

        // Sine wave at frequency 3
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

        // Complex input
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

        // DC component only
        for i in 0..nfft {
            fin[i] = KissCpx::new(1.0, 0.0);
        }

        opus_fft(&st, &fin, &mut fout);

        // DC should be nfft * scale = 1, others should be 0
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

        // Initialize with some data
        for i in 0..nfft {
            fin[i] = KissCpx::new(
                (2.0 * PI * 7.0 * i as f32 / nfft as f32).sin(),
                (2.0 * PI * 11.0 * i as f32 / nfft as f32).cos(),
            );
        }

        // Warm up
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
        println!("  Total time for {} FFT+IFFT pairs: {:?}", iterations, elapsed);
        println!("  Time per FFT or IFFT: {:.2} ns", ns_per_iter);
        println!("  Throughput: {:.2} M points/sec", (nfft as f64) / ns_per_iter * 1000.0);
    }
}
