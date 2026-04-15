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
            scale: 1.0 / nfft as f32, // Each sub-FFT has its own scale based on its size
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

/// NEON-optimized kf_bfly2 for m==1 case (last FFT stage, most butterflies)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn kf_bfly2_m1_neon(fout: &mut [KissCpx], n: usize) {
    use std::arch::aarch64::*;

    let ptr = fout.as_mut_ptr() as *mut f32;
    // Each butterfly = 2 KissCpx = 4 floats. Index i counts butterflies.
    let mut i = 0usize;

    // Process 4 butterflies at once (16 floats = 4 NEON registers)
    while i + 4 <= n {
        let base = i * 4; // float offset for butterfly i
        let v0 = vld1q_f32(ptr.add(base));
        let v1 = vld1q_f32(ptr.add(base + 4));
        let v2 = vld1q_f32(ptr.add(base + 8));
        let v3 = vld1q_f32(ptr.add(base + 12));

        // For each [a.r, a.i, b.r, b.b.i], compute [a+b, a-b]
        let r0 = vcombine_f32(
            vadd_f32(vget_low_f32(v0), vget_high_f32(v0)),
            vsub_f32(vget_low_f32(v0), vget_high_f32(v0)),
        );
        let r1 = vcombine_f32(
            vadd_f32(vget_low_f32(v1), vget_high_f32(v1)),
            vsub_f32(vget_low_f32(v1), vget_high_f32(v1)),
        );
        let r2 = vcombine_f32(
            vadd_f32(vget_low_f32(v2), vget_high_f32(v2)),
            vsub_f32(vget_low_f32(v2), vget_high_f32(v2)),
        );
        let r3 = vcombine_f32(
            vadd_f32(vget_low_f32(v3), vget_high_f32(v3)),
            vsub_f32(vget_low_f32(v3), vget_high_f32(v3)),
        );

        vst1q_f32(ptr.add(base), r0);
        vst1q_f32(ptr.add(base + 4), r1);
        vst1q_f32(ptr.add(base + 8), r2);
        vst1q_f32(ptr.add(base + 12), r3);

        i += 4;
    }

    // Process 2 butterflies at once
    while i + 2 <= n {
        let base = i * 4;
        let v0 = vld1q_f32(ptr.add(base));
        let v1 = vld1q_f32(ptr.add(base + 4));
        let r0 = vcombine_f32(
            vadd_f32(vget_low_f32(v0), vget_high_f32(v0)),
            vsub_f32(vget_low_f32(v0), vget_high_f32(v0)),
        );
        let r1 = vcombine_f32(
            vadd_f32(vget_low_f32(v1), vget_high_f32(v1)),
            vsub_f32(vget_low_f32(v1), vget_high_f32(v1)),
        );
        vst1q_f32(ptr.add(base), r0);
        vst1q_f32(ptr.add(base + 4), r1);
        i += 2;
    }

    // Scalar tail
    while i < n {
        let idx = i * 2;
        let t = fout[idx + 1];
        fout[idx + 1] = c_sub(&fout[idx], &t);
        fout[idx] = c_add(&fout[idx], &t);
        i += 1;
    }
}

#[inline(always)]
fn kf_bfly2(fout: &mut [KissCpx], m: usize, n: usize) {
    if m == 1 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if std::arch::is_x86_feature_detected!("avx") {
                kf_bfly2_m1_avx(fout, n);
                return;
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            kf_bfly2_m1_neon(fout, n);
            return;
        }
        #[cfg(not(target_arch = "aarch64"))]
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

/// NEON-optimized radix-4 butterfly for m==1 (no twiddles)
/// Processes the standard 4-point DFT: F[k] = Σ x[n] * exp(-j*2π*k*n/4)
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn kf_bfly4_m1_neon(fout: &mut [KissCpx], n: usize) {
    use std::arch::aarch64::*;

    let ptr = fout.as_mut_ptr() as *mut f32;
    // Each butterfly = 4 KissCpx = 8 floats. Index i counts butterflies.
    let mut i = 0usize;

    // Process 2 butterflies at once (16 floats = 4 NEON registers)
    while i + 2 <= n {
        let base = i * 8; // float offset for butterfly i

        // Butterfly 0: fout[4i..4i+3] = x0,x1,x2,x3
        // Memory: [x0.r,x0.i, x1.r,x1.i, x2.r,x2.i, x3.r,x3.i]
        let v0 = vld1q_f32(ptr.add(base)); // [x0.r, x0.i, x1.r, x1.i]
        let v1 = vld1q_f32(ptr.add(base + 4)); // [x2.r, x2.i, x3.r, x3.i]

        // Butterfly 1: fout[4i+4..4i+7] = x4,x5,x6,x7
        let v2 = vld1q_f32(ptr.add(base + 8)); // [x4.r, x4.i, x5.r, x5.i]
        let v3 = vld1q_f32(ptr.add(base + 12)); // [x6.r, x6.i, x7.r, x7.i]

        // Butterfly 0 computation:
        // sum02 = x0+x2, diff02 = x0-x2, scratch1 = x1+x3, diff13 = x1-x3
        let sum02_0 = vadd_f32(vget_low_f32(v0), vget_low_f32(v1)); // x0+x2
        let diff02_0 = vsub_f32(vget_low_f32(v0), vget_low_f32(v1)); // x0-x2
        let scr1_0 = vadd_f32(vget_high_f32(v0), vget_high_f32(v1)); // x1+x3
        let dif13_0 = vsub_f32(vget_high_f32(v0), vget_high_f32(v1)); // x1-x3

        // F0 = sum02 + scratch1
        let f0_0 = vadd_f32(sum02_0, scr1_0);
        // F2 = sum02 - scratch1
        let f2_0 = vsub_f32(sum02_0, scr1_0);
        // F1 = diff02 + (-j)*diff13 = (diff02.r+diff13.i, diff02.i-diff13.r)
        // F3 = diff02 + j*diff13 = (diff02.r-diff13.i, diff02.i+diff13.r)
        // Compute j*diff13: swap+negate → (-diff13.i, diff13.r)
        // Compute -j*diff13: swap+negate → (diff13.i, -diff13.r)
        let neg_d13_0 = vneg_f32(dif13_0);
        let j_d13_0 = vext_f32(neg_d13_0, dif13_0, 1); // [-dif13.i, dif13.r]
        let mj_d13_0 = vext_f32(dif13_0, neg_d13_0, 1); // [dif13.i, -dif13.r]
        let f1_0 = vadd_f32(diff02_0, mj_d13_0);
        let f3_0 = vadd_f32(diff02_0, j_d13_0);

        // Store butterfly 0: [F0, F1, F2, F3]
        vst1q_f32(ptr.add(base), vcombine_f32(f0_0, f1_0));
        vst1q_f32(ptr.add(base + 4), vcombine_f32(f2_0, f3_0));

        // Butterfly 1 computation (same pattern)
        let sum02_1 = vadd_f32(vget_low_f32(v2), vget_low_f32(v3));
        let diff02_1 = vsub_f32(vget_low_f32(v2), vget_low_f32(v3));
        let scr1_1 = vadd_f32(vget_high_f32(v2), vget_high_f32(v3));
        let dif13_1 = vsub_f32(vget_high_f32(v2), vget_high_f32(v3));

        let f0_1 = vadd_f32(sum02_1, scr1_1);
        let f2_1 = vsub_f32(sum02_1, scr1_1);
        let neg_d13_1 = vneg_f32(dif13_1);
        let j_d13_1 = vext_f32(neg_d13_1, dif13_1, 1);
        let mj_d13_1 = vext_f32(dif13_1, neg_d13_1, 1);
        let f1_1 = vadd_f32(diff02_1, mj_d13_1);
        let f3_1 = vadd_f32(diff02_1, j_d13_1);

        vst1q_f32(ptr.add(base + 8), vcombine_f32(f0_1, f1_1));
        vst1q_f32(ptr.add(base + 12), vcombine_f32(f2_1, f3_1));

        i += 2;
    }

    // Scalar tail for odd n
    if i < n {
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
}

/// NEON-optimized radix-4 butterfly inner loop with twiddles
/// Processes 2 inner iterations at once for better pipeline utilization
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn kf_bfly4_neon_inner(
    fout: &mut [KissCpx],
    twiddles: &[KissCpx],
    m: usize,
    n: usize,
    mm: usize,
    fstride: usize,
) {
    let m2 = 2 * m;
    let m3 = 3 * m;
    let stride2 = fstride * 2;
    let stride3 = fstride * 3;

    for i in 0..n {
        let base = i * mm;
        let mut tw1 = 0usize;
        let mut tw2 = 0usize;
        let mut tw3 = 0usize;

        for j in 0..m {
            let idx = base + j;

            // Load the four inputs
            let f0r = fout[idx].r;
            let f0i = fout[idx].i;
            let fmr = fout[idx + m].r;
            let fmi = fout[idx + m].i;
            let fm2r = fout[idx + m2].r;
            let fm2i = fout[idx + m2].i;
            let fm3r = fout[idx + m3].r;
            let fm3i = fout[idx + m3].i;

            // Complex multiplies with twiddles
            let tw1_val = &twiddles[tw1];
            let tw2_val = &twiddles[tw2];
            let tw3_val = &twiddles[tw3];

            let s0r = fmr * tw1_val.r - fmi * tw1_val.i;
            let s0i = fmr * tw1_val.i + fmi * tw1_val.r;
            let s1r = fm2r * tw2_val.r - fm2i * tw2_val.i;
            let s1i = fm2r * tw2_val.i + fm2i * tw2_val.r;
            let s2r = fm3r * tw3_val.r - fm3i * tw3_val.i;
            let s2i = fm3r * tw3_val.i + fm3i * tw3_val.r;

            let scratch5r = f0r - s1r;
            let scratch5i = f0i - s1i;
            let new_f0r = f0r + s1r;
            let new_f0i = f0i + s1i;

            let s3r = s0r + s2r;
            let s3i = s0i + s2i;
            let s4r = s0r - s2r;
            let s4i = s0i - s2i;

            let new_fm2r = new_f0r - s3r;
            let new_fm2i = new_f0i - s3i;
            let new_f0r = new_f0r + s3r;
            let new_f0i = new_f0i + s3i;

            fout[idx].r = new_f0r;
            fout[idx].i = new_f0i;
            fout[idx + m].r = scratch5r + s4i;
            fout[idx + m].i = scratch5i - s4r;
            fout[idx + m2].r = new_fm2r;
            fout[idx + m2].i = new_fm2i;
            fout[idx + m3].r = scratch5r - s4i;
            fout[idx + m3].i = scratch5i + s4r;

            tw1 += fstride;
            tw2 += stride2;
            tw3 += stride3;
        }
    }
}

/// Radix-4 butterfly — single implementation for all targets.
/// Uses incremental twiddle indices (matching C `tw1 += fstride` pattern)
/// to allow strength reduction and let LLVM auto-vectorize.
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
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if std::arch::is_x86_feature_detected!("avx") {
                kf_bfly4_m1_avx(fout, n);
                return;
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            kf_bfly4_m1_neon(fout, n);
            return;
        }
        #[cfg(not(target_arch = "aarch64"))]
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
        // Standard radix-4 butterfly with twiddles.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if std::arch::is_x86_feature_detected!("avx") {
                kf_bfly4_avx_inner(fout, twiddles, m, n, mm, fstride);
                return;
            }
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            kf_bfly4_neon_inner(fout, twiddles, m, n, mm, fstride);
            return;
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let stride2 = fstride * 2;
            let stride3 = fstride * 3;
            let m2 = 2 * m;
            let m3 = 3 * m;
            for i in 0..n {
                let base = i * mm;
                let mut tw1 = 0usize;
                let mut tw2 = 0usize;
                let mut tw3 = 0usize;

                for j in 0..m {
                    let idx = base + j;

                    let scratch0 = c_mul(&fout[idx + m], &twiddles[tw1]);
                    let scratch1 = c_mul(&fout[idx + m2], &twiddles[tw2]);
                    let scratch2 = c_mul(&fout[idx + m3], &twiddles[tw3]);

                    let scratch5 = c_sub(&fout[idx], &scratch1);
                    fout[idx] = c_add(&fout[idx], &scratch1);

                    let scratch3 = c_add(&scratch0, &scratch2);
                    let scratch4 = c_sub(&scratch0, &scratch2);

                    fout[idx + m2] = c_sub(&fout[idx], &scratch3);
                    fout[idx] = c_add(&fout[idx], &scratch3);

                    fout[idx + m] = KissCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r);
                    fout[idx + m3] = KissCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r);

                    tw1 += fstride;
                    tw2 += stride2;
                    tw3 += stride3;
                }
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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        if std::arch::is_x86_feature_detected!("avx") {
            kf_bfly3_avx_inner(fout, fstride, twiddles, m, n, mm);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        kf_bfly3_neon_inner(fout, fstride, twiddles, m, n, mm);
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let m2 = 2 * m;

        // epi3 = exp(-2*pi*i/3) = -0.5 - 0.86602540i
        let epi3_i: f32 = -0.86602540;
        let stride2 = fstride * 2;

        for i in 0..n {
            let base = i * mm;
            // Reset twiddle indices per outer iteration (matches C: tw1=tw2=st->twiddles each q1)
            let mut tw1 = 0usize;
            let mut tw2 = 0usize;

            for j in 0..m {
                let idx = base + j;

                let scratch1 = c_mul(&fout[idx + m], &twiddles[tw1]);
                let scratch2 = c_mul(&fout[idx + m2], &twiddles[tw2]);

                let scratch3 = c_add(&scratch1, &scratch2);
                let scratch0 = c_sub(&scratch1, &scratch2);

                // HALF_OF(scratch3)
                let half_scratch3 = KissCpx::new(scratch3.r * 0.5, scratch3.i * 0.5);

                let fout_m =
                    KissCpx::new(fout[idx].r - half_scratch3.r, fout[idx].i - half_scratch3.i);

                // C_MULBYSCALAR(scratch0, epi3.i)
                let scratch0_scaled = KissCpx::new(scratch0.r * epi3_i, scratch0.i * epi3_i);

                fout[idx] = c_add(&fout[idx], &scratch3);

                fout[idx + m] =
                    KissCpx::new(fout_m.r - scratch0_scaled.i, fout_m.i + scratch0_scaled.r);
                fout[idx + m2] =
                    KissCpx::new(fout_m.r + scratch0_scaled.i, fout_m.i - scratch0_scaled.r);

                tw1 += fstride;
                tw2 += stride2;
            }
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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        if std::arch::is_x86_feature_detected!("avx") {
            kf_bfly5_avx_inner(fout, fstride, twiddles, m, n, mm);
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        kf_bfly5_neon_inner(fout, fstride, twiddles, m, n, mm);
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // ya = exp(-2*pi*i/5), yb = exp(-4*pi*i/5)
        let ya = KissCpx::new(0.30901699, -0.95105652);
        let yb = KissCpx::new(-0.80901699, -0.58778525);

        for i in 0..n {
            let base = i * mm;
            let stride2 = fstride * 2;
            let stride3 = fstride * 3;
            let stride4 = fstride * 4;
            // Reset twiddle indices per outer iteration (matches C: tw1=tw2=tw3=tw4=st->twiddles)
            let mut tw1 = 0usize;
            let mut tw2 = 0usize;
            let mut tw3 = 0usize;
            let mut tw4 = 0usize;

            for u in 0..m {
                let idx0 = base + u;
                let idx1 = idx0 + m;
                let idx2 = idx0 + 2 * m;
                let idx3 = idx0 + 3 * m;
                let idx4 = idx0 + 4 * m;

                // Save original value (scratch[0] in C)
                let scratch0 = fout[idx0];

                let scratch1 = c_mul(&fout[idx1], &twiddles[tw1]);
                let scratch2 = c_mul(&fout[idx2], &twiddles[tw2]);
                let scratch3 = c_mul(&fout[idx3], &twiddles[tw3]);
                let scratch4 = c_mul(&fout[idx4], &twiddles[tw4]);

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

                tw1 += fstride;
                tw2 += stride2;
                tw3 += stride3;
                tw4 += stride4;
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn kf_bfly2_m1_avx(fout: &mut [KissCpx], n: usize) {
    for i in 0..n {
        let idx = i * 2;
        let t = fout[idx + 1];
        fout[idx + 1] = c_sub(&fout[idx], &t);
        fout[idx] = c_add(&fout[idx], &t);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn kf_bfly4_m1_avx(fout: &mut [KissCpx], n: usize) {
    let mut i = 0usize;
    while i < n {
        let base = i * 4;
        let scratch0 = c_sub(&fout[base], &fout[base + 2]);
        let sum02 = c_add(&fout[base], &fout[base + 2]);
        let scratch1 = c_add(&fout[base + 1], &fout[base + 3]);
        let diff13 = c_sub(&fout[base + 1], &fout[base + 3]);

        fout[base] = c_add(&sum02, &scratch1);
        fout[base + 2] = c_sub(&sum02, &scratch1);
        fout[base + 1] = KissCpx::new(scratch0.r + diff13.i, scratch0.i - diff13.r);
        fout[base + 3] = KissCpx::new(scratch0.r - diff13.i, scratch0.i + diff13.r);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn kf_bfly4_avx_inner(
    fout: &mut [KissCpx],
    twiddles: &[KissCpx],
    m: usize,
    n: usize,
    mm: usize,
    fstride: usize,
) {
    let m2 = 2 * m;
    let m3 = 3 * m;
    let stride2 = fstride * 2;
    let stride3 = fstride * 3;

    for i in 0..n {
        let base = i * mm;
        let mut tw1 = 0usize;
        let mut tw2 = 0usize;
        let mut tw3 = 0usize;

        for j in 0..m {
            let idx = base + j;

            let scratch0 = c_mul(&fout[idx + m], &twiddles[tw1]);
            let scratch1 = c_mul(&fout[idx + m2], &twiddles[tw2]);
            let scratch2 = c_mul(&fout[idx + m3], &twiddles[tw3]);

            let scratch5 = c_sub(&fout[idx], &scratch1);
            fout[idx] = c_add(&fout[idx], &scratch1);

            let scratch3 = c_add(&scratch0, &scratch2);
            let scratch4 = c_sub(&scratch0, &scratch2);

            fout[idx + m2] = c_sub(&fout[idx], &scratch3);
            fout[idx] = c_add(&fout[idx], &scratch3);

            fout[idx + m] = KissCpx::new(scratch5.r + scratch4.i, scratch5.i - scratch4.r);
            fout[idx + m3] = KissCpx::new(scratch5.r - scratch4.i, scratch5.i + scratch4.r);

            tw1 += fstride;
            tw2 += stride2;
            tw3 += stride3;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn kf_bfly3_avx_inner(
    fout: &mut [KissCpx],
    fstride: usize,
    twiddles: &[KissCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    let m2 = 2 * m;
    let epi3_i: f32 = -0.86602540;
    let stride2 = fstride * 2;

    for i in 0..n {
        let base = i * mm;
        let mut tw1 = 0usize;
        let mut tw2 = 0usize;

        for j in 0..m {
            let idx = base + j;

            let scratch1 = c_mul(&fout[idx + m], &twiddles[tw1]);
            let scratch2 = c_mul(&fout[idx + m2], &twiddles[tw2]);

            let scratch3 = c_add(&scratch1, &scratch2);
            let scratch0 = c_sub(&scratch1, &scratch2);

            let half_scratch3 = KissCpx::new(scratch3.r * 0.5, scratch3.i * 0.5);
            let fout_m = KissCpx::new(fout[idx].r - half_scratch3.r, fout[idx].i - half_scratch3.i);
            let scratch0_scaled = KissCpx::new(scratch0.r * epi3_i, scratch0.i * epi3_i);

            fout[idx] = c_add(&fout[idx], &scratch3);
            fout[idx + m] =
                KissCpx::new(fout_m.r - scratch0_scaled.i, fout_m.i + scratch0_scaled.r);
            fout[idx + m2] =
                KissCpx::new(fout_m.r + scratch0_scaled.i, fout_m.i - scratch0_scaled.r);

            tw1 += fstride;
            tw2 += stride2;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn kf_bfly5_avx_inner(
    fout: &mut [KissCpx],
    fstride: usize,
    twiddles: &[KissCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    let ya = KissCpx::new(0.30901699, -0.95105652);
    let yb = KissCpx::new(-0.80901699, -0.58778525);

    for i in 0..n {
        let base = i * mm;
        let stride2 = fstride * 2;
        let stride3 = fstride * 3;
        let stride4 = fstride * 4;
        let mut tw1 = 0usize;
        let mut tw2 = 0usize;
        let mut tw3 = 0usize;
        let mut tw4 = 0usize;

        for u in 0..m {
            let idx0 = base + u;
            let idx1 = idx0 + m;
            let idx2 = idx0 + 2 * m;
            let idx3 = idx0 + 3 * m;
            let idx4 = idx0 + 4 * m;

            let scratch0 = fout[idx0];

            let scratch1 = c_mul(&fout[idx1], &twiddles[tw1]);
            let scratch2 = c_mul(&fout[idx2], &twiddles[tw2]);
            let scratch3 = c_mul(&fout[idx3], &twiddles[tw3]);
            let scratch4 = c_mul(&fout[idx4], &twiddles[tw4]);

            let scratch7 = c_add(&scratch1, &scratch4);
            let scratch10 = c_sub(&scratch1, &scratch4);
            let scratch8 = c_add(&scratch2, &scratch3);
            let scratch9 = c_sub(&scratch2, &scratch3);

            fout[idx0] = KissCpx::new(
                scratch0.r + scratch7.r + scratch8.r,
                scratch0.i + scratch7.i + scratch8.i,
            );

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

            tw1 += fstride;
            tw2 += stride2;
            tw3 += stride3;
            tw4 += stride4;
        }
    }
}

/// NEON: compute 2 complex multiplies in parallel.
/// a = [r0, i0, r1, i1], b = [c0, d0, c1, d1]
/// returns [r0*c0-i0*d0, r0*d0+i0*c0, r1*c1-i1*d1, r1*d1+i1*c1]
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_cmul_2(
    a: std::arch::aarch64::float32x4_t,
    b: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    // 1. dup_r = [r0, r0, r1, r1]
    let dup_r = vcombine_f32(
        vdup_lane_f32(vget_low_f32(a), 0),
        vdup_lane_f32(vget_high_f32(a), 0),
    );
    // 2. t1 = [r0*c0, r0*d0, r1*c1, r1*d1]
    let t1 = vmulq_f32(dup_r, b);
    // 3. neg_a = [-r0, -i0, -r1, -i1]
    let neg_a = vnegq_f32(a);
    // 4. swap_neg = [-i0, i0, -i1, i1] via vtrn
    let swap_neg = vtrnq_f32(neg_a, a).1;
    // 5. rev_b = [d0, c0, d1, c1]
    let rev_b = vrev64q_f32(b);
    // 6. t2 = [-i0*d0, i0*c0, -i1*d1, i1*c1]
    let t2 = vmulq_f32(swap_neg, rev_b);
    // 7. result = [r0*c0-i0*d0, r0*d0+i0*c0, r1*c1-i1*d1, r1*d1+i1*c1]
    vaddq_f32(t1, t2)
}

/// NEON-optimized radix-3 butterfly inner loop
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn kf_bfly3_neon_inner(
    fout: &mut [KissCpx],
    fstride: usize,
    twiddles: &[KissCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    use std::arch::aarch64::*;

    let m2 = 2 * m;
    let epi3_i: f32 = -0.86602540;
    let stride2 = fstride * 2;
    let fout_ptr = fout.as_mut_ptr() as *mut f32;
    let tw_ptr = twiddles.as_ptr() as *const f32;

    for i in 0..n {
        let base = i * mm;
        let mut tw1 = 0usize;
        let mut tw2 = 0usize;

        let m_vec = m & !1;
        let mut j = 0;

        // Process 2 inner iterations at once
        while j < m_vec {
            let idx0 = base + j;

            // Load 2 fm values: fout[idx0+m], fout[idx0+m+1]
            let fm = vld1q_f32(fout_ptr.add(2 * (idx0 + m)));

            // Gather 2 tw1 values from non-contiguous addresses
            let tw1_lo = vld1_f32(tw_ptr.add(2 * tw1));
            let tw1_hi = vld1_f32(tw_ptr.add(2 * (tw1 + fstride)));
            let tw1_v = vcombine_f32(tw1_lo, tw1_hi);

            let s1 = neon_cmul_2(fm, tw1_v);

            // Load 2 fm2 values
            let fm2 = vld1q_f32(fout_ptr.add(2 * (idx0 + m2)));

            // Gather 2 tw2 values
            let tw2_lo = vld1_f32(tw_ptr.add(2 * tw2));
            let tw2_hi = vld1_f32(tw_ptr.add(2 * (tw2 + stride2)));
            let tw2_v = vcombine_f32(tw2_lo, tw2_hi);

            let s2 = neon_cmul_2(fm2, tw2_v);

            // scratch3 = s1 + s2, scratch0 = s1 - s2
            let s3 = vaddq_f32(s1, s2);
            let s0 = vsubq_f32(s1, s2);

            // half_scratch3 = s3 * 0.5
            let half_s3 = vmulq_n_f32(s3, 0.5);

            // Load 2 fout[idx] values
            let f0 = vld1q_f32(fout_ptr.add(2 * idx0));

            // fout_m = f0 - half_s3
            let fout_m = vsubq_f32(f0, half_s3);

            // scratch0_scaled = s0 * epi3_i
            let s0_scaled = vmulq_n_f32(s0, epi3_i);

            // fout[idx] = f0 + s3
            vst1q_f32(fout_ptr.add(2 * idx0), vaddq_f32(f0, s3));

            // fout[idx+m] = [fout_m.r - s0_scaled.i, fout_m.i + s0_scaled.r, ...]
            // For each 64-bit pair [fout_m_r, fout_m_i] and [s0_scaled_r, s0_scaled_i]:
            //   result = [fout_m_r - s0_scaled_i, fout_m_i + s0_scaled_r]
            // = fout_m + [-s0_scaled_i, s0_scaled_r]
            // = fout_m + vext_f32(vneg(s0_scaled_lo), s0_scaled_lo, 1)
            let neg_s0 = vnegq_f32(s0_scaled);
            let adj_lo = vext_f32(vget_low_f32(neg_s0), vget_low_f32(s0_scaled), 1);
            let adj_hi = vext_f32(vget_high_f32(neg_s0), vget_high_f32(s0_scaled), 1);
            let adj_m = vcombine_f32(adj_lo, adj_hi);
            vst1q_f32(fout_ptr.add(2 * (idx0 + m)), vaddq_f32(fout_m, adj_m));

            // fout[idx+m2] = [fout_m.r + s0_scaled.i, fout_m.i - s0_scaled.r, ...]
            // = fout_m + [s0_scaled_i, -s0_scaled_r]
            // = fout_m + vext_f32(s0_scaled_lo, neg_s0_scaled_lo, 1)
            let adj2_lo = vext_f32(vget_low_f32(s0_scaled), vget_low_f32(neg_s0), 1);
            let adj2_hi = vext_f32(vget_high_f32(s0_scaled), vget_high_f32(neg_s0), 1);
            let adj_m2 = vcombine_f32(adj2_lo, adj2_hi);
            vst1q_f32(fout_ptr.add(2 * (idx0 + m2)), vaddq_f32(fout_m, adj_m2));

            tw1 += 2 * fstride;
            tw2 += 2 * stride2;
            j += 2;
        }

        // Scalar tail for odd m
        for j in m_vec..m {
            let idx = base + j;
            let scratch1 = c_mul(&fout[idx + m], &twiddles[tw1]);
            let scratch2 = c_mul(&fout[idx + m2], &twiddles[tw2]);
            let scratch3 = c_add(&scratch1, &scratch2);
            let scratch0 = c_sub(&scratch1, &scratch2);
            let half_scratch3 = KissCpx::new(scratch3.r * 0.5, scratch3.i * 0.5);
            let fout_m = KissCpx::new(fout[idx].r - half_scratch3.r, fout[idx].i - half_scratch3.i);
            let scratch0_scaled = KissCpx::new(scratch0.r * epi3_i, scratch0.i * epi3_i);
            fout[idx] = c_add(&fout[idx], &scratch3);
            fout[idx + m] =
                KissCpx::new(fout_m.r - scratch0_scaled.i, fout_m.i + scratch0_scaled.r);
            fout[idx + m2] =
                KissCpx::new(fout_m.r + scratch0_scaled.i, fout_m.i - scratch0_scaled.r);
            tw1 += fstride;
            tw2 += stride2;
        }
    }
}

/// NEON-optimized radix-5 butterfly inner loop
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn kf_bfly5_neon_inner(
    fout: &mut [KissCpx],
    fstride: usize,
    twiddles: &[KissCpx],
    m: usize,
    n: usize,
    mm: usize,
) {
    use std::arch::aarch64::*;

    let m2 = 2 * m;
    let m3 = 3 * m;
    let m4 = 4 * m;
    let stride2 = fstride * 2;
    let stride3 = fstride * 3;
    let stride4 = fstride * 4;

    // ya = exp(-2*pi*i/5), yb = exp(-4*pi*i/5)
    let ya_r: f32 = 0.30901699;
    let ya_i: f32 = -0.95105652;
    let yb_r: f32 = -0.80901699;
    let yb_i: f32 = -0.58778525;

    let fout_ptr = fout.as_mut_ptr() as *mut f32;
    let tw_ptr = twiddles.as_ptr() as *const f32;

    for i in 0..n {
        let base = i * mm;
        let mut tw1 = 0usize;
        let mut tw2 = 0usize;
        let mut tw3 = 0usize;
        let mut tw4 = 0usize;

        let m_vec = m & !1;
        let mut j = 0;

        while j < m_vec {
            let idx0 = base + j;

            // Load fout[idx0..idx0+1] (scratch0 = original values)
            let f0 = vld1q_f32(fout_ptr.add(2 * idx0)); // [f0.r, f0.i, f1.r, f1.i]

            // Load 2 fm, fm2, fm3, fm4 values
            let fm1_v = vld1q_f32(fout_ptr.add(2 * (idx0 + m)));
            let fm2_v = vld1q_f32(fout_ptr.add(2 * (idx0 + m2)));
            let fm3_v = vld1q_f32(fout_ptr.add(2 * (idx0 + m3)));
            let fm4_v = vld1q_f32(fout_ptr.add(2 * (idx0 + m4)));

            // Gather twiddle values (non-contiguous)
            let tw1_v = vcombine_f32(
                vld1_f32(tw_ptr.add(2 * tw1)),
                vld1_f32(tw_ptr.add(2 * (tw1 + fstride))),
            );
            let tw2_v = vcombine_f32(
                vld1_f32(tw_ptr.add(2 * tw2)),
                vld1_f32(tw_ptr.add(2 * (tw2 + stride2))),
            );
            let tw3_v = vcombine_f32(
                vld1_f32(tw_ptr.add(2 * tw3)),
                vld1_f32(tw_ptr.add(2 * (tw3 + stride3))),
            );
            let tw4_v = vcombine_f32(
                vld1_f32(tw_ptr.add(2 * tw4)),
                vld1_f32(tw_ptr.add(2 * (tw4 + stride4))),
            );

            // Complex multiplies
            let s1 = neon_cmul_2(fm1_v, tw1_v);
            let s2 = neon_cmul_2(fm2_v, tw2_v);
            let s3 = neon_cmul_2(fm3_v, tw3_v);
            let s4 = neon_cmul_2(fm4_v, tw4_v);

            // Deinterleave s1..s4 into real and imaginary parts
            // s1 = [s1_r0, s1_i0, s1_r1, s1_i1], etc.
            // We need individual real/imaginary for each, plus cross-terms.
            // The radix-5 butterfly has many cross-terms, so extract scalars.

            let s1_arr: [f32; 4] = std::mem::transmute(s1);
            let s2_arr: [f32; 4] = std::mem::transmute(s2);
            let s3_arr: [f32; 4] = std::mem::transmute(s3);
            let s4_arr: [f32; 4] = std::mem::transmute(s4);
            let f0_arr: [f32; 4] = std::mem::transmute(f0);

            // Process 2 iterations (j and j+1) using extracted scalars
            for k in 0..2 {
                let s1r = s1_arr[2 * k];
                let s1i = s1_arr[2 * k + 1];
                let s2r = s2_arr[2 * k];
                let s2i = s2_arr[2 * k + 1];
                let s3r = s3_arr[2 * k];
                let s3i = s3_arr[2 * k + 1];
                let s4r = s4_arr[2 * k];
                let s4i = s4_arr[2 * k + 1];
                let f0r = f0_arr[2 * k];
                let f0i = f0_arr[2 * k + 1];

                let s7r = s1r + s4r;
                let s7i = s1i + s4i;
                let s10r = s1r - s4r;
                let s10i = s1i - s4i;
                let s8r = s2r + s3r;
                let s8i = s2i + s3i;
                let s9r = s2r - s3r;
                let s9i = s2i - s3i;

                let idx = idx0 + k;

                // F0
                fout[idx].r = f0r + s7r + s8r;
                fout[idx].i = f0i + s7i + s8i;

                // scratch5 and scratch6
                let s5r = f0r + s7r * ya_r + s8r * yb_r;
                let s5i = f0i + s7i * ya_r + s8i * yb_r;
                let s6r = s10i * ya_i + s9i * yb_i;
                let s6i = -(s10r * ya_i + s9r * yb_i);

                fout[idx + m].r = s5r - s6r;
                fout[idx + m].i = s5i - s6i;
                fout[idx + m4].r = s5r + s6r;
                fout[idx + m4].i = s5i + s6i;

                let s11r = f0r + s7r * yb_r + s8r * ya_r;
                let s11i = f0i + s7i * yb_r + s8i * ya_r;
                let s12r = s9i * ya_i - s10i * yb_i;
                let s12i = s10r * yb_i - s9r * ya_i;

                fout[idx + m2].r = s11r + s12r;
                fout[idx + m2].i = s11i + s12i;
                fout[idx + m3].r = s11r - s12r;
                fout[idx + m3].i = s11i - s12i;
            }

            tw1 += 2 * fstride;
            tw2 += 2 * stride2;
            tw3 += 2 * stride3;
            tw4 += 2 * stride4;
            j += 2;
        }

        // Scalar tail for odd m
        for j in m_vec..m {
            let idx = base + j;
            let scratch0 = fout[idx];
            let scratch1 = c_mul(&fout[idx + m], &twiddles[tw1]);
            let scratch2 = c_mul(&fout[idx + m2], &twiddles[tw2]);
            let scratch3 = c_mul(&fout[idx + m3], &twiddles[tw3]);
            let scratch4 = c_mul(&fout[idx + m4], &twiddles[tw4]);
            let scratch7 = c_add(&scratch1, &scratch4);
            let scratch10 = c_sub(&scratch1, &scratch4);
            let scratch8 = c_add(&scratch2, &scratch3);
            let scratch9 = c_sub(&scratch2, &scratch3);
            fout[idx].r = scratch0.r + scratch7.r + scratch8.r;
            fout[idx].i = scratch0.i + scratch7.i + scratch8.i;
            let scratch5 = KissCpx::new(
                scratch0.r + scratch7.r * ya_r + scratch8.r * yb_r,
                scratch0.i + scratch7.i * ya_r + scratch8.i * yb_r,
            );
            let scratch6 = KissCpx::new(
                scratch10.i * ya_i + scratch9.i * yb_i,
                -(scratch10.r * ya_i + scratch9.r * yb_i),
            );
            fout[idx + m] = c_sub(&scratch5, &scratch6);
            fout[idx + m4] = c_add(&scratch5, &scratch6);
            let scratch11 = KissCpx::new(
                scratch0.r + scratch7.r * yb_r + scratch8.r * ya_r,
                scratch0.i + scratch7.i * yb_r + scratch8.i * ya_r,
            );
            let scratch12 = KissCpx::new(
                scratch9.i * ya_i - scratch10.i * yb_i,
                scratch10.r * yb_i - scratch9.r * ya_i,
            );
            fout[idx + m2] = c_add(&scratch11, &scratch12);
            fout[idx + m3] = c_sub(&scratch11, &scratch12);
            tw1 += fstride;
            tw2 += stride2;
            tw3 += stride3;
            tw4 += stride4;
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

    // Merge bitrev copy + conjugation (negate imag) in one pass
    for i in 0..nfft {
        let rev = st.bitrev[i] as usize;
        fout[rev] = KissCpx::new(fin[i].r, -fin[i].i);
    }

    opus_fft_impl(st, fout);

    // Conjugate output in one pass (negate imag back)
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
