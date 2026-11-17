//! The 320-point forward FFT of the neural network code: libopus
//! `opus_fft_c` (float) on the precomputed state `kfft` of
//! `dnn/lpcnet_tables.c` (factors 5, 4, 4, 4), with the butterflies of
//! `celt/kiss_fft.c`.

use super::lpcnet_tables::{FFT_BITREV, FFT_FACTORS, FFT_SCALE, FFT_TWIDDLES};

/// A complex value (`kiss_fft_cpx`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cpx {
    pub r: f32,
    pub i: f32,
}

/// The FFT size.
pub const NFFT: usize = 320;

#[inline(always)]
fn tw(k: usize) -> Cpx {
    let t = FFT_TWIDDLES[k];
    Cpx { r: t[0], i: t[1] }
}

/// `C_MUL`.
#[inline(always)]
fn cmul(a: Cpx, b: Cpx) -> Cpx {
    Cpx { r: a.r * b.r - a.i * b.i, i: a.r * b.i + a.i * b.r }
}

#[inline(always)]
fn cadd(a: Cpx, b: Cpx) -> Cpx {
    Cpx { r: a.r + b.r, i: a.i + b.i }
}

#[inline(always)]
fn csub(a: Cpx, b: Cpx) -> Cpx {
    Cpx { r: a.r - b.r, i: a.i - b.i }
}

/// `kf_bfly4`.
fn bfly4(f: &mut [Cpx], fstride: usize, m: usize, n: usize, mm: usize) {
    if m == 1 {
        for k in 0..n {
            let b = 4 * k;
            let s0 = csub(f[b], f[b + 2]);
            f[b] = cadd(f[b], f[b + 2]);
            let s1 = cadd(f[b + 1], f[b + 3]);
            f[b + 2] = csub(f[b], s1);
            f[b] = cadd(f[b], s1);
            let s1 = csub(f[b + 1], f[b + 3]);
            f[b + 1] = Cpx { r: s0.r + s1.i, i: s0.i - s1.r };
            f[b + 3] = Cpx { r: s0.r - s1.i, i: s0.i + s1.r };
        }
    } else {
        let (m2, m3) = (2 * m, 3 * m);
        for i in 0..n {
            let base = i * mm;
            for j in 0..m {
                let p = base + j;
                let s0 = cmul(f[p + m], tw(j * fstride));
                let s1 = cmul(f[p + m2], tw(j * fstride * 2));
                let s2 = cmul(f[p + m3], tw(j * fstride * 3));
                let s5 = csub(f[p], s1);
                f[p] = cadd(f[p], s1);
                let s3 = cadd(s0, s2);
                let s4 = csub(s0, s2);
                f[p + m2] = csub(f[p], s3);
                f[p] = cadd(f[p], s3);
                f[p + m] = Cpx { r: s5.r + s4.i, i: s5.i - s4.r };
                f[p + m3] = Cpx { r: s5.r - s4.i, i: s5.i + s4.r };
            }
        }
    }
}

/// `kf_bfly5` (float: `ya`, `yb` are twiddles).
fn bfly5(f: &mut [Cpx], fstride: usize, m: usize, n: usize, mm: usize) {
    let ya = tw(fstride * m);
    let yb = tw(fstride * 2 * m);
    for i in 0..n {
        let base = i * mm;
        for u in 0..m {
            let (p0, p1, p2, p3, p4) = (base + u, base + u + m, base + u + 2 * m, base + u + 3 * m, base + u + 4 * m);
            let s0 = f[p0];
            let s1 = cmul(f[p1], tw(u * fstride));
            let s2 = cmul(f[p2], tw(2 * u * fstride));
            let s3 = cmul(f[p3], tw(3 * u * fstride));
            let s4 = cmul(f[p4], tw(4 * u * fstride));
            let s7 = cadd(s1, s4);
            let s10 = csub(s1, s4);
            let s8 = cadd(s2, s3);
            let s9 = csub(s2, s3);
            f[p0].r += s7.r + s8.r;
            f[p0].i += s7.i + s8.i;
            let s5 = Cpx { r: s0.r + (s7.r * ya.r + s8.r * yb.r), i: s0.i + (s7.i * ya.r + s8.i * yb.r) };
            let s6 = Cpx { r: s10.i * ya.i + s9.i * yb.i, i: -(s10.r * ya.i + s9.r * yb.i) };
            f[p1] = csub(s5, s6);
            f[p4] = cadd(s5, s6);
            let s11 = Cpx { r: s0.r + (s7.r * yb.r + s8.r * ya.r), i: s0.i + (s7.i * yb.r + s8.i * ya.r) };
            let s12 = Cpx { r: s9.i * ya.i - s10.i * yb.i, i: s10.r * yb.i - s9.r * ya.i };
            f[p2] = cadd(s11, s12);
            f[p3] = csub(s11, s12);
        }
    }
}

/// `opus_fft_c(&kfft, fin, fout)`: scaled by 1/320.
pub fn opus_fft(fin: &[Cpx; NFFT], fout: &mut [Cpx; NFFT]) {
    for (i, x) in fin.iter().enumerate() {
        fout[FFT_BITREV[i] as usize] = Cpx { r: FFT_SCALE * x.r, i: FFT_SCALE * x.i };
    }
    // opus_fft_impl with shift 0.
    let f = &FFT_FACTORS;
    let mut fstride = [0usize; 8];
    fstride[0] = 1;
    let mut l = 0;
    loop {
        let p = f[2 * l];
        let m = f[2 * l + 1];
        fstride[l + 1] = fstride[l] * p;
        l += 1;
        if m == 1 {
            break;
        }
    }
    let mut m = f[2 * l - 1];
    for i in (0..l).rev() {
        let m2 = if i != 0 { f[2 * i - 1] } else { 1 };
        match f[2 * i] {
            4 => bfly4(fout, fstride[i], m, fstride[i], m2),
            5 => bfly5(fout, fstride[i], m, fstride[i], m2),
            _ => unreachable!("kfft has only radix 4 and 5 stages"),
        }
        m = m2;
    }
}
