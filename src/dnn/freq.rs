//! Band energies, cepstra and LPC of the LPCNet features (libopus
//! `dnn/freq.c`). The mixed float/double arithmetic of the C code is kept:
//! where C promotes to `double`, so does this port.

#[cfg(not(feature = "std"))]
use crate::compat::Math;

use super::burg::silk_burg_analysis;
use super::fft::{Cpx, NFFT, opus_fft};
use super::lpcnet_tables::{DCT_TABLE, HALF_WINDOW};

pub const LPC_ORDER: usize = 16;
pub const PREEMPHASIS: f32 = 0.85;
pub const FRAME_SIZE_5MS: usize = 2;
pub const OVERLAP_SIZE_5MS: usize = 2;
pub const TRAINING_OFFSET_5MS: usize = 1;
pub const WINDOW_SIZE_5MS: usize = FRAME_SIZE_5MS + OVERLAP_SIZE_5MS;
pub const FRAME_SIZE: usize = 80 * FRAME_SIZE_5MS;
pub const OVERLAP_SIZE: usize = 80 * OVERLAP_SIZE_5MS;
pub const TRAINING_OFFSET: usize = 80 * TRAINING_OFFSET_5MS;
pub const WINDOW_SIZE: usize = FRAME_SIZE + OVERLAP_SIZE;
pub const FREQ_SIZE: usize = WINDOW_SIZE / 2 + 1;
pub const NB_BANDS: usize = 18;
pub const NB_BANDS_1: usize = NB_BANDS - 1;

const EBAND5MS: [usize; 18] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40,
];

const COMPENSATION: [f32; 18] = [
    0.8, 1., 1., 1., 1., 1., 1., 1., 0.666667, 0.5, 0.5, 0.5, 0.333333, 0.25, 0.25, 0.2, 0.166667,
    0.173913,
];

/// libopus `MAX16` on doubles: `a > b ? a : b`.
#[inline(always)]
fn maxd(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

/// `compute_band_energy_inverse`.
fn compute_band_energy_inverse(band_e: &mut [f32; NB_BANDS], x: &[Cpx]) {
    let mut sum = [0.0f32; NB_BANDS];
    for i in 0..NB_BANDS - 1 {
        let band_size = (EBAND5MS[i + 1] - EBAND5MS[i]) * WINDOW_SIZE_5MS;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let v = x[EBAND5MS[i] * WINDOW_SIZE_5MS + j];
            let mut tmp = v.r * v.r;
            tmp += v.i * v.i;
            let tmp = (1.0f64 / (tmp as f64 + 1e-9)) as f32;
            sum[i] += (1.0 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[0] *= 2.0;
    sum[NB_BANDS - 1] *= 2.0;
    *band_e = sum;
}

/// `lpcn_lpc` (float build: the Q shifts are no-ops). Returns the error.
fn lpcn_lpc(lpc: &mut [f32], rc: &mut [f32], ac: &[f32], p: usize) -> f32 {
    let mut error = ac[0];
    lpc[..p].fill(0.0);
    rc[..p].fill(0.0);
    if ac[0] != 0.0 {
        for i in 0..p {
            let mut rr = 0.0f32;
            for j in 0..i {
                rr += lpc[j] * ac[i - j];
            }
            rr += ac[i + 1];
            let r = -rr / error;
            rc[i] = r;
            lpc[i] = r;
            for j in 0..(i + 1) >> 1 {
                let tmp1 = lpc[j];
                let tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + r * tmp2;
                lpc[i - 1 - j] = tmp2 + r * tmp1;
            }
            error -= r * r * error;
            if error < 0.001f32 * ac[0] {
                break;
            }
        }
    }
    error
}

/// `lpcn_compute_band_energy`.
pub fn lpcn_compute_band_energy(band_e: &mut [f32; NB_BANDS], x: &[Cpx]) {
    let mut sum = [0.0f32; NB_BANDS];
    for i in 0..NB_BANDS - 1 {
        let band_size = (EBAND5MS[i + 1] - EBAND5MS[i]) * WINDOW_SIZE_5MS;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let v = x[EBAND5MS[i] * WINDOW_SIZE_5MS + j];
            let mut tmp = v.r * v.r;
            tmp += v.i * v.i;
            sum[i] += (1.0 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[0] *= 2.0;
    sum[NB_BANDS - 1] *= 2.0;
    *band_e = sum;
}

/// `compute_burg_cepstrum`.
fn compute_burg_cepstrum(pcm: &[f32], burg_cepstrum: &mut [f32], len: usize, order: usize) {
    assert!(order <= LPC_ORDER && len <= FRAME_SIZE);
    let mut burg_in = [0.0f32; FRAME_SIZE];
    let mut burg_lpc = [0.0f32; LPC_ORDER];
    let mut x = [0.0f32; WINDOW_SIZE];
    let mut eburg = [0.0f32; NB_BANDS];
    let mut lpc = [Cpx::default(); FREQ_SIZE];
    let mut ly = [0.0f32; NB_BANDS];
    let mut log_max: f32 = -2.0;
    let mut follow: f32 = -2.0;
    for i in 0..len - 1 {
        burg_in[i] = pcm[i + 1] - PREEMPHASIS * pcm[i];
    }
    let mut g = silk_burg_analysis(&mut burg_lpc, &burg_in, 1e-3f32, len - 1, 1, order);
    g /= (len - 2 * (order - 1)) as f32;
    x[0] = 1.0;
    for i in 0..order {
        x[i + 1] = (-burg_lpc[i] as f64 * 0.995f64.powf((i + 1) as f64)) as f32;
    }
    forward_transform(&mut lpc, &x);
    compute_band_energy_inverse(&mut eburg, &lpc);
    let w = 1.0f32 / (WINDOW_SIZE as f32 * WINDOW_SIZE as f32 * WINDOW_SIZE as f32);
    for e in eburg.iter_mut() {
        *e = (*e as f64 * (0.45 * g as f64 * w as f64)) as f32;
    }
    for i in 0..NB_BANDS {
        let l = (1e-2 + eburg[i] as f64).log10() as f32;
        ly[i] = maxd((log_max - 8.0) as f64, maxd(follow as f64 - 2.5, l as f64)) as f32;
        log_max = if log_max > ly[i] { log_max } else { ly[i] };
        follow = maxd(follow as f64 - 2.5, ly[i] as f64) as f32;
    }
    dct(burg_cepstrum, &ly);
    burg_cepstrum[0] += -4.0;
}

/// `burg_cepstral_analysis`: the cepstra of the two halves of a frame, as
/// their mean and difference.
pub fn burg_cepstral_analysis(ceps: &mut [f32], x: &[f32]) {
    compute_burg_cepstrum(x, &mut ceps[..NB_BANDS], FRAME_SIZE / 2, LPC_ORDER);
    compute_burg_cepstrum(
        &x[FRAME_SIZE / 2..],
        &mut ceps[NB_BANDS..2 * NB_BANDS],
        FRAME_SIZE / 2,
        LPC_ORDER,
    );
    for i in 0..NB_BANDS {
        let c0 = ceps[i];
        let c1 = ceps[NB_BANDS + i];
        ceps[i] = (0.5 * (c0 + c1) as f64) as f32;
        ceps[NB_BANDS + i] = c0 - c1;
    }
}

/// `interp_band_gain`.
fn interp_band_gain(g: &mut [f32; FREQ_SIZE], band_e: &[f32]) {
    for i in 0..NB_BANDS - 1 {
        let band_size = (EBAND5MS[i + 1] - EBAND5MS[i]) * WINDOW_SIZE_5MS;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            g[EBAND5MS[i] * WINDOW_SIZE_5MS + j] = (1.0 - frac) * band_e[i] + frac * band_e[i + 1];
        }
    }
}

/// `dct`.
pub fn dct(out: &mut [f32], input: &[f32]) {
    let s = (2.0f64 / NB_BANDS as f64).sqrt();
    for i in 0..NB_BANDS {
        let mut sum = 0.0f32;
        for j in 0..NB_BANDS {
            sum += input[j] * DCT_TABLE[j * NB_BANDS + i];
        }
        out[i] = (sum as f64 * s) as f32;
    }
}

/// `idct`.
fn idct(out: &mut [f32], input: &[f32]) {
    let s = (2.0f64 / NB_BANDS as f64).sqrt();
    for i in 0..NB_BANDS {
        let mut sum = 0.0f32;
        for j in 0..NB_BANDS {
            sum += input[j] * DCT_TABLE[i * NB_BANDS + j];
        }
        out[i] = (sum as f64 * s) as f32;
    }
}

/// `forward_transform`: the first `FREQ_SIZE` bins of the FFT of the real
/// `input[..WINDOW_SIZE]`.
pub fn forward_transform(out: &mut [Cpx], input: &[f32]) {
    let mut x = [Cpx::default(); NFFT];
    let mut y = [Cpx::default(); NFFT];
    for (c, &v) in x.iter_mut().zip(&input[..WINDOW_SIZE]) {
        c.r = v;
    }
    opus_fft(&x, &mut y);
    out[..FREQ_SIZE].copy_from_slice(&y[..FREQ_SIZE]);
}

/// `inverse_transform`.
fn inverse_transform(out: &mut [f32; WINDOW_SIZE], input: &[Cpx]) {
    let mut x = [Cpx::default(); NFFT];
    let mut y = [Cpx::default(); NFFT];
    x[..FREQ_SIZE].copy_from_slice(&input[..FREQ_SIZE]);
    for i in FREQ_SIZE..WINDOW_SIZE {
        x[i].r = x[WINDOW_SIZE - i].r;
        x[i].i = -x[WINDOW_SIZE - i].i;
    }
    opus_fft(&x, &mut y);
    out[0] = WINDOW_SIZE as f32 * y[0].r;
    for i in 1..WINDOW_SIZE {
        out[i] = WINDOW_SIZE as f32 * y[WINDOW_SIZE - i].r;
    }
}

/// `lpc_from_bands`.
fn lpc_from_bands(lpc: &mut [f32], ex: &[f32]) -> f32 {
    let mut ac = [0.0f32; LPC_ORDER + 1];
    let mut rc = [0.0f32; LPC_ORDER];
    let mut xr = [0.0f32; FREQ_SIZE];
    let mut x_auto_f = [Cpx::default(); FREQ_SIZE];
    let mut x_auto = [0.0f32; WINDOW_SIZE];
    interp_band_gain(&mut xr, ex);
    xr[FREQ_SIZE - 1] = 0.0;
    for (c, &v) in x_auto_f.iter_mut().zip(&xr) {
        c.r = v;
    }
    inverse_transform(&mut x_auto, &x_auto_f);
    ac.copy_from_slice(&x_auto[..LPC_ORDER + 1]);
    // -40 dB noise floor; 320/12 is an integer division in C.
    ac[0] = (ac[0] as f64 + (ac[0] as f64 * 1e-4 + (320 / 12) as f64 / 38.0)) as f32;
    for i in 1..LPC_ORDER + 1 {
        ac[i] = (ac[i] as f64 * (1.0 - 6e-5 * i as f64 * i as f64)) as f32;
    }
    lpcn_lpc(lpc, &mut rc, &ac, LPC_ORDER)
}

/// `lpc_weighting`.
pub fn lpc_weighting(lpc: &mut [f32], gamma: f32) {
    let mut gamma_i = gamma;
    for v in lpc[..LPC_ORDER].iter_mut() {
        *v *= gamma_i;
        gamma_i *= gamma;
    }
}

/// `lpc_from_cepstrum`.
pub fn lpc_from_cepstrum(lpc: &mut [f32], cepstrum: &[f32]) -> f32 {
    let mut ex = [0.0f32; NB_BANDS];
    let mut tmp = [0.0f32; NB_BANDS];
    tmp.copy_from_slice(&cepstrum[..NB_BANDS]);
    tmp[0] += 4.0;
    idct(&mut ex, &tmp);
    for i in 0..NB_BANDS {
        ex[i] = (10.0f64.powf(ex[i] as f64) * COMPENSATION[i] as f64) as f32;
    }
    lpc_from_bands(lpc, &ex)
}

/// `apply_window`.
pub fn apply_window(x: &mut [f32]) {
    for i in 0..OVERLAP_SIZE {
        x[i] *= HALF_WINDOW[i];
        x[WINDOW_SIZE - 1 - i] *= HALF_WINDOW[i];
    }
}
