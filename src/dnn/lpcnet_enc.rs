//! The LPCNet feature analysis (libopus `dnn/lpcnet_enc.c`): cepstrum,
//! pitch (PitchDNN) and pitch correlation of each 10 ms frame at 16 kHz.
//! Deep PLC runs it on the decoded signal, DRED on the encoder input.

#[cfg(not(feature = "std"))]
use crate::compat::Math;

use super::common::{celt_fir, celt_inner_prod, celt_log2, celt_pitch_xcorr};
use super::fft::Cpx;
use super::freq::{
    FRAME_SIZE, FREQ_SIZE, LPC_ORDER, NB_BANDS, OVERLAP_SIZE, PREEMPHASIS, TRAINING_OFFSET,
    WINDOW_SIZE, apply_window, dct, forward_transform, lpc_from_cepstrum, lpcn_compute_band_energy,
};
use super::models::pitchdnn::PitchDnn;
use super::pitchdnn::{PITCH_MAX_PERIOD, PITCH_MIN_PERIOD, PitchDnnState, compute_pitchdnn};
use super::vec::c_int;
use crate::Arch;

pub const NB_FEATURES: usize = 20;
pub const NB_TOTAL_FEATURES: usize = 36;
pub const LPCNET_FRAME_SIZE: usize = 160;
pub const PITCH_FRAME_SIZE: usize = 320;
pub const PITCH_BUF_SIZE: usize = PITCH_MAX_PERIOD + PITCH_FRAME_SIZE;
pub const PITCH_IF_MAX_FREQ: usize = 30;
pub const PITCH_IF_FEATURES: usize = 3 * PITCH_IF_MAX_FREQ - 2;

/// libopus `LPCNetEncState`.
#[derive(Clone)]
pub struct LpcnetEncState<'a> {
    pub pitchdnn: PitchDnnState<'a>,
    pub analysis_mem: [f32; OVERLAP_SIZE],
    pub mem_preemph: f32,
    pub prev_if: [Cpx; PITCH_IF_MAX_FREQ],
    pub if_features: [f32; PITCH_IF_FEATURES],
    pub xcorr_features: [f32; PITCH_MAX_PERIOD - PITCH_MIN_PERIOD],
    pub dnn_pitch: f32,
    pub pitch_mem: [f32; LPC_ORDER],
    pub pitch_filt: f32,
    pub exc_buf: [f32; PITCH_BUF_SIZE],
    pub lp_buf: [f32; PITCH_BUF_SIZE],
    pub lp_mem: [f32; 4],
    pub lpc: [f32; LPC_ORDER],
    pub features: [f32; NB_TOTAL_FEATURES],
    pub sig_mem: [f32; LPC_ORDER],
    pub burg_cepstrum: [f32; 2 * NB_BANDS],
}

impl<'a> LpcnetEncState<'a> {
    /// `lpcnet_encoder_init`: zero state with the PitchDNN model.
    pub fn new(pitchdnn: PitchDnn<'a>) -> Self {
        LpcnetEncState {
            pitchdnn: PitchDnnState::new(pitchdnn),
            analysis_mem: [0.0; OVERLAP_SIZE],
            mem_preemph: 0.0,
            prev_if: [Cpx::default(); PITCH_IF_MAX_FREQ],
            if_features: [0.0; PITCH_IF_FEATURES],
            xcorr_features: [0.0; PITCH_MAX_PERIOD - PITCH_MIN_PERIOD],
            dnn_pitch: 0.0,
            pitch_mem: [0.0; LPC_ORDER],
            pitch_filt: 0.0,
            exc_buf: [0.0; PITCH_BUF_SIZE],
            lp_buf: [0.0; PITCH_BUF_SIZE],
            lp_mem: [0.0; 4],
            lpc: [0.0; LPC_ORDER],
            features: [0.0; NB_TOTAL_FEATURES],
            sig_mem: [0.0; LPC_ORDER],
            burg_cepstrum: [0.0; 2 * NB_BANDS],
        }
    }

    /// `lpcnet_encoder_init` again: clears everything but the model.
    pub fn reset(&mut self) {
        *self = LpcnetEncState::new(self.pitchdnn.model);
    }
}

/// `frame_analysis`.
fn frame_analysis(
    st: &mut LpcnetEncState<'_>,
    x_out: &mut [Cpx; FREQ_SIZE],
    ex: &mut [f32; NB_BANDS],
    input: &[f32],
) {
    let mut x = [0.0f32; WINDOW_SIZE];
    x[..OVERLAP_SIZE].copy_from_slice(&st.analysis_mem);
    x[OVERLAP_SIZE..].copy_from_slice(&input[..FRAME_SIZE]);
    st.analysis_mem
        .copy_from_slice(&input[FRAME_SIZE - OVERLAP_SIZE..FRAME_SIZE]);
    apply_window(&mut x);
    forward_transform(x_out, &x);
    lpcn_compute_band_energy(ex, x_out);
}

/// `biquad` (the reordered form of the C code, with its `+ 1e-30f`).
fn biquad(y: &mut [f32], mem: &mut [f32], b: &[f32; 2], a: &[f32; 2], n: usize) {
    let mut mem0 = mem[0];
    let mut mem1 = mem[1];
    for v in y[..n].iter_mut() {
        let xi = *v;
        let yi = xi + mem0;
        let mem00 = mem0;
        mem0 = (b[0] - a[0]) * xi + mem1 - a[0] * mem0;
        mem1 = (b[1] - a[1]) * xi + 1e-30f32 - a[1] * mem00;
        *v = yi;
    }
    mem[0] = mem0;
    mem[1] = mem1;
}

/// `celt_log10`.
#[inline]
fn celt_log10(x: f32) -> f32 {
    0.3010299957f32 * celt_log2(x)
}

#[inline(always)]
fn clamp1(x: f32) -> f32 {
    // MAX16(-1.f, MIN16(1.f, x))
    let m = if 1.0 < x { 1.0 } else { x };
    if -1.0 > m { -1.0 } else { m }
}

/// `compute_frame_features`.
pub fn compute_frame_features(st: &mut LpcnetEncState<'_>, input: &[f32], arch: Arch) {
    let mut aligned_in = [0.0f32; FRAME_SIZE];
    let mut ly = [0.0f32; NB_BANDS];
    let mut xf = [Cpx::default(); FREQ_SIZE];
    let mut ex = [0.0f32; NB_BANDS];
    let mut xcorr = [0.0f32; PITCH_MAX_PERIOD];
    let mut x = [0.0f32; FRAME_SIZE + LPC_ORDER];
    let mut ener_norm = [0.0f32; PITCH_MAX_PERIOD - PITCH_MIN_PERIOD];
    const LP_B: [f32; 2] = [-0.84946, 1.0];
    const LP_A: [f32; 2] = [-1.54220, 0.70781];
    aligned_in[..TRAINING_OFFSET]
        .copy_from_slice(&st.analysis_mem[OVERLAP_SIZE - TRAINING_OFFSET..]);
    frame_analysis(st, &mut xf, &mut ex, input);
    st.if_features[0] =
        clamp1((1.0f32 / 64.0) * (10.0f32 * celt_log10(1e-15f32 + xf[0].r * xf[0].r) - 6.0));
    for i in 1..PITCH_IF_MAX_FREQ {
        let a = xf[i];
        let b = st.prev_if[i];
        // C_MULC
        let mut prod = Cpx {
            r: a.r * b.r + a.i * b.i,
            i: a.i * b.r - a.r * b.i,
        };
        let norm_1 =
            (1.0f32 as f64 / ((1e-15f32 + prod.r * prod.r + prod.i * prod.i) as f64).sqrt()) as f32;
        prod.r *= norm_1;
        prod.i *= norm_1;
        st.if_features[3 * i - 2] = prod.r;
        st.if_features[3 * i - 1] = prod.i;
        st.if_features[3 * i] = clamp1(
            (1.0f32 / 64.0) * (10.0f32 * celt_log10(1e-15f32 + a.r * a.r + a.i * a.i) - 6.0),
        );
    }
    st.prev_if.copy_from_slice(&xf[..PITCH_IF_MAX_FREQ]);
    let mut log_max: f32 = -2.0;
    let mut follow: f32 = -2.0;
    for i in 0..NB_BANDS {
        let l = celt_log10(1e-2f32 + ex[i]);
        let f = follow - 2.5;
        let m = if f > l { f } else { l };
        let lm = log_max - 8.0;
        ly[i] = if lm > m { lm } else { m };
        log_max = if log_max > ly[i] { log_max } else { ly[i] };
        follow = if f > ly[i] { f } else { ly[i] };
    }
    dct(&mut st.features, &ly);
    st.features[0] -= 4.0;
    let feats = st.features;
    lpc_from_cepstrum(&mut st.lpc, &feats);
    for i in 0..LPC_ORDER {
        st.features[NB_BANDS + 2 + i] = st.lpc[i];
    }
    st.exc_buf
        .copy_within(FRAME_SIZE..FRAME_SIZE + PITCH_MAX_PERIOD, 0);
    st.lp_buf
        .copy_within(FRAME_SIZE..FRAME_SIZE + PITCH_MAX_PERIOD, 0);
    aligned_in[TRAINING_OFFSET..].copy_from_slice(&input[..FRAME_SIZE - TRAINING_OFFSET]);
    x[..LPC_ORDER].copy_from_slice(&st.pitch_mem);
    x[LPC_ORDER..].copy_from_slice(&aligned_in);
    st.pitch_mem
        .copy_from_slice(&aligned_in[FRAME_SIZE - LPC_ORDER..]);
    celt_fir(
        &x,
        &st.lpc,
        &mut st.lp_buf[PITCH_MAX_PERIOD..],
        FRAME_SIZE,
        LPC_ORDER,
    );
    for i in 0..FRAME_SIZE {
        st.exc_buf[PITCH_MAX_PERIOD + i] = st.lp_buf[PITCH_MAX_PERIOD + i] + 0.7f32 * st.pitch_filt;
        st.pitch_filt = st.lp_buf[PITCH_MAX_PERIOD + i];
    }
    biquad(
        &mut st.lp_buf[PITCH_MAX_PERIOD..],
        &mut st.lp_mem,
        &LP_B,
        &LP_A,
        FRAME_SIZE,
    );
    {
        let buf = &st.exc_buf;
        celt_pitch_xcorr(
            &buf[PITCH_MAX_PERIOD..],
            buf,
            &mut xcorr,
            FRAME_SIZE,
            PITCH_MAX_PERIOD - PITCH_MIN_PERIOD,
        );
        let ener0 = celt_inner_prod(
            &buf[PITCH_MAX_PERIOD..],
            &buf[PITCH_MAX_PERIOD..],
            FRAME_SIZE,
        );
        let mut ener1 = celt_inner_prod(buf, buf, FRAME_SIZE) as f64;
        for i in 0..PITCH_MAX_PERIOD - PITCH_MIN_PERIOD {
            let ener = ((1.0f32 + ener0) as f64 + ener1) as f32;
            st.xcorr_features[i] = 2.0 * xcorr[i];
            ener_norm[i] = ener;
            ener1 += buf[i + FRAME_SIZE] as f64 * buf[i + FRAME_SIZE] as f64
                - buf[i] as f64 * buf[i] as f64;
        }
        for i in 0..PITCH_MAX_PERIOD - PITCH_MIN_PERIOD {
            st.xcorr_features[i] /= ener_norm[i];
        }
    }
    let (iff, xcf) = (st.if_features, st.xcorr_features);
    st.dnn_pitch = compute_pitchdnn(&mut st.pitchdnn, &iff, &xcf, arch);
    let pitch = c_int(
        (0.5 + 256.0 / 2.0f32.powf_f64((1.0 / 60.0) * ((st.dnn_pitch as f64 + 1.5) * 60.0)))
            .floor(),
    ) as usize;
    let lp = &st.lp_buf;
    let xx = celt_inner_prod(&lp[PITCH_MAX_PERIOD..], &lp[PITCH_MAX_PERIOD..], FRAME_SIZE);
    let yy = celt_inner_prod(
        &lp[PITCH_MAX_PERIOD - pitch..],
        &lp[PITCH_MAX_PERIOD - pitch..],
        FRAME_SIZE,
    );
    let xy = celt_inner_prod(
        &lp[PITCH_MAX_PERIOD..],
        &lp[PITCH_MAX_PERIOD - pitch..],
        FRAME_SIZE,
    );
    let mut frame_corr = (xy as f64 / ((1.0f32 + xx * yy) as f64).sqrt()) as f32;
    frame_corr = (((1.0f32 as f64 + ((5.0f32 * frame_corr) as f64).exp()).ln())
        / (1.0 + (5.0f32 as f64).exp()).ln()) as f32;
    st.features[NB_BANDS] = st.dnn_pitch;
    st.features[NB_BANDS + 1] = frame_corr - 0.5f32;
}

/// `pow(2.f, y)` with the float base promoted to double.
trait PowF64 {
    fn powf_f64(self, y: f64) -> f64;
}
impl PowF64 for f32 {
    #[inline]
    fn powf_f64(self, y: f64) -> f64 {
        (self as f64).powf(y)
    }
}

/// `preemphasis`: `y[i] = x[i] + mem; mem = -coef*x[i]`, in place.
pub fn preemphasis(x: &mut [f32], mem: &mut f32, coef: f32, n: usize) {
    for v in x[..n].iter_mut() {
        let yi = *v + *mem;
        *mem = -coef * *v;
        *v = yi;
    }
}

/// `lpcnet_compute_single_frame_features_float`: the features of one
/// 10 ms frame (`FRAME_SIZE` samples at 16 kHz, in the int16 range).
pub fn lpcnet_compute_single_frame_features_float(
    st: &mut LpcnetEncState<'_>,
    pcm: &[f32],
    features: &mut [f32; NB_TOTAL_FEATURES],
    arch: Arch,
) {
    let mut x = [0.0f32; FRAME_SIZE];
    x.copy_from_slice(&pcm[..FRAME_SIZE]);
    preemphasis(&mut x, &mut st.mem_preemph, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(st, &x, arch);
    features.copy_from_slice(&st.features);
}

/// `lpcnet_compute_single_frame_features`: as the float version, from
/// int16 samples.
pub fn lpcnet_compute_single_frame_features(
    st: &mut LpcnetEncState<'_>,
    pcm: &[i16],
    features: &mut [f32; NB_TOTAL_FEATURES],
    arch: Arch,
) {
    let mut x = [0.0f32; FRAME_SIZE];
    for (o, &v) in x.iter_mut().zip(&pcm[..FRAME_SIZE]) {
        *o = v as f32;
    }
    lpcnet_compute_single_frame_features_float(st, &x, features, arch);
}
