//! FARGAN, the vocoder of deep PLC and DRED (libopus `dnn/fargan.c`): it
//! turns one feature vector per 10 ms into 160 samples at 16 kHz.

#[cfg(not(feature = "std"))]
use crate::compat::Math;

use super::freq::NB_BANDS;
use super::lpcnet_enc::{LPCNET_FRAME_SIZE, NB_FEATURES};
use super::models::fargan::*;
use super::nnet::{compute_generic_conv1d, compute_generic_dense, compute_generic_gru, compute_glu, compute_glu_inplace, Activation};
use super::pitchdnn::PITCH_MAX_PERIOD;
use super::vec::{c_int, max32, min32};

pub const FARGAN_CONT_SAMPLES: usize = 320;
pub const FARGAN_NB_SUBFRAMES: usize = 4;
pub const FARGAN_SUBFRAME_SIZE: usize = 40;
pub const FARGAN_FRAME_SIZE: usize = FARGAN_NB_SUBFRAMES * FARGAN_SUBFRAME_SIZE;
pub const FARGAN_COND_SIZE: usize = COND_NET_FDENSE2_OUT_SIZE / FARGAN_NB_SUBFRAMES;
pub const FARGAN_DEEMPHASIS: f32 = 0.85;
pub const SIG_NET_INPUT_SIZE: usize = FARGAN_COND_SIZE + 2 * FARGAN_SUBFRAME_SIZE + 4;
pub const SIG_NET_FWC0_STATE_SIZE: usize = 2 * SIG_NET_INPUT_SIZE;
const SKIP_CAT_SIZE: usize = SIG_NET_GRU1_OUT_SIZE
    + SIG_NET_GRU2_OUT_SIZE
    + SIG_NET_GRU3_OUT_SIZE
    + SIG_NET_FWC0_CONV_OUT_SIZE
    + 2 * FARGAN_SUBFRAME_SIZE;

/// libopus `FARGANState`.
#[derive(Clone)]
pub struct FarganState<'a> {
    pub model: Fargan<'a>,
    pub cont_initialized: bool,
    pub deemph_mem: f32,
    pub pitch_buf: [f32; PITCH_MAX_PERIOD],
    pub cond_conv1_state: [f32; COND_NET_FCONV1_STATE_SIZE],
    pub fwc0_mem: [f32; SIG_NET_FWC0_STATE_SIZE],
    pub gru1_state: [f32; SIG_NET_GRU1_STATE_SIZE],
    pub gru2_state: [f32; SIG_NET_GRU2_STATE_SIZE],
    pub gru3_state: [f32; SIG_NET_GRU3_STATE_SIZE],
    pub last_period: i32,
}

impl<'a> FarganState<'a> {
    /// `fargan_init`: zero state with the given model.
    pub fn new(model: Fargan<'a>) -> Self {
        FarganState {
            model,
            cont_initialized: false,
            deemph_mem: 0.0,
            pitch_buf: [0.0; PITCH_MAX_PERIOD],
            cond_conv1_state: [0.0; COND_NET_FCONV1_STATE_SIZE],
            fwc0_mem: [0.0; SIG_NET_FWC0_STATE_SIZE],
            gru1_state: [0.0; SIG_NET_GRU1_STATE_SIZE],
            gru2_state: [0.0; SIG_NET_GRU2_STATE_SIZE],
            gru3_state: [0.0; SIG_NET_GRU3_STATE_SIZE],
            last_period: 0,
        }
    }
}

/// The pitch period of a feature vector:
/// `(int)floor(.5+256./pow(2.f,((1./60.)*((features[NB_BANDS]+1.5)*60))))`.
pub(crate) fn feature_period(features: &[f32]) -> i32 {
    c_int((0.5 + 256.0 / 2.0f64.powf((1.0 / 60.0) * ((features[NB_BANDS] as f64 + 1.5) * 60.0))).floor())
}

/// `compute_fargan_cond`.
fn compute_fargan_cond(st: &mut FarganState<'_>, cond: &mut [f32], features: &[f32], period: i32) {
    let model = &st.model;
    let mut dense_in = [0.0f32; NB_FEATURES + COND_NET_PEMBED_OUT_SIZE];
    let mut conv1_in = [0.0f32; COND_NET_FCONV1_IN_SIZE];
    let mut fdense2_in = [0.0f32; COND_NET_FCONV1_OUT_SIZE];
    assert_eq!(model.cond_net_fdense1.nb_inputs, NB_FEATURES + COND_NET_PEMBED_OUT_SIZE);
    let pembed = model.cond_net_pembed.float_weights.expect("cond_net_pembed weights");
    let row = (period - 32).clamp(0, 223) as usize * COND_NET_PEMBED_OUT_SIZE;
    for (k, v) in dense_in[NB_FEATURES..].iter_mut().enumerate() {
        *v = pembed.at(row + k);
    }
    dense_in[..NB_FEATURES].copy_from_slice(&features[..NB_FEATURES]);
    compute_generic_dense(&model.cond_net_fdense1, &mut conv1_in, &dense_in, Activation::Tanh);
    compute_generic_conv1d(
        &model.cond_net_fconv1,
        &mut fdense2_in,
        &mut st.cond_conv1_state,
        &conv1_in,
        COND_NET_FCONV1_IN_SIZE,
        Activation::Tanh,
    );
    compute_generic_dense(&model.cond_net_fdense2, cond, &fdense2_in, Activation::Tanh);
}

/// `fargan_deemphasis`.
fn fargan_deemphasis(pcm: &mut [f32], deemph_mem: &mut f32) {
    for v in pcm[..FARGAN_SUBFRAME_SIZE].iter_mut() {
        *v += FARGAN_DEEMPHASIS * *deemph_mem;
        *deemph_mem = *v;
    }
}

/// `run_fargan_subframe`: 40 samples into `pcm`.
fn run_fargan_subframe(st: &mut FarganState<'_>, pcm: &mut [f32], cond: &[f32], period: i32) {
    assert!(st.cont_initialized);
    let model = st.model;
    let mut fwc0_in = [0.0f32; SIG_NET_INPUT_SIZE];
    let mut gru1_in = [0.0f32; SIG_NET_FWC0_CONV_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    let mut gru2_in = [0.0f32; SIG_NET_GRU1_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    let mut gru3_in = [0.0f32; SIG_NET_GRU2_OUT_SIZE + 2 * FARGAN_SUBFRAME_SIZE];
    let mut pred = [0.0f32; FARGAN_SUBFRAME_SIZE + 4];
    let mut prev = [0.0f32; FARGAN_SUBFRAME_SIZE];
    let mut pitch_gate = [0.0f32; 4];
    let mut gain = [0.0f32; 1];
    let mut skip_cat = [0.0f32; SKIP_CAT_SIZE];
    let mut skip_out = [0.0f32; SIG_NET_SKIP_DENSE_OUT_SIZE];

    compute_generic_dense(&model.sig_net_cond_gain_dense, &mut gain, cond, Activation::Linear);
    let gain = (gain[0] as f64).exp() as f32;
    let gain_1 = 1.0f32 / (1e-5f32 + gain);

    let mut pos = PITCH_MAX_PERIOD as i32 - period - 2;
    for p in pred.iter_mut() {
        *p = min32(1.0, max32(-1.0, gain_1 * st.pitch_buf[pos.max(0) as usize]));
        pos += 1;
        if pos == PITCH_MAX_PERIOD as i32 {
            pos -= period;
        }
    }
    for (i, p) in prev.iter_mut().enumerate() {
        *p = max32(-1.0, min32(1.0, gain_1 * st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE + i]));
    }

    fwc0_in[..FARGAN_COND_SIZE].copy_from_slice(&cond[..FARGAN_COND_SIZE]);
    fwc0_in[FARGAN_COND_SIZE..FARGAN_COND_SIZE + FARGAN_SUBFRAME_SIZE + 4].copy_from_slice(&pred);
    fwc0_in[FARGAN_COND_SIZE + FARGAN_SUBFRAME_SIZE + 4..].copy_from_slice(&prev);

    compute_generic_conv1d(
        &model.sig_net_fwc0_conv,
        &mut gru1_in,
        &mut st.fwc0_mem,
        &fwc0_in,
        SIG_NET_INPUT_SIZE,
        Activation::Tanh,
    );
    assert_eq!(model.sig_net_fwc0_glu_gate.nb_outputs, SIG_NET_FWC0_GLU_GATE_OUT_SIZE);
    compute_glu_inplace(&model.sig_net_fwc0_glu_gate, &mut gru1_in);

    compute_generic_dense(&model.sig_net_gain_dense_out, &mut pitch_gate, &gru1_in, Activation::Sigmoid);

    let g = SIG_NET_FWC0_GLU_GATE_OUT_SIZE;
    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru1_in[g + i] = pitch_gate[0] * pred[i + 2];
    }
    gru1_in[g + FARGAN_SUBFRAME_SIZE..g + 2 * FARGAN_SUBFRAME_SIZE].copy_from_slice(&prev);
    compute_generic_gru(&model.sig_net_gru1_input, &model.sig_net_gru1_recurrent, &mut st.gru1_state, &gru1_in);
    compute_glu(&model.sig_net_gru1_glu_gate, &mut gru2_in, &st.gru1_state);

    let g = SIG_NET_GRU1_OUT_SIZE;
    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru2_in[g + i] = pitch_gate[1] * pred[i + 2];
    }
    gru2_in[g + FARGAN_SUBFRAME_SIZE..g + 2 * FARGAN_SUBFRAME_SIZE].copy_from_slice(&prev);
    compute_generic_gru(&model.sig_net_gru2_input, &model.sig_net_gru2_recurrent, &mut st.gru2_state, &gru2_in);
    compute_glu(&model.sig_net_gru2_glu_gate, &mut gru3_in, &st.gru2_state);

    let g = SIG_NET_GRU2_OUT_SIZE;
    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru3_in[g + i] = pitch_gate[2] * pred[i + 2];
    }
    gru3_in[g + FARGAN_SUBFRAME_SIZE..g + 2 * FARGAN_SUBFRAME_SIZE].copy_from_slice(&prev);
    compute_generic_gru(&model.sig_net_gru3_input, &model.sig_net_gru3_recurrent, &mut st.gru3_state, &gru3_in);
    let o3 = SIG_NET_GRU1_OUT_SIZE + SIG_NET_GRU2_OUT_SIZE;
    compute_glu(&model.sig_net_gru3_glu_gate, &mut skip_cat[o3..], &st.gru3_state);

    skip_cat[..SIG_NET_GRU1_OUT_SIZE].copy_from_slice(&gru2_in[..SIG_NET_GRU1_OUT_SIZE]);
    skip_cat[SIG_NET_GRU1_OUT_SIZE..o3].copy_from_slice(&gru3_in[..SIG_NET_GRU2_OUT_SIZE]);
    let o4 = o3 + SIG_NET_GRU3_OUT_SIZE;
    skip_cat[o4..o4 + SIG_NET_FWC0_CONV_OUT_SIZE].copy_from_slice(&gru1_in[..SIG_NET_FWC0_CONV_OUT_SIZE]);
    let o5 = o4 + SIG_NET_FWC0_CONV_OUT_SIZE;
    for i in 0..FARGAN_SUBFRAME_SIZE {
        skip_cat[o5 + i] = pitch_gate[3] * pred[i + 2];
    }
    skip_cat[o5 + FARGAN_SUBFRAME_SIZE..o5 + 2 * FARGAN_SUBFRAME_SIZE].copy_from_slice(&prev);

    compute_generic_dense(&model.sig_net_skip_dense, &mut skip_out, &skip_cat, Activation::Tanh);
    compute_glu_inplace(&model.sig_net_skip_glu_gate, &mut skip_out);

    compute_generic_dense(&model.sig_net_sig_dense_out, pcm, &skip_out, Activation::Tanh);
    for v in pcm[..FARGAN_SUBFRAME_SIZE].iter_mut() {
        *v *= gain;
    }

    st.pitch_buf.copy_within(FARGAN_SUBFRAME_SIZE.., 0);
    st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE..].copy_from_slice(&pcm[..FARGAN_SUBFRAME_SIZE]);
    fargan_deemphasis(pcm, &mut st.deemph_mem);
}

/// `fargan_cont`: primes the vocoder with the last `FARGAN_CONT_SAMPLES`
/// samples (`pcm0`, float in [-1, 1]) and five feature vectors.
pub fn fargan_cont(st: &mut FarganState<'_>, pcm0: &[f32], features0: &[f32]) {
    let mut cond = [0.0f32; COND_NET_FDENSE2_OUT_SIZE];
    let mut x0 = [0.0f32; FARGAN_CONT_SAMPLES];
    let mut dummy = [0.0f32; FARGAN_SUBFRAME_SIZE];
    let mut period = 0;
    for i in 0..5 {
        let features = &features0[i * NB_FEATURES..(i + 1) * NB_FEATURES];
        st.last_period = period;
        period = feature_period(features);
        compute_fargan_cond(st, &mut cond, features, period);
    }
    x0[0] = 0.0;
    for i in 1..FARGAN_CONT_SAMPLES {
        x0[i] = pcm0[i] - FARGAN_DEEMPHASIS * pcm0[i - 1];
    }
    st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_FRAME_SIZE..].copy_from_slice(&x0[..FARGAN_FRAME_SIZE]);
    st.cont_initialized = true;
    for i in 0..FARGAN_NB_SUBFRAMES {
        let lp = st.last_period;
        run_fargan_subframe(st, &mut dummy, &cond[i * FARGAN_COND_SIZE..], lp);
        let src = FARGAN_FRAME_SIZE + i * FARGAN_SUBFRAME_SIZE;
        st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE..].copy_from_slice(&x0[src..src + FARGAN_SUBFRAME_SIZE]);
    }
    st.deemph_mem = pcm0[FARGAN_CONT_SAMPLES - 1];
}

/// `fargan_synthesize`: 160 float samples from one feature vector.
pub fn fargan_synthesize(st: &mut FarganState<'_>, pcm: &mut [f32], features: &[f32]) {
    assert!(st.cont_initialized);
    let mut cond = [0.0f32; COND_NET_FDENSE2_OUT_SIZE];
    let period = feature_period(features);
    compute_fargan_cond(st, &mut cond, features, period);
    for sub in 0..FARGAN_NB_SUBFRAMES {
        let lp = st.last_period;
        run_fargan_subframe(
            st,
            &mut pcm[sub * FARGAN_SUBFRAME_SIZE..],
            &cond[sub * FARGAN_COND_SIZE..],
            lp,
        );
    }
    st.last_period = period;
}

/// `fargan_synthesize_int`: as `fargan_synthesize`, rounded to int16.
pub fn fargan_synthesize_int(st: &mut FarganState<'_>, pcm: &mut [i16], features: &[f32]) {
    let mut fpcm = [0.0f32; FARGAN_FRAME_SIZE];
    fargan_synthesize(st, &mut fpcm, features);
    for (o, &v) in pcm[..LPCNET_FRAME_SIZE].iter_mut().zip(&fpcm) {
        let s = min32(32767.0, max32(-32767.0, 32768.0f32 * v));
        *o = c_int((0.5 + s as f64).floor()) as i16;
    }
}
