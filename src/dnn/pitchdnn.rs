//! PitchDNN, the pitch estimator of the LPCNet features (libopus
//! `dnn/pitchdnn.c`).

#[cfg(not(feature = "std"))]
use crate::compat::Math;

use super::models::pitchdnn::{
    PitchDnn, DENSE_DOWNSAMPLER_OUT_SIZE, DENSE_FINAL_UPSAMPLER_OUT_SIZE, DENSE_IF_UPSAMPLER_1_OUT_SIZE,
    DENSE_IF_UPSAMPLER_2_OUT_SIZE, GRU_1_STATE_SIZE,
};
use super::nnet::{compute_conv2d, compute_generic_dense, compute_generic_gru, Activation};

pub const PITCH_MIN_PERIOD: usize = 32;
pub const PITCH_MAX_PERIOD: usize = 256;
pub const NB_XCORR_FEATURES: usize = PITCH_MAX_PERIOD - PITCH_MIN_PERIOD;

/// libopus `PitchDNNState`.
#[derive(Clone)]
pub struct PitchDnnState<'a> {
    pub model: PitchDnn<'a>,
    pub gru_state: [f32; GRU_1_STATE_SIZE],
    pub xcorr_mem1: [f32; (NB_XCORR_FEATURES + 2) * 2],
    pub xcorr_mem2: [f32; (NB_XCORR_FEATURES + 2) * 2 * 8],
    pub xcorr_mem3: [f32; (NB_XCORR_FEATURES + 2) * 2 * 8],
}

impl<'a> PitchDnnState<'a> {
    /// `pitchdnn_init` with the given model: zero state.
    pub fn new(model: PitchDnn<'a>) -> Self {
        PitchDnnState {
            model,
            gru_state: [0.0; GRU_1_STATE_SIZE],
            xcorr_mem1: [0.0; (NB_XCORR_FEATURES + 2) * 2],
            xcorr_mem2: [0.0; (NB_XCORR_FEATURES + 2) * 2 * 8],
            xcorr_mem3: [0.0; (NB_XCORR_FEATURES + 2) * 2 * 8],
        }
    }

    /// Clears the state and keeps the model.
    pub fn reset(&mut self) {
        *self = PitchDnnState::new(self.model);
    }
}

/// `compute_pitchdnn`: the pitch as `log2(256/period)`-like value in
/// [-1.5, 1.5] from the instantaneous-frequency and cross-correlation
/// features.
pub fn compute_pitchdnn(st: &mut PitchDnnState<'_>, if_features: &[f32], xcorr_features: &[f32]) -> f32 {
    let mut if1_out = [0.0f32; DENSE_IF_UPSAMPLER_1_OUT_SIZE];
    let mut downsampler_in = [0.0f32; NB_XCORR_FEATURES + DENSE_IF_UPSAMPLER_2_OUT_SIZE];
    let mut downsampler_out = [0.0f32; DENSE_DOWNSAMPLER_OUT_SIZE];
    let mut conv1_tmp1 = [0.0f32; (NB_XCORR_FEATURES + 2) * 8];
    let mut conv1_tmp2 = [0.0f32; (NB_XCORR_FEATURES + 2) * 8];
    let mut output = [0.0f32; DENSE_FINAL_UPSAMPLER_OUT_SIZE];
    let model = st.model;
    compute_generic_dense(&model.dense_if_upsampler_1, &mut if1_out, if_features, Activation::Tanh);
    compute_generic_dense(
        &model.dense_if_upsampler_2,
        &mut downsampler_in[NB_XCORR_FEATURES..],
        &if1_out,
        Activation::Tanh,
    );
    conv1_tmp1[1..1 + NB_XCORR_FEATURES].copy_from_slice(&xcorr_features[..NB_XCORR_FEATURES]);
    compute_conv2d(
        &model.conv2d_1,
        &mut conv1_tmp2[1..],
        &mut st.xcorr_mem1,
        &conv1_tmp1,
        NB_XCORR_FEATURES,
        NB_XCORR_FEATURES + 2,
        Activation::Tanh,
    );
    compute_conv2d(
        &model.conv2d_2,
        &mut downsampler_in,
        &mut st.xcorr_mem2,
        &conv1_tmp2,
        NB_XCORR_FEATURES,
        NB_XCORR_FEATURES,
        Activation::Tanh,
    );
    compute_generic_dense(&model.dense_downsampler, &mut downsampler_out, &downsampler_in, Activation::Tanh);
    compute_generic_gru(&model.gru_1_input, &model.gru_1_recurrent, &mut st.gru_state, &downsampler_out);
    compute_generic_dense(&model.dense_final_upsampler, &mut output, &st.gru_state, Activation::Linear);
    let mut pos = 0usize;
    let mut maxval = -1.0f32;
    for (i, &v) in output[..180].iter().enumerate() {
        if v > maxval {
            pos = i;
            maxval = v;
        }
    }
    let mut sum = 0.0f32;
    let mut count = 0.0f32;
    for i in pos.saturating_sub(2)..=(pos + 2).min(179) {
        let p = (output[i] as f64).exp() as f32;
        sum += p * i as f32;
        count += p;
    }
    (((1.0f32 / 60.0f32) * (sum / count)) as f64 - 1.5) as f32
}
