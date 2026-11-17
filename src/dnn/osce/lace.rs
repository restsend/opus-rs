//! LACE, the light SILK enhancer (libopus `dnn/osce.c`): a feature network
//! conditions two adaptive comb filters and one adaptive convolution.

#[cfg(not(feature = "std"))]
use crate::compat::Math;

use super::nndsp::{
    AdaCombConfig, AdaCombState, AdaConvConfig, AdaConvState, adacomb_process_frame,
    adaconv_process_frame, compute_overlap_window,
};
use crate::dnn::models::lace::*;
use crate::dnn::nnet::{
    Activation, compute_generic_conv1d, compute_generic_dense, compute_generic_gru,
};

/// libopus `LACE`: the layers and the overlap window.
#[derive(Clone, Debug)]
pub struct Lace<'a> {
    pub layers: LaceLayers<'a>,
    pub window: [f32; LACE_OVERLAP_SIZE],
}

impl<'a> Lace<'a> {
    /// `init_lace` from initialized layers.
    pub fn new(layers: LaceLayers<'a>) -> Self {
        let mut window = [0.0; LACE_OVERLAP_SIZE];
        compute_overlap_window(&mut window);
        Lace { layers, window }
    }
}

/// libopus `LACEState`.
#[derive(Clone, Debug)]
pub struct LaceState {
    pub feature_net_conv2_state: [f32; LACE_FNET_CONV2_STATE_SIZE],
    pub feature_net_gru_state: [f32; LACE_COND_DIM],
    pub cf1_state: AdaCombState,
    pub cf2_state: AdaCombState,
    pub af1_state: AdaConvState,
    pub preemph_mem: f32,
    pub deemph_mem: f32,
}

impl LaceState {
    /// `reset_lace_state`: all zero.
    pub const fn new() -> Self {
        LaceState {
            feature_net_conv2_state: [0.0; LACE_FNET_CONV2_STATE_SIZE],
            feature_net_gru_state: [0.0; LACE_COND_DIM],
            cf1_state: AdaCombState::new(),
            cf2_state: AdaCombState::new(),
            af1_state: AdaConvState::new(),
            preemph_mem: 0.0,
            deemph_mem: 0.0,
        }
    }
}

impl Default for LaceState {
    fn default() -> Self {
        Self::new()
    }
}

/// `compute_lace_numbits_embedding` and `compute_nolace_numbits_embedding`
/// (with `logscale` set): sines of the scaled log bit count. The bit count
/// is limited from above only, as the C macro `CLIP` returns its argument
/// when it is below the minimum.
pub fn numbits_embedding(
    emb: &mut [f32],
    numbits: f32,
    min_val: f32,
    max_val: f32,
    scales: &[f32; 8],
) {
    let numbits = (numbits as f64).ln() as f32;
    let clipped = if (if numbits < min_val { min_val } else { numbits }) > max_val {
        max_val
    } else {
        numbits
    };
    let x = clipped - (max_val + min_val) / 2.0;
    for (e, &s) in emb[..8].iter_mut().zip(scales) {
        *e = ((x * s - 0.5f32) as f64).sin() as f32;
    }
}

const LACE_NUMBITS_SCALES: [f32; 8] = [
    LACE_NUMBITS_SCALE_0,
    LACE_NUMBITS_SCALE_1,
    LACE_NUMBITS_SCALE_2,
    LACE_NUMBITS_SCALE_3,
    LACE_NUMBITS_SCALE_4,
    LACE_NUMBITS_SCALE_5,
    LACE_NUMBITS_SCALE_6,
    LACE_NUMBITS_SCALE_7,
];

const CONV1_IN: usize =
    LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM + 2 * LACE_NUMBITS_EMBEDDING_DIM;
const BUF: usize = 4 * if LACE_COND_DIM > LACE_HIDDEN_FEATURE_DIM {
    LACE_COND_DIM
} else {
    LACE_HIDDEN_FEATURE_DIM
};

/// `lace_feature_net`: `LACE_COND_DIM` conditioning values per subframe.
fn lace_feature_net(
    lace: &Lace<'_>,
    state: &mut LaceState,
    output: &mut [f32; 4 * LACE_COND_DIM],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[i32; 4],
) {
    let layers = &lace.layers;
    let mut input_buffer = [0.0f32; if BUF > CONV1_IN { BUF } else { CONV1_IN }];
    let mut output_buffer = [0.0f32; BUF];
    let mut numbits_embedded = [0.0f32; 2 * LACE_NUMBITS_EMBEDDING_DIM];
    let lo = (LACE_NUMBITS_RANGE_LOW as f64).ln() as f32;
    let hi = (LACE_NUMBITS_RANGE_HIGH as f64).ln() as f32;

    numbits_embedding(
        &mut numbits_embedded,
        numbits[0],
        lo,
        hi,
        &LACE_NUMBITS_SCALES,
    );
    numbits_embedding(
        &mut numbits_embedded[LACE_NUMBITS_EMBEDDING_DIM..],
        numbits[1],
        lo,
        hi,
        &LACE_NUMBITS_SCALES,
    );

    let emb = layers
        .lace_pitch_embedding
        .float_weights
        .expect("pitch embedding without float weights");
    for i in 0..4 {
        input_buffer[..LACE_NUM_FEATURES]
            .copy_from_slice(&features[i * LACE_NUM_FEATURES..(i + 1) * LACE_NUM_FEATURES]);
        let row = periods[i] as usize * LACE_PITCH_EMBEDDING_DIM;
        for j in 0..LACE_PITCH_EMBEDDING_DIM {
            input_buffer[LACE_NUM_FEATURES + j] = emb.at(row + j);
        }
        input_buffer[LACE_NUM_FEATURES + LACE_PITCH_EMBEDDING_DIM..CONV1_IN]
            .copy_from_slice(&numbits_embedded);
        compute_generic_conv1d(
            &layers.lace_fnet_conv1,
            &mut output_buffer[i * LACE_HIDDEN_FEATURE_DIM..],
            &mut [],
            &input_buffer,
            CONV1_IN,
            Activation::Tanh,
        );
    }

    input_buffer[..4 * LACE_HIDDEN_FEATURE_DIM]
        .copy_from_slice(&output_buffer[..4 * LACE_HIDDEN_FEATURE_DIM]);
    compute_generic_conv1d(
        &layers.lace_fnet_conv2,
        &mut output_buffer,
        &mut state.feature_net_conv2_state,
        &input_buffer,
        4 * LACE_HIDDEN_FEATURE_DIM,
        Activation::Tanh,
    );

    input_buffer[..4 * LACE_COND_DIM].copy_from_slice(&output_buffer[..4 * LACE_COND_DIM]);
    compute_generic_dense(
        &layers.lace_fnet_tconv,
        &mut output_buffer,
        &input_buffer,
        Activation::Tanh,
    );

    input_buffer[..4 * LACE_COND_DIM].copy_from_slice(&output_buffer[..4 * LACE_COND_DIM]);
    for i in 0..4 {
        compute_generic_gru(
            &layers.lace_fnet_gru_input,
            &layers.lace_fnet_gru_recurrent,
            &mut state.feature_net_gru_state,
            &input_buffer[i * LACE_COND_DIM..],
        );
        output[i * LACE_COND_DIM..(i + 1) * LACE_COND_DIM]
            .copy_from_slice(&state.feature_net_gru_state);
    }
}

/// The parameters of the cf1 comb filter.
pub const CF1: AdaCombConfig = AdaCombConfig {
    frame_size: LACE_FRAME_SIZE,
    overlap_size: LACE_OVERLAP_SIZE,
    kernel_size: LACE_CF1_KERNEL_SIZE,
    left_padding: LACE_CF1_LEFT_PADDING,
    filter_gain_a: LACE_CF1_FILTER_GAIN_A,
    filter_gain_b: LACE_CF1_FILTER_GAIN_B,
    log_gain_limit: LACE_CF1_LOG_GAIN_LIMIT,
};

/// The parameters of the cf2 comb filter.
pub const CF2: AdaCombConfig = AdaCombConfig {
    frame_size: LACE_FRAME_SIZE,
    overlap_size: LACE_OVERLAP_SIZE,
    kernel_size: LACE_CF2_KERNEL_SIZE,
    left_padding: LACE_CF2_LEFT_PADDING,
    filter_gain_a: LACE_CF2_FILTER_GAIN_A,
    filter_gain_b: LACE_CF2_FILTER_GAIN_B,
    log_gain_limit: LACE_CF2_LOG_GAIN_LIMIT,
};

/// The parameters of the af1 adaptive filter.
pub const AF1: AdaConvConfig = AdaConvConfig {
    frame_size: LACE_FRAME_SIZE,
    overlap_size: LACE_OVERLAP_SIZE,
    in_channels: LACE_AF1_IN_CHANNELS,
    out_channels: LACE_AF1_OUT_CHANNELS,
    kernel_size: LACE_AF1_KERNEL_SIZE,
    left_padding: LACE_AF1_LEFT_PADDING,
    filter_gain_a: LACE_AF1_FILTER_GAIN_A,
    filter_gain_b: LACE_AF1_FILTER_GAIN_B,
    shape_gain: LACE_AF1_SHAPE_GAIN,
};

/// `lace_process_20ms_frame`: enhances the 320 samples `x_in` (16 kHz,
/// scaled to +-1) into `x_out` with the frame's `features`, bit counts
/// and pitch lags.
pub fn lace_process_20ms_frame(
    lace: &Lace<'_>,
    state: &mut LaceState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[i32; 4],
) {
    const F: usize = LACE_FRAME_SIZE;
    let layers = &lace.layers;
    let mut feature_buffer = [0.0f32; 4 * LACE_COND_DIM];
    let mut output_buffer = [0.0f32; 4 * F];

    for i in 0..4 * F {
        output_buffer[i] = x_in[i] - LACE_PREEMPH * state.preemph_mem;
        state.preemph_mem = x_in[i];
    }

    lace_feature_net(lace, state, &mut feature_buffer, features, numbits, periods);

    for i in 0..4 {
        adacomb_process_frame(
            &mut state.cf1_state,
            &mut output_buffer[i * F..(i + 1) * F],
            &feature_buffer[i * LACE_COND_DIM..],
            &layers.lace_cf1_kernel,
            &layers.lace_cf1_gain,
            &layers.lace_cf1_global_gain,
            periods[i] as usize,
            &CF1,
            &lace.window,
        );
    }
    for i in 0..4 {
        adacomb_process_frame(
            &mut state.cf2_state,
            &mut output_buffer[i * F..(i + 1) * F],
            &feature_buffer[i * LACE_COND_DIM..],
            &layers.lace_cf2_kernel,
            &layers.lace_cf2_gain,
            &layers.lace_cf2_global_gain,
            periods[i] as usize,
            &CF2,
            &lace.window,
        );
    }
    for i in 0..4 {
        let mut x = [0.0f32; F];
        x.copy_from_slice(&output_buffer[i * F..(i + 1) * F]);
        adaconv_process_frame(
            &mut state.af1_state,
            &mut output_buffer[i * F..(i + 1) * F],
            &x,
            &feature_buffer[i * LACE_COND_DIM..],
            &layers.lace_af1_kernel,
            &layers.lace_af1_gain,
            &AF1,
            &lace.window,
        );
    }

    for i in 0..4 * F {
        x_out[i] = output_buffer[i] + LACE_PREEMPH * state.deemph_mem;
        state.deemph_mem = x_out[i];
    }
}
