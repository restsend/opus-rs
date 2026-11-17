//! NoLACE, the non-linear SILK enhancer (libopus `dnn/osce.c`): LACE's
//! feature network and comb filters followed by three rounds of adaptive
//! shaping and mixing of two channels.

use super::lace::numbits_embedding;
use super::nndsp::{
    AdaCombConfig, AdaCombState, AdaConvConfig, AdaConvState, AdaShapeConfig, AdaShapeState,
    adacomb_process_frame, adaconv_process_frame, adashape_process_frame, compute_overlap_window,
};
use crate::dnn::models::nolace::*;
use crate::dnn::nnet::{
    Activation, compute_generic_conv1d, compute_generic_dense, compute_generic_gru,
};

#[cfg(not(feature = "std"))]
use crate::compat::Math;

/// libopus `NoLACE`: the layers and the overlap window.
#[derive(Clone, Debug)]
pub struct Nolace<'a> {
    pub layers: NolaceLayers<'a>,
    pub window: [f32; NOLACE_OVERLAP_SIZE],
}

impl<'a> Nolace<'a> {
    /// `init_nolace` from initialized layers.
    pub fn new(layers: NolaceLayers<'a>) -> Self {
        let mut window = [0.0; NOLACE_OVERLAP_SIZE];
        compute_overlap_window(&mut window);
        Nolace { layers, window }
    }
}

/// libopus `NoLACEState`.
#[derive(Clone, Debug)]
pub struct NolaceState {
    pub feature_net_conv2_state: [f32; NOLACE_FNET_CONV2_STATE_SIZE],
    pub feature_net_gru_state: [f32; NOLACE_COND_DIM],
    pub post_cf1_state: [f32; NOLACE_COND_DIM],
    pub post_cf2_state: [f32; NOLACE_COND_DIM],
    pub post_af1_state: [f32; NOLACE_COND_DIM],
    pub post_af2_state: [f32; NOLACE_COND_DIM],
    pub post_af3_state: [f32; NOLACE_COND_DIM],
    pub cf1_state: AdaCombState,
    pub cf2_state: AdaCombState,
    pub af1_state: AdaConvState,
    pub af2_state: AdaConvState,
    pub af3_state: AdaConvState,
    pub af4_state: AdaConvState,
    pub tdshape1_state: AdaShapeState,
    pub tdshape2_state: AdaShapeState,
    pub tdshape3_state: AdaShapeState,
    pub preemph_mem: f32,
    pub deemph_mem: f32,
}

impl NolaceState {
    /// `reset_nolace_state`: all zero.
    pub const fn new() -> Self {
        NolaceState {
            feature_net_conv2_state: [0.0; NOLACE_FNET_CONV2_STATE_SIZE],
            feature_net_gru_state: [0.0; NOLACE_COND_DIM],
            post_cf1_state: [0.0; NOLACE_COND_DIM],
            post_cf2_state: [0.0; NOLACE_COND_DIM],
            post_af1_state: [0.0; NOLACE_COND_DIM],
            post_af2_state: [0.0; NOLACE_COND_DIM],
            post_af3_state: [0.0; NOLACE_COND_DIM],
            cf1_state: AdaCombState::new(),
            cf2_state: AdaCombState::new(),
            af1_state: AdaConvState::new(),
            af2_state: AdaConvState::new(),
            af3_state: AdaConvState::new(),
            af4_state: AdaConvState::new(),
            tdshape1_state: AdaShapeState::new(),
            tdshape2_state: AdaShapeState::new(),
            tdshape3_state: AdaShapeState::new(),
            preemph_mem: 0.0,
            deemph_mem: 0.0,
        }
    }
}

impl Default for NolaceState {
    fn default() -> Self {
        Self::new()
    }
}

const NOLACE_NUMBITS_SCALES: [f32; 8] = [
    NOLACE_NUMBITS_SCALE_0,
    NOLACE_NUMBITS_SCALE_1,
    NOLACE_NUMBITS_SCALE_2,
    NOLACE_NUMBITS_SCALE_3,
    NOLACE_NUMBITS_SCALE_4,
    NOLACE_NUMBITS_SCALE_5,
    NOLACE_NUMBITS_SCALE_6,
    NOLACE_NUMBITS_SCALE_7,
];

const CONV1_IN: usize =
    NOLACE_NUM_FEATURES + NOLACE_PITCH_EMBEDDING_DIM + 2 * NOLACE_NUMBITS_EMBEDDING_DIM;
const BUF: usize = 4 * if NOLACE_COND_DIM > NOLACE_HIDDEN_FEATURE_DIM {
    NOLACE_COND_DIM
} else {
    NOLACE_HIDDEN_FEATURE_DIM
};
const COND: usize = NOLACE_COND_DIM;

/// `nolace_feature_net`: `NOLACE_COND_DIM` conditioning values per subframe.
fn nolace_feature_net(
    nolace: &Nolace<'_>,
    state: &mut NolaceState,
    output: &mut [f32; 4 * COND],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[i32; 4],
) {
    let layers = &nolace.layers;
    let mut input_buffer = [0.0f32; BUF];
    let mut output_buffer = [0.0f32; BUF];
    let mut numbits_embedded = [0.0f32; 2 * NOLACE_NUMBITS_EMBEDDING_DIM];
    let lo = (NOLACE_NUMBITS_RANGE_LOW as f64).ln() as f32;
    let hi = (NOLACE_NUMBITS_RANGE_HIGH as f64).ln() as f32;

    numbits_embedding(
        &mut numbits_embedded,
        numbits[0],
        lo,
        hi,
        &NOLACE_NUMBITS_SCALES,
    );
    numbits_embedding(
        &mut numbits_embedded[NOLACE_NUMBITS_EMBEDDING_DIM..],
        numbits[1],
        lo,
        hi,
        &NOLACE_NUMBITS_SCALES,
    );

    let emb = layers
        .nolace_pitch_embedding
        .float_weights
        .expect("pitch embedding without float weights");
    for i in 0..4 {
        input_buffer[..NOLACE_NUM_FEATURES]
            .copy_from_slice(&features[i * NOLACE_NUM_FEATURES..(i + 1) * NOLACE_NUM_FEATURES]);
        let row = periods[i] as usize * NOLACE_PITCH_EMBEDDING_DIM;
        for j in 0..NOLACE_PITCH_EMBEDDING_DIM {
            input_buffer[NOLACE_NUM_FEATURES + j] = emb.at(row + j);
        }
        input_buffer[NOLACE_NUM_FEATURES + NOLACE_PITCH_EMBEDDING_DIM..CONV1_IN]
            .copy_from_slice(&numbits_embedded);
        compute_generic_conv1d(
            &layers.nolace_fnet_conv1,
            &mut output_buffer[i * NOLACE_HIDDEN_FEATURE_DIM..],
            &mut [],
            &input_buffer,
            CONV1_IN,
            Activation::Tanh,
        );
    }

    input_buffer[..4 * NOLACE_HIDDEN_FEATURE_DIM]
        .copy_from_slice(&output_buffer[..4 * NOLACE_HIDDEN_FEATURE_DIM]);
    compute_generic_conv1d(
        &layers.nolace_fnet_conv2,
        &mut output_buffer,
        &mut state.feature_net_conv2_state,
        &input_buffer,
        4 * NOLACE_HIDDEN_FEATURE_DIM,
        Activation::Tanh,
    );

    input_buffer[..4 * COND].copy_from_slice(&output_buffer[..4 * COND]);
    compute_generic_dense(
        &layers.nolace_fnet_tconv,
        &mut output_buffer,
        &input_buffer,
        Activation::Tanh,
    );

    input_buffer[..4 * COND].copy_from_slice(&output_buffer[..4 * COND]);
    for i in 0..4 {
        compute_generic_gru(
            &layers.nolace_fnet_gru_input,
            &layers.nolace_fnet_gru_recurrent,
            &mut state.feature_net_gru_state,
            &input_buffer[i * COND..],
        );
        output[i * COND..(i + 1) * COND].copy_from_slice(&state.feature_net_gru_state);
    }
}

const fn comb(
    kernel_size: usize,
    left_padding: usize,
    a: f32,
    b: f32,
    limit: f32,
) -> AdaCombConfig {
    AdaCombConfig {
        frame_size: NOLACE_FRAME_SIZE,
        overlap_size: NOLACE_OVERLAP_SIZE,
        kernel_size,
        left_padding,
        filter_gain_a: a,
        filter_gain_b: b,
        log_gain_limit: limit,
    }
}

#[allow(clippy::too_many_arguments)]
const fn conv(i: usize, o: usize, k: usize, lp: usize, a: f32, b: f32, s: f32) -> AdaConvConfig {
    AdaConvConfig {
        frame_size: NOLACE_FRAME_SIZE,
        overlap_size: NOLACE_OVERLAP_SIZE,
        in_channels: i,
        out_channels: o,
        kernel_size: k,
        left_padding: lp,
        filter_gain_a: a,
        filter_gain_b: b,
        shape_gain: s,
    }
}

/// The parameters of the cf1 comb filter.
pub const CF1: AdaCombConfig = comb(
    NOLACE_CF1_KERNEL_SIZE,
    NOLACE_CF1_LEFT_PADDING,
    NOLACE_CF1_FILTER_GAIN_A,
    NOLACE_CF1_FILTER_GAIN_B,
    NOLACE_CF1_LOG_GAIN_LIMIT,
);
/// The parameters of the cf2 comb filter.
pub const CF2: AdaCombConfig = comb(
    NOLACE_CF2_KERNEL_SIZE,
    NOLACE_CF2_LEFT_PADDING,
    NOLACE_CF2_FILTER_GAIN_A,
    NOLACE_CF2_FILTER_GAIN_B,
    NOLACE_CF2_LOG_GAIN_LIMIT,
);
/// The parameters of the af1 adaptive filter.
pub const AF1: AdaConvConfig = conv(
    NOLACE_AF1_IN_CHANNELS,
    NOLACE_AF1_OUT_CHANNELS,
    NOLACE_AF1_KERNEL_SIZE,
    NOLACE_AF1_LEFT_PADDING,
    NOLACE_AF1_FILTER_GAIN_A,
    NOLACE_AF1_FILTER_GAIN_B,
    NOLACE_AF1_SHAPE_GAIN,
);
/// The parameters of the af2 adaptive filter.
pub const AF2: AdaConvConfig = conv(
    NOLACE_AF2_IN_CHANNELS,
    NOLACE_AF2_OUT_CHANNELS,
    NOLACE_AF2_KERNEL_SIZE,
    NOLACE_AF2_LEFT_PADDING,
    NOLACE_AF2_FILTER_GAIN_A,
    NOLACE_AF2_FILTER_GAIN_B,
    NOLACE_AF2_SHAPE_GAIN,
);
/// The parameters of the af3 adaptive filter.
pub const AF3: AdaConvConfig = conv(
    NOLACE_AF3_IN_CHANNELS,
    NOLACE_AF3_OUT_CHANNELS,
    NOLACE_AF3_KERNEL_SIZE,
    NOLACE_AF3_LEFT_PADDING,
    NOLACE_AF3_FILTER_GAIN_A,
    NOLACE_AF3_FILTER_GAIN_B,
    NOLACE_AF3_SHAPE_GAIN,
);
/// The parameters of the af4 adaptive filter.
pub const AF4: AdaConvConfig = conv(
    NOLACE_AF4_IN_CHANNELS,
    NOLACE_AF4_OUT_CHANNELS,
    NOLACE_AF4_KERNEL_SIZE,
    NOLACE_AF4_LEFT_PADDING,
    NOLACE_AF4_FILTER_GAIN_A,
    NOLACE_AF4_FILTER_GAIN_B,
    NOLACE_AF4_SHAPE_GAIN,
);
/// The parameters of the tdshape1 shaping stage.
pub const TDSHAPE1: AdaShapeConfig = AdaShapeConfig {
    feature_dim: NOLACE_TDSHAPE1_FEATURE_DIM,
    frame_size: NOLACE_TDSHAPE1_FRAME_SIZE,
    avg_pool_k: NOLACE_TDSHAPE1_AVG_POOL_K,
    interpolate_k: 1,
};
/// The parameters of the tdshape2 shaping stage.
pub const TDSHAPE2: AdaShapeConfig = AdaShapeConfig {
    feature_dim: NOLACE_TDSHAPE2_FEATURE_DIM,
    frame_size: NOLACE_TDSHAPE2_FRAME_SIZE,
    avg_pool_k: NOLACE_TDSHAPE2_AVG_POOL_K,
    interpolate_k: 1,
};
/// The parameters of the tdshape3 shaping stage.
pub const TDSHAPE3: AdaShapeConfig = AdaShapeConfig {
    feature_dim: NOLACE_TDSHAPE3_FEATURE_DIM,
    frame_size: NOLACE_TDSHAPE3_FRAME_SIZE,
    avg_pool_k: NOLACE_TDSHAPE3_AVG_POOL_K,
    interpolate_k: 1,
};

/// `nolace_process_20ms_frame`: enhances the 320 samples `x_in` (16 kHz,
/// scaled to +-1) into `x_out` with the frame's `features`, bit counts
/// and pitch lags.
pub fn nolace_process_20ms_frame(
    nolace: &Nolace<'_>,
    state: &mut NolaceState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[i32; 4],
) {
    const F: usize = NOLACE_FRAME_SIZE;
    const _: () = assert!(
        NOLACE_AF1_OUT_CHANNELS == 2
            && NOLACE_AF2_OUT_CHANNELS == 2
            && NOLACE_AF3_OUT_CHANNELS == 2
    );
    let layers = &nolace.layers;
    let window = &nolace.window;
    let mut feature_buffer = [0.0f32; 4 * COND];
    let mut feature_transform_buffer = [0.0f32; 4 * COND];
    let mut x_buffer1 = [0.0f32; 8 * F];
    let mut x_buffer2 = [0.0f32; 8 * F];

    for i in 0..4 * F {
        x_buffer1[i] = x_in[i] - NOLACE_PREEMPH * state.preemph_mem;
        state.preemph_mem = x_in[i];
    }

    nolace_feature_net(
        nolace,
        state,
        &mut feature_buffer,
        features,
        numbits,
        periods,
    );

    // 1st comb filtering stage.
    for i in 0..4 {
        adacomb_process_frame(
            &mut state.cf1_state,
            &mut x_buffer1[i * F..(i + 1) * F],
            &feature_buffer[i * COND..],
            &layers.nolace_cf1_kernel,
            &layers.nolace_cf1_gain,
            &layers.nolace_cf1_global_gain,
            periods[i] as usize,
            &CF1,
            window,
        );
        compute_generic_conv1d(
            &layers.nolace_post_cf1,
            &mut feature_transform_buffer[i * COND..],
            &mut state.post_cf1_state,
            &feature_buffer[i * COND..],
            COND,
            Activation::Tanh,
        );
    }
    feature_buffer.copy_from_slice(&feature_transform_buffer);

    // 2nd comb filtering stage.
    for i in 0..4 {
        adacomb_process_frame(
            &mut state.cf2_state,
            &mut x_buffer1[i * F..(i + 1) * F],
            &feature_buffer[i * COND..],
            &layers.nolace_cf2_kernel,
            &layers.nolace_cf2_gain,
            &layers.nolace_cf2_global_gain,
            periods[i] as usize,
            &CF2,
            window,
        );
        compute_generic_conv1d(
            &layers.nolace_post_cf2,
            &mut feature_transform_buffer[i * COND..],
            &mut state.post_cf2_state,
            &feature_buffer[i * COND..],
            COND,
            Activation::Tanh,
        );
    }
    feature_buffer.copy_from_slice(&feature_transform_buffer);

    // Adaptive filtering into two channels.
    for i in 0..4 {
        adaconv_process_frame(
            &mut state.af1_state,
            &mut x_buffer2[i * 2 * F..],
            &x_buffer1[i * F..],
            &feature_buffer[i * COND..],
            &layers.nolace_af1_kernel,
            &layers.nolace_af1_gain,
            &AF1,
            window,
        );
        compute_generic_conv1d(
            &layers.nolace_post_af1,
            &mut feature_transform_buffer[i * COND..],
            &mut state.post_af1_state,
            &feature_buffer[i * COND..],
            COND,
            Activation::Tanh,
        );
    }
    feature_buffer.copy_from_slice(&feature_transform_buffer);

    // First shape-mix round: shape the second channel, then mix.
    for i in 0..4 {
        adashape_process_frame(
            &mut state.tdshape1_state,
            &mut x_buffer2[i * 2 * F + F..(i + 1) * 2 * F],
            &feature_buffer[i * COND..],
            &layers.nolace_tdshape1_alpha1_f,
            &layers.nolace_tdshape1_alpha1_t,
            &layers.nolace_tdshape1_alpha2,
            &TDSHAPE1,
        );
        adaconv_process_frame(
            &mut state.af2_state,
            &mut x_buffer1[i * 2 * F..],
            &x_buffer2[i * 2 * F..],
            &feature_buffer[i * COND..],
            &layers.nolace_af2_kernel,
            &layers.nolace_af2_gain,
            &AF2,
            window,
        );
        compute_generic_conv1d(
            &layers.nolace_post_af2,
            &mut feature_transform_buffer[i * COND..],
            &mut state.post_af2_state,
            &feature_buffer[i * COND..],
            COND,
            Activation::Tanh,
        );
    }
    feature_buffer.copy_from_slice(&feature_transform_buffer);

    // Second shape-mix round.
    for i in 0..4 {
        adashape_process_frame(
            &mut state.tdshape2_state,
            &mut x_buffer1[i * 2 * F + F..(i + 1) * 2 * F],
            &feature_buffer[i * COND..],
            &layers.nolace_tdshape2_alpha1_f,
            &layers.nolace_tdshape2_alpha1_t,
            &layers.nolace_tdshape2_alpha2,
            &TDSHAPE2,
        );
        adaconv_process_frame(
            &mut state.af3_state,
            &mut x_buffer2[i * 2 * F..],
            &x_buffer1[i * 2 * F..],
            &feature_buffer[i * COND..],
            &layers.nolace_af3_kernel,
            &layers.nolace_af3_gain,
            &AF3,
            window,
        );
        compute_generic_conv1d(
            &layers.nolace_post_af3,
            &mut feature_transform_buffer[i * COND..],
            &mut state.post_af3_state,
            &feature_buffer[i * COND..],
            COND,
            Activation::Tanh,
        );
    }
    feature_buffer.copy_from_slice(&feature_transform_buffer);

    // Third shape-mix round, mixing down to one channel.
    for i in 0..4 {
        adashape_process_frame(
            &mut state.tdshape3_state,
            &mut x_buffer2[i * 2 * F + F..(i + 1) * 2 * F],
            &feature_buffer[i * COND..],
            &layers.nolace_tdshape3_alpha1_f,
            &layers.nolace_tdshape3_alpha1_t,
            &layers.nolace_tdshape3_alpha2,
            &TDSHAPE3,
        );
        adaconv_process_frame(
            &mut state.af4_state,
            &mut x_buffer1[i * F..],
            &x_buffer2[i * 2 * F..],
            &feature_buffer[i * COND..],
            &layers.nolace_af4_kernel,
            &layers.nolace_af4_gain,
            &AF4,
            window,
        );
    }

    for i in 0..4 * F {
        x_out[i] = x_buffer1[i] + NOLACE_PREEMPH * state.deemph_mem;
        state.deemph_mem = x_out[i];
    }
}
