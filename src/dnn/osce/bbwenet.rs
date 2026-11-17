//! BBWENet, the blind bandwidth extension from 16 to 48 kHz (libopus
//! `dnn/osce.c`): an adaptive filter splits the signal into three
//! channels, which are upsampled twice (2x, then 1.5x), shaped or
//! distorted non-linearly and mixed again.

use super::nndsp::{
    adaconv_process_frame, adashape_process_frame, compute_overlap_window, AdaConvConfig, AdaConvState,
    AdaShapeConfig, AdaShapeState,
};
use super::OSCE_BWE_OUTPUT_DELAY;
use crate::dnn::common::{celt_log, celt_sin};
use crate::dnn::models::bbwenet::*;
use crate::dnn::nnet::{compute_generic_conv1d, compute_generic_dense, compute_generic_gru, Activation};

/// libopus `BBWENet`: the layers and the overlap windows of the three
/// adaptive filters.
#[derive(Clone, Debug)]
pub struct Bbwenet<'a> {
    pub layers: BbwenetLayers<'a>,
    pub window16: [f32; BBWENET_AF1_OVERLAP_SIZE],
    pub window32: [f32; BBWENET_AF2_OVERLAP_SIZE],
    pub window48: [f32; BBWENET_AF3_OVERLAP_SIZE],
}

impl<'a> Bbwenet<'a> {
    /// `init_bbwenet` from initialized layers.
    pub fn new(layers: BbwenetLayers<'a>) -> Self {
        let mut b = Bbwenet {
            layers,
            window16: [0.0; BBWENET_AF1_OVERLAP_SIZE],
            window32: [0.0; BBWENET_AF2_OVERLAP_SIZE],
            window48: [0.0; BBWENET_AF3_OVERLAP_SIZE],
        };
        compute_overlap_window(&mut b.window16);
        compute_overlap_window(&mut b.window32);
        compute_overlap_window(&mut b.window48);
        b
    }
}

/// libopus `resamp_state`: the memories of `upsamp_2x` and `interpol_3_2`
/// for one channel.
#[derive(Clone, Debug, Default)]
pub struct ResampState {
    pub upsamp_buffer: [[f32; 3]; 2],
    pub interpol_buffer: [f32; 8],
}

/// libopus `BBWENetState`.
#[derive(Clone, Debug)]
pub struct BbwenetState {
    pub feature_net_conv1_state: [f32; BBWENET_FNET_CONV1_STATE_SIZE],
    pub feature_net_conv2_state: [f32; BBWENET_FNET_CONV2_STATE_SIZE],
    pub feature_net_gru_state: [f32; BBWENET_FNET_GRU_STATE_SIZE],
    /// `outbut_buffer`: the last `OSCE_BWE_OUTPUT_DELAY` output samples.
    pub output_buffer: [i16; OSCE_BWE_OUTPUT_DELAY],
    pub af1_state: AdaConvState,
    pub af2_state: AdaConvState,
    pub af3_state: AdaConvState,
    pub tdshape1_state: AdaShapeState,
    pub tdshape2_state: AdaShapeState,
    pub resampler_state: [ResampState; 3],
}

impl BbwenetState {
    /// `reset_bbwenet_state`: all zero.
    pub const fn new() -> Self {
        const R: ResampState = ResampState { upsamp_buffer: [[0.0; 3]; 2], interpol_buffer: [0.0; 8] };
        BbwenetState {
            feature_net_conv1_state: [0.0; BBWENET_FNET_CONV1_STATE_SIZE],
            feature_net_conv2_state: [0.0; BBWENET_FNET_CONV2_STATE_SIZE],
            feature_net_gru_state: [0.0; BBWENET_FNET_GRU_STATE_SIZE],
            output_buffer: [0; OSCE_BWE_OUTPUT_DELAY],
            af1_state: AdaConvState::new(),
            af2_state: AdaConvState::new(),
            af3_state: AdaConvState::new(),
            tdshape1_state: AdaShapeState::new(),
            tdshape2_state: AdaShapeState::new(),
            resampler_state: [R; 3],
        }
    }
}

impl Default for BbwenetState {
    fn default() -> Self {
        Self::new()
    }
}

const GRU: usize = BBWENET_FNET_GRU_STATE_SIZE;

/// `bbwe_feature_net`: `BBWENET_COND_DIM` conditioning values per 5 ms
/// subframe (two per frame of features).
fn bbwe_feature_net(
    bbwenet: &Bbwenet<'_>,
    state: &mut BbwenetState,
    output: &mut [f32; 4 * BBWENET_COND_DIM],
    features: &[f32],
    num_frames: usize,
) {
    const _: () = assert!(
        GRU == BBWENET_FNET_TCONV_OUT_CHANNELS
            && BBWENET_FNET_TCONV_OUT_CHANNELS == BBWENET_FNET_CONV2_OUT_SIZE
            && BBWENET_FNET_CONV2_OUT_SIZE == BBWENET_FNET_CONV1_OUT_SIZE
            && BBWENET_FNET_TCONV_STRIDE == 2
            && GRU == BBWENET_COND_DIM
    );
    let layers = &bbwenet.layers;
    let mut input_buffer = [0.0f32; 4 * GRU];
    let mut output_buffer = [0.0f32; 4 * GRU];

    for i in 0..num_frames {
        compute_generic_conv1d(
            &layers.bbwenet_fnet_conv1,
            &mut output_buffer[i * BBWENET_FNET_CONV1_OUT_SIZE..],
            &mut state.feature_net_conv1_state,
            &features[i * BBWENET_FEATURE_DIM..],
            BBWENET_FEATURE_DIM,
            Activation::Tanh,
        );
    }
    input_buffer[..num_frames * GRU].copy_from_slice(&output_buffer[..num_frames * GRU]);

    for i in 0..num_frames {
        compute_generic_conv1d(
            &layers.bbwenet_fnet_conv2,
            &mut output_buffer[i * BBWENET_FNET_CONV2_OUT_SIZE..],
            &mut state.feature_net_conv2_state,
            &input_buffer[i * BBWENET_FNET_CONV1_OUT_SIZE..],
            BBWENET_FNET_CONV1_OUT_SIZE,
            Activation::Tanh,
        );
    }
    input_buffer[..num_frames * GRU].copy_from_slice(&output_buffer[..num_frames * GRU]);

    const TC: usize = BBWENET_FNET_TCONV_OUT_CHANNELS * BBWENET_FNET_TCONV_STRIDE;
    for i in 0..num_frames {
        compute_generic_dense(
            &layers.bbwenet_fnet_tconv,
            &mut output_buffer[i * TC..],
            &input_buffer[i * BBWENET_FNET_CONV2_OUT_SIZE..],
            Activation::Tanh,
        );
    }
    input_buffer[..num_frames * TC].copy_from_slice(&output_buffer[..num_frames * TC]);

    for i in 0..BBWENET_FNET_TCONV_STRIDE * num_frames {
        compute_generic_gru(
            &layers.bbwenet_fnet_gru_input,
            &layers.bbwenet_fnet_gru_recurrent,
            &mut state.feature_net_gru_state,
            &input_buffer[i * BBWENET_FNET_TCONV_OUT_CHANNELS..],
        );
        output[i * GRU..(i + 1) * GRU].copy_from_slice(&state.feature_net_gru_state);
    }
}

const HQ_2X_EVEN: [f32; 3] = [0.026641845703125, 0.228668212890625, -0.4036407470703125];
const HQ_2X_ODD: [f32; 3] = [0.104583740234375, 0.3932037353515625, -0.152496337890625];

// Double literals in float arrays: each is rounded from its double value.
const FRAC_01_24: [f32; 8] = [
    0.00576782f64 as f32,
    -0.01831055f64 as f32,
    0.01882935f64 as f32,
    0.9328308f64 as f32,
    0.09143066f64 as f32,
    -0.04196167f64 as f32,
    0.01296997f64 as f32,
    -0.00140381f64 as f32,
];

const FRAC_17_24: [f32; 8] = [
    -3.14331055e-03f64 as f32,
    2.73437500e-02f64 as f32,
    -1.06414795e-01f64 as f32,
    3.64685059e-01f64 as f32,
    8.03863525e-01f64 as f32,
    -1.02233887e-01f64 as f32,
    1.61437988e-02f64 as f32,
    -1.22070312e-04f64 as f32,
];

const FRAC_09_24: [f32; 8] = [
    -0.00146484f64 as f32,
    0.02313232f64 as f32,
    -0.12072754f64 as f32,
    0.7315979f64 as f32,
    0.4621277f64 as f32,
    -0.12075806f64 as f32,
    0.0295105f64 as f32,
    -0.00326538f64 as f32,
];

/// `apply_valin_activation`: `x sin(log(|x| + 1e-6))` with the CELT
/// approximations of `sin` and `log`.
pub fn apply_valin_activation(x: &mut [f32]) {
    for v in x.iter_mut() {
        let y = (v.abs() as f64 + 1e-6f32 as f64) as f32;
        *v *= celt_sin(celt_log(y));
    }
}

/// An 8-tap FIR output: the products summed left to right from the first.
#[inline(always)]
fn fir8(x: &[f32], h: &[f32; 8]) -> f32 {
    let mut s = x[0] * h[0];
    for k in 1..8 {
        s += x[k] * h[k];
    }
    s
}

/// The history length of `interpol_3_2` (`DELAY_SAMPLES`).
const DELAY_SAMPLES: usize = 8;

/// `interpol_3_2`: 1.5x upsampling of `x_in` (an even number of samples)
/// with three fractional-delay filters.
pub fn interpol_3_2(state: &mut ResampState, x_out: &mut [f32], x_in: &[f32]) {
    let n = x_in.len();
    let mut buffer = [0.0f32; 8 * BBWENET_FRAME_SIZE16 + DELAY_SAMPLES];
    assert!(n > 1 && n < 8 * BBWENET_FRAME_SIZE16 && n.is_multiple_of(2));

    buffer[..DELAY_SAMPLES].copy_from_slice(&state.interpol_buffer);
    buffer[DELAY_SAMPLES..DELAY_SAMPLES + n].copy_from_slice(x_in);

    let mut o = 0;
    for i in (0..n).step_by(2) {
        x_out[o] = fir8(&buffer[i..], &FRAC_01_24);
        x_out[o + 1] = fir8(&buffer[i..], &FRAC_17_24);
        x_out[o + 2] = fir8(&buffer[i + 1..], &FRAC_09_24);
        o += 3;
    }

    state.interpol_buffer.copy_from_slice(&buffer[n..n + DELAY_SAMPLES]);
}

/// One allpass section of `upsamp_2x`: returns the output, updates `s`.
#[inline(always)]
fn allpass(s: &mut f32, x: f32, c: f32) -> f32 {
    let y = x - *s;
    let xx = y * c;
    let out = *s + xx;
    *s = x + xx;
    out
}

/// `upsamp_2x`: 2x upsampling with two chains of three allpass sections
/// (the even and the odd output samples).
pub fn upsamp_2x(state: &mut ResampState, x_out: &mut [f32], x_in: &[f32]) {
    let n = x_in.len();
    assert!(n > 1 && n < 4 * BBWENET_FRAME_SIZE16);
    let [s_even, s_odd] = &mut state.upsamp_buffer;
    for (k, &x) in x_in.iter().enumerate() {
        let t1 = allpass(&mut s_even[0], x, HQ_2X_EVEN[0]);
        let t2 = allpass(&mut s_even[1], t1, HQ_2X_EVEN[1]);
        x_out[2 * k] = allpass(&mut s_even[2], t2, 1.0 + HQ_2X_EVEN[2]);

        let t1 = allpass(&mut s_odd[0], x, HQ_2X_ODD[0]);
        let t2 = allpass(&mut s_odd[1], t1, HQ_2X_ODD[1]);
        x_out[2 * k + 1] = allpass(&mut s_odd[2], t2, 1.0 + HQ_2X_ODD[2]);
    }
}

/// The parameters of the af1 adaptive filter.
pub const AF1: AdaConvConfig = AdaConvConfig {
    frame_size: BBWENET_AF1_FRAME_SIZE,
    overlap_size: BBWENET_AF1_OVERLAP_SIZE,
    in_channels: BBWENET_AF1_IN_CHANNELS,
    out_channels: BBWENET_AF1_OUT_CHANNELS,
    kernel_size: BBWENET_AF1_KERNEL_SIZE,
    left_padding: BBWENET_AF1_LEFT_PADDING,
    filter_gain_a: BBWENET_AF1_FILTER_GAIN_A,
    filter_gain_b: BBWENET_AF1_FILTER_GAIN_B,
    shape_gain: BBWENET_AF1_SHAPE_GAIN,
};
/// The parameters of the af2 adaptive filter.
pub const AF2: AdaConvConfig = AdaConvConfig {
    frame_size: BBWENET_AF2_FRAME_SIZE,
    overlap_size: BBWENET_AF2_OVERLAP_SIZE,
    in_channels: BBWENET_AF2_IN_CHANNELS,
    out_channels: BBWENET_AF2_OUT_CHANNELS,
    kernel_size: BBWENET_AF2_KERNEL_SIZE,
    left_padding: BBWENET_AF2_LEFT_PADDING,
    filter_gain_a: BBWENET_AF2_FILTER_GAIN_A,
    filter_gain_b: BBWENET_AF2_FILTER_GAIN_B,
    shape_gain: BBWENET_AF2_SHAPE_GAIN,
};
/// The parameters of the af3 adaptive filter.
pub const AF3: AdaConvConfig = AdaConvConfig {
    frame_size: BBWENET_AF3_FRAME_SIZE,
    overlap_size: BBWENET_AF3_OVERLAP_SIZE,
    in_channels: BBWENET_AF3_IN_CHANNELS,
    out_channels: BBWENET_AF3_OUT_CHANNELS,
    kernel_size: BBWENET_AF3_KERNEL_SIZE,
    left_padding: BBWENET_AF3_LEFT_PADDING,
    filter_gain_a: BBWENET_AF3_FILTER_GAIN_A,
    filter_gain_b: BBWENET_AF3_FILTER_GAIN_B,
    shape_gain: BBWENET_AF3_SHAPE_GAIN,
};
/// The parameters of the tdshape1 shaping stage.
pub const TDSHAPE1: AdaShapeConfig = AdaShapeConfig {
    feature_dim: BBWENET_TDSHAPE1_FEATURE_DIM,
    frame_size: BBWENET_TDSHAPE1_FRAME_SIZE,
    avg_pool_k: BBWENET_TDSHAPE1_AVG_POOL_K,
    interpolate_k: BBWENET_TDSHAPE1_INTERPOLATE_K,
};
/// The parameters of the tdshape2 shaping stage.
pub const TDSHAPE2: AdaShapeConfig = AdaShapeConfig {
    feature_dim: BBWENET_TDSHAPE2_FEATURE_DIM,
    frame_size: BBWENET_TDSHAPE2_FRAME_SIZE,
    avg_pool_k: BBWENET_TDSHAPE2_AVG_POOL_K,
    interpolate_k: BBWENET_TDSHAPE2_INTERPOLATE_K,
};

/// `bbwenet_process_frames`: extends `num_frames` (1 or 2) frames of 10 ms
/// at 16 kHz (`x_in`, scaled to +-1) to 48 kHz (`x_out`, three times as
/// many samples) with the frames' `OSCE_BWE_FEATURE_DIM` features each.
pub fn bbwenet_process_frames(
    bbwenet: &Bbwenet<'_>,
    state: &mut BbwenetState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    num_frames: usize,
) {
    const F16: usize = BBWENET_AF1_FRAME_SIZE;
    const F32: usize = BBWENET_TDSHAPE1_FRAME_SIZE;
    const F48: usize = BBWENET_TDSHAPE2_FRAME_SIZE;
    const _: () = assert!(
        BBWENET_AF1_OUT_CHANNELS == 3
            && BBWENET_AF2_OUT_CHANNELS == 3
            && BBWENET_AF3_OUT_CHANNELS == 1
            && 2 * F16 == F32
            && BBWENET_AF2_FRAME_SIZE == F32
            && 3 * F32 == 2 * F48
            && BBWENET_AF3_FRAME_SIZE == F48
    );
    assert!(num_frames <= 2);
    let layers = &bbwenet.layers;
    let num_subframes = 2 * num_frames;
    let mut latent_features = [0.0f32; 4 * BBWENET_COND_DIM];
    let mut x_buffer1 = [0.0f32; 3 * 3 * 4 * 3 * BBWENET_FRAME_SIZE16];
    let mut x_buffer2 = [0.0f32; 3 * 3 * 4 * 3 * BBWENET_FRAME_SIZE16];
    const COND: usize = BBWENET_COND_DIM;

    bbwe_feature_net(bbwenet, state, &mut latent_features, features, num_frames);

    // First adaptive filter: three channels at 16 kHz.
    for i in 0..num_subframes {
        adaconv_process_frame(
            &mut state.af1_state,
            &mut x_buffer1[i * 3 * F16..],
            &x_in[i * F16..],
            &latent_features[i * COND..],
            &layers.bbwenet_af1_kernel,
            &layers.bbwenet_af1_gain,
            &AF1,
            &bbwenet.window16,
        );
    }

    // 2x upsampling, shaping of channel 2 and activation of channel 3.
    for i in 0..num_subframes {
        for ch in 0..3 {
            upsamp_2x(
                &mut state.resampler_state[ch],
                &mut x_buffer2[i * 3 * F32 + ch * F32..],
                &x_buffer1[i * 3 * F16 + ch * F16..i * 3 * F16 + (ch + 1) * F16],
            );
        }
        adashape_process_frame(
            &mut state.tdshape1_state,
            &mut x_buffer2[i * 3 * F32 + F32..i * 3 * F32 + 2 * F32],
            &latent_features[i * COND..],
            &layers.bbwenet_tdshape1_alpha1_f,
            &layers.bbwenet_tdshape1_alpha1_t,
            &layers.bbwenet_tdshape1_alpha2,
            &TDSHAPE1,
        );
        apply_valin_activation(&mut x_buffer2[i * 3 * F32 + 2 * F32..(i + 1) * 3 * F32]);
    }

    // Mixing at 32 kHz.
    for i in 0..num_subframes {
        adaconv_process_frame(
            &mut state.af2_state,
            &mut x_buffer1[i * 3 * F32..],
            &x_buffer2[i * 3 * F32..],
            &latent_features[i * COND..],
            &layers.bbwenet_af2_kernel,
            &layers.bbwenet_af2_gain,
            &AF2,
            &bbwenet.window32,
        );
    }

    // 1.5x interpolation, shaping of channel 2 and activation of channel 3.
    for i in 0..num_subframes {
        for ch in 0..3 {
            interpol_3_2(
                &mut state.resampler_state[ch],
                &mut x_buffer2[i * 3 * F48 + ch * F48..],
                &x_buffer1[i * 3 * F32 + ch * F32..i * 3 * F32 + (ch + 1) * F32],
            );
        }
        adashape_process_frame(
            &mut state.tdshape2_state,
            &mut x_buffer2[i * 3 * F48 + F48..i * 3 * F48 + 2 * F48],
            &latent_features[i * COND..],
            &layers.bbwenet_tdshape2_alpha1_f,
            &layers.bbwenet_tdshape2_alpha1_t,
            &layers.bbwenet_tdshape2_alpha2,
            &TDSHAPE2,
        );
        apply_valin_activation(&mut x_buffer2[i * 3 * F48 + 2 * F48..(i + 1) * 3 * F48]);
    }

    // Final mixing to one channel at 48 kHz.
    for i in 0..num_subframes {
        adaconv_process_frame(
            &mut state.af3_state,
            &mut x_out[i * F48..],
            &x_buffer2[i * 3 * F48..],
            &latent_features[i * COND..],
            &layers.bbwenet_af3_kernel,
            &layers.bbwenet_af3_gain,
            &AF3,
            &bbwenet.window48,
        );
    }
}
