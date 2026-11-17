//! The adaptive filters of OSCE (libopus `dnn/nndsp.c`): `adaconv`
//! (adaptive convolution with cross-faded kernels), `adacomb` (adaptive
//! comb filter around the pitch lag) and `adashape` (temporal envelope
//! shaping).

#[cfg(not(feature = "std"))]
use crate::compat::Math;

use crate::dnn::common::{celt_log, celt_pitch_xcorr};
use crate::dnn::nnet::{
    compute_activation_inplace, compute_generic_conv1d, compute_generic_dense, Activation, LinearLayer,
};

pub const ADACONV_MAX_KERNEL_SIZE: usize = 32;
pub const ADACONV_MAX_INPUT_CHANNELS: usize = 3;
pub const ADACONV_MAX_OUTPUT_CHANNELS: usize = 3;
pub const ADACONV_MAX_FRAME_SIZE: usize = 240;
pub const ADACONV_MAX_OVERLAP_SIZE: usize = 120;

pub const ADACOMB_MAX_LAG: usize = 300;
pub const ADACOMB_MAX_KERNEL_SIZE: usize = 16;
pub const ADACOMB_MAX_FRAME_SIZE: usize = 80;
pub const ADACOMB_MAX_OVERLAP_SIZE: usize = 40;

pub const ADASHAPE_MAX_INPUT_DIM: usize = 512;
pub const ADASHAPE_MAX_FRAME_SIZE: usize = 240;

/// libopus `AdaConvState`.
#[derive(Clone, Debug)]
pub struct AdaConvState {
    pub history: [f32; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS],
    pub last_kernel: [f32; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS * ADACONV_MAX_OUTPUT_CHANNELS],
    pub last_gain: f32,
}

/// libopus `AdaCombState`.
#[derive(Clone, Debug)]
pub struct AdaCombState {
    pub history: [f32; ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG],
    pub last_kernel: [f32; ADACOMB_MAX_KERNEL_SIZE],
    pub last_global_gain: f32,
    pub last_pitch_lag: usize,
}

/// libopus `AdaShapeState`.
#[derive(Clone, Debug)]
pub struct AdaShapeState {
    pub conv_alpha1f_state: [f32; ADASHAPE_MAX_INPUT_DIM],
    pub conv_alpha1t_state: [f32; ADASHAPE_MAX_INPUT_DIM],
    pub conv_alpha2_state: [f32; ADASHAPE_MAX_FRAME_SIZE],
    pub interpolate_state: f32,
}

impl AdaConvState {
    /// `init_adaconv_state`: all zero.
    pub const fn new() -> Self {
        AdaConvState {
            history: [0.0; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS],
            last_kernel: [0.0; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS * ADACONV_MAX_OUTPUT_CHANNELS],
            last_gain: 0.0,
        }
    }
}

impl AdaCombState {
    /// `init_adacomb_state`: all zero.
    pub const fn new() -> Self {
        AdaCombState {
            history: [0.0; ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG],
            last_kernel: [0.0; ADACOMB_MAX_KERNEL_SIZE],
            last_global_gain: 0.0,
            last_pitch_lag: 0,
        }
    }
}

impl AdaShapeState {
    /// `init_adashape_state`: all zero.
    pub const fn new() -> Self {
        AdaShapeState {
            conv_alpha1f_state: [0.0; ADASHAPE_MAX_INPUT_DIM],
            conv_alpha1t_state: [0.0; ADASHAPE_MAX_INPUT_DIM],
            conv_alpha2_state: [0.0; ADASHAPE_MAX_FRAME_SIZE],
            interpolate_state: 0.0,
        }
    }
}

impl Default for AdaConvState {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AdaCombState {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AdaShapeState {
    fn default() -> Self {
        Self::new()
    }
}

/// The fixed parameters of one `adaconv_process_frame` stage.
#[derive(Clone, Copy, Debug)]
pub struct AdaConvConfig {
    pub frame_size: usize,
    pub overlap_size: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub left_padding: usize,
    pub filter_gain_a: f32,
    pub filter_gain_b: f32,
    pub shape_gain: f32,
}

/// The fixed parameters of one `adacomb_process_frame` stage.
#[derive(Clone, Copy, Debug)]
pub struct AdaCombConfig {
    pub frame_size: usize,
    pub overlap_size: usize,
    pub kernel_size: usize,
    pub left_padding: usize,
    pub filter_gain_a: f32,
    pub filter_gain_b: f32,
    pub log_gain_limit: f32,
}

/// The fixed parameters of one `adashape_process_frame` stage.
#[derive(Clone, Copy, Debug)]
pub struct AdaShapeConfig {
    pub feature_dim: usize,
    pub frame_size: usize,
    pub avg_pool_k: usize,
    pub interpolate_k: usize,
}

/// `compute_overlap_window`: `0.5 + 0.5 cos(pi (i + 0.5) / n)` for
/// `i < n = window.len()`, with `M_PI` and `cos` in double.
pub fn compute_overlap_window(window: &mut [f32]) {
    let n = window.len();
    for (i, w) in window.iter_mut().enumerate() {
        let arg = core::f64::consts::PI * (i as f32 + 0.5f32) as f64 / n as f64;
        *w = (0.5f32 as f64 + 0.5f32 as f64 * arg.cos()) as f32;
    }
}

/// `scale_kernel`: normalizes each output channel's kernel to unit 2-norm
/// over input channels and taps, then multiplies it by that channel's gain.
fn scale_kernel(kernel: &mut [f32], in_channels: usize, out_channels: usize, kernel_size: usize, gain: &[f32]) {
    let per_out = in_channels * kernel_size;
    for (k, &g) in kernel[..out_channels * per_out].chunks_exact_mut(per_out).zip(gain) {
        let mut norm = 0.0f32;
        for &v in k.iter() {
            norm += v * v;
        }
        let norm = (1.0f32 as f64 / (1e-6f32 as f64 + (norm as f64).sqrt())) as f32;
        for v in k.iter_mut() {
            *v *= norm * g;
        }
    }
}

/// `transform_gains`: `exp(a g + b)` with `exp` in double.
fn transform_gains(gains: &mut [f32], filter_gain_a: f32, filter_gain_b: f32) {
    for g in gains.iter_mut() {
        *g = ((filter_gain_a * *g + filter_gain_b) as f64).exp() as f32;
    }
}

/// `adaconv_process_frame`: filters `in_channels` channels of
/// `frame_size` samples (`x_in`, channel after channel) into
/// `out_channels` channels (`x_out`) with a kernel computed from
/// `features`, cross-faded over `overlap_size` samples from the previous
/// frame's kernel with `window`. Only the causal case
/// (`left_padding == kernel_size - 1`) exists, as in libopus.
pub fn adaconv_process_frame(
    state: &mut AdaConvState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    kernel_layer: &LinearLayer<'_>,
    gain_layer: &LinearLayer<'_>,
    cfg: &AdaConvConfig,
    window: &[f32],
) {
    const IN_STRIDE_MAX: usize = ADACONV_MAX_FRAME_SIZE + ADACONV_MAX_KERNEL_SIZE;
    let AdaConvConfig { frame_size, overlap_size, in_channels, out_channels, kernel_size, left_padding, .. } = *cfg;
    let mut output_buffer = [0.0f32; ADACONV_MAX_FRAME_SIZE * ADACONV_MAX_OUTPUT_CHANNELS];
    let mut kernel_buffer = [0.0f32; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS * ADACONV_MAX_OUTPUT_CHANNELS];
    let mut input_buffer = [0.0f32; ADACONV_MAX_INPUT_CHANNELS * IN_STRIDE_MAX];
    let mut channel_buffer0 = [0.0f32; ADACONV_MAX_OVERLAP_SIZE];
    let mut channel_buffer1 = [0.0f32; ADACONV_MAX_FRAME_SIZE];
    let mut gain_buffer = [0.0f32; ADACONV_MAX_OUTPUT_CHANNELS];

    assert!(cfg.shape_gain == 1.0);
    assert!(left_padding + 1 == kernel_size);
    assert!(kernel_size < frame_size);
    assert!(kernel_size <= ADACONV_MAX_KERNEL_SIZE && frame_size <= ADACONV_MAX_FRAME_SIZE);
    assert!(overlap_size <= ADACONV_MAX_OVERLAP_SIZE);
    assert!(in_channels <= ADACONV_MAX_INPUT_CHANNELS && out_channels <= ADACONV_MAX_OUTPUT_CHANNELS);

    // Each input channel: kernel_size history samples, then the frame.
    let stride = kernel_size + frame_size;
    for ch in 0..in_channels {
        input_buffer[ch * stride..ch * stride + kernel_size]
            .copy_from_slice(&state.history[ch * kernel_size..(ch + 1) * kernel_size]);
        input_buffer[ch * stride + kernel_size..(ch + 1) * stride]
            .copy_from_slice(&x_in[ch * frame_size..(ch + 1) * frame_size]);
    }

    compute_generic_dense(kernel_layer, &mut kernel_buffer, features, Activation::Linear);
    compute_generic_dense(gain_layer, &mut gain_buffer, features, Activation::Tanh);
    transform_gains(&mut gain_buffer[..out_channels], cfg.filter_gain_a, cfg.filter_gain_b);
    scale_kernel(&mut kernel_buffer, in_channels, out_channels, kernel_size, &gain_buffer);

    for oc in 0..out_channels {
        let out = &mut output_buffer[oc * frame_size..(oc + 1) * frame_size];
        for ic in 0..in_channels {
            let k = (oc * in_channels + ic) * kernel_size;
            let mut kernel0 = [0.0f32; ADACONV_MAX_KERNEL_SIZE];
            let mut kernel1 = [0.0f32; ADACONV_MAX_KERNEL_SIZE];
            kernel0[..kernel_size].copy_from_slice(&state.last_kernel[k..k + kernel_size]);
            kernel1[..kernel_size].copy_from_slice(&kernel_buffer[k..k + kernel_size]);
            // p_input + ic*stride - left_padding, with p_input = input_buffer + kernel_size.
            let y = &input_buffer[kernel_size + ic * stride - left_padding..];
            celt_pitch_xcorr(&kernel0, y, &mut channel_buffer0, ADACONV_MAX_KERNEL_SIZE, overlap_size);
            celt_pitch_xcorr(&kernel1, y, &mut channel_buffer1, ADACONV_MAX_KERNEL_SIZE, frame_size);
            for i in 0..overlap_size {
                out[i] += window[i] * channel_buffer0[i];
                out[i] += (1.0 - window[i]) * channel_buffer1[i];
            }
            for i in overlap_size..frame_size {
                out[i] += channel_buffer1[i];
            }
        }
    }

    x_out[..out_channels * frame_size].copy_from_slice(&output_buffer[..out_channels * frame_size]);

    for ch in 0..in_channels {
        let src = ch * stride + frame_size;
        state.history[ch * kernel_size..(ch + 1) * kernel_size].copy_from_slice(&input_buffer[src..src + kernel_size]);
    }
    let n = kernel_size * in_channels * out_channels;
    state.last_kernel[..n].copy_from_slice(&kernel_buffer[..n]);
}

/// `adacomb_process_frame` in place on `x` (one channel of `frame_size`
/// samples, as libopus calls it): adds the comb-filtered signal around
/// `pitch_lag` and scales by a global gain, both computed from `features`
/// and cross-faded over `overlap_size` samples from the previous frame's
/// lag, kernel and gain.
#[allow(clippy::too_many_arguments)]
pub fn adacomb_process_frame(
    state: &mut AdaCombState,
    x: &mut [f32],
    features: &[f32],
    kernel_layer: &LinearLayer<'_>,
    gain_layer: &LinearLayer<'_>,
    global_gain_layer: &LinearLayer<'_>,
    pitch_lag: usize,
    cfg: &AdaCombConfig,
    window: &[f32],
) {
    const HIST: usize = ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG;
    let AdaCombConfig { frame_size, overlap_size, kernel_size, left_padding, .. } = *cfg;
    let mut output_buffer = [0.0f32; ADACOMB_MAX_FRAME_SIZE];
    let mut output_buffer_last = [0.0f32; ADACOMB_MAX_FRAME_SIZE];
    let mut kernel_buffer = [0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    let mut input_buffer = [0.0f32; ADACOMB_MAX_FRAME_SIZE + HIST];
    let mut gain = [0.0f32; 1];
    let mut global_gain = [0.0f32; 1];

    assert!(kernel_size <= ADACOMB_MAX_KERNEL_SIZE && frame_size <= ADACOMB_MAX_FRAME_SIZE);
    assert!(overlap_size <= frame_size && left_padding <= kernel_size);
    assert!(pitch_lag <= ADACOMB_MAX_LAG && state.last_pitch_lag <= ADACOMB_MAX_LAG);

    // History (kernel_size + ADACOMB_MAX_LAG samples), then the frame.
    let p = kernel_size + ADACOMB_MAX_LAG;
    input_buffer[..p].copy_from_slice(&state.history[..p]);
    input_buffer[p..p + frame_size].copy_from_slice(&x[..frame_size]);

    compute_generic_dense(kernel_layer, &mut kernel_buffer, features, Activation::Linear);
    compute_generic_dense(gain_layer, &mut gain, features, Activation::Relu);
    compute_generic_dense(global_gain_layer, &mut global_gain, features, Activation::Tanh);
    let gain = ((cfg.log_gain_limit - gain[0]) as f64).exp() as f32;
    let global_gain = ((cfg.filter_gain_a * global_gain[0] + cfg.filter_gain_b) as f64).exp() as f32;
    scale_kernel(&mut kernel_buffer, 1, 1, kernel_size, &[gain]);

    let mut kernel = [0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    let mut last_kernel = [0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    kernel[..kernel_size].copy_from_slice(&kernel_buffer[..kernel_size]);
    last_kernel[..kernel_size].copy_from_slice(&state.last_kernel[..kernel_size]);

    celt_pitch_xcorr(
        &last_kernel,
        &input_buffer[p - left_padding - state.last_pitch_lag..],
        &mut output_buffer_last,
        ADACOMB_MAX_KERNEL_SIZE,
        overlap_size,
    );
    celt_pitch_xcorr(
        &kernel,
        &input_buffer[p - left_padding - pitch_lag..],
        &mut output_buffer,
        ADACOMB_MAX_KERNEL_SIZE,
        frame_size,
    );
    let last_gg = state.last_global_gain;
    let p_input = &input_buffer[p..p + frame_size];
    for i in 0..overlap_size {
        output_buffer[i] =
            last_gg * window[i] * output_buffer_last[i] + global_gain * (1.0 - window[i]) * output_buffer[i];
    }
    for i in 0..overlap_size {
        output_buffer[i] += (window[i] * last_gg + (1.0 - window[i]) * global_gain) * p_input[i];
    }
    for i in overlap_size..frame_size {
        output_buffer[i] = global_gain * (output_buffer[i] + p_input[i]);
    }
    x[..frame_size].copy_from_slice(&output_buffer[..frame_size]);

    state.last_kernel[..kernel_size].copy_from_slice(&kernel_buffer[..kernel_size]);
    state.history[..p].copy_from_slice(&input_buffer[frame_size..frame_size + p]);
    state.last_pitch_lag = pitch_lag;
    state.last_global_gain = global_gain;
}

/// `adashape_process_frame` in place on `x` (`frame_size` samples, as
/// libopus calls it): multiplies the signal by a gain curve computed from
/// `features` and its temporal envelope (mean magnitude over `avg_pool_k`
/// samples), upsampled by linear interpolation by `interpolate_k`.
#[allow(clippy::too_many_arguments)]
pub fn adashape_process_frame(
    state: &mut AdaShapeState,
    x: &mut [f32],
    features: &[f32],
    alpha1f: &LinearLayer<'_>,
    alpha1t: &LinearLayer<'_>,
    alpha2: &LinearLayer<'_>,
    cfg: &AdaShapeConfig,
) {
    let AdaShapeConfig { feature_dim, frame_size, avg_pool_k, interpolate_k } = *cfg;
    let mut in_buffer = [0.0f32; ADASHAPE_MAX_INPUT_DIM + ADASHAPE_MAX_FRAME_SIZE];
    let mut out_buffer = [0.0f32; ADASHAPE_MAX_FRAME_SIZE];
    let mut tmp_buffer = [0.0f32; ADASHAPE_MAX_FRAME_SIZE];
    let hidden_dim = frame_size / interpolate_k;
    let f = 1.0f32 / avg_pool_k as f32;

    assert!(frame_size % avg_pool_k == 0 && frame_size % interpolate_k == 0);
    assert!(frame_size <= ADASHAPE_MAX_FRAME_SIZE);
    assert!(feature_dim + frame_size / avg_pool_k + 1 < ADASHAPE_MAX_INPUT_DIM);

    let tenv_size = frame_size / avg_pool_k;
    in_buffer[..feature_dim].copy_from_slice(&features[..feature_dim]);

    // Temporal envelope after the features: log mean magnitudes minus their
    // mean, then the mean.
    let mut mean = 0.0f32;
    {
        let tenv = &mut in_buffer[feature_dim..feature_dim + tenv_size + 1];
        for i in 0..tenv_size {
            for k in 0..avg_pool_k {
                tenv[i] += x[i * avg_pool_k + k].abs();
            }
            tenv[i] = celt_log(tenv[i] * f + 1.52587890625e-05f32);
            mean += tenv[i];
        }
        mean /= tenv_size as f32;
        for v in tenv[..tenv_size].iter_mut() {
            *v -= mean;
        }
        tenv[tenv_size] = mean;
    }

    compute_generic_conv1d(alpha1f, &mut out_buffer, &mut state.conv_alpha1f_state, &in_buffer, feature_dim, Activation::Linear);
    compute_generic_conv1d(
        alpha1t,
        &mut tmp_buffer,
        &mut state.conv_alpha1t_state,
        &in_buffer[feature_dim..],
        tenv_size + 1,
        Activation::Linear,
    );
    // Leaky ReLU, with the double constant 0.2.
    for i in 0..hidden_dim {
        let tmp = out_buffer[i] + tmp_buffer[i];
        in_buffer[i] = if tmp >= 0.0 { tmp } else { (0.2f64 * tmp as f64) as f32 };
    }
    compute_generic_conv1d(alpha2, &mut tmp_buffer, &mut state.conv_alpha2_state, &in_buffer, hidden_dim, Activation::Linear);

    for i in 0..hidden_dim {
        for k in 0..interpolate_k {
            let alpha = (k + 1) as f32 / interpolate_k as f32;
            out_buffer[i * interpolate_k + k] = alpha * tmp_buffer[i] + (1.0 - alpha) * state.interpolate_state;
        }
        state.interpolate_state = tmp_buffer[i];
    }

    compute_activation_inplace(&mut out_buffer[..frame_size], Activation::Exp);
    for (v, &g) in x[..frame_size].iter_mut().zip(&out_buffer) {
        *v *= g;
    }
}
