//! Layer types and layer functions of libopus (`dnn/nnet.h`, `dnn/nnet.c`,
//! `dnn/nnet_arch.h`), with the generic kernels of `vec.rs`.
//!
//! Inputs and outputs may be longer than a layer reads or writes; only the
//! first `nb_inputs`/`nb_outputs` values are used.

use super::vec::{
    MAX_INPUTS, cgemv8x4, sgemv, softmax, sparse_cgemv8x4, sparse_sgemv8x4, vec_sigmoid, vec_tanh,
};
use super::weights::{F32s, I32s};

/// The activation after a layer (`ACTIVATION_*`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum Activation {
    Linear = 0,
    Sigmoid = 1,
    Tanh = 2,
    Relu = 3,
    /// `exp` of each value divided by their sum. (`SOFTMAX_HACK` is
    /// defined in nnet.c only; the compiled `compute_activation_c` of
    /// nnet_default.c normalizes.)
    Softmax = 4,
    Swish = 5,
    Exp = 6,
}

/// `MAX_RNN_NEURONS_ALL` with every model compiled in: the largest GRU
/// state of FARGAN, PLC, DRED, OSCE and the bandwidth extension.
pub const MAX_RNN_NEURONS_ALL: usize = 192;
/// `MAX_CONV_INPUTS_ALL`: IMAX(DRED_MAX_CONV_INPUTS, 1024).
pub const MAX_CONV_INPUTS_ALL: usize = 1024;
/// `MAX_ACTIVATIONS` (the buffer of `vec_swish`).
pub const MAX_ACTIVATIONS: usize = 4096;
/// `MAX_CONV2D_INPUTS`.
pub const MAX_CONV2D_INPUTS: usize = 8192;

/// An affine layer `y = W x + b` (libopus `LinearLayer`). The float
/// weights take precedence over the int8 weights when both exist.
///
/// - `weights` (int8) and sparse `float_weights`: blocks of 8 rows by 4
///   columns; a dense layer has the blocks of rows 0-7 for columns 0, 4,
///   ... then those of rows 8-15.
/// - dense `float_weights`: column-major, `w[j*nb_outputs + i]`.
/// - `weights_idx`: per block of 8 rows a count, then that many column
///   positions.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearLayer<'a> {
    pub bias: Option<F32s<'a>>,
    /// The bias for unsigned inputs of the SIMD kernels (`USE_SU_BIAS`);
    /// the generic kernels do not use it.
    pub subias: Option<F32s<'a>>,
    pub weights: Option<&'a [u8]>,
    pub float_weights: Option<F32s<'a>>,
    pub weights_idx: Option<I32s<'a>>,
    pub diag: Option<F32s<'a>>,
    pub scale: Option<F32s<'a>>,
    pub nb_inputs: usize,
    pub nb_outputs: usize,
}

/// A 2D convolution (libopus `Conv2dLayer`). Weight `[i][m][t][h]` is at
/// `i*in_channels*ktime*kheight + m*ktime*kheight + t*kheight + h`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Conv2dLayer<'a> {
    pub bias: Option<F32s<'a>>,
    pub float_weights: Option<F32s<'a>>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub ktime: usize,
    pub kheight: usize,
}

/// `compute_linear_c`.
pub fn compute_linear(linear: &LinearLayer<'_>, out: &mut [f32], input: &[f32]) {
    let m = linear.nb_inputs;
    let n = linear.nb_outputs;
    if let Some(fw) = linear.float_weights {
        match linear.weights_idx {
            Some(idx) => sparse_sgemv8x4(out, fw, idx, n, input),
            None => sgemv(out, fw, n, m, n, input),
        }
    } else if let Some(w) = linear.weights {
        let scale = linear.scale.expect("int8 layer without scale");
        match linear.weights_idx {
            Some(idx) => sparse_cgemv8x4(out, w, idx, scale, n, m, input),
            None => cgemv8x4(out, w, scale, n, m, input),
        }
    } else {
        out[..n].fill(0.0);
    }
    if let Some(bias) = linear.bias {
        for (i, o) in out[..n].iter_mut().enumerate() {
            *o += bias.at(i);
        }
    }
    if let Some(diag) = linear.diag {
        debug_assert_eq!(3 * m, n);
        for i in 0..m {
            out[i] += diag.at(i) * input[i];
            out[i + m] += diag.at(i + m) * input[i];
            out[i + 2 * m] += diag.at(i + 2 * m) * input[i];
        }
    }
}

/// `relu`: `x < 0 ? 0 : x`.
#[inline(always)]
fn relu(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { x }
}

/// `compute_activation_c` on `x` in place.
pub fn compute_activation_inplace(x: &mut [f32], activation: Activation) {
    match activation {
        Activation::Sigmoid => {
            for v in x.iter_mut() {
                *v = super::vec::sigmoid_approx(*v);
            }
        }
        Activation::Tanh => {
            for v in x.iter_mut() {
                *v = super::vec::tanh_approx(*v);
            }
        }
        Activation::Swish => {
            for v in x.iter_mut() {
                *v *= super::vec::sigmoid_approx(*v);
            }
        }
        Activation::Relu => {
            for v in x.iter_mut() {
                *v = relu(*v);
            }
        }
        Activation::Exp => {
            for v in x.iter_mut() {
                *v = super::vec::lpcnet_exp(*v);
            }
        }
        Activation::Softmax => softmax_normalized(x),
        Activation::Linear => {}
    }
}

/// The ACTIVATION_SOFTMAX branch of `compute_activation_c`.
fn softmax_normalized(x: &mut [f32]) {
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = super::vec::lpcnet_exp(*v);
        sum += *v;
    }
    let sum = (1.0f64 / (sum as f64 + 1e-30)) as f32;
    for v in x.iter_mut() {
        *v *= sum;
    }
}

/// `compute_activation_c`: `output[..n] = activation(input[..n])`.
pub fn compute_activation(output: &mut [f32], input: &[f32], n: usize, activation: Activation) {
    let (output, input) = (&mut output[..n], &input[..n]);
    match activation {
        Activation::Sigmoid => vec_sigmoid(output, input),
        Activation::Tanh => vec_tanh(output, input),
        Activation::Swish => {
            let mut tmp = [0.0f32; MAX_ACTIVATIONS];
            vec_sigmoid(&mut tmp[..n], input);
            for ((o, &x), &t) in output.iter_mut().zip(input).zip(&tmp[..n]) {
                *o = x * t;
            }
        }
        Activation::Relu => {
            for (o, &x) in output.iter_mut().zip(input) {
                *o = relu(x);
            }
        }
        Activation::Exp => softmax(output, input),
        Activation::Softmax => {
            output.copy_from_slice(input);
            softmax_normalized(output);
        }
        Activation::Linear => output.copy_from_slice(input),
    }
}

/// `compute_generic_dense`.
pub fn compute_generic_dense(
    layer: &LinearLayer<'_>,
    output: &mut [f32],
    input: &[f32],
    activation: Activation,
) {
    compute_linear(layer, output, input);
    compute_activation_inplace(&mut output[..layer.nb_outputs], activation);
}

/// `compute_generic_gru`: one step of a GRU with state `state`
/// (`recurrent_weights.nb_inputs` values).
pub fn compute_generic_gru(
    input_weights: &LinearLayer<'_>,
    recurrent_weights: &LinearLayer<'_>,
    state: &mut [f32],
    input: &[f32],
) {
    let n = recurrent_weights.nb_inputs;
    assert_eq!(3 * n, recurrent_weights.nb_outputs);
    assert_eq!(input_weights.nb_outputs, recurrent_weights.nb_outputs);
    assert!(n <= MAX_RNN_NEURONS_ALL);
    let mut zrh = [0.0f32; 3 * MAX_RNN_NEURONS_ALL];
    let mut recur = [0.0f32; 3 * MAX_RNN_NEURONS_ALL];
    compute_linear(input_weights, &mut zrh, input);
    compute_linear(recurrent_weights, &mut recur, state);
    for i in 0..2 * n {
        zrh[i] += recur[i];
    }
    compute_activation_inplace(&mut zrh[..2 * n], Activation::Sigmoid);
    let (zr, h) = zrh.split_at_mut(2 * n);
    let (z, r) = zr.split_at(n);
    for i in 0..n {
        h[i] += recur[2 * n + i] * r[i];
    }
    compute_activation_inplace(&mut h[..n], Activation::Tanh);
    for i in 0..n {
        h[i] = z[i] * state[i] + (1.0 - z[i]) * h[i];
    }
    state[..n].copy_from_slice(&h[..n]);
}

/// `compute_glu`: `output = input * sigmoid(W input + b)`.
pub fn compute_glu(layer: &LinearLayer<'_>, output: &mut [f32], input: &[f32]) {
    assert_eq!(layer.nb_inputs, layer.nb_outputs);
    let n = layer.nb_outputs;
    let mut act2 = [0.0f32; MAX_INPUTS];
    compute_linear(layer, &mut act2, input);
    compute_activation_inplace(&mut act2[..n], Activation::Sigmoid);
    for i in 0..n {
        output[i] = input[i] * act2[i];
    }
}

/// `compute_glu` with the output in place of the input.
pub fn compute_glu_inplace(layer: &LinearLayer<'_>, x: &mut [f32]) {
    assert_eq!(layer.nb_inputs, layer.nb_outputs);
    let n = layer.nb_outputs;
    let mut act2 = [0.0f32; MAX_INPUTS];
    compute_linear(layer, &mut act2, x);
    compute_activation_inplace(&mut act2[..n], Activation::Sigmoid);
    for i in 0..n {
        x[i] *= act2[i];
    }
}

/// `compute_generic_conv1d`: the layer input is the memory
/// (`nb_inputs - input_size` values) followed by `input`.
pub fn compute_generic_conv1d(
    layer: &LinearLayer<'_>,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    activation: Activation,
) {
    let ni = layer.nb_inputs;
    assert!(ni <= MAX_CONV_INPUTS_ALL);
    let mut tmp = [0.0f32; MAX_CONV_INPUTS_ALL];
    let hist = ni - input_size;
    tmp[..hist].copy_from_slice(&mem[..hist]);
    tmp[hist..ni].copy_from_slice(&input[..input_size]);
    compute_linear(layer, output, &tmp);
    compute_activation_inplace(&mut output[..layer.nb_outputs], activation);
    mem[..hist].copy_from_slice(&tmp[input_size..input_size + hist]);
}

/// `compute_generic_conv1d_dilation`.
pub fn compute_generic_conv1d_dilation(
    layer: &LinearLayer<'_>,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    dilation: usize,
    activation: Activation,
) {
    let ni = layer.nb_inputs;
    assert!(ni <= MAX_CONV_INPUTS_ALL);
    let ksize = ni / input_size;
    let mut tmp = [0.0f32; MAX_CONV_INPUTS_ALL];
    let hist = ni - input_size;
    if dilation == 1 {
        tmp[..hist].copy_from_slice(&mem[..hist]);
    } else {
        for i in 0..ksize - 1 {
            let src = i * input_size * dilation;
            tmp[i * input_size..(i + 1) * input_size].copy_from_slice(&mem[src..src + input_size]);
        }
    }
    tmp[hist..ni].copy_from_slice(&input[..input_size]);
    compute_linear(layer, output, &tmp);
    compute_activation_inplace(&mut output[..layer.nb_outputs], activation);
    if dilation == 1 {
        mem[..hist].copy_from_slice(&tmp[input_size..input_size + hist]);
    } else {
        let total = input_size * dilation * (ksize - 1);
        mem.copy_within(input_size..total, 0);
        mem[total - input_size..total].copy_from_slice(&input[..input_size]);
    }
}

/// `conv2d_float`: each product is added to the output on its own.
fn conv2d_float(
    out: &mut [f32],
    weights: F32s<'_>,
    in_channels: usize,
    out_channels: usize,
    ktime: usize,
    kheight: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    let in_stride = height + kheight - 1;
    for i in 0..out_channels {
        let o = &mut out[i * hstride..i * hstride + height];
        o.fill(0.0);
        for m in 0..in_channels {
            for t in 0..ktime {
                for h in 0..kheight {
                    let w = weights.at(i * in_channels * ktime * kheight
                        + m * ktime * kheight
                        + t * kheight
                        + h);
                    let x = &input[t * in_channels * in_stride + m * in_stride + h..];
                    for j in 0..height {
                        o[j] += w * x[j];
                    }
                }
            }
        }
    }
}

/// `conv2d_3x3_float`: the nine products of an input channel are summed
/// first, then added to the output.
fn conv2d_3x3_float(
    out: &mut [f32],
    weights: F32s<'_>,
    in_channels: usize,
    out_channels: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    let in_stride = height + 2;
    for i in 0..out_channels {
        let o = &mut out[i * hstride..i * hstride + height];
        o.fill(0.0);
        for m in 0..in_channels {
            let wb = i * in_channels * 9 + m * 9;
            let w: [f32; 9] = core::array::from_fn(|k| weights.at(wb + k));
            let x0 = &input[m * in_stride..];
            let x1 = &input[in_channels * in_stride + m * in_stride..];
            let x2 = &input[2 * in_channels * in_stride + m * in_stride..];
            for j in 0..height {
                o[j] += w[0] * x0[j]
                    + w[1] * x0[j + 1]
                    + w[2] * x0[j + 2]
                    + w[3] * x1[j]
                    + w[4] * x1[j + 1]
                    + w[5] * x1[j + 2]
                    + w[6] * x2[j]
                    + w[7] * x2[j + 1]
                    + w[8] * x2[j + 2];
            }
        }
    }
}

/// `compute_conv2d_c`: one time step of a 2D convolution whose memory holds
/// the previous `ktime - 1` inputs.
pub fn compute_conv2d(
    conv: &Conv2dLayer<'_>,
    out: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    height: usize,
    hstride: usize,
    activation: Activation,
) {
    let time_stride = conv.in_channels * (height + conv.kheight - 1);
    assert!(conv.ktime * time_stride <= MAX_CONV2D_INPUTS);
    let hist = (conv.ktime - 1) * time_stride;
    let mut in_buf = [0.0f32; MAX_CONV2D_INPUTS];
    in_buf[..hist].copy_from_slice(&mem[..hist]);
    in_buf[hist..hist + time_stride].copy_from_slice(&input[..time_stride]);
    mem[..hist].copy_from_slice(&in_buf[time_stride..time_stride + hist]);
    let w = conv.float_weights.expect("conv2d layer without weights");
    if conv.kheight == 3 && conv.ktime == 3 {
        conv2d_3x3_float(
            out,
            w,
            conv.in_channels,
            conv.out_channels,
            &in_buf,
            height,
            hstride,
        );
    } else {
        conv2d_float(
            out,
            w,
            conv.in_channels,
            conv.out_channels,
            conv.ktime,
            conv.kheight,
            &in_buf,
            height,
            hstride,
        );
    }
    if let Some(bias) = conv.bias {
        for i in 0..conv.out_channels {
            for j in 0..height {
                out[i * hstride + j] += bias.at(i);
            }
        }
    }
    for i in 0..conv.out_channels {
        compute_activation_inplace(&mut out[i * hstride..i * hstride + height], activation);
    }
}
