//! Layer types and layer functions of libopus (`dnn/nnet.h`, `dnn/nnet.c`,
//! `dnn/nnet_arch.h`), with the generic kernels of `vec.rs`.
//!
//! Inputs and outputs may be longer than a layer reads or writes; only the
//! first `nb_inputs`/`nb_outputs` values are used.

use super::vec::{Elem, cgemv8x4_at, sgemv, sparse_sgemv8x4, vec_apply};
use super::weights::{F32s, I32s};
use crate::Arch;

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
    /// The bias of the int8 layers at the x86 levels, whose kernels take
    /// unsigned inputs (`USE_SU_BIAS`); the generic kernels do not use it.
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

impl LinearLayer<'_> {
    /// A layer without weights, for the models' `EMPTY`.
    pub(crate) const EMPTY: Self = Self {
        bias: None,
        subias: None,
        weights: None,
        float_weights: None,
        weights_idx: None,
        diag: None,
        scale: None,
        nb_inputs: 0,
        nb_outputs: 0,
    };
}

impl Conv2dLayer<'_> {
    /// A layer without weights, for the models' `EMPTY`.
    pub(crate) const EMPTY: Self = Self {
        bias: None,
        float_weights: None,
        in_channels: 0,
        out_channels: 0,
        ktime: 0,
        kheight: 0,
    };
}

/// Runs `f` on `n` zeroed values on the stack, zeroing only the smallest
/// of 64, 256, 1024, 2048 and 8192 values that holds them: the scratch
/// buffers of the layers, which libopus sizes for the largest layer.
#[inline(always)]
pub(crate) fn with_zeroed<T: Copy + Default, R>(n: usize, f: impl FnOnce(&mut [T]) -> R) -> R {
    let z = T::default();
    match n {
        0..=64 => f(&mut [z; 64][..n]),
        65..=256 => f(&mut [z; 256][..n]),
        257..=1024 => f(&mut [z; 1024][..n]),
        1025..=2048 => f(&mut [z; 2048][..n]),
        _ => f(&mut [z; 8192][..n]),
    }
}

/// `compute_linear` at `arch` (libopus's `compute_linear_c` or the x86
/// level's).
pub fn compute_linear(linear: &LinearLayer<'_>, out: &mut [f32], input: &[f32], arch: Arch) {
    let m = linear.nb_inputs;
    let n = linear.nb_outputs;
    let mut bias = linear.bias;
    if let Some(fw) = linear.float_weights {
        match linear.weights_idx {
            Some(idx) => sparse_sgemv8x4(out, fw, idx, n, input, arch),
            None => sgemv(out, fw, n, m, n, input, arch),
        }
    } else if let Some(w) = linear.weights {
        let scale = linear.scale.expect("int8 layer without scale");
        if cgemv8x4_at(out, w, linear.weights_idx, scale, n, m, input, arch) {
            bias = linear.subias;
        }
    } else {
        out[..n].fill(0.0);
    }
    if let Some(bias) = bias {
        for (i, o) in out[..n].iter_mut().enumerate() {
            *o += bias.at(i);
        }
    }
    if let Some(diag) = linear.diag {
        debug_assert_eq!(3 * m, n);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(avx2) = arch.avx2() {
            return super::vec_x86::diag(avx2, out, diag, input, m);
        }
        diag_by(out, diag, input, m, |d, x, y| y + d * x);
    }
}

/// The diagonal term of a GRU's recurrent weights (`diag`, three per
/// input), with `madd(d, x, y)` adding a product to a sum.
#[inline(always)]
pub(crate) fn diag_by(
    out: &mut [f32],
    diag: F32s<'_>,
    input: &[f32],
    m: usize,
    madd: impl Fn(f32, f32, f32) -> f32,
) {
    for i in 0..m {
        out[i] = madd(diag.at(i), input[i], out[i]);
        out[i + m] = madd(diag.at(i + m), input[i], out[i + m]);
        out[i + 2 * m] = madd(diag.at(i + 2 * m), input[i], out[i + 2 * m]);
    }
}

/// `relu`: `x < 0 ? 0 : x`.
#[inline(always)]
fn relu(x: f32) -> f32 {
    if x < 0.0 { 0.0 } else { x }
}

/// `compute_activation` on `x` in place: libopus's `compute_activation_c`
/// at the scalar level, `compute_activation_sse2`/`_sse4_1`/`_avx2` at the
/// x86 levels, which differ in vec.h's element-wise functions.
pub fn compute_activation_inplace(x: &mut [f32], activation: Activation, arch: Arch) {
    match activation {
        Activation::Sigmoid => vec_apply(x, Elem::Sigmoid, arch),
        Activation::Tanh => vec_apply(x, Elem::Tanh, arch),
        Activation::Swish => vec_apply(x, Elem::Swish, arch),
        Activation::Exp => vec_apply(x, Elem::Exp, arch),
        Activation::Relu => {
            for v in x.iter_mut() {
                *v = relu(*v);
            }
        }
        Activation::Softmax => {
            vec_apply(x, Elem::Exp, arch);
            let sum = x.iter().fold(0.0f32, |s, &v| s + v);
            let sum = (1.0f64 / (sum as f64 + 1e-30)) as f32;
            for v in x.iter_mut() {
                *v *= sum;
            }
        }
        Activation::Linear => {}
    }
}

/// `compute_activation`: `output[..n] = activation(input[..n])`.
pub fn compute_activation(
    output: &mut [f32],
    input: &[f32],
    n: usize,
    activation: Activation,
    arch: Arch,
) {
    output[..n].copy_from_slice(&input[..n]);
    compute_activation_inplace(&mut output[..n], activation, arch);
}

/// `compute_generic_dense`.
pub fn compute_generic_dense(
    layer: &LinearLayer<'_>,
    output: &mut [f32],
    input: &[f32],
    activation: Activation,
    arch: Arch,
) {
    compute_linear(layer, output, input, arch);
    compute_activation_inplace(&mut output[..layer.nb_outputs], activation, arch);
}

/// `compute_generic_gru`: one step of a GRU with state `state`
/// (`recurrent_weights.nb_inputs` values).
pub fn compute_generic_gru(
    input_weights: &LinearLayer<'_>,
    recurrent_weights: &LinearLayer<'_>,
    state: &mut [f32],
    input: &[f32],
    arch: Arch,
) {
    let n = recurrent_weights.nb_inputs;
    assert_eq!(3 * n, recurrent_weights.nb_outputs);
    assert_eq!(input_weights.nb_outputs, recurrent_weights.nb_outputs);
    assert!(n <= MAX_RNN_NEURONS_ALL);
    with_zeroed(6 * n, |buf: &mut [f32]| {
        let (zrh, recur) = buf.split_at_mut(3 * n);
        compute_linear(input_weights, zrh, input, arch);
        compute_linear(recurrent_weights, recur, state, arch);
        for i in 0..2 * n {
            zrh[i] += recur[i];
        }
        compute_activation_inplace(&mut zrh[..2 * n], Activation::Sigmoid, arch);
        let (zr, h) = zrh.split_at_mut(2 * n);
        let (z, r) = zr.split_at(n);
        for i in 0..n {
            h[i] += recur[2 * n + i] * r[i];
        }
        compute_activation_inplace(&mut h[..n], Activation::Tanh, arch);
        for i in 0..n {
            h[i] = z[i] * state[i] + (1.0 - z[i]) * h[i];
        }
        state[..n].copy_from_slice(&h[..n]);
    })
}

/// `compute_glu`: `output = input * sigmoid(W input + b)`.
pub fn compute_glu(layer: &LinearLayer<'_>, output: &mut [f32], input: &[f32], arch: Arch) {
    assert_eq!(layer.nb_inputs, layer.nb_outputs);
    let n = layer.nb_outputs;
    with_zeroed(n, |act2: &mut [f32]| {
        compute_linear(layer, act2, input, arch);
        compute_activation_inplace(act2, Activation::Sigmoid, arch);
        for i in 0..n {
            output[i] = input[i] * act2[i];
        }
    })
}

/// `compute_glu` with the output in place of the input.
pub fn compute_glu_inplace(layer: &LinearLayer<'_>, x: &mut [f32], arch: Arch) {
    assert_eq!(layer.nb_inputs, layer.nb_outputs);
    let n = layer.nb_outputs;
    with_zeroed(n, |act2: &mut [f32]| {
        compute_linear(layer, act2, x, arch);
        compute_activation_inplace(act2, Activation::Sigmoid, arch);
        for i in 0..n {
            x[i] *= act2[i];
        }
    })
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
    arch: Arch,
) {
    let ni = layer.nb_inputs;
    assert!(ni <= MAX_CONV_INPUTS_ALL);
    let hist = ni - input_size;
    with_zeroed(ni, |tmp: &mut [f32]| {
        tmp[..hist].copy_from_slice(&mem[..hist]);
        tmp[hist..ni].copy_from_slice(&input[..input_size]);
        compute_linear(layer, output, tmp, arch);
        compute_activation_inplace(&mut output[..layer.nb_outputs], activation, arch);
        mem[..hist].copy_from_slice(&tmp[input_size..input_size + hist]);
    })
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
    arch: Arch,
) {
    let ni = layer.nb_inputs;
    assert!(ni <= MAX_CONV_INPUTS_ALL);
    let ksize = ni / input_size;
    let hist = ni - input_size;
    with_zeroed(ni, |tmp: &mut [f32]| {
        if dilation == 1 {
            tmp[..hist].copy_from_slice(&mem[..hist]);
        } else {
            for i in 0..ksize - 1 {
                let src = i * input_size * dilation;
                tmp[i * input_size..(i + 1) * input_size]
                    .copy_from_slice(&mem[src..src + input_size]);
            }
        }
        tmp[hist..ni].copy_from_slice(&input[..input_size]);
        compute_linear(layer, output, tmp, arch);
        compute_activation_inplace(&mut output[..layer.nb_outputs], activation, arch);
        if dilation == 1 {
            mem[..hist].copy_from_slice(&tmp[input_size..input_size + hist]);
        } else {
            let total = input_size * dilation * (ksize - 1);
            mem.copy_within(input_size..total, 0);
            mem[total - input_size..total].copy_from_slice(&input[..input_size]);
        }
    })
}

/// `conv2d_float`: each product is added to the output on its own, with
/// `madd(w, x, y)`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
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
    madd: impl Fn(f32, f32, f32) -> f32,
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
                        o[j] = madd(w, x[j], o[j]);
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
/// `conv2d_3x3_float`: the nine products of an input channel are summed
/// first, then added to the output. `madd(w, x, y)` adds the first
/// product to the second and each later product to the sum.
#[inline(always)]
fn conv2d_3x3_float(
    out: &mut [f32],
    weights: F32s<'_>,
    in_channels: usize,
    out_channels: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
    madd: impl Fn(f32, f32, f32) -> f32,
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
                let mut sum = madd(w[0], x0[j], w[1] * x0[j + 1]);
                sum = madd(w[2], x0[j + 2], sum);
                sum = madd(w[3], x1[j], sum);
                sum = madd(w[4], x1[j + 1], sum);
                sum = madd(w[5], x1[j + 2], sum);
                sum = madd(w[6], x2[j], sum);
                sum = madd(w[7], x2[j + 1], sum);
                sum = madd(w[8], x2[j + 2], sum);
                o[j] += sum;
            }
        }
    }
}

/// The convolution of `compute_conv2d` on `input` (memory and new input),
/// with `madd(w, x, y)` adding a product to a sum.
#[inline(always)]
pub(crate) fn conv2d_by(
    conv: &Conv2dLayer<'_>,
    out: &mut [f32],
    input: &[f32],
    height: usize,
    hstride: usize,
    madd: impl Fn(f32, f32, f32) -> f32,
) {
    let w = conv.float_weights.expect("conv2d layer without weights");
    let (inc, outc) = (conv.in_channels, conv.out_channels);
    if conv.kheight == 3 && conv.ktime == 3 {
        conv2d_3x3_float(out, w, inc, outc, input, height, hstride, madd);
    } else {
        let (kt, kh) = (conv.ktime, conv.kheight);
        conv2d_float(out, w, inc, outc, kt, kh, input, height, hstride, madd);
    }
}

/// `compute_conv2d` at `arch`: one time step of a 2D convolution whose memory holds
/// the previous `ktime - 1` inputs.
pub fn compute_conv2d(
    conv: &Conv2dLayer<'_>,
    out: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    height: usize,
    hstride: usize,
    activation: Activation,
    arch: Arch,
) {
    let time_stride = conv.in_channels * (height + conv.kheight - 1);
    assert!(conv.ktime * time_stride <= MAX_CONV2D_INPUTS);
    let hist = (conv.ktime - 1) * time_stride;
    with_zeroed(hist + time_stride, |in_buf: &mut [f32]| {
        in_buf[..hist].copy_from_slice(&mem[..hist]);
        in_buf[hist..].copy_from_slice(&input[..time_stride]);
        mem[..hist].copy_from_slice(&in_buf[time_stride..]);
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(avx2) = arch.avx2() {
            return super::vec_x86::conv2d(avx2, conv, out, in_buf, height, hstride);
        }
        conv2d_by(conv, out, in_buf, height, hstride, |w, x, y| y + w * x);
    });
    if let Some(bias) = conv.bias {
        for i in 0..conv.out_channels {
            for j in 0..height {
                out[i * hstride + j] += bias.at(i);
            }
        }
    }
    for i in 0..conv.out_channels {
        compute_activation_inplace(
            &mut out[i * hstride..i * hstride + height],
            activation,
            arch,
        );
    }
}
