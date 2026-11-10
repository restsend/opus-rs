//! The small network of the signal analysis (libopus `src/mlp.c`): a dense
//! layer, a GRU and a dense output layer estimating the probability of music
//! and of activity (speech or music rather than noise) from 25 features.

use crate::mlp_data::*;

/// libopus `WEIGHTS_SCALE`: the weights are Q7.
const WEIGHTS_SCALE: f32 = 1.0 / 128.0;

/// libopus `MAX_NEURONS`.
pub(crate) const MAX_NEURONS: usize = 32;

/// libopus `tansig_approx`: a rational approximation of tanh.
#[inline]
fn tansig_approx(x: f32) -> f32 {
    // libopus's 952.52801514f, 96.39235687f, 0.60863042f, 952.72399902f,
    // 413.36801147f and 11.88600922f.
    const N0: f32 = 952.528;
    const N1: f32 = 96.39236;
    const N2: f32 = 0.6086304;
    const D0: f32 = 952.724;
    const D1: f32 = 413.368;
    const D2: f32 = 11.886009;
    let x2 = x * x;
    let num = (N2 * x2 + N1) * x2 + N0;
    let den = (D2 * x2 + D1) * x2 + D0;
    let num = num * x / den;
    // MAX32(-1, MIN32(1, num))
    let num = if 1.0 < num { 1.0 } else { num };
    if -1.0 > num { -1.0 } else { num }
}

#[inline]
fn sigmoid_approx(x: f32) -> f32 {
    0.5 + 0.5 * tansig_approx(0.5 * x)
}

/// libopus `gemm_accum`: `out[i] += sum_j weights[j * col_stride + i] * x[j]`.
fn gemm_accum(
    out: &mut [f32],
    weights: &[i8],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    for i in 0..rows {
        for j in 0..cols {
            out[i] += f32::from(weights[j * col_stride + i]) * x[j];
        }
    }
}

struct DenseLayer {
    bias: &'static [i8],
    input_weights: &'static [i8],
    nb_inputs: usize,
    nb_neurons: usize,
    sigmoid: bool,
}

struct GruLayer {
    bias: &'static [i8],
    input_weights: &'static [i8],
    recurrent_weights: &'static [i8],
    nb_inputs: usize,
    nb_neurons: usize,
}

const LAYER0: DenseLayer = DenseLayer {
    bias: &LAYER0_BIAS,
    input_weights: &LAYER0_WEIGHTS,
    nb_inputs: 25,
    nb_neurons: 32,
    sigmoid: false,
};

const LAYER1: GruLayer = GruLayer {
    bias: &LAYER1_BIAS,
    input_weights: &LAYER1_WEIGHTS,
    recurrent_weights: &LAYER1_RECUR_WEIGHTS,
    nb_inputs: 32,
    nb_neurons: 24,
};

const LAYER2: DenseLayer = DenseLayer {
    bias: &LAYER2_BIAS,
    input_weights: &LAYER2_WEIGHTS,
    nb_inputs: 24,
    nb_neurons: 2,
    sigmoid: true,
};

/// libopus `analysis_compute_dense`.
fn compute_dense(layer: &DenseLayer, output: &mut [f32], input: &[f32]) {
    let (m, n) = (layer.nb_inputs, layer.nb_neurons);
    for i in 0..n {
        output[i] = f32::from(layer.bias[i]);
    }
    gemm_accum(output, layer.input_weights, n, m, n, input);
    for v in &mut output[..n] {
        *v *= WEIGHTS_SCALE;
    }
    for v in &mut output[..n] {
        *v = if layer.sigmoid {
            sigmoid_approx(*v)
        } else {
            tansig_approx(*v)
        };
    }
}

/// libopus `analysis_compute_gru`.
fn compute_gru(gru: &GruLayer, state: &mut [f32], input: &[f32]) {
    let (m, n) = (gru.nb_inputs, gru.nb_neurons);
    let stride = 3 * n;
    let mut tmp = [0.0f32; MAX_NEURONS];
    let mut z = [0.0f32; MAX_NEURONS];
    let mut r = [0.0f32; MAX_NEURONS];
    let mut h = [0.0f32; MAX_NEURONS];
    // Update gate.
    for i in 0..n {
        z[i] = f32::from(gru.bias[i]);
    }
    gemm_accum(&mut z, gru.input_weights, n, m, stride, input);
    gemm_accum(&mut z, gru.recurrent_weights, n, n, stride, state);
    for v in &mut z[..n] {
        *v = sigmoid_approx(WEIGHTS_SCALE * *v);
    }
    // Reset gate.
    for i in 0..n {
        r[i] = f32::from(gru.bias[n + i]);
    }
    gemm_accum(&mut r, &gru.input_weights[n..], n, m, stride, input);
    gemm_accum(&mut r, &gru.recurrent_weights[n..], n, n, stride, state);
    for v in &mut r[..n] {
        *v = sigmoid_approx(WEIGHTS_SCALE * *v);
    }
    // Output.
    for i in 0..n {
        h[i] = f32::from(gru.bias[2 * n + i]);
    }
    for i in 0..n {
        tmp[i] = state[i] * r[i];
    }
    gemm_accum(&mut h, &gru.input_weights[2 * n..], n, m, stride, input);
    gemm_accum(&mut h, &gru.recurrent_weights[2 * n..], n, n, stride, &tmp);
    for i in 0..n {
        h[i] = z[i] * state[i] + (1.0 - z[i]) * tansig_approx(WEIGHTS_SCALE * h[i]);
    }
    state[..n].copy_from_slice(&h[..n]);
}

/// Run the network on one frame's `features`, carrying the GRU state in
/// `rnn_state`: returns the music probability and the activity probability.
pub(crate) fn compute_probabilities(
    features: &[f32; 25],
    rnn_state: &mut [f32; MAX_NEURONS],
) -> (f32, f32) {
    let mut layer_out = [0.0f32; MAX_NEURONS];
    let mut frame_probs = [0.0f32; 2];
    compute_dense(&LAYER0, &mut layer_out, features);
    compute_gru(&LAYER1, rnn_state, &layer_out);
    compute_dense(&LAYER2, &mut frame_probs, rnn_state);
    (frame_probs[0], frame_probs[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tansig_is_bounded_and_odd() {
        for x in [-20.0f32, -3.0, -0.5, 0.0, 0.25, 2.0, 50.0] {
            let y = tansig_approx(x);
            assert!((-1.0..=1.0).contains(&y));
            assert_eq!(y, -tansig_approx(-x));
        }
        assert_eq!(tansig_approx(0.0), 0.0);
    }
}
