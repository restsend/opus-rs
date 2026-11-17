/* Helpers for the neural network tests (feature `dnn`). */
#include <string.h>
#include "nnet.h"

/* The generic kernels, for the differential tests of the layer functions. */
void ref_compute_linear(const LinearLayer *l, float *out, const float *in) { compute_linear_c(l, out, in); }
void ref_compute_activation(float *out, const float *in, int n, int act) { compute_activation_c(out, in, n, act); }
void ref_compute_generic_dense(const LinearLayer *l, float *out, const float *in, int act) { compute_generic_dense(l, out, in, act, 0); }
void ref_compute_generic_gru(const LinearLayer *i, const LinearLayer *r, float *state, const float *in) { compute_generic_gru(i, r, state, in, 0); }
void ref_compute_glu(const LinearLayer *l, float *out, const float *in) { compute_glu(l, out, in, 0); }
void ref_compute_generic_conv1d(const LinearLayer *l, float *out, float *mem, const float *in, int n, int act) { compute_generic_conv1d(l, out, mem, in, n, act, 0); }
void ref_compute_generic_conv1d_dilation(const LinearLayer *l, float *out, float *mem, const float *in, int n, int dil, int act) { compute_generic_conv1d_dilation(l, out, mem, in, n, dil, act, 0); }
void ref_compute_conv2d(const Conv2dLayer *c, float *out, float *mem, const float *in, int h, int hs, int act) { compute_conv2d_c(c, out, mem, in, h, hs, act); }
