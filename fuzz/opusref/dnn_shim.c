/* Helpers for the neural network tests (feature `dnn`). */
#include <string.h>
#include "nnet.h"
#include "cpu_support.h"

/* The record writer of libopus's dnn/write_lpcnet_weights.c, into a buffer:
   writes `list` in the blob format and returns the byte count (or the count
   needed when `cap` is too small). */
int ref_write_weights(const WeightArray *list, unsigned char *out, int cap)
{
  int i = 0;
  int pos = 0;
  while (list[i].name != NULL) {
    WeightHead h;
    int block = (list[i].size + WEIGHT_BLOCK_SIZE - 1) / WEIGHT_BLOCK_SIZE * WEIGHT_BLOCK_SIZE;
    memcpy(h.head, "DNNw", 4);
    h.version = WEIGHT_BLOB_VERSION;
    h.type = list[i].type;
    h.size = list[i].size;
    h.block_size = block;
    memset(h.name, 0, sizeof(h.name));
    strncpy(h.name, list[i].name, sizeof(h.name));
    h.name[sizeof(h.name) - 1] = 0;
    if (pos + WEIGHT_BLOCK_SIZE + block <= cap) {
      memcpy(out + pos, &h, WEIGHT_BLOCK_SIZE);
      memcpy(out + pos + WEIGHT_BLOCK_SIZE, list[i].data, list[i].size);
      memset(out + pos + WEIGHT_BLOCK_SIZE + list[i].size, 0, block - list[i].size);
    }
    pos += WEIGHT_BLOCK_SIZE + block;
    i++;
  }
  return pos;
}

/* The layer functions at the reference's level (opus_select_arch(): 0 in the
   scalar build, OPUSREF_ARCH or the CPU's in the simd build), for the
   differential tests of the layer functions. */
void ref_compute_linear(const LinearLayer *l, float *out, const float *in) { compute_linear(l, out, in, opus_select_arch()); }
void ref_compute_activation(float *out, const float *in, int n, int act) { compute_activation(out, in, n, act, opus_select_arch()); }
void ref_compute_generic_dense(const LinearLayer *l, float *out, const float *in, int act) { compute_generic_dense(l, out, in, act, opus_select_arch()); }
void ref_compute_generic_gru(const LinearLayer *i, const LinearLayer *r, float *state, const float *in) { compute_generic_gru(i, r, state, in, opus_select_arch()); }
void ref_compute_glu(const LinearLayer *l, float *out, const float *in) { compute_glu(l, out, in, opus_select_arch()); }
void ref_compute_generic_conv1d(const LinearLayer *l, float *out, float *mem, const float *in, int n, int act) { compute_generic_conv1d(l, out, mem, in, n, act, opus_select_arch()); }
void ref_compute_generic_conv1d_dilation(const LinearLayer *l, float *out, float *mem, const float *in, int n, int dil, int act) { compute_generic_conv1d_dilation(l, out, mem, in, n, dil, act, opus_select_arch()); }
void ref_compute_conv2d(const Conv2dLayer *c, float *out, float *mem, const float *in, int h, int hs, int act) { compute_conv2d(c, out, mem, in, h, hs, act, opus_select_arch()); }

/* LPCNet feature analysis. */
#include <stdlib.h>
#include "lpcnet.h"
#include "lpcnet_private.h"
#include "freq.h"
#include "burg.h"
#include "pitchdnn.h"

LPCNetEncState *ref_lpcnet_encoder_create(void) { return lpcnet_encoder_create(); }
void ref_lpcnet_encoder_destroy(LPCNetEncState *st) { lpcnet_encoder_destroy(st); }
void ref_lpcnet_features_float(LPCNetEncState *st, const float *pcm, float *features) {
  lpcnet_compute_single_frame_features_float(st, pcm, features, opus_select_arch());
}
void ref_lpcnet_features(LPCNetEncState *st, const opus_int16 *pcm, float *features) {
  lpcnet_compute_single_frame_features(st, pcm, features, opus_select_arch());
}
void ref_burg_cepstral_analysis(float *ceps, const float *x) { burg_cepstral_analysis(ceps, x); }
float ref_lpc_from_cepstrum(float *lpc, const float *cep) { return lpc_from_cepstrum(lpc, cep); }
void ref_forward_transform(kiss_fft_cpx *out, const float *in) { forward_transform(out, in); }
float ref_silk_burg_analysis(float *A, const float *x, float min_inv_gain, int subfr_length, int nb_subfr, int D) {
  return silk_burg_analysis(A, x, min_inv_gain, subfr_length, nb_subfr, D);
}
PitchDNNState *ref_pitchdnn_create(void) {
  PitchDNNState *st = malloc(sizeof(*st));
  pitchdnn_init(st);
  return st;
}
void ref_pitchdnn_destroy(PitchDNNState *st) { free(st); }
float ref_compute_pitchdnn(PitchDNNState *st, const float *if_features, const float *xcorr_features) {
  return compute_pitchdnn(st, if_features, xcorr_features, opus_select_arch());
}
