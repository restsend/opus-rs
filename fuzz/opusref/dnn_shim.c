/* Helpers for the neural network tests (feature `dnn`). */
#include <string.h>
#include "nnet.h"

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

/* The generic kernels, for the differential tests of the layer functions. */
void ref_compute_linear(const LinearLayer *l, float *out, const float *in) { compute_linear_c(l, out, in); }
void ref_compute_activation(float *out, const float *in, int n, int act) { compute_activation_c(out, in, n, act); }
void ref_compute_generic_dense(const LinearLayer *l, float *out, const float *in, int act) { compute_generic_dense(l, out, in, act, 0); }
void ref_compute_generic_gru(const LinearLayer *i, const LinearLayer *r, float *state, const float *in) { compute_generic_gru(i, r, state, in, 0); }
void ref_compute_glu(const LinearLayer *l, float *out, const float *in) { compute_glu(l, out, in, 0); }
void ref_compute_generic_conv1d(const LinearLayer *l, float *out, float *mem, const float *in, int n, int act) { compute_generic_conv1d(l, out, mem, in, n, act, 0); }
void ref_compute_generic_conv1d_dilation(const LinearLayer *l, float *out, float *mem, const float *in, int n, int dil, int act) { compute_generic_conv1d_dilation(l, out, mem, in, n, dil, act, 0); }
void ref_compute_conv2d(const Conv2dLayer *c, float *out, float *mem, const float *in, int h, int hs, int act) { compute_conv2d_c(c, out, mem, in, h, hs, act); }
