/* Wrappers for libopus header-only macros/inlines, so tests can call them. */
#include "SigProc_FIX.h"
#include "celt.h"
#include "modes.h"
#include "entenc.h"
#include "entdec.h"

#define W2(name, expr) opus_int32 ref_##name(opus_int32 a, opus_int32 b) { return (expr); }
#define W3(name, expr) opus_int32 ref_##name(opus_int32 a, opus_int32 b, opus_int32 c) { return (expr); }
W2(silk_lshift_sat32, silk_LSHIFT_SAT32(a, b))
W2(silk_add_sat32, silk_ADD_SAT32(a, b))
W2(silk_sub_sat32, silk_SUB_SAT32(a, b))
W2(silk_add_pos_sat32, silk_ADD_POS_SAT32(a, b))
W2(silk_rshift_round, silk_RSHIFT_ROUND(a, b))
W2(silk_smulwb, silk_SMULWB(a, b))
W2(silk_smulwt, silk_SMULWT(a, b))
W2(silk_smulww, silk_SMULWW(a, b))
W2(silk_smulbb, silk_SMULBB(a, b))
W2(silk_smmul, silk_SMMUL(a, b))
W2(silk_div32_16, silk_DIV32_16(a, b))
W3(silk_smlawb, silk_SMLAWB(a, b, c))
W3(silk_smlawt, silk_SMLAWT(a, b, c))
W3(silk_smlaww, silk_SMLAWW(a, b, c))
W3(silk_smlabb, silk_SMLABB(a, b, c))
W3(silk_limit_32, silk_LIMIT_32(a, b, c))
W3(silk_add_lshift32, silk_ADD_LSHIFT32(a, b, c))
W3(silk_add_rshift32, silk_ADD_RSHIFT32(a, b, c))
W3(silk_sub_lshift32, silk_SUB_LSHIFT32(a, b, c))
W3(silk_div32_varq, silk_DIV32_varQ(a, b, c))
opus_int32 ref_silk_inverse32_varq(opus_int32 b, opus_int32 q) { return silk_INVERSE32_varQ(b, q); }
opus_int32 ref_silk_sat16(opus_int32 a) { return silk_SAT16(a); }
opus_int32 ref_silk_clz32(opus_int32 a) { return silk_CLZ32(a); }
opus_int32 ref_silk_sqrt_approx(opus_int32 a) { return silk_SQRT_APPROX(a); }
opus_int32 ref_silk_rand(opus_int32 a) { return silk_RAND(a); }

const CELTMode *ref_celt_mode48(void) { return opus_custom_mode_create(48000, 960, NULL); }

#include "laplace.h"
/* ec_laplace_encode of `*value` into `cap` bytes, twice: the buffer after
   ec_enc_done and the value the first call codes. */
void ref_laplace_encode(int *value, unsigned fs, int decay, unsigned char *buf, int cap) {
  ec_enc enc;
  int again;
  ec_enc_init(&enc, buf, cap);
  ec_laplace_encode(&enc, value, fs, decay);
  again = *value;
  ec_laplace_encode(&enc, &again, fs, decay);
  ec_enc_done(&enc);
}
/* ec_laplace_decode of the first value in `buf`. */
int ref_laplace_decode(unsigned fs, int decay, unsigned char *buf, int len) {
  ec_dec dec;
  ec_dec_init(&dec, buf, len);
  return ec_laplace_decode(&dec, fs, decay);
}

#ifndef FIXED_POINT
/* The static mode's MDCT and FFT, for kernel comparisons. */
#include "mdct.h"
#include "kiss_fft.h"
void ref_mdct_forward(float *in, float *out, int shift, int stride) {
  const CELTMode *m = ref_celt_mode48();
  clt_mdct_forward_c(&m->mdct, in, out, m->window, m->overlap, shift, stride, 0);
}
void ref_mdct_backward(float *in, float *out, int shift, int stride) {
  const CELTMode *m = ref_celt_mode48();
  clt_mdct_backward_c(&m->mdct, in, out, m->window, m->overlap, shift, stride, 0);
}
void ref_fft_impl(float *x, int shift) {
  opus_fft_impl(ref_celt_mode48()->mdct.kfft[shift], (kiss_fft_cpx *)x);
}
const float *ref_mdct_trig(void) { return ref_celt_mode48()->mdct.trig; }
const float *ref_window(void) { return ref_celt_mode48()->window; }

/* A bare CELT encoder (celt_encoder_init as opus_encoder.c sets it up,
   without signalling), for comparisons with opus-rs's CeltEncoder. */
#include <stdlib.h>
void *ref_celt_encoder_create(int fs, int channels) {
  CELTEncoder *st = malloc(celt_encoder_get_size(channels));
  if (!st || celt_encoder_init(st, fs, channels, 0) != OPUS_OK) {
    free(st);
    return NULL;
  }
  opus_custom_encoder_ctl(st, CELT_SET_SIGNALLING_REQUEST, 0);
  return st;
}
void ref_celt_encoder_destroy(void *st) { free(st); }
int ref_celt_encoder_ctl(void *st, int request, int value) {
  return opus_custom_encoder_ctl((CELTEncoder *)st, request, value);
}
int ref_celt_encode(void *st, const float *pcm, int frame_size, unsigned char *out, int bytes) {
  return celt_encode_with_ec((CELTEncoder *)st, pcm, frame_size, out, bytes, NULL);
}
/* celt_encode_with_ec into a range coder that already holds `prefix`
   equiprobable bits, as a Hybrid frame's CELT layer follows SILK. */
int ref_celt_encode_after(void *st, const float *pcm, int frame_size, unsigned char *out,
                          int bytes, int prefix) {
  ec_enc enc;
  ec_enc_init(&enc, out, bytes);
  for (int i = 0; i < prefix; i++) ec_enc_bit_logp(&enc, i & 1, 1);
  return celt_encode_with_ec((CELTEncoder *)st, pcm, frame_size, NULL, bytes, &enc);
}
/* celt_inner_prod and dual_inner_prod, macros that select the kernel of
   the level `arch`. */
#include "pitch.h"
float ref_celt_inner_prod(const float *x, const float *y, int n, int arch) {
  (void)arch;
  return celt_inner_prod(x, y, n, arch);
}
void ref_dual_inner_prod(const float *x, const float *y1, const float *y2, int n,
                         float *xy1, float *xy2, int arch) {
  (void)arch;
  dual_inner_prod(x, y1, y2, n, xy1, xy2, arch);
}
void ref_celt_pitch_xcorr(const float *x, const float *y, float *xcorr, int len, int max_pitch,
                          int arch) {
  (void)arch;
  celt_pitch_xcorr(x, y, xcorr, len, max_pitch, arch);
}
#endif
