/* Helpers for the OSCE tests (tests/dnn_osce.rs): the model, the SILK
   decoder data osce_enhance_frame reads, and the nndsp kernels. */
#include <stdlib.h>
#include <string.h>
#include "osce.h"
#include "osce_features.h"
#include "nndsp.h"
#include "main.h"

/* The built-in models, as silk_InitDecoder loads them. */
OSCEModel *ref_osce_model_new(void) {
  OSCEModel *m = calloc(1, sizeof(*m));
  m->loaded = osce_load_models(m, NULL, 0) == 0;
  return m;
}
void ref_osce_model_set_loaded(OSCEModel *m, int loaded) { m->loaded = loaded; }
void ref_osce_free(void *p) { free(p); }

/* A layer of the model by its name. */
const LinearLayer *ref_osce_layer(const OSCEModel *m, const char *name) {
#define L(mod, f) if (!strcmp(name, #f)) return &m->mod.layers.f;
  L(lace, lace_pitch_embedding) L(lace, lace_fnet_conv1) L(lace, lace_fnet_conv2) L(lace, lace_fnet_tconv)
  L(lace, lace_fnet_gru_input) L(lace, lace_fnet_gru_recurrent) L(lace, lace_cf1_kernel) L(lace, lace_cf1_gain)
  L(lace, lace_cf1_global_gain) L(lace, lace_cf2_kernel) L(lace, lace_cf2_gain) L(lace, lace_cf2_global_gain)
  L(lace, lace_af1_kernel) L(lace, lace_af1_gain)
  L(nolace, nolace_pitch_embedding) L(nolace, nolace_fnet_conv1) L(nolace, nolace_fnet_conv2)
  L(nolace, nolace_fnet_tconv) L(nolace, nolace_fnet_gru_input) L(nolace, nolace_fnet_gru_recurrent)
  L(nolace, nolace_cf1_kernel) L(nolace, nolace_cf1_gain) L(nolace, nolace_cf1_global_gain)
  L(nolace, nolace_cf2_kernel) L(nolace, nolace_cf2_gain) L(nolace, nolace_cf2_global_gain)
  L(nolace, nolace_af1_kernel) L(nolace, nolace_af1_gain) L(nolace, nolace_tdshape1_alpha1_f)
  L(nolace, nolace_tdshape1_alpha1_t) L(nolace, nolace_tdshape1_alpha2) L(nolace, nolace_tdshape2_alpha1_f)
  L(nolace, nolace_tdshape2_alpha1_t) L(nolace, nolace_tdshape2_alpha2) L(nolace, nolace_tdshape3_alpha1_f)
  L(nolace, nolace_tdshape3_alpha1_t) L(nolace, nolace_tdshape3_alpha2) L(nolace, nolace_af2_kernel)
  L(nolace, nolace_af2_gain) L(nolace, nolace_af3_kernel) L(nolace, nolace_af3_gain) L(nolace, nolace_af4_kernel)
  L(nolace, nolace_af4_gain) L(nolace, nolace_post_cf1) L(nolace, nolace_post_cf2) L(nolace, nolace_post_af1)
  L(nolace, nolace_post_af2) L(nolace, nolace_post_af3)
  L(bbwenet, bbwenet_fnet_conv1) L(bbwenet, bbwenet_fnet_conv2) L(bbwenet, bbwenet_fnet_gru_input)
  L(bbwenet, bbwenet_fnet_gru_recurrent) L(bbwenet, bbwenet_fnet_tconv) L(bbwenet, bbwenet_tdshape1_alpha1_f)
  L(bbwenet, bbwenet_tdshape1_alpha1_t) L(bbwenet, bbwenet_tdshape1_alpha2) L(bbwenet, bbwenet_tdshape2_alpha1_f)
  L(bbwenet, bbwenet_tdshape2_alpha1_t) L(bbwenet, bbwenet_tdshape2_alpha2) L(bbwenet, bbwenet_af1_kernel)
  L(bbwenet, bbwenet_af1_gain) L(bbwenet, bbwenet_af2_kernel) L(bbwenet, bbwenet_af2_gain)
  L(bbwenet, bbwenet_af3_kernel) L(bbwenet, bbwenet_af3_gain)
#undef L
  return NULL;
}

/* The overlap windows the models computed. */
const float *ref_osce_window(const OSCEModel *m, int which) {
  switch (which) {
    case 0: return m->lace.window;
    case 1: return m->nolace.window;
    case 2: return m->bbwenet.window16;
    case 3: return m->bbwenet.window32;
    default: return m->bbwenet.window48;
  }
}

/* nndsp. */
void ref_compute_overlap_window(float *w, int n) { compute_overlap_window(w, n); }
AdaConvState *ref_adaconv_state_new(void) {
  AdaConvState *st = malloc(sizeof(*st));
  init_adaconv_state(st);
  return st;
}
AdaCombState *ref_adacomb_state_new(void) {
  AdaCombState *st = malloc(sizeof(*st));
  init_adacomb_state(st);
  return st;
}
AdaShapeState *ref_adashape_state_new(void) {
  AdaShapeState *st = malloc(sizeof(*st));
  init_adashape_state(st);
  return st;
}
void ref_adaconv_process_frame(AdaConvState *st, float *out, const float *in, const float *features,
                               const LinearLayer *kernel, const LinearLayer *gain, int feature_dim, int frame_size,
                               int overlap_size, int in_channels, int out_channels, int kernel_size, int left_padding,
                               float gain_a, float gain_b, float shape_gain, float *window) {
  adaconv_process_frame(st, out, in, features, kernel, gain, feature_dim, frame_size, overlap_size, in_channels,
                        out_channels, kernel_size, left_padding, gain_a, gain_b, shape_gain, window, 0);
}
void ref_adacomb_process_frame(AdaCombState *st, float *out, const float *in, const float *features,
                               const LinearLayer *kernel, const LinearLayer *gain, const LinearLayer *global_gain,
                               int pitch_lag, int feature_dim, int frame_size, int overlap_size, int kernel_size,
                               int left_padding, float gain_a, float gain_b, float log_gain_limit, float *window) {
  adacomb_process_frame(st, out, in, features, kernel, gain, global_gain, pitch_lag, feature_dim, frame_size,
                        overlap_size, kernel_size, left_padding, gain_a, gain_b, log_gain_limit, window, 0);
}
void ref_adashape_process_frame(AdaShapeState *st, float *out, const float *in, const float *features,
                                const LinearLayer *alpha1f, const LinearLayer *alpha1t, const LinearLayer *alpha2,
                                int feature_dim, int frame_size, int avg_pool_k, int interpolate_k) {
  adashape_process_frame(st, out, in, features, alpha1f, alpha1t, alpha2, feature_dim, frame_size, avg_pool_k,
                         interpolate_k, 0);
}

/* A SILK decoder channel and the control data of its current frame. */
typedef struct {
  silk_decoder_state dec;
  silk_decoder_control ctrl;
} RefOsceDec;

RefOsceDec *ref_osce_dec_new(void) {
  RefOsceDec *d = calloc(1, sizeof(*d));
  silk_init_decoder(&d->dec);
  return d;
}
void ref_osce_dec_set(RefOsceDec *d, int fs_khz, int nb_subfr, int lpc_order, int signal_type, const int *pitch_l,
                      const opus_int32 *gains_q16, const opus_int16 *pred_coef_q12, const opus_int16 *ltp_coef_q14) {
  int i;
  d->dec.fs_kHz = fs_khz;
  d->dec.nb_subfr = nb_subfr;
  d->dec.LPC_order = lpc_order;
  d->dec.indices.signalType = signal_type;
  for (i = 0; i < MAX_NB_SUBFR; i++) {
    d->ctrl.pitchL[i] = pitch_l[i];
    d->ctrl.Gains_Q16[i] = gains_q16[i];
  }
  for (i = 0; i < MAX_LPC_ORDER; i++) {
    d->ctrl.PredCoef_Q12[0][i] = pred_coef_q12[i];
    d->ctrl.PredCoef_Q12[1][i] = pred_coef_q12[MAX_LPC_ORDER + i];
  }
  for (i = 0; i < LTP_ORDER * MAX_NB_SUBFR; i++) d->ctrl.LTPCoef_Q14[i] = ltp_coef_q14[i];
}
void ref_osce_dec_features(RefOsceDec *d, const opus_int16 *xq, int num_bits, float *features, float *numbits,
                           int *periods) {
  osce_calculate_features(&d->dec, &d->ctrl, features, numbits, periods, xq, num_bits);
}
void ref_osce_enhance_frame(OSCEModel *m, RefOsceDec *d, opus_int16 *xq, int num_bits) {
  osce_enhance_frame(m, &d->dec, &d->ctrl, xq, num_bits, 0);
}
void ref_osce_reset(RefOsceDec *d, int method) { osce_reset(&d->dec.osce, method); }
int ref_osce_method(const RefOsceDec *d) { return d->dec.osce.method; }
int ref_osce_reset_count(const RefOsceDec *d) { return d->dec.osce.features.reset; }
float ref_osce_numbits_smooth(const RefOsceDec *d) { return d->dec.osce.features.numbits_smooth; }

/* Bandwidth extension. */
silk_OSCE_BWE_struct *ref_osce_bwe_new(void) { return calloc(1, sizeof(silk_OSCE_BWE_struct)); }
void ref_osce_bwe_reset(silk_OSCE_BWE_struct *st) { osce_bwe_reset(st); }
void ref_osce_bwe(OSCEModel *m, silk_OSCE_BWE_struct *st, opus_int16 *xq48, opus_int16 *xq16, int len) {
  osce_bwe(m, st, xq48, xq16, len, 0);
}
void ref_osce_bwe_features(silk_OSCE_BWE_struct *st, float *features, const opus_int16 *xq16, int len) {
  osce_bwe_calculate_features(&st->features, features, xq16, len);
}
void ref_osce_cross_fade_10ms(float *x_enhanced, float *x_in, int len) { osce_cross_fade_10ms(x_enhanced, x_in, len); }
void ref_osce_bwe_cross_fade_10ms(opus_int16 *x_fadein, opus_int16 *x_fadeout, int len) {
  osce_bwe_cross_fade_10ms(x_fadein, x_fadeout, len);
}
