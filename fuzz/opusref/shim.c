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
#endif
