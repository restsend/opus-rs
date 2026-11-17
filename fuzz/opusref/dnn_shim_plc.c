/* Deep PLC (FARGAN, LPCNet PLC) for the tests of feature `dnn`. */
#include <stdlib.h>
#include "lpcnet.h"
#include "lpcnet_private.h"
#include "fargan.h"

FARGANState *ref_fargan_create(void) {
  FARGANState *st = malloc(sizeof(*st));
  fargan_init(st);
  return st;
}
void ref_fargan_destroy(FARGANState *st) { free(st); }
void ref_fargan_cont(FARGANState *st, const float *pcm0, const float *features0) { fargan_cont(st, pcm0, features0); }
void ref_fargan_synthesize(FARGANState *st, float *pcm, const float *features) { fargan_synthesize(st, pcm, features); }
void ref_fargan_synthesize_int(FARGANState *st, opus_int16 *pcm, const float *features) { fargan_synthesize_int(st, pcm, features); }

LPCNetPLCState *ref_lpcnet_plc_create(void) {
  LPCNetPLCState *st = malloc(sizeof(*st));
  lpcnet_plc_init(st);
  return st;
}
void ref_lpcnet_plc_destroy(LPCNetPLCState *st) { free(st); }
void ref_lpcnet_plc_reset(LPCNetPLCState *st) { lpcnet_plc_reset(st); }
void ref_lpcnet_plc_update(LPCNetPLCState *st, opus_int16 *pcm) { lpcnet_plc_update(st, pcm); }
void ref_lpcnet_plc_conceal(LPCNetPLCState *st, opus_int16 *pcm) { lpcnet_plc_conceal(st, pcm); }
void ref_lpcnet_plc_fec_add(LPCNetPLCState *st, const float *features) { lpcnet_plc_fec_add(st, features); }
void ref_lpcnet_plc_fec_clear(LPCNetPLCState *st) { lpcnet_plc_fec_clear(st); }
