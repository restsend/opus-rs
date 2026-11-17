/* libopus dnn/osce.c with exported wrappers of its static functions, for
   the OSCE tests (tests/dnn_osce.rs). The angle brackets find osce.c on
   the include path (opus-1.6.1/dnn). */
#include <osce.c>
#include <stdlib.h>

void ref_lace_numbits_embedding(float *emb, float numbits, float min_val, float max_val) {
  compute_lace_numbits_embedding(emb, numbits, LACE_NUMBITS_EMBEDDING_DIM, min_val, max_val, 1);
}

LACEState *ref_lace_state_new(void) {
  LACEState *st = calloc(1, sizeof(*st));
  reset_lace_state(st);
  return st;
}
NoLACEState *ref_nolace_state_new(void) {
  NoLACEState *st = calloc(1, sizeof(*st));
  reset_nolace_state(st);
  return st;
}
BBWENetState *ref_bbwenet_state_new(void) {
  BBWENetState *st = calloc(1, sizeof(*st));
  reset_bbwenet_state(st);
  return st;
}

void ref_lace_process_20ms_frame(OSCEModel *m, LACEState *st, float *out, const float *in, const float *features,
                                 const float *numbits, const int *periods) {
  lace_process_20ms_frame(&m->lace, st, out, in, features, numbits, periods, 0);
}
void ref_nolace_process_20ms_frame(OSCEModel *m, NoLACEState *st, float *out, const float *in, const float *features,
                                   const float *numbits, const int *periods) {
  nolace_process_20ms_frame(&m->nolace, st, out, in, features, numbits, periods, 0);
}
void ref_bbwenet_process_frames(OSCEModel *m, BBWENetState *st, float *out, const float *in, const float *features,
                                int num_frames) {
  bbwenet_process_frames(&m->bbwenet, st, out, in, features, num_frames, 0);
}

resamp_state *ref_resamp_state_new(void) { return calloc(1, sizeof(resamp_state)); }
void ref_upsamp_2x(resamp_state *st, float *out, const float *in, int n) { upsamp_2x(st, out, in, n); }
void ref_interpol_3_2(resamp_state *st, float *out, const float *in, int n) { interpol_3_2(st, out, in, n); }
void ref_apply_valin_activation(float *x, int n) { apply_valin_activation(x, n); }
