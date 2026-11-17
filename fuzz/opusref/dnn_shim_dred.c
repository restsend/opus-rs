/* Helpers for the DRED tests (feature `dnn`): create and inspect the C
   states of dnn/dred_*.c without the Rust side knowing their layout. */
#include <stdlib.h>
#include <string.h>
#include "dred_encoder.h"
#include "dred_decoder.h"
#include "dred_coding.h"
#include "dred_rdovae_enc.h"
#include "dred_rdovae_dec.h"
#include "dred_rdovae_enc_data.h"
#include "dred_rdovae_dec_data.h"
#include "celt/entenc.h"
#include "celt/entdec.h"
#include "celt/laplace.h"

/* Non-static in dred_encoder.c, without a header declaration. */
void filter_df2t(const float *in, float *out, int len, float b0, const float *b, const float *a, int order, float *mem);

int ref_compute_quantizer(int q0, int dQ, int qmax, int i) { return compute_quantizer(q0, dQ, qmax, i); }

/* Codes `n` values with ec_laplace_encode_p0 into `buf` (`cap` bytes,
   all of them written by ec_enc_done); returns ec_tell before ec_enc_done,
   or -1 on a coder error. */
int ref_laplace_p0_encode(const int *vals, const opus_uint16 *p0, const opus_uint16 *decay, int n,
                          unsigned char *buf, int cap)
{
  ec_enc enc;
  int i;
  int tell;
  ec_enc_init(&enc, buf, cap);
  for (i = 0; i < n; i++) ec_laplace_encode_p0(&enc, vals[i], p0[i], decay[i]);
  tell = ec_tell(&enc);
  ec_enc_done(&enc);
  if (enc.error) return -1;
  return tell;
}

/* Decodes `n` values with ec_laplace_decode_p0; returns the final ec_tell. */
int ref_laplace_p0_decode(int *vals, const opus_uint16 *p0, const opus_uint16 *decay, int n,
                          const unsigned char *buf, int len)
{
  ec_dec dec;
  int i;
  ec_dec_init(&dec, (unsigned char *)buf, len);
  for (i = 0; i < n; i++) vals[i] = ec_laplace_decode_p0(&dec, p0[i], decay[i]);
  return ec_tell(&dec);
}

/* RDOVAE encoder: the model and a zeroed state in one allocation. */
typedef struct { RDOVAEEnc model; RDOVAEEncState st; } RefRdovaeEnc;

RefRdovaeEnc *ref_rdovae_enc_create(void)
{
  RefRdovaeEnc *e = calloc(1, sizeof(*e));
  if (init_rdovaeenc(&e->model, rdovaeenc_arrays) != 0) { free(e); return NULL; }
  return e;
}
void ref_rdovae_enc_destroy(RefRdovaeEnc *e) { free(e); }
void ref_rdovae_encode_dframe(RefRdovaeEnc *e, float *latents, float *initial_state, const float *input)
{
  dred_rdovae_encode_dframe(&e->st, &e->model, latents, initial_state, input, 0);
}
/* The state as floats: the GRU states then the conv states, in field order.
   Returns `initialized`; `*n` receives the float count. */
int ref_rdovae_enc_state(const RefRdovaeEnc *e, float *out, int *n)
{
  const RDOVAEEncState *s = &e->st;
  int k = 0;
#define PUT(f) memcpy(out + k, s->f, sizeof(s->f)); k += sizeof(s->f) / sizeof(float);
  PUT(gru1_state) PUT(gru2_state) PUT(gru3_state) PUT(gru4_state) PUT(gru5_state)
  PUT(conv1_state) PUT(conv2_state) PUT(conv3_state) PUT(conv4_state) PUT(conv5_state)
#undef PUT
  *n = k;
  return s->initialized;
}

/* RDOVAE decoder. */
typedef struct { RDOVAEDec model; RDOVAEDecState st; } RefRdovaeDec;

RefRdovaeDec *ref_rdovae_dec_create(void)
{
  RefRdovaeDec *d = calloc(1, sizeof(*d));
  if (init_rdovaedec(&d->model, rdovaedec_arrays) != 0) { free(d); return NULL; }
  return d;
}
void ref_rdovae_dec_destroy(RefRdovaeDec *d) { free(d); }
void ref_rdovae_dec_init_states(RefRdovaeDec *d, const float *initial_state)
{
  dred_rdovae_dec_init_states(&d->st, &d->model, initial_state, 0);
}
void ref_rdovae_decode_qframe(RefRdovaeDec *d, float *qframe, const float *z)
{
  dred_rdovae_decode_qframe(&d->st, &d->model, qframe, z, 0);
}
void ref_rdovae_decode_all(RefRdovaeDec *d, float *features, const float *state, const float *latents, int nb)
{
  DRED_rdovae_decode_all(&d->model, features, state, latents, nb, 0);
}
int ref_rdovae_dec_state(const RefRdovaeDec *d, float *out, int *n)
{
  const RDOVAEDecState *s = &d->st;
  int k = 0;
#define PUT(f) memcpy(out + k, s->f, sizeof(s->f)); k += sizeof(s->f) / sizeof(float);
  PUT(gru1_state) PUT(gru2_state) PUT(gru3_state) PUT(gru4_state) PUT(gru5_state)
  PUT(conv1_state) PUT(conv2_state) PUT(conv3_state) PUT(conv4_state) PUT(conv5_state)
#undef PUT
  *n = k;
  return s->initialized;
}

/* DRED encoder. */
DREDEnc *ref_dred_encoder_create(int fs, int channels)
{
  DREDEnc *e = calloc(1, sizeof(*e));
  dred_encoder_init(e, fs, channels);
  return e;
}
void ref_dred_encoder_destroy(DREDEnc *e) { free(e); }
void ref_dred_encoder_reset(DREDEnc *e) { dred_encoder_reset(e); }
void ref_dred_compute_latents(DREDEnc *e, const float *pcm, int frame_size, int extra_delay)
{
  dred_compute_latents(e, pcm, frame_size, extra_delay, 0);
}
int ref_dred_encode_silk_frame(DREDEnc *e, unsigned char *buf, int max_chunks, int max_bytes, int q0, int dQ,
                               int qmax, unsigned char *activity_mem)
{
  return dred_encode_silk_frame(e, buf, max_chunks, max_bytes, q0, dQ, qmax, activity_mem, 0);
}
/* The integer fields: loaded, Fs, channels, input_buffer_fill, dred_offset,
   latent_offset, last_extra_dred_offset, latents_buffer_fill. */
void ref_dred_encoder_ints(const DREDEnc *e, int *out)
{
  out[0] = e->loaded;
  out[1] = e->Fs;
  out[2] = e->channels;
  out[3] = e->input_buffer_fill;
  out[4] = e->dred_offset;
  out[5] = e->latent_offset;
  out[6] = e->last_extra_dred_offset;
  out[7] = e->latents_buffer_fill;
}
/* The float arrays: 0 input_buffer, 1 latents_buffer, 2 state_buffer,
   3 resample_mem. Returns the array and its length in `*n`. */
const float *ref_dred_encoder_floats(const DREDEnc *e, int which, int *n)
{
  switch (which) {
  case 0: *n = sizeof(e->input_buffer) / sizeof(float); return e->input_buffer;
  case 1: *n = sizeof(e->latents_buffer) / sizeof(float); return e->latents_buffer;
  case 2: *n = sizeof(e->state_buffer) / sizeof(float); return e->state_buffer;
  default: *n = sizeof(e->resample_mem) / sizeof(float); return e->resample_mem;
  }
}
/* The RDOVAE encoder state inside a DREDEnc, as ref_rdovae_enc_state. */
int ref_dred_encoder_rdovae_state(const DREDEnc *e, float *out, int *n)
{
  RefRdovaeEnc tmp;
  memcpy(&tmp.st, &e->rdovae_enc, sizeof(tmp.st));
  return ref_rdovae_enc_state(&tmp, out, n);
}
void ref_filter_df2t(const float *in, float *out, int len, float b0, const float *b, const float *a, int order,
                     float *mem)
{
  filter_df2t(in, out, len, b0, b, a, order, mem);
}

/* DRED decoder. */
OpusDRED *ref_dred_create(void) { return calloc(1, sizeof(OpusDRED)); }
void ref_dred_destroy(OpusDRED *d) { free(d); }
int ref_dred_ec_decode(OpusDRED *d, const unsigned char *bytes, int num_bytes, int max_dred_features,
                       int dred_frame_offset)
{
  return dred_ec_decode(d, bytes, num_bytes, max_dred_features, dred_frame_offset);
}
/* nb_latents, process_stage, dred_offset. */
void ref_dred_ints(const OpusDRED *d, int *out)
{
  out[0] = d->nb_latents;
  out[1] = d->process_stage;
  out[2] = d->dred_offset;
}
/* 0 fec_features, 1 state, 2 latents (mutable, to seed them). */
float *ref_dred_floats(OpusDRED *d, int which, int *n)
{
  switch (which) {
  case 0: *n = sizeof(d->fec_features) / sizeof(float); return d->fec_features;
  case 1: *n = sizeof(d->state) / sizeof(float); return d->state;
  default: *n = sizeof(d->latents) / sizeof(float); return d->latents;
  }
}
/* The decoding step of opus_dred_process with the built-in model. */
void ref_dred_process(OpusDRED *d, const RefRdovaeDec *m)
{
  DRED_rdovae_decode_all(&m->model, d->fec_features, d->state, d->latents, d->nb_latents, 0);
  d->process_stage = 2;
}
