#include <string.h>
#include "entdec.h"
#include "laplace.h"
#define OFZ_MAX 8192
typedef struct { const char *file; int line; int val; } ofz_sym;
static ofz_sym syms[OFZ_MAX];
static int n, on;
static void rec(const char *f, int l, int v) { if (on && n < OFZ_MAX) { syms[n].file = f; syms[n].line = l; syms[n].val = v; n++; } }
void ofz_trace_start(void) { n = 0; on = 1; }
int ofz_trace_stop(void) { on = 0; return n; }
void ofz_trace_get(int i, const char **f, int *l, int *v) { *f = syms[i].file; *l = syms[i].line; *v = syms[i].val; }
int ofz_icdf(ec_dec *d, const unsigned char *icdf, unsigned ftb, const char *f, int l) { int v = ec_dec_icdf(d, icdf, ftb); rec(f, l, v); return v; }
int ofz_bit_logp(ec_dec *d, unsigned logp, const char *f, int l) { int v = ec_dec_bit_logp(d, logp); rec(f, l, v); return v; }
opus_uint32 ofz_uint(ec_dec *d, opus_uint32 ft, const char *f, int l) { opus_uint32 v = ec_dec_uint(d, ft); rec(f, l, (int)v); return v; }
opus_uint32 ofz_bits(ec_dec *d, unsigned bits, const char *f, int l) { opus_uint32 v = ec_dec_bits(d, bits); rec(f, l, (int)v); return v; }
unsigned ofz_decode(ec_dec *d, unsigned ft, const char *f, int l) { unsigned v = ec_decode(d, ft); rec(f, l, (int)v); return v; }
int ofz_laplace(ec_dec *d, unsigned fs, int decay, const char *f, int l) { int v = ec_laplace_decode(d, fs, decay); rec(f, l, v); return v; }

/* SILK input capture: the 16-bit samples silk_Encode derives from its input
 * (RES2INT16, enc_API.c), for the encoder's coded calls. */
#include "API.h"
#include "float_cast.h"
static opus_int16 silk_in[2 * 5760];
static int silk_n;
opus_int ofz_silk_Encode(void *enc, silk_EncControlStruct *ctl, const opus_res *in, opus_int n,
                         ec_enc *rc, opus_int32 *bytes, const opus_int prefill, int activity) {
  if (!prefill && rc) {
    int i, total = n * ctl->nChannelsAPI;
    for (i = 0; i < total && i < 2 * 5760; i++) silk_in[i] = FLOAT2INT16(in[i]);
    silk_n = i;
  }
  return silk_Encode(enc, ctl, in, n, rc, bytes, prefill, activity);
}
int ofz_take_silk_input(const opus_int16 **p) { int n = silk_n; *p = silk_in; silk_n = 0; return n; }
