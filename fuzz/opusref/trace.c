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
