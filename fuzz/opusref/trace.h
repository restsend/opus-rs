/* Records every range-decoder call made by the wrapped libopus files, with
 * the libopus source line, so two packets can be compared symbol by symbol. */
#ifndef OFZ_TRACE_H
#define OFZ_TRACE_H
#include "entdec.h"
#include "laplace.h"
int ofz_icdf(ec_dec *d, const unsigned char *icdf, unsigned ftb, const char *f, int l);
int ofz_bit_logp(ec_dec *d, unsigned logp, const char *f, int l);
opus_uint32 ofz_uint(ec_dec *d, opus_uint32 ft, const char *f, int l);
opus_uint32 ofz_bits(ec_dec *d, unsigned bits, const char *f, int l);
unsigned ofz_decode(ec_dec *d, unsigned ft, const char *f, int l);
int ofz_laplace(ec_dec *d, unsigned fs, int decay, const char *f, int l);
#define ec_dec_icdf(d, i, b) ofz_icdf(d, i, b, __FILE__, __LINE__)
#define ec_dec_bit_logp(d, p) ofz_bit_logp(d, p, __FILE__, __LINE__)
#define ec_dec_uint(d, t) ofz_uint(d, t, __FILE__, __LINE__)
#define ec_dec_bits(d, b) ofz_bits(d, b, __FILE__, __LINE__)
#define ec_decode(d, t) ofz_decode(d, t, __FILE__, __LINE__)
#define ec_laplace_decode(d, fs, dec) ofz_laplace(d, fs, dec, __FILE__, __LINE__)
#endif
