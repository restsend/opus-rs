#ifndef OFZ_DUMP_H
#define OFZ_DUMP_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int ofz_dump_on(void) { static int on = -1; if (on < 0) on = getenv("OFZ_DUMP") != NULL; return on; }
static void ofz_dumpf(const char *t, const float *x, int n) { int i; if (!ofz_dump_on()) return; fprintf(stderr, "D %s f %d", t, n); for (i = 0; i < n; i++) { unsigned u; memcpy(&u, &x[i], 4); fprintf(stderr, " %08x", u); } fprintf(stderr, "\n"); }
static void ofz_dumpi(const char *t, const int *x, int n) { int i; if (!ofz_dump_on()) return; fprintf(stderr, "D %s i %d", t, n); for (i = 0; i < n; i++) fprintf(stderr, " %d", x[i]); fprintf(stderr, "\n"); }
static void ofz_dump1f(const char *t, float v) { ofz_dumpf(t, &v, 1); }
static void ofz_dump1i(const char *t, int v) { ofz_dumpi(t, &v, 1); }
#endif
