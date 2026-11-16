# Differential fuzzing against libopus

These targets compare opus-rs with libopus 1.6.1 instead of only looking for
crashes. The reference is built from source by `fuzz/opusref` as a float,
scalar-C library (no intrinsics, no runtime CPU dispatch, `-ffp-contract=off`),
so its results do not depend on the host CPU.

## Setup

- libopus 1.6.1 sources in `opus-1.6.1/` at the repository root (gitignored;
  the existing interop tests use the same place), or `OPUSREF_SRC=<dir>`.
- Nightly Rust and `cargo-fuzz`.
- opus-rs's `fuzzing` feature: it exposes `comb_filter` and a runtime switch
  for the x86 AVX/FMA kernels, so each target runs opus-rs with SIMD off
  (comparable to scalar libopus) and on (compared with SIMD off).

```sh
cargo +nightly fuzz build
cargo +nightly fuzz run diff_decoder -- -len_control=0 -max_len=8192
```

## Targets

| Target | Compares | Oracles |
| --- | --- | --- |
| `diff_decoder` | Sequences of `decode`, `decode_fec`, loss and reset calls with arbitrary packets, frame sizes and valid libopus-encoded packets with forced mode switches | Same result and sample count, same final range after every call, PCM within `OFZ_DEC_TOL` (4e-6 of max(1, peak): below CELT float rounding, above one SILK LSB), SIMD vs scalar, no panic on absurd frame sizes |
| `diff_encoder` | Generated signals (silence, DC, clipping, low pitch, impulses, voice-like, raw) with settings that change between segments, including bitrates up to 4 Mb/s | libopus decodes every packet with the encoder's final range, CBR packet length, lookahead, ToC, byte identity in CELT-only streams (`OFZ_STRICT_BYTES=1`), per-frame fidelity |
| `diff_kernels` | DSP functions against the same libopus functions: `haar1`, `exp_rotation`, PVQ search, `celt_fir`, `silk_sum_sqr_shift`, `comb_filter`, SILK fixed-point macros, LPC, autocorrelation | Bit-exact where the arithmetic is element-wise or integer, a relative tolerance where the summation order differs, and for the PVQ search a pulse count of K and an objective no worse than libopus's |
| `api_args` | Arbitrary call sequences on `RangeCoder` and the rate functions | Built with debug assertions: a violated `get_unchecked` precondition aborts, and AddressSanitizer reports out-of-bounds accesses; ordinary panics are allowed |

All targets add IJON-style feedback (`common::ijon_set`/`ijon_max`): states
such as (previous packet mode, call, packet mode) or a value's magnitude
bucket count as coverage, so mode transitions and extreme values are explored.

### Locating encoder differences

opus-rs and libopus produce different packets from the first non-silent frame
in almost every configuration, so byte identity alone says little. The
reference decoder records every range-decoder call with its libopus source
line (`opusref/trace.h`). The `bytes` oracle decodes both packets and reports
the first symbol that differs, for example `celt_decoder.c:1426` (a dynalloc
boost) or `length` (same symbols, different size). Findings are grouped by
that site.

### aarch64

The NEON kernels run under qemu-user:

```sh
CC_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnu-gcc \
CXX_aarch64_unknown_linux_gnu=aarch64-unknown-linux-gnu-g++ \
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-unknown-linux-gnu-gcc \
  cargo +nightly fuzz build -s none --target aarch64-unknown-linux-gnu diff_kernels
qemu-aarch64 <target-dir>/aarch64-unknown-linux-gnu/release/diff_kernels -len_control=0
```

## Surveys

`./survey.sh <name> <target> <seconds> [ENV=...]` keeps fuzzing after a
finding: the first input of each signature goes to `survey/<name>/`, with one
line per signature in `findings.txt`. Run the target binary on a saved input,
without `OFZ_SURVEY`, to reproduce it.

`OFZ_SUPPRESS` skips divergences that are already recorded, so a survey can
find the next one:

| Name | Skips |
| --- | --- |
| `tiny-frame` | Packets libopus rejects that opus-rs plays as loss, and 0/1-byte frames inside longer packets (libopus conceals them) |
| `fec-invalid` | `decode_fec` with a packet libopus rejects |
| `toc` | The ToC comparison in `diff_encoder` |
| `simd` | The SIMD encoder in `diff_encoder` |
| `rate`, `icdf` | Those calls in `api_args` |

Tolerances: `OFZ_DEC_TOL`, `OFZ_PLC_TOL`, `OFZ_SIMD_TOL`, `OFZ_KTOL`.
