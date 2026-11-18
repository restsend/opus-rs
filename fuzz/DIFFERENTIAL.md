# Differential fuzzing against libopus

These targets compare opus-rs with libopus (main at f1afc4222a) instead of only looking for
crashes. The reference is built from source by `fuzz/opusref` as a float,
scalar-C library (no intrinsics, no runtime CPU dispatch, `-ffp-contract=off`),
so its results do not depend on the host CPU.

## Setup

- The libopus sources with their neural model data in `libopus/` at the
  repository root (gitignored; `scripts/fetch-libopus.sh` fetches the pinned
  revision), or `OPUSREF_SRC=<dir>`.
- Nightly Rust and `cargo-fuzz`.
- opus-rs's `fuzzing` feature: it exposes `comb_filter` and the
  `SIMD_ENABLED` switch the targets flip between a scalar run and a SIMD
  run. The switch has no effect yet: opus-rs's one target-specific kernel,
  SILK's AVX2 delayed-decision filter, is chosen by runtime CPU detection
  and gives the scalar path's integer results, which the SILK comparisons
  with fixed-point libopus cover on AVX2 machines.

```sh
cargo +nightly fuzz build
cargo +nightly fuzz run diff_decoder -- -len_control=0 -max_len=8192
```

## Reference patches

`opusref/patches/*.patch` fix libopus bugs in the reference that opus-rs
fixes too. `opusref/build.rs` applies them in name order (`patch -p1`) to
copies of the `.c` files they change, in its `OUT_DIR`; the libopus tree,
`OPUSREF_SRC` and the flake's source stay unchanged. Each patch is the diff
proposed upstream and starts with a description: the bug it fixes and the
reference builds it changes (float, `dnn`, `fixed`, `simd`). A patch is
dropped when the pinned libopus revision contains the fix.

## Running every check

`scripts/test-all.sh` (from the repository root) runs everything and prints a
summary; it exits non-zero if a step fails:

- `cargo test --release`, with and without `--features deep-plc,dred,osce`,
  and with `--features silk-fixed`;
- the `no_std` builds (`--no-default-features --features libm`, also with the
  neural features and with `silk-fixed`) and the `no_std` library and
  integration tests;
- the opusref tests: float (with the float SILK grid `silk_float`),
  `--features dnn`, `--features fixed --test silk_fixed`;
- `opusref/examples/parity.rs`, which must report every 48 kHz CELT
  configuration byte-identical with libopus;
- the regression replay `tests/regressions.rs` (below) for the default, dnn
  and fixed builds.

`scripts/test-all.sh --fuzz` also builds the targets with `cargo fuzz build`
(nightly cargo and cargo-fuzz on `PATH`) and runs `regressions/check.sh`,
then the same for the fixed build (`--features fixed`, `OFZ_BUILD=fixed`).
The script needs stable cargo, a C and C++ compiler, pkg-config, cmake,
libopus (for the interop dev-dependencies) and the pinned libopus sources
(`OPUSREF_SRC` or `libopus/`); the dev shell provides them.

## Regression reproducers

`regressions/manifest.tsv` and `regressions/manifest.d/*.tsv` list one
reproducer per fixed divergence: an input for a target with its environment
and the output pattern that shows the bug, or `test:<name>` for an opusref
test. The target bodies live in the fuzz crate's library (`src/`), which the
libFuzzer binaries in `fuzz_targets/` call, so `cargo test --test
regressions` replays every input on the stable toolchain, each in its own
process, and fails if a pattern appears; `regressions/check.sh` does the same
with the cargo-fuzz binaries. A target suffix selects the build: `@dnn` and
`@fixed` entries run only with `--features dnn` or `--features fixed` (and
such a build runs only those), `@aarch64` entries only on aarch64. The
replay keeps the test profile's debug assertions and overflow checks, as
cargo-fuzz does; `stack-buffer-overflow` patterns need AddressSanitizer and
show only in check.sh. `REGRESSIONS.md` maps every fix to the test or
reproducer that guards it.

## Targets

| Target | Compares | Oracles |
| --- | --- | --- |
| `diff_decoder` | Sequences of `decode`, `decode_fec`, loss, `reset` (against `OPUS_RESET_STATE`), new-decoder and control calls (gain, phase inversion, ignore extensions, the output format: float, `opus_decode` or `opus_decode24`) with arbitrary packets, frame sizes and valid libopus-encoded packets with forced mode switches | Same result and sample count, same final range after every call, same control results and values, same sample rate, bandwidth, pitch and last packet duration, PCM within `OFZ_DEC_TOL` (4e-6 of max(1, peak): below CELT float rounding, above one SILK LSB), SIMD vs scalar, no panic on absurd frame sizes |
| `diff_encoder` | Generated signals (silence, DC, clipping, low pitch, impulses, voice-like, raw) in all five applications with settings that change between segments, including bitrates up to 4 Mb/s; per segment also the frame duration, an output buffer of 0-1599 bytes, the 16-bit and 24-bit entry points, an invalid frame size, a `reset` (against `OPUS_RESET_STATE`) or new encoders, and encoder controls (application, bitrate, complexity, loss, FEC 0-2, VBR, LSB depth, force channels, prediction and phase inversion disabled, expert frame duration, the 24-bit entry point) | Both encoders accept or reject each control alike and report the same value and the same bitrate query after each frame, both encoders accept or reject each call, libopus decodes every packet with the encoder's final range, CBR packet length, lookahead, ToC, SILK input, byte identity where `common::strict_bytes` says the encoders run the same algorithm, the CELT-only applications so far (`OFZ_STRICT_BYTES=1`), per-frame fidelity |
| `diff_kernels` | DSP functions against the same libopus functions: `haar1`, `exp_rotation`, PVQ search, `celt_fir`, `silk_sum_sqr_shift`, `comb_filter`, SILK fixed-point macros, LPC, autocorrelation | Bit-exact where the arithmetic is element-wise or integer, a relative tolerance where the summation order differs, and for the PVQ search a pulse count of K and an objective no worse than libopus's |
| `api_args` | Arbitrary call sequences on `RangeCoder` and the rate functions, and libopus API calls with arbitrary arguments, buffer lengths and frame sizes (`api_calls`: encoder controls and encodes, decodes in every format, the packet functions, the repacketizer and padding, multistream and projection codecs of any layout) | Built with debug assertions: a violated `get_unchecked` precondition aborts, and AddressSanitizer reports out-of-bounds accesses; ordinary panics of the low-level functions are allowed, a panic of a libopus API call is a crash, and the encoder controls and valid encodes return libopus's error codes |
| `diff_multistream` | Random channel layouts (1-8 channels, streams, coupled streams, mapping) decoding random and libopus-encoded multistream packets, losses and FEC in float, 16-bit and 24-bit output, with resets and controls (gain, complexity, phase inversion) | Both accept or reject the layout, the same result and sample count per call, PCM within `OFZ_DEC_TOL` (plus one step of integer output), the same final range and queries, the same control results |
| `diff_ms_encoder` | Random layouts (1-8 channels, streams, coupled streams, mapping) or the surround layouts of mapping families 0, 1, 2 and 255, and applications encoding generated signals of every frame duration into buffers of 0-5999 bytes through the float, 16-bit and 24-bit entry points, with the multistream controls and resets | Both accept or reject the layout and each call with the same error code, libopus's multistream decoder decodes every packet to its duration with the encoder's final range, the first stream's ToC agrees (every stream's where `common::strict_bytes` holds, with the bytes), the bitrate query agrees where the stream modes do |
| `diff_projection` | Ambisonics projection encoders of any channel count and mapping family (mostly valid ones) and decoders of the encoder's or arbitrary layouts and demixing matrices, on generated signals through the three entry points, arbitrary and libopus-encoded packets, losses, the three output formats and bitrate changes | Both create or reject alike with the same error code, streams and demixing matrix, the same call results, byte identity where `common::strict_bytes` holds, decoded PCM within `OFZ_DEC_TOL` per decoded channel (plus one integer step each) |
| `diff_repacketizer` | Sequences of repacketizer calls (`cat`, `out_range`, `out`, `init`), `opus_packet_pad`/`unpad` and the multistream variants, `opus_packet_parse` with the packet queries, and `opus_pcm_soft_clip`, on arbitrary packets and libopus-encoded ones with extensions in their padding | The same result or error code per call, identical written bytes, the same frames, counts and queries, bit-identical soft clipping and state |

All targets add IJON-style feedback (`common::ijon_set`/`ijon_max`): states
such as (previous packet mode, call, packet mode) or a value's magnitude
bucket count as coverage, so mode transitions and extreme values are explored.

### Locating encoder differences

CELT-only packets are byte-identical with the scalar float reference in the
48 kHz configurations of `opusref/examples/parity.rs` (mono and stereo,
10/20 ms, 24-510 kb/s, VBR, CVBR and CBR) and in `opusref/tests/celt_parity.rs`;
other sampling rates and the fuzzer's settings still find differences. The
reference decoder records every range-decoder call with its libopus source
line (`opusref/trace.h`). The `bytes` oracle decodes both packets and reports
the first symbol that differs, for example `celt_decoder.c:1426` (a dynalloc
boost) or `length` (same symbols, different size). Findings are grouped by
that site.

`opusref/examples/encdiff.rs` prints that first differing symbol per frame
for one CELT-only configuration. To find the computation behind it, both
encoders dump intermediate values (pre-emphasised input, MDCT, band
energies, allocation, ...) under `OFZ_DUMP=1`: opus-rs through `dumpf!`/
`dumpi!` with the `fuzzing` feature, the reference through a copy of
libopus instrumented by `opusref/dump/instrument.py`:

```sh
opusref/dump/instrument.py ../libopus /tmp/ref-dump
cd opusref && OPUSREF_SRC=/tmp/ref-dump CARGO_TARGET_DIR=<separate dir> \
  cargo build --release --example encdiff
OFZ_DUMP=1 <dir>/release/examples/encdiff 1 20 64 0 voice 3 2>dump.txt
dump/compare.py dump.txt 0   # first frame, value by value
```

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
| `simd` | The SIMD encoder in `diff_encoder` and the SIMD decoder's oracle in `diff_decoder` |
| `rate`, `icdf` | Those calls in `api_args` |
| `silk-input` | The SILK input comparison in `diff_encoder` |

Tolerances: `OFZ_DEC_TOL`, `OFZ_PLC_TOL`, `OFZ_SIMD_TOL`, `OFZ_KTOL`.

## SIMD reference

On x86-64, `--features simd` builds the reference as libopus's configure
builds it (intrinsics, run-time CPU detection, GCC's contraction in the
`-mfma` files). opus-rs reproduces its arithmetic at its x86 levels
(`opus_rs::Arch`). The pair runs at the levels the environment names:
`OPUS_RS_ARCH` (`scalar`, `sse2`, `sse4_1`, `avx2`, `native`; unset:
`scalar`) for opus-rs and `OPUSREF_ARCH` (libopus's 0-4; unset: the
CPU's) for the reference:

```sh
cargo +nightly fuzz build diff_encoder --features simd
OPUS_RS_ARCH=native <target-dir>/release/diff_encoder ...
OPUS_RS_ARCH=sse2 OPUSREF_ARCH=2 <target-dir>/release/diff_encoder ...
```

The `simd` encoder and decoder of the targets run at the CPU's level.

With `--features simd,dnn` the reference also builds libopus's x86 neural
network kernels (`dnn/x86/`), and the neural features of both sides (deep
PLC, DRED, OSCE, the bandwidth extension) run at the pair's levels.

## SILK encoder

opus-rs's SILK encoder is libopus's float encoder (`silk/float/`), the one of
libopus's default and SIMD builds, and its SILK input is the float build's:
against the default reference, SILK packets, the SILK input of every
application and the DTX decisions are exact (`opusref/tests/silk_float.rs`
over a grid of SILK configurations, `diff_encoder`'s `silk-input` oracle).
The encoder follows the SIMD build at its levels too: `silk_float` runs
against the `simd` reference (the Avx2 level's inner product,
`silk_inner_product_FLP_avx2`, takes fused multiply-adds).

## Fixed-point reference

`--features fixed` builds the reference as fixed-point libopus
(`opusref/build.rs`) and opus-rs with its Cargo feature `silk-fixed`, the
SILK encoder of fixed-point libopus (`silk/fixed/`). Against it, SILK
packets and the VoIP SILK input are exact (`opusref/tests/silk_fixed.rs`);
the Audio and low-delay SILK input follows the float build in opus-rs, so
skip that comparison:

```sh
cargo +nightly fuzz build diff_encoder --features fixed
OFZ_SUPPRESS=toc,simd,silk-input <target-dir>/release/diff_encoder ...
```

Both opus-rs's fixed-point SILK encoder and this reference carry the fix of
`opusref/patches/silk-fixed-min-inv-gain.patch`. Lower fidelity of SILK
frames of the fixed build than of the float reference comes from the
fixed-point encoder, for example its LPC analysis of the rounded 16-bit
`LPC_in_pre` on quiet frames after loud ones (reproducers f1 and f2 hold
such frames).

## Neural network features

`--features dnn` builds the reference as libopus with
`--enable-deep-plc --enable-dred --enable-osce` (see `opusref/build.rs`) and
opus-rs with `deep-plc`, `dred` and `osce`; `diff_decoder` then also picks
a decoder complexity (0-10) and the bandwidth extension from its input:

```sh
cargo +nightly fuzz run diff_decoder --features dnn -- -len_control=0 -max_len=8192
```

Deep PLC, LACE/NoLACE and BBWENet are bit-exact with libopus where the
decoded signal they start from is; any difference before a loss grows
through the networks, so the decoder tolerances then do not hold for
concealed frames. The directed tests are `opusref/tests/dnn_*.rs`
(`cargo test -p opusref --features dnn`).

With `--features dnn`, `diff_decoder` also receives DRED: parses of the
last packet, of raw bytes and of garbled packets, the processing, and
concealment with DRED at offsets inside the parsed range (`DredBurst`
makes packets with DRED from a libopus encoder; `OFZ_DEBUG` prints each
DRED call's results).
