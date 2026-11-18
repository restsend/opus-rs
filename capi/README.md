# libopus from opus-rs

`capi/` builds `libopus.so.0` and `libopus.a` with libopus's C API
(`opus.h`, `opus_multistream.h`, `opus_projection.h`) from opus-rs, so C
programs built against libopus link and run against opus-rs unchanged.
It started as PR #115 by Object905, which wrote the C layer, its test
harness and the RFC 8251 conformance runs; this version calls opus-rs for
everything libopus has.

Every function keeps libopus's name, signature, argument checks and error
codes, and calls the opus-rs API of the same libopus function: the
encoder, decoder and every ctl, packet parsing, the repacketizer and
padding, soft clipping, the multistream and surround encoders and the
multistream decoder, the projection (ambisonics) encoder and decoder, and
with the `dred` feature DRED.

Every state is one flat block, as in libopus: `opus_*_get_size()` and
`opus_*_init()` place it in caller memory, `free()` releases it and
`memcpy` copies it (libopus's tests copy decoder and encoder states).
opus-rs is built without its `heap` feature for this, and a multistream
or projection state is a header (`MultistreamLayout`,
`MultistreamEncoderHeader`) followed by its stream codecs, used through
opus-rs's `MultistreamDecoderView`, `MultistreamEncoderView` and
`ProjectionDecoderView`.

## Build and use

With Nix:

```sh
nix build .#libopus        # libopus's default build
nix build .#libopus-dnn    # with --enable-deep-plc, --enable-dred, --enable-osce
```

The package is laid out as libopus installs itself:

```
lib/libopus.so.0          soname libopus.so.0
lib/libopus.so -> libopus.so.0
lib/libopus.a
lib/pkgconfig/opus.pc     Cflags: -I${includedir}/opus, Libs: -lopus
include/opus/opus.h opus_defines.h opus_multistream.h opus_projection.h opus_types.h
```

so `pkg-config opus` and `-lopus` users build against it unchanged. To
use it for every package of a nixpkgs set:

```nix
nixpkgs.overlays = [ (final: prev: { libopus = opus-rs.packages.${final.system}.libopus; }) ];
```

With cargo, `cargo build --release` in `capi/` writes `libopus.so` (soname
`libopus.so.0`) and `libopus.a` to the target directory; the features
`deep-plc`, `dred` and `osce` are libopus's configure options of the same
names. A program that loads the library by its soname needs a
`libopus.so.0` link next to it.

The headers are libopus's own, from the revision opus-rs follows
(`f1afc4222a`, `OPUSREF_SRC` in the dev shell or `libopus/` from
`scripts/fetch-libopus.sh`).

## Tests

`capi/test.sh <dir with libopus.so.0> <work dir> [--dred]` runs:

- libopus's test programs, compiled unmodified from the pinned sources:
  `test_opus_api`, `test_opus_decode`, `test_opus_encode` (with
  `opus_encode_regressions.c`), `test_opus_padding`,
  `test_opus_projection` (with libopus's `src/mapping_matrix.c`, whose
  internal functions it also tests) and, with `--dred`, `test_opus_dred`.
  `test_opus_extensions` tests libopus's internal extension functions and
  is left out.
- every RFC 8251 test vector case (12 vectors, mono and stereo, 5 rates),
  decoded and compared as libopus's `tests/run_vectors.sh` does.
- a cross-check with the reference libopus (pkg-config `opus`): each
  `opus_demo` encodes what the other decodes, over 16 settings, with the
  final range checked on every packet.

`scripts/test-all.sh` runs it for the default and the neural build (steps
`capi` and `capi-dnn`), and `nix flake check` on `packages.libopus`.
`cargo test` in `capi/` calls the C API from Rust; `cargo +nightly miri
test` runs those tests under Miri, except the one of the C-variadic ctls.

Results on x86-64 Linux:

| | PR #115 | now |
|---|---|---|
| `test_opus_api` | fails at the encoder size check | fails at the encoder size check; the rest passes |
| `test_opus_decode` | pass | pass |
| `test_opus_encode` | fails (no multistream encoder) | pass |
| `test_opus_padding` | pass | pass |
| `test_opus_projection` | not run (no projection) | pass |
| `test_opus_dred` | not run (no DRED) | pass (`libopus-dnn`) |
| RFC 8251 vectors | 95/120 | 120/120 |
| cross-check | pass | pass |

`test_opus_api` requires `opus_encoder_get_size()` to be at most 256 KiB;
opus-rs's encoder is 304800 bytes, as it keeps CELT's per-call buffers in
its state. `test.sh` reports that check as a known failure and runs a copy
of the test with the bound raised to 512 KiB, which passes.

## Differences from libopus

- `opus_custom_*` is not exported: libopus exports it only with
  `--enable-custom-modes`.
- The private ctls `OPUS_SET_VOICE_RATIO`, `OPUS_SET_LFE` and
  `OPUS_SET_ENERGY_MASK` return `OPUS_UNIMPLEMENTED`; the multistream
  encoder sets the latter two on its streams itself. `OPUS_SET_QEXT` and
  `OPUS_SET_DNN_BLOB` are `OPUS_UNIMPLEMENTED`, as in libopus's default
  build.
- `OPUS_SET_EXPERT_FRAME_DURATION` on a multistream encoder rejects values
  outside `OPUS_FRAMESIZE_*` with `OPUS_BAD_ARG`; libopus stores them and
  fails the next encode call.
- The encoder state is 304800 bytes (libopus: about 110 KiB) and
  `opus_rs::OpusEncoder::new` builds it on the stack: `opus_encoder_create`
  and `opus_encoder_init` need between 768 KiB and 1 MiB of stack (x86-64,
  release build), and crash on a thread with less, such as musl's default
  128 KiB. Encoding a 20 ms stereo frame needs up to 128 KiB, decoding one
  less than 64 KiB. The decoder state is 63760 bytes.
- `libopus.a` holds the Rust standard library: a program that links
  another Rust static library besides it can get duplicate symbols.
- `opus_get_version_string()` returns `libopus 1.6.1-opus-rs`.
