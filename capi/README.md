# opus-rs-capi

The libopus C API over opus-rs. `cargo build --release` produces
`target/release/libopus.so` (soname `libopus.so.0`) and `libopus.a`, exporting
libopus 1.6.1's functions with their signatures and argument checking, so C
programs built against libopus's headers link and run against opus-rs.

```sh
cd capi && cargo build --release
cc app.c -I/usr/include/opus -Ltarget/release -lopus -lm
# or, for an existing binary:
ln -s libopus.so target/release/libopus.so.0
LD_LIBRARY_PATH=target/release ./app
```

The codec state is one flat block, as in libopus (opus-rs is built without
`heap`): `opus_*_get_size()` / `opus_*_init()` on caller memory and `memcpy`
of a state work. libopus's tests rely on this: they `memcpy` and `memset`
decoder states.

The `opus_*_ctl()` functions are C-variadic functions written in Rust
(`args: ...`, stable since Rust 1.99), with no C shim.

## Testing with libopus's test suite

The libopus sources come from `LIBOPUS_SRC`, `../opus-1.6.1`, or the copy the
`opusic-sys` dev-dependency unpacks in the cargo registry. The cross-check and
`--ref` need a reference libopus (pkg-config `opus`). The test vectors are
https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz,
which unpacks to `opus_newvectors/`.

```sh
capi/run_libopus_tests.sh                 # test programs + cross-check
capi/conformance.sh --ref capi/target/opus_newvectors   # every vector case
```

`run_libopus_tests.sh` compiles libopus's test programs, unmodified, against
this library and runs them. `conformance.sh` runs every test vector case in
parallel (12 vectors × mono/stereo × 5 output rates) and lists each failure
with `opus_compare`'s error; with `--ref` it runs the reference libopus too,
which should pass all of them.

### Results

| test | result |
|---|---|
| `test_opus_padding` | pass |
| `test_opus_decode` | pass (random and adversarial packets, PLC, state copies) |
| `test_opus_api` | fails one check: `opus_encoder_get_size()` must be ≤ 256 KiB; it is 317 KiB. With that bound relaxed, all of it passes (~118M calls: every ctl, packet parsing, repacketizer, multistream decoder) |
| `test_opus_encode` | fails at its first multistream encode (not implemented) |
| cross-check | pass: libopus encodes / opus-rs decodes and the reverse over 16 settings, the range coder state checked per packet |
| test vectors | 95/120 (reference libopus: 120/120) |

### Test vector failures

No case fails to decode: in every packet of every vector the range decoder
ends in the encoder's state, so the bitstream is parsed exactly. The failures
are in the output, judged by `opus_compare`, which passes a weighted error up
to about 0.277. They fall into two groups.

**Output below SILK's internal rate (8 and 12 kHz).** Large errors (6 to 88):

| vector | content | 12 kHz | 8 kHz |
|---|---|---|---|
| 02 | SILK NB | pass | pass |
| 03 | SILK MB | pass | 23.7 |
| 04 | SILK WB | 15.2 | 88.0 |
| 05 | Hybrid SWB | 6.6–8.9 | 31.1 |
| 06 | Hybrid FB | 6.4–6.5 | 29.6 |
| 12 | SILK NB/MB/WB, Hybrid SWB | 14.6 | 71.6 |

A vector fails exactly when its SILK layer runs at a higher rate than the
output (MB is 12 kHz, WB and Hybrid 16 kHz), and passes at 16 kHz and above.
That points at SILK's output resampling when it downsamples. This is
unconfirmed.

**testvector07 (CELT, all bandwidths, mono/stereo switches).** Fails just
over the threshold at every rate except 16 kHz mono: error 0.28 to 0.36
against `testvector07m.dec`, and about 0.54 to 0.59 against
`testvector07.dec`. The other CELT vectors (01, 08–11) pass at every rate.

All 25 failing cases (`conformance.sh` output; the first error is against
`testvectorNN.dec`, the second against `testvectorNNm.dec`):

```
testvector07 48000 mono   error 0.538594 (.dec) / 0.356175 (m.dec)
testvector07 48000 stereo error 0.591799 (.dec) / 0.304483 (m.dec)
testvector07 24000 mono   error 0.537134 (.dec) / 0.350395 (m.dec)
testvector07 24000 stereo error 0.590130 (.dec) / 0.294463 (m.dec)
testvector07 16000 stereo error 0.586232 (.dec) / 0.280777 (m.dec)
testvector04 12000 mono   error 15.233897 (.dec) / 15.233897 (m.dec)
testvector04 12000 stereo error 15.128353 (.dec) / 15.128353 (m.dec)
testvector05 12000 mono   error 6.573266 (.dec) / 6.570979 (m.dec)
testvector05 12000 stereo error 8.866322 (.dec) / 8.873761 (m.dec)
testvector06 12000 mono   error 6.430621 (.dec) / 6.430621 (m.dec)
testvector06 12000 stereo error 6.501177 (.dec) / 6.500511 (m.dec)
testvector07 12000 stereo error 0.584466 (.dec) / 0.279432 (m.dec)
testvector12 12000 mono   error 14.631038 (.dec) / 14.631038 (m.dec)
testvector12 12000 stereo error 14.531920 (.dec) / 14.531920 (m.dec)
testvector03  8000 mono   error 23.752755 (.dec) / 23.752755 (m.dec)
testvector03  8000 stereo error 23.694767 (.dec) / 23.694767 (m.dec)
testvector04  8000 mono   error 88.041362 (.dec) / 88.041362 (m.dec)
testvector04  8000 stereo error 87.857932 (.dec) / 87.857932 (m.dec)
testvector05  8000 mono   error 31.160963 (.dec) / 31.160961 (m.dec)
testvector05  8000 stereo error 31.147843 (.dec) / 31.147613 (m.dec)
testvector06  8000 mono   error 29.630099 (.dec) / 29.630099 (m.dec)
testvector06  8000 stereo error 29.583030 (.dec) / 29.583031 (m.dec)
testvector07  8000 stereo error 0.581190 (.dec) / 0.279549 (m.dec)
testvector12  8000 mono   error 71.715824 (.dec) / 71.715824 (m.dec)
testvector12  8000 stereo error 71.573301 (.dec) / 71.573301 (m.dec)
```

Content of the vectors, from their ToC bytes:

| vector | packets | modes | channels |
|---|---|---|---|
| 01 | 2147 | CELT FB | stereo |
| 02 | 1185 | SILK NB | mono/stereo |
| 03 | 998 | SILK MB | mono/stereo |
| 04 | 1265 | SILK WB | mono/stereo |
| 05 | 2037 | Hybrid SWB | mono/stereo |
| 06 | 1876 | Hybrid FB | mono/stereo |
| 07 | 4186 | CELT NB/WB/SWB/FB, ~equal shares | mono/stereo |
| 08 | 1247 | CELT NB–FB, 5 SILK NB | mono/stereo |
| 09 | 1337 | mostly CELT FB, 5 SILK NB | mono/stereo |
| 10 | 1912 | CELT FB, Hybrid FB | mono/stereo |
| 11 | 553 | CELT FB | stereo |
| 12 | 1332 | SILK NB/MB/WB, Hybrid SWB | mono |

## Reproducing

Environment of the results above: opus-rs 5d8230f (0.1.42), rustc 1.99.0,
gcc 16.2.1, x86_64 Linux; libopus 1.6.1 test sources from opusic-sys 0.7.5;
reference libopus: Arch `opus 1.6.1-1` (float build, custom modes enabled).

The programs land in `capi/target/libopus-tests/` (`capi/run_libopus_tests.sh
build` builds them alone). From there:

```sh
# one test vector case: decode (range coder checked per packet), then compare
./opus_demo -d 12000 1 -ignore_extensions $V/testvector04.bit out.pcm
./opus_compare -r 12000 $V/testvector04.dec out.pcm      # -s for stereo
# the same with the reference libopus
./opus_demo_ref -d 12000 1 -ignore_extensions $V/testvector04.bit out.pcm

# a cross-check: libopus encodes, opus-rs decodes (and swap the two)
./opus_demo_ref -e audio 48000 2 64000 speech_stereo.raw x.bit
./opus_demo -d 48000 2 x.bit x.pcm

# test_opus_api past the encoder size check: a copy with the bound raised
cp $SRC/tests/test_opus_api.c api_relaxed.c
sed -i 's/(i<=2048||i>1<<18)/(i<=2048||i>1<<19)/' api_relaxed.c
cc -O2 -Icfg -I$SRC/include -I$SRC/celt -I$SRC/silk -I$SRC/src -I$SRC/tests \
    api_relaxed.c -o api_relaxed -L. -Wl,-rpath,'$ORIGIN' -l:libopus.so.0 -lm
```

`test_opus_decode` and `test_opus_encode` draw a random seed per run (printed
first; the failure message gives `SEED=`); a run is repeated with
`./test_opus_decode <seed>` or `SEED=<seed> ./test_opus_decode`. The passing runs' seeds were not recorded.

## Not covered

Passing tests here say less than they may seem to:

- The encoder is only checked for producing packets libopus decodes with the
  same range coder state. Its output is not compared with libopus's encoder
  (bitrate, quality, mode and bandwidth decisions), and only over 16 settings
  on one speech file read at several rates.
- That the encoder reports a non-zero final range on every packet (without
  which `opus_demo` skips the comparison) was checked for one setting
  (audio, 48 kHz stereo, 64 kb/s).
- Decoder output is compared with libopus only through the test vectors:
  not for PLC, FEC (`decode_fec`), decoder gain, soft clipping, `opus_decode24`,
  or multistream streams. `test_opus_decode` checks robustness and API
  behaviour, not output values.
- `opus_encode24` is not exercised by any test.
- Only x86_64 Linux was tested. The variadic ctls rely on Rust's C-variadic
  support for the target's ABI.

## Gaps

- Multistream and projection (ambisonics) encoding, DRED: return
  `OPUS_UNIMPLEMENTED`, as a libopus built without them would. Multistream
  decoding works. This blocks `test_opus_encode`, whose regressions use
  surround and ambisonics encoders.
- `opus_custom_*` is not exported.
- Encoder state is 317 KiB, about 3× libopus's 107–123 KiB. The largest part is per-call
  CELT scratch that opus-rs keeps in the state, where libopus uses the stack:
  the `w_*` buffers, ~129 KB of `CeltEncoder`'s 174 KB. Decoder state is close to libopus's (175 KiB
  vs 151–160 KiB).
- Accepted, validated and reported back, but not acted on (opus-rs has no
  counterpart): `OPUS_SET_FORCE_CHANNELS`, `OPUS_SET_LSB_DEPTH`,
  `OPUS_SET_PREDICTION_DISABLED`, `OPUS_SET_PHASE_INVERSION_DISABLED` (encoder
  and decoder), `OPUS_SET_FORCE_MODE`; `OPUS_GET_PITCH` reports 0.
- `OPUS_APPLICATION_RESTRICTED_SILK` is `OPUS_UNIMPLEMENTED`;
  `RESTRICTED_CELT` codes as `RESTRICTED_LOWDELAY`.
- The repacketizer drops extensions found in padding instead of carrying them
  over.

## Harness pitfalls

- cargo writes `libopus.so`, but programs link to the soname
  `libopus.so.0`; without a `libopus.so.0` next to them they silently load
  the system libopus. `run_libopus_tests.sh` links through a `libopus.so.0`
  symlink and refuses to run unless `ldd` shows it.
- `run_vectors.sh` exits 0 when it finds no vectors, and stops at the first
  failing vector of each rate. `run_libopus_tests.sh test-vectors` counts the
  comparisons that matched, and `conformance.sh` runs every case.
- `opus_demo` only compares range coder states when the stored one is
  non-zero. The encoder sets it on every packet (checked), and a corrupted
  packet makes both decoders report a mismatch, with the same wrong value.
