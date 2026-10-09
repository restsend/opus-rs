# opus-rs

A pure-Rust implementation of the [Opus audio codec](https://opus-codec.org/) (RFC 6716), ported from the reference C implementation (libopus 1.6).

> **Production-ready**

## Features

- **Pure Rust** — no C dependencies
- **High Performance:** Competitive with C libopus on x64/aarch64
- **`#![no_std]` + no `alloc`:** fully heap-free — runs on bare metal, RTOS, and WebAssembly with **no global allocator**

## Quick Start

```rust
use opus_rs::{OpusEncoder, OpusDecoder, Application};

// Encode
let mut encoder = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
encoder.bitrate_bps = 16000;
encoder.use_cbr = true;

let input = vec![0.0f32; 320]; // 20ms frame at 16kHz
let mut output = vec![0u8; 256];
let bytes = encoder.encode(&input, 320, &mut output).unwrap();

// Encode PCM16 directly (issue #28): skip the f32 round-trip in the
// SILK/VoIP path. Output is byte-for-byte identical to `encode()` called
// with the same audio converted via `sample as f32 / 32768.0`.
let input_i16 = vec![0i16; 320]; // 20ms frame at 16kHz
let bytes = encoder.encode_i16(&input_i16, 320, &mut output).unwrap();

// Decode
let mut decoder = OpusDecoder::new(16000, 1).unwrap();
let mut pcm = vec![0.0f32; 320];
let samples = decoder.decode(&output[..bytes], 320, &mut pcm).unwrap();
```

### WAV Roundtrip

```bash
# Rust encoder/decoder
cargo run --example wav_test
```

## `#![no_std]` + no-`alloc` Support

By default opus-rs runs on `std` with the **`heap` feature**: the large codec state
(`OpusEncoder`/`OpusDecoder` working buffers) is allocated with `Box`, so the public
structs are tiny (~1 KB each) and constructing them uses almost no stack.

For `#![no_std]`, disable the default features. The crate then builds as
`#![no_std]` with **no `alloc`** — fully heap-free, no global allocator required —
and every working buffer is an inline, fixed-size array (sized to the Opus worst
case); growable state uses an internal `FixedVec<T, N>` (a tiny `Vec`-like type over
`[MaybeUninit<T>; N]`). In that configuration the structs are large, so place them
in a `static` (wrap init in your own `Once`-like guard) or a dedicated buffer.

```toml
[dependencies.opus-rs]
version = "0.1"
default-features = false
features = ["libm"]
```

The public API (`OpusEncoder`/`OpusDecoder` with `encode`/`decode` writing into caller-provided buffers) is identical in both configurations.

### Feature flags

| Feature  | Default | Description |
|----------|---------|-------------|
| `std`    | yes     | Enables OS-backed runtime x86 SIMD detection (CPUID). Without it the crate is `#![no_std]`. |
| `heap`   | yes     | Allocates the large codec state on the heap (`Box`), shrinking `OpusEncoder`/`OpusDecoder` to ~1 KB so they can live on ordinary stacks. Requires `std`. |
| `libm`   | no      | Required for `#![no_std]` builds — provides float math via a pure-Rust port of musl `libm`. Not needed when `std` is on. |

> Runtime dependency footprint: `std` build → **0 deps**; `no_std` build → **1 dep** (`libm`, pure Rust).

### Notes for `no_std` users

- **Large structs:** without `heap`, every working buffer is inline, so
  `OpusEncoder`/`OpusDecoder` are large (~250 KB / ~175 KB each). Place them in a
  `static` (wrap init in your own `Once`-like guard) or a dedicated buffer rather
  than on the stack. Test threads use a 16 MB stack (see `.cargo/config.toml`).
- **x86 SIMD:** without `std`, AVX/AVX2 dispatch falls back to compile-time detection. Build with `RUSTFLAGS="-C target-feature=+avx2"` to enable it. aarch64 NEON is unconditional.
- **Packets are capped** at the RFC 6716 maximum of 1276 bytes (the heap-free range-coder buffer is sized accordingly); standalone `RangeCoder`/CELT use supports up to 2048 bytes.
- **Verified targets:** `thumbv7em-none-eabi` (Cortex-M), `wasm32-unknown-unknown`, and any Linux no_std target. Check with `scripts/check_no_std.sh`.

## Performance

Criterion benchmark (`cargo bench --bench opus_vs_c_bench`) with 20 samples, 100 ms warm-up, 500 ms measurement, real speech input (`fixtures/answer_16k.wav`), mono encode-only. All numbers below are wall-clock time for the full frame set.

### vs C Opus (libopus 1.6.1) on x86-64 (AVX2/FMA)

Measured on AMD Ryzen 7 5700X, compiled with `--release` (opt-level=3 + ThinLTO).

| Config | Pure Rust | C Opus | Ratio |
|--------|-----------|--------|-------|
| 8 kHz / 20 ms VoIP | **39.9 ms** | 40.6 ms | 0.98× (**Rust 2% faster**) |
| 16 kHz / 20 ms VoIP | **66.8 ms** | 67.1 ms | 1.00× (**Rust 0.5% faster**) |
| 16 kHz / 10 ms VoIP | 73.2 ms | **72.5 ms** | 1.01× (within noise) |
| 48 kHz / 20 ms Audio | **25.1 ms** | 28.4 ms | 0.88× (**Rust 12% faster**) |
| 48 kHz / 10 ms Audio | **29.7 ms** | 31.2 ms | 0.95× (**Rust 5% faster**) |

### vs C Opus (libopus 1.6.1) on Apple Silicon

Measured on Apple Silicon M-series (aarch64), compiled with `--release` (opt-level=3 + ThinLTO), latest run on 2026-04-23.

| Config | Pure Rust | C Opus | Ratio |
|--------|-----------|--------|-------|
| 8 kHz / 20 ms VoIP | 31.47 ms | **31.20 ms** | 1.01× (C 0.9% faster) |
| 16 kHz / 20 ms VoIP | **51.19 ms** | 52.81 ms | 0.97× (**Rust 3.1% faster**) |
| 16 kHz / 10 ms VoIP | 55.69 ms | **55.49 ms** | 1.00× (within noise) |
| 48 kHz / 20 ms Audio | **13.97 ms** | 19.39 ms | 0.72× (**Rust 28% faster**) |
| 48 kHz / 10 ms Audio | **16.19 ms** | 20.28 ms | 0.80× (**Rust 20% faster**) |


## Release Notes

### 0.1.37

- **Fix: starved CBR budgets (issues #45, #46).** Budgets below 3 bytes per
  frame (and other starved budgets) now emit libopus's TOC-only "PLC" packet
  — padded under CBR, with the 40/60 ms multiframe TOC rules — instead of
  failing with a range-coder error (debug builds panicked in
  `RangeCoder::shrink`). Budgets below ~6 kb/s (9 kb/s above 20 ms) switch to
  CELT-only mode and code small valid packets, as libopus does
  (opus_encoder.c:1226-1286, 1525-1527). Known gap: at 5-6 byte CELT packets
  the payload's tail bits differ from libopus's, costing some SNR — tracked
  for follow-up.
- **Fix: 60 ms SILK packets cap the first two frames at 2/5 and 3/4 of the
  packet (enc_API.c parity, from PR #44).** Without the caps a 60 ms packet's
  early frames could leave the last one under its no-pulse fallback cost.
- **CELT transients are analysed on unfiltered history (PR #47).** The
  tone/transient analysis saw a prefiltered overlap head against an
  unfiltered body, reading a comb-filter step as a transient on steady
  pitched input; 48 kHz Audio at 20-24 kbps lost its first ~200 ms. Now
  lag 0, correlation 1.00 from the first window, and only the onset frame
  is coded with short blocks.

### Unreleased

- **Fix: CELT from 24 kHz input stuck at ~20 dB SNR (issue #54).** When
  `run_prefilter` cancelled the pitch pre-filter for a frame (libopus 1.6's
  `cancel_pitch`), it restored the unfiltered input over the whole frame.
  libopus still fades the previous frame's comb filter out over the overlap
  (celt_encoder.c:1572-1584), and the decoder's post-filter always applies
  that fade. So every cancel right after an "on" frame left 120 samples the
  decoder post-filtered but the encoder never pre-filtered. From 24 kHz the
  zero-stuffed input is dominated by its image above 12 kHz, so the filter
  was cancelled whenever the pitch period was odd, in 18 of 66 frames on the
  issue's chirp. The SNR was 20.1 / 20.9 / 21.1 dB at 32 / 64 / 128 kbps and
  is now 29.4 / 36.2 / 41.9 dB. libopus 1.6.1 scores 38.8 / 41.9 dB at 64 /
  128 kbps (35.2 / 41.1 with its tonality analysis off, which opus-rs does
  not port). The 48 kHz chirp packets are byte-identical.
- **Fix: CELT flagged transients on steady pitched input (issue #38).** The
  encoder ran `tone_detect` and `transient_analysis` on a buffer whose
  overlap head was the previous frame's *prefiltered* signal while the body
  was unfiltered; libopus 1.6 analyses the unfiltered `prefilter_mem` tail
  (celt_encoder.c:2017). Wherever the comb filter was active, the step at the
  head read as a transient: on the issue's chirp, 30 of 65 frames were coded
  with short blocks at every bitrate (libopus: only the onset frame). With
  bits to spare this was inaudible, but 48 kHz `Audio` at 20-24 kbps
  (CELT-only) lost its first ~200 ms (correlation 0.56-0.66); it is now lag 0
  and correlation 1.00 from the first window, like libopus's CELT.
  `prefilter_mem` is also kept current while the comb filter is off (stereo,
  hybrid, complexity < 5), as libopus's always-run `run_prefilter` does.

### 0.1.36

- **Fix: in-band FEC emitted no LBRR at CBR (issue #36).** The LBRR payload
  reused the main frame's quantized pulses with only the first gain index
  raised, so it cost as many bits as the main frame and never fit a CBR
  packet. `silk_LBRR_encode` is now ported faithfully: the excitation is
  re-quantized through the noise-shaping quantizer at the raised LBRR gain
  with its own NSQ state and gain chain, gated on speech activity, with
  `silk_setup_LBRR` (gain increase 7, then `max(7 − 0.2·loss, 2)`) and
  opus_encoder.c's `decide_fec` (rate thresholds + hysteresis + bandwidth
  reduction). CBR mono now recovers the previous packet through libopus
  `decode_fec` at 3.8 dB median (previously 0.0 dB: no LBRR at all).
- **Fix: 48 kHz SILK/hybrid encoder delay and startup (issue #38).** The
  encoder input was resampled with the stateless `down2`/`down2_3` decimators
  plus an internal-rate delay buffer, lagging libopus's pre-skip by up to 17
  samples. The encoder now runs the libopus `silk_resampler` (encoder
  direction: `delay_matrix_enc` + private down-FIR, ported with the
  `silk_Resampler_*_COEFS` tables) on the API-rate input, and prefills the
  SILK encoder on CELT→SILK restarts. Measured with the issue's harness:
  Voip 32 kbps lag −5..−2 (libopus's own encoder: identical), Audio 24 kbps
  startup correlation 0.56 → ≥ 0.77.
- **Fix: panics found by fuzzing.** 2.5 ms CELT frames aborted in the
  encoder's lm search (off-by-one: lm=0 was excluded from the valid set);
  2.5/5 ms frames reached the SILK encoder, whose VAD band split is only
  valid for ≥ 10 ms frames — SILK-only now rejects sub-10 ms frame sizes
  (they are CELT-only durations in libopus) and the VAD bails out on
  non-multiple-of-8 lengths.
- **no_std:** the crate builds and the full test suite runs with
  `--no-default-features --features libm` (`RUST_MIN_STACK=33554432`, see
  `scripts/check_no_std.sh`). 13 fuzz targets run clean.

### 0.1.35

- **Fix: SIGILL on AVX-without-FMA CPUs (issue #30, PR #31).** Nine kernels
  compiled with `#[target_feature(enable = "avx,fma")]` / `"avx2,fma"` were
  dispatched after checking only AVX/AVX2. New `compat::x86_has_avx_fma()` /
  `x86_has_avx2_fma()` probes gate them, so Sandy/Ivy Bridge and VMs that mask
  FMA no longer fault on the first `vfmadd`.
- **Fix: frame-loss concealment (issue #15, PR #32).** Packets of 0 or 1 bytes
  were decoded using the placeholder byte's ToC, which switched the decoder's
  mode/bandwidth/SILK rate and could panic in the resampler. Losses now conceal
  in the previous mode, as libopus does; an empty packet conceals `frame_size`
  samples instead of erroring, and a mono marker no longer errors on a stereo
  decoder.
- **Fix: in-band FEC LBRR desync (issue #27, PR #33).** LBRR frames were coded
  with `CODE_INDEPENDENTLY_NO_LTP_SCALING` while every decoder reads
  `CODE_INDEPENDENTLY`, so a voiced LBRR frame desynced the main frame that
  followed it. The LBRR gain bump now raises only index 0 of independently
  coded frames, as libopus does.
- **Fix: SILK decoder alignment with libopus (issue #34).** Mono SILK output
  read from `w_silk_buf[0][2..]` instead of libopus's `[1..]`, running one
  internal-rate sample early, and equal API/SILK-internal rates bypassed
  `silk_resampler`, dropping its `inputDelay`. Mono SILK now matches libopus at
  lag 0 at 16 kHz and 48 kHz.
- **Fix: 40/60 ms frames (issue #35).** Frame sizes were validated by
  `sampling_rate % frame_size == 0`, which rejected 60 ms (16.67 frames/s)
  before encoding. Validation is now duration-based. SILK-only carries a
  60 ms frame directly (3 internal SILK frames); CELT and Hybrid, which cap a
  frame at 20 ms, repacketize 20 ms sub-frames into one code 1/2/3 packet, as
  libopus does. This also fixes the code-3 VBR packet parser, which read frame
  lengths and frame data interleaved instead of all lengths first (libopus
  layout) — a latent bug that only surfaced once the encoder emitted code 3.
- **Fix: 24 kHz CELT decoded to garbage (issue #37).** The CELT encoder always
  coded all 21 bands, while the decoder derives its end band from the TOC
  (SWB = 19 at 24 kHz); coding one band more than the decoder reads
  desynchronised the range coder, turning every 24 kHz CELT packet into noise.
  The encoder now takes `end_band` from the Opus bandwidth
  (`CELT_SET_END_BAND`). It also zero-stuffs the API-rate input by
  `resampling_factor(Fs)` and scales/zeroes the MDCT output for non-48 kHz
  rates, as libopus `celt_preemphasis` / `compute_mdcts` do. 24 kHz `Audio`
  now matches libopus (per-window correlation ~1.0, was ~0.04).
- **Tests:** new libopus-oracle suites for SILK multi-frame/FEC (#27),
  frame-loss concealment (#15), mono alignment (#34), 40/60 ms frames (#35)
  and 24 kHz encoding (#37).

### 0.1.34

- **New: `OpusEncoder::encode_i16()` — native PCM16 entry point (issue #28).**
  Callers holding PCM16 audio no longer need to convert to f32 before encoding.
  In the SILK/VoIP path the integer high-pass biquad (`hp_cutoff_i16`) is fed
  directly from the input — no f32 round-trip, no `[i16; 11520]` temporary
  conversion buffer. Output is byte-for-byte identical to `encode()` called
  with the same audio converted via `sample as f32 / 32768.0` (SILK-only,
  Hybrid and CELT-only modes; verified by a parity test matrix across
  8/12/16/24/48 kHz, mono/stereo, complexity 0-10, CBR/VBR, in-band FEC).
  The float `encode()` API is unchanged.
- **Fix: `no_std` builds (`--no-default-features --features libm`) compile
  again.** The `multistream` decode module (added in 0.1.33) uses `Vec`-based
  plumbing and is now gated behind the default `std` feature; the core codec
  remains heap-free `no_std`.
- **Zero warnings:** `cargo check` and `cargo clippy --all-targets` are clean
  for the default and `no_std` feature sets (dead CELT state fields removed,
  tests/examples/benches lint-clean).

### 0.1.29

- **Default build is now heap-backed (issue #12).** A new `heap` feature (on by
  default, requires `std`) allocates the large codec state with `Box`, shrinking
  `OpusEncoder`/`OpusDecoder` from ~250 KB / ~175 KB to ~1 KB each. Constructing,
  encoding and decoding now fit on a 768 KiB stack (previously `OpusEncoder::new`
  alone needed ~850 KiB in release / ~2 MiB in debug, overflowing small-stack
  threads such as Windows' 1 MiB main thread). Disable default features for the
  heap-free `#![no_std]` build, which keeps the inline layout.
- **Tests:** add `tests/stack_usage_test.rs` asserting the structs stay <4 KiB
  with `heap`, >100 KiB without it, and that a full construct + encode + decode
  round-trip fits in a 768 KiB stack thread.

### 0.1.28

- **Fix: high-bitrate bitstream interop with libopus (issue #11).** The ported `BAND_ALLOCATION` table was missing one `200` in its final row (7 entries instead of 8), shifting the whole row. That row is only selected by the allocator at high bitrates, so encoded stereo streams above ~160 kbps decoded to garbage in libopus (and the crate decoder mis-decoded libopus streams at the same rates). Table now byte-identical to libopus.
- **Fix: `celt_pvq_u`/`celt_pvq_v` panic for `k >= 129`.** `compute_u`'s fixed-size buffer is now domain-checked (`k <= MAX_PVQ_K = 128`); out-of-domain calls return `u32::MAX` instead of panicking or silently wrapping.
- **Fix: silent u32 overflow in PVQ codebook sizes.** `compute_u`/`unext`/`celt_pvq_v` use saturating arithmetic, so an out-of-domain `(n, k)` can never produce a plausible-but-wrong codebook size.
- **Fix: `celt_pvq_u_lookup` row-extent aliasing.** Lookups whose `max(n, k)` exceeds a row's actual coverage now compute instead of reading the next row's block.
- **Fix: `OpusEncoder::encode` now returns an error when the range coder overflows its packet budget** (previously bytes were silently dropped).
- **Tests:** add full-domain PVQ table verification vs exact big-int recurrence, out-of-domain saturation checks, and a high-bitrate (128–192 kbps) stereo interop test that cross-checks against libopus.

### 0.1.27

- **`#![no_std]` with no `alloc`**: the crate is now fully heap-free — no global allocator is required. All `Vec`/`Box` working buffers were replaced with an internal `FixedVec<T, N>` (inline `[MaybeUninit<T>; N]`); `std::sync::LazyLock` was replaced with a heap-free `OnceCell`; no_std float math (`sin`/`cos`/`sqrt`/…) is routed through an optional `libm` feature (pure-Rust musl port).
  - Verified on `thumbv7em-none-eabi` (Cortex-M), `wasm32-unknown-unknown`, and Linux no_std. `scripts/check_no_std.sh` reproduces.
  - Feature flags: `std` (default) → 0 runtime deps; `default-features = false, features = ["libm"]` → 1 dep (`libm`).
  - Encoder/decoder structs are inline and large (~250 KB / ~175 KB); place them in a `static` or dedicated buffer on constrained targets.
- **Performance:** unchanged vs 0.1.26 (measured within ±0.5% on `opus_real`; no per-frame allocation overhead).
- **Conformance:** RFC 6716 1276-byte packet cap enforced on the Opus path; the standalone `RangeCoder`/CELT buffer supports up to 2048 bytes.
- `wav_test` example now tags output files by build (`_std` / `_nostd`) for A/B listening across build modes.

## License

See [COPYING](COPYING) for the original Opus license (BSD-3-Clause).

## Links

- **RustPBX**: <https://github.com/restsend/rustpbx>
- **RustRTC**: <https://github.com/restsend/rustrtc>
- **SIP Stack**: <https://github.com/restsend/rsipstack>
- **Rust Voice Agent**: <https://github.com/restsend/active-call>
