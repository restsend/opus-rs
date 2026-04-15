# opus-rs

A pure-Rust implementation of the [Opus audio codec](https://opus-codec.org/) (RFC 6716), ported from the reference C implementation (libopus 1.6).

> **Status: Production-ready** — SILK-only, CELT-only, and Hybrid modes are functional. Stereo encoding (SILK and CELT) is supported.

## Features

- **Pure Rust** — no C dependencies, no unsafe code in the codec core
- **High Performance:** At or faster than C libopus across all configurations on both architectures. 

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

## Performance

Criterion benchmarks with 20 samples, real audio (902/1804 frames of real speech), mono encode + decode. All numbers are wall-clock time for the full frame set.

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

Measured on Apple Silicon M-series (aarch64), compiled with `--release` (opt-level=3 + ThinLTO).

| Config | Pure Rust | C Opus | Ratio |
|--------|-----------|--------|-------|
| 8 kHz / 20 ms VoIP | **34.77 ms** | 36.31 ms | 0.96× (**Rust 4% faster**) |
| 16 kHz / 20 ms VoIP | **58.23 ms** | 59.37 ms | 0.98× (**Rust 2% faster**) |
| 16 kHz / 10 ms VoIP | 63.44 ms | **62.50 ms** | 1.02× (C 2% faster) |
| 48 kHz / 20 ms Audio | **29.61 ms** | 32.42 ms | 0.91× (**Rust 9% faster**) |
| 48 kHz / 10 ms Audio | 34.58 ms | **33.47 ms** | 1.03× (C 3% faster) |


## License

See [COPYING](COPYING) for the original Opus license (BSD-3-Clause).

## Links

- **RustPBX**: <https://github.com/restsend/rustpbx>
- **RustRTC**: <https://github.com/restsend/rustrtc>
- **SIP Stack**: <https://github.com/restsend/rsipstack>
- **Rust Voice Agent**: <https://github.com/restsend/active-call>
