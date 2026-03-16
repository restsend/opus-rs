# opus-rs

A pure-Rust implementation of the [Opus audio codec](https://opus-codec.org/) (RFC 6716), ported from the reference C implementation (libopus 1.6).

> **Status: Production-ready** — SILK-only, CELT-only, and Hybrid modes are functional. Stereo encoding (SILK and CELT) is supported.

## Features

- **Pure Rust** — no C dependencies, no unsafe code in the codec core
- **SILK encoder & decoder** — narrowband (8 kHz), mediumband (12 kHz), wideband (16 kHz)
- **CELT encoder & decoder** — fullband (48 kHz) with MDCT, PVQ, energy quantization
- **Hybrid mode** — SILK for low frequencies + CELT for high frequencies
- **Range coder** — entropy coding with ICDF tables and Laplace distribution
- **VAD** — voice activity detection
- **HP filter** — variable-cutoff high-pass filter for VOIP mode
- **CBR / VBR** — both constant and variable bitrate modes
- **LBRR** — in-band forward error correction
- **Resampler** — high-quality resampling (up2, up2_hq)
- **Stereo** — mid-side encoding for both SILK and CELT


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

## Testing

```bash
cargo test
```

All 156 tests pass, covering MDCT identity, PVQ consistency, SILK/CELT/Hybrid encode/decode roundtrip, resampler tests, and more.

### WAV Roundtrip

```bash
# Rust encoder/decoder
cargo run --example wav_test

# Compare with C libopus (requires opusic-sys)
cargo run --example wav_test_c
```

### Stereo Tests

```bash
cargo run --example stereo_test
```

## Performance

Run all benchmarks:

```bash
cargo bench
```

Run a specific benchmark:

```bash
cargo bench -- silk_pitch_analysis_core  # Pitch analysis only
cargo bench -- silk_nsq               # Noise shape quantizer
cargo bench -- silk_burg_modified_fix  # LPC analysis
```

### SILK Encoder (Rust, complexity=0)

| Sample Rate | Frame Size | Time per Frame | Throughput |
|-------------|------------|-----------------|------------|
| 8 kHz       | 20 ms      | 13.0 µs        | 23.5 MiB/s |
| 16 kHz      | 20 ms      | 24.3 µs        | 25.1 MiB/s |
| 16 kHz      | 10 ms      | 13.6 µs        | 22.4 MiB/s |

### SILK vs C Reference (libopus)

| Config              | 8kHz/20ms | 16kHz/20ms | 16kHz/10ms |
|---------------------|-----------|-------------|------------|
| Rust (cx0)          | 13.0 µs  | 24.3 µs    | 13.6 µs   |
| C libopus (cx0)     | 10.8 µs  | 18.3 µs    | 11.2 µs   |
| C libopus (cx9)     | 62.2 µs  | 132.4 µs   | 66.6 µs   |

Rust implementation uses complexity=0 (fast mode). Performance is comparable to C at the same complexity level. C at complexity=9 (default quality) is 5-6x slower.

### SILK Core Algorithms

| Function               | Time (16kHz WB) | Time (8kHz NB) |
|-----------------------|-----------------|---------------|
| burg_modified_fix     | 2.1 µs         | 0.93 µs       |
| autocorrelation       | 1.8 µs         | 0.4 µs        |
| inner product         | 0.1 µs         | 0.1 µs        |
| pitch_analysis_core  | 7.4 µs         | 3.1 µs        |
| nsq (voiced)          | 8.1 µs         | 4.2 µs        |
| nsq (unvoiced)        | 6.8 µs         | 3.3 µs        |

## License

See [COPYING](COPYING) for the original Opus license (BSD-3-Clause).

## Links

- **RustPBX**: <https://github.com/restsend/rustpbx>
- **RustRTC**: <https://github.com/restsend/rustrtc>
- **SIP Stack**: <https://github.com/restsend/rsipstack>
- **Rust Voice Agent**: <https://github.com/restsend/active-call>
