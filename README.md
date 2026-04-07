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

All 170+ tests pass, covering MDCT identity, PVQ consistency, SILK/CELT/Hybrid encode/decode roundtrip, resampler tests, and more.

### WAV Roundtrip

```bash
# Rust encoder/decoder
cargo run --example wav_test
```

### Stereo Tests

```bash
cargo run --example stereo_test
```

## Performance: Full Opus Encoder + Decoder Roundtrip (complexity=0)

Latest measurements are from 2026-04-07 (`cargo bench --bench opus_bench -- opus_vs_c --measurement-time 1`).

| Config           | Rust      | C (opus-sys) | C faster by |
|------------------|-----------|--------------|-------------|
| 8kHz/20ms VoIP   | 16.68 µs  | 13.35 µs     | 1.25×       |
| 16kHz/20ms VoIP  | 29.33 µs  | 21.21 µs     | 1.38×       |
| 16kHz/10ms VoIP  | 15.87 µs  | 13.01 µs     | 1.22×       |
| 48kHz/20ms Audio | 126.21 µs | 24.35 µs     | 5.18×       |
| 48kHz/10ms Audio | 70.19 µs  | 13.09 µs     | 5.36×       |

SILK (VoIP) is ~1.2-1.4× slower than C; CELT-heavy 48k Audio is ~5.2-5.4× slower.

## Performance: 48k Audio Cost Split (encode/decode/roundtrip)

Latest measurements are from 2026-04-07 (`cargo bench --bench opus_bench -- opus_audio_split_vs_c --measurement-time 1`).

| Path @ 48k       | Rust      | C (opus-sys) | C faster by |
|------------------|-----------|--------------|-------------|
| 20ms encode      | 79.11 µs  | 14.75 µs     | 5.36×       |
| 20ms decode      | 24.26 µs  | 9.50 µs      | 2.55×       |
| 20ms roundtrip   | 126.21 µs | 24.35 µs     | 5.18×       |
| 10ms encode      | 46.21 µs  | 7.93 µs      | 5.83×       |
| 10ms decode      | 14.88 µs  | 4.86 µs      | 3.06×       |
| 10ms roundtrip   | 70.19 µs  | 13.09 µs     | 5.36×       |


## License

See [COPYING](COPYING) for the original Opus license (BSD-3-Clause).

## Links

- **RustPBX**: <https://github.com/restsend/rustpbx>
- **RustRTC**: <https://github.com/restsend/rustrtc>
- **SIP Stack**: <https://github.com/restsend/rsipstack>
- **Rust Voice Agent**: <https://github.com/restsend/active-call>
