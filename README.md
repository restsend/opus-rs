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

## Performance

### vs C Opus (libopus 1.6.1) on Apple Silicon

Latest measurements on Apple Silicon M-series (aarch64), compiled with `-O3` + ThinLTO:

#### Encode + Decode Roundtrip (20ms frames, mono)

| Mode | Sample Rate | Application | Time/Frame | Throughput |
|------|-------------|-------------|------------|------------|
| **VoIP** | 8 kHz | Narrowband | **~17 µs** | 58k+ frames/sec |
| **VoIP** | 16 kHz | Wideband | **~30 µs** | 33k+ frames/sec |
| **Audio** | 48 kHz | Fullband | **~126 µs** | 7.9k+ frames/sec |

#### Encoder Performance (20ms frames)

| Mode | Sample Rate | Channels | C Opus | Pure Rust | Ratio | Codec |
|------|-------------|----------|--------|-----------|-------|-------|
| **VoIP** | 16 kHz | mono | ~70 µs | **~26 µs** | **0.37×** | SILK |
| **Audio** | 16 kHz | mono | - | **~21 µs** | - | SILK |
| **VoIP** | 48 kHz | mono | ~14 µs | **~82 µs** | **5.9×** | CELT/Hybrid |
| **Audio** | 48 kHz | mono | - | **~82 µs** | - | CELT |
| **VoIP** | 48 kHz | stereo | - | **~136 µs** | - | CELT |
| **Audio** | 48 kHz | stereo | - | **~135 µs** | - | CELT |

#### Decoder Performance (20ms frames)

| Mode | Sample Rate | Channels | C Opus | Pure Rust | Ratio | Codec |
|------|-------------|----------|--------|-----------|-------|-------|
| **VoIP** | 16 kHz | mono | ~4.4 µs | **~6.5 µs** | **1.48×** | SILK |
| **VoIP/Audio** | 48 kHz | mono | - | **~43 µs** | - | CELT |
| **VoIP/Audio** | 48 kHz | stereo | ~27 µs | **~74 µs** | **2.74×** | CELT |

**Summary:**
- ✅ **SILK encoding (16 kHz VoIP)**: Pure Rust is **2.7× faster** than C
- ⚠️ **CELT encoding (48 kHz)**: ~6× slower than C (optimization in progress)
- ⚠️ **SILK decoding**: **1.5× slower** than C (very close!)
- ❌ **CELT decoding**: **2.7× slower** than C (needs optimization)

## License

See [COPYING](COPYING) for the original Opus license (BSD-3-Clause).

## Links

- **RustPBX**: <https://github.com/restsend/rustpbx>
- **RustRTC**: <https://github.com/restsend/rustrtc>
- **SIP Stack**: <https://github.com/restsend/rsipstack>
- **Rust Voice Agent**: <https://github.com/restsend/active-call>
