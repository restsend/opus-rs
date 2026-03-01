# opus-rs

A pure-Rust implementation of the [Opus audio codec](https://opus-codec.org/) (RFC 6716), ported from the reference C implementation (libopus 1.6).

> **Status: Production-ready** — SILK-only, CELT-only, and Hybrid modes are functional. Stereo encoding for SILK is in progress.

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
- **Stereo** — mid-side encoding (CELT complete, SILK in progress)


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

## Roadmap

- [x] SILK-only encode & decode (NB/MB/WB)
- [x] CELT-only encode & decode (FB)
- [x] Range coder (entropy coding)
- [x] Rate control loop (CBR/VBR)
- [x] VAD & DTX framework
- [x] HP variable cutoff filter
- [x] Hybrid mode (SILK + CELT)
- [x] NLSF interpolation for multi-frame packets
- [x] High-quality resampler (up2, up2_hq) for decoder
- [x] LBRR (forward error correction)
- [x] Stereo encoding for CELT (mid-side)
- [ ] SILK bitstream bit-exact match with C reference (minor state differences, functionally correct)
- [ ] Stereo encoding for SILK (in progress)

## License

This project is a clean-room Rust port of the Opus reference implementation. See [COPYING](COPYING) for the original Opus license (BSD-3-Clause).

## Links

- **Repository**: <https://github.com/restsend/opus-rs>
- **Opus specification**: [RFC 6716](https://www.rfc-editor.org/rfc/rfc6716)
- **Reference C implementation**: <https://opus-codec.org/>
