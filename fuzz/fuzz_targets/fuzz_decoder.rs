//! Fuzz test for OpusDecoder
//! Tests decoding with arbitrary byte patterns to detect overflows and panics

#![no_main]

use libfuzzer_sys::fuzz_target;
use opus_rs::OpusDecoder;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // Parse TOC byte from first byte
    let toc = data[0];

    // Determine channels from TOC
    let channels = if toc & 0x04 != 0 { 2 } else { 1 };

    // Determine sampling rate (use 48kHz for maximum compatibility)
    let sampling_rate = 48000;

    // Create decoder
    let mut decoder = match OpusDecoder::new(sampling_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Determine frame size based on TOC
    let mode = if toc & 0x80 != 0 {
        2 // CeltOnly
    } else if toc & 0x60 == 0x60 {
        1 // Hybrid
    } else {
        0 // SilkOnly
    };

    let frame_size = match mode {
        0 => {
            // SILK mode: 10, 20, 40, or 60 ms
            let config = (toc >> 3) & 0x03;
            let ms = match config {
                0 => 10,
                1 => 20,
                2 => 40,
                3 => 60,
                _ => 20,
            };
            (sampling_rate as i64 * ms / 1000) as usize
        }
        1 => {
            // Hybrid: 10 or 20 ms
            let config = (toc >> 3) & 0x01;
            let ms = if config == 0 { 10 } else { 20 };
            (sampling_rate as i64 * ms / 1000) as usize
        }
        _ => {
            // CELT: 2.5, 5, 10, or 20 ms
            let config = (toc >> 3) & 0x03;
            let ms = match config {
                0 => 2.5,
                1 => 5.0,
                2 => 10.0,
                3 => 20.0,
                _ => 20.0,
            };
            (sampling_rate as f64 * ms / 1000.0) as usize
        }
    };

    // Clamp frame_size to reasonable bounds
    let frame_size = frame_size.clamp(80, 5760);

    // Allocate output buffer
    let mut output = vec![0.0f32; frame_size * channels];

    // Try to decode (may fail with invalid data, that's OK)
    let _ = decoder.decode(data, frame_size, &mut output);
});
