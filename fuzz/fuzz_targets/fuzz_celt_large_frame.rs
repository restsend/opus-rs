//! Fuzz test for CELT decoder with large frame sizes
//! Specifically tests the integer underflow bug fix in celt.rs:1454
//! where frame_size > decode_buffer_size + overlap caused panic

#![no_main]

use libfuzzer_sys::fuzz_target;
use opus_rs::OpusDecoder;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    // Focus on 48kHz which gives the largest frame sizes
    let sampling_rate = 48000;
    let channels = if data[0] % 2 == 0 { 1 } else { 2 };

    // Create decoder
    let mut decoder = match OpusDecoder::new(sampling_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Test valid CELT frame sizes at 48kHz
    // CELT supports: 2.5ms, 5ms, 10ms, 20ms (120, 240, 480, 960 samples @ 48kHz)
    // Note: 40ms and 60ms are SILK-only frame sizes, not CELT
    let valid_celt_frame_sizes: Vec<usize> = match data[1] % 4 {
        0 => vec![120],           // 2.5ms - smallest
        1 => vec![240],           // 5ms
        2 => vec![480],           // 10ms - common
        _ => vec![960],           // 20ms - most common
    };

    for frame_size in valid_celt_frame_sizes {
        // Allocate output buffer
        let mut output = vec![0.0f32; frame_size * channels];

        // Try to decode with various data patterns
        // The data is arbitrary bytes, simulating corrupted/malformed packets
        let _ = decoder.decode(data, frame_size, &mut output);

        // Also test with CELT-specific TOC byte patterns
        // CELT mode TOC: top bit set (0x80)
        // Frame size config in bits 3-4
        let toc_configs = [
            0x80 | (0 << 3), // CELT, 2.5ms
            0x80 | (1 << 3), // CELT, 5ms
            0x80 | (2 << 3), // CELT, 10ms
            0x80 | (3 << 3), // CELT, 20ms
        ];

        for toc in toc_configs {
            let mut packet = vec![toc];
            packet.extend_from_slice(&data[2..]);

            // Reset decoder for each attempt
            let mut decoder2 = match OpusDecoder::new(sampling_rate, channels) {
                Ok(d) => d,
                Err(_) => continue,
            };

            let _ = decoder2.decode(&packet, frame_size, &mut output);
        }
    }

    // Test edge cases around the boundary (invalid frame sizes but should not panic)
    // decode_buffer_size = 3072, overlap = 120
    // Original bug: frame_size > 2048 + 120 = 2168 caused underflow
    // Now: frame_size > 3072 + 120 should be handled gracefully
    let edge_case_sizes = [
        2048,           // Exactly old buffer size
        2168,           // Old boundary (2048 + 120)
        2169,           // Just over old boundary
        3072,           // New buffer size
        3192,           // New buffer size + overlap (3072 + 120)
    ];

    for frame_size in edge_case_sizes {
        let mut output = vec![0.0f32; frame_size * channels];
        // These are invalid frame sizes, decode should handle gracefully
        let _ = decoder.decode(data, frame_size, &mut output);
    }
});
