//! Fuzz test for SILK decoder
//! Tests SILK decoding with various frame sizes and packet configurations

#![no_main]

use libfuzzer_sys::fuzz_target;
use opus_rs::OpusDecoder;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // SILK supports 8, 12, 16, and 24 kHz (narrowband to wideband)
    let sampling_rates = [8000, 12000, 16000, 24000, 48000];

    for &sampling_rate in &sampling_rates {
        let channels = if data[0] % 2 == 0 { 1 } else { 2 };

        let mut decoder = match OpusDecoder::new(sampling_rate, channels) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // SILK frame sizes: 10, 20, 40, 60 ms
        let frame_sizes: Vec<usize> = vec![
            (sampling_rate as i64 * 10 / 1000) as usize,  // 10ms
            (sampling_rate as i64 * 20 / 1000) as usize,  // 20ms
            (sampling_rate as i64 * 40 / 1000) as usize,  // 40ms
            (sampling_rate as i64 * 60 / 1000) as usize,  // 60ms
        ];

        for frame_size in frame_sizes {
            if frame_size == 0 {
                continue;
            }

            let mut output = vec![0.0f32; frame_size * channels];

            // Test with raw data
            let _ = decoder.decode(data, frame_size, &mut output);

            // Test with SILK TOC bytes
            // SILK mode TOC: bits 5-6 = 0, bit 7 = 0
            // Frame config in bits 3-4
            let silk_tocs = [
                0x00, // SILK 10ms
                0x08, // SILK 20ms
                0x10, // SILK 40ms
                0x18, // SILK 60ms
            ];

            for &toc in &silk_tocs {
                let toc_with_channels = toc | if channels == 2 { 0x04 } else { 0x00 };
                let mut packet = vec![toc_with_channels];
                packet.extend_from_slice(&data[1..]);

                let _ = decoder.decode(&packet, frame_size, &mut output);
            }
        }

        // Test boundary frame sizes for SILK
        let boundary_sizes: Vec<usize> = vec![
            sampling_rate as usize / 100,      // 10ms
            sampling_rate as usize / 50,       // 20ms
            sampling_rate as usize / 25,       // 40ms
            sampling_rate as usize * 3 / 50,   // 60ms
            sampling_rate as usize / 25 + 1,   // Just over 40ms
            sampling_rate as usize * 3 / 50 + 1, // Just over 60ms
        ];

        for frame_size in boundary_sizes {
            if frame_size == 0 {
                continue;
            }
            let mut output = vec![0.0f32; frame_size * channels];
            let _ = decoder.decode(data, frame_size, &mut output);
        }
    }
});
