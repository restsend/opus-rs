//! Fuzz test for packet parsing and TOC byte handling
//! Tests all possible TOC byte configurations and packet structures

#![no_main]

use libfuzzer_sys::fuzz_target;
use opus_rs::OpusDecoder;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Test all sample rates
    for &sampling_rate in &[8000, 12000, 16000, 24000, 48000] {
        for &channels in &[1, 2] {
            let mut decoder = match OpusDecoder::new(sampling_rate, channels) {
                Ok(d) => d,
                Err(_) => continue,
            };

            // Test all 256 possible TOC bytes
            for toc in 0u8..=255 {
                let mut packet = vec![toc];
                if data.len() > 1 {
                    packet.extend_from_slice(&data[1..]);
                }

                // Determine frame size from TOC for valid packets
                // TOC format: [config:5][s:1][c:2]
                // config = bits 3-7, s = bit 2 (stereo), c = bits 0-1 (channel config)

                let config = (toc >> 3) & 0x1F;
                let mode_flag = (toc >> 7) & 0x01;

                // Frame sizes based on mode
                let frame_sizes: Vec<usize> = if mode_flag == 1 {
                    // CELT mode: 2.5, 5, 10, 20 ms
                    let period = config & 0x03;
                    let frame_rate = 400 >> period;
                    if frame_rate == 0 || sampling_rate % frame_rate != 0 {
                        vec![120, 240, 480, 960] // Default CELT sizes
                    } else {
                        vec![(sampling_rate / frame_rate) as usize]
                    }
                } else if (toc >> 5) & 0x03 == 0x03 {
                    // Hybrid mode: 10 or 20 ms
                    vec![
                        sampling_rate as usize / 100,
                        sampling_rate as usize / 50,
                    ]
                } else {
                    // SILK mode: 10, 20, 40, 60 ms
                    let silk_config = (toc >> 3) & 0x03;
                    let ms = match silk_config {
                        0 => 10,
                        1 => 20,
                        2 => 40,
                        _ => 60,
                    };
                    vec![(sampling_rate as i64 * ms / 1000) as usize]
                };

                for frame_size in frame_sizes {
                    if frame_size == 0 || frame_size > 5760 {
                        continue;
                    }

                    let mut output = vec![0.0f32; frame_size * channels];
                    let _ = decoder.decode(&packet, frame_size, &mut output);
                }

                // Also test with some arbitrary frame sizes
                let arbitrary_sizes = [120, 240, 480, 960, 1920, 2880];
                for frame_size in arbitrary_sizes {
                    if frame_size > 0 {
                        let mut output = vec![0.0f32; frame_size * channels];
                        let _ = decoder.decode(&packet, frame_size, &mut output);
                    }
                }
            }

            // Test packets with various structures
            let packet_structures: Vec<Vec<u8>> = vec![
                // Single byte packet
                vec![data[0]],
                // TOC + minimal data
                vec![data[0], 0],
                // TOC + padding
                vec![data[0], 0, 0, 0, 0],
                // TOC + random data
                data.to_vec(),
                // Long packet
                vec![data[0]; 4000],
            ];

            for packet in &packet_structures {
                for &frame_size in &[120, 240, 480, 960] {
                    let mut output = vec![0.0f32; frame_size * channels];
                    let _ = decoder.decode(packet, frame_size, &mut output);
                }
            }
        }
    }

    // Test with mismatched channel configurations
    // Create mono decoder, test with stereo-like TOC
    if let Ok(mut mono_decoder) = OpusDecoder::new(48000, 1) {
        let stereo_tocs = [0x04, 0x0C, 0x14, 0x1C, 0x84, 0x8C, 0x94, 0x9C];
        for &toc in &stereo_tocs {
            let packet = vec![toc];
            for &frame_size in &[120, 480, 960] {
                let mut output = vec![0.0f32; frame_size];
                let _ = mono_decoder.decode(&packet, frame_size, &mut output);
            }
        }
    }

    // Create stereo decoder, test with mono-like TOC
    if let Ok(mut stereo_decoder) = OpusDecoder::new(48000, 2) {
        let mono_tocs = [0x00, 0x08, 0x10, 0x18, 0x80, 0x88, 0x90, 0x98];
        for &toc in &mono_tocs {
            let packet = vec![toc];
            for &frame_size in &[120, 480, 960] {
                let mut output = vec![0.0f32; frame_size * 2];
                let _ = stereo_decoder.decode(&packet, frame_size, &mut output);
            }
        }
    }
});
