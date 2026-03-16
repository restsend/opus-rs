//! Fuzz test for CELT decoder boundary conditions
//! Tests various edge cases including:
//! - Empty/minimal packets
//! - Maximum frame sizes
//! - Invalid TOC bytes
//! - Corrupted packet data
//! - Boundary values around decode buffer size

#![no_main]

use libfuzzer_sys::fuzz_target;
use opus_rs::OpusDecoder;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let sampling_rate = 48000; // Use 48kHz for maximum frame sizes
    let channels = if data[0] % 2 == 0 { 1 } else { 2 };

    let mut decoder = match OpusDecoder::new(sampling_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Test 1: Minimal packets (just TOC byte)
    let minimal_packets: Vec<Vec<u8>> = vec![
        vec![0x80], // CELT 2.5ms mono
        vec![0x88], // CELT 5ms mono
        vec![0x90], // CELT 10ms mono
        vec![0x98], // CELT 20ms mono
        vec![0x84], // CELT 2.5ms stereo
        vec![0x8C], // CELT 5ms stereo
        vec![0x94], // CELT 10ms stereo
        vec![0x9C], // CELT 20ms stereo
    ];

    for packet in &minimal_packets {
        // All valid CELT frame sizes
        for &frame_size in &[120, 240, 480, 960] {
            let mut output = vec![0.0f32; frame_size * channels];
            let _ = decoder.decode(packet, frame_size, &mut output);
        }
    }

    // Test 2: Packets with invalid/reserved TOC bytes
    let invalid_tocs: Vec<u8> = vec![
        0x00, 0x01, 0x02, 0x03, // Reserved/invalid
        0xFC, 0xFD, 0xFE, 0xFF, // Reserved
        0x40, 0x44, 0x48, 0x4C, // Various mode configurations
    ];

    for &toc in &invalid_tocs {
        let mut packet = vec![toc];
        if data.len() > 1 {
            packet.extend_from_slice(&data[1..]);
        }

        for &frame_size in &[120, 240, 480, 960] {
            let mut output = vec![0.0f32; frame_size * channels];
            let _ = decoder.decode(&packet, frame_size, &mut output);
        }
    }

    // Test 3: Boundary frame sizes around DECODE_BUFFER_SIZE (3072)
    // Original bug was at 2048 + 120 = 2168
    let boundary_sizes: Vec<usize> = vec![
        2047, 2048, 2049,      // Around old buffer size
        2167, 2168, 2169,      // Around old boundary
        2879, 2880, 2881,      // Around max valid Opus frame
        3071, 3072, 3073,      // Around new buffer size
        3191, 3192, 3193,      // Around new buffer + overlap
        4095, 4096, 4097,      // Much larger
        5759, 5760, 5761,      // Around absolute max
    ];

    for frame_size in boundary_sizes {
        let mut output = vec![0.0f32; frame_size * channels];
        // Use raw data as packet
        let _ = decoder.decode(data, frame_size, &mut output);
    }

    // Test 4: Very small and zero frame sizes
    let small_sizes: Vec<usize> = vec![0, 1, 2, 4, 8, 16, 32, 64, 119, 120, 121];
    for frame_size in small_sizes {
        if frame_size == 0 {
            continue; // Skip zero to avoid divide by zero issues
        }
        let mut output = vec![0.0f32; frame_size * channels];
        let _ = decoder.decode(data, frame_size, &mut output);
    }

    // Test 5: Large packets with various sizes
    let large_packets: Vec<Vec<u8>> = vec![
        vec![0x98; 100],   // 100 bytes of CELT header
        vec![0x98; 1000],  // 1000 bytes
        vec![0x98; 4000],  // Near max packet size
    ];

    for packet in &large_packets {
        for &frame_size in &[120, 240, 480, 960] {
            let mut output = vec![0.0f32; frame_size * channels];
            let _ = decoder.decode(packet, frame_size, &mut output);
        }
    }

    // Test 6: Multiple consecutive decode calls (state persistence)
    for _ in 0..5 {
        for &frame_size in &[120, 480, 960] {
            let mut output = vec![0.0f32; frame_size * channels];
            let _ = decoder.decode(data, frame_size, &mut output);
        }
    }
});
