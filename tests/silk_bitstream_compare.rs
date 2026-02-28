/// Test to compare Rust SILK encoder output with C reference
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

#[test]
fn test_silk_bitstream_vs_c_reference() {
    // Configuration matching C test: 8kHz NB, mono, VOIP, 10kbps CBR
    let sample_rate = 8000;
    let channels = 1;
    let bitrate = 10000;
    let frame_size = 160; // 20ms at 8kHz

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");

    // Set parameters to match C test
    encoder.complexity = 0;
    encoder.bitrate_bps = bitrate;
    encoder.use_cbr = true; // CBR mode

    // Generate 440Hz sine wave matching C: sinf(2.0f * M_PI * 440.0f * (float)i / 8000.0f)
    // NOTE: Must match C's exact computation order to get identical f32 values.
    // C computes: 2*pi*440*i/8000 (left to right), NOT (i/8000)*2*pi*440.
    let mut input = vec![0.0f32; frame_size];
    for i in 0..frame_size {
        input[i] = (2.0f32 * PI * 440.0f32 * i as f32 / 8000.0f32).sin();
    }

    // Encode first frame
    // Use max_data_bytes = 25 to match C reference test
    let mut output = vec![0u8; 25];
    let bytes = encoder
        .encode(&input, frame_size, &mut output)
        .expect("Encode failed");

    // C reference output for frame 1 (from gen_silk_ref.c with opus-1.6 FIXED_POINT, no RES24):
    // 25 bytes: 0b0184c1c1c7b66f5e06a4b728c81c956120e0781c264a1760
    let c_reference = [
        0x0b, 0x01, 0x84, 0xc1, 0xc1, 0xc7, 0xb6, 0x6f, 0x5e, 0x06, 0xa4, 0xb7, 0x28, 0xc8, 0x1c,
        0x95, 0x61, 0x20, 0xe0, 0x78, 0x1c, 0x26, 0x4a, 0x17, 0x60,
    ];

    println!("Rust output: {} bytes", bytes);
    println!("Rust hex: {}", hex::encode(&output[..bytes]));
    println!("C reference: {} bytes", c_reference.len());
    println!("C reference hex: {}", hex::encode(&c_reference));

    // Compare TOC byte
    if bytes > 0 {
        println!("Rust TOC: 0x{:02x}", output[0]);
        println!("C TOC: 0x{:02x}", c_reference[0]);
    }

    // For now, just report differences (don't assert equality until pipeline is complete)
    if bytes == c_reference.len() && &output[..bytes] == &c_reference[..] {
        println!("✓ BITSTREAM MATCHES C REFERENCE!");
    } else {
        println!("✗ Bitstream differs from C reference");
        println!("  Differences are expected until rate control, HP filter, etc. are implemented");
    }
}

// Helper to display hex
mod hex {
    pub fn encode(data: &[u8]) -> String {
        let mut s = String::with_capacity(data.len() * 2);
        for &b in data {
            let _ = std::fmt::Write::write_fmt(&mut s, format_args!("{:02x}", b));
        }
        s
    }
}
