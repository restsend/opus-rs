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

    // C reference output for frame 1 (from opusic-sys v0.5.8 wrapping opus-1.5.2 FIXED_POINT):
    // 25 bytes when encoded with max_data_bytes=1275
    // When encoded with max_data_bytes=25, C produces:
    let c_reference = unsafe {
        let mut err: i32 = 0;
        let enc = opusic_sys::opus_encoder_create(
            8000, 1, opusic_sys::OPUS_APPLICATION_VOIP, &mut err,
        );
        assert_eq!(err, opusic_sys::OPUS_OK);
        opusic_sys::opus_encoder_ctl(enc, opusic_sys::OPUS_SET_BITRATE_REQUEST, 10000i32);
        opusic_sys::opus_encoder_ctl(enc, opusic_sys::OPUS_SET_VBR_REQUEST, 0i32);
        opusic_sys::opus_encoder_ctl(enc, opusic_sys::OPUS_SET_COMPLEXITY_REQUEST, 0i32);
        opusic_sys::opus_encoder_ctl(
            enc,
            opusic_sys::OPUS_SET_MAX_BANDWIDTH_REQUEST,
            opusic_sys::OPUS_BANDWIDTH_NARROWBAND as i32,
        );
        let mut buf = vec![0u8; 25];
        let n = opusic_sys::opus_encode_float(enc, input.as_ptr(), 160, buf.as_mut_ptr(), 25);
        assert!(n > 0, "C opus_encode_float failed: {}", n);
        let result = buf[..n as usize].to_vec();
        opusic_sys::opus_encoder_destroy(enc);
        result
    };

    println!("Rust output: {} bytes", bytes);
    println!("Rust hex: {}", hex::encode(&output[..bytes]));
    println!("C reference: {} bytes", c_reference.len());
    println!("C reference hex: {}", hex::encode(&c_reference));

    // Compare TOC byte
    if bytes > 0 {
        println!("Rust TOC: 0x{:02x}", output[0]);
        println!("C TOC: 0x{:02x}", c_reference[0]);
    }

    // Assert packet length matches
    assert_eq!(
        bytes,
        c_reference.len(),
        "Packet length mismatch: Rust={} C={}",
        bytes,
        c_reference.len()
    );

    // Find first difference byte
    let mut first_diff: Option<usize> = None;
    for i in 0..bytes {
        if output[i] != c_reference[i] {
            first_diff = Some(i);
            break;
        }
    }

    if let Some(d) = first_diff {
        // Report matching prefix length
        println!(
            "First byte difference at index {}: Rust=0x{:02x} C=0x{:02x}",
            d, output[d], c_reference[d]
        );
        println!("Matching prefix: {} bytes", d);
        // Assert bitstream equality — this will fail with a clear diagnostic
        assert_eq!(
            &output[..bytes],
            &c_reference[..],
            "Bitstream differs from C reference starting at byte {} (Rust=0x{:02x} C=0x{:02x}). \
             Matching prefix: {} bytes. \
             Rust: {} \
             C:    {}",
            d, output[d], c_reference[d], d,
            hex::encode(&output[..bytes]),
            hex::encode(&c_reference)
        );
    } else {
        println!("✓ BITSTREAM MATCHES C REFERENCE EXACTLY!");
    }
}

/// Test TOC byte structure is correct for SILK NB 20ms CBR
#[test]
fn test_silk_toc_byte_structure() {
    let sample_rate = 8000;
    let frame_size = 160;

    let mut encoder = OpusEncoder::new(sample_rate, 1, Application::Voip)
        .expect("Failed to create encoder");
    encoder.complexity = 0;
    encoder.bitrate_bps = 10000;
    encoder.use_cbr = true;

    let mut input = vec![0.0f32; frame_size];
    for i in 0..frame_size {
        input[i] = (2.0f32 * PI * 440.0f32 * i as f32 / 8000.0f32).sin();
    }

    let mut output = vec![0u8; 25];
    let bytes = encoder.encode(&input, frame_size, &mut output).expect("Encode failed");

    assert!(bytes >= 3, "Packet too short: {}", bytes);

    // TOC byte: SILK-only NB 20ms mono = 0x0b
    // bits[7:3] = config (NB 20ms = 0b00001)
    // bit[2]    = stereo = 0
    // bits[1:0] = frame_count_code = 3 (Code 3)
    assert_eq!(output[0], 0x0b, "TOC byte mismatch: got 0x{:02x}", output[0]);

    // Code 3 second byte: count byte, 1 frame, no padding = 0x01
    assert_eq!(output[1], 0x01, "Count byte mismatch: got 0x{:02x}", output[1]);
}

/// Test that multiple consecutive frames produce correct output sizes
#[test]
fn test_silk_multi_frame_sizes() {
    let sample_rate = 8000;
    let frame_size = 160;

    let mut encoder = OpusEncoder::new(sample_rate, 1, Application::Voip)
        .expect("Failed to create encoder");
    encoder.complexity = 0;
    encoder.bitrate_bps = 10000;
    encoder.use_cbr = true;

    for frame_idx in 0..5 {
        let mut input = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            let t = (frame_idx * frame_size + i) as f32 / sample_rate as f32;
            input[i] = (2.0f32 * PI * 440.0f32 * t).sin();
        }

        let mut output = vec![0u8; 25];
        let bytes = encoder.encode(&input, frame_size, &mut output).expect("Encode failed");

        assert!(bytes >= 3, "Frame {}: packet too short: {}", frame_idx, bytes);
        assert!(bytes <= 25, "Frame {}: packet too large: {}", frame_idx, bytes);
        println!("Frame {}: {} bytes, hex: {}", frame_idx, bytes, hex::encode(&output[..bytes]));
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
