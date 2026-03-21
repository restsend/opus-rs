use std::f32::consts::PI;

const FRAME_SIZE: usize = 160;
const SAMPLE_RATE: usize = 8000;

/// Generate 440Hz sine wave
fn generate_input() -> Vec<f32> {
    (0..FRAME_SIZE)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
        .collect()
}

/// Convert f32 to i16 (matching what C encoder expects)
fn to_i16(samples: &[f32]) -> Vec<i16> {
    samples.iter().map(|&x| (x * 32767.0) as i16).collect()
}

/// Hex encode
fn hex_encode(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len() * 2);
    for &b in data {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[test]
fn test_encoder_bitstream_comparison() {
    println!("\n========================================");
    println!("SILK Encoder Bitstream Comparison");
    println!("========================================\n");

    let input = generate_input();
    let input_i16 = to_i16(&input);

    println!("Input: {} samples", input.len());
    println!("Input[0..10] = {:?}", &input[0..10]);
    println!("Input_i16[0..10] = {:?}", &input_i16[0..10]);

    // ========================================
    // 1. Encode with C FIXED_POINT encoder
    // ========================================
    let c_encoder_ptr: *mut opusic_sys::OpusEncoder = unsafe {
        let mut error = 0i32;
        let ptr = opusic_sys::opus_encoder_create(
            SAMPLE_RATE as i32,
            1,
            opusic_sys::OPUS_APPLICATION_VOIP as i32,
            &mut error,
        );
        if error != 0 {
            panic!("Failed to create C encoder: error {}", error);
        }
        ptr
    };

    unsafe {
        opusic_sys::opus_encoder_ctl(c_encoder_ptr, opusic_sys::OPUS_SET_BITRATE_REQUEST, 10000);
        opusic_sys::opus_encoder_ctl(c_encoder_ptr, opusic_sys::OPUS_SET_COMPLEXITY_REQUEST, 0);
        opusic_sys::opus_encoder_ctl(
            c_encoder_ptr,
            opusic_sys::OPUS_SET_VBR_REQUEST,
            0, // CBR
        );
    }

    let mut c_packet = vec![0u8; 400];
    let c_bytes = unsafe {
        opusic_sys::opus_encode(
            c_encoder_ptr,
            input_i16.as_ptr(),
            FRAME_SIZE as i32,
            c_packet.as_mut_ptr(),
            c_packet.len() as i32,
        )
    };

    unsafe {
        opusic_sys::opus_encoder_destroy(c_encoder_ptr);
    }

    if c_bytes < 0 {
        panic!("C encode failed: {}", c_bytes);
    }
    c_packet.truncate(c_bytes as usize);

    println!("\n--- C FIXED_POINT Encoder ---");
    println!("Output: {} bytes", c_bytes);
    println!("Hex: {}", hex_encode(&c_packet));

    // ========================================
    // 2. Encode with Rust encoder
    // ========================================
    let mut rust_encoder =
        opus_rs::OpusEncoder::new(SAMPLE_RATE as i32, 1, opus_rs::Application::Voip)
            .expect("Failed to create Rust encoder");
    rust_encoder.bitrate_bps = 10000;
    rust_encoder.complexity = 0;
    rust_encoder.use_cbr = true;

    let mut rust_packet = vec![0u8; 400];
    let rust_bytes = rust_encoder
        .encode(&input, FRAME_SIZE, &mut rust_packet)
        .expect("Rust encode failed");
    rust_packet.truncate(rust_bytes);

    println!("\n--- Rust FIXED_POINT Encoder ---");
    println!("Output: {} bytes", rust_bytes);
    println!("Hex: {}", hex_encode(&rust_packet));

    // ========================================
    // 3. Compare byte by byte
    // ========================================
    println!("\n--- Byte Comparison ---");
    let c_bytes_len = c_bytes as usize;
    let compare_len = c_bytes_len.min(rust_bytes);

    let mut first_diff_idx: Option<usize> = None;
    for i in 0..compare_len {
        if c_packet[i] != rust_packet[i] {
            first_diff_idx = Some(i);
            println!(
                "DIFF at byte {}: C={:02x} vs Rust={:02x}",
                i, c_packet[i], rust_packet[i]
            );
            // Show context
            let start = i.saturating_sub(2);
            let end = (i + 3).min(compare_len);
            println!(
                "  C bytes {}..{}: {:02x?}",
                start,
                end,
                &c_packet[start..end]
            );
            println!(
                "  Rust bytes {}..{}: {:02x?}",
                start,
                end,
                &rust_packet[start..end]
            );
            break;
        }
    }

    if let Some(idx) = first_diff_idx {
        println!("\n*** First difference at byte {} ***", idx);

        // If first diff is at byte 4+, it means the first 4 bytes (TOC + count) match
        // Let's analyze what's encoded in those first bytes
        if idx >= 4 {
            println!("\nBytes 0-3 match: TOC + count bytes identical");
            println!("Difference starts in the frame payload");
        }
    } else {
        println!("Packets are identical!");
    }
}

#[test]
fn test_multi_frame_comparison() {
    println!("\n========================================");
    println!("Multi-frame Encoder Comparison");
    println!("========================================\n");

    let n_frames = 5;
    let mut c_all_packets = Vec::new();
    let mut rust_all_packets = Vec::new();

    // C encoder
    let c_encoder_ptr: *mut opusic_sys::OpusEncoder = unsafe {
        let mut error = 0i32;
        let ptr = opusic_sys::opus_encoder_create(
            SAMPLE_RATE as i32,
            1,
            opusic_sys::OPUS_APPLICATION_VOIP as i32,
            &mut error,
        );
        if error != 0 {
            panic!("Failed to create C encoder: error {}", error);
        }
        ptr
    };

    unsafe {
        opusic_sys::opus_encoder_ctl(c_encoder_ptr, opusic_sys::OPUS_SET_BITRATE_REQUEST, 10000);
        opusic_sys::opus_encoder_ctl(c_encoder_ptr, opusic_sys::OPUS_SET_COMPLEXITY_REQUEST, 0);
        opusic_sys::opus_encoder_ctl(c_encoder_ptr, opusic_sys::OPUS_SET_VBR_REQUEST, 0);
    }

    // Rust encoder
    let mut rust_encoder =
        opus_rs::OpusEncoder::new(SAMPLE_RATE as i32, 1, opus_rs::Application::Voip)
            .expect("Failed to create Rust encoder");
    rust_encoder.bitrate_bps = 10000;
    rust_encoder.complexity = 0;
    rust_encoder.use_cbr = true;

    for frame_idx in 0..n_frames {
        // Generate input for this frame
        let t_offset = frame_idx * FRAME_SIZE;
        let input: Vec<f32> = (0..FRAME_SIZE)
            .map(|i| (2.0 * PI * 440.0 * (t_offset + i) as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let input_i16 = to_i16(&input);

        // C encode
        let mut c_packet = vec![0u8; 400];
        let c_bytes = unsafe {
            opusic_sys::opus_encode(
                c_encoder_ptr,
                input_i16.as_ptr(),
                FRAME_SIZE as i32,
                c_packet.as_mut_ptr(),
                c_packet.len() as i32,
            )
        };
        c_packet.truncate(c_bytes as usize);
        c_all_packets.push(c_packet);

        // Rust encode
        let mut rust_packet = vec![0u8; 400];
        let rust_bytes = rust_encoder
            .encode(&input, FRAME_SIZE, &mut rust_packet)
            .expect("Rust encode failed");
        rust_packet.truncate(rust_bytes);
        rust_all_packets.push(rust_packet);

        println!("Frame {}: C={}B, Rust={}B", frame_idx, c_bytes, rust_bytes);
        if &c_all_packets[frame_idx] != &rust_all_packets[frame_idx] {
            println!("  C hex: {}", hex_encode(&c_all_packets[frame_idx]));
            println!("  Rust hex: {}", hex_encode(&rust_all_packets[frame_idx]));
        } else {
            println!("  IDENTICAL");
        }
    }

    unsafe {
        opusic_sys::opus_encoder_destroy(c_encoder_ptr);
    }
}
