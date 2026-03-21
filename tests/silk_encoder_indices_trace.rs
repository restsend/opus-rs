use std::f32::consts::PI;

const FRAME_SIZE: usize = 160;
const SAMPLE_RATE: usize = 8000;

/// Generate 440Hz sine wave
fn generate_input() -> Vec<f32> {
    (0..FRAME_SIZE)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin())
        .collect()
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
fn test_trace_encoder_indices() {
    println!("\n========================================");
    println!("SILK Encoder Indices Tracing");
    println!("========================================\n");

    let input = generate_input();

    // Use Rust encoder with max complexity to ensure all features are used
    let mut rust_encoder =
        opus_rs::OpusEncoder::new(SAMPLE_RATE as i32, 1, opus_rs::Application::Voip)
            .expect("Failed to create Rust encoder");
    rust_encoder.bitrate_bps = 10000;
    rust_encoder.complexity = 0;
    rust_encoder.use_cbr = true;

    // Encode and get the packet
    let mut rust_packet = vec![0u8; 400];
    let rust_bytes = rust_encoder
        .encode(&input, FRAME_SIZE, &mut rust_packet)
        .expect("Rust encode failed");
    rust_packet.truncate(rust_bytes);

    println!("Rust packet: {} bytes", rust_bytes);
    println!("Hex: {}", hex_encode(&rust_packet));

    // Also encode with C for comparison
    // Use 32768.0 scaling to match libopus behavior (round(x * 32768.0))
    let input_i16: Vec<i16> = input
        .iter()
        .map(|x| (*x * 32768.0).clamp(-32768.0, 32767.0) as i16)
        .collect();

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

    c_packet.truncate(c_bytes as usize);

    println!("\nC packet: {} bytes", c_bytes);
    println!("Hex: {}", hex_encode(&c_packet));

    println!("\n--- Side-by-side comparison ---");
    let len = (rust_bytes as usize).min(c_bytes as usize);
    for i in 0..len {
        let same = rust_packet[i] == c_packet[i];
        if !same {
            println!(
                "Byte {}: Rust={:02x} C={:02x} DIFF",
                i, rust_packet[i], c_packet[i]
            );
        } else {
            println!(
                "Byte {}: {:02x} = {:02x} SAME",
                i, rust_packet[i], c_packet[i]
            );
        }
    }
}
