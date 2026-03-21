use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

fn generate_sine(freq: f32, n_samples: usize, sample_rate: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| (2.0 * PI * freq * i as f32 / sample_rate as f32).sin())
        .collect()
}

fn to_i16(samples: &[f32]) -> Vec<i16> {
    samples.iter().map(|&x| (x * 32767.0) as i16).collect()
}

fn to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&x| x as f32 / 32768.0).collect()
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f64 = samples.iter().map(|x| (*x as f64).powi(2)).sum();
    (sum / samples.len() as f64).sqrt() as f32
}

fn max_abs(samples: &[f32]) -> f32 {
    samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max)
}

/// Compare two audio signals sample-by-sample
fn compare_signals(original: &[f32], decoded: &[f32], frame_idx: usize) {
    let n = original.len().min(decoded.len());
    let mut max_diff = 0.0f32;
    let mut sum_sq_diff = 0.0f64;

    for i in 0..n {
        let diff = (original[i] - decoded[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_sq_diff += (diff as f64).powi(2);
    }

    let rms_diff = (sum_sq_diff / n as f64).sqrt() as f32;

    println!(
        "  Frame {}: max_diff={:.6}, rms_diff={:.6}, orig_rms={:.6}, dec_rms={:.6}",
        frame_idx,
        max_diff,
        rms_diff,
        rms(original),
        rms(decoded)
    );

    // Check for click/pop artifacts (sudden large amplitude changes)
    if n > 1 {
        let mut max_sample_diff = 0.0f32;
        for i in 1..n {
            let sample_diff = (decoded[i] - decoded[i - 1]).abs();
            if sample_diff > max_sample_diff {
                max_sample_diff = sample_diff;
            }
        }
        if max_sample_diff > 0.5 {
            println!(
                "  WARNING: Possible click/pop at frame {}! max_sample_diff={}",
                frame_idx, max_sample_diff
            );
        }
    }
}

// =============================================================================
// Test 1: C FIXED_POINT encoder → Rust decoder
// =============================================================================

#[test]
fn test_c_fixedpoint_to_rust_decoder() {
    let sample_rate: i32 = 8000;
    let channels = 1;
    let frame_size = 160; // 20ms at 8kHz
    let bitrate = 10000;

    println!("\n=== Test: C FIXED_POINT → Rust Decoder ===");
    println!(
        "Config: {}Hz, {}ch, {}bps, {} samples/frame",
        sample_rate, channels, bitrate, frame_size
    );

    // Generate test signal
    let input = generate_sine(440.0, frame_size, sample_rate as usize);
    let input_i16 = to_i16(&input);
    let input_rms = rms(&input);
    println!("Input: {} samples, RMS={:.6}", input.len(), input_rms);

    // Use opusic_sys (C FLOAT) encoder for comparison
    // Note: opusic_sys uses FLOAT SILK, not FIXED_POINT
    // For true FIXED_POINT comparison, we need to use the locally compiled libopus.a
    let c_encoder_ptr: *mut opusic_sys::OpusEncoder = unsafe {
        let mut error = 0i32;
        let ptr = opusic_sys::opus_encoder_create(
            sample_rate as i32,
            channels as i32,
            opusic_sys::OPUS_APPLICATION_VOIP as i32,
            &mut error,
        );
        if error != 0 {
            panic!("Failed to create C encoder: error {}", error);
        }
        ptr
    };

    unsafe {
        opusic_sys::opus_encoder_ctl(
            c_encoder_ptr,
            opusic_sys::OPUS_SET_BITRATE_REQUEST,
            bitrate as i32,
        );
        opusic_sys::opus_encoder_ctl(c_encoder_ptr, opusic_sys::OPUS_SET_COMPLEXITY_REQUEST, 0);
    }

    // Encode with C encoder
    let mut c_packet = vec![0u8; 400];
    let c_bytes = unsafe {
        opusic_sys::opus_encode(
            c_encoder_ptr,
            input_i16.as_ptr(),
            frame_size as i32,
            c_packet.as_mut_ptr(),
            c_packet.len() as i32,
        )
    };

    if c_bytes < 0 {
        panic!("C encoder failed: error {}", c_bytes);
    }
    c_packet.truncate(c_bytes as usize);

    println!("C encoded: {} bytes", c_bytes);
    println!("Packet hex: {}", hex::encode(&c_packet));

    unsafe {
        opusic_sys::opus_encoder_destroy(c_encoder_ptr);
    }

    // Decode with Rust decoder
    let mut rust_decoder =
        OpusDecoder::new(sample_rate, channels).expect("Failed to create Rust decoder");

    let mut rust_output = vec![0.0f32; frame_size];
    let rust_decoded = rust_decoder
        .decode(&c_packet, frame_size, &mut rust_output)
        .expect("Rust decode failed");

    println!("Rust decoded: {} samples", rust_decoded);

    // Compare
    compare_signals(&input, &rust_output, 0);

    // Verify output quality
    let output_rms = rms(&rust_output);
    let ratio = output_rms / input_rms;
    println!(
        "Energy ratio: {:.3} (should be ~1.0 for clean signal)",
        ratio
    );

    if ratio < 0.1 {
        println!("FAIL: Output energy is too low - possible decoder bug!");
    } else if ratio > 10.0 {
        println!("FAIL: Output energy is too high - possible decoder bug!");
    } else {
        println!("PASS: Output energy is reasonable");
    }
}

// =============================================================================
// Test 2: Rust encoder → C decoder
// =============================================================================

#[test]
fn test_rust_to_c_decoder() {
    let sample_rate: i32 = 8000;
    let channels = 1;
    let frame_size = 160;
    let bitrate = 10000;

    println!("\n=== Test: Rust Encoder → C Decoder ===");

    let input = generate_sine(440.0, frame_size, sample_rate as usize);
    let input_rms = rms(&input);
    println!("Input: {} samples, RMS={:.6}", input.len(), input_rms);

    // Encode with Rust encoder
    let mut rust_encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create Rust encoder");
    rust_encoder.bitrate_bps = bitrate;
    rust_encoder.complexity = 0;

    let mut rust_packet = vec![0u8; 400];
    let rust_bytes = rust_encoder
        .encode(&input, frame_size, &mut rust_packet)
        .expect("Rust encode failed");
    rust_packet.truncate(rust_bytes);

    println!("Rust encoded: {} bytes", rust_bytes);
    println!("Packet hex: {}", hex::encode(&rust_packet));

    // Decode with C decoder
    let c_decoder_ptr: *mut opusic_sys::OpusDecoder = unsafe {
        let mut error = 0i32;
        let ptr = opusic_sys::opus_decoder_create(sample_rate as i32, channels as i32, &mut error);
        if error != 0 {
            panic!("Failed to create C decoder: error {}", error);
        }
        ptr
    };

    let mut c_output_i16 = vec![0i16; frame_size];
    let c_decoded = unsafe {
        opusic_sys::opus_decode(
            c_decoder_ptr,
            rust_packet.as_ptr(),
            rust_bytes as i32,
            c_output_i16.as_mut_ptr(),
            frame_size as i32,
            0,
        )
    };

    if c_decoded < 0 {
        panic!("C decoder failed: error {}", c_decoded);
    }

    let c_output = to_f32(&c_output_i16[..c_decoded as usize]);
    println!("C decoded: {} samples", c_decoded);

    unsafe {
        opusic_sys::opus_decoder_destroy(c_decoder_ptr);
    }

    // Compare
    compare_signals(&input, &c_output, 0);

    let output_rms = rms(&c_output);
    let ratio = output_rms / input_rms;
    println!("Energy ratio: {:.3}", ratio);

    if ratio < 0.1 {
        println!("FAIL: Output energy too low!");
    } else if ratio > 10.0 {
        println!("FAIL: Output energy too high!");
    } else {
        println!("PASS: Energy reasonable");
    }
}

// =============================================================================
// Test 3: Multi-frame test
// =============================================================================

#[test]
fn test_multi_frame_cross_decode() {
    let sample_rate: i32 = 8000;
    let channels = 1;
    let frame_size = 160;
    let bitrate = 10000;
    let n_frames = 5;

    println!("\n=== Test: Multi-frame Cross-Decode ===");

    // Generate multi-frame input
    let mut full_input = Vec::new();
    for i in 0..n_frames {
        let freq = 440.0 + (i as f32 * 50.0); // Vary frequency per frame
        let frame = generate_sine(freq, frame_size, sample_rate as usize);
        full_input.extend(frame);
    }

    let input_i16 = to_i16(&full_input);
    let input_rms = rms(&full_input);
    println!(
        "Total input: {} samples, RMS={:.6}",
        full_input.len(),
        input_rms
    );

    // Encode with C encoder
    let c_encoder_ptr: *mut opusic_sys::OpusEncoder = unsafe {
        let mut error = 0i32;
        let ptr = opusic_sys::opus_encoder_create(
            sample_rate as i32,
            channels as i32,
            opusic_sys::OPUS_APPLICATION_VOIP as i32,
            &mut error,
        );
        if error != 0 {
            panic!("Failed to create C encoder: error {}", error);
        }
        ptr
    };

    unsafe {
        opusic_sys::opus_encoder_ctl(
            c_encoder_ptr,
            opusic_sys::OPUS_SET_BITRATE_REQUEST,
            bitrate as i32,
        );
        opusic_sys::opus_encoder_ctl(c_encoder_ptr, opusic_sys::OPUS_SET_COMPLEXITY_REQUEST, 0);
        opusic_sys::opus_encoder_ctl(c_encoder_ptr, opusic_sys::OPUS_SET_VBR_REQUEST, 0); // CBR
    }

    // Encode frame by frame
    let mut all_packets = Vec::new();
    for frame_idx in 0..n_frames {
        let frame_start = frame_idx * frame_size;
        let frame_input = &input_i16[frame_start..frame_start + frame_size];

        let mut packet = vec![0u8; 400];
        let bytes = unsafe {
            opusic_sys::opus_encode(
                c_encoder_ptr,
                frame_input.as_ptr(),
                frame_size as i32,
                packet.as_mut_ptr(),
                packet.len() as i32,
            )
        };

        if bytes < 0 {
            panic!("Frame {} encode failed: {}", frame_idx, bytes);
        }
        packet.truncate(bytes as usize);
        all_packets.push(packet);

        println!(
            "Frame {}: {} bytes, hex: {}",
            frame_idx,
            bytes,
            hex::encode(&all_packets[frame_idx])
        );
    }

    unsafe {
        opusic_sys::opus_encoder_destroy(c_encoder_ptr);
    }

    // Decode with Rust decoder
    let mut rust_decoder =
        OpusDecoder::new(sample_rate, channels).expect("Failed to create Rust decoder");

    println!("\nDecoding with Rust...");
    let mut full_output = Vec::new();

    for frame_idx in 0..n_frames {
        let mut frame_output = vec![0.0f32; frame_size];
        let decoded = rust_decoder
            .decode(&all_packets[frame_idx], frame_size, &mut frame_output)
            .expect("Rust decode failed");

        compare_signals(
            &full_input[frame_idx * frame_size..],
            &frame_output,
            frame_idx,
        );
        full_output.extend(&frame_output[..decoded]);
    }

    // Overall quality
    let output_rms = rms(&full_output);
    let ratio = output_rms / input_rms;
    println!("\nOverall energy ratio: {:.3}", ratio);

    if ratio < 0.1 {
        println!("FAIL: Overall output energy too low!");
    } else {
        println!("PASS");
    }
}

// =============================================================================
// Test 4: Check for clicks/pops (high frequency artifacts)
// =============================================================================

#[test]
fn test_no_clicks_or_pops() {
    let sample_rate: i32 = 8000;
    let channels = 1;
    let frame_size = 160;
    let bitrate = 10000;

    println!("\n=== Test: Click/Pop Detection ===");

    // Use a silent input (no signal = any artifact is obvious)
    let input = vec![0.0f32; frame_size];
    let input_i16 = to_i16(&input);

    // Encode with C
    let c_encoder_ptr: *mut opusic_sys::OpusEncoder = unsafe {
        let mut error = 0i32;
        let ptr = opusic_sys::opus_encoder_create(
            sample_rate as i32,
            channels as i32,
            opusic_sys::OPUS_APPLICATION_VOIP as i32,
            &mut error,
        );
        if error != 0 {
            panic!("Failed to create C encoder: error {}", error);
        }
        ptr
    };

    unsafe {
        opusic_sys::opus_encoder_ctl(
            c_encoder_ptr,
            opusic_sys::OPUS_SET_BITRATE_REQUEST,
            bitrate as i32,
        );
        opusic_sys::opus_encoder_ctl(c_encoder_ptr, opusic_sys::OPUS_SET_COMPLEXITY_REQUEST, 0);
    }

    let mut packet = vec![0u8; 400];
    let bytes = unsafe {
        opusic_sys::opus_encode(
            c_encoder_ptr,
            input_i16.as_ptr(),
            frame_size as i32,
            packet.as_mut_ptr(),
            packet.len() as i32,
        )
    };

    unsafe {
        opusic_sys::opus_encoder_destroy(c_encoder_ptr);
    }

    if bytes < 0 {
        panic!("Encode failed: {}", bytes);
    }
    packet.truncate(bytes as usize);

    println!("Encoded silence: {} bytes", bytes);

    // Decode with Rust
    let mut rust_decoder =
        OpusDecoder::new(sample_rate, channels).expect("Failed to create Rust decoder");

    let mut output = vec![0.0f32; frame_size];
    let decoded = rust_decoder
        .decode(&packet, frame_size, &mut output)
        .expect("Rust decode failed");

    // Check for artifacts
    let max_amp = max_abs(&output);
    println!("Max amplitude in decoded silence: {:.6}", max_amp);

    // Check for sudden jumps (clicks)
    let mut max_jump = 0.0f32;
    for i in 1..decoded {
        let jump = (output[i] - output[i - 1]).abs();
        if jump > max_jump {
            max_jump = jump;
        }
    }
    println!("Max sample-to-sample jump: {:.6}", max_jump);

    if max_amp > 0.01 {
        println!("WARNING: Non-zero output from silence input!");
    }
    if max_jump > 0.1 {
        println!("WARNING: Possible click detected!");
    }
    if max_amp < 0.001 && max_jump < 0.01 {
        println!("PASS: Clean silence output");
    }
}

// =============================================================================
// Helper
// =============================================================================

mod hex {
    pub fn encode(data: &[u8]) -> String {
        let mut s = String::with_capacity(data.len() * 2);
        for &b in data {
            let _ = std::fmt::Write::write_fmt(&mut s, format_args!("{:02x}", b));
        }
        s
    }
}
