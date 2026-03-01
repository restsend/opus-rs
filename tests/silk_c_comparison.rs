/// Compare Rust SILK encoder output against C libopus via opusic-sys
/// Uses exactly the same float input and encoder settings for both
use std::f32::consts::PI;

#[test]
fn test_rust_vs_c_silk_encode() {
    // Generate test signal: 440 Hz sine at 8kHz, 20ms frame (160 samples)
    let mut input = vec![0.0f32; 160];
    for i in 0..160 {
        input[i] = (2.0 * PI * 440.0 * i as f32 / 8000.0).sin();
    }

    // ---- Encode with C (opusic-sys) ----
    let c_opus = unsafe {
        let mut err: i32 = 0;
        let enc = opusic_sys::opus_encoder_create(
            8000, // Fs
            1,    // channels
            opusic_sys::OPUS_APPLICATION_VOIP,
            &mut err,
        );
        assert_eq!(err, opusic_sys::OPUS_OK, "C encoder creation failed");

        opusic_sys::opus_encoder_ctl(enc, opusic_sys::OPUS_SET_BITRATE_REQUEST, 10000i32);
        opusic_sys::opus_encoder_ctl(enc, opusic_sys::OPUS_SET_VBR_REQUEST, 0i32);
        opusic_sys::opus_encoder_ctl(enc, opusic_sys::OPUS_SET_COMPLEXITY_REQUEST, 0i32);
        // Force narrowband for SILK-only
        opusic_sys::opus_encoder_ctl(
            enc,
            opusic_sys::OPUS_SET_MAX_BANDWIDTH_REQUEST,
            opusic_sys::OPUS_BANDWIDTH_NARROWBAND as i32,
        );

        let mut buf = vec![0u8; 1275];
        let n = opusic_sys::opus_encode_float(enc, input.as_ptr(), 160, buf.as_mut_ptr(), 1275);
        assert!(n > 0, "C opus_encode_float failed: {}", n);

        let result = buf[..n as usize].to_vec();
        opusic_sys::opus_encoder_destroy(enc);
        result
    };

    // ---- Encode with Rust ----
    use opus_rs::{Application, OpusEncoder};
    let mut encoder = OpusEncoder::new(8000, 1, Application::Voip).expect("Rust encoder failed");
    encoder.complexity = 0;
    encoder.bitrate_bps = 10000;
    encoder.use_cbr = true;

    let mut rust_buf = vec![0u8; 1275];
    let rust_bytes = encoder
        .encode(&input, 160, &mut rust_buf)
        .expect("Rust encode failed");
    let rust_opus = rust_buf[..rust_bytes].to_vec();

    // ---- Compare ----
    println!("C output ({} bytes):    {}", c_opus.len(), hex(&c_opus));
    println!(
        "Rust output ({} bytes): {}",
        rust_opus.len(),
        hex(&rust_opus)
    );

    // Find first byte difference
    let max_len = c_opus.len().max(rust_opus.len());
    let min_len = c_opus.len().min(rust_opus.len());
    let mut first_diff = None;
    for i in 0..min_len {
        if c_opus[i] != rust_opus[i] {
            first_diff = Some(i);
            break;
        }
    }

    if let Some(d) = first_diff {
        println!(
            "First difference at byte {}: C=0x{:02x} Rust=0x{:02x}",
            d, c_opus[d], rust_opus[d]
        );
        // Show bit-level difference
        let c_bits = format!("{:08b}", c_opus[d]);
        let r_bits = format!("{:08b}", rust_opus[d]);
        println!("  C bits:    {}", c_bits);
        println!("  Rust bits: {}", r_bits);
    } else if c_opus.len() != rust_opus.len() {
        println!(
            "Bytes match up to {} but lengths differ ({} vs {})",
            min_len,
            c_opus.len(),
            rust_opus.len()
        );
    } else {
        println!("PERFECT MATCH!");
    }

    // Decode both to compare SILK parameters
    if c_opus.len() >= 3 && rust_opus.len() >= 3 {
        println!("\n=== C SILK parameters ===");
        decode_silk_indices(&c_opus);
        println!("\n=== Rust SILK parameters ===");
        decode_silk_indices(&rust_opus);
    }

    // Multi-frame test: encode 5 consecutive frames
    println!("\n=== Multi-frame comparison (5 frames) ===");
    let mut c_enc = unsafe {
        let mut err = 0i32;
        let enc =
            opusic_sys::opus_encoder_create(8000, 1, opusic_sys::OPUS_APPLICATION_VOIP, &mut err);
        opusic_sys::opus_encoder_ctl(enc, opusic_sys::OPUS_SET_BITRATE_REQUEST, 10000i32);
        opusic_sys::opus_encoder_ctl(enc, opusic_sys::OPUS_SET_VBR_REQUEST, 0i32);
        opusic_sys::opus_encoder_ctl(enc, opusic_sys::OPUS_SET_COMPLEXITY_REQUEST, 0i32);
        opusic_sys::opus_encoder_ctl(
            enc,
            opusic_sys::OPUS_SET_MAX_BANDWIDTH_REQUEST,
            opusic_sys::OPUS_BANDWIDTH_NARROWBAND as i32,
        );
        enc
    };
    let mut rust_enc = OpusEncoder::new(8000, 1, Application::Voip).expect("Rust encoder failed");
    rust_enc.complexity = 0;
    rust_enc.bitrate_bps = 10000;
    rust_enc.use_cbr = true;

    let mut total_match = 0;
    let mut total_frames = 0;
    for frame_idx in 0..5 {
        // Generate successive frames of sine
        let mut frame = vec![0.0f32; 160];
        for i in 0..160 {
            let t = (frame_idx * 160 + i) as f32 / 8000.0;
            frame[i] = (2.0 * PI * 440.0 * t).sin();
        }

        let c_pkt = unsafe {
            let mut buf = vec![0u8; 1275];
            let n =
                opusic_sys::opus_encode_float(c_enc, frame.as_ptr(), 160, buf.as_mut_ptr(), 1275);
            assert!(n > 0);
            buf[..n as usize].to_vec()
        };

        let mut r_buf = vec![0u8; 1275];
        let r_n = rust_enc.encode(&frame, 160, &mut r_buf).unwrap();
        let r_pkt = r_buf[..r_n].to_vec();

        let matches = c_pkt == r_pkt;
        if matches {
            total_match += 1;
        }
        total_frames += 1;

        println!(
            "Frame {}: C={} bytes, Rust={} bytes, match={}",
            frame_idx,
            c_pkt.len(),
            r_pkt.len(),
            matches
        );
        if !matches {
            // Find first diff
            for i in 0..c_pkt.len().min(r_pkt.len()) {
                if c_pkt[i] != r_pkt[i] {
                    println!(
                        "  First diff at byte {}: C=0x{:02x} Rust=0x{:02x}",
                        i, c_pkt[i], r_pkt[i]
                    );
                    break;
                }
            }
        }
    }
    println!("\nMatched {}/{} frames", total_match, total_frames);

    unsafe {
        opusic_sys::opus_encoder_destroy(c_enc);
    }
}

fn hex(data: &[u8]) -> String {
    data.iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

fn decode_silk_indices(opus_pkt: &[u8]) {
    // TOC byte
    let toc = opus_pkt[0];
    let config = (toc >> 3) & 0x1f;
    let stereo = (toc >> 2) & 1;
    let code = toc & 0x03;
    println!("  TOC: config={} stereo={} code={}", config, stereo, code);

    if code == 3 && opus_pkt.len() > 2 {
        // Code 3: second byte is count/flag
        let count_byte = opus_pkt[1];
        let n_frames = count_byte & 0x3f;
        println!("  Code 3: {} frames", n_frames);

        // SILK payload starts after TOC + count byte
        let silk = &opus_pkt[2..];
        if silk.len() > 6 {
            // Quick peek at first few bytes
            print!("  SILK payload ({} bytes): ", silk.len());
            for i in 0..silk.len().min(10) {
                print!("{:02x}", silk[i]);
            }
            println!();
        }
    }
}
