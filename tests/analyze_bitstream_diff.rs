use opus_rs::{Application, OpusEncoder};
use std::f64::consts::PI;

fn gen_silk_test_pcm(n_samples: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let val_f64 = (2.0 * PI * 440.0 * i as f64 / 8000.0).sin();
            let i16_val = (val_f64 * 16383.0) as i16;
            i16_val as f32 / 32768.0
        })
        .collect()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Reference frames from opus_demo (voip, 8kHz, 1ch, 12kbps CBR, complexity 10)
const REF_FRAMES: &[&str] = &[
    "0b018455a4e3c2206bd13d16c0f1d332bbe6fca978f3eac09e538202d180",
    "0b4101ac3140db3d937238af06f7e79b1c5633a1a31b781a8b390a5fd000",
    "0b41069b2b3431aeb2ee203024f749c535df096f03f0f5ac000000000000",
    "0b41049b2b3431af52bc8194161b42a77fd294ddc3315dd6804000000000",
    "0b41089b2b3431af52bc7d809b6a10fadb98075eaeb50000000000000000",
    "0b41019b276b5871305f453d4cb0b49d2ad5b4cdbd357ea9352d1c988000",
    "0b41069b2b3431af52bc0e19c44e4561a70230a88cae5860000000000000",
    "0b41049b2b3431aeb2ee1e34a16fe019f229a13dac28fc6d27f800000000",
    "0b41089b2b3431aeb2ee2ae92eedad4354553414c43c0000000000000000",
    "0b41089b2b34328f5dad8e3708b5d032e78caf628a1c0000000000000000",
];

fn hex_to_bytes(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// Decode SILK frame header bits
fn decode_silk_header(bits: &[u8]) -> (u32, u32, u32, u32) {
    // Range decoder simulation
    let mut rng = 0x8000_ffffu32;
    let mut val = ((bits[0] as u32) << 16) | ((bits[1] as u32) << 8) | (bits[2] as u32);
    let mut offs = 0usize;

    // Decode VAD (logp=1) - bit
    rng >>= 1;
    let vad = if val >= rng {
        val -= rng;
        1
    } else {
        0
    };

    // Normalize
    while rng < 0x8000 {
        rng <<= 8;
        offs += 1;
        if offs + 3 < bits.len() {
            val = (val << 8) | (bits[offs + 3] as u32);
        }
    }

    // Decode LBRR flag (logp=1) - bit
    rng >>= 1;
    let lbrr = if val >= rng {
        val -= rng;
        1
    } else {
        0
    };

    // Normalize
    while rng < 0x8000 {
        rng <<= 8;
        offs += 1;
        if offs + 3 < bits.len() {
            val = (val << 8) | (bits[offs + 3] as u32);
        }
    }

    // Signal type/offset decode
    // For VAD=1, uses silk_type_offset_iCDF = { 224, 112, 44, 15, 3, 2, 1, 0 }
    let icdf: [u8; 8] = [224, 112, 44, 15, 3, 2, 1, 0];
    let ft = 256u32;
    let r = rng / ft;
    let mut k = 0usize;
    let mut tmp = r * (icdf[0] as u32);
    while val < tmp && k < 7 {
        k += 1;
        tmp = r * (icdf[k] as u32);
    }

    // The type_offset value
    let type_offset = k as u32;
    let signal_type = (type_offset + 2) / 2;
    let quant_offset_type = (type_offset + 2) % 2;

    (vad, lbrr, signal_type, quant_offset_type)
}

#[test]
fn test_detailed_bitstream_analysis() {
    let sample_rate = 8000i32;
    let channels = 1;
    let frame_size = 160usize;
    let n_frames = 10;

    let all_pcm = gen_silk_test_pcm(n_frames * frame_size + frame_size);

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 12000;
    encoder.use_cbr = true;
    encoder.complexity = 10;

    println!("\n========================================================");
    println!("SILK Bitstream Detailed Analysis");
    println!("========================================================\n");

    for frame_idx in 0..n_frames {
        let frame = &all_pcm[frame_idx * frame_size..(frame_idx + 1) * frame_size];
        let mut pkt_buf = vec![0u8; 200];
        let pkt_len = encoder
            .encode(frame, frame_size, &mut pkt_buf)
            .expect("Encode failed");
        let pkt = &pkt_buf[..pkt_len];

        let ref_bytes = hex_to_bytes(REF_FRAMES[frame_idx]);

        println!("Frame {}:", frame_idx);
        println!("  Rust: {}", hex_encode(pkt));
        println!("  Ref:  {}", hex_encode(&ref_bytes));

        // TOC byte analysis
        let toc = pkt[0];
        let ref_toc = ref_bytes[0];
        println!("  TOC: rust={:02x} ref={:02x}", toc, ref_toc);

        // SILK frame data (after TOC)
        let silk_data = &pkt[1..];
        let ref_silk = &ref_bytes[1..];

        // Binary comparison
        println!("  SILK data (binary):");
        for (i, (r, c)) in silk_data.iter().zip(ref_silk.iter()).enumerate() {
            if r != c {
                println!(
                    "    Byte {}: rust={:02x} ({:08b}) ref={:02x} ({:08b})",
                    i, r, r, c, c
                );
            }
        }

        // Decode headers
        if pkt[0] == ref_bytes[0] {
            let (vad, lbrr, sig_type, quant_off) = decode_silk_header(silk_data);
            let (ref_vad, ref_lbrr, ref_sig_type, ref_quant_off) = decode_silk_header(ref_silk);
            println!(
                "  Rust header: VAD={} LBRR={} sig_type={} quant_off={}",
                vad, lbrr, sig_type, quant_off
            );
            println!(
                "  Ref header:  VAD={} LBRR={} sig_type={} quant_off={}",
                ref_vad, ref_lbrr, ref_sig_type, ref_quant_off
            );
        }

        // Count matching bytes
        let matching = silk_data
            .iter()
            .zip(ref_silk.iter())
            .filter(|(a, b)| a == b)
            .count();
        println!(
            "  Matching: {}/{} bytes",
            matching,
            silk_data.len().min(ref_silk.len())
        );
        println!();
    }
}

/// Analyze the actual encoded values
#[test]
fn test_frame0_detailed_decode() {
    let sample_rate = 8000i32;
    let channels = 1;
    let frame_size = 160usize;

    let all_pcm = gen_silk_test_pcm(2 * frame_size);

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 12000;
    encoder.use_cbr = true;
    encoder.complexity = 10;

    let frame = &all_pcm[0..frame_size];
    let mut pkt_buf = vec![0u8; 200];
    let pkt_len = encoder
        .encode(frame, frame_size, &mut pkt_buf)
        .expect("Encode failed");
    let rust_pkt = &pkt_buf[..pkt_len];

    let ref_pkt = hex_to_bytes(REF_FRAMES[0]);

    println!("\n========================================================");
    println!("Frame 0 Detailed Decode Comparison");
    println!("========================================================\n");

    println!("Rust packet: {}", hex_encode(rust_pkt));
    println!("Ref packet:  {}", hex_encode(&ref_pkt));

    // Byte-by-byte analysis
    println!("\nByte-by-byte comparison:");
    for i in 0..rust_pkt.len().min(ref_pkt.len()) {
        let r = rust_pkt[i];
        let c = ref_pkt[i];
        let diff = if r != c { " *** DIFF ***" } else { "" };
        println!(
            "  [{:2}] rust={:02x} ({:08b}) ref={:02x} ({:08b}){}",
            i, r, r, c, c, diff
        );
    }

    // The first byte (0x0b) is the TOC: SILK-only mode, 8kHz, 20ms
    println!("\nTOC byte (0x0b):");
    println!(
        "  Config: {} -> SILK-only, 8kHz, 20ms frame",
        rust_pkt[0] >> 3
    );
    println!("  Stereo: {}", rust_pkt[0] & 0x04 != 0);
    println!("  Frame count code: {}", rust_pkt[0] & 0x03);

    // Parse SILK frame data
    println!("\nSILK frame data (after TOC):");
    let silk_rust = &rust_pkt[1..];
    let silk_ref = &ref_pkt[1..];

    // First bits: VAD flag (1 bit, logp=1)
    // High bit of first byte
    println!(
        "  Byte 0: rust={:02x} ({:08b}) ref={:02x} ({:08b})",
        silk_rust[0], silk_rust[0], silk_ref[0], silk_ref[0]
    );

    // In the SILK bitstream:
    // - First bit (MSB of byte 0) = VAD flag
    // - Second bit = LBRR flag
    // - Then signal_type + quant_offset_type (via iCDF)
    // - Then gain values

    let vad_rust = (silk_rust[0] >> 7) & 1;
    let lbrr_rust = (silk_rust[0] >> 6) & 1;
    let vad_ref = (silk_ref[0] >> 7) & 1;
    let lbrr_ref = (silk_ref[0] >> 6) & 1;

    println!("\n  Rust: VAD={} LBRR={}", vad_rust, lbrr_rust);
    println!("  Ref:  VAD={} LBRR={}", vad_ref, lbrr_ref);

    // The rest depends on range coding, so we need proper decoding
    // For now, just show the raw hex
    println!("\nRemaining SILK bytes:");
    println!("  Rust: {}", hex_encode(&silk_rust[1..]));
    println!("  Ref:  {}", hex_encode(&silk_ref[1..]));
}

/// Compare internal encoder state values
/// Note: This test requires adding debug output to the encoder
#[test]
fn test_encoder_state_comparison() {
    let sample_rate = 8000i32;
    let channels = 1;
    let frame_size = 160usize;

    let all_pcm = gen_silk_test_pcm(2 * frame_size);

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 12000;
    encoder.use_cbr = true;
    encoder.complexity = 10;

    let frame = &all_pcm[0..frame_size];
    let mut pkt_buf = vec![0u8; 200];

    println!("\n========================================================");
    println!("Encoder State After Frame 0");
    println!("========================================================\n");

    let pkt_len = encoder
        .encode(frame, frame_size, &mut pkt_buf)
        .expect("Encode failed");

    println!("Encoded {} bytes", pkt_len);
    println!("Packet: {}", hex_encode(&pkt_buf[..pkt_len]));

    // The reference frame 0
    let ref_pkt = hex_to_bytes(REF_FRAMES[0]);
    println!("Reference: {}", hex_encode(&ref_pkt));

    // Show what we know about the encoder state
    println!("\nEncoder configuration:");
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Channels: {}", channels);
    println!("  Frame size: {} samples", frame_size);
    println!("  Bitrate: {} bps", 12000);
    println!("  CBR: true");
    println!("  Complexity: 10");
}
