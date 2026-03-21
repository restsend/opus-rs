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

/// Reference frames from opus_demo
const REF_FRAMES: &[&str] = &[
    "0b018455a4e3c2206bd13d16c0f1d332bbe6fca978f3eac09e538202d180",
    "0b4101ac3140db3d937238af06f7e79b1c5633a1a31b781a8b390a5fd000",
];

fn hex_to_bytes(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// Decode the first few bits of SILK bitstream to understand the flags
#[test]
fn test_decode_silk_flags() {
    let rust_pkt = hex_to_bytes("0b4101847a6e601c63439bf06044fd59c27de1308172d16c65733925a000");
    let ref_pkt = hex_to_bytes("0b018455a4e3c2206bd13d16c0f1d332bbe6fca978f3eac09e538202d180");

    println!("\n=== SILK Flag Analysis ===\n");

    // TOC byte (same for both)
    println!("TOC byte: 0x{:02x}", rust_pkt[0]);
    println!("  Config: {} (SILK-only, 8kHz, 20ms)", rust_pkt[0] >> 3);
    println!("  Stereo: {}", rust_pkt[0] & 0x04 != 0);
    println!("  Frame count code: {}", rust_pkt[0] & 0x03);
    println!();

    let silk_rust = &rust_pkt[1..];
    let silk_ref = &ref_pkt[1..];
    println!("First 3 bytes (hex):");
    println!(
        "  Rust: {:02x} {:02x} {:02x}",
        silk_rust[0], silk_rust[1], silk_rust[2]
    );
    println!(
        "  Ref:  {:02x} {:02x} {:02x}",
        silk_ref[0], silk_ref[1], silk_ref[2]
    );

    let rng_rust: u32 = 0x8000_ffff;
    let val_rust: u32 =
        ((silk_rust[0] as u32) << 16) | ((silk_rust[1] as u32) << 8) | (silk_rust[2] as u32);

    let rng_ref: u32 = 0x8000_ffff;
    let val_ref: u32 =
        ((silk_ref[0] as u32) << 16) | ((silk_ref[1] as u32) << 8) | (silk_ref[2] as u32);

    println!("\nRange decoder initial state:");
    println!("  Rust: rng={:08x} val={:08x}", rng_rust, val_rust);
    println!("  Ref:  rng={:08x} val={:08x}", rng_ref, val_ref);

    // The first symbol is encoded with icdf = [192, 0], ft=256
    // This reserves 2 bits for flags
    // decode_icdf: k = 0 if val >= (rng/256 * icdf[0]), else search for k
    // rng/256 = 0x8000_ffff / 256 = 0x0080_00ff (approx)
    let r_rust = rng_rust / 256;
    let r_ref = rng_ref / 256;
    let threshold_0_rust = r_rust * 192; // icdf[0] = 192
    let threshold_0_ref = r_ref * 192;

    println!("\nFirst symbol decode (VAD/LBRR flags placeholder):");
    println!(
        "  Rust: r={:08x} threshold={:08x} val={:08x}",
        r_rust, threshold_0_rust, val_rust
    );
    println!(
        "  Ref:  r={:08x} threshold={:08x} val={:08x}",
        r_ref, threshold_0_ref, val_ref
    );

    // For k=0: val >= r*192 (high probability)
    // For k=1: val < r*192 (low probability - the patched bits are non-zero)
    let k_rust = if val_rust >= threshold_0_rust { 0 } else { 1 };
    let k_ref = if val_ref >= threshold_0_ref { 0 } else { 1 };

    println!("  Rust decoded k={} (flags placeholder)", k_rust);
    println!("  Ref decoded k={} (flags placeholder)", k_ref);

    // The patched bits should be: (VAD << 1) | LBRR
    // For Rust: first byte 0x41 = 0100 0001
    // For Ref:  first byte 0x01 = 0000 0001

    println!("\nPatched bits analysis:");
    // The patched bits go into the high bits of the first byte
    // patch_initial_bits(val, nbits=2) patches bits [7:6]
    // val = (VAD << 1) | LBRR

    // Extract what was patched
    // For 2 bits patched at position [7:6]:
    // val_patched = (first_byte >> 6) & 0x3
    let patched_rust = (silk_rust[0] >> 6) & 0x3;
    let patched_ref = (silk_ref[0] >> 6) & 0x3;

    println!(
        "  Rust patched value: {:02b} (VAD={}, LBRR={})",
        patched_rust,
        (patched_rust >> 1) & 1,
        patched_rust & 1
    );
    println!(
        "  Ref patched value:  {:02b} (VAD={}, LBRR={})",
        patched_ref,
        (patched_ref >> 1) & 1,
        patched_ref & 1
    );

    // So:
    // Rust: VAD=1, LBRR=0 (patched = 10 = 2) -> but we see 0x41 = 01 000001
    // Ref:  VAD=0, LBRR=0 (patched = 00 = 0) -> 0x01 = 00 000001

    // Wait, let me recalculate:
    // 0x41 = 0100 0001 = bits 7..0 = 0 1 0 0 0 0 0 1
    // high 2 bits = 01 = 1 (binary)
    // 0x01 = 0000 0001 = high 2 bits = 00 = 0

    // If patched value = (VAD << 1) | LBRR
    // Rust: patched = 1 = 01 = VAD=0, LBRR=1
    // Ref:  patched = 0 = 00 = VAD=0, LBRR=0

    println!("\n*** ANALYSIS ***");
    println!(
        "Rust: patched bits = {} -> VAD={}, LBRR={}",
        patched_rust,
        (patched_rust >> 1) & 1,
        patched_rust & 1
    );
    println!(
        "Ref:  patched bits = {} -> VAD={}, LBRR={}",
        patched_ref,
        (patched_ref >> 1) & 1,
        patched_ref & 1
    );

    // The difference is in LBRR flag!
    // Rust sets LBRR=1, but Ref sets LBRR=0
    // This causes all subsequent bits to differ
}

/// Check what signal_type the Rust encoder detects
#[test]
fn test_rust_encoder_signal_detection() {
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

    println!("\n=== Rust Encoder Signal Detection ===\n");

    println!("Input frame stats:");
    let min = frame.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = frame.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = frame.iter().sum();
    let avg = sum / frame.len() as f32;
    println!("  min={}, max={}, avg={}", min, max, avg);

    // Check energy
    let energy: f64 = frame.iter().map(|&x| (x as f64) * (x as f64)).sum();
    println!("  energy={:.6}", energy);

    // Encode
    let pkt_len = encoder
        .encode(frame, frame_size, &mut pkt_buf)
        .expect("Encode failed");

    println!("\nEncoded packet: {} bytes", pkt_len);
    println!("Hex: {}", hex_encode(&pkt_buf[..pkt_len]));

    // Reference
    let ref_pkt = hex_to_bytes(REF_FRAMES[0]);
    println!("Ref:  {}", hex_encode(&ref_pkt));

    // The 440Hz sine wave should be detected as VOICED (VAD=1)
    // But the bitstream shows VAD=0 for Rust...

    // Let's check the encoder configuration
    println!("\nEncoder config:");
    println!("  FEC enabled: {}", encoder.use_inband_fec);
    println!("  Packet loss: {}%", encoder.packet_loss_perc);
}

/// Test with explicit FEC disabled
#[test]
fn test_with_fec_explicitly_disabled() {
    let sample_rate = 8000i32;
    let channels = 1;
    let frame_size = 160usize;

    let all_pcm = gen_silk_test_pcm(2 * frame_size);

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 12000;
    encoder.use_cbr = true;
    encoder.complexity = 10;
    // Explicitly disable FEC
    encoder.use_inband_fec = false;
    encoder.packet_loss_perc = 0;

    let frame = &all_pcm[0..frame_size];
    let mut pkt_buf = vec![0u8; 200];

    let pkt_len = encoder
        .encode(frame, frame_size, &mut pkt_buf)
        .expect("Encode failed");

    println!("\n=== With FEC Explicitly Disabled ===\n");
    println!("Rust: {}", hex_encode(&pkt_buf[..pkt_len]));
    println!("Ref:  {}", REF_FRAMES[0]);

    // Check first diff
    let rust = &pkt_buf[..pkt_len];
    let ref_pkt = hex_to_bytes(REF_FRAMES[0]);
    for i in 0..rust.len().min(ref_pkt.len()) {
        if rust[i] != ref_pkt[i] {
            println!(
                "\nFirst diff at byte {}: rust={:02x} ref={:02x}",
                i, rust[i], ref_pkt[i]
            );
            break;
        }
    }
}
