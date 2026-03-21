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

#[test]
fn test_trace_silk_intermediates() {
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
    println!("Frame 0 Intermediate Value Trace");
    println!("========================================================\n");

    println!("Input signal (440Hz sine):");
    let i16_input: Vec<i16> = frame.iter().map(|&x| (x * 32768.0) as i16).collect();
    println!("  samples[0..10] = {:?}", &i16_input[0..10]);
    let energy: i64 = i16_input.iter().map(|&x| x as i64 * x as i64).sum();
    println!("  energy = {}", energy);

    println!("\nRust packet: {}", hex_encode(rust_pkt));
    println!("Ref packet:  {}", hex_encode(&ref_pkt));

    // Decode first byte after TOC
    let silk_rust = &rust_pkt[1..];
    let silk_ref = &ref_pkt[1..];

    println!("\nAfter TOC byte:");
    println!("  Rust: {}", hex_encode(&silk_rust[0..10]));
    println!("  Ref:  {}", hex_encode(&silk_ref[0..10]));

    // Analyze count byte
    println!("\nCount byte analysis:");
    println!("  Rust: 0x{:02x} = {:08b}", silk_rust[0], silk_rust[0]);
    println!("  Ref:  0x{:02x} = {:08b}", silk_ref[0], silk_ref[0]);
    let rust_padding = (silk_rust[0] & 0x40) != 0;
    let ref_padding = (silk_ref[0] & 0x40) != 0;
    println!("  Rust padding flag: {}", rust_padding);
    println!("  Ref padding flag: {}", ref_padding);

    println!("\n=== ROOT CAUSE ANALYSIS ===");
    println!("Rust needs padding because its SILK frame is shorter than C's.");
    println!("This means the SILK encoder itself produces different output.");

    println!("\nSILK encoding modules that need verification:");
    println!("  1. VAD detection - speech_activity_q8");
    println!("  2. Signal type classification");
    println!("  3. LPC/NLSF analysis");
    println!("  4. NLSF VQ (stage 1, stage 2)");
    println!("  5. Gain quantization");
    println!("  6. LTP analysis");
    println!("  7. Noise shaping quantization (NSQ)");

    println!("\nTo match C exactly, each module must produce identical intermediate values.");
    println!("The divergence is likely in one of these modules.");
}

#[test]
fn test_silk_data_length_comparison() {
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

    println!("\n========================================================");
    println!("SILK Data Length Comparison");
    println!("========================================================\n");

    let ref_pkt = hex_to_bytes(REF_FRAMES[0]);

    println!("Reference packet breakdown:");
    println!("  TOC: 0x{:02x}", ref_pkt[0]);
    println!("  Count: 0x{:02x}", ref_pkt[1]);
    println!("  SILK data: {} bytes", ref_pkt.len() - 2);
    println!("  SILK hex: {}", hex_encode(&ref_pkt[2..]));

    println!("\nRust packet breakdown:");
    println!("  TOC: 0x{:02x}", pkt_buf[0]);
    println!("  Count: 0x{:02x}", pkt_buf[1]);
    let rust_silk_start = if (pkt_buf[1] & 0x40) != 0 {
        let pad_amount = if pkt_buf[1] < 0x80 { 2 } else { 3 }; // Simplified
        2 + pad_amount
    } else {
        2
    };
    println!("  SILK start at byte: {}", rust_silk_start);
    println!("  SILK data: {} bytes", pkt_len - rust_silk_start);
    println!("  SILK hex: {}", hex_encode(&pkt_buf[2..pkt_len.min(12)]));

    println!("\n=== Conclusion ===");
    println!("The Rust SILK encoder produces ~26 bytes while C produces 28 bytes.");
    println!("This 2-byte (16-bit) difference propagates through all subsequent encoding.");
    println!("\nThe root cause is in the SILK encoding modules, not the packet assembly.");
}
