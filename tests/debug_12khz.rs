#[test]
fn test_debug_12khz() {
    use opus_rs::{Application, OpusEncoder};
    use std::f32::consts::PI;
    
    let sample_rate = 12000i32;
    let channels = 1;
    let frame_size = 240; // 20ms at 12kHz
    
    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 14000;
    encoder.complexity = 0;
    encoder.use_cbr = true;
    
    // Generate 440Hz sine
    let mut input = vec![0.0f32; frame_size];
    for i in 0..frame_size {
        let val_f64 = (2.0 * PI * 440.0 * i as f32 / sample_rate as f32).sin();
        let i16_val = (val_f64 * 16383.0) as i16;
        input[i] = i16_val as f32 / 32768.0;
    }
    
    let mut output = vec![0u8; 500];
    let n = encoder.encode(&input, frame_size, &mut output)
        .expect("Encode failed");
    
    println!("Packet length: {} bytes", n);
    println!("First 32 bytes:");
    for (i, &b) in output[..n.min(32)].iter().enumerate() {
        print!("{:02x} ", b);
        if (i + 1) % 16 == 0 {
            println!();
        }
    }
    println!();
    
    println!("Expected (C): 2b410183f90625db5f23e65d1a9903a78b483c14977f64a54250001701b6d3bf4bc000");
    println!("Got (Rust):  ");
    for &b in &output[..n.min(36)] {
        print!("{:02x}", b);
    }
    println!();
}
