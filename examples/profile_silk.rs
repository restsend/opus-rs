/// Profiling test: measure time spent in each major SILK encoder function
/// Run with: cargo test --test profile_silk -- --nocapture

use opus_rs::{Application, OpusEncoder};
use std::time::Instant;

fn main() {
    let sample_rate = 16000;
    let frame_size = 320;

    // Generate test input
    let input: Vec<f32> = (0..frame_size)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            (2.0 * std::f64::consts::PI * 440.0 * t).sin() as f32 * 0.5
        })
        .collect();

    let mut encoder = OpusEncoder::new(sample_rate, 1, Application::Voip).unwrap();
    encoder.bitrate_bps = 20000;
    encoder.complexity = 0;

    // Warmup
    for _ in 0..10 {
        let mut output = vec![0u8; 256];
        encoder.encode(&input, frame_size, &mut output).unwrap();
    }

    // Profile: measure multiple iterations
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        let mut output = vec![0u8; 256];
        encoder.encode(&input, frame_size, &mut output).unwrap();
    }
    let elapsed = start.elapsed();

    println!("=== SILK Encoder Profile ({}) ===", sample_rate);
    println!("Total time for {} iterations: {:?}", iterations, elapsed);
    println!("Average per frame: {:?}", elapsed / iterations as u32);
    println!("Throughput: {:.2} frames/sec", iterations as f64 / elapsed.as_secs_f64());
}
