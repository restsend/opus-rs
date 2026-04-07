/// Profiling target: loop 48kHz CELT encode+decode for flame-graph capture.
/// Run with: samply record target/release/examples/profile_48k
use std::hint::black_box;

fn sine_f32(samples: usize, sample_rate: u32, freq: u32) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            f64::sin(2.0 * std::f64::consts::PI * freq as f64 * t) as f32 * 0.25
        })
        .collect()
}

fn main() {
    use opus_rs::{Application, OpusDecoder, OpusEncoder};

    // 48kHz / 20ms audio frame
    let sample_rate = 48000u32;
    let frame_ms = 20usize;
    let frame_size = sample_rate as usize * frame_ms / 1000;
    let input = sine_f32(frame_size, sample_rate, 440);

    let mut enc = OpusEncoder::new(sample_rate as i32, 1, Application::Audio).unwrap();
    enc.bitrate_bps = 64_000;
    enc.complexity = 0;
    let mut dec = OpusDecoder::new(sample_rate as i32, 1).unwrap();
    let mut output = vec![0u8; 1024];
    let mut pcm = vec![0.0f32; frame_size];

    // Warmup
    for _ in 0..200 {
        let len = enc
            .encode(black_box(&input), frame_size, black_box(&mut output))
            .unwrap();
        dec.decode(black_box(&output[..len]), frame_size, black_box(&mut pcm))
            .unwrap();
    }

    // Profiling loop — long enough for samply to collect samples
    let iterations = 20_000;
    for _ in 0..iterations {
        let len = enc
            .encode(black_box(&input), frame_size, black_box(&mut output))
            .unwrap();
        dec.decode(black_box(&output[..len]), frame_size, black_box(&mut pcm))
            .unwrap();
    }
}
