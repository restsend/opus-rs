use std::hint::black_box;

// Generate test audio - 440 Hz sine wave
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

    // Test configurations: (sample_rate, frame_ms, app_type, bitrate)
    let configs: &[(u32, usize, &str, i32)] = &[
        (8000, 20, "voip", 20_000),
        (16000, 20, "voip", 20_000),
        (48000, 20, "audio", 64_000),
    ];

    // Warm up
    for _ in 0..3 {
        for &(sr, fs_ms, app_str, br) in configs {
            let frame_size = sr as usize * fs_ms / 1000;
            let input = sine_f32(frame_size, sr, 440);
            let app = if app_str == "audio" {
                Application::Audio
            } else {
                Application::Voip
            };
            let mut enc = OpusEncoder::new(sr as i32, 1, app).unwrap();
            enc.bitrate_bps = br;
            let mut output = vec![0u8; 1024];
            let mut dec = OpusDecoder::new(sr as i32, 1).unwrap();
            let mut pcm = vec![0.0f32; frame_size];

            for _ in 0..100 {
                let len = enc
                    .encode(black_box(&input), frame_size, black_box(&mut output))
                    .unwrap();
                dec.decode(black_box(&output[..len]), frame_size, black_box(&mut pcm))
                    .unwrap();
            }
        }
    }

    println!("Warmup done. Starting measurement...");

    // Actual measurement
    for &(sr, fs_ms, app_str, br) in configs {
        let frame_size = sr as usize * fs_ms / 1000;
        let input = sine_f32(frame_size, sr, 440);
        let app = if app_str == "audio" {
            Application::Audio
        } else {
            Application::Voip
        };
        let mut enc = OpusEncoder::new(sr as i32, 1, app).unwrap();
        enc.bitrate_bps = br;
        let mut output = vec![0u8; 1024];
        let mut dec = OpusDecoder::new(sr as i32, 1).unwrap();
        let mut pcm = vec![0.0f32; frame_size];

        let start = std::time::Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            let len = enc
                .encode(black_box(&input), frame_size, black_box(&mut output))
                .unwrap();
            dec.decode(black_box(&output[..len]), frame_size, black_box(&mut pcm))
                .unwrap();
        }

        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;
        println!(
            "{:>6}Hz/{:>3}ms {}: {:.2} µs/frame ({:.2} ms total for {} iterations)",
            sr,
            fs_ms,
            app_str,
            ns_per_op / 1000.0,
            elapsed.as_secs_f64() * 1000.0,
            iterations
        );
    }

    println!("Done!");
}
