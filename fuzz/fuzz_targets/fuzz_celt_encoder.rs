#![no_main]

use libfuzzer_sys::fuzz_target;
use opus_rs::{Application, OpusEncoder};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    // CELT supports all sample rates
    let sampling_rate = match data[0] % 5 {
        0 => 8000,
        1 => 12000,
        2 => 16000,
        3 => 24000,
        _ => 48000,
    };

    let channels = if data[1] % 2 == 0 { 1 } else { 2 };

    // Use Audio mode for CELT
    let mut encoder = match OpusEncoder::new(sampling_rate, channels, Application::Audio) {
        Ok(e) => e,
        Err(_) => return,
    };

    // CELT frame sizes depend on sample rate
    // At 48kHz: 120 (2.5ms), 240 (5ms), 480 (10ms), 960 (20ms)
    let frame_size = match data[2] % 4 {
        0 => sampling_rate as usize / 400, // 2.5ms
        1 => sampling_rate as usize / 200, // 5ms
        2 => sampling_rate as usize / 100, // 10ms
        _ => sampling_rate as usize / 50,  // 20ms
    };

    // Skip invalid frame sizes
    if frame_size == 0 || frame_size > 5760 {
        return;
    }

    // Test various bitrates
    encoder.bitrate_bps = match data[3] % 8 {
        0 => 6000,
        1 => 16000,
        2 => 32000,
        3 => 64000,
        4 => 96000,
        5 => 128000,
        6 => 256000,
        _ => 510000,
    };

    // Test complexity
    encoder.complexity = (data[4] % 11) as i32 - 1;

    // Generate input with various spectral characteristics
    let samples_needed = frame_size * channels;
    let mut input = vec![0.0f32; samples_needed];

    let spectral_type = data[5] % 8;
    for i in 0..samples_needed {
        let t = i as f32 / sampling_rate as f32;
        input[i] = match spectral_type {
            // Pure tone
            0 => (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5,
            // Multiple harmonics
            1 => {
                let mut v = 0.0;
                for h in 1..=5 {
                    v += (2.0 * std::f32::consts::PI * 440.0 * h as f32 * t).sin() / h as f32;
                }
                v * 0.2
            }
            // Low frequency
            2 => (2.0 * std::f32::consts::PI * 50.0 * t).sin() * 0.5,
            // High frequency
            3 => (2.0 * std::f32::consts::PI * 15000.0 * t).sin() * 0.5,
            // Silence with occasional clicks
            4 => {
                let idx = (i + 6) % data.len();
                if data[idx] > 250 {
                    0.9
                } else {
                    0.001 * (data[idx] as f32 - 128.0) / 128.0
                }
            }
            // Amplitude modulated
            5 => {
                let carrier = (2.0 * std::f32::consts::PI * 1000.0 * t).sin();
                let modulator = (2.0 * std::f32::consts::PI * 5.0 * t).sin();
                carrier * modulator * 0.4
            }
            // From fuzz data
            6 => {
                let idx = (i + 6) % data.len();
                (data[idx] as f32 - 128.0) / 128.0
            }
            // Extreme values
            _ => {
                let idx = (i + 6) % data.len();
                if data[idx] > 200 {
                    0.99
                } else if data[idx] < 55 {
                    -0.99
                } else {
                    0.0
                }
            }
        };
        // Ensure no NaN/Inf
        if !input[i].is_finite() {
            input[i] = 0.0;
        }
        input[i] = input[i].clamp(-1.0, 1.0);
    }

    // Encode
    let mut output = vec![0u8; 4000];
    let _ = encoder.encode(&input, frame_size, &mut output);
});
