//! Fuzz test for Hybrid mode (SILK + CELT)
//! Tests the hybrid mode where lower bands are from SILK and upper bands from CELT

#![no_main]

use libfuzzer_sys::fuzz_target;
use opus_rs::{Application, OpusDecoder, OpusEncoder};

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }

    // Hybrid mode typically uses 48kHz for full bandwidth
    let sampling_rates = [24000, 48000];

    for &sampling_rate in &sampling_rates {
        let channels = if data[0] % 2 == 0 { 1 } else { 2 };

        // Create encoder with VoIP or Audio mode (hybrid uses both)
        let application = if data[1] % 2 == 0 {
            Application::Voip
        } else {
            Application::Audio
        };

        let mut encoder = match OpusEncoder::new(sampling_rate, channels, application) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let mut decoder = match OpusDecoder::new(sampling_rate, channels) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Hybrid frame sizes: typically 10ms or 20ms
        let frame_sizes: Vec<usize> = vec![
            sampling_rate as usize / 100,  // 10ms
            sampling_rate as usize / 50,   // 20ms
        ];

        for frame_size in &frame_sizes {
            if *frame_size == 0 {
                continue;
            }

            // Configure encoder with various settings
            encoder.bitrate_bps = match data[2] % 6 {
                0 => 16000,
                1 => 32000,
                2 => 64000,
                3 => 96000,
                4 => 128000,
                _ => 256000,
            };

            encoder.complexity = (data[3] % 11) as i32;
            encoder.use_cbr = data[4] % 2 == 0;

            // Generate test input
            let samples_needed = frame_size * channels;
            let mut input = vec![0.0f32; samples_needed];

            // Fill with various audio patterns
            for i in 0..samples_needed {
                let pattern_type = data[5] % 6;
                input[i] = match pattern_type {
                    // Sine wave
                    0 => {
                        let t = i as f32 / sampling_rate as f32;
                        (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
                    }
                    // Mixed frequencies (more realistic for hybrid)
                    1 => {
                        let t = i as f32 / sampling_rate as f32;
                        let low = (2.0 * std::f32::consts::PI * 200.0 * t).sin();
                        let high = (2.0 * std::f32::consts::PI * 8000.0 * t).sin();
                        (low * 0.6 + high * 0.4) * 0.3
                    }
                    // From fuzz data
                    2 => {
                        let idx = (i + 6) % data.len();
                        (data[idx] as f32 - 128.0) / 256.0
                    }
                    // Silence
                    3 => 0.0,
                    // DC offset
                    4 => 0.1,
                    // Impulses
                    _ => {
                        if i % (frame_size / 10) == 0 { 0.8 } else { 0.0 }
                    }
                };

                // Clamp to valid range
                if !input[i].is_finite() {
                    input[i] = 0.0;
                }
                input[i] = input[i].clamp(-1.0, 1.0);
            }

            // Encode
            let mut encoded = vec![0u8; 4000];
            if let Ok(encoded_len) = encoder.encode(&input, *frame_size, &mut encoded) {
                if encoded_len > 0 {
                    // Decode
                    let mut decoded = vec![0.0f32; frame_size * channels];
                    let _ = decoder.decode(&encoded[..encoded_len], *frame_size, &mut decoded);

                    // Multiple decode passes to test state
                    for _ in 0..3 {
                        let _ = decoder.decode(&encoded[..encoded_len], *frame_size, &mut decoded);
                    }
                }
            }

            // Also test decoding raw data as hybrid packets
            let mut output = vec![0.0f32; frame_size * channels];
            let _ = decoder.decode(data, *frame_size, &mut output);
        }
    }
});
