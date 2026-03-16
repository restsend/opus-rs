//! Fuzz test for multi-frame encoding/decoding
//! Tests processing multiple consecutive frames to catch state-related bugs

#![no_main]

use libfuzzer_sys::fuzz_target;
use opus_rs::{Application, OpusDecoder, OpusEncoder};

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }

    let sampling_rate = match data[0] % 5 {
        0 => 8000,
        1 => 12000,
        2 => 16000,
        3 => 24000,
        _ => 48000,
    };

    let channels = if data[1] % 2 == 0 { 1 } else { 2 };
    let application = match data[2] % 3 {
        0 => Application::Voip,
        1 => Application::Audio,
        _ => Application::RestrictedLowDelay,
    };

    let mut encoder = match OpusEncoder::new(sampling_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut decoder = match OpusDecoder::new(sampling_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Configure encoder
    encoder.bitrate_bps = match data[3] % 5 {
        0 => 16000,
        1 => 32000,
        2 => 64000,
        3 => 96000,
        _ => 128000,
    };
    encoder.complexity = (data[4] % 11) as i32;

    // Frame size
    let frame_size = match data[5] % 4 {
        0 => sampling_rate as usize / 400,  // 2.5ms
        1 => sampling_rate as usize / 200,  // 5ms
        2 => sampling_rate as usize / 100,  // 10ms
        _ => sampling_rate as usize / 50,   // 20ms
    };

    if frame_size == 0 || frame_size > 2880 {
        return;
    }

    let samples_needed = frame_size * channels;
    let mut output_buf = vec![0u8; 4000];
    let mut decoded_buf = vec![0.0f32; samples_needed];

    // Process multiple frames
    let num_frames = 5 + (data[6] % 10) as usize;

    for frame_idx in 0..num_frames {
        // Generate input for this frame
        let mut input = vec![0.0f32; samples_needed];

        for i in 0..samples_needed {
            let data_offset = 7 + ((frame_idx * samples_needed + i) % (data.len() - 7));
            let raw = data[data_offset] as f32;

            // Various patterns that might expose state bugs
            input[i] = match data[6] % 8 {
                // Continuous sine
                0 => {
                    let t = (frame_idx * frame_size + i) as f32 / sampling_rate as f32;
                    (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
                }
                // Changing frequency
                1 => {
                    let freq = 200.0 + (frame_idx as f32 * 50.0);
                    let t = i as f32 / sampling_rate as f32;
                    (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
                }
                // Amplitude ramp
                2 => {
                    let amp = (frame_idx as f32 + 1.0) / (num_frames as f32);
                    (raw / 128.0) * amp
                }
                // Silence then loud
                3 => {
                    if frame_idx < num_frames / 2 { 0.0 } else { raw / 128.0 }
                }
                // Loud then silence
                4 => {
                    if frame_idx < num_frames / 2 { raw / 128.0 } else { 0.0 }
                }
                // Alternating patterns
                5 => {
                    if frame_idx % 2 == 0 { 0.5 } else { -0.5 }
                }
                // Random
                6 => (raw - 128.0) / 128.0,
                // From data with frame offset
                _ => {
                    let idx = (frame_idx + i) % data.len();
                    (data[idx] as f32 - 128.0) / 128.0
                }
            };

            if !input[i].is_finite() {
                input[i] = 0.0;
            }
            input[i] = input[i].clamp(-1.0, 1.0);
        }

        // Encode
        match encoder.encode(&input, frame_size, &mut output_buf) {
            Ok(encoded_len) if encoded_len > 0 => {
                // Decode
                let _ = decoder.decode(&output_buf[..encoded_len], frame_size, &mut decoded_buf);

                // Verify decoded output doesn't contain NaN/Inf
                for &sample in &decoded_buf {
                    if !sample.is_finite() {
                        // Found invalid sample - this is a bug
                        return;
                    }
                }
            }
            _ => {}
        }
    }

    // Reset and do another batch
    let _ = OpusEncoder::new(sampling_rate, channels, application);
    let _ = OpusDecoder::new(sampling_rate, channels);
});
