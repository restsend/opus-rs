#![no_main]

use libfuzzer_sys::fuzz_target;
use opus_rs::{Application, OpusEncoder};

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }

    // Parse fuzz input to determine test parameters
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

    let frame_size_multiplier = (data[3] % 4 + 1) as usize;
    let frame_size = (sampling_rate as usize / 100) * frame_size_multiplier; // 10ms, 20ms, 30ms, 40ms

    // Create encoder
    let mut encoder = match OpusEncoder::new(sampling_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };

    // Convert remaining bytes to f32 samples
    let samples_needed = frame_size * channels;
    let mut input = vec![0.0f32; samples_needed];

    // Fill input with data interpreted as f32, with clamping
    for (i, sample) in input.iter_mut().enumerate() {
        if i * 4 + 4 <= data.len() {
            let bytes = [
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ];
            let raw = f32::from_le_bytes(bytes);
            // Clamp to reasonable audio range to avoid inf/nan
            *sample = raw.clamp(-10.0, 10.0);
        } else {
            // Use smaller pattern for remaining samples
            let idx = i % (data.len() - 4).max(1);
            *sample = (data[idx] as f32 - 128.0) / 128.0;
        }
    }

    // Encode
    let mut output = vec![0u8; 4000];
    let _ = encoder.encode(&input, frame_size, &mut output);
});
