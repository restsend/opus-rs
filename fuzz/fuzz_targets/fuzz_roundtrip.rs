#![no_main]

use libfuzzer_sys::fuzz_target;
use opus_rs::{Application, OpusDecoder, OpusEncoder};

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }

    // Parse parameters
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

    let frame_size = match data[3] % 4 {
        0 => sampling_rate as usize / 400, // 2.5ms
        1 => sampling_rate as usize / 200, // 5ms
        2 => sampling_rate as usize / 100, // 10ms
        _ => sampling_rate as usize / 50,  // 20ms
    };

    if frame_size < 40 {
        return;
    }

    let mut encoder = match OpusEncoder::new(sampling_rate, channels, application) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut decoder = match OpusDecoder::new(sampling_rate, channels) {
        Ok(d) => d,
        Err(_) => return,
    };

    let samples_needed = frame_size * channels;
    let mut input = vec![0.0f32; samples_needed];

    for (i, sample) in input.iter_mut().enumerate() {
        let offset = 4 + (i * 4) % (data.len() - 4).max(1);
        if offset + 4 <= data.len() {
            let raw = f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            *sample = raw.clamp(-1.0, 1.0);
        }
    }

    let mut encoded = vec![0u8; 4000];
    let encoded_len = match encoder.encode(&input, frame_size, &mut encoded) {
        Ok(len) => len,
        Err(_) => return,
    };

    let mut decoded = vec![0.0f32; frame_size * channels];
    let _ = decoder.decode(&encoded[..encoded_len], frame_size, &mut decoded);
});
