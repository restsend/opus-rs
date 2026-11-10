#![no_main]

use libfuzzer_sys::fuzz_target;
use opus_rs::{Application, OpusDecoder, OpusEncoder};

// One encoder driven through a sequence of frames whose duration, settings,
// input and output buffer change between calls. Every packet must fit the
// buffer, decode in opus-rs to its duration with finite samples, and end in
// the encoder's final range.
fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let rate = [8000usize, 12000, 16000, 24000, 48000][data[0] as usize % 5];
    let channels = 1 + data[1] as usize % 2;
    let application = [
        Application::Voip,
        Application::Audio,
        Application::RestrictedLowDelay,
    ][data[2] as usize % 3];
    let mut encoder = OpusEncoder::new(rate as i32, channels, application).unwrap();
    let mut decoder = OpusDecoder::new(rate as i32, channels).unwrap();
    let mut packet = vec![0u8; 8000];
    let mut output = vec![0.0f32; 5760 * channels];
    let mut seed = u32::from_le_bytes([data[0], data[1], data[2], 1]);
    let mut cursor = 3;
    for _ in 0..24 {
        if cursor + 6 > data.len() {
            break;
        }
        let op = &data[cursor..cursor + 6];
        cursor += 6;
        // 2.5, 5, 10, 20, 40, 60, 80, 100 or 120 ms.
        let units = [1usize, 2, 4, 8, 16, 24, 32, 40, 48][op[0] as usize % 9];
        let frame = rate / 400 * units;
        encoder.bitrate_bps = 500 + (u16::from_le_bytes([op[1], op[2]]) as i32) * 8;
        encoder.complexity = (op[3] % 11) as i32;
        encoder.use_cbr = op[3] & 0x10 != 0;
        encoder.set_vbr_constraint(op[3] & 0x20 != 0);
        encoder.set_dtx(op[3] & 0x40 != 0);
        encoder.use_inband_fec = op[3] & 0x80 != 0;
        encoder.packet_loss_perc = (op[4] % 40) as i32;
        let room = [2usize, 3, 8, 40, 200, 1276, 8000][op[5] as usize % 7];
        // Input: tone, noise or digital silence at a fuzzed level, as f32 or i16.
        let level = (op[4] >> 6) as f32 / 3.0;
        let mut input = vec![0.0f32; frame * channels];
        for (i, x) in input.iter_mut().enumerate() {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *x = match op[5] >> 6 {
                0 => level * (i as f32 * 0.07).sin(),
                1 => level * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0),
                2 => 0.0,
                _ => level * (2.0 * ((seed >> 8) as f32 / (1u32 << 24) as f32) - 1.0).signum(),
            };
        }
        let result = if op[5] & 0x20 != 0 {
            let pcm: Vec<i16> = input.iter().map(|x| (x * 32767.0) as i16).collect();
            encoder.encode_i16(&pcm, frame, &mut packet[..room])
        } else {
            encoder.encode(&input, frame, &mut packet[..room])
        };
        if let Ok(n) = result {
            assert!(n >= 1 && n <= room, "{n} bytes for a buffer of {room}");
            let decoded = decoder
                .decode(&packet[..n], 5760, &mut output)
                .expect("encoder packet rejected");
            assert_eq!(decoded, frame);
            assert!(output[..decoded * channels].iter().all(|x| x.is_finite()));
            assert_eq!(decoder.final_range(), encoder.final_range());
        }
    }
});
