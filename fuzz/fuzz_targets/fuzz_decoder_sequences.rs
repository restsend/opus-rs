#![no_main]

use libfuzzer_sys::fuzz_target;
use opus_rs::OpusDecoder;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let rate = [8000, 12000, 16000, 24000, 48000][data[0] as usize % 5];
    let channels = 1 + data[1] as usize % 2;
    let mut decoder = OpusDecoder::new(rate, channels).unwrap();
    let mut output = vec![0.0; 5760 * channels];
    let mut cursor = 2;
    for _ in 0..64 {
        if cursor + 4 > data.len() {
            break;
        }
        let operation = data[cursor] % 3;
        let units = 1 + data[cursor + 1] as usize % 48;
        let length = u16::from_le_bytes([data[cursor + 2], data[cursor + 3]]) as usize;
        cursor += 4;
        let end = (cursor + length).min(data.len());
        let packet = &data[cursor..end];
        cursor = end;
        let capacity = if operation == 0 {
            rate as usize * 120 / 1000
        } else {
            units * (rate as usize / 400)
        };
        let result = match operation {
            0 => decoder.decode(packet, capacity, &mut output),
            1 => decoder.decode_fec(packet, capacity, &mut output),
            _ => decoder.decode(&[], capacity, &mut output),
        };
        if let Ok(samples) = result {
            assert!(samples <= capacity);
            assert!(output[..samples * channels].iter().all(|x| x.is_finite()));
        }
    }
});
