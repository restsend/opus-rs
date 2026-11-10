use std::path::Path;

pub fn speech_pcm() -> Vec<i16> {
    let wav = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/answer_16k.wav"))
        .expect("speech fixture");
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    let mut offset = 12;
    while offset + 8 <= wav.len() {
        let len = u32::from_le_bytes(wav[offset + 4..offset + 8].try_into().unwrap()) as usize;
        let data = &wav[offset + 8..offset + 8 + len];
        if &wav[offset..offset + 4] == b"fmt " {
            assert_eq!(u16::from_le_bytes(data[..2].try_into().unwrap()), 1);
            assert_eq!(u16::from_le_bytes(data[2..4].try_into().unwrap()), 1);
            assert_eq!(u32::from_le_bytes(data[4..8].try_into().unwrap()), 16000);
            assert_eq!(u16::from_le_bytes(data[14..16].try_into().unwrap()), 16);
        } else if &wav[offset..offset + 4] == b"data" {
            return data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                .collect();
        }
        offset += 8 + len + (len & 1);
    }
    panic!("speech fixture has no PCM data");
}

pub fn speech(sr: usize, channels: usize, samples: usize) -> Vec<f32> {
    let pcm = speech_pcm();
    (0..samples)
        .flat_map(|i| {
            let at = i * 16000 / sr;
            let fraction = (i * 16000 % sr) as f32 / sr as f32;
            let a = pcm[at] as f32;
            let b = pcm[(at + 1).min(pcm.len() - 1)] as f32;
            std::iter::repeat_n((a + fraction * (b - a)) / 32768.0, channels)
        })
        .collect()
}
