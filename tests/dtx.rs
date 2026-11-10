//! Discontinuous transmission, as libopus `OPUS_SET_DTX` and
//! `OPUS_GET_IN_DTX`.
//!
//! With DTX on, libopus sends a packet of the ToC byte alone for each frame
//! after 200 ms of inactivity, which decoders conceal with comfort noise,
//! and codes one frame every 400 ms to refresh the background. SILK's voice
//! activity detection decides it for SILK and Hybrid frames (counting SILK
//! frames: 10 frames of silence first, then 20 in DTX for each coded one);
//! libopus's own decision, in milliseconds, covers frames of digital
//! silence when its signal analysis runs (complexity 7 and up, 16 kHz and
//! up), and VoIP with DTX codes in SILK.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

/// Two seconds of the speech fixture at `rate` (decimated or repeated from
/// 16 kHz), then `silence_s` seconds of noise at `floor`.
fn fixture_then(rate: usize, floor: f32, silence_s: usize) -> Vec<f32> {
    let wav = std::fs::read("fixtures/answer_16k.wav").unwrap();
    let speech: Vec<f32> = wav[44..44 + 2 * 2 * 16000]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&b| i16::from_le_bytes(b) as f32 / 32768.0)
        .collect();
    let mut out: Vec<f32> = (0..2 * rate).map(|i| speech[i * 16000 / rate]).collect();
    let mut seed = 1u32;
    out.extend((0..silence_s * rate).map(|_| {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        floor * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0)
    }));
    out
}

/// One second of speech-like sound, then `silence_s` of the given floor.
fn speech_then(rate: usize, floor: f32, silence_s: usize) -> Vec<f32> {
    let mut phase = 0.0f32;
    let mut seed = 1u32;
    (0..rate * (1 + silence_s))
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = floor * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0);
            if i >= rate {
                return noise;
            }
            let t = i as f32 / rate as f32;
            phase += 2.0 * PI * (150.0 + 30.0 * (2.0 * PI * 0.7 * t).sin()) / rate as f32;
            let mut v = 0.0;
            for h in 1..=12 {
                v += 0.12 / h as f32 * (h as f32 * phase).sin();
            }
            v * (0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin()) + noise
        })
        .collect()
}

/// Packet lengths for `pcm` in `frame`-sample frames; every packet must
/// decode in libopus. Also returns `in_dtx()` after each packet.
fn lengths(
    enc: &mut OpusEncoder,
    rate: usize,
    pcm: &[f32],
    frame: usize,
) -> (Vec<usize>, Vec<bool>) {
    let mut dec = CDec::new(rate as u32, Channels::Mono).unwrap();
    let mut pkt = [0u8; 1500];
    let mut out = vec![0f32; 5760];
    let (mut lens, mut dtx) = (Vec::new(), Vec::new());
    for x in pcm.chunks_exact(frame) {
        let n = enc.encode(x, frame, &mut pkt).unwrap();
        assert_eq!(dec.decode_float(&pkt[..n], &mut out, false).unwrap(), frame);
        if n == 1 {
            assert_eq!(enc.final_range(), 0);
        } else {
            assert_eq!(dec.get_final_range().unwrap(), enc.final_range());
        }
        lens.push(n);
        dtx.push(enc.in_dtx());
    }
    (lens, dtx)
}

#[test]
fn dtx_is_off_by_default() {
    let enc = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
    assert!(!enc.dtx());
    assert!(!enc.in_dtx());
}

#[test]
fn silk_dtx_follows_voice_activity() {
    // Complexity 5: SILK's DTX, driven by its VAD, for a quiet noise floor.
    for (rate, frame_ms) in [(16000usize, 20usize), (8000, 20), (16000, 10), (48000, 20)] {
        let frame = rate * frame_ms / 1000;
        let pcm = fixture_then(rate, 0.0005, 3);
        let mut enc = OpusEncoder::new(rate as i32, 1, Application::Voip).unwrap();
        enc.bitrate_bps = 24000;
        enc.complexity = 5;
        let (plain, _) = lengths(&mut enc, rate, &pcm, frame);
        assert!(
            plain.iter().all(|&n| n > 1),
            "{rate} Hz: DTX without set_dtx"
        );

        let mut enc = OpusEncoder::new(rate as i32, 1, Application::Voip).unwrap();
        enc.bitrate_bps = 24000;
        enc.complexity = 5;
        enc.set_dtx(true);
        assert!(enc.dtx());
        let (lens, dtx) = lengths(&mut enc, rate, &pcm, frame);
        let speech = 2000 / frame_ms;
        assert!(
            lens[..speech].iter().all(|&n| n > 1),
            "{rate} Hz: DTX during speech: {lens:?}"
        );
        // In the silence: 10 SILK frames coded, then 20 ToC-only packets for
        // each coded one.
        let quiet = &lens[speech + 15..];
        let toc_only = quiet.iter().filter(|&&n| n == 1).count();
        assert!(
            toc_only * 21 >= quiet.len() * 19,
            "{rate} Hz {frame_ms} ms: {toc_only} of {} ToC-only: {lens:?}",
            quiet.len()
        );
        let first = lens[speech..].iter().position(|&n| n == 1).unwrap();
        assert!(
            (10..20).contains(&first),
            "{rate} Hz {frame_ms} ms: first DTX packet {first} frames into the silence"
        );
        let coded: Vec<usize> = quiet
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 1)
            .map(|(i, _)| i)
            .collect();
        for w in coded.windows(2) {
            assert_eq!(
                w[1] - w[0],
                21,
                "{rate} Hz {frame_ms} ms: refresh interval {lens:?}"
            );
        }
        assert!(
            dtx[speech + 20..].iter().all(|&d| d),
            "{rate} Hz: in_dtx during silence"
        );
        assert!(!dtx[speech / 2], "{rate} Hz: in_dtx during speech");
    }
}

#[test]
fn digital_silence_dtx_counts_milliseconds() {
    // Complexity 9 at 48 kHz: digital silence goes through libopus's own
    // DTX, in CELT as in SILK: 200 ms coded, then one coded frame per 400 ms.
    for (app, frame_ms) in [
        (Application::Audio, 20usize),
        (Application::Audio, 10),
        (Application::Voip, 20),
    ] {
        let rate = 48000usize;
        let frame = rate * frame_ms / 1000;
        let pcm = speech_then(rate, 0.0, 3);
        let mut enc = OpusEncoder::new(rate as i32, 1, app).unwrap();
        enc.bitrate_bps = 64000;
        enc.complexity = 9;
        enc.set_dtx(true);
        let (lens, dtx) = lengths(&mut enc, rate, &pcm, frame);
        let speech = 1000 / frame_ms;
        let first = lens[speech..].iter().position(|&n| n == 1).unwrap();
        assert_eq!(
            first,
            200 / frame_ms,
            "{app:?} {frame_ms} ms: first DTX packet: {lens:?}"
        );
        let coded: Vec<usize> = lens[speech + first..]
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 1)
            .map(|(i, _)| i)
            .collect();
        assert!(!coded.is_empty());
        for w in coded.windows(2) {
            assert_eq!(
                (w[1] - w[0]) * frame_ms,
                400 + frame_ms,
                "{app:?} {frame_ms} ms: refresh interval: {lens:?}"
            );
        }
        assert!(
            dtx[speech + first..].iter().all(|&d| d),
            "{app:?}: in_dtx during silence"
        );
    }
}
