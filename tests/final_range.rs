//! `final_range()` reports the range coder's final state, as libopus
//! `OPUS_GET_FINAL_RANGE` does.
//!
//! An encoder and a decoder that agree on every symbol of a frame end in the
//! same range coder state, which is how Opus implementations check each
//! other: libopus's `opus_demo` compares the decoder's value with the one
//! each official test vector stores per packet. The decoder must report what
//! libopus's encoder reports for the same packets (multi-frame packets and
//! the redundant frames of mode switches included), the encoder what
//! libopus's decoder reports for its packets, and a concealed frame 0.

use opus::{Application as CApp, Bitrate, Channels, Decoder as CDec, Encoder as CEnc};
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

fn channels(ch: usize) -> Channels {
    if ch == 2 {
        Channels::Stereo
    } else {
        Channels::Mono
    }
}

/// A drifting harmonic series with some noise under a syllable envelope;
/// the right channel is a quieter copy.
fn signal(rate: usize, ch: usize, n: usize) -> Vec<f32> {
    let mut seed = 0x1234_5678u32;
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f32 / rate as f32;
        phase += 2.0 * PI * (150.0 + 30.0 * (2.0 * PI * 0.7 * t).sin()) / rate as f32;
        let mut v = 0.0;
        for h in 1..=12 {
            v += 0.12 / h as f32 * (h as f32 * phase).sin();
        }
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        v += 0.03 * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0);
        let v = v * (0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin());
        out.push(v);
        if ch == 2 {
            out.push(0.6 * v);
        }
    }
    out
}

#[test]
fn encoder_final_range_matches_libopus_decoder() {
    // (rate, channels, application, bitrate, frame in 0.1 ms, CBR)
    let cases = [
        (16000, 1, Application::Voip, 24000, 200, false),
        (8000, 2, Application::Voip, 16000, 200, true),
        (48000, 1, Application::Voip, 24000, 200, false),
        (48000, 2, Application::Audio, 96000, 200, false),
        (24000, 1, Application::Audio, 64000, 100, true),
        (48000, 1, Application::Audio, 96000, 600, false),
    ];
    let mut pkt = [0u8; 1500];
    let mut out = vec![0f32; 5760 * 2];
    for (rate, ch, app, bitrate, tenths, cbr) in cases {
        let frame = rate * tenths / 10000;
        let pcm = signal(rate, ch, frame * 20);
        let mut enc = OpusEncoder::new(rate as i32, ch, app).unwrap();
        enc.bitrate_bps = bitrate;
        enc.use_cbr = cbr;
        let mut c_dec = CDec::new(rate as u32, channels(ch)).unwrap();
        let mut rs_dec = OpusDecoder::new(rate as i32, ch).unwrap();
        for (f, x) in pcm.chunks_exact(frame * ch).enumerate() {
            let n = enc.encode(x, frame, &mut pkt).unwrap();
            c_dec.decode_float(&pkt[..n], &mut out, false).unwrap();
            rs_dec.decode(&pkt[..n], 5760, &mut out).unwrap();
            let what = format!("{rate} Hz, {ch} ch, {app:?}, {bitrate} b/s, frame {f}");
            assert_ne!(enc.final_range(), 0, "{what}");
            assert_eq!(
                enc.final_range(),
                c_dec.get_final_range().unwrap(),
                "libopus, {what}"
            );
            assert_eq!(rs_dec.final_range(), enc.final_range(), "opus-rs, {what}");
        }
    }
}

#[test]
fn decoder_final_range_matches_libopus_encoder() {
    let mut pkt = [0u8; 4000];
    let mut out = vec![0f32; 5760 * 2];
    for (rate, ch) in [(48000usize, 2usize), (48000, 1), (16000, 1)] {
        let mut enc = CEnc::new(rate as u32, channels(ch), CApp::Voip).unwrap();
        let mut dec = OpusDecoder::new(rate as i32, ch).unwrap();
        let pcm = signal(rate, ch, rate * 4);
        let (mut pos, mut f) = (0usize, 0usize);
        let mut seen = String::new();
        while pos + rate * 3 / 50 <= rate * 4 {
            // 10, 20, 40 and 60 ms frames; SILK, Hybrid and CELT bitrates,
            // whose switches carry redundant frames.
            let frame = [rate / 100, rate / 50, rate / 25, 3 * rate / 50][(f / 5) % 4];
            let bitrate = [10000, 20000, 64000, 128000][(f / 9) % 4];
            enc.set_bitrate(Bitrate::Bits(bitrate)).unwrap();
            let n = enc
                .encode_float(&pcm[pos * ch..(pos + frame) * ch], &mut pkt)
                .unwrap();
            dec.decode(&pkt[..n], 5760, &mut out).unwrap();
            let want = enc.get_final_range().unwrap();
            assert_eq!(
                dec.final_range(),
                want,
                "{rate} Hz, {ch} ch, packet {f}: {:02x?}",
                &pkt[..n.min(4)]
            );
            let mode = match pkt[0] >> 3 {
                0..=11 => 'S',
                12..=15 => 'H',
                _ => 'C',
            };
            if !seen.contains(mode) {
                seen.push(mode);
            }
            pos += frame;
            f += 1;
        }
        if rate == 48000 {
            assert_eq!(seen.len(), 3, "modes seen: {seen}");
        }
    }
}

#[test]
fn concealed_frames_report_zero() {
    let mut enc = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let pcm = signal(48000, 1, 960);
    let mut pkt = [0u8; 1500];
    let mut out = vec![0f32; 960];
    let n = enc.encode(&pcm, 960, &mut pkt).unwrap();
    dec.decode(&pkt[..n], 960, &mut out).unwrap();
    assert_eq!(dec.final_range(), enc.final_range());
    assert_ne!(dec.final_range(), 0);
    dec.decode(&[], 960, &mut out).unwrap();
    assert_eq!(dec.final_range(), 0, "lost packet");
    dec.decode(&pkt[..n], 960, &mut out).unwrap();
    dec.decode(&pkt[..1], 960, &mut out).unwrap();
    assert_eq!(dec.final_range(), 0, "ToC-only packet");
}
