//! The Hybrid redundancy flag must be coded only where decoders read it.
//!
//! After the SILK layer of a Hybrid frame, decoders read the redundancy
//! flag only when at least 37 bits are left (RFC 6716 section 4.5.1,
//! libopus opus_decode_frame), and libopus's encoder writes it on the same
//! condition. The encoder always wrote it, so in frames whose SILK layer
//! left less room it coded a symbol the decoder never read, and the CELT
//! layer after it was decoded from a different range coder state.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

fn signal(rate: usize, n: usize) -> Vec<f32> {
    let mut seed = 7u32;
    let mut phase = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / rate as f32;
            phase += 2.0 * PI * (160.0 + 20.0 * (2.0 * PI * 0.8 * t).sin()) / rate as f32;
            let mut v = 0.0;
            for h in 1..=12 {
                v += 0.2 / h as f32 * (h as f32 * phase).sin();
            }
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (v + 0.05 * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0))
                * (0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin())
        })
        .collect()
}

#[test]
fn low_rate_hybrid_cbr_decodes_to_the_encoders_final_range() {
    let mut pkt = [0u8; 1500];
    let mut out = vec![0f32; 5760];
    let mut hybrid = 0;
    for rate in [48000usize, 24000] {
        for app in [Application::Voip, Application::Audio] {
            for bitrate in [6000, 8000, 10000, 12000] {
                for frame in [rate / 100, rate / 50] {
                    let pcm = signal(rate, frame * 50);
                    let mut enc = OpusEncoder::new(rate as i32, 1, app).unwrap();
                    enc.bitrate_bps = bitrate;
                    enc.use_cbr = true;
                    let mut dec = CDec::new(rate as u32, Channels::Mono).unwrap();
                    for (f, x) in pcm.chunks_exact(frame).enumerate() {
                        let n = enc.encode(x, frame, &mut pkt).unwrap();
                        hybrid += usize::from((12..16).contains(&(pkt[0] >> 3)));
                        dec.decode_float(&pkt[..n], &mut out, false).unwrap();
                        assert_eq!(
                            dec.get_final_range().unwrap(),
                            enc.final_range(),
                            "{rate} Hz, {app:?}, {bitrate} b/s, {frame} samples, frame {f}: {:02x?}",
                            &pkt[..n.min(4)]
                        );
                    }
                }
            }
        }
    }
    assert!(hybrid > 500, "{hybrid} Hybrid packets");
}
