//! A Hybrid frame's CELT intensity band must not lie below its start band.
//!
//! Stereo CELT codes, from an intensity band up, only the mid and the
//! inversion flags; the band is coded relative to the first coded band, 17
//! in a Hybrid frame. The encoder clamped its choice to bands 0 and up, so
//! at Hybrid rates where it picked a band below 17 the coded value wrapped
//! around: the decoder read another band, and other symbols from there on.
//! libopus clamps the band to the coded range.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

fn signal(rate: usize, n: usize) -> Vec<f32> {
    let mut phase = 0.0f32;
    let mut seed = 5u32;
    let mut out = Vec::with_capacity(2 * n);
    for i in 0..n {
        let t = i as f32 / rate as f32;
        phase += 2.0 * PI * (150.0 + 30.0 * (2.0 * PI * 0.6 * t).sin()) / rate as f32;
        let mut v = 0.0;
        for h in 1..=30 {
            v += 0.1 / h as f32 * (h as f32 * phase).sin();
        }
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let n = 0.05 * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0);
        out.push(v + n);
        out.push(0.6 * v - n);
    }
    out
}

#[test]
fn hybrid_stereo_decodes_to_the_encoders_final_range() {
    let mut pkt = [0u8; 1500];
    let mut out = vec![0f32; 5760 * 2];
    let mut hybrid = 0;
    for rate in [48000usize, 24000] {
        for app in [Application::Voip, Application::Audio] {
            for bitrate in [24000, 32000] {
                let frame = rate / 50;
                let pcm = signal(rate, frame * 40);
                let mut enc = OpusEncoder::new(rate as i32, 2, app).unwrap();
                enc.bitrate_bps = bitrate;
                enc.use_cbr = true;
                let mut dec = CDec::new(rate as u32, Channels::Stereo).unwrap();
                for (f, x) in pcm.chunks_exact(frame * 2).enumerate() {
                    let n = enc.encode(x, frame, &mut pkt).unwrap();
                    hybrid += usize::from((12..16).contains(&(pkt[0] >> 3)));
                    dec.decode_float(&pkt[..n], &mut out, false).unwrap();
                    assert_eq!(
                        dec.get_final_range().unwrap(),
                        enc.final_range(),
                        "{rate} Hz, {app:?}, {bitrate} b/s, frame {f}"
                    );
                }
            }
        }
    }
    assert!(hybrid > 100, "{hybrid} Hybrid packets");
}
