//! CELT frames that code no transient flag must be coded as not transient.
//!
//! The flag needs frames of 5 ms or more and three bits of room; a decoder
//! takes a frame without it as not transient. The encoder kept the decision
//! of its transient analysis anyway and coded the frame's time-frequency
//! flags with the transient probabilities (and its allocation with the
//! transient's anti-collapse reserve), so from that point it wrote other
//! symbols than a decoder reads: 2.5 ms frames with an onset decoded to
//! noise. libopus clears the decision when no flag is coded.
//!
//! Clicks and noise bursts are coded in 2.5 ms frames at 24 and 48 kHz;
//! libopus's decoder must end every packet in the encoder's final range
//! state.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};

/// A quiet tone with a decaying noise burst every 23 ms.
fn bursts(rate: usize, ch: usize, n: usize) -> Vec<f32> {
    let mut seed = 0x8251u32;
    let period = rate * 23 / 1000;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
        let age = (i % period) as f32 / rate as f32;
        let v = 0.05 * (i as f32 * 0.03).sin() + 0.7 * noise * (-age * 400.0).exp();
        for c in 0..ch {
            out.push(if c == 0 { v } else { 0.8 * v });
        }
    }
    out
}

#[test]
fn short_frames_with_onsets_decode_to_the_encoders_final_range() {
    let mut pkt = [0u8; 1500];
    let mut out = vec![0f32; 5760 * 2];
    let mut checked = 0;
    for rate in [48000usize, 24000] {
        for ch in [1usize, 2] {
            // RestrictedLowDelay codes every frame in CELT; Audio does at
            // 48 kHz and these rates.
            for app in [Application::Audio, Application::RestrictedLowDelay] {
                for bitrate in [64000, 128000] {
                    for cbr in [false, true] {
                        let frame = rate / 400;
                        let pcm = bursts(rate, ch, frame * 120);
                        let mut enc = OpusEncoder::new(rate as i32, ch, app).unwrap();
                        enc.bitrate_bps = bitrate;
                        enc.use_cbr = cbr;
                        let channels = if ch == 2 {
                            Channels::Stereo
                        } else {
                            Channels::Mono
                        };
                        let mut dec = CDec::new(rate as u32, channels).unwrap();
                        for (f, x) in pcm.chunks_exact(frame * ch).enumerate() {
                            let n = enc.encode(x, frame, &mut pkt).unwrap();
                            dec.decode_float(&pkt[..n], &mut out, false).unwrap();
                            assert_eq!(
                                dec.get_final_range().unwrap(),
                                enc.final_range(),
                                "{rate} Hz, {ch} ch, {app:?}, {bitrate} b/s, cbr {cbr}, frame {f}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(checked, 32 * 120);
}
