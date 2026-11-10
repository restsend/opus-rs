//! 80, 100 and 120 ms frames, as libopus accepts them.
//!
//! libopus encodes them as several shorter frames in one packet: SILK as
//! 2 x 40 ms, 5 x 20 ms or 2 x 60 ms, CELT and Hybrid as 20 ms frames
//! (encode_multiframe_packet). opus-rs rejected them as an invalid frame
//! size.

use opus::{Channels, Decoder as CDec, packet};
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

fn signal(rate: usize, ch: usize, n: usize) -> Vec<f32> {
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f32 / rate as f32;
        phase += 2.0 * PI * (150.0 + 30.0 * (2.0 * PI * 0.7 * t).sin()) / rate as f32;
        let mut v = 0.0;
        for h in 1..=12 {
            v += 0.12 / h as f32 * (h as f32 * phase).sin();
        }
        let v = v * (0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin());
        for c in 0..ch {
            out.push(if c == 0 { v } else { 0.7 * v });
        }
    }
    out
}

#[test]
fn long_frames_encode_in_every_mode() {
    let mut pkt = vec![0u8; 8000];
    let mut out = vec![0f32; 5760 * 2];
    let mut modes = String::new();
    for rate in [8000usize, 16000, 48000] {
        for ch in [1usize, 2] {
            for app in [
                Application::Voip,
                Application::Audio,
                Application::RestrictedLowDelay,
            ] {
                for ms in [80usize, 100, 120] {
                    for (bitrate, cbr) in [(16000, false), (64000, false), (32000, true)] {
                        let frame = rate * ms / 1000;
                        let pcm = signal(rate, ch, frame * 4);
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
                            let what = format!(
                                "{rate} Hz, {ch} ch, {app:?}, {ms} ms, {bitrate} b/s, cbr {cbr}, frame {f}"
                            );
                            let n = enc
                                .encode(x, frame, &mut pkt)
                                .unwrap_or_else(|e| panic!("{what}: {e}"));
                            assert_eq!(
                                packet::get_nb_samples(&pkt[..n], rate as u32).unwrap(),
                                frame,
                                "{what}"
                            );
                            assert_eq!(
                                dec.decode_float(&pkt[..n], &mut out, false).unwrap(),
                                frame,
                                "{what}"
                            );
                            assert_eq!(dec.get_final_range().unwrap(), enc.final_range(), "{what}");
                            let mode = match pkt[0] >> 3 {
                                0..=11 => 'S',
                                12..=15 => 'H',
                                _ => 'C',
                            };
                            if !modes.contains(mode) {
                                modes.push(mode);
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(modes.len(), 3, "modes: {modes}");
}
