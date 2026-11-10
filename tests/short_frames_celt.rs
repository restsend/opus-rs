//! 2.5 and 5 ms frames must encode in every application, in CELT.
//!
//! SILK and Hybrid frames are 10 ms or longer. libopus overrides the mode
//! choice to CELT for shorter frames, in every application and at every
//! bandwidth (opus_encode_native); opus-rs returned "Unsupported frame
//! size" whenever its mode choice was SILK or Hybrid, which is most VoIP and
//! Audio configurations.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};

#[test]
fn short_frames_encode_as_celt_in_every_application() {
    let mut pkt = [0u8; 1500];
    let mut out = vec![0f32; 5760 * 2];
    for rate in [8000usize, 12000, 16000, 24000, 48000] {
        for ch in [1usize, 2] {
            for app in [
                Application::Voip,
                Application::Audio,
                Application::RestrictedLowDelay,
            ] {
                for (units, config) in [(1usize, 0u8), (2, 1)] {
                    for bitrate in [12000, 64000] {
                        let frame = rate / 400 * units;
                        let mut enc = OpusEncoder::new(rate as i32, ch, app).unwrap();
                        enc.bitrate_bps = bitrate;
                        let channels = if ch == 2 {
                            Channels::Stereo
                        } else {
                            Channels::Mono
                        };
                        let mut dec = CDec::new(rate as u32, channels).unwrap();
                        for f in 0..10 {
                            let x: Vec<f32> = (0..frame * ch)
                                .map(|i| 0.3 * ((f * frame * ch + i) as f32 * 0.07).sin())
                                .collect();
                            let what = format!(
                                "{rate} Hz, {ch} ch, {app:?}, {} ms, {bitrate} b/s, frame {f}",
                                units as f32 * 2.5
                            );
                            let n = enc
                                .encode(&x, frame, &mut pkt)
                                .unwrap_or_else(|e| panic!("{what}: {e}"));
                            assert!(pkt[0] >= 0x80, "{what}: not CELT ({:#04x})", pkt[0]);
                            assert_eq!((pkt[0] >> 3) & 3, config, "{what}: frame duration");
                            assert_eq!(
                                dec.decode_float(&pkt[..n], &mut out, false).unwrap(),
                                frame,
                                "{what}"
                            );
                        }
                    }
                }
            }
        }
    }
}
