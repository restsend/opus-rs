//! CELT frames at 12 kHz must be signalled as wideband.
//!
//! CELT has no mediumband, and a ToC cannot signal a mediumband CELT frame:
//! libopus codes such frames as wideband. At 12 kHz the encoder coded the
//! 17 wideband bands but wrote a narrowband ToC, so decoders read 13 bands
//! and took the rest of each frame for something else: every CELT packet at
//! 12 kHz (restricted low delay, frames too short or budgets too small for
//! SILK) decoded to noise, about -3 dB SNR.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

fn signal(rate: usize, ch: usize, n: usize) -> Vec<f32> {
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f32 / rate as f32;
        phase += 2.0 * PI * (180.0 + 40.0 * (2.0 * PI * 0.5 * t).sin()) / rate as f32;
        let mut v = 0.0;
        for h in 1..=15 {
            v += 0.1 / h as f32 * (h as f32 * phase).sin();
        }
        for c in 0..ch {
            out.push(if c == 0 { v } else { 0.7 * v });
        }
    }
    out
}

#[test]
fn twelve_khz_celt_is_wideband_and_decodes() {
    let rate = 12000usize;
    let frame = rate / 50;
    let mut pkt = [0u8; 1500];
    let mut buf = vec![0f32; 5760 * 2];
    for ch in [1usize, 2] {
        for bitrate in [24000, 48000] {
            let pcm = signal(rate, ch, frame * 100);
            let mut enc =
                OpusEncoder::new(rate as i32, ch, Application::RestrictedLowDelay).unwrap();
            enc.bitrate_bps = bitrate;
            let lag = enc.lookahead();
            let channels = if ch == 2 {
                Channels::Stereo
            } else {
                Channels::Mono
            };
            let mut dec = CDec::new(rate as u32, channels).unwrap();
            let mut out = Vec::new();
            for x in pcm.chunks_exact(frame * ch) {
                let n = enc.encode(x, frame, &mut pkt).unwrap();
                // CELT-only, wideband: configurations 20 to 23.
                assert_eq!(
                    pkt[0] >> 3,
                    23,
                    "{ch} ch, {bitrate} b/s: ToC {:#04x}",
                    pkt[0]
                );
                let k = dec.decode_float(&pkt[..n], &mut buf, false).unwrap();
                out.extend_from_slice(&buf[..k * ch]);
            }
            let skip = 5 * frame * ch;
            let (mut s, mut e) = (0f64, 0f64);
            for i in skip..pcm.len() - lag * ch {
                let (a, b) = (pcm[i] as f64, out[i + lag * ch] as f64);
                s += a * a;
                e += (a - b) * (a - b);
            }
            let snr = 10.0 * (s / e).log10();
            assert!(snr > 8.0, "{ch} ch, {bitrate} b/s: SNR {snr:.1} dB");
        }
    }
}
