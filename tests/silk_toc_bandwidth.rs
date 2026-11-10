//! A SILK-only ToC must signal the rate SILK codes at.
//!
//! With in-band FEC and packet loss the encoder may lower a frame's
//! bandwidth so that FEC fits (libopus decide_fec). opus-rs kept coding SILK
//! at the API rate (up to 16 kHz) but wrote the lowered bandwidth into the
//! ToC, so decoders read 16 kHz SILK frames as 12 kHz ones: at 16 kHz with
//! 10 % loss and 20 kb/s CBR every packet said mediumband and decoded to
//! noise (-23 dB SNR). libopus lowers SILK's rate through its switching
//! state machine and takes the ToC bandwidth from the rate SILK actually
//! used, so every packet must decode in libopus to the encoder's final
//! range.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

fn signal(rate: usize, ch: usize, n: usize) -> Vec<f32> {
    let mut seed = 3u32;
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f32 / rate as f32;
        phase += 2.0 * PI * (140.0 + 30.0 * (2.0 * PI * 0.7 * t).sin()) / rate as f32;
        let mut v = 0.0;
        for h in 1..=20 {
            v += 0.15 / h as f32 * (h as f32 * phase).sin();
        }
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = (v + 0.04 * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0))
            * (0.55 + 0.45 * (2.0 * PI * 3.0 * t).sin());
        for c in 0..ch {
            out.push(if c == 0 { v } else { 0.7 * v });
        }
    }
    out
}

#[test]
fn fec_keeps_the_toc_at_the_silk_rate() {
    let (rate, frame) = (16000usize, 320usize);
    let mut pkt = [0u8; 1500];
    let mut buf = vec![0f32; 5760 * 2];
    for ch in [1usize, 2] {
        for bitrate in [16000, 20000, 24000] {
            let pcm = signal(rate, ch, frame * 60);
            let mut enc = OpusEncoder::new(rate as i32, ch, Application::Voip).unwrap();
            enc.bitrate_bps = bitrate;
            enc.use_cbr = true;
            enc.use_inband_fec = true;
            enc.packet_loss_perc = 10;
            let lag = enc.lookahead();
            let channels = if ch == 2 {
                Channels::Stereo
            } else {
                Channels::Mono
            };
            let mut dec = CDec::new(rate as u32, channels).unwrap();
            let mut out = Vec::new();
            for (f, x) in pcm.chunks_exact(frame * ch).enumerate() {
                let n = enc.encode(x, frame, &mut pkt).unwrap();
                let what = format!("{ch} ch, {bitrate} b/s, frame {f}: ToC {:#04x}", pkt[0]);
                // SILK-only, 20 ms: configuration 1 (narrowband), 5
                // (mediumband) or 9 (wideband).
                assert!([1, 5, 9].contains(&(pkt[0] >> 3)), "{what}");
                let k = dec.decode_float(&pkt[..n], &mut buf, false).unwrap();
                assert_eq!(dec.get_final_range().unwrap(), enc.final_range(), "{what}");
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
            assert!(snr > 3.0, "{ch} ch, {bitrate} b/s: SNR {snr:.1} dB");
        }
    }
}
