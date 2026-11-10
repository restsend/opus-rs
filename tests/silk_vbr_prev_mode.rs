//! SILK-only VBR packets must count as SILK for the next mode decision.
//!
//! The encoder recorded each packet's mode except for SILK-only VBR
//! packets, so after a single CELT frame every following SILK frame looked
//! like the first one after a switch from CELT, and the SILK encoder was
//! reset and primed again on each of them. Decoders carry their SILK state
//! on, so the stream lost its quality for good: about 5 to 7 dB of SNR.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder, Signal};
use std::f32::consts::PI;

fn signal(rate: usize, ch: usize, n: usize) -> Vec<f32> {
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f32 / rate as f32;
        phase += 2.0 * PI * (140.0 + 30.0 * (2.0 * PI * 0.7 * t).sin()) / rate as f32;
        let mut v = 0.0;
        for h in 1..=20 {
            v += 0.15 / h as f32 * (h as f32 * phase).sin();
        }
        let v = v * (0.55 + 0.45 * (2.0 * PI * 3.0 * t).sin());
        for c in 0..ch {
            out.push(if c == 0 { v } else { 0.7 * v });
        }
    }
    out
}

fn snr_after(ch: usize, celt_frame: Option<usize>) -> f64 {
    let (rate, frame) = (16000usize, 320usize);
    let pcm = signal(rate, ch, frame * 150);
    let mut enc = OpusEncoder::new(rate as i32, ch, Application::Voip).unwrap();
    enc.bitrate_bps = 24000;
    // Speech, coded in SILK at this rate.
    enc.set_signal(Signal::Voice);
    let lag = enc.lookahead();
    let channels = if ch == 2 {
        Channels::Stereo
    } else {
        Channels::Mono
    };
    let mut dec = CDec::new(rate as u32, channels).unwrap();
    let mut pkt = [0u8; 1500];
    let mut buf = vec![0f32; 5760 * 2];
    let mut out = Vec::new();
    for (f, x) in pcm.chunks_exact(frame * ch).enumerate() {
        // Tiny output buffers (below 6 kb/s) for two frames make the second
        // one CELT: libopus may code the first of them in the old mode to
        // end it with a redundant frame (to_celt).
        let small = celt_frame.is_some_and(|c| f == c || f == c + 1);
        let room = if small { 8 } else { 1500 };
        let n = enc.encode(x, frame, &mut pkt[..room]).unwrap();
        let celt = pkt[0] >= 0x80;
        if celt_frame.is_some_and(|c| f == c + 1) {
            assert!(celt, "frame {f} must be CELT");
        } else if !small {
            assert!(!celt, "frame {f} must be SILK");
        }
        let k = dec.decode_float(&pkt[..n], &mut buf, false).unwrap();
        out.extend_from_slice(&buf[..k * ch]);
    }
    let (mut s, mut e) = (0f64, 0f64);
    for i in 20 * frame * ch..pcm.len() - lag * ch {
        let (a, b) = (pcm[i] as f64, out[i + lag * ch] as f64);
        s += a * a;
        e += (a - b) * (a - b);
    }
    10.0 * (s / e).log10()
}

#[test]
fn silk_keeps_its_quality_after_a_celt_frame() {
    for ch in [1, 2] {
        let plain = snr_after(ch, None);
        let switched = snr_after(ch, Some(5));
        assert!(
            switched > plain - 1.0,
            "{ch} ch: {plain:.2} dB without the CELT frames, {switched:.2} dB with them"
        );
    }
}
