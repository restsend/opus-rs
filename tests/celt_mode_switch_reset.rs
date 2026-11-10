//! The first CELT frames after a mode switch must decode at the right level.
//!
//! At a switch into CELT or Hybrid mode, libopus's encoder resets its CELT
//! encoder, primes it with the 2.5 ms before the frame and codes the frame
//! without inter-frame prediction (opus_encode_frame_native), as decoders
//! restart their CELT state there. opus-rs kept its CELT state, so the
//! energy of the first CELT frame after a Hybrid stretch was predicted from
//! stale values: about 12 dB too quiet in libopus's decoder, 8 dB in the
//! next frame.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};

#[test]
fn hybrid_to_celt_switches_keep_the_level() {
    let (rate, frame) = (48000usize, 960usize);
    let mut seed = 5u32;
    let pcm: Vec<f32> = (0..frame * 300)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            0.3 * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0)
        })
        .collect();
    let mut enc = OpusEncoder::new(rate as i32, 1, Application::Voip).unwrap();
    let lag = enc.lookahead();
    let mut dec = CDec::new(rate as u32, Channels::Mono).unwrap();
    let mut pkt = [0u8; 1500];
    let mut buf = vec![0f32; 5760];
    let (mut out, mut modes) = (Vec::new(), Vec::new());
    for (f, x) in pcm.chunks_exact(frame).enumerate() {
        enc.bitrate_bps = if (f / 30) % 2 == 0 { 96000 } else { 16000 };
        let n = enc.encode(x, frame, &mut pkt).unwrap();
        modes.push(pkt[0] >= 0x80);
        let k = dec.decode_float(&pkt[..n], &mut buf, false).unwrap();
        out.extend_from_slice(&buf[..k]);
    }
    let level_db = |f: usize| {
        let (mut s, mut d) = (0f64, 0f64);
        for i in f * frame..(f + 1) * frame {
            s += (pcm[i] as f64).powi(2);
            d += (out[i + lag] as f64).powi(2);
        }
        10.0 * (d / s).log10()
    };
    let mut switches = 0;
    for f in 1..modes.len() - 2 {
        if modes[f] && !modes[f - 1] {
            switches += 1;
            for g in [f, f + 1] {
                let db = level_db(g);
                assert!(
                    db.abs() < 3.0,
                    "frame {g}, after the switch at {f}: level {db:+.1} dB"
                );
            }
        }
    }
    assert!(switches >= 4, "{switches} switches to CELT");
}
