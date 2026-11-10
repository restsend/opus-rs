//! Constrained VBR can be switched off, as with libopus
//! `OPUS_SET_VBR_CONSTRAINT`.
//!
//! libopus constrains CELT-only VBR by default, so that every frame fits a
//! buffer of one frame at the target bitrate, and lets the application turn
//! that off; opus-rs always constrained it. Unconstrained frames vary more
//! around the target (a transient takes more than the constraint allows),
//! and VBR frames stay within 510 kb/s, as libopus caps them.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder, Signal};

/// Noise bursts with sharp onsets over a steady tone: frames that want very
/// different sizes.
fn signal(rate: usize, n: usize) -> Vec<f32> {
    let mut seed = 21u32;
    (0..n)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
            let age = (i % (rate / 3)) as f32 / rate as f32;
            0.1 * (i as f32 * 0.05).sin() + 0.6 * noise * (-age * 30.0).exp()
        })
        .collect()
}

fn sizes(constrained: bool, rate: usize, frame: usize, bitrate: i32) -> Vec<usize> {
    let pcm = signal(rate, frame * 200);
    let mut enc = OpusEncoder::new(rate as i32, 1, Application::Audio).unwrap();
    // Music, coded in CELT-only at these rates.
    enc.set_signal(Signal::Music);
    assert!(enc.vbr_constraint(), "constrained by default");
    enc.set_vbr_constraint(constrained);
    assert_eq!(enc.vbr_constraint(), constrained);
    enc.bitrate_bps = bitrate;
    let mut dec = CDec::new(rate as u32, Channels::Mono).unwrap();
    let mut pkt = [0u8; 1500];
    let mut out = vec![0f32; 5760];
    pcm.chunks_exact(frame)
        .map(|x| {
            let n = enc.encode(x, frame, &mut pkt).unwrap();
            assert!(pkt[0] >= 0x80, "CELT-only expected");
            assert_eq!(dec.decode_float(&pkt[..n], &mut out, false).unwrap(), frame);
            assert_eq!(dec.get_final_range().unwrap(), enc.final_range());
            n
        })
        .collect()
}

#[test]
fn unconstrained_vbr_varies_more_and_keeps_the_rate() {
    for (rate, frame, bitrate) in [
        (48000usize, 960usize, 64000),
        (48000, 480, 96000),
        (24000, 480, 48000),
    ] {
        let target = bitrate as f64 * frame as f64 / rate as f64 / 8.0;
        let c = sizes(true, rate, frame, bitrate);
        let u = sizes(false, rate, frame, bitrate);
        let mean = |v: &[usize]| v[10..].iter().sum::<usize>() as f64 / (v.len() - 10) as f64;
        let max = |v: &[usize]| *v[10..].iter().max().unwrap() as f64;
        println!(
            "{rate} Hz, {frame}, {bitrate}: target {target:.0}, constrained mean {:.1} max {}, unconstrained mean {:.1} max {}",
            mean(&c),
            max(&c),
            mean(&u),
            max(&u)
        );
        for (name, v) in [("constrained", &c), ("unconstrained", &u)] {
            assert!(
                (mean(v) / target - 1.0).abs() < 0.2,
                "{name}: mean {:.1} for a target of {target:.1}",
                mean(v)
            );
        }
        assert!(
            max(&c) <= 2.0 * target + 2.0,
            "constrained frames stay within two frames' budget: {}",
            max(&c)
        );
        assert!(
            max(&u) > max(&c),
            "unconstrained frames vary more: {} vs {}",
            max(&u),
            max(&c)
        );
    }
}

#[test]
fn vbr_frames_stay_within_510_kbps() {
    for (frame, cap) in [(120usize, 159usize), (240, 318), (480, 637)] {
        let pcm = signal(48000, frame * 50);
        let mut enc = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
        enc.bitrate_bps = 510000;
        enc.set_vbr_constraint(false);
        let mut pkt = [0u8; 1500];
        for x in pcm.chunks_exact(frame) {
            let n = enc.encode(x, frame, &mut pkt).unwrap();
            assert!(n <= cap + 1, "{frame} samples: {n} bytes");
        }
    }
}
