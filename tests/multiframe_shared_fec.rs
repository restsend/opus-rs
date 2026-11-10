//! The 20 ms frames of a 40/60 ms Hybrid packet must share one bandwidth.
//!
//! A 40 or 60 ms Hybrid packet is built from 20 ms frames under the first
//! frame's ToC. With in-band FEC each frame decided again whether to lower
//! the bandwidth to make room for FEC, and the hysteresis of that decision
//! changes after the first frame: the first frame went super-wideband, the
//! next ones fullband, all under a super-wideband ToC, and decoders read the
//! later frames with the wrong band layout. libopus decides bandwidth and FEC
//! once per packet.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

fn voice(rate: usize, n: usize) -> Vec<f32> {
    let mut seed = 5u32;
    let mut phase = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / rate as f32;
            phase += 2.0 * PI * (140.0 + 30.0 * (2.0 * PI * 0.7 * t).sin()) / rate as f32;
            let mut v = 0.0;
            for h in (1..=20).take_while(|h| 140.0 * *h as f32 <= rate as f32 / 2.0) {
                v += 0.15 / h as f32 * (h as f32 * phase).sin();
            }
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
            (v + 0.04 * noise) * (0.55 + 0.45 * (2.0 * PI * 3.0 * t).sin())
        })
        .collect()
}

#[test]
fn split_hybrid_packets_with_fec_decode_in_step() {
    let rate = 48000usize;
    let pcm = voice(rate, rate * 2);
    let mut out = vec![0f32; 5760];
    let mut pkt = [0u8; 1500];
    // (frame ms, bitrate, CBR, loss %, complexity)
    let cases = [
        (40usize, 27000, false, 15, 5),
        (40, 27000, false, 30, 5),
        (40, 27000, false, 30, 9),
        (40, 27000, true, 30, 5),
        (40, 32000, false, 15, 2),
        (40, 32000, true, 15, 0),
        (40, 32000, true, 15, 5),
        (40, 32000, true, 30, 2),
        (40, 40000, false, 15, 0),
        (40, 40000, true, 30, 0),
        (40, 24000, false, 30, 9),
        (60, 24000, false, 30, 5),
        (60, 27000, false, 30, 5),
        (60, 32000, false, 15, 2),
        (60, 32000, true, 15, 5),
    ];
    for (ms, bitrate, cbr, loss, complexity) in cases {
        let frame = rate * ms / 1000;
        let mut enc = OpusEncoder::new(rate as i32, 1, Application::Voip).unwrap();
        enc.bitrate_bps = bitrate;
        enc.use_cbr = cbr;
        enc.complexity = complexity;
        enc.use_inband_fec = true;
        enc.packet_loss_perc = loss;
        let mut dec = CDec::new(rate as u32, Channels::Mono).unwrap();
        for (f, x) in pcm.chunks_exact(frame).enumerate().take(8) {
            let what = format!(
                "{ms} ms, {bitrate} b/s, CBR {cbr}, {loss} % loss, complexity {complexity}, packet {f}"
            );
            let n = enc.encode(x, frame, &mut pkt).unwrap();
            assert_eq!(
                dec.decode_float(&pkt[..n], &mut out, false).unwrap(),
                frame,
                "{what}"
            );
            assert_eq!(
                dec.get_final_range().unwrap(),
                enc.final_range(),
                "{what}: libopus decodes other frames (ToC {:#04x})",
                pkt[0]
            );
        }
    }
}
