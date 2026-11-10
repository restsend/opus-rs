//! SILK must follow a change of frame duration between packets.
//!
//! The SILK encoder set its packet layout (frames per packet, subframes per
//! frame) only when it was created or changed rate. After a switch from 20
//! to 40 ms it went on coding one 20 ms frame under a 40 ms ToC, after a
//! switch the other way it wrote a two-frame header for one frame, and some
//! switches panicked on the input slices. Decoders read such packets as
//! noise. libopus reconfigures SILK for every frame and drops the previous
//! layout's FEC data.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

fn signal(rate: usize, ch: usize, n: usize) -> Vec<f32> {
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f32 / rate as f32;
        phase += 2.0 * PI * (130.0 + 25.0 * (2.0 * PI * 0.9 * t).sin()) / rate as f32;
        let mut v = 0.0;
        for h in 1..=20 {
            v += 0.15 / h as f32 * (h as f32 * phase).sin();
        }
        let v = v * (0.6 + 0.4 * (2.0 * PI * 2.5 * t).sin());
        for c in 0..ch {
            out.push(if c == 0 { v } else { 0.8 * v });
        }
    }
    out
}

#[test]
fn changing_the_frame_duration_keeps_silk_in_sync() {
    let mut pkt = [0u8; 1500];
    let mut out = vec![0f32; 5760 * 2];
    for rate in [16000usize, 8000] {
        for ch in [1usize, 2] {
            for fec in [false, true] {
                let pcm = signal(rate, ch, rate * 6);
                let mut enc = OpusEncoder::new(rate as i32, ch, Application::Voip).unwrap();
                enc.bitrate_bps = 24000;
                enc.use_inband_fec = fec;
                enc.packet_loss_perc = if fec { 10 } else { 0 };
                let channels = if ch == 2 {
                    Channels::Stereo
                } else {
                    Channels::Mono
                };
                let mut dec = CDec::new(rate as u32, channels).unwrap();
                let durations_ms = [20, 40, 60, 10, 20, 60, 40, 10, 60, 20];
                let mut pos = 0usize;
                for (i, ms) in durations_ms.iter().cycle().take(40).enumerate() {
                    let frame = rate * ms / 1000;
                    for _ in 0..3 {
                        let n = enc
                            .encode(&pcm[pos * ch..(pos + frame) * ch], frame, &mut pkt)
                            .unwrap();
                        assert_eq!(dec.decode_float(&pkt[..n], &mut out, false).unwrap(), frame);
                        assert_eq!(
                            dec.get_final_range().unwrap(),
                            enc.final_range(),
                            "{rate} Hz, {ch} ch, fec {fec}, switch {i} to {ms} ms"
                        );
                        pos += frame;
                    }
                }
            }
        }
    }
}
