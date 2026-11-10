//! 10 ms SILK frames at 12 kHz must code exactly their 120 pulses.
//!
//! The pulses are coded in shell blocks of 16, so a 10 ms frame at 12 kHz
//! (120 samples) ends in half a block. libopus zeroes the rest of that block
//! (silk_encode_pulses). The encoder read it from its pulse buffer instead,
//! where a previous, longer frame can have left pulses: their LSBs and signs
//! were coded although the shell code gives the block no pulses there, and
//! decoders read the rest of the frame from the wrong place. That happens on
//! the first 10 ms frames after a switch from 20 ms or longer.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};

fn signal(n: usize, ch: usize, seed: u32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n * ch)
        .map(|i| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let noise = (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
            let square = if (i / ch / 9).is_multiple_of(2) {
                0.9
            } else {
                -0.9
            };
            0.7 * square + 0.3 * noise
        })
        .collect()
}

#[test]
fn ten_ms_frames_at_12_khz_after_longer_frames_decode_in_step() {
    let rate = 12000usize;
    let mut out = vec![0f32; 5760 * 2];
    let mut pkt = [0u8; 1500];
    let mut cases = 0;
    for ch in [1usize, 2] {
        let channels = if ch == 2 {
            Channels::Stereo
        } else {
            Channels::Mono
        };
        for complexity in 0..=10 {
            for (bitrate, cbr) in [
                (32000, true),
                (64000, false),
                (128000, true),
                (430000, false),
            ] {
                for long_ms in [20usize, 40, 60] {
                    let mut enc = OpusEncoder::new(rate as i32, ch, Application::Voip).unwrap();
                    enc.bitrate_bps = bitrate;
                    enc.use_cbr = cbr;
                    enc.complexity = complexity;
                    let mut dec = CDec::new(rate as u32, channels).unwrap();
                    let pcm = signal(rate, ch, 7 + complexity as u32);
                    // Two long frames, then four 10 ms frames.
                    let mut pos = 0;
                    for (f, ms) in [long_ms, long_ms, 10, 10, 10, 10].into_iter().enumerate() {
                        let n = rate * ms / 1000;
                        let x = &pcm[pos * ch..(pos + n) * ch];
                        pos += n;
                        let len = enc.encode(x, n, &mut pkt).unwrap();
                        let what = format!(
                            "{ch} ch, complexity {complexity}, {bitrate} b/s, {long_ms} ms then 10 ms, frame {f}"
                        );
                        assert_eq!(
                            dec.decode_float(&pkt[..len], &mut out, false).unwrap(),
                            n,
                            "{what}"
                        );
                        assert_eq!(
                            dec.get_final_range().unwrap(),
                            enc.final_range(),
                            "{what}: libopus decodes another frame"
                        );
                    }
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 264);
}

#[test]
fn pulses_past_the_frame_end_are_not_coded() {
    // A 10 ms frame at 12 kHz has 120 pulses, seven and a half shell blocks
    // of 16. The buffer's last eight entries may still hold a longer
    // frame's pulses; the frame must code as if they were zero.
    use opus_rs::range_coder::RangeCoder;
    use opus_rs::silk::encode_pulses::silk_encode_pulses;
    let mut clean = [0i8; 128];
    for (i, p) in clean.iter_mut().take(120).enumerate() {
        *p = [0, 1, -1, 2, 0, -3, 1, 0][i % 8];
    }
    let mut stale = clean;
    stale[120..].copy_from_slice(&[4, -2, 1, -1, 3, -5, 2, 1]);
    let code = |pulses: &[i8]| {
        let mut rc = RangeCoder::new_encoder(256);
        silk_encode_pulses(&mut rc, 2, 0, pulses, 120);
        rc.done();
        (rc.tell(), rc.buf[..256].to_vec())
    };
    assert_eq!(code(&stale), code(&clean));
}
