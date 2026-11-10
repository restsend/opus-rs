//! 40 and 60 ms CELT and Hybrid packets must carry only their frames.
//!
//! These packets are built from 20 ms frames, each first encoded as a packet
//! of its own. Under CBR such a sub-packet can be a code 3 packet: a frame
//! below 3 bytes, or over its budget, is sent empty and padded to the CBR
//! size. The encoder copied each sub-packet after its ToC as the frame, so
//! the code 3 header and padding went out as audio data and decoders decoded
//! them instead of concealing an empty frame. libopus's repacketizer takes
//! each sub-packet's frame, gives every frame an equal share of the CBR size
//! less the header, and pads the packet to exactly that size.
//!
//! In VBR the output buffer was not shared out either: each 20 ms frame
//! could take 1275 bytes, so a 60 ms packet at 256 kb/s needed about 1900
//! bytes and failed with "Output buffer too small" in a 1500-byte buffer.
//! libopus caps each frame at its share of the buffer and at the bitrate's
//! bytes per frame.

use opus::{Channels, Decoder as CDec};
use opus_rs::{Application, OpusEncoder};

fn signal(n: usize) -> Vec<f32> {
    let mut seed = 11u32;
    (0..n)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            0.3 * (i as f32 * 0.05).sin() + 0.1 * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0)
        })
        .collect()
}

#[test]
fn vbr_multiframe_packets_fit_the_output_buffer() {
    let rate = 48000usize;
    let mut pkt = [0u8; 1500];
    let mut out = vec![0f32; 5760 * 2];
    for ch in [1usize, 2] {
        for (bitrate, room) in [(256000, 1500usize), (510000, 1500), (128000, 400)] {
            for frame in [rate / 25, 3 * rate / 50] {
                let pcm: Vec<f32> = signal(frame * 10 * ch);
                let mut enc = OpusEncoder::new(rate as i32, ch, Application::Audio).unwrap();
                enc.bitrate_bps = bitrate;
                let channels = if ch == 2 {
                    Channels::Stereo
                } else {
                    Channels::Mono
                };
                let mut dec = CDec::new(rate as u32, channels).unwrap();
                for (f, x) in pcm.chunks_exact(frame * ch).enumerate() {
                    let what = format!(
                        "{ch} ch, {bitrate} b/s, {frame} samples in {room} bytes, frame {f}"
                    );
                    let n = enc
                        .encode(x, frame, &mut pkt[..room])
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert_eq!(
                        dec.decode_float(&pkt[..n], &mut out, false).unwrap(),
                        frame,
                        "{what}"
                    );
                    assert_eq!(dec.get_final_range().unwrap(), enc.final_range(), "{what}");
                }
            }
        }
    }
}

#[test]
fn cbr_multiframe_packets_hold_their_frames_at_the_cbr_size() {
    let rate = 48000usize;
    let mut pkt = [0u8; 1500];
    let mut out = vec![0f32; 5760];
    for app in [Application::Audio, Application::RestrictedLowDelay] {
        // 1 kb/s gives 20 ms frames below 3 bytes, sent empty.
        for bitrate in [1000, 48000, 96000] {
            for frame in [rate / 25, 3 * rate / 50] {
                let pcm = signal(frame * 20);
                let mut enc = OpusEncoder::new(rate as i32, 1, app).unwrap();
                enc.bitrate_bps = bitrate;
                enc.use_cbr = true;
                let cbr = (bitrate as usize * frame / rate + 4) / 8;
                let mut dec = CDec::new(rate as u32, Channels::Mono).unwrap();
                for (f, x) in pcm.chunks_exact(frame).enumerate() {
                    let n = enc.encode(x, frame, &mut pkt).unwrap();
                    let what = format!(
                        "{app:?}, {bitrate} b/s, {frame} samples, frame {f}: {:02x?}",
                        &pkt[..n.min(6)]
                    );
                    assert_eq!(n, cbr, "{what}");
                    assert_eq!(
                        dec.decode_float(&pkt[..n], &mut out, false).unwrap(),
                        frame,
                        "{what}"
                    );
                    assert_eq!(dec.get_final_range().unwrap(), enc.final_range(), "{what}");
                }
            }
        }
    }
}
