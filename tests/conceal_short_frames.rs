//! Frames of 0 or 1 bytes must be concealed, as libopus does.
//!
//! libopus conceals a frame of 0 or 1 bytes in the mode of the frame before
//! it ("Payloads of 1 (2 including ToC) or 0 trigger the PLC/DTX",
//! opus_decode_frame). Such frames come from encoders whose budget ran out
//! (libopus and opus-rs then send a single zero byte) and from multi-frame
//! packets with empty frames. opus-rs decoded the byte as SILK or CELT data,
//! and CELT empty frames as silence, so its output differed from libopus's
//! there and for several frames after (down to 1 dB from libopus). Such a
//! frame must now decode exactly like a lost one; for SILK, whose
//! concealment matches libopus's exactly, also like libopus.

use opus::{Application as CApp, Bitrate, Channels, Decoder as CDec, Encoder as CEnc};
use opus_rs::OpusDecoder;
use std::f32::consts::PI;

fn signal(rate: usize, n: usize) -> Vec<f32> {
    let mut phase = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / rate as f32;
            phase += 2.0 * PI * (150.0 + 30.0 * (2.0 * PI * 0.7 * t).sin()) / rate as f32;
            let mut v = 0.0;
            for h in 1..=12 {
                v += 0.12 / h as f32 * (h as f32 * phase).sin();
            }
            v * (0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin())
        })
        .collect()
}

fn snr(a: &[f32], b: &[f32]) -> f64 {
    let (mut s, mut e) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        s += (*x as f64).powi(2);
        e += (*x as f64 - *y as f64).powi(2);
    }
    if e == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (s / e).log10()
    }
}

#[test]
fn short_frames_are_concealed_like_libopus() {
    // (name, rate, application, bitrate): SILK, Hybrid and CELT streams.
    let streams = [
        ("SILK", 16000usize, CApp::Voip, 16000),
        ("Hybrid", 48000, CApp::Voip, 24000),
        ("CELT", 48000, CApp::Audio, 64000),
    ];
    // (name, packet 15 rebuilt from its ToC byte).
    type Edit = (&'static str, fn(u8) -> Vec<u8>);
    let edits: [Edit; 4] = [
        ("one zero byte", |toc| vec![toc & 0xfc, 0x00]),
        ("one byte", |toc| vec![toc & 0xfc, 0xa5]),
        ("code 1, two 1-byte frames", |toc| {
            vec![(toc & 0xfc) | 1, 0x5a, 0xa5]
        }),
        ("code 3, one empty frame", |toc| vec![toc | 3, 0x01]),
    ];
    for (name, rate, app, bitrate) in streams {
        let frame = rate / 50;
        let pcm = signal(rate, frame * 30);
        let mut enc = CEnc::new(rate as u32, Channels::Mono, app).unwrap();
        enc.set_bitrate(Bitrate::Bits(bitrate)).unwrap();
        let mut pkt = [0u8; 1500];
        let packets: Vec<Vec<u8>> = pcm
            .chunks_exact(frame)
            .map(|x| {
                let n = enc.encode_float(x, &mut pkt).unwrap();
                pkt[..n].to_vec()
            })
            .collect();
        for (edit, make) in edits {
            let mut stream = packets.clone();
            stream[15] = make(packets[15][0]);
            let frames = if stream[15][0] & 3 == 1 { 2 } else { 1 };
            let mut c_dec = CDec::new(rate as u32, Channels::Mono).unwrap();
            let mut rs_dec = OpusDecoder::new(rate as i32, 1).unwrap();
            // The same stream with packet 15 lost instead.
            let mut lost_dec = OpusDecoder::new(rate as i32, 1).unwrap();
            let (mut a, mut b, mut c) = (vec![0f32; 5760], vec![0f32; 5760], vec![0f32; 5760]);
            for (i, p) in stream.iter().enumerate() {
                let n = c_dec.decode_float(p, &mut a, false).unwrap();
                assert_eq!(
                    rs_dec.decode(p, 5760, &mut b),
                    Ok(n),
                    "{name}, {edit}, packet {i}"
                );
                let lost: &[u8] = if i == 15 { &[] } else { p };
                lost_dec.decode(lost, frames * frame, &mut c).unwrap();
                if i >= 15 {
                    assert_eq!(
                        b[..n],
                        c[..n],
                        "{name}, {edit}, packet {i}: not decoded as a lost frame"
                    );
                    // Concealment is not normative (RFC 6716 section 4.4);
                    // opus-rs's SILK concealment matches libopus's exactly.
                    if name == "SILK" {
                        let db = snr(&a[..n], &b[..n]);
                        assert!(
                            db > 60.0,
                            "{name}, {edit}, packet {i}: {db:.1} dB from libopus"
                        );
                    }
                }
            }
        }
    }
}
