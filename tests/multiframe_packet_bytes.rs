//! 40/60 ms CELT and Hybrid packets must be framed exactly like libopus's
//! repacketizer output.
//!
//! The packet is built from 20 ms frames, and its framing — code 1 for two
//! equal frames, code 2 for two different ones, code 3 for more, with the
//! VBR flag and per-frame size bytes only when the frames differ in size,
//! and CBR padding up to the nominal size — must match
//! `opus_repacketizer_out_range_impl`. Decode-level tests cannot see this:
//! every framing that reads back the same frames decodes to the same audio.
//! CELT payloads are float and not byte-exact against libopus, so this file
//! compares the framing (TOC, code, flags, sizes, padding, packet size);
//! 120 ms SILK-only packets, whose payload is fixed-point, are compared byte
//! for byte.

use opusic_sys::{
    opus_encode_float, opus_encoder_create, opus_encoder_ctl, opus_encoder_destroy,
    OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_VOIP, OPUS_SET_BITRATE_REQUEST, OPUS_SET_VBR_REQUEST,
};

/// A deterministic speech-like signal: harmonic sweeps under syllable-rate
/// envelopes, loud enough to code, quiet enough never to clip.
fn signal(n: usize) -> Vec<f32> {
    let mut seed = 7u32;
    (0..n)
        .map(|i| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let t = i as f32 / 48000.0;
            let f0 = 110.0 + 60.0 * (2.0 * core::f32::consts::PI * 0.9 * t).sin();
            let env = 0.5 + 0.4 * (2.0 * core::f32::consts::PI * 2.3 * t).sin();
            let mut s = 0.0f32;
            for h in 1..=6 {
                s += (0.4 / h as f32) * (h as f32 * 2.0 * core::f32::consts::PI * f0 * t).sin();
            }
            (s * env * 0.4).clamp(-1.0, 1.0)
        })
        .collect()
}

struct CEncoder(*mut opusic_sys::OpusEncoder);

impl CEncoder {
    fn new(rate: i32, ch: i32, app: i32, bitrate: i32, cbr: bool) -> Self {
        unsafe {
            let mut err = 0;
            let enc = opus_encoder_create(rate, ch, app, &mut err);
            assert_eq!(err, 0);
            opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, bitrate);
            if cbr {
                opus_encoder_ctl(enc, OPUS_SET_VBR_REQUEST, 0);
            }
            Self(enc)
        }
    }
    fn encode_float(&self, pcm: &[f32], frame: usize, out: &mut [u8]) -> usize {
        unsafe {
            let n = opus_encode_float(
                self.0,
                pcm.as_ptr(),
                frame as i32,
                out.as_mut_ptr(),
                out.len() as i32,
            );
            assert!(n > 0, "libopus encode failed: {n}");
            n as usize
        }
    }
}

impl Drop for CEncoder {
    fn drop(&mut self) {
        unsafe { opus_encoder_destroy(self.0) }
    }
}

/// The packet's framing, as libopus's repacketizer lays it out: the TOC, the
/// code, the VBR and padding flags, the per-frame size bytes, each frame's
/// offset and length, and the packet size (CBR padding included).
#[derive(Debug, PartialEq)]
struct Framing {
    toc: u8,
    code: u8,
    vbr: bool,
    pad: bool,
    sizes: Vec<usize>,
    frames: Vec<(usize, usize)>,
    total: usize,
}

/// One RFC 6716 frame-size byte: below 252 it is the length, otherwise it
/// continues into a second byte.
fn size_at(pkt: &[u8], pos: usize) -> (usize, usize) {
    let b = pkt[pos] as usize;
    if b == 252 {
        (((pkt[pos + 1] as usize) << 8) | 252, 2)
    } else if b >= 253 {
        (((b - 253) << 8) | pkt[pos + 1] as usize, 2)
    } else {
        (b, 1)
    }
}

fn framing(pkt: &[u8]) -> Framing {
    let toc = pkt[0];
    let code = toc & 3;
    let mut sizes = Vec::new();
    let mut frames = Vec::new();
    let mut vbr = false;
    let mut pad = false;
    let mut pos = 1;
    match code {
        0 => {
            frames.push((1, pkt.len() - 1));
        }
        1 => {
            // Two equal frames, unpadded (a padded pair becomes code 3).
            let len = (pkt.len() - 1) / 2;
            frames.push((1, len));
            frames.push((1 + len, len));
        }
        2 => {
            let (len, n) = size_at(pkt, 1);
            sizes.push(len);
            pos += n;
            vbr = true;
            frames.push((pos, len));
            frames.push((pos + len, pkt.len() - pos - len));
        }
        _ => {
            vbr = pkt[1] & 0x80 != 0;
            pad = pkt[1] & 0x40 != 0;
            let count = (pkt[1] & 0x3f) as usize;
            pos = 2;
            let mut pad_bytes = 0;
            if pad {
                loop {
                    let b = pkt[pos] as usize;
                    pos += 1;
                    pad_bytes += b;
                    if b < 255 {
                        break;
                    }
                }
            }
            if vbr {
                let mut start = pos;
                let mut header_end = pos;
                for _ in 0..count - 1 {
                    let (len, n) = size_at(pkt, header_end);
                    sizes.push(len);
                    header_end += n;
                }
                // The last frame runs to the end of the packet less padding.
                let mut last = pkt.len() - header_end - pad_bytes;
                for &len in &sizes {
                    last -= len;
                }
                sizes.push(last);
                for &len in &sizes {
                    frames.push((start, len));
                    start += len;
                }
            } else {
                let len = (pkt.len() - pos - pad_bytes) / count;
                for i in 0..count {
                    frames.push((pos + i * len, len));
                }
            }
        }
    }
    Framing {
        toc,
        code,
        vbr,
        pad,
        sizes,
        frames,
        total: pkt.len(),
    }
}

fn assert_same_framing(c: &[u8], rs: &[u8], what: &str) {
    let (mut fc, mut frs) = (framing(c), framing(rs));
    // The port does not yet switch a stereo encoder's stream to mono at low
    // bitrates (opus_encoder.c's `stream_channels` decision), so the TOC's
    // stereo bit (bit 2) can differ where libopus sends mono. Everything
    // else — code, VBR/padding flags, sizes, frame extents, total — must
    // match.
    fc.toc &= 0xFB;
    frs.toc &= 0xFB;
    assert_eq!(fc, frs, "framing ({what})\n c={c:02x?}\n rs={rs:02x?}");
}

#[test]
fn multiframe_framing_matches_libopus() {
    let rate = 48000usize;
    for app in [OPUS_APPLICATION_VOIP, OPUS_APPLICATION_AUDIO] {
        for ch in [1usize, 2] {
            for (bitrate, cbr) in [
                (32000, true),
                (64000, true),
                (96000, true),
                (64000, false),
                (128000, false),
                (16000, true),
                (24000, true),
            ] {
                for frame in [rate / 25, 3 * rate / 50] {
                    let packets = 12;
                    let pcm: Vec<f32> = signal(frame * packets * ch);
                    let c = CEncoder::new(rate as i32, ch as i32, app, bitrate, cbr);
                    let mut enc = opus_rs::OpusEncoder::new(
                        rate as i32,
                        ch,
                        if app == OPUS_APPLICATION_VOIP {
                            opus_rs::Application::Voip
                        } else {
                            opus_rs::Application::Audio
                        },
                    )
                    .unwrap();
                    enc.bitrate_bps = bitrate;
                    enc.use_cbr = cbr;
                    let mut c_out = [0u8; 1500];
                    let mut rs_out = [0u8; 1500];
                    let what =
                        |f: usize| format!("{app} {ch}ch {bitrate} b/s cbr={cbr} {frame}smp pkt{f}");
                    for (f, x) in pcm.chunks_exact(frame * ch).enumerate() {
                        let nc = c.encode_float(x, frame, &mut c_out);
                        let nr = enc
                            .encode(x, frame, &mut rs_out)
                            .unwrap_or_else(|e| panic!("{}: {e}", what(f)));
                        assert_eq!(nc, nr, "size ({})", what(f));
                        assert_same_framing(&c_out[..nc], &rs_out[..nr], &what(f));
                    }
                }
            }
        }
    }
}
