//! A packet lost right after a switch to CELT must be concealed in CELT.
//!
//! libopus ends the last SILK or Hybrid frame before a switch to CELT with a
//! redundant 5 ms CELT frame. If the first CELT packet is then lost, its
//! decoder conceals it in CELT, continuing from that frame, and decodes the
//! following CELT packets on from there ("Run PLC using last used mode (CELT
//! if we ended with CELT redundancy)", opus_decode_frame). opus-rs concealed
//! such a packet in SILK and then decoded the next CELT packets from the
//! state of the redundant frame, 20 ms back.
//!
//! The oracle is libopus 1.6.1 (`opusic-sys`), whose concealment this crate
//! ports; the `opus` crate's older libopus does not have this rule. Stereo
//! streams are compared: from the 5 ms of history the redundant frame leaves,
//! opus-rs's periodic concealment of a mono stream does not follow libopus's
//! closely enough for a fixed bound.

use opus_rs::OpusDecoder;
use opusic_sys as ffi;
use std::f32::consts::PI;
use std::ffi::c_int;

struct Encoder(*mut ffi::OpusEncoder);

impl Encoder {
    fn new(rate: usize, channels: usize) -> Self {
        let mut err: c_int = 0;
        let ptr = unsafe {
            ffi::opus_encoder_create(
                rate as i32,
                channels as c_int,
                ffi::OPUS_APPLICATION_VOIP,
                &mut err,
            )
        };
        assert!(!ptr.is_null() && err == 0, "opus_encoder_create: {err}");
        Self(ptr)
    }

    fn encode(&mut self, pcm: &[f32], frame: usize, bitrate: i32) -> Vec<u8> {
        let mut out = [0u8; 1500];
        let n = unsafe {
            assert_eq!(
                ffi::opus_encoder_ctl(self.0, ffi::OPUS_SET_BITRATE_REQUEST, bitrate as c_int),
                0
            );
            ffi::opus_encode_float(self.0, pcm.as_ptr(), frame as c_int, out.as_mut_ptr(), 1500)
        };
        assert!(n > 0, "opus_encode_float: {n}");
        out[..n as usize].to_vec()
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        unsafe { ffi::opus_encoder_destroy(self.0) }
    }
}

struct Decoder(*mut ffi::OpusDecoder, usize);

impl Decoder {
    fn new(rate: usize, channels: usize) -> Self {
        let mut err: c_int = 0;
        let ptr = unsafe { ffi::opus_decoder_create(rate as i32, channels as c_int, &mut err) };
        assert!(!ptr.is_null() && err == 0, "opus_decoder_create: {err}");
        Self(ptr, channels)
    }

    /// An empty packet conceals one frame of `out.len() / channels` samples.
    fn decode(&mut self, data: &[u8], out: &mut [f32]) -> usize {
        let (ptr, len) = if data.is_empty() {
            (std::ptr::null(), 0)
        } else {
            (data.as_ptr(), data.len() as c_int)
        };
        let n = unsafe {
            ffi::opus_decode_float(
                self.0,
                ptr,
                len,
                out.as_mut_ptr(),
                (out.len() / self.1) as c_int,
                0,
            )
        };
        assert!(n > 0, "opus_decode_float: {n}");
        n as usize
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe { ffi::opus_decoder_destroy(self.0) }
    }
}

fn signal(rate: usize, ch: usize, n: usize) -> Vec<f32> {
    let mut seed = 7u32;
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f32 / rate as f32;
        phase += 2.0 * PI * (140.0 + 30.0 * (2.0 * PI * 0.7 * t).sin()) / rate as f32;
        let mut v = 0.0;
        for h in (1..=20).take_while(|h| 140.0 * *h as f32 <= rate as f32 / 2.0) {
            v += 0.15 / h as f32 * (h as f32 * phase).sin();
        }
        let mut noise = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        };
        let x = (v + 0.04 * noise()) * (0.55 + 0.45 * (2.0 * PI * 3.0 * t).sin());
        out.push(x);
        if ch == 2 {
            out.push(0.7 * x + 0.03 * noise());
        }
    }
    out
}

fn snr(a: &[f32], b: &[f32]) -> f64 {
    let (mut s, mut e) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        s += (*x as f64).powi(2);
        e += (*x as f64 - *y as f64).powi(2);
    }
    10.0 * (s / e.max(1e-30)).log10()
}

#[test]
fn loss_after_a_switch_to_celt_is_concealed_in_celt() {
    for (rate, ch) in [
        (48000usize, 2usize),
        (24000, 2),
        (16000, 2),
        (12000, 2),
        (8000, 2),
    ] {
        let frame = rate / 50;
        let pcm = signal(rate, ch, rate * 4);
        let mut enc = Encoder::new(rate, ch);
        // 12 and 128 kb/s in turn: SILK or Hybrid, then CELT.
        let packets: Vec<Vec<u8>> = pcm
            .chunks_exact(frame * ch)
            .enumerate()
            .map(|(f, x)| enc.encode(x, frame, if (f / 25) % 2 == 0 { 12000 } else { 128000 }))
            .collect();
        let celt = |p: &[u8]| p[0] & 0x80 != 0;
        let mut switches = 0;
        for f in 1..packets.len() - 3 {
            if !celt(&packets[f]) || celt(&packets[f - 1]) {
                continue;
            }
            switches += 1;
            let mut c_dec = Decoder::new(rate, ch);
            let mut rs_dec = OpusDecoder::new(rate as i32, ch).unwrap();
            let (mut a, mut b) = (vec![0f32; 5760 * 2], vec![0f32; 5760 * 2]);
            for (i, p) in packets.iter().enumerate().take(f + 3) {
                let n = if i == f {
                    c_dec.decode(&[], &mut a[..frame * ch]);
                    rs_dec.decode(&[], frame, &mut b).unwrap()
                } else {
                    c_dec.decode(p, &mut a);
                    rs_dec.decode(p, 5760, &mut b).unwrap()
                };
                if i >= f {
                    let db = snr(&a[..n * ch], &b[..n * ch]);
                    assert!(
                        db > 40.0,
                        "{rate} Hz, {ch} ch, packet {f} lost, packet {i}: {db:.1} dB from libopus"
                    );
                }
            }
        }
        assert!(switches >= 3, "{rate} Hz, {ch} ch: {switches} switches");
    }
}
