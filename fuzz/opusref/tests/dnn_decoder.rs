//! The opus-rs decoder with its neural network features against the
//! DNN-enabled libopus decoder at the same complexity, on packets of the
//! reference encoder with losses.
#![cfg(feature = "dnn")]
use opusref::dnn::{Rng, speechlike};
use opusref::*;

const OPUS_SET_COMPLEXITY: i32 = 4010;
const OPUS_SET_BITRATE: i32 = 4002;
const OPUS_SET_MAX_BANDWIDTH: i32 = 4004;
const OPUS_SET_PACKET_LOSS_PERC: i32 = 4014;
const OPUS_SET_INBAND_FEC: i32 = 4012;

/// One encoder setup.
#[derive(Clone, Copy, Debug)]
pub struct Setup {
    pub fs: i32,
    pub channels: usize,
    pub app: i32,
    pub bitrate: i32,
    pub max_bandwidth: i32,
    /// Frame length in ms (2.5 ms units are not used here).
    pub frame_ms: usize,
}

/// Packets of the reference encoder for `seconds` of a speech-like signal.
pub fn encode(s: Setup, seconds: usize, seed: u64) -> Vec<Vec<u8>> {
    let sig16 = speechlike(16000 * seconds, seed);
    let n = s.fs as usize * seconds;
    let mut pcm = vec![0f32; n * s.channels];
    for i in 0..n {
        let v = sig16[(i * 16000 / s.fs as usize).min(sig16.len() - 1)] / 32768.0;
        for c in 0..s.channels {
            pcm[i * s.channels + c] = if c == 0 { v } else { 0.8 * v };
        }
    }
    let frame = s.fs as usize * s.frame_ms / 1000;
    let mut out = Vec::new();
    unsafe {
        let mut err = 0;
        let enc = opus_encoder_create(s.fs, s.channels as i32, s.app, &mut err);
        assert_eq!(err, 0);
        opus_encoder_ctl(enc, OPUS_SET_BITRATE, s.bitrate);
        opus_encoder_ctl(enc, OPUS_SET_MAX_BANDWIDTH, s.max_bandwidth);
        opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY, 9);
        opus_encoder_ctl(enc, OPUS_SET_INBAND_FEC, 0);
        opus_encoder_ctl(enc, OPUS_SET_PACKET_LOSS_PERC, 0);
        for f in pcm.chunks_exact(frame * s.channels) {
            let mut pkt = [0u8; 1500];
            let len = opus_encode_float(enc, f.as_ptr(), frame as i32, pkt.as_mut_ptr(), 1500);
            assert!(len > 0);
            out.push(pkt[..len as usize].to_vec());
        }
        opus_encoder_destroy(enc);
    }
    out
}

/// Decodes `packets` (None = lost) with both decoders at `complexity` and
/// returns the largest difference and the index of the first frame that
/// differs.
pub fn compare(s: Setup, packets: &[Option<Vec<u8>>], complexity: i32) -> (f32, Option<usize>) {
    let frame = s.fs as usize * s.frame_ms / 1000;
    let mut rs = opus_rs::OpusDecoder::new(s.fs, s.channels).unwrap();
    rs.set_complexity(complexity).unwrap();
    let mut maxd = 0f32;
    let mut first = None;
    unsafe {
        let mut err = 0;
        let dec = opus_decoder_create(s.fs, s.channels as i32, &mut err);
        opus_decoder_ctl(dec, OPUS_SET_COMPLEXITY, complexity);
        for (k, p) in packets.iter().enumerate() {
            let mut a = vec![0f32; frame * s.channels];
            let mut b = vec![0f32; frame * s.channels];
            let (ptr, len) = match p {
                Some(p) => (p.as_ptr(), p.len() as i32),
                None => (core::ptr::null(), 0),
            };
            let na = opus_decode_float(dec, ptr, len, a.as_mut_ptr(), frame as i32, 0);
            let nb = rs.decode(p.as_deref().unwrap_or(&[]), frame, &mut b).unwrap();
            assert_eq!(na as usize, nb, "frame {k}");
            for (x, y) in a.iter().zip(&b) {
                let d = (x - y).abs();
                if d > 0.0 && first.is_none() {
                    first = Some(k);
                }
                maxd = maxd.max(d);
            }
        }
        opus_decoder_destroy(dec);
    }
    (maxd, first)
}

/// Bursts of 1 to `max_burst` lost packets, a burst starting with
/// probability `rate` per packet; the first second is never lost.
pub fn with_losses(packets: Vec<Vec<u8>>, seed: u64, rate: u32, max_burst: u32, frame_ms: usize) -> Vec<Option<Vec<u8>>> {
    let mut r = Rng(seed);
    let mut left = 0;
    packets
        .into_iter()
        .enumerate()
        .map(|(k, p)| {
            if k * frame_ms >= 1000 && left == 0 && r.below(100) < rate {
                left = 1 + r.below(max_burst);
            }
            if left > 0 {
                left -= 1;
                None
            } else {
                Some(p)
            }
        })
        .collect()
}

pub const SILK_WB_16K: Setup =
    Setup { fs: 16000, channels: 1, app: 2048, bitrate: 20000, max_bandwidth: 1103, frame_ms: 20 };
pub const SILK_WB_48K: Setup =
    Setup { fs: 48000, channels: 1, app: 2048, bitrate: 20000, max_bandwidth: 1103, frame_ms: 20 };
pub const SILK_WB_16K_10MS: Setup =
    Setup { fs: 16000, channels: 1, app: 2048, bitrate: 20000, max_bandwidth: 1103, frame_ms: 10 };
pub const SILK_NB_8K: Setup =
    Setup { fs: 8000, channels: 1, app: 2048, bitrate: 12000, max_bandwidth: 1101, frame_ms: 20 };
pub const SILK_WB_STEREO: Setup =
    Setup { fs: 16000, channels: 2, app: 2048, bitrate: 32000, max_bandwidth: 1103, frame_ms: 20 };
pub const CELT_48K: Setup =
    Setup { fs: 48000, channels: 1, app: 2051, bitrate: 48000, max_bandwidth: 1105, frame_ms: 20 };
pub const CELT_48K_STEREO: Setup =
    Setup { fs: 48000, channels: 2, app: 2051, bitrate: 64000, max_bandwidth: 1105, frame_ms: 20 };
pub const CELT_48K_10MS: Setup =
    Setup { fs: 48000, channels: 1, app: 2051, bitrate: 48000, max_bandwidth: 1105, frame_ms: 10 };
pub const CELT_16K: Setup =
    Setup { fs: 16000, channels: 1, app: 2051, bitrate: 32000, max_bandwidth: 1105, frame_ms: 20 };

/// The largest difference between opus-rs's output at two complexities.
pub fn rs_effect(s: Setup, packets: &[Option<Vec<u8>>], c0: i32, c1: i32) -> f32 {
    let frame = s.fs as usize * s.frame_ms / 1000;
    let mut d0 = opus_rs::OpusDecoder::new(s.fs, s.channels).unwrap();
    let mut d1 = opus_rs::OpusDecoder::new(s.fs, s.channels).unwrap();
    d0.set_complexity(c0).unwrap();
    d1.set_complexity(c1).unwrap();
    let mut maxd = 0f32;
    for p in packets {
        let mut a = vec![0f32; frame * s.channels];
        let mut b = vec![0f32; frame * s.channels];
        d0.decode(p.as_deref().unwrap_or(&[]), frame, &mut a).unwrap();
        d1.decode(p.as_deref().unwrap_or(&[]), frame, &mut b).unwrap();
        maxd = a.iter().zip(&b).fold(maxd, |m, (x, y)| m.max((x - y).abs()));
    }
    maxd
}

pub const HYBRID_48K: Setup =
    Setup { fs: 48000, channels: 1, app: 2048, bitrate: 32000, max_bandwidth: 1105, frame_ms: 20 };

/// Every setup, and whether deep PLC conceals its losses (SILK at 16 kHz
/// and CELT; narrowband SILK keeps the classic PLC).
pub const SETUPS: [(&str, Setup, bool); 10] = [
    ("silk wb 16k", SILK_WB_16K, true),
    ("silk wb 48k", SILK_WB_48K, true),
    ("silk wb 16k 10ms", SILK_WB_16K_10MS, true),
    ("silk nb 8k", SILK_NB_8K, false),
    ("silk wb stereo", SILK_WB_STEREO, true),
    ("hybrid 48k", HYBRID_48K, true),
    ("celt 48k", CELT_48K, true),
    ("celt 48k stereo", CELT_48K_STEREO, true),
    ("celt 48k 10ms", CELT_48K_10MS, true),
    ("celt 16k", CELT_16K, true),
];

/// Deep PLC (complexity 5) with random loss bursts, short and long (up to
/// 400 ms, past CELT's 200 ms limit for neural concealment): bit-exact with
/// libopus in every mode, and different from complexity 4 where deep PLC
/// applies.
#[test]
fn deep_plc_matches_libopus() {
    for (name, s, neural) in SETUPS {
        for (seed, rate, burst) in [(3u64, 12u32, 6u32), (4, 4, 20)] {
            let packets = with_losses(encode(s, 6, seed), seed, rate, burst, s.frame_ms);
            assert_eq!(compare(s, &packets, 5), (0.0, None), "{name} seed {seed}");
            if neural {
                assert!(rs_effect(s, &packets, 4, 5) > 0.01, "{name} seed {seed}: deep PLC did not run");
            }
        }
    }
}

/// Prints how far opus-rs is from libopus for every setup and complexity.
#[test]
#[ignore]
fn deep_plc_survey() {
    for (name, s, _) in SETUPS {
        let packets = with_losses(encode(s, 6, 7), 3, 12, 6, s.frame_ms);
        println!("{name}: opus-rs complexity 0 vs 5 differ by {:e}", rs_effect(s, &packets, 0, 5));
        for c in [0, 4, 5, 10] {
            let (d, f) = compare(s, &packets, c);
            println!("{name} complexity {c}: max diff {d:e}, first differing frame {f:?}");
        }
    }
}

/// LACE (complexity 6) and NoLACE (7), with deep PLC, on every setup with
/// random loss bursts and with two bursts two frames apart (which exercise
/// decode_core's loss-recovery tap): bit-exact with libopus, and different
/// from complexity 5 where OSCE applies (20 ms frames of a 16 kHz SILK
/// layer).
#[test]
fn osce_matches_libopus() {
    for (name, s, _) in SETUPS {
        let base = encode(s, 6, 7);
        let mut burst = base.iter().cloned().map(Some).collect::<Vec<_>>();
        let k = 1000 / s.frame_ms;
        for i in (k..k + 6).chain(k + 8..k + 12) {
            burst[i] = None;
        }
        let random = with_losses(base, 3, 12, 6, s.frame_ms);
        let enhanced = s.app == 2048 && s.frame_ms == 20 && s.max_bandwidth != 1101;
        for packets in [&burst, &random] {
            for c in [6, 7] {
                assert_eq!(compare(s, packets, c), (0.0, None), "{name} complexity {c}");
                if enhanced {
                    assert!(rs_effect(s, packets, 5, c) > 0.01, "{name} complexity {c}: OSCE did not run");
                }
            }
        }
    }
}
