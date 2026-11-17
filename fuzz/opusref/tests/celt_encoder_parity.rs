//! opus-rs's CeltEncoder against a bare libopus CELT encoder
//! (celt_encode_with_ec without a range coder), byte for byte, for the
//! settings the Opus layer cannot reach with identical packets: Hybrid
//! start bands, other sampling rates, direct CELT use.
use opus_rs::CeltEncoder;
use opus_rs::range_coder::RangeCoder;
use opusref::*;

const MAX: i32 = -1;

#[derive(Clone, Copy)]
struct Cfg {
    fs: i32,
    ch: usize,
    /// Samples per channel at `fs`.
    n: usize,
    bytes: usize,
    bitrate: i32,
    vbr: bool,
    cvbr: bool,
    start: usize,
    end: usize,
    /// None keeps both encoders' default.
    complexity: Option<i32>,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            fs: 48000,
            ch: 1,
            n: 960,
            bytes: 160,
            bitrate: MAX,
            vbr: false,
            cvbr: false,
            start: 0,
            end: 21,
            complexity: Some(10),
        }
    }
}

/// Encodes `frames` frames of `signal(frame, sample, channel)` with both
/// encoders; returns per frame whether the packets are identical.
fn identical(cfg: Cfg, frames: usize, signal: impl Fn(usize, usize, usize) -> f32) -> Vec<bool> {
    let Cfg { fs, ch, n, bytes, .. } = cfg;
    let mut e = CeltEncoder::with_sampling_rate(opus_rs::modes::default_mode(), ch, fs);
    e.set_end_band(cfg.end);
    e.set_bitrate(cfg.bitrate);
    e.set_vbr(cfg.vbr);
    e.set_constrained_vbr(cfg.cvbr);
    let r = unsafe { ref_celt_encoder_create(fs, ch as i32) };
    assert!(!r.is_null());
    unsafe {
        ref_celt_encoder_ctl(r, SET_BITRATE, cfg.bitrate);
        ref_celt_encoder_ctl(r, SET_VBR, cfg.vbr as i32);
        ref_celt_encoder_ctl(r, SET_VBR_CONSTRAINT, cfg.cvbr as i32);
        ref_celt_encoder_ctl(r, 10010, cfg.start as i32); // CELT_SET_START_BAND
        ref_celt_encoder_ctl(r, 10012, cfg.end as i32); // CELT_SET_END_BAND
    }
    if let Some(c) = cfg.complexity {
        e.complexity = c;
        unsafe { ref_celt_encoder_ctl(r, SET_COMPLEXITY, c) };
    }
    let out = (0..frames)
        .map(|f| {
            let pcm: Vec<f32> = (0..n * ch).map(|i| signal(f, i / ch, i % ch)).collect();
            let planar: Vec<f32> = (0..n * ch).map(|i| pcm[(i % n) * ch + i / n]).collect();
            let mut rc = RangeCoder::new_encoder(bytes as u32);
            let vbr = cfg.vbr && cfg.bitrate != MAX;
            e.encode_with_budget_vbr(&planar, n, &mut rc, cfg.start, (bytes * 8) as i32, vbr);
            rc.done();
            let a = rc.buf[..rc.storage as usize].to_vec();
            let mut b = vec![0u8; bytes];
            let m = unsafe { ref_celt_encode(r, pcm.as_ptr(), n as i32, b.as_mut_ptr(), bytes as i32) };
            assert!(m > 0, "libopus returned {m}");
            a == b[..m as usize]
        })
        .collect();
    unsafe { ref_celt_encoder_destroy(r) };
    out
}

/// Noise from a linear congruential generator, `amp` peak.
fn noise(seed: u32, amp: f32) -> impl Fn(usize, usize, usize) -> f32 {
    move |f, i, c| {
        let mut s = seed ^ (f as u32).wrapping_mul(0x9e37_79b9) ^ (i as u32).wrapping_mul(0x85eb_ca6b) ^ c as u32;
        for _ in 0..3 {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        }
        ((s >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * 2.0 * amp
    }
}

/// The CELT layer of a Hybrid frame (start band 17) runs the dynalloc
/// over the log energies of every band, also those below the start band
/// (celt_encoder.c:2106, quant_bands.c:553-570).
#[test]
fn hybrid_dynalloc_sees_the_low_band_energies() {
    let cfg = Cfg { bitrate: 48000, vbr: true, start: 17, end: 21, bytes: 400, ..Cfg::default() };
    assert_eq!(identical(cfg, 4, noise(1, 0.3)), [true; 4]);
}

/// Under CBR with a bitrate, the frame takes the rate's bytes when the
/// buffer holds more (celt_encoder.c:1876-1890).
#[test]
fn cbr_frames_take_the_rate_bytes() {
    let cfg = Cfg { bitrate: 24000, bytes: 400, ..Cfg::default() };
    assert_eq!(identical(cfg, 3, noise(1, 0.3)), [true; 3]);
}

/// A frame codes at most 1275 bytes (celt_encoder.c:1859).
#[test]
fn frames_code_at_most_1275_bytes() {
    let cfg = Cfg { bytes: 1500, ch: 2, ..Cfg::default() };
    assert_eq!(identical(cfg, 2, noise(1, 0.3)), [true; 2]);
}

/// transient_analysis sums plain squares (celt_encoder.c:371, PSHR32 is
/// an identity in the float build), so its 1e-15 epsilons weigh the same
/// on a very quiet onset.
#[test]
fn quiet_onsets_weigh_the_epsilon_like_libopus() {
    let cfg = Cfg { n: 240, bytes: 80, ..Cfg::default() };
    let chirp = |f: usize, i: usize, _| {
        let t = (f * 240 + i) as f32 / 48000.0;
        (t * t * 20000.0).sin() * 1e-6
    };
    assert_eq!(identical(cfg, 2, chirp), [true; 2]);
}

/// Intensity stereo weighs the channels by bandE/(EPSILON +
/// sqrt(EPSILON + left^2 + right^2)) (bands.c:391-395); a quiet DC
/// leaves bands at the energy floor.
#[test]
fn intensity_stereo_weights_follow_libopus() {
    let cfg = Cfg { fs: 16000, ch: 2, n: 320, bytes: 300, end: 17, ..Cfg::default() };
    assert_eq!(identical(cfg, 20, |_, _, _| 3e-7), [true; 20]);
}

/// A frame with a sample beyond +-65536 clips its input to +-2.0 before
/// the pre-emphasis (celt_encoder.c:2010-2016, 592-597).
#[test]
fn input_beyond_65536_is_clipped() {
    let cfg = Cfg { ..Cfg::default() };
    let loud = |f: usize, i: usize, _| ((f * 960 + i) as f32 * 0.05).sin() * if f == 1 { 1e5 } else { 1.5 };
    assert_eq!(identical(cfg, 3, loud), [true; 3]);
}

/// A new CELT encoder runs at complexity 5 (celt_encoder.c:220).
#[test]
fn default_complexity_is_5() {
    let cfg = Cfg { ch: 2, complexity: None, ..Cfg::default() };
    let clicks = |f: usize, i: usize, c: usize| if i % 300 == 7 * c { 0.9 } else { noise(3, 0.01)(f, i, c) };
    assert_eq!(identical(cfg, 4, clicks), [true; 4]);
}
