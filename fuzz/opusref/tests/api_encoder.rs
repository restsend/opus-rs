//! The encoder API against libopus's: the controls of `opus_encoder_ctl`
//! and the `opus_encode*` entry points.
//!
//! Packets are compared as the differential fuzzer does: libopus decodes
//! every opus-rs packet with the encoder's final range and the ToC agrees;
//! RestrictedLowDelay (CELT-only) packets are byte-identical.
use core::ffi::c_void;
use opus_rs::{Application, OpusEncoder};
use opusref::*;

/// A voiced test signal, frame `f` of `frame` samples per channel.
fn signal(fs: i32, ch: usize, frame: usize, f: usize) -> Vec<f32> {
    let pitch = 150.0 + 40.0 * ((f as f32) * 0.3).sin();
    (0..frame * ch)
        .map(|i| {
            let t = (f * frame + i / ch) as f32 / fs as f32;
            (1..12)
                .map(|h| ((h as f32) * pitch * t * 6.2832).sin() / h as f32)
                .sum::<f32>()
                * if i % ch == 0 { 0.2 } else { 0.13 }
        })
        .collect()
}

/// An opus-rs encoder, a libopus encoder with the same settings, and a
/// libopus decoder for the opus-rs packets.
struct Pair {
    rs: OpusEncoder,
    c: *mut c_void,
    dec: *mut c_void,
    fs: i32,
    ch: usize,
    app: i32,
    /// The signal's gain.
    scale: f32,
    /// Encode through `encode_i16` / `opus_encode`.
    int16: bool,
}

impl Pair {
    fn new(fs: i32, ch: usize, app: Application) -> Self {
        let mut err = 0;
        let c = unsafe { opus_encoder_create(fs, ch as i32, app as i32, &mut err) };
        assert!(!c.is_null() && err == 0);
        let dec = unsafe { opus_decoder_create(fs, ch as i32, &mut err) };
        assert!(!dec.is_null() && err == 0);
        Pair {
            rs: OpusEncoder::new(fs, ch, app).unwrap(),
            c,
            dec,
            fs,
            ch,
            app: app as i32,
            scale: 1.0,
            int16: false,
        }
    }

    fn ctl(&mut self, req: i32, value: i32) -> i32 {
        unsafe { opus_encoder_ctl(self.c, req, value) }
    }

    fn get(&mut self, req: i32) -> i32 {
        let mut v = 0i32;
        let r = unsafe { opus_encoder_ctl(self.c, req, &mut v as *mut i32) };
        assert_eq!(r, 0, "get {req}");
        v
    }

    /// Encodes frame `f` on both and checks the packets.
    fn encode(&mut self, frame: usize, f: usize, what: &str) {
        let pcm: Vec<f32> = signal(self.fs, self.ch, frame, f).iter().map(|x| x * self.scale).collect();
        let (mut a, mut b) = ([0u8; 1500], [0u8; 1500]);
        let (n, m) = if self.int16 {
            let pcm: Vec<i16> = pcm.iter().map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16).collect();
            let n = self.rs.encode_i16(&pcm, frame, &mut a);
            let m = unsafe { opus_encode(self.c, pcm.as_ptr(), frame as i32, b.as_mut_ptr(), 1500) };
            (n, m)
        } else {
            let n = self.rs.encode(&pcm, frame, &mut a);
            let m = unsafe { opus_encode_float(self.c, pcm.as_ptr(), frame as i32, b.as_mut_ptr(), 1500) };
            (n, m)
        };
        assert_eq!(n.is_ok(), m > 0, "{what}: {n:?} vs {m}");
        let Ok(n) = n else { return };
        self.check(&a[..n], &b[..m as usize], frame, what);
    }

    /// The packet oracles: libopus decodes `pkt` with the encoder's final
    /// range, the ToC equals libopus's `reference`, and CELT-only streams
    /// are byte-identical.
    fn check(&mut self, pkt: &[u8], reference: &[u8], frame: usize, what: &str) {
        let mut out = vec![0f32; frame * self.ch];
        let d = unsafe {
            opus_decode_float(self.dec, pkt.as_ptr(), pkt.len() as i32, out.as_mut_ptr(), frame as i32, 0)
        };
        assert_eq!(d, frame as i32, "{what}: libopus decodes the opus-rs packet");
        let mut r = 0u32;
        unsafe { opus_decoder_ctl(self.dec, GET_FINAL_RANGE, &mut r as *mut u32) };
        if pkt.len() > 1 {
            assert_eq!(r, self.rs.final_range(), "{what}: final range");
        }
        assert_eq!(pkt[0] & 0xfc, reference[0] & 0xfc, "{what}: ToC");
        if self.app == Application::RestrictedLowDelay as i32 {
            assert_eq!(pkt, reference, "{what}: bytes");
        }
    }
}

impl Drop for Pair {
    fn drop(&mut self) {
        unsafe {
            opus_encoder_destroy(self.c);
            opus_decoder_destroy(self.dec);
        }
    }
}

const CONFIGS: [(i32, usize, Application, usize, i32); 6] = [
    (48000, 2, Application::RestrictedLowDelay, 960, 96000),
    (48000, 1, Application::RestrictedLowDelay, 240, 64000),
    (48000, 2, Application::Audio, 960, 64000),
    (16000, 1, Application::Voip, 320, 16000),
    (24000, 2, Application::Voip, 480, 32000),
    (8000, 1, Application::Audio, 160, 12000),
];

/// After a reset both encoders start over: the packets after it agree as
/// the first packets of a stream do, also for a second reset.
#[test]
fn reset_matches_opus_reset_state() {
    for (fs, ch, app, frame, rate) in CONFIGS {
        let mut p = Pair::new(fs, ch, app);
        p.rs.bitrate_bps = rate;
        p.ctl(SET_BITRATE, rate);
        for f in 0..30 {
            if f == 9 || f == 20 {
                p.rs.reset();
                p.ctl(RESET_STATE, 0);
            }
            p.encode(frame, f, &format!("{fs} {ch} {app:?} frame {f}"));
        }
        // A reset keeps the settings.
        assert_eq!(p.get(GET_BITRATE), rate);
    }
}

/// A reset encoder codes CELT-only streams like a new one.
#[test]
fn reset_codes_like_a_new_encoder() {
    let (fs, ch, frame) = (48000, 2, 960);
    let mut a = OpusEncoder::new(fs, ch, Application::RestrictedLowDelay).unwrap();
    let mut out = [0u8; 1500];
    for f in 0..7 {
        a.encode(&signal(fs, ch, frame, f), frame, &mut out).unwrap();
    }
    a.reset();
    let mut b = OpusEncoder::new(fs, ch, Application::RestrictedLowDelay).unwrap();
    for f in 0..10 {
        let pcm = signal(fs, ch, frame, f + 3);
        let mut out2 = [0u8; 1500];
        let n = a.encode(&pcm, frame, &mut out).unwrap();
        let m = b.encode(&pcm, frame, &mut out2).unwrap();
        assert_eq!(out[..n], out2[..m], "frame {f}");
    }
}

fn code(r: Result<(), &str>) -> i32 {
    r.map_or_else(|e| opus_rs::error::code(e).unwrap(), |()| 0)
}

/// `OPUS_SET_LSB_DEPTH`: quiet signals code alike at every depth, through
/// both entry points; out-of-range depths are rejected.
#[test]
fn lsb_depth_matches_libopus() {
    for (fs, ch, app, frame, rate) in CONFIGS {
        for depth in [8, 11, 16, 20, 24] {
            for int16 in [false, true] {
                let mut p = Pair::new(fs, ch, app);
                p.rs.bitrate_bps = rate;
                p.ctl(SET_BITRATE, rate);
                p.scale = 0.004;
                p.int16 = int16;
                for bad in [7, 25, -1] {
                    assert_eq!(code(p.rs.set_lsb_depth(bad)), p.ctl(SET_LSB_DEPTH, bad));
                }
                assert_eq!(code(p.rs.set_lsb_depth(depth)), p.ctl(SET_LSB_DEPTH, depth));
                assert_eq!(p.rs.lsb_depth(), p.get(GET_LSB_DEPTH));
                for f in 0..12 {
                    p.encode(frame, f, &format!("{fs} {ch} {app:?} depth {depth} i16 {int16} frame {f}"));
                }
            }
        }
    }
}

/// `OPUS_SET_FORCE_CHANNELS`: stereo input coded mono or stereo at any
/// rate; invalid counts are rejected.
#[test]
fn force_channels_matches_libopus() {
    for (fs, ch, app, frame, _) in CONFIGS {
        for rate in [12000, 64000] {
            for (force, value) in [(Some(1), 1), (Some(2), 2), (None, -1000)] {
                let mut p = Pair::new(fs, ch, app);
                p.rs.bitrate_bps = rate;
                p.ctl(SET_BITRATE, rate);
                for (bad, c) in [(Some(0), 0), (Some(3), 3)] {
                    assert_eq!(code(p.rs.set_force_channels(bad)), p.ctl(SET_FORCE_CHANNELS, c));
                }
                assert_eq!(code(p.rs.set_force_channels(force)), p.ctl(SET_FORCE_CHANNELS, value));
                assert_eq!(p.rs.force_channels().map_or(-1000, |v| v as i32), p.get(GET_FORCE_CHANNELS));
                for f in 0..12 {
                    p.encode(frame, f, &format!("{fs} {ch} {app:?} {rate} force {force:?} frame {f}"));
                }
            }
        }
    }
}
