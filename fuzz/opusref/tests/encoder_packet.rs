//! Packet-level decisions of the Opus encoder against libopus 1.6.1
//! (opus_encode_native): the packet's bitrate and budget, the ToC-only
//! packets of starved budgets, the mono/stereo decision and multiframe
//! packets. Each test codes the same audio with the same settings in both
//! encoders and compares the packets.
use opus_rs::{Application, OpusEncoder};
use opusref as c;

struct Pair {
    rs: OpusEncoder,
    re: *mut c::OpusEncoder,
    fs: i32,
    ch: usize,
    t: usize,
}

impl Pair {
    fn new(fs: i32, ch: usize, app: Application) -> Self {
        let mut err = 0;
        let re = unsafe { c::opus_encoder_create(fs, ch as i32, app as i32, &mut err) };
        assert!(!re.is_null() && err == 0);
        Pair { rs: OpusEncoder::new(fs, ch, app).unwrap(), re, fs, ch, t: 0 }
    }

    fn ctl(&mut self, req: i32, val: i32) {
        assert_eq!(unsafe { c::opus_encoder_ctl(self.re, req, val) }, 0);
    }

    fn bitrate(&mut self, bps: i32) {
        self.rs.bitrate_bps = bps;
        self.ctl(c::SET_BITRATE, bps);
    }

    fn cbr(&mut self) {
        self.rs.use_cbr = true;
        self.ctl(c::SET_VBR, 0);
    }

    /// The next `frame` samples per channel of a voiced signal with some
    /// noise, different in each channel.
    fn pcm(&mut self, frame: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(frame * self.ch);
        for i in 0..frame {
            let t = (self.t + i) as f32 / self.fs as f32;
            let env = 0.5 + 0.5 * (6.0 * t).sin();
            for c in 0..self.ch {
                let mut x = 0.0;
                for h in 1..12 {
                    let f = 140.0 * h as f32 * (1.0 + 0.05 * c as f32);
                    x += (2.0 * std::f32::consts::PI * f * t + h as f32).sin() / h as f32;
                }
                let n = ((self.t + i) as u32).wrapping_mul(2654435761).wrapping_add(c as u32 * 7919);
                x += ((n >> 8) as f32 / (1 << 24) as f32 - 0.5) * 0.05;
                out.push(0.25 * env * x);
            }
        }
        self.t += frame;
        out
    }

    /// One frame into a `budget`-byte buffer: (opus-rs, libopus) packets,
    /// or the errors.
    fn encode(&mut self, frame: usize, budget: usize) -> (Result<Vec<u8>, String>, Result<Vec<u8>, i32>) {
        let pcm = self.pcm(frame);
        let mut a = vec![0u8; budget];
        let mut b = vec![0u8; budget];
        let r = self.rs.encode(&pcm, frame, &mut a).map(|n| a[..n].to_vec()).map_err(String::from);
        let n = unsafe {
            c::opus_encode_float(self.re, pcm.as_ptr(), frame as i32, b.as_mut_ptr(), budget as i32)
        };
        let l = if n < 0 { Err(n) } else { Ok(b[..n as usize].to_vec()) };
        (r, l)
    }

    /// Codes `frames` frames and asserts that both encoders accept or both
    /// reject each call, and that `same` holds for the packets.
    fn check(&mut self, frames: usize, frame: usize, budget: usize, same: fn(&[u8], &[u8]) -> bool) {
        for f in 0..frames {
            let (r, l) = self.encode(frame, budget);
            match (&r, &l) {
                (Ok(a), Ok(b)) => assert!(same(a, b), "frame {f}: opus-rs {a:02x?}, libopus {b:02x?}"),
                (Err(_), Err(_)) => {}
                _ => panic!("frame {f}: opus-rs {r:?}, libopus {l:?}"),
            }
        }
    }
}

/// The same ToC configuration and stereo flag.
fn toc(a: &[u8], b: &[u8]) -> bool {
    a[0] & 0xfc == b[0] & 0xfc
}

/// The same ToC and length.
fn toc_len(a: &[u8], b: &[u8]) -> bool {
    toc(a, b) && a.len() == b.len()
}

/// The same bytes.
fn bytes(a: &[u8], b: &[u8]) -> bool {
    a == b
}

impl Drop for Pair {
    fn drop(&mut self) {
        unsafe { c::opus_encoder_destroy(self.re) }
    }
}

/// The packet's bitrate is at most what the output buffer holds
/// (user_bitrate_to_bitrate), so a 60-byte buffer at 64 kb/s codes SILK
/// at 24 kb/s.
#[test]
fn vbr_bitrate_is_capped_by_the_buffer() {
    let mut p = Pair::new(16000, 1, Application::Voip);
    p.bitrate(64000);
    p.check(8, 320, 60, toc);
}

/// Under CBR the bitrate is the rate of the whole bytes it codes: 1190 b/s
/// at 20 ms is 3 bytes, 1200 b/s, enough for a coded frame.
#[test]
fn cbr_bitrate_is_whole_bytes() {
    let mut p = Pair::new(48000, 1, Application::Audio);
    p.bitrate(1190);
    p.cbr();
    p.check(8, 960, 1500, bytes);
}

/// The default bitrate is OPUS_AUTO: 60*Fs/frame_size + Fs*channels, here
/// 99000 b/s for 48 kHz stereo 20 ms frames.
#[test]
fn default_bitrate_is_auto() {
    let mut p = Pair::new(48000, 2, Application::Audio);
    p.check(8, 960, 1500, bytes);
}

/// OPUS_BITRATE_MAX codes as many bits as the buffer holds.
#[test]
fn bitrate_max_fills_the_buffer() {
    let mut p = Pair::new(48000, 1, Application::Audio);
    p.bitrate(OpusEncoder::BITRATE_MAX);
    p.check(8, 960, 200, bytes);
}

/// Below 9 kb/s of buffer at 10 ms libopus codes CELT, comparing the
/// buffer with the bit budget floored to bytes: 11 bytes against 90 bits
/// (11 bytes) keeps SILK.
#[test]
fn small_buffer_threshold_floors_to_bytes() {
    let mut p = Pair::new(16000, 1, Application::Voip);
    p.bitrate(12000);
    p.check(8, 160, 11, toc);
}

/// A new encoder is at fullband (and in Hybrid mode) at every rate and
/// application, like libopus (opus_encoder.c:319-320).
#[test]
fn initial_bandwidth_is_fullband() {
    for fs in [8000, 12000, 16000, 24000, 48000] {
        for app in [Application::Voip, Application::Audio, Application::RestrictedLowDelay] {
            let p = Pair::new(fs, 1, app);
            let mut bw = 0i32;
            unsafe { c::opus_encoder_ctl(p.re, 4009, &mut bw as *mut i32) }; // OPUS_GET_BANDWIDTH
            assert_eq!(p.rs.bandwidth() as i32, bw, "{fs} Hz {app:?}");
        }
    }
}
