//! The decoder API against libopus's: the controls and entry points of
//! `opus_decoder_ctl` and `opus_decode*` on libopus-encoded streams with
//! losses, compared after every call.
use core::ffi::c_void;
use opus_rs::OpusDecoder;
use opusref::*;

const SILK: i32 = 1000;
const HYBRID: i32 = 1001;
const CELT: i32 = 1002;

/// A libopus encoder in `force` mode (`OPUS_SET_FORCE_MODE`) producing
/// packets of a voiced signal.
struct Source {
    enc: *mut c_void,
    fs: i32,
    ch: usize,
    frame: usize,
    n: usize,
}

impl Source {
    fn new(fs: i32, ch: usize, frame: usize, kbps: i32, force: i32) -> Self {
        let mut err = 0;
        let enc = unsafe { opus_encoder_create(fs, ch as i32, 2049, &mut err) };
        assert!(!enc.is_null() && err == 0);
        unsafe {
            opus_encoder_ctl(enc, SET_BITRATE, kbps * 1000);
            opus_encoder_ctl(enc, 11002, force);
            opus_encoder_ctl(enc, SET_INBAND_FEC, 1);
            opus_encoder_ctl(enc, SET_PACKET_LOSS_PERC, 20);
        }
        Source {
            enc,
            fs,
            ch,
            frame,
            n: 0,
        }
    }

    fn next(&mut self) -> Vec<u8> {
        let (fs, ch, frame) = (self.fs, self.ch, self.frame);
        let f = self.n;
        self.n += 1;
        let pitch = 150.0 + 40.0 * ((f as f32) * 0.3).sin();
        let pcm: Vec<f32> = (0..frame * ch)
            .map(|i| {
                let t = (f * frame + i / ch) as f32 / fs as f32;
                (1..12)
                    .map(|h| ((h as f32) * pitch * t * 6.2832).sin() / h as f32)
                    .sum::<f32>()
                    * if i % ch == 0 { 0.2 } else { 0.1 }
            })
            .collect();
        let mut pkt = [0u8; 1500];
        let len =
            unsafe { opus_encode_float(self.enc, pcm.as_ptr(), frame as i32, pkt.as_mut_ptr(), 1500) };
        assert!(len > 0);
        pkt[..len as usize].to_vec()
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        unsafe { opus_encoder_destroy(self.enc) }
    }
}

/// An opus-rs decoder and a libopus decoder side by side.
struct Pair {
    rs: OpusDecoder,
    c: *mut c_void,
    ch: usize,
}

impl Pair {
    fn new(fs: i32, ch: usize) -> Self {
        let mut err = 0;
        let c = unsafe { opus_decoder_create(fs, ch as i32, &mut err) };
        assert!(!c.is_null() && err == 0);
        Pair {
            rs: OpusDecoder::new(fs, ch).unwrap(),
            c,
            ch,
        }
    }

    fn ctl(&mut self, req: i32, value: i32) -> i32 {
        unsafe { opus_decoder_ctl(self.c, req, value) }
    }

    fn get(&mut self, req: i32) -> i32 {
        let mut v = 0i32;
        let r = unsafe { opus_decoder_ctl(self.c, req, &mut v as *mut i32) };
        assert_eq!(r, 0);
        v
    }

    /// Decodes `pkt` (`None`: a loss) on both, `fec` as `decode_fec`, and
    /// asserts the same result, final range and samples.
    fn decode(&mut self, pkt: Option<&[u8]>, frame: usize, fec: bool, what: &str) {
        let (mut a, mut b) = (vec![0f32; frame * self.ch], vec![0f32; frame * self.ch]);
        let p = pkt.unwrap_or(&[]);
        let n = if fec {
            self.rs.decode_fec(p, frame, &mut a)
        } else {
            self.rs.decode(p, frame, &mut a)
        };
        let ptr = pkt.map_or(core::ptr::null(), |p| p.as_ptr());
        let m = unsafe {
            opus_decode_float(self.c, ptr, p.len() as i32, b.as_mut_ptr(), frame as i32, fec as i32)
        };
        assert_eq!(n.map_or(-1, |n| n as i32).signum(), m.signum(), "{what}: {n:?} vs {m}");
        let Ok(n) = n else { return };
        assert_eq!(n as i32, m, "{what}");
        let mut r = 0u32;
        unsafe { opus_decoder_ctl(self.c, GET_FINAL_RANGE, &mut r as *mut u32) };
        assert_eq!(self.rs.final_range(), r, "{what}: final range");
        assert_close(&a[..n * self.ch], &b[..n * self.ch], what);
    }
}

impl Drop for Pair {
    fn drop(&mut self) {
        unsafe { opus_decoder_destroy(self.c) }
    }
}

/// Within the differential fuzzer's tolerance: 4e-6 of max(1, peak).
fn assert_close(a: &[f32], b: &[f32], what: &str) {
    let peak = b.iter().fold(1f32, |p, x| p.max(x.abs()));
    if let Some(i) = (0..a.len()).find(|&i| !((a[i] - b[i]).abs() <= 4e-6 * peak)) {
        panic!("{what}: sample {i}: opus-rs {:e}, libopus {:e}", a[i], b[i]);
    }
}

/// The decoder complexity: 10 with the neural network features, so the
/// deep PLC conceals the losses.
fn complexity(p: &mut Pair) {
    if cfg!(feature = "dnn") {
        p.rs.set_complexity(10).unwrap();
        p.ctl(SET_COMPLEXITY, 10);
    }
}

/// A reset in the middle of a stream, next to losses and FEC, decodes like
/// libopus's `OPUS_RESET_STATE`; with the deep PLC its FARGAN state
/// survives the reset in both.
#[test]
fn reset_matches_opus_reset_state() {
    for (fs, ch, frame, kbps, force) in [
        (48000, 2, 960, 64, CELT),
        (48000, 1, 960, 32, HYBRID),
        (16000, 1, 320, 16, SILK),
        (16000, 2, 640, 24, SILK),
        (24000, 2, 480, 40, HYBRID),
    ] {
        let mut src = Source::new(fs, ch, frame, kbps, force);
        let mut p = Pair::new(fs, ch);
        complexity(&mut p);
        for f in 0..40 {
            let pkt = src.next();
            let what = format!("{fs} {ch} mode {force} frame {f}");
            match f % 10 {
                3 | 7 => p.decode(None, frame, false, &what),
                4 => p.decode(Some(&pkt), frame, true, &what),
                5 => {
                    p.rs.reset();
                    p.ctl(RESET_STATE, 0);
                    p.decode(None, frame, false, &what);
                }
                _ => p.decode(Some(&pkt), frame, false, &what),
            }
            if f == 21 {
                p.rs.reset();
                p.ctl(RESET_STATE, 0);
            }
        }
    }
}

/// A reset keeps the settings: the complexity stays.
#[test]
fn reset_keeps_the_complexity() {
    let mut p = Pair::new(48000, 1);
    p.rs.set_complexity(7).unwrap();
    p.ctl(SET_COMPLEXITY, 7);
    p.rs.reset();
    p.ctl(RESET_STATE, 0);
    assert_eq!(p.rs.complexity(), p.get(GET_COMPLEXITY));
}
