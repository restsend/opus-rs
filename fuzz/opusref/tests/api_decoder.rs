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
        self.check_getters(what);
    }

    /// The decoder's queries agree with libopus's.
    fn check_getters(&mut self, what: &str) {
        let rs = &self.rs;
        let bw = rs.bandwidth().map_or(0, |b| b as i32);
        let got = (rs.sample_rate(), bw, rs.pitch(), rs.last_packet_duration() as i32);
        let want = (
            self.get(GET_SAMPLE_RATE),
            self.get(GET_BANDWIDTH),
            self.get(GET_PITCH),
            self.get(GET_LAST_PACKET_DURATION),
        );
        assert_eq!(got, want, "{what}: sample rate, bandwidth, pitch, last packet duration");
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

/// `OPUS_SET_GAIN`: the output scales alike, and out-of-range gains are
/// rejected and keep the gain.
#[test]
fn gain_matches_opus_set_gain() {
    for (fs, ch, frame, kbps, force) in [(48000, 2, 960, 64, CELT), (16000, 1, 320, 16, SILK)] {
        for gain in [-32768, -1536, -1, 1, 256, 1536, 32767] {
            let mut src = Source::new(fs, ch, frame, kbps, force);
            let mut p = Pair::new(fs, ch);
            p.rs.set_gain(gain).unwrap();
            assert_eq!(p.ctl(SET_GAIN, gain), 0);
            for bad in [-32769, 32768, i32::MIN] {
                assert_eq!(p.rs.set_gain(bad), Err(opus_rs::error::BAD_ARG));
                assert_eq!(p.ctl(SET_GAIN, bad), -1);
            }
            assert_eq!(p.rs.gain(), p.get(GET_GAIN));
            for f in 0..12 {
                let pkt = src.next();
                let what = format!("{fs} {ch} gain {gain} frame {f}");
                if f % 5 == 4 {
                    p.decode(None, frame, false, &what);
                } else {
                    p.decode(Some(&pkt), frame, false, &what);
                }
            }
        }
    }
}

/// `OPUS_SET_PHASE_INVERSION_DISABLED`: on by default in a mono decoder,
/// off in a stereo one; both settings decode intensity stereo alike.
#[test]
fn phase_inversion_matches_libopus() {
    for out in [1, 2] {
        for disabled in [false, true] {
            // Low-rate stereo CELT codes the upper bands as intensity stereo.
            let mut src = Source::new(48000, 2, 960, 24, CELT);
            let mut p = Pair::new(48000, out);
            assert_eq!(
                p.rs.phase_inversion_disabled() as i32,
                p.get(GET_PHASE_INVERSION_DISABLED)
            );
            p.rs.set_phase_inversion_disabled(disabled);
            assert_eq!(p.ctl(SET_PHASE_INVERSION_DISABLED, disabled as i32), 0);
            assert_eq!(p.ctl(SET_PHASE_INVERSION_DISABLED, 2), -1);
            assert_eq!(
                p.rs.phase_inversion_disabled() as i32,
                p.get(GET_PHASE_INVERSION_DISABLED)
            );
            p.rs.reset();
            p.ctl(RESET_STATE, 0);
            assert_eq!(p.rs.phase_inversion_disabled(), disabled);
            for f in 0..12 {
                let pkt = src.next();
                p.decode(Some(&pkt), 960, false, &format!("{out} {disabled} frame {f}"));
            }
        }
    }
}

/// `OPUS_SET_IGNORE_EXTENSIONS`, and the queries before the first packet.
#[test]
fn ignore_extensions_and_initial_queries() {
    let mut p = Pair::new(24000, 2);
    p.check_getters("new decoder");
    assert_eq!(p.rs.ignore_extensions() as i32, p.get(GET_IGNORE_EXTENSIONS));
    p.rs.set_ignore_extensions(true);
    assert_eq!(p.ctl(SET_IGNORE_EXTENSIONS, 1), 0);
    assert_eq!(p.ctl(SET_IGNORE_EXTENSIONS, 2), -1);
    assert_eq!(p.rs.ignore_extensions() as i32, p.get(GET_IGNORE_EXTENSIONS));
    p.decode(None, 480, false, "loss before any packet");
}

/// The output format of a call.
#[derive(Clone, Copy, Debug)]
enum Format {
    F32,
    I16,
    I24,
}

impl Pair {
    /// Decodes `pkt` (`None`: a loss) on both in `format` and asserts the
    /// same result and samples within the float tolerance plus one step.
    fn decode_as(&mut self, format: Format, pkt: Option<&[u8]>, frame: usize, fec: bool, what: &str) {
        let what = &format!("{what} {format:?}");
        let n = frame * self.ch;
        let p = pkt.unwrap_or(&[]);
        let ptr = pkt.map_or(core::ptr::null(), |p| p.as_ptr());
        let (len, f) = (p.len() as i32, frame as i32);
        let (r, m, a, b, scale): (Result<usize, &str>, i32, Vec<f32>, Vec<f32>, f32) = match format {
            Format::F32 => return self.decode(pkt, frame, fec, what),
            Format::I16 => {
                let (mut a, mut b) = (vec![0i16; n], vec![0i16; n]);
                let r = if fec {
                    self.rs.decode_fec_i16(p, frame, &mut a)
                } else {
                    self.rs.decode_i16(p, frame, &mut a)
                };
                let m = unsafe { opus_decode(self.c, ptr, len, b.as_mut_ptr(), f, fec as i32) };
                let g = |v: Vec<i16>| v.iter().map(|&x| x as f32).collect();
                (r, m, g(a), g(b), 32768.0)
            }
            Format::I24 => {
                let (mut a, mut b) = (vec![0i32; n], vec![0i32; n]);
                let r = if fec {
                    self.rs.decode_fec24(p, frame, &mut a)
                } else {
                    self.rs.decode24(p, frame, &mut a)
                };
                let m = unsafe { opus_decode24(self.c, ptr, len, b.as_mut_ptr(), f, fec as i32) };
                let g = |v: Vec<i32>| v.iter().map(|&x| x as f32).collect();
                (r, m, g(a), g(b), 8_388_608.0)
            }
        };
        match r {
            Ok(k) => assert_eq!(k as i32, m, "{what}"),
            Err(e) => {
                assert!(m < 0, "{what}: opus-rs {e}, libopus {m}");
                if let Some(c) = opus_rs::error::code(e) {
                    assert_eq!(c, m, "{what}");
                }
                return;
            }
        }
        let k = m as usize * self.ch;
        let peak = b[..k].iter().fold(scale, |p, x| p.max(x.abs()));
        if let Some(i) = (0..k).find(|&i| !((a[i] - b[i]).abs() <= 1.0 + 4e-6 * peak)) {
            panic!("{what}: sample {i}: opus-rs {}, libopus {}", a[i], b[i]);
        }
        let mut rng = 0u32;
        unsafe { opus_decoder_ctl(self.c, GET_FINAL_RANGE, &mut rng as *mut u32) };
        assert_eq!(self.rs.final_range(), rng, "{what}: final range");
        self.check_getters(what);
    }
}

/// `opus_decode` and `opus_decode24`: loud streams that soft clipping
/// shapes, mixed with float calls (which clear the soft clipping state),
/// losses, FEC and concealments longer than 120 ms.
#[test]
fn integer_decoding_matches_libopus() {
    use Format::*;
    for (fs, ch, frame, kbps, force) in [
        (48000, 2, 960, 128, CELT),
        (48000, 1, 480, 64, CELT),
        (16000, 1, 320, 24, SILK),
        (24000, 2, 480, 40, HYBRID),
    ] {
        let mut src = Source::new(fs, ch, frame, kbps, force);
        let mut p = Pair::new(fs, ch);
        p.rs.set_gain(2048).unwrap();
        p.ctl(SET_GAIN, 2048);
        for f in 0..40 {
            let pkt = src.next();
            let format = [I16, I16, I24, F32, I16][f % 5];
            let what = format!("{fs} {ch} mode {force} frame {f}");
            match f % 9 {
                3 => p.decode_as(format, None, frame, false, &what),
                4 => p.decode_as(format, Some(&pkt), 3 * frame, true, &what),
                6 => p.decode_as(format, None, 7 * fs as usize / 50, false, &what),
                7 => p.decode_as(format, Some(&pkt), 8 * fs as usize / 50, true, &what),
                _ => p.decode_as(format, Some(&pkt), frame * 6, false, &what),
            }
        }
        p.decode_as(I16, None, 0, false, "frame size 0");
        p.decode_as(I24, Some(&[0xfb]), 960, false, "code 3 without count");
    }
}
