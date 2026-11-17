//! Shared pieces of the differential harnesses: safe wrappers around the
//! reference libopus build (`opusref`), failure reporting and the
//! signal generators for encoder inputs.
#![allow(dead_code)]

use arbitrary::Arbitrary;
use core::ffi::c_int;
use core::sync::atomic::Ordering;
use opusref as c;

pub const RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

/// A differential finding. `sig` groups findings that likely share a cause.
pub struct Finding {
    pub oracle: &'static str,
    pub sig: String,
    pub msg: String,
}

/// `fail!(oracle, sig, fmt...)` returns a [`Finding`] from the current function.
#[macro_export]
macro_rules! fail {
    ($oracle:expr, $sig:expr, $($arg:tt)*) => {
        return Err($crate::common::Finding { oracle: $oracle, sig: $sig.to_string(), msg: format!($($arg)*) })
    };
}

/// Without `OFZ_SURVEY`, a finding panics with `DIFF[<oracle>]` (a libFuzzer
/// crash). With `OFZ_SURVEY=<dir>`, the first input of each new signature is
/// saved to `<dir>/<target>-<n>.bin` with a line in `<dir>/findings.txt`, and
/// fuzzing continues, so one run surveys many distinct findings.
pub fn report(target: &str, data: &[u8], r: Result<(), Finding>) {
    use std::io::Write;
    use std::sync::Mutex;
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let Err(f) = r else { return };
    if f.oracle == "panic" && std::env::var("OFZ_SURVEY").is_err() {
        panic!("{}", f.msg);
    }
    let Ok(dir) = std::env::var("OFZ_SURVEY") else {
        panic!("DIFF[{}] {} :: {}", f.oracle, f.sig, f.msg);
    };
    let key = format!("{}|{}", f.oracle, f.sig);
    let mut seen = SEEN.lock().unwrap();
    if seen.contains(&key) {
        return;
    }
    seen.push(key.clone());
    let n = seen.len();
    let path = format!("{dir}/{target}-{n}.bin");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(&path, data).unwrap();
    let mut log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(format!("{dir}/findings.txt"))
        .unwrap();
    writeln!(log, "{target}\t{key}\t{path}\t{}", f.msg).unwrap();
    eprintln!("NEW FINDING {key}: {}", f.msg);
}

pub fn set_simd(on: bool) {
    opus_rs::fuzzing::SIMD_ENABLED.store(on, Ordering::Relaxed);
}

/// Reads a tolerance or switch from the environment once per process.
pub fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Runs `f`; in survey mode a panic becomes a `panic` finding keyed by its
/// source location, so the survey continues past it.
pub fn guarded(f: impl FnOnce() -> Result<(), Finding>) -> Result<(), Finding> {
    use std::cell::RefCell;
    thread_local!(static LOC: RefCell<String> = const { RefCell::new(String::new()) });
    if std::env::var("OFZ_SURVEY").is_err() {
        return f();
    }
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            let loc = info
                .location()
                .map(|l| format!("{}:{}", l.file(), l.line()))
                .unwrap_or_default();
            let msg = info
                .payload()
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| info.payload().downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            LOC.with(|l| *l.borrow_mut() = format!("{loc}\t{msg}"));
        }))
    });
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(_) => {
            let s = LOC.with(|l| l.borrow().clone());
            let (loc, msg) = s.split_once('\t').unwrap_or((&s, ""));
            Err(Finding {
                oracle: "panic",
                sig: loc.to_string(),
                msg: msg.to_string(),
            })
        }
    }
}

/// Frame sizes of a packet as libopus parses it, or `None` if it is invalid.
pub fn parse_packet(pkt: &[u8]) -> Option<Vec<i16>> {
    let mut toc = 0u8;
    let mut frames = [core::ptr::null(); 48];
    let mut size = [0i16; 48];
    let n = unsafe {
        c::opus_packet_parse(
            pkt.as_ptr(),
            pkt.len() as i32,
            &mut toc,
            frames.as_mut_ptr(),
            size.as_mut_ptr(),
            core::ptr::null_mut(),
        )
    };
    (n > 0).then(|| size[..n as usize].to_vec())
}

/// Known divergences to skip, from `OFZ_SUPPRESS` (comma-separated), so a
/// survey can look past findings that are already recorded.
pub fn suppressed(name: &str) -> bool {
    use std::sync::OnceLock;
    static S: OnceLock<Vec<String>> = OnceLock::new();
    S.get_or_init(|| {
        std::env::var("OFZ_SUPPRESS")
            .unwrap_or_default()
            .split(',')
            .map(str::to_owned)
            .collect()
    })
    .iter()
    .any(|s| s == name)
}

/// IJON-style state feedback: libFuzzer treats each byte of this section as
/// extra coverage, so inputs that reach a new (state, transition) pair, or a
/// new magnitude bucket of an interesting value, are kept in the corpus.
#[used]
#[unsafe(link_section = "__libfuzzer_extra_counters")]
static mut IJON: [u8; 1 << 14] = [0; 1 << 14];

/// Marks the abstract state `key` as reached (IJON_SET).
pub fn ijon_set(key: u64) {
    let h = key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 50;
    unsafe {
        let p = &raw mut IJON[h as usize];
        *p = (*p).wrapping_add(1);
    }
}

/// Rewards larger `value` for `slot` by its log2 bucket (an IJON_MAX approximation).
pub fn ijon_max(slot: u64, value: u64) {
    ijon_set(slot << 8 | (64 - value.leading_zeros()) as u64);
}

pub struct RefDec(*mut c::OpusDecoder);
impl RefDec {
    pub fn new(fs: i32, ch: usize) -> Self {
        let mut err = 0;
        let p = unsafe { c::opus_decoder_create(fs, ch as c_int, &mut err) };
        assert!(!p.is_null() && err == 0, "reference decoder init");
        RefDec(p)
    }
    /// `data == None` is packet loss. Returns samples per channel or an error code.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        out: &mut [f32],
        frame_size: usize,
        fec: bool,
    ) -> i32 {
        let ch = out.len() / frame_size.max(1);
        assert!(out.len() >= frame_size * ch.max(1));
        let (ptr, len) = data.map_or((core::ptr::null(), 0), |d| (d.as_ptr(), d.len() as i32));
        unsafe {
            c::opus_decode_float(
                self.0,
                ptr,
                len,
                out.as_mut_ptr(),
                frame_size as c_int,
                fec as c_int,
            )
        }
    }
    /// [`Self::decode`] through `opus_decode` or `opus_decode24` for an
    /// integer `fmt`, the samples scaled back to float.
    pub fn decode_as(
        &mut self,
        fmt: crate::diff_decoder::Format,
        data: Option<&[u8]>,
        out: &mut [f32],
        frame_size: usize,
        fec: bool,
    ) -> i32 {
        use crate::diff_decoder::Format;
        let (ptr, len) = data.map_or((core::ptr::null(), 0), |d| (d.as_ptr(), d.len() as i32));
        let (f, fec) = (frame_size as c_int, fec as c_int);
        match fmt {
            Format::F32 => self.decode(data, out, frame_size, fec != 0),
            Format::I16 => {
                let mut o = vec![0i16; out.len()];
                let r = unsafe { c::opus_decode(self.0, ptr, len, o.as_mut_ptr(), f, fec) };
                out.iter_mut().zip(&o).for_each(|(v, &x)| *v = x as f32 / 32768.0);
                r
            }
            Format::I24 => {
                let mut o = vec![0i32; out.len()];
                let r = unsafe { c::opus_decode24(self.0, ptr, len, o.as_mut_ptr(), f, fec) };
                out.iter_mut().zip(&o).for_each(|(v, &x)| *v = x as f32 / 8_388_608.0);
                r
            }
        }
    }
    pub fn final_range(&mut self) -> u32 {
        let mut r = 0u32;
        unsafe { c::opus_decoder_ctl(self.0, c::GET_FINAL_RANGE, &mut r as *mut u32) };
        r
    }
    /// `OPUS_RESET_STATE`.
    pub fn reset(&mut self) {
        unsafe { c::opus_decoder_ctl(self.0, c::RESET_STATE) };
    }
    /// An integer decoder control (`opus_decoder_ctl(st, req, value)`),
    /// returning libopus's result.
    pub fn ctl(&mut self, req: c_int, value: c_int) -> c_int {
        unsafe { c::opus_decoder_ctl(self.0, req, value) }
    }
    /// An integer decoder query (`opus_decoder_ctl(st, req, &value)`).
    pub fn get(&mut self, req: c_int) -> i32 {
        let mut v = 0i32;
        unsafe { c::opus_decoder_ctl(self.0, req, &mut v as *mut i32) };
        v
    }
}
impl Drop for RefDec {
    fn drop(&mut self) {
        unsafe { c::opus_decoder_destroy(self.0) }
    }
}

pub struct RefEnc(*mut c::OpusEncoder);
impl RefEnc {
    pub fn new(fs: i32, ch: usize, app: c_int) -> Self {
        let mut err = 0;
        let p = unsafe { c::opus_encoder_create(fs, ch as c_int, app, &mut err) };
        assert!(!p.is_null() && err == 0, "reference encoder init");
        RefEnc(p)
    }
    pub fn ctl(&mut self, req: c_int, val: i32) {
        let r = unsafe { c::opus_encoder_ctl(self.0, req, val as c_int) };
        assert_eq!(r, 0, "reference ctl {req}={val}");
    }
    pub fn get(&mut self, req: c_int) -> i32 {
        let mut v = 0i32;
        unsafe { c::opus_encoder_ctl(self.0, req, &mut v as *mut i32) };
        v
    }
    /// An integer encoder control, returning libopus's result.
    pub fn try_ctl(&mut self, req: c_int, val: i32) -> c_int {
        unsafe { c::opus_encoder_ctl(self.0, req, val as c_int) }
    }
    /// `OPUS_RESET_STATE`.
    pub fn reset(&mut self) {
        unsafe { c::opus_encoder_ctl(self.0, c::RESET_STATE) };
    }
    pub fn encode_i16(&mut self, pcm: &[i16], frame_size: usize, out: &mut [u8]) -> i32 {
        unsafe {
            c::opus_encode(
                self.0,
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        }
    }
    pub fn encode(&mut self, pcm: &[f32], frame_size: usize, out: &mut [u8]) -> i32 {
        unsafe {
            c::opus_encode_float(
                self.0,
                pcm.as_ptr(),
                frame_size as c_int,
                out.as_mut_ptr(),
                out.len() as i32,
            )
        }
    }
    pub fn final_range(&mut self) -> u32 {
        self.get(c::GET_FINAL_RANGE) as u32
    }
}
impl Drop for RefEnc {
    fn drop(&mut self) {
        unsafe { c::opus_encoder_destroy(self.0) }
    }
}

/// Largest absolute sample difference, and the reference's peak (at least 1).
pub fn max_diff(a: &[f32], b: &[f32]) -> (f32, f32, usize) {
    let mut worst = (0.0f32, 0usize);
    let mut peak = 1.0f32;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        if d > worst.0 || (d.is_nan() && !worst.0.is_nan()) {
            worst = (d, i);
        }
        peak = peak.max(y.abs());
    }
    (worst.0, peak, worst.1)
}

/// One segment of encoder input. Generators cover the triggers that random
/// audio rarely produces: digital silence, DC offset, clipping, very low
/// pitch, impulses and abrupt level changes.
#[derive(Arbitrary, Debug, Clone)]
pub enum Gen {
    Silence,
    Dc {
        level: i8,
    },
    Sine {
        freq: u16,
        amp: u8,
    },
    /// Pitch glide; `lo` maps to 30..=285 Hz, the range of bass and low voices.
    Glide {
        lo: u8,
        hi: u8,
        amp: u8,
    },
    /// Square wave, optionally above full scale.
    Square {
        freq: u16,
        amp: u8,
    },
    Noise {
        amp: u8,
    },
    /// Harmonic source with formant-like weighting, like voiced speech.
    Voice {
        f0: u8,
        amp: u8,
        vowel: u8,
    },
    Impulses {
        period: u16,
        amp: u8,
    },
    /// Raw bytes as samples in [-2, 2).
    Raw {
        bytes: Vec<u8>,
    },
}

pub struct Synth {
    pub fs: f64,
    pub phase: [f64; 2],
    pub t: u64,
    pub seed: u32,
}

impl Synth {
    pub fn new(fs: i32) -> Self {
        Synth {
            fs: fs as f64,
            phase: [0.0; 2],
            t: 0,
            seed: 0x1234_5678,
        }
    }
    fn rand(&mut self) -> f32 {
        self.seed = self
            .seed
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        (self.seed >> 8) as f32 / (1u32 << 23) as f32 * 2.0 - 1.0
    }
    /// Fills `out` (interleaved, `ch` channels). The second channel is a
    /// delayed, scaled copy so stereo decisions vary.
    pub fn fill(&mut self, g: &Gen, out: &mut [f32], ch: usize, seg_pos: usize, seg_len: usize) {
        let n = out.len() / ch;
        for i in 0..n {
            let s = self.sample(g, (seg_pos * n + i) as f64 / (seg_len * n).max(1) as f64, i);
            out[i * ch] = s;
            if ch == 2 {
                out[i * ch + 1] = 0.7 * s + 0.1 * self.rand() * (s != 0.0) as u8 as f32;
            }
            self.t += 1;
        }
    }
    fn sample(&mut self, g: &Gen, frac: f64, i: usize) -> f32 {
        use core::f64::consts::TAU;
        let amp = |a: u8| a as f64 / 255.0;
        match g {
            Gen::Silence => 0.0,
            Gen::Dc { level } => *level as f32 / 128.0,
            Gen::Sine { freq, amp: a } => {
                self.phase[0] += TAU * (*freq as f64 % 24000.0) / self.fs;
                (amp(*a) * self.phase[0].sin()) as f32
            }
            Gen::Glide { lo, hi, amp: a } => {
                let f0 = 30.0 + *lo as f64;
                let f1 = 30.0 + *hi as f64;
                self.phase[0] += TAU * (f0 + (f1 - f0) * frac) / self.fs;
                let p = self.phase[0];
                (amp(*a) * (p.sin() + 0.5 * (2.0 * p).sin() + 0.25 * (3.0 * p).sin()) / 1.75) as f32
            }
            Gen::Square { freq, amp: a } => {
                self.phase[0] += TAU * (1.0 + *freq as f64 % 4000.0) / self.fs;
                (amp(*a) * 1.5 * self.phase[0].sin().signum()) as f32
            }
            Gen::Noise { amp: a } => amp(*a) as f32 * self.rand(),
            Gen::Voice { f0, amp: a, vowel } => {
                let f0 = 80.0 + *f0 as f64;
                self.phase[0] += TAU * f0 / self.fs;
                let formants = [
                    [700.0, 1200.0],
                    [300.0, 2300.0],
                    [500.0, 900.0],
                    [400.0, 1900.0],
                ];
                let fm = formants[*vowel as usize % 4];
                let mut s = 0.0;
                let mut h = 1.0;
                while h * f0 < (self.fs / 2.0).min(8000.0) {
                    let f = h * f0;
                    let w = 1.0 / (1.0 + ((f - fm[0]) / 150.0).powi(2))
                        + 0.5 / (1.0 + ((f - fm[1]) / 200.0).powi(2));
                    s += w * (h * self.phase[0]).sin() / h.sqrt();
                    h += 1.0;
                }
                (amp(*a) * 0.5 * s) as f32
            }
            Gen::Impulses { period, amp: a } => {
                let p = (*period as u64).max(16);
                if self.t % p == 0 { amp(*a) as f32 } else { 0.0 }
            }
            Gen::Raw { bytes } => {
                if bytes.is_empty() {
                    0.0
                } else {
                    let b = bytes[i % bytes.len()] as i8;
                    b as f32 / 64.0
                }
            }
        }
    }
}
