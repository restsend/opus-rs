//! Mono concealment must search the pitch of the history libopus searches.
//!
//! The first concealed CELT frame repeats the last pitch period, found on the
//! decoded history decimated by two (`pitch_downsample`, libopus pitch.c):
//! `x_lp[i]` is `.25 x[2i - 1] + .5 x[2i] + .25 x[2i + 1]`. On aarch64 the mono
//! path computed four outputs at a time from four consecutive samples instead
//! of samples two apart, so only one output in four was right and the pitch
//! search ran on a warped history. A stereo decoder took the scalar path and
//! matched libopus. Mono concealment then differed from libopus's wherever it
//! was periodic: at a loss, at the 5 ms bridge of a switch out of CELT, and
//! after a short history.
//!
//! The oracle is libopus 1.6.1 (`opusic-sys`), whose concealment this crate
//! ports.

use opus_rs::OpusDecoder;
use opus_rs::pitch::pitch_downsample;
use opusic_sys as ffi;
use std::f32::consts::PI;
use std::ffi::c_int;

/// `OPUS_SET_FORCE_MODE` and its modes (libopus `opus_private.h`).
const OPUS_SET_FORCE_MODE_REQUEST: c_int = 11002;
const MODE_SILK_ONLY: c_int = 1000;
const MODE_HYBRID: c_int = 1001;
const MODE_CELT_ONLY: c_int = 1002;

const RATE: usize = 48000;

struct Encoder(*mut ffi::OpusEncoder);

impl Encoder {
    /// A 48 kHz mono encoder held in `mode` at `bitrate`.
    fn new(mode: c_int, bitrate: i32) -> Self {
        let mut err: c_int = 0;
        let ptr = unsafe {
            ffi::opus_encoder_create(RATE as i32, 1, ffi::OPUS_APPLICATION_AUDIO, &mut err)
        };
        assert!(!ptr.is_null() && err == 0, "opus_encoder_create: {err}");
        let bandwidth = if mode == MODE_SILK_ONLY {
            ffi::OPUS_BANDWIDTH_WIDEBAND
        } else {
            ffi::OPUS_BANDWIDTH_FULLBAND
        };
        for (request, value) in [
            (OPUS_SET_FORCE_MODE_REQUEST, mode),
            (ffi::OPUS_SET_BITRATE_REQUEST, bitrate),
            (ffi::OPUS_SET_BANDWIDTH_REQUEST, bandwidth),
        ] {
            assert_eq!(unsafe { ffi::opus_encoder_ctl(ptr, request, value) }, 0);
        }
        Self(ptr)
    }

    fn encode(&mut self, pcm: &[f32]) -> Vec<u8> {
        let mut out = [0u8; 1500];
        let n = unsafe {
            ffi::opus_encode_float(
                self.0,
                pcm.as_ptr(),
                pcm.len() as c_int,
                out.as_mut_ptr(),
                1500,
            )
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

struct Decoder(*mut ffi::OpusDecoder);

impl Decoder {
    fn new(rate: usize) -> Self {
        let mut err: c_int = 0;
        let ptr = unsafe { ffi::opus_decoder_create(rate as i32, 1, &mut err) };
        assert!(!ptr.is_null() && err == 0, "opus_decoder_create: {err}");
        Self(ptr)
    }

    /// An empty packet conceals `out.len()` samples.
    fn decode(&mut self, data: &[u8], out: &mut [f32]) -> usize {
        let (ptr, len) = if data.is_empty() {
            (std::ptr::null(), 0)
        } else {
            (data.as_ptr(), data.len() as c_int)
        };
        let n = unsafe {
            ffi::opus_decode_float(self.0, ptr, len, out.as_mut_ptr(), out.len() as c_int, 0)
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

/// A voiced signal: harmonics of a gliding 140 Hz pitch, a little noise,
/// under a syllable-rate envelope.
fn signal(n: usize) -> Vec<f32> {
    let mut seed = 7u32;
    let mut phase = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            phase += 2.0 * PI * (140.0 + 30.0 * (2.0 * PI * 0.7 * t).sin()) / RATE as f32;
            let mut v = 0.0;
            for h in 1..=40 {
                v += 0.15 / h as f32 * (h as f32 * phase).sin();
            }
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
            (v + 0.04 * noise) * (0.55 + 0.45 * (2.0 * PI * 3.0 * t).sin())
        })
        .collect()
}

/// libopus packets of `signal` in frames of `frame` samples, from an encoder
/// held in `mode`.
fn packets(mode: c_int, bitrate: i32, frame: usize, count: usize) -> Vec<Vec<u8>> {
    let mut enc = Encoder::new(mode, bitrate);
    signal(frame * count)
        .chunks_exact(frame)
        .map(|x| enc.encode(x))
        .collect()
}

fn snr(want: &[f32], got: &[f32]) -> f64 {
    let (mut s, mut e) = (0f64, 0f64);
    for (x, y) in want.iter().zip(got) {
        s += (*x as f64).powi(2);
        e += (*x as f64 - *y as f64).powi(2);
    }
    10.0 * (s / e.max(1e-30)).log10()
}

#[test]
fn mono_pitch_downsample_matches_stereo_with_a_silent_channel() {
    // Twice the decimated length plus the overlap: the size of the CELT
    // decode history, where the SIMD path is taken.
    let x = signal(2048 + 120);
    let silent = vec![0.0f32; x.len()];
    let (mut mono, mut stereo) = ([0.0f32; 1024], [0.0f32; 1024]);
    pitch_downsample(&[&x], &mut mono, 1024, 1, 2);
    pitch_downsample(&[&x, &silent], &mut stereo, 1024, 2, 2);
    let peak = stereo.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = mono
        .iter()
        .zip(&stereo)
        .map(|(a, b)| (a - b).abs())
        .enumerate()
        .fold((0, 0.0f32), |w, (i, d)| if d > w.1 { (i, d) } else { w });
    assert!(
        worst.1 <= 1e-6 * peak,
        "mono x_lp[{}] differs from stereo by {} (peak {peak})",
        worst.0,
        worst.1
    );
}

#[test]
fn switch_out_of_celt_without_redundancy_bridges_like_libopus() {
    // A CELT stream followed by a Hybrid or SILK one, as when the encoder
    // restarts: the switch carries no redundant frame, so the decoder
    // bridges it with 5 ms of the CELT history concealed forward.
    const SWITCH: usize = 25;
    let celt = packets(MODE_CELT_ONLY, 64000, 960, 30);
    let mut worst = (f64::INFINITY, String::new());
    for (mode, bitrate, name) in [
        (MODE_HYBRID, 24000, "Hybrid"),
        (MODE_SILK_ONLY, 16000, "SILK"),
    ] {
        let next = packets(mode, bitrate, 960, 30);
        let stream: Vec<&[u8]> = (0..30)
            .map(|i| {
                if i < SWITCH {
                    &celt[i][..]
                } else {
                    &next[i][..]
                }
            })
            .collect();
        for rate in [48000usize, 24000, 16000] {
            let mut c_dec = Decoder::new(rate);
            let mut rs_dec = OpusDecoder::new(rate as i32, 1).unwrap();
            let (mut a, mut b) = (vec![0f32; 5760], vec![0f32; 5760]);
            for (i, p) in stream.iter().enumerate().take(SWITCH + 2) {
                let n = c_dec.decode(p, &mut a);
                assert_eq!(rs_dec.decode(p, 5760, &mut b).unwrap(), n);
                let db = snr(&a[..n], &b[..n]);
                if i >= SWITCH && db < worst.0 {
                    worst = (db, format!("CELT to {name}, {rate} Hz, packet {i}"));
                }
            }
        }
    }
    println!("worst: {}: {:.1} dB from libopus", worst.1, worst.0);
    assert!(
        worst.0 > 40.0,
        "{}: {:.1} dB from libopus",
        worst.1,
        worst.0
    );
}

#[test]
fn loss_after_a_short_history_conceals_like_libopus() {
    // A decoder that has decoded a single 5 ms CELT packet holds 5 ms of
    // history, as after a redundant frame, and conceals the next 20 ms in
    // four 5 ms frames, periodic from that history.
    let celt = packets(MODE_CELT_ONLY, 64000, 240, 90);
    let (mut worst, mut below) = ((f64::INFINITY, String::new()), 0);
    for rate in [48000usize, 24000, 16000] {
        let f5 = rate / 200;
        for start in (10..90).step_by(10) {
            let mut c_dec = Decoder::new(rate);
            let mut rs_dec = OpusDecoder::new(rate as i32, 1).unwrap();
            let (mut a, mut b) = (vec![0f32; 5760], vec![0f32; 5760]);
            c_dec.decode(&celt[start], &mut a);
            rs_dec.decode(&celt[start], 5760, &mut b).unwrap();
            let (mut want, mut got) = (Vec::new(), Vec::new());
            for _ in 0..4 {
                let n = c_dec.decode(&[], &mut a[..f5]);
                assert_eq!(rs_dec.decode(&[], f5, &mut b).unwrap(), n);
                want.extend_from_slice(&a[..n]);
                got.extend_from_slice(&b[..n]);
            }
            let db = snr(&want, &got);
            below += usize::from(db <= 40.0);
            if db < worst.0 {
                worst = (db, format!("{rate} Hz, loss after packet {start} alone"));
            }
        }
    }
    println!(
        "worst: {}: {:.1} dB from libopus; {below} of 24 at 40 dB or less",
        worst.1, worst.0
    );
    assert!(
        below == 0,
        "{}: {:.1} dB from libopus; {below} of 24 at 40 dB or less",
        worst.1,
        worst.0
    );
}
