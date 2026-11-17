//! OSCE feature extraction and cross-fades of opus-rs (`dnn::osce`)
//! against libopus's C on the same inputs: every float and every sample
//! must have the same bits.
#![cfg(feature = "dnn")]
use core::ffi::{c_int, c_void};
use opus_rs::dnn::osce::{self as rs, features as rf};
use opusref::dnn::{Rng, same_bits};
use opusref::dnn_osce as c;

fn check(what: &str, rust: &[f32], cref: &[f32]) {
    assert_eq!(rust.len(), cref.len());
    if let Some(i) = same_bits(rust, cref) {
        panic!("{what}: first difference at {i}: rust {:e} ({:#x}), C {:e} ({:#x})", rust[i], rust[i].to_bits(), cref[i], cref[i].to_bits());
    }
}

fn check_i16(what: &str, rust: &[i16], cref: &[i16]) {
    assert_eq!(rust.len(), cref.len());
    if let Some(i) = rust.iter().zip(cref).position(|(a, b)| a != b) {
        panic!("{what}: first difference at {i}: rust {}, C {}", rust[i], cref[i]);
    }
}

/// A C allocation freed on drop.
struct CBox(*mut c_void);
impl Drop for CBox {
    fn drop(&mut self) {
        unsafe { c::ref_osce_free(self.0) }
    }
}

fn uni_vec(r: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| r.uni() * scale).collect()
}

/// A synthetic speech-like 16 kHz signal: harmonics of a drifting pitch
/// under a syllable envelope, with noise, pauses and clipping.
struct Synth {
    phase: f64,
    f0: f64,
    t: u64,
}

impl Synth {
    fn new() -> Self {
        Synth { phase: 0.0, f0: 140.0, t: 0 }
    }
    fn frame(&mut self, r: &mut Rng, len: usize) -> Vec<i16> {
        let mut out = Vec::with_capacity(len);
        let loud = match r.below(8) {
            0 => 0.0,
            1 => 40000.0,
            _ => 2000.0 + r.below(12000) as f64,
        };
        let noise = r.below(1500) as f64;
        for _ in 0..len {
            self.f0 = (self.f0 + r.uni() as f64 * 0.3).clamp(70.0, 400.0);
            self.phase += 2.0 * std::f64::consts::PI * self.f0 / 16000.0;
            let env = 0.5 + 0.5 * (self.t as f64 * 2.0 * std::f64::consts::PI / 4000.0).sin();
            let mut s = 0.0;
            for h in 1..12 {
                s += (h as f64 * self.phase).sin() / h as f64;
            }
            let v = loud * env * s * 0.5 + noise * r.uni() as f64;
            out.push(v.round().clamp(-32768.0, 32767.0) as i16);
            self.t += 1;
        }
        out
    }
}

/// Random but plausible SILK frame parameters.
fn silk_params(r: &mut Rng, fs_khz: i32, nb_subfr: usize) -> rs::OsceFrameInput {
    let lpc_order = if fs_khz == 16 { 16 } else { 10 };
    let mut p = rs::OsceFrameInput {
        fs_khz,
        nb_subfr,
        lpc_order,
        signal_type: r.below(3) as i32,
        num_bits: 20 + r.below(700) as i32,
        ..Default::default()
    };
    for k in 0..4 {
        p.pitch_l[k] = 32 + r.below(257) as i32;
        p.gains_q16[k] = match r.below(6) {
            0 => r.below(1 << 16) as i32,
            1 => (r.below(1 << 30) as i32).max(1),
            _ => ((1u32 << (12 + r.below(14))) + r.below(1 << 12)) as i32,
        };
    }
    for h in 0..2 {
        for i in 0..lpc_order {
            p.pred_coef_q12[h][i] = (r.uni() * 6000.0 / (1 + i) as f32) as i16;
        }
    }
    for v in p.ltp_coef_q14.iter_mut() {
        *v = (r.uni() * 9000.0 + 2000.0) as i16;
    }
    p
}

fn c_set(d: *mut c_void, p: &rs::OsceFrameInput) {
    let pred: Vec<i16> = p.pred_coef_q12.iter().flatten().copied().collect();
    unsafe {
        c::ref_osce_dec_set(
            d, p.fs_khz, p.nb_subfr as c_int, p.lpc_order as c_int, p.signal_type, p.pitch_l.as_ptr(),
            p.gains_q16.as_ptr(), pred.as_ptr(), p.ltp_coef_q14.as_ptr(),
        )
    };
}

#[test]
fn features() {
    let mut r = Rng(0xfea7);
    let mut synth = Synth::new();
    let mut st = rf::OsceFeatureState::new();
    let d = CBox(unsafe { c::ref_osce_dec_new() });
    unsafe { c::ref_osce_reset(d.0, 2) };
    for frame in 0..300 {
        let nb_subfr = if frame % 11 == 5 { 2 } else { 4 };
        let p = silk_params(&mut r, 16, nb_subfr);
        let xq = synth.frame(&mut r, 80 * nb_subfr);
        let mut f = [0.0f32; 4 * rs::OSCE_FEATURE_DIM];
        let mut nb = [0.0f32; 2];
        let mut per = [0i32; 4];
        rf::osce_calculate_features(&mut st, &p, &mut f, &mut nb, &mut per, &xq);
        let mut fc = [0.0f32; 4 * rs::OSCE_FEATURE_DIM];
        let mut nbc = [0.0f32; 2];
        let mut perc = [0i32; 4];
        c_set(d.0, &p);
        unsafe { c::ref_osce_dec_features(d.0, xq.as_ptr(), p.num_bits, fc.as_mut_ptr(), nbc.as_mut_ptr(), perc.as_mut_ptr()) };
        let n = nb_subfr * rs::OSCE_FEATURE_DIM;
        check(&format!("features frame {frame}"), &f[..n], &fc[..n]);
        check(&format!("numbits frame {frame}"), &nb, &nbc);
        assert_eq!(per[..nb_subfr], perc[..nb_subfr], "periods frame {frame}");
    }
}

#[test]
fn cross_fades() {
    let mut r = Rng(0xc7055);
    for _ in 0..50 {
        let mut a = uni_vec(&mut r, 320, 1.0);
        let mut ac = a.clone();
        let mut x = uni_vec(&mut r, 320, 1.0);
        rf::osce_cross_fade_10ms(&mut a, &x);
        unsafe { c::ref_osce_cross_fade_10ms(ac.as_mut_ptr(), x.as_mut_ptr(), 320) };
        check("osce_cross_fade_10ms", &a, &ac);

        let mut i: Vec<i16> = (0..960).map(|_| r.next_u32() as i16).collect();
        let mut ic = i.clone();
        let mut o: Vec<i16> = (0..960).map(|_| r.next_u32() as i16).collect();
        rf::osce_bwe_cross_fade_10ms(&mut i, &o);
        unsafe { c::ref_osce_bwe_cross_fade_10ms(ic.as_mut_ptr(), o.as_mut_ptr(), 960) };
        check_i16("osce_bwe_cross_fade_10ms", &i, &ic);
    }
}
