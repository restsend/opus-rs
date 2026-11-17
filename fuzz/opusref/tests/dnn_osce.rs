//! OSCE of opus-rs (`dnn::osce`) against libopus's C on the same inputs:
//! the nndsp kernels with the real model layers, the feature extraction,
//! LACE, NoLACE, BBWENet and the whole `osce_enhance_frame`/`osce_bwe`
//! over frame sequences with resets and method switches. Every float and
//! every sample must have the same bits.
#![cfg(feature = "dnn")]
use core::ffi::{c_int, c_void};
use opus_rs::dnn::nnet::LinearLayer;
use opus_rs::dnn::osce::{
    self as rs, OsceMethod, bbwenet as rb, features as rf, lace as rl, nndsp, nolace as rn,
};
use opusref::dnn::{Rng, same_bits};
use opusref::dnn_osce::{self as c, layer};

fn check(what: &str, rust: &[f32], cref: &[f32]) {
    assert_eq!(rust.len(), cref.len());
    if let Some(i) = same_bits(rust, cref) {
        panic!(
            "{what}: first difference at {i}: rust {:e} ({:#x}), C {:e} ({:#x})",
            rust[i],
            rust[i].to_bits(),
            cref[i],
            cref[i].to_bits()
        );
    }
}

fn check_i16(what: &str, rust: &[i16], cref: &[i16]) {
    assert_eq!(rust.len(), cref.len());
    if let Some(i) = rust.iter().zip(cref).position(|(a, b)| a != b) {
        panic!(
            "{what}: first difference at {i}: rust {}, C {}",
            rust[i], cref[i]
        );
    }
}

/// A C allocation freed on drop.
struct CBox(*mut c_void);
impl Drop for CBox {
    fn drop(&mut self) {
        unsafe { c::ref_osce_free(self.0) }
    }
}

struct Model {
    c: CBox,
    rs: rs::OsceModel<'static>,
}

fn model() -> Model {
    Model {
        c: CBox(unsafe { c::ref_osce_model_new() }),
        rs: rs::OsceModel::builtin().unwrap(),
    }
}

impl Model {
    fn layer(&self, name: &str) -> *const opusref::dnn::LinearLayer {
        layer(self.c.0, name)
    }
}

fn uni_vec(r: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| r.uni() * scale).collect()
}

#[test]
fn overlap_windows() {
    for n in [1usize, 7, 40, 80, 120] {
        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        nndsp::compute_overlap_window(&mut a);
        unsafe { c::ref_compute_overlap_window(b.as_mut_ptr(), n as c_int) };
        check(&format!("window {n}"), &a, &b);
    }
    let m = model();
    let rsw: [&[f32]; 5] = [
        &m.rs.lace.window,
        &m.rs.nolace.window,
        &m.rs.bbwenet.window16,
        &m.rs.bbwenet.window32,
        &m.rs.bbwenet.window48,
    ];
    for (i, w) in rsw.iter().enumerate() {
        let cw =
            unsafe { core::slice::from_raw_parts(c::ref_osce_window(m.c.0, i as c_int), w.len()) };
        check(&format!("model window {i}"), w, cw);
    }
}

#[test]
fn adaconv() {
    let m = model();
    let l = &m.rs;
    let cases: [(
        &str,
        &LinearLayer,
        &LinearLayer,
        nndsp::AdaConvConfig,
        usize,
        &[f32],
    ); 7] = [
        (
            "lace_af1",
            &l.lace.layers.lace_af1_kernel,
            &l.lace.layers.lace_af1_gain,
            rl::AF1,
            128,
            &l.lace.window,
        ),
        (
            "nolace_af1",
            &l.nolace.layers.nolace_af1_kernel,
            &l.nolace.layers.nolace_af1_gain,
            rn::AF1,
            160,
            &l.nolace.window,
        ),
        (
            "nolace_af2",
            &l.nolace.layers.nolace_af2_kernel,
            &l.nolace.layers.nolace_af2_gain,
            rn::AF2,
            160,
            &l.nolace.window,
        ),
        (
            "nolace_af4",
            &l.nolace.layers.nolace_af4_kernel,
            &l.nolace.layers.nolace_af4_gain,
            rn::AF4,
            160,
            &l.nolace.window,
        ),
        (
            "bbwenet_af1",
            &l.bbwenet.layers.bbwenet_af1_kernel,
            &l.bbwenet.layers.bbwenet_af1_gain,
            rb::AF1,
            128,
            &l.bbwenet.window16,
        ),
        (
            "bbwenet_af2",
            &l.bbwenet.layers.bbwenet_af2_kernel,
            &l.bbwenet.layers.bbwenet_af2_gain,
            rb::AF2,
            128,
            &l.bbwenet.window32,
        ),
        (
            "bbwenet_af3",
            &l.bbwenet.layers.bbwenet_af3_kernel,
            &l.bbwenet.layers.bbwenet_af3_gain,
            rb::AF3,
            128,
            &l.bbwenet.window48,
        ),
    ];
    let mut r = Rng(0x0dd5_eed1);
    for (name, kl, gl, cfg, fdim, window) in cases {
        let ck = m.layer(&format!("{name}_kernel"));
        let cg = m.layer(&format!("{name}_gain"));
        let mut st = nndsp::AdaConvState::new();
        let cst = CBox(unsafe { c::ref_adaconv_state_new() });
        for frame in 0..40 {
            let scale = if frame % 7 == 3 { 0.0 } else { 0.5 };
            let x = uni_vec(&mut r, cfg.in_channels * cfg.frame_size, scale);
            let feat = uni_vec(&mut r, fdim, 1.0);
            let mut out = vec![0.0f32; cfg.out_channels * cfg.frame_size];
            let mut out_c = out.clone();
            nndsp::adaconv_process_frame(&mut st, &mut out, &x, &feat, kl, gl, &cfg, window);
            unsafe {
                c::ref_adaconv_process_frame(
                    cst.0,
                    out_c.as_mut_ptr(),
                    x.as_ptr(),
                    feat.as_ptr(),
                    ck,
                    cg,
                    fdim as c_int,
                    cfg.frame_size as c_int,
                    cfg.overlap_size as c_int,
                    cfg.in_channels as c_int,
                    cfg.out_channels as c_int,
                    cfg.kernel_size as c_int,
                    cfg.left_padding as c_int,
                    cfg.filter_gain_a,
                    cfg.filter_gain_b,
                    cfg.shape_gain,
                    window.as_ptr(),
                )
            };
            check(&format!("adaconv {name} frame {frame}"), &out, &out_c);
        }
    }
}

#[test]
fn adacomb() {
    let m = model();
    let l = &m.rs;
    let cases: [(&str, [&LinearLayer; 3], nndsp::AdaCombConfig, usize, &[f32]); 4] = [
        (
            "lace_cf1",
            [
                &l.lace.layers.lace_cf1_kernel,
                &l.lace.layers.lace_cf1_gain,
                &l.lace.layers.lace_cf1_global_gain,
            ],
            rl::CF1,
            128,
            &l.lace.window,
        ),
        (
            "lace_cf2",
            [
                &l.lace.layers.lace_cf2_kernel,
                &l.lace.layers.lace_cf2_gain,
                &l.lace.layers.lace_cf2_global_gain,
            ],
            rl::CF2,
            128,
            &l.lace.window,
        ),
        (
            "nolace_cf1",
            [
                &l.nolace.layers.nolace_cf1_kernel,
                &l.nolace.layers.nolace_cf1_gain,
                &l.nolace.layers.nolace_cf1_global_gain,
            ],
            rn::CF1,
            160,
            &l.nolace.window,
        ),
        (
            "nolace_cf2",
            [
                &l.nolace.layers.nolace_cf2_kernel,
                &l.nolace.layers.nolace_cf2_gain,
                &l.nolace.layers.nolace_cf2_global_gain,
            ],
            rn::CF2,
            160,
            &l.nolace.window,
        ),
    ];
    let mut r = Rng(0xc0b_f117);
    for (name, [kl, gl, ggl], cfg, fdim, window) in cases {
        let ck = m.layer(&format!("{name}_kernel"));
        let cg = m.layer(&format!("{name}_gain"));
        let cgg = m.layer(&format!("{name}_global_gain"));
        let mut st = nndsp::AdaCombState::new();
        let cst = CBox(unsafe { c::ref_adacomb_state_new() });
        let mut lag = 100;
        for frame in 0..80 {
            if r.below(3) != 0 {
                lag = match r.below(4) {
                    0 => 7,
                    1 => 300,
                    _ => 7 + r.below(294) as usize,
                };
            }
            let x = uni_vec(&mut r, cfg.frame_size, 0.5);
            let feat = uni_vec(&mut r, fdim, 1.0);
            let mut out = x.clone();
            let mut out_c = vec![0.0f32; cfg.frame_size];
            nndsp::adacomb_process_frame(&mut st, &mut out, &feat, kl, gl, ggl, lag, &cfg, window);
            unsafe {
                c::ref_adacomb_process_frame(
                    cst.0,
                    out_c.as_mut_ptr(),
                    x.as_ptr(),
                    feat.as_ptr(),
                    ck,
                    cg,
                    cgg,
                    lag as c_int,
                    fdim as c_int,
                    cfg.frame_size as c_int,
                    cfg.overlap_size as c_int,
                    cfg.kernel_size as c_int,
                    cfg.left_padding as c_int,
                    cfg.filter_gain_a,
                    cfg.filter_gain_b,
                    cfg.log_gain_limit,
                    window.as_ptr(),
                )
            };
            check(&format!("adacomb {name} frame {frame}"), &out, &out_c);
        }
    }
}

#[test]
fn adashape() {
    let m = model();
    let l = &m.rs;
    let n = &l.nolace.layers;
    let b = &l.bbwenet.layers;
    let cases: [(&str, [&LinearLayer; 3], nndsp::AdaShapeConfig); 5] = [
        (
            "nolace_tdshape1",
            [
                &n.nolace_tdshape1_alpha1_f,
                &n.nolace_tdshape1_alpha1_t,
                &n.nolace_tdshape1_alpha2,
            ],
            rn::TDSHAPE1,
        ),
        (
            "nolace_tdshape2",
            [
                &n.nolace_tdshape2_alpha1_f,
                &n.nolace_tdshape2_alpha1_t,
                &n.nolace_tdshape2_alpha2,
            ],
            rn::TDSHAPE2,
        ),
        (
            "nolace_tdshape3",
            [
                &n.nolace_tdshape3_alpha1_f,
                &n.nolace_tdshape3_alpha1_t,
                &n.nolace_tdshape3_alpha2,
            ],
            rn::TDSHAPE3,
        ),
        (
            "bbwenet_tdshape1",
            [
                &b.bbwenet_tdshape1_alpha1_f,
                &b.bbwenet_tdshape1_alpha1_t,
                &b.bbwenet_tdshape1_alpha2,
            ],
            rb::TDSHAPE1,
        ),
        (
            "bbwenet_tdshape2",
            [
                &b.bbwenet_tdshape2_alpha1_f,
                &b.bbwenet_tdshape2_alpha1_t,
                &b.bbwenet_tdshape2_alpha2,
            ],
            rb::TDSHAPE2,
        ),
    ];
    let mut r = Rng(0x5aa9e);
    for (name, [a1f, a1t, a2], cfg) in cases {
        let ca1f = m.layer(&format!("{name}_alpha1_f"));
        let ca1t = m.layer(&format!("{name}_alpha1_t"));
        let ca2 = m.layer(&format!("{name}_alpha2"));
        let mut st = nndsp::AdaShapeState::new();
        let cst = CBox(unsafe { c::ref_adashape_state_new() });
        for frame in 0..60 {
            let scale = [0.0, 1e-4, 0.1, 0.9][frame % 4];
            let x = uni_vec(&mut r, cfg.frame_size, scale);
            let feat = uni_vec(&mut r, cfg.feature_dim, 1.0);
            let mut out = x.clone();
            let mut out_c = vec![0.0f32; cfg.frame_size];
            nndsp::adashape_process_frame(&mut st, &mut out, &feat, a1f, a1t, a2, &cfg);
            unsafe {
                c::ref_adashape_process_frame(
                    cst.0,
                    out_c.as_mut_ptr(),
                    x.as_ptr(),
                    feat.as_ptr(),
                    ca1f,
                    ca1t,
                    ca2,
                    cfg.feature_dim as c_int,
                    cfg.frame_size as c_int,
                    cfg.avg_pool_k as c_int,
                    cfg.interpolate_k as c_int,
                )
            };
            check(&format!("adashape {name} frame {frame}"), &out, &out_c);
        }
    }
}

#[test]
fn numbits_embedding() {
    use opus_rs::dnn::models::lace::*;
    let scales = [
        LACE_NUMBITS_SCALE_0,
        LACE_NUMBITS_SCALE_1,
        LACE_NUMBITS_SCALE_2,
        LACE_NUMBITS_SCALE_3,
        LACE_NUMBITS_SCALE_4,
        LACE_NUMBITS_SCALE_5,
        LACE_NUMBITS_SCALE_6,
        LACE_NUMBITS_SCALE_7,
    ];
    let lo = 50f64.ln() as f32;
    let hi = 650f64.ln() as f32;
    let mut r = Rng(77);
    let mut values: Vec<f32> = vec![0.0, 1.0, 10.0, 49.0, 50.0, 650.0, 651.0, 5000.0];
    values.extend((0..500).map(|_| r.below(100000) as f32 / 100.0));
    for v in values {
        let mut a = [0.0f32; 8];
        let mut b = [0.0f32; 8];
        rl::numbits_embedding(&mut a, v, lo, hi, &scales);
        unsafe { c::ref_lace_numbits_embedding(b.as_mut_ptr(), v, lo, hi) };
        check(&format!("numbits embedding {v}"), &a, &b);
    }
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
        Synth {
            phase: 0.0,
            f0: 140.0,
            t: 0,
        }
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
            d,
            p.fs_khz,
            p.nb_subfr as c_int,
            p.lpc_order as c_int,
            p.signal_type,
            p.pitch_l.as_ptr(),
            p.gains_q16.as_ptr(),
            pred.as_ptr(),
            p.ltp_coef_q14.as_ptr(),
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
        unsafe {
            c::ref_osce_dec_features(
                d.0,
                xq.as_ptr(),
                p.num_bits,
                fc.as_mut_ptr(),
                nbc.as_mut_ptr(),
                perc.as_mut_ptr(),
            )
        };
        let n = nb_subfr * rs::OSCE_FEATURE_DIM;
        check(&format!("features frame {frame}"), &f[..n], &fc[..n]);
        check(&format!("numbits frame {frame}"), &nb, &nbc);
        assert_eq!(per[..nb_subfr], perc[..nb_subfr], "periods frame {frame}");
    }
}

/// LACE and NoLACE on features from the real extraction, compared as floats.
#[test]
fn lace_nolace_frames() {
    let m = model();
    let mut r = Rng(0x1ace);
    let mut synth = Synth::new();
    let mut fst = rf::OsceFeatureState::new();
    let mut ls = rl::LaceState::new();
    let mut ns = rn::NolaceState::new();
    let cls = CBox(unsafe { c::ref_lace_state_new() });
    let cns = CBox(unsafe { c::ref_nolace_state_new() });
    for frame in 0..120 {
        let p = silk_params(&mut r, 16, 4);
        let xq = synth.frame(&mut r, 320);
        let mut f = [0.0f32; 4 * rs::OSCE_FEATURE_DIM];
        let mut nb = [0.0f32; 2];
        let mut per = [0i32; 4];
        rf::osce_calculate_features(&mut fst, &p, &mut f, &mut nb, &mut per, &xq);
        let x: Vec<f32> = xq.iter().map(|&v| v as f32 * (1.0 / 32768.0)).collect();
        let mut out = [0.0f32; 320];
        let mut out_c = [0.0f32; 320];
        rl::lace_process_20ms_frame(&m.rs.lace, &mut ls, &mut out, &x, &f, &nb, &per);
        unsafe {
            c::ref_lace_process_20ms_frame(
                m.c.0,
                cls.0,
                out_c.as_mut_ptr(),
                x.as_ptr(),
                f.as_ptr(),
                nb.as_ptr(),
                per.as_ptr(),
            )
        };
        check(&format!("lace frame {frame}"), &out, &out_c);
        rn::nolace_process_20ms_frame(&m.rs.nolace, &mut ns, &mut out, &x, &f, &nb, &per);
        unsafe {
            c::ref_nolace_process_20ms_frame(
                m.c.0,
                cns.0,
                out_c.as_mut_ptr(),
                x.as_ptr(),
                f.as_ptr(),
                nb.as_ptr(),
                per.as_ptr(),
            )
        };
        check(&format!("nolace frame {frame}"), &out, &out_c);
    }
}

fn method_of(i: u32) -> OsceMethod {
    [OsceMethod::None, OsceMethod::Lace, OsceMethod::Nolace][i as usize]
}

/// `osce_enhance_frame` on a decoder channel over a frame sequence, as
/// silk_Decode drives it: method switches through `osce_reset`, lost
/// frames (reset), frames at other rates or lengths (reset inside), and
/// frames without a loaded model.
#[test]
fn enhance_frame_sequence() {
    let m = model();
    let mut total_changed = 0;
    for seed in 0..3u64 {
        let mut r = Rng(0xe7a7ce + seed);
        let mut synth = Synth::new();
        let mut st = rs::Osce::new();
        let d = CBox(unsafe { c::ref_osce_dec_new() });
        assert_eq!(unsafe { c::ref_osce_method(d.0) }, st.method as c_int);
        let mut requested = OsceMethod::Nolace;
        let mut changed = 0;
        for frame in 0..150 {
            if r.below(12) == 0 {
                requested = method_of(r.below(3));
            }
            // silk_Decode: reset when the requested method changes.
            if st.method != requested {
                rs::osce_reset(&mut st, requested);
                unsafe { c::ref_osce_reset(d.0, requested as c_int) };
            }
            let lost = r.below(25) == 0;
            if lost {
                // decode_frame's PLC branch.
                let method = st.method;
                rs::osce_reset(&mut st, method);
                unsafe { c::ref_osce_reset(d.0, method as c_int) };
                continue;
            }
            let (fs, nb_subfr) = match r.below(20) {
                0 => (8, 4),
                1 => (12, 4),
                2 => (16, 2),
                _ => (16, 4),
            };
            let loaded = r.below(15) != 0;
            unsafe { c::ref_osce_model_set_loaded(m.c.0, loaded as c_int) };
            let p = silk_params(&mut r, fs, nb_subfr);
            let mut xq = synth.frame(&mut r, 320);
            let mut xq_c = xq.clone();
            let before = xq.clone();
            rs::osce_enhance_frame(loaded.then_some(&m.rs), &mut st, &p, &mut xq);
            changed += (xq
                .iter()
                .zip(&before)
                .filter(|(a, b)| a.abs_diff(**b) > 1)
                .count()
                > 100) as usize;
            c_set(d.0, &p);
            unsafe { c::ref_osce_enhance_frame(m.c.0, d.0, xq_c.as_mut_ptr(), p.num_bits) };
            let what = format!(
                "seed {seed} frame {frame} ({:?}, fs {fs}, nb_subfr {nb_subfr}, loaded {loaded})",
                st.method
            );
            check_i16(&what, &xq, &xq_c);
            assert_eq!(
                unsafe { c::ref_osce_method(d.0) },
                st.method as c_int,
                "{what}"
            );
            assert_eq!(
                unsafe { c::ref_osce_reset_count(d.0) },
                st.features.reset,
                "{what}"
            );
            let smooth = unsafe { c::ref_osce_numbits_smooth(d.0) };
            assert_eq!(
                smooth.to_bits(),
                st.features.numbits_smooth.to_bits(),
                "{what}"
            );
        }
        unsafe { c::ref_osce_model_set_loaded(m.c.0, 1) };
        total_changed += changed;
    }
    // The enhancers did run and change the signal.
    assert!(total_changed > 100, "only {total_changed} frames enhanced");
}

#[test]
fn resamplers_and_activation() {
    let mut r = Rng(0x2e5a);
    let mut st = rb::ResampState::default();
    let cst = CBox(unsafe { c::ref_resamp_state_new() });
    for frame in 0..50 {
        let x = uni_vec(&mut r, 80, 1.0);
        let mut y = vec![0.0f32; 160];
        let mut yc = y.clone();
        rb::upsamp_2x(&mut st, &mut y, &x);
        unsafe { c::ref_upsamp_2x(cst.0, yc.as_mut_ptr(), x.as_ptr(), 80) };
        check(&format!("upsamp_2x frame {frame}"), &y, &yc);
        let mut z = vec![0.0f32; 240];
        let mut zc = z.clone();
        rb::interpol_3_2(&mut st, &mut z, &y);
        unsafe { c::ref_interpol_3_2(cst.0, zc.as_mut_ptr(), y.as_ptr(), 160) };
        check(&format!("interpol_3_2 frame {frame}"), &z, &zc);
        let mut a = uni_vec(&mut r, 240, [1e-7, 1e-3, 1.0, 3.0][frame % 4]);
        let mut ac = a.clone();
        rb::apply_valin_activation(&mut a);
        unsafe { c::ref_apply_valin_activation(ac.as_mut_ptr(), 240) };
        check(&format!("valin frame {frame}"), &a, &ac);
    }
}

#[test]
fn bwe_features() {
    let mut r = Rng(0xb3e);
    let mut synth = Synth::new();
    let mut st = rs::OsceBwe::new();
    rs::osce_bwe_reset(&mut st);
    let cst = CBox(unsafe { c::ref_osce_bwe_new() });
    unsafe { c::ref_osce_bwe_reset(cst.0) };
    for frame in 0..200 {
        let len = if r.below(2) == 0 { 160 } else { 320 };
        let xq = synth.frame(&mut r, len);
        let n = len / 160 * rs::OSCE_BWE_FEATURE_DIM;
        let mut f = vec![0.0f32; n];
        let mut fc = vec![0.0f32; n];
        rf::osce_bwe_calculate_features(&mut st.features, &mut f, &xq);
        unsafe { c::ref_osce_bwe_features(cst.0, fc.as_mut_ptr(), xq.as_ptr(), len as c_int) };
        check(&format!("bwe features frame {frame}"), &f, &fc);
    }
}

/// BBWENet on features from the real extraction, compared as floats.
#[test]
fn bbwenet_frames() {
    let m = model();
    let mut r = Rng(0xbb3e);
    let mut synth = Synth::new();
    let mut fst = rs::OsceBwe::new();
    rs::osce_bwe_reset(&mut fst);
    let mut st = rb::BbwenetState::new();
    let cst = CBox(unsafe { c::ref_bbwenet_state_new() });
    for frame in 0..100 {
        let num_frames = 1 + r.below(2) as usize;
        let xq = synth.frame(&mut r, 160 * num_frames);
        let mut f = vec![0.0f32; num_frames * rs::OSCE_BWE_FEATURE_DIM];
        rf::osce_bwe_calculate_features(&mut fst.features, &mut f, &xq);
        let x: Vec<f32> = xq.iter().map(|&v| v as f32 * (1.0 / 32768.0)).collect();
        let mut out = vec![0.0f32; 480 * num_frames];
        let mut out_c = out.clone();
        rb::bbwenet_process_frames(&m.rs.bbwenet, &mut st, &mut out, &x, &f, num_frames);
        unsafe {
            c::ref_bbwenet_process_frames(
                m.c.0,
                cst.0,
                out_c.as_mut_ptr(),
                x.as_ptr(),
                f.as_ptr(),
                num_frames as c_int,
            )
        };
        check(
            &format!("bbwenet frame {frame} ({num_frames} frames)"),
            &out,
            &out_c,
        );
    }
}

/// `osce_bwe` over a sequence of 10 and 20 ms frames with resets.
#[test]
fn bwe_sequence() {
    let m = model();
    let mut r = Rng(0xb3e5e9);
    let mut synth = Synth::new();
    let mut st = rs::OsceBwe::new();
    let cst = CBox(unsafe { c::ref_osce_bwe_new() });
    let mut energy = 0.0;
    for frame in 0..150 {
        if frame == 0 || r.below(30) == 0 {
            rs::osce_bwe_reset(&mut st);
            unsafe { c::ref_osce_bwe_reset(cst.0) };
        }
        let len = if r.below(3) == 0 { 160 } else { 320 };
        let xq = synth.frame(&mut r, len);
        let mut xq_c = xq.clone();
        let mut out = vec![0i16; 3 * len];
        let mut out_c = out.clone();
        rs::osce_bwe(&m.rs, &mut st, &mut out, &xq);
        unsafe {
            c::ref_osce_bwe(
                m.c.0,
                cst.0,
                out_c.as_mut_ptr(),
                xq_c.as_mut_ptr(),
                len as c_int,
            )
        };
        check_i16(&format!("osce_bwe frame {frame} (len {len})"), &out, &out_c);
        energy += out.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
    }
    assert!(energy > 1e9, "silent bandwidth extension");
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
