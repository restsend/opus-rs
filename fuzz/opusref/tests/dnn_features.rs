//! The LPCNet feature analysis of opus-rs (FFT, band energies, Burg
//! cepstrum, LPC, PitchDNN, `lpcnet_compute_single_frame_features`)
//! against libopus: every output must have the same bits.
#![cfg(feature = "dnn")]
use opus_rs::dnn::{builtin, burg, fft::Cpx, freq, lpcnet_enc, pitchdnn};
use opusref::dnn::{self as c, Rng, same_bits};

/// A voiced-speech-like test signal at 16 kHz in the int16 range: a few
/// harmonics of a gliding pitch with an envelope, plus noise and silences.
pub fn speechlike(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Rng(seed);
    let mut phase = 0.0f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / 16000.0;
            let f0 = 110.0 + 70.0 * (t * 1.3).sin() + 30.0 * (t * 5.1).sin();
            phase += 2.0 * std::f64::consts::PI * f0 / 16000.0;
            let env = ((t * 2.7).sin() * 0.5 + 0.5).powi(2);
            let gate = if (i / 4000) % 5 == 4 { 0.0 } else { 1.0 };
            let mut v = 0.0;
            for h in 1..12 {
                v += (phase * h as f64).sin() / h as f64;
            }
            (gate * env * 9000.0 * v) as f32 + 300.0 * r.uni()
        })
        .collect()
}

#[test]
fn forward_transform_matches() {
    let mut r = Rng(1);
    for _ in 0..50 {
        let x: Vec<f32> = (0..freq::WINDOW_SIZE).map(|_| r.uni() * 1000.0).collect();
        let mut a = vec![0f32; 2 * freq::FREQ_SIZE];
        let mut b = [Cpx::default(); freq::FREQ_SIZE];
        unsafe { c::ref_forward_transform(a.as_mut_ptr(), x.as_ptr()) };
        freq::forward_transform(&mut b, &x);
        let b: Vec<f32> = b.iter().flat_map(|c| [c.r, c.i]).collect();
        assert_eq!(same_bits(&a, &b), None);
    }
}

#[test]
fn burg_and_cepstrum_match() {
    let sig = speechlike(16000, 2);
    let mut r = Rng(3);
    for f in 0..90 {
        let x = &sig[f * freq::FRAME_SIZE..(f + 1) * freq::FRAME_SIZE];
        let mut a = vec![0f32; 2 * freq::NB_BANDS];
        let mut b = vec![0f32; 2 * freq::NB_BANDS];
        unsafe { c::ref_burg_cepstral_analysis(a.as_mut_ptr(), x.as_ptr()) };
        freq::burg_cepstral_analysis(&mut b, x);
        assert_eq!(same_bits(&a, &b), None, "burg cepstrum frame {f}");

        let mut la = vec![0f32; 16];
        let mut lb = vec![0f32; 16];
        let ea = unsafe { c::ref_lpc_from_cepstrum(la.as_mut_ptr(), a.as_ptr()) };
        let eb = freq::lpc_from_cepstrum(&mut lb, &b);
        assert_eq!(same_bits(&la, &lb), None, "lpc frame {f}");
        assert_eq!(ea.to_bits(), eb.to_bits(), "lpc error frame {f}");

        let d = 1 + r.below(16) as usize;
        let sub = 20 + r.below(60) as usize;
        let nb = 1 + r.below(4) as usize;
        let y: Vec<f32> = (0..sub * nb).map(|i| sig[f * 100 + i] + r.uni()).collect();
        let mig = [1e-3f32, 1e-2, 0.25][f % 3];
        let mut aa = vec![0f32; 16];
        let mut ab = vec![0f32; 16];
        let ga = unsafe { c::ref_silk_burg_analysis(aa.as_mut_ptr(), y.as_ptr(), mig, sub as i32, nb as i32, d as i32) };
        let gb = burg::silk_burg_analysis(&mut ab, &y, mig, sub, nb, d);
        assert_eq!(same_bits(&aa[..d], &ab[..d]), None, "burg frame {f}");
        assert_eq!(ga.to_bits(), gb.to_bits(), "burg energy frame {f}");
    }
}

#[test]
fn pitchdnn_matches() {
    let mut r = Rng(4);
    let cst = unsafe { c::ref_pitchdnn_create() };
    let mut rst = pitchdnn::PitchDnnState::new(builtin::pitchdnn().unwrap());
    for f in 0..100 {
        let iff: Vec<f32> = (0..88).map(|_| r.uni()).collect();
        let xcf: Vec<f32> = (0..pitchdnn::NB_XCORR_FEATURES).map(|_| r.uni()).collect();
        let a = unsafe { c::ref_compute_pitchdnn(cst, iff.as_ptr(), xcf.as_ptr()) };
        let b = pitchdnn::compute_pitchdnn(&mut rst, &iff, &xcf);
        assert_eq!(a.to_bits(), b.to_bits(), "frame {f}: {a} vs {b}");
    }
    unsafe { c::ref_pitchdnn_destroy(cst) };
}

#[test]
fn frame_features_match() {
    for (seed, scale) in [(5u64, 1.0f32), (6, 0.01), (7, 3.0)] {
        let sig: Vec<f32> = speechlike(16000 * 4, seed).iter().map(|x| x * scale).collect();
        let cst = unsafe { c::ref_lpcnet_encoder_create() };
        let mut rst = lpcnet_enc::LpcnetEncState::new(builtin::pitchdnn().unwrap());
        for (f, x) in sig.chunks_exact(freq::FRAME_SIZE).enumerate() {
            let mut a = [0f32; lpcnet_enc::NB_TOTAL_FEATURES];
            let mut b = [0f32; lpcnet_enc::NB_TOTAL_FEATURES];
            unsafe { c::ref_lpcnet_features_float(cst, x.as_ptr(), a.as_mut_ptr()) };
            lpcnet_enc::lpcnet_compute_single_frame_features_float(&mut rst, x, &mut b);
            assert_eq!(same_bits(&a, &b), None, "seed {seed} frame {f}\n{a:?}\n{b:?}");
        }
        unsafe { c::ref_lpcnet_encoder_destroy(cst) };
    }
}
