//! FARGAN and the deep PLC state machine of opus-rs against libopus: the
//! synthesized samples must have the same bits.
#![cfg(feature = "dnn")]
use opus_rs::dnn::fargan::{self, FarganState};
use opus_rs::dnn::lpcnet_enc::{self, LpcnetEncState, NB_FEATURES, NB_TOTAL_FEATURES};
use opus_rs::dnn::lpcnet_plc::{LpcnetPlcState, PlcModels};
use opus_rs::dnn::{builtin, freq::FRAME_SIZE};
use opusref::dnn::{Rng, same_bits, speechlike};
use opusref::dnn_plc as c;

/// The features of every frame of `sig` (int16-range floats).
fn features_of(sig: &[f32]) -> Vec<[f32; NB_TOTAL_FEATURES]> {
    let mut st = LpcnetEncState::new(builtin::pitchdnn().unwrap());
    sig.chunks_exact(FRAME_SIZE)
        .map(|x| {
            let mut f = [0f32; NB_TOTAL_FEATURES];
            lpcnet_enc::lpcnet_compute_single_frame_features_float(&mut st, x, &mut f);
            f
        })
        .collect()
}

#[test]
fn fargan_matches() {
    let sig = speechlike(16000 * 3, 11);
    let feats = features_of(&sig);
    let cst = unsafe { c::ref_fargan_create() };
    let mut rst = FarganState::new(builtin::fargan().unwrap());
    for start in [10usize, 120] {
        let pcm0: Vec<f32> = sig[start * FRAME_SIZE - 320..start * FRAME_SIZE]
            .iter()
            .map(|x| x / 32768.0)
            .collect();
        let f0: Vec<f32> = feats[start - 5..start]
            .iter()
            .flat_map(|f| f[..NB_FEATURES].to_vec())
            .collect();
        unsafe { c::ref_fargan_cont(cst, pcm0.as_ptr(), f0.as_ptr()) };
        fargan::fargan_cont(&mut rst, &pcm0, &f0);
        for k in 0..40 {
            let f = &feats[start + k];
            let mut a = [0f32; 160];
            let mut b = [0f32; 160];
            unsafe { c::ref_fargan_synthesize(cst, a.as_mut_ptr(), f.as_ptr()) };
            fargan::fargan_synthesize(&mut rst, &mut b, f);
            assert_eq!(same_bits(&a, &b), None, "start {start} frame {k}");
            let mut ia = [0i16; 160];
            let mut ib = [0i16; 160];
            unsafe { c::ref_fargan_synthesize_int(cst, ia.as_mut_ptr(), f.as_ptr()) };
            fargan::fargan_synthesize_int(&mut rst, &mut ib, f);
            assert_eq!(ia, ib, "start {start} frame {k} int");
        }
    }
    unsafe { c::ref_fargan_destroy(cst) };
}

#[test]
fn plc_sequences_match() {
    let sig = speechlike(16000 * 6, 12);
    let feats = features_of(&sig);
    let mut r = Rng(13);
    let cst = unsafe { c::ref_lpcnet_plc_create() };
    let mut rst = LpcnetPlcState::new(PlcModels::builtin().unwrap());
    let mut lost = 0;
    for (f, x) in sig.chunks_exact(FRAME_SIZE).enumerate() {
        let mut pcm: Vec<i16> = x
            .iter()
            .map(|&v| v.clamp(-32768.0, 32767.0) as i16)
            .collect();
        let roll = r.below(100);
        if f % 211 == 210 {
            unsafe { c::ref_lpcnet_plc_reset(cst) };
            rst.reset();
        }
        // DRED-like features for some losses.
        if roll < 8 {
            for k in 0..1 + r.below(4) as usize {
                if r.below(3) == 0 {
                    unsafe { c::ref_lpcnet_plc_fec_add(cst, core::ptr::null()) };
                    rst.fec_add(None);
                } else {
                    let ff = &feats[(f + k).min(feats.len() - 1)];
                    unsafe { c::ref_lpcnet_plc_fec_add(cst, ff.as_ptr()) };
                    rst.fec_add(Some(ff));
                }
            }
        } else if roll < 10 {
            unsafe { c::ref_lpcnet_plc_fec_clear(cst) };
            rst.fec_clear();
        }
        // Bursts of losses of 1 to 12 frames.
        if lost == 0 && roll < 25 {
            lost = 1 + r.below(12);
        }
        if lost > 0 {
            lost -= 1;
            let mut a = [0i16; FRAME_SIZE];
            let mut b = [0i16; FRAME_SIZE];
            unsafe { c::ref_lpcnet_plc_conceal(cst, a.as_mut_ptr()) };
            rst.conceal(&mut b);
            assert_eq!(a, b, "frame {f}");
        } else {
            unsafe { c::ref_lpcnet_plc_update(cst, pcm.as_mut_ptr()) };
            rst.update(&pcm);
        }
    }
    unsafe { c::ref_lpcnet_plc_destroy(cst) };
}
