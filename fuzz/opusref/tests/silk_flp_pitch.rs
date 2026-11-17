//! The float SILK pitch estimator (pitch_analysis_core_FLP.c) against
//! libopus's on periodic and noisy frames at every SILK rate, frame length
//! and complexity: voicing, lags, indices and the normalized correlation,
//! bit for bit.
#![cfg(not(feature = "fixed"))]
use opus_rs::silk::float::pitch_analysis_core_flp::silk_pitch_analysis_core_flp;
use opusref::*;
use std::os::raw::c_int;

unsafe extern "C" {
    fn silk_pitch_analysis_core_FLP(
        frame: *const f32,
        pitch_out: *mut c_int,
        lag_index: *mut i16,
        contour_index: *mut i8,
        ltp_corr: *mut f32,
        prev_lag: c_int,
        search_thres1: f32,
        search_thres2: f32,
        fs_khz: c_int,
        complexity: c_int,
        nb_subfr: c_int,
        arch: c_int,
    ) -> c_int;
}

#[test]
fn pitch_analysis_core_matches_libopus() {
    let _ = unsafe { silk_lin2log(1) };
    let mut seed = 5u32;
    let mut rand = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        seed >> 8
    };
    let mut voiced = 0;
    for t in 0..12_000 {
        let fs_khz = [8, 12, 16][t % 3];
        let nb_subfr = [2, 4][(t / 3) % 2];
        let complexity = (t / 6) % 3;
        let len = (20 + nb_subfr * 5) * fs_khz;
        // A pulse train with a random period, gliding a little, in noise.
        let period = 2.0 * fs_khz as f32 + (rand() % (16 * fs_khz as u32)) as f32;
        let glide = (rand() % 100) as f32 / 1e4;
        let noise = [0.0, 0.05, 0.5, 2.0][rand() as usize % 4];
        let amp = [10.0, 1000.0, 20000.0][rand() as usize % 3];
        let mut phase = 0.0f32;
        let frame: Vec<f32> = (0..len)
            .map(|i| {
                phase += 1.0 / (period * (1.0 + glide * i as f32 / len as f32));
                let pulse = if phase.fract() < 1.0 / period { 1.0 } else { 0.0 };
                let n = (rand() as f32 / (1 << 23) as f32 - 1.0) * noise;
                ((pulse + n) * amp).round()
            })
            .collect();
        let prev_lag = if rand() % 2 == 0 { 0 } else { (period * 1.0) as i32 };
        let ltp_corr0 = (rand() % 1000) as f32 / 1000.0;
        let thres1 = 0.5 + (rand() % 400) as f32 / 1000.0;
        let thres2 = 0.2 + (rand() % 500) as f32 / 1000.0;

        let mut pitch_r = [0i32; 4];
        let mut corr_r = ltp_corr0;
        let est = silk_pitch_analysis_core_flp(
            &frame,
            &mut pitch_r,
            &mut corr_r,
            prev_lag,
            thres1,
            thres2,
            fs_khz,
            complexity,
            nb_subfr,
            opus_rs::Arch::Scalar,
        );
        let mut pitch_c = [0 as c_int; 4];
        let (mut lag_c, mut contour_c, mut corr_c) = (0i16, 0i8, ltp_corr0);
        let unvoiced_c = unsafe {
            silk_pitch_analysis_core_FLP(
                frame.as_ptr(),
                pitch_c.as_mut_ptr(),
                &mut lag_c,
                &mut contour_c,
                &mut corr_c,
                prev_lag,
                thres1,
                thres2,
                fs_khz as c_int,
                complexity as c_int,
                nb_subfr as c_int,
                0,
            )
        };
        let r = (
            est.as_ref().map(|e| (e.lag_index, e.contour_index)),
            pitch_r,
            corr_r.to_bits(),
        );
        let c = (
            (unvoiced_c == 0).then_some((lag_c, contour_c)),
            pitch_c,
            corr_c.to_bits(),
        );
        assert_eq!(r, c, "case {t}: {fs_khz} kHz {nb_subfr} subframes complexity {complexity}");
        voiced += est.is_some() as usize;
    }
    // Both outcomes are covered.
    assert!((1000..11_000).contains(&voiced), "{voiced} voiced");
}
