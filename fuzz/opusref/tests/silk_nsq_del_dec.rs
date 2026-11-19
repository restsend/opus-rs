//! silk_nsq_del_dec against libopus's silk_NSQ_del_dec (silk/NSQ_del_dec.c;
//! with the feature `simd` the kernel libopus picks at each x86 level,
//! silk/x86/NSQ_del_dec_avx2.c at arch 4), frame by frame on one carried
//! quantizer state: random frames, an unvoiced frame after a lag of 3, and
//! reconstructed samples past the 32-bit range of SMULWW.
use opus_rs::Arch;
use opus_rs::silk::{SideInfoIndices, SilkEncoderStateCommon, SilkNSQState, silk_nsq_del_dec};
use opusref::*;
use std::os::raw::c_int;

/// libopus's `silk_nsq_state`.
#[repr(C)]
#[derive(Clone, Copy)]
struct CNsq {
    xq: [i16; 640],
    s_ltp_shp_q14: [i32; 640],
    s_lpc_q14: [i32; 96],
    s_ar2_q14: [i32; 24],
    s_lf_ar_q14: i32,
    s_diff_shp_q14: i32,
    lag_prev: c_int,
    s_ltp_buf_idx: c_int,
    s_ltp_shp_buf_idx: c_int,
    rand_seed: i32,
    prev_gain_q16: i32,
    rewhite_flag: c_int,
}

unsafe extern "C" {
    fn ref_silk_nsq_state_size() -> usize;
    fn ref_silk_nsq_del_dec(
        arch: c_int,
        dims: *const c_int,
        ind: *const i8,
        nsq: *mut CNsq,
        x16: *const i16,
        pulses: *mut i8,
        pred_coef_q12: *const i16,
        ltp_coef_q14: *const i16,
        ar_q13: *const i16,
        harm_shape_gain_q14: *const c_int,
        tilt_q14: *const c_int,
        lf_shp_q14: *const i32,
        gains_q16: *const i32,
        pitch_l: *const c_int,
        lambda_q10: c_int,
        ltp_scale_q14: c_int,
    ) -> c_int;
}

/// One frame's inputs.
struct Frame {
    fs_khz: i32,
    nb_subfr: i32,
    lpc_order: i32,
    shaping_order: i32,
    warping_q16: i32,
    n_states: i32,
    signal_type: i8,
    quant_offset_type: i8,
    nlsf_interp_coef_q2: i8,
    seed: i8,
    x16: Vec<i16>,
    pred_coef_q12: [i16; 32],
    ltp_coef_q14: [i16; 20],
    ar_q13: [i16; 96],
    harm_shape_gain_q14: [i32; 4],
    tilt_q14: [i32; 4],
    lf_shp_q14: [i32; 4],
    gains_q16: [i32; 4],
    pitch_l: [i32; 4],
    lambda_q10: i32,
    ltp_scale_q14: i32,
}

fn to_c(s: &SilkNSQState) -> CNsq {
    CNsq {
        xq: s.xq,
        s_ltp_shp_q14: s.s_ltp_shp_q14,
        s_lpc_q14: s.s_lpc_q14,
        s_ar2_q14: s.s_ar2_q14,
        s_lf_ar_q14: s.s_lf_ar_q14,
        s_diff_shp_q14: s.s_diff_shp_q14,
        lag_prev: s.lag_prev,
        s_ltp_buf_idx: s.s_ltp_buf_idx,
        s_ltp_shp_buf_idx: s.s_ltp_shp_buf_idx,
        rand_seed: s.rand_seed,
        prev_gain_q16: s.prev_gain_q16,
        rewhite_flag: s.rewhite_flag,
    }
}

/// Runs `f` on both sides at `arch` (libopus's arch number `c`) and
/// compares the seed, the pulses and the whole quantizer state.
fn check(f: &Frame, nsq: &mut SilkNSQState, arch: Arch, c: c_int, what: &str) {
    let subfr_length = 5 * f.fs_khz;
    let frame_length = subfr_length * f.nb_subfr;
    let ltp_mem_length = 20 * f.fs_khz;
    let mut common = SilkEncoderStateCommon::default();
    common.arch = arch;
    common.fs_khz = f.fs_khz;
    common.nb_subfr = f.nb_subfr;
    common.frame_length = frame_length;
    common.subfr_length = subfr_length;
    common.ltp_mem_length = ltp_mem_length;
    common.predict_lpc_order = f.lpc_order;
    common.shaping_lpc_order = f.shaping_order;
    common.warping_q16 = f.warping_q16;
    common.n_states_delayed_decision = f.n_states;
    let mut indices = SideInfoIndices::default();
    indices.signal_type = f.signal_type;
    indices.quant_offset_type = f.quant_offset_type;
    indices.nlsf_interp_coef_q2 = f.nlsf_interp_coef_q2;
    indices.seed = f.seed;
    let mut c_nsq = to_c(nsq);
    let n = frame_length as usize;
    let (mut pr, mut pc) = (vec![0i8; n], vec![0i8; n]);
    let seed_r = silk_nsq_del_dec(
        &common,
        nsq,
        &indices,
        &f.x16,
        &mut pr,
        &f.pred_coef_q12,
        &f.ltp_coef_q14,
        &f.ar_q13,
        &f.harm_shape_gain_q14,
        &f.tilt_q14,
        &f.lf_shp_q14,
        &f.gains_q16,
        &f.pitch_l,
        f.lambda_q10,
        f.ltp_scale_q14,
    );
    let dims = [
        f.fs_khz,
        f.nb_subfr,
        frame_length,
        subfr_length,
        ltp_mem_length,
        f.lpc_order,
        f.shaping_order,
        f.warping_q16,
        f.n_states,
    ];
    let ind = [
        f.signal_type,
        f.quant_offset_type,
        f.nlsf_interp_coef_q2,
        f.seed,
    ];
    let seed_c = unsafe {
        ref_silk_nsq_del_dec(
            c,
            dims.as_ptr(),
            ind.as_ptr(),
            &mut c_nsq,
            f.x16.as_ptr(),
            pc.as_mut_ptr(),
            f.pred_coef_q12.as_ptr(),
            f.ltp_coef_q14.as_ptr(),
            f.ar_q13.as_ptr(),
            f.harm_shape_gain_q14.as_ptr(),
            f.tilt_q14.as_ptr(),
            f.lf_shp_q14.as_ptr(),
            f.gains_q16.as_ptr(),
            f.pitch_l.as_ptr(),
            f.lambda_q10,
            f.ltp_scale_q14,
        )
    };
    assert_eq!(seed_r, seed_c, "{what} {arch:?}: seed");
    assert_eq!(pr, pc, "{what} {arch:?}: pulses");
    let r = to_c(nsq);
    assert_eq!(r.xq, c_nsq.xq, "{what} {arch:?}: xq");
    assert_eq!(
        r.s_ltp_shp_q14, c_nsq.s_ltp_shp_q14,
        "{what} {arch:?}: sLTP_shp"
    );
    assert_eq!(r.s_lpc_q14, c_nsq.s_lpc_q14, "{what} {arch:?}: sLPC");
    assert_eq!(r.s_ar2_q14, c_nsq.s_ar2_q14, "{what} {arch:?}: sAR2");
    assert_eq!(
        [
            r.s_lf_ar_q14,
            r.s_diff_shp_q14,
            r.lag_prev,
            r.s_ltp_buf_idx,
            r.s_ltp_shp_buf_idx,
            r.rand_seed,
            r.prev_gain_q16,
            r.rewhite_flag
        ],
        [
            c_nsq.s_lf_ar_q14,
            c_nsq.s_diff_shp_q14,
            c_nsq.lag_prev,
            c_nsq.s_ltp_buf_idx,
            c_nsq.s_ltp_shp_buf_idx,
            c_nsq.rand_seed,
            c_nsq.prev_gain_q16,
            c_nsq.rewhite_flag
        ],
        "{what} {arch:?}: scalars"
    );
}

/// The levels to compare at, with libopus's arch numbers.
fn levels() -> Vec<(Arch, c_int)> {
    if cfg!(feature = "simd") {
        [(Arch::Sse2, 2), (Arch::Sse4_1, 3), (Arch::Avx2, 4)]
            .into_iter()
            .filter(|&(a, _)| a <= Arch::native())
            .collect()
    } else {
        vec![(Arch::Scalar, 0)]
    }
}

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + (self.next() % (hi - lo + 1) as u32) as i32
    }
}

fn random_frame(r: &mut Rng, fs_khz: i32) -> Frame {
    let nb_subfr = if r.next() % 2 == 0 { 4 } else { 2 };
    let signal_type = r.range(0, 2) as i8;
    let n = (5 * fs_khz * nb_subfr) as usize;
    let lpc_order = if fs_khz == 16 { 16 } else { 10 };
    let shaping_order = 2 * r.range(4, 12);
    let mut f = Frame {
        fs_khz,
        nb_subfr,
        lpc_order,
        shaping_order,
        warping_q16: if r.next() % 2 == 0 {
            0
        } else {
            r.range(0, 20000)
        },
        n_states: r.range(1, 4),
        signal_type,
        quant_offset_type: r.range(0, 1) as i8,
        nlsf_interp_coef_q2: r.range(0, 4) as i8,
        seed: r.range(0, 3) as i8,
        x16: (0..n).map(|_| r.range(-12000, 12000) as i16).collect(),
        pred_coef_q12: [0; 32],
        ltp_coef_q14: [0; 20],
        ar_q13: [0; 96],
        harm_shape_gain_q14: [0; 4],
        tilt_q14: [0; 4],
        lf_shp_q14: [0; 4],
        gains_q16: [0; 4],
        pitch_l: [0; 4],
        lambda_q10: r.range(300, 3000),
        ltp_scale_q14: [15565, 12288, 8192][r.range(0, 2) as usize],
    };
    for (k, a) in f.pred_coef_q12.iter_mut().enumerate() {
        if k % 16 < lpc_order as usize {
            *a = (r.range(-1500, 1500) >> (k % 16 / 3)) as i16;
        }
    }
    for a in &mut f.ltp_coef_q14 {
        *a = r.range(-2000, 6000) as i16;
    }
    for a in &mut f.ar_q13 {
        *a = r.range(-3000, 3000) as i16;
    }
    let lag = r.range(2 * fs_khz, 18 * fs_khz - 3);
    for k in 0..4 {
        f.harm_shape_gain_q14[k] = r.range(0, 8000);
        f.tilt_q14[k] = r.range(-6000, 2000);
        f.lf_shp_q14[k] = (r.range(0, 16000) << 16) | (r.range(-16000, 0) & 0xFFFF);
        f.gains_q16[k] = r.range(1 << 16, 1 << 23);
        if signal_type == 2 {
            f.pitch_l[k] = (lag + r.range(-4, 4)).max(2 * fs_khz);
        }
    }
    f
}

#[test]
fn random_frames_match_libopus() {
    let _ = unsafe { silk_lin2log(1) };
    assert_eq!(
        unsafe { ref_silk_nsq_state_size() },
        std::mem::size_of::<CNsq>()
    );
    for (arch, c) in levels() {
        let mut r = Rng(7);
        for fs_khz in [8, 12, 16] {
            let mut nsq = SilkNSQState {
                prev_gain_q16: 65536,
                lag_prev: 100,
                ..Default::default()
            };
            nsq.s_ltp_shp_buf_idx = 20 * fs_khz;
            nsq.s_ltp_buf_idx = 20 * fs_khz;
            for i in 0..200 {
                let f = random_frame(&mut r, fs_khz);
                check(&f, &mut nsq, arch, c, &format!("frame {i} at {fs_khz} kHz"));
            }
        }
    }
}

/// An unvoiced frame after a lag of 3: libopus limits the decision delay to
/// lag - LTP_ORDER/2 - 1 = 0.
#[test]
fn unvoiced_after_lag_3_matches_libopus() {
    let _ = unsafe { silk_lin2log(1) };
    for (arch, c) in levels() {
        let mut r = Rng(3);
        for i in 0..40 {
            let mut f = random_frame(&mut r, 16);
            f.signal_type = (i % 2) as i8;
            f.n_states = 2 + i % 3;
            let mut nsq = SilkNSQState {
                prev_gain_q16: 65536,
                lag_prev: 3,
                ..Default::default()
            };
            check(&f, &mut nsq, arch, c, &format!("lag 3, frame {i}"));
        }
    }
}

/// Reconstructed samples near 2^26 in Q14 with a gain of 2^21 in Q10: their
/// product passes 2^47, where SMULWW leaves the 32-bit range.
/// NSQ_del_dec_avx2.c rounds the 64-bit product (3 and 4 states).
#[test]
fn large_reconstruction_matches_libopus() {
    let _ = unsafe { silk_lin2log(1) };
    for (arch, c) in levels() {
        for n_states in 1..=4 {
            for &gain in &[(1 << 27) + (1 << 8), (1 << 27) + (1 << 23), 3 << 26] {
                let mut r = Rng(5);
                let mut f = random_frame(&mut r, 16);
                f.signal_type = 0;
                f.n_states = n_states;
                f.pred_coef_q12 = [0; 32];
                f.pred_coef_q12[0] = 4096;
                f.pred_coef_q12[16] = 4096;
                f.ar_q13 = [0; 96];
                f.harm_shape_gain_q14 = [0; 4];
                f.tilt_q14 = [0; 4];
                f.lf_shp_q14 = [0; 4];
                f.gains_q16 = [gain; 4];
                f.x16.fill(0);
                let mut nsq = SilkNSQState {
                    prev_gain_q16: gain,
                    lag_prev: 100,
                    ..Default::default()
                };
                nsq.s_lpc_q14 = [1 << 26; 96];
                check(
                    &f,
                    &mut nsq,
                    arch,
                    c,
                    &format!("{n_states} states, gain {gain}"),
                );
            }
        }
    }
}
