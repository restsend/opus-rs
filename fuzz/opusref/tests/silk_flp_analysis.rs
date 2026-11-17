//! The float SILK encoder's LTP and LPC helpers (find_LTP_FLP.c,
//! corrMatrix_FLP.c, LTP_analysis_filter_FLP.c, residual_energy_FLP.c,
//! wrappers_FLP.c) against libopus's, bit for bit.
#![cfg(not(feature = "fixed"))]
use opus_rs::silk::float::lpc_analysis_flp::silk_residual_energy_flp;
use opus_rs::silk::float::ltp_analysis_flp::*;
use opus_rs::silk::float::wrappers_flp::*;
use opusref::*;
use std::os::raw::c_int;

unsafe extern "C" {
    fn silk_find_LTP_FLP(
        xx: *mut f32,
        x_x: *mut f32,
        r: *const f32,
        lag: *const c_int,
        subfr_length: c_int,
        nb_subfr: c_int,
        arch: c_int,
    );
    fn silk_LTP_analysis_filter_FLP(
        res: *mut f32,
        x: *const f32,
        b: *const f32,
        pitch_l: *const c_int,
        inv_gains: *const f32,
        subfr_length: c_int,
        nb_subfr: c_int,
        pre_length: c_int,
    );
    fn silk_residual_energy_FLP(
        nrgs: *mut f32,
        x: *const f32,
        a: *const [f32; 16],
        gains: *const f32,
        subfr_length: c_int,
        nb_subfr: c_int,
        order: c_int,
    );
    fn silk_A2NLSF_FLP(nlsf: *mut i16, ar: *const f32, order: c_int);
    fn silk_NLSF2A_FLP(ar: *mut f32, nlsf: *const i16, order: c_int, arch: c_int);
    fn silk_quant_LTP_gains_FLP(
        b: *mut f32,
        cbk_index: *mut i8,
        periodicity_index: *mut i8,
        sum_log_gain_q7: *mut i32,
        pred_gain_db: *mut f32,
        xx: *const f32,
        x_x: *const f32,
        subfr_len: c_int,
        nb_subfr: c_int,
        arch: c_int,
    );
}

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1 << 23) as f32 - 1.0
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn ltp_analysis_matches_libopus() {
    let _ = unsafe { silk_lin2log(1) };
    let mut r = Rng(6);
    for t in 0..10_000 {
        let fs_khz = [8, 12, 16][t % 3];
        let nb_subfr = [2, 4][(t / 3) % 2];
        let subfr_length = 5 * fs_khz;
        let mem = 20 * fs_khz;
        // A periodic residual in noise, with the history the lags reach and
        // the LTP_ORDER samples after the frame that find_LTP's energy reads.
        let period = 2 * fs_khz + r.next() as usize % (16 * fs_khz);
        let amp = [1.0, 300.0, 30000.0][r.next() as usize % 3];
        let res: Vec<f32> = (0..mem + nb_subfr * subfr_length + 5)
            .map(|i| ((if i % period == 0 { 1.0 } else { 0.0 } + 0.1 * r.unit()) * amp).round())
            .collect();
        let lag: [i32; 4] = core::array::from_fn(|_| {
            (period as i32 + (r.next() % 5) as i32 - 2).clamp(2 * fs_khz as i32, 18 * fs_khz as i32)
        });
        let (mut xx_r, mut xx_c) = ([0f32; 100], [0f32; 100]);
        let (mut x_x_r, mut x_x_c) = ([0f32; 20], [0f32; 20]);
        silk_find_ltp_flp(&mut xx_r, &mut x_x_r, &res, mem, &lag, subfr_length, nb_subfr);
        unsafe {
            silk_find_LTP_FLP(
                xx_c.as_mut_ptr(),
                x_x_c.as_mut_ptr(),
                res[mem..].as_ptr(),
                lag.as_ptr(),
                subfr_length as c_int,
                nb_subfr as c_int,
                0,
            )
        };
        assert_eq!((bits(&xx_r), bits(&x_x_r)), (bits(&xx_c), bits(&x_x_c)), "find_LTP case {t}");

        // Quantized LTP gains and their prediction gain.
        let (mut b_r, mut b_c) = ([0f32; 20], [0f32; 20]);
        let (mut ix_r, mut ix_c) = ([0i8; 4], [0i8; 4]);
        let (mut per_r, mut per_c) = (0i8, 0i8);
        let sum0 = (r.next() % 2000) as i32;
        let (mut sum_r, mut sum_c) = (sum0, sum0);
        let gain_r = silk_quant_ltp_gains_flp(
            &mut b_r,
            &mut ix_r,
            &mut per_r,
            &mut sum_r,
            &xx_r,
            &x_x_r,
            subfr_length as i32,
            nb_subfr,
        );
        let mut gain_c = 0f32;
        unsafe {
            silk_quant_LTP_gains_FLP(
                b_c.as_mut_ptr(),
                ix_c.as_mut_ptr(),
                &mut per_c,
                &mut sum_c,
                &mut gain_c,
                xx_c.as_ptr(),
                x_x_c.as_ptr(),
                subfr_length as c_int,
                nb_subfr as c_int,
                0,
            )
        };
        assert_eq!(
            (bits(&b_r), ix_r, per_r, sum_r, gain_r.to_bits()),
            (bits(&b_c), ix_c, per_c, sum_c, gain_c.to_bits()),
            "quant_LTP_gains case {t}"
        );

        // The LTP residual of the input, with the LPC order of preceding
        // samples (16 at 16 kHz).
        let order = if fs_khz == 16 { 16 } else { 10 };
        let inv_gains: [f32; 4] = core::array::from_fn(|_| 1.0 / (1.0 + 100.0 * r.unit().abs()));
        let n = nb_subfr * (subfr_length + order);
        let (mut ltp_r, mut ltp_c) = (vec![0f32; n], vec![0f32; n]);
        silk_ltp_analysis_filter_flp(&mut ltp_r, &res, mem - order, &b_r, &lag, &inv_gains, subfr_length, nb_subfr, order);
        unsafe {
            silk_LTP_analysis_filter_FLP(
                ltp_c.as_mut_ptr(),
                res[mem - order..].as_ptr(),
                b_c.as_ptr(),
                lag.as_ptr(),
                inv_gains.as_ptr(),
                subfr_length as c_int,
                nb_subfr as c_int,
                order as c_int,
            )
        };
        assert_eq!(bits(&ltp_r), bits(&ltp_c), "LTP analysis filter case {t}");

        // Its residual energies with an LPC filter per frame half.
        let mut a = [[0f32; 16]; 2];
        for h in &mut a {
            for v in &mut h[..order] {
                *v = 0.3 * r.unit();
            }
        }
        let gains: [f32; 4] = core::array::from_fn(|_| 1.0 + 1000.0 * r.unit().abs());
        let (mut nrg_r, mut nrg_c) = ([0f32; 4], [0f32; 4]);
        silk_residual_energy_flp(&mut nrg_r, &ltp_r, &a, &gains, subfr_length, nb_subfr, order);
        unsafe {
            silk_residual_energy_FLP(
                nrg_c.as_mut_ptr(),
                ltp_c.as_ptr(),
                a.as_ptr(),
                gains.as_ptr(),
                subfr_length as c_int,
                nb_subfr as c_int,
                order as c_int,
            )
        };
        assert_eq!(bits(&nrg_r), bits(&nrg_c), "residual energy case {t}");
    }
}

#[test]
fn nlsf_conversions_match_libopus() {
    let mut r = Rng(7);
    for t in 0..20_000 {
        let order = [10, 16][t % 2];
        // Bandwidth-expanded random filters, and NLSFs back to filters.
        let mut a = [0f32; 16];
        let mut c = 1.0f32;
        for v in &mut a[..order] {
            c *= 0.9;
            *v = r.unit() * c;
        }
        let (mut n_r, mut n_c) = ([0i16; 16], [0i16; 16]);
        silk_a2nlsf_flp(&mut n_r[..order], &a[..order]);
        unsafe { silk_A2NLSF_FLP(n_c.as_mut_ptr(), a.as_ptr(), order as c_int) };
        assert_eq!(n_r, n_c, "A2NLSF {a:?}");
        let (mut a_r, mut a_c) = ([0f32; 16], [0f32; 16]);
        silk_nlsf2a_flp(&mut a_r[..order], &n_r[..order]);
        unsafe { silk_NLSF2A_FLP(a_c.as_mut_ptr(), n_c.as_ptr(), order as c_int, 0) };
        assert_eq!(bits(&a_r), bits(&a_c), "NLSF2A {n_r:?}");
    }
}
