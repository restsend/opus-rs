//! The float SILK encoder's signal processing functions (silk/float/,
//! SigProc_FLP.h) against libopus's, bit for bit, on random inputs of the
//! lengths and orders the encoder uses.
#![cfg(not(feature = "fixed"))]
use opus_rs::silk::float::sigproc_flp::*;
use opusref::*;
use std::os::raw::c_int;

unsafe extern "C" {
    fn silk_energy_FLP(data: *const f32, n: c_int) -> f64;
    fn silk_inner_product_FLP_c(a: *const f32, b: *const f32, n: c_int) -> f64;
    fn silk_autocorrelation_FLP(r: *mut f32, x: *const f32, n: c_int, count: c_int, arch: c_int);
    fn silk_bwexpander_FLP(ar: *mut f32, d: c_int, chirp: f32);
    fn silk_k2a_FLP(a: *mut f32, rc: *const f32, order: i32);
    fn silk_schur_FLP(rc: *mut f32, ac: *const f32, order: c_int) -> f32;
    fn silk_LPC_inverse_pred_gain_FLP(a: *const f32, order: i32) -> f32;
    fn silk_LPC_analysis_filter_FLP(r: *mut f32, a: *const f32, s: *const f32, n: c_int, order: c_int);
    fn silk_burg_modified_FLP(
        a: *mut f32,
        x: *const f32,
        min_inv_gain: f32,
        subfr_length: c_int,
        nb_subfr: c_int,
        d: c_int,
        arch: c_int,
    ) -> f32;
    fn silk_scale_copy_vector_FLP(out: *mut f32, x: *const f32, gain: f32, n: c_int);
    fn silk_scale_vector_FLP(x: *mut f32, gain: f32, n: c_int);
    fn silk_insertion_sort_decreasing_FLP(a: *mut f32, idx: *mut c_int, l: c_int, k: c_int);
    fn silk_apply_sine_window_FLP(out: *mut f32, x: *const f32, win_type: c_int, n: c_int);
    fn silk_warped_autocorrelation_FLP(c: *mut f32, x: *const f32, warping: f32, n: c_int, order: c_int);
}

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0 >> 8
    }
    /// A uniform value in [-1, 1).
    fn unit(&mut self) -> f32 {
        self.next() as f32 / (1 << 23) as f32 - 1.0
    }
    /// A signal of `n` samples, 16-bit scale, smoothed by a random amount so
    /// that it is predictable, with occasional silence.
    fn signal(&mut self, n: usize) -> Vec<f32> {
        let amp = [0.0, 1.0, 300.0, 30000.0][self.next() as usize % 4];
        let pole = self.unit().abs() * 0.99;
        let mut y = 0.0f32;
        (0..n)
            .map(|_| {
                y = pole * y + self.unit() * amp;
                y.round()
            })
            .collect()
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn energy_and_inner_products_match_libopus() {
    let _ = unsafe { silk_lin2log(1) };
    let mut r = Rng(1);
    for _ in 0..20_000 {
        let n = r.next() as usize % 400;
        let a = r.signal(n);
        let b = r.signal(n);
        let c = unsafe { silk_energy_FLP(a.as_ptr(), n as c_int) };
        assert_eq!(silk_energy_flp(&a).to_bits(), c.to_bits(), "energy n {n}");
        let c = unsafe { silk_inner_product_FLP_c(a.as_ptr(), b.as_ptr(), n as c_int) };
        assert_eq!(silk_inner_product_flp(&a, &b).to_bits(), c.to_bits(), "inner product n {n}");
        let count = 1 + r.next() as usize % 25;
        let (mut rs, mut cs) = (vec![0f32; count], vec![0f32; count]);
        silk_autocorrelation_flp(&mut rs[..count.min(n)], &a);
        unsafe { silk_autocorrelation_FLP(cs.as_mut_ptr(), a.as_ptr(), n as c_int, count as c_int, 0) };
        assert_eq!(bits(&rs), bits(&cs), "autocorrelation n {n} count {count}");
    }
}

#[test]
fn lpc_functions_match_libopus() {
    let mut r = Rng(2);
    for _ in 0..20_000 {
        let order = [6, 8, 10, 12, 16][r.next() as usize % 5];
        // Autocorrelation of a signal, through Schur and k2a.
        let x = r.signal(240);
        let mut ac = [0f32; 17];
        silk_autocorrelation_flp(&mut ac[..order + 1], &x);
        ac[0] += ac[0] * 1e-3 + 1.0;
        let (mut rc_r, mut rc_c) = ([0f32; 16], [0f32; 16]);
        let nrg_r = silk_schur_flp(&mut rc_r, &ac, order);
        let nrg_c = unsafe { silk_schur_FLP(rc_c.as_mut_ptr(), ac.as_ptr(), order as c_int) };
        assert_eq!((nrg_r.to_bits(), bits(&rc_r)), (nrg_c.to_bits(), bits(&rc_c)), "schur");
        let (mut a_r, mut a_c) = ([0f32; 16], [0f32; 16]);
        silk_k2a_flp(&mut a_r[..order], &rc_r[..order]);
        unsafe { silk_k2a_FLP(a_c.as_mut_ptr(), rc_c.as_ptr(), order as i32) };
        assert_eq!(bits(&a_r), bits(&a_c), "k2a");
        let chirp = 0.9 + 0.1 * r.unit().abs();
        silk_bwexpander_flp(&mut a_r[..order], chirp);
        unsafe { silk_bwexpander_FLP(a_c.as_mut_ptr(), order as c_int, chirp) };
        assert_eq!(bits(&a_r), bits(&a_c), "bwexpander");
        // Random coefficients too, unstable ones included.
        let mut a = a_r;
        if r.next() % 2 == 0 {
            for v in &mut a[..order] {
                *v = r.unit() * 2.0;
            }
        }
        let g_c = unsafe { silk_LPC_inverse_pred_gain_FLP(a.as_ptr(), order as i32) };
        assert_eq!(silk_lpc_inverse_pred_gain_flp(&a[..order]).to_bits(), g_c.to_bits(), "inverse prediction gain");
        let (mut res_r, mut res_c) = (vec![1f32; 240], vec![1f32; 240]);
        silk_lpc_analysis_filter_flp(&mut res_r, &a[..order], &x, 240);
        unsafe { silk_LPC_analysis_filter_FLP(res_c.as_mut_ptr(), a.as_ptr(), x.as_ptr(), 240, order as c_int) };
        assert_eq!(bits(&res_r), bits(&res_c), "analysis filter");
    }
}

#[test]
fn burg_matches_libopus() {
    let mut r = Rng(3);
    for t in 0..20_000 {
        let d = [10, 16][t % 2];
        let nb_subfr = [2, 4][(t / 2) % 2];
        let subfr_length = d + [40, 60, 80][r.next() as usize % 3];
        let x = r.signal(nb_subfr * subfr_length);
        // From no limit to the limit after a reset.
        let min_inv_gain = [1e-6, 1e-4, 4e-4, 1e-2, 0.2][r.next() as usize % 5];
        let (mut a_r, mut a_c) = ([0f32; 16], [0f32; 16]);
        let n_r = silk_burg_modified_flp(&mut a_r, &x, min_inv_gain, subfr_length, nb_subfr, d);
        let n_c = unsafe {
            silk_burg_modified_FLP(
                a_c.as_mut_ptr(),
                x.as_ptr(),
                min_inv_gain,
                subfr_length as c_int,
                nb_subfr as c_int,
                d as c_int,
                0,
            )
        };
        assert_eq!((n_r.to_bits(), bits(&a_r)), (n_c.to_bits(), bits(&a_c)), "burg d {d} {nb_subfr}x{subfr_length}");
    }
}

#[test]
fn vector_window_and_sort_functions_match_libopus() {
    let mut r = Rng(4);
    for _ in 0..20_000 {
        let n = 4 * (1 + r.next() as usize % 60);
        let x = r.signal(n);
        let gain = r.unit() * 3.0;
        let (mut y_r, mut y_c) = (vec![0f32; n], vec![0f32; n]);
        silk_scale_copy_vector_flp(&mut y_r, &x, gain);
        unsafe { silk_scale_copy_vector_FLP(y_c.as_mut_ptr(), x.as_ptr(), gain, n as c_int) };
        assert_eq!(bits(&y_r), bits(&y_c), "scale copy");
        silk_scale_vector_flp(&mut y_r, gain);
        unsafe { silk_scale_vector_FLP(y_c.as_mut_ptr(), gain, n as c_int) };
        assert_eq!(bits(&y_r), bits(&y_c), "scale");
        for win_type in [1, 2] {
            silk_apply_sine_window_flp(&mut y_r, &x, win_type, n);
            unsafe { silk_apply_sine_window_FLP(y_c.as_mut_ptr(), x.as_ptr(), win_type, n as c_int) };
            assert_eq!(bits(&y_r), bits(&y_c), "sine window {win_type} n {n}");
        }
        let order = 2 * (1 + r.next() as usize % 12);
        let warping = r.unit().abs() * 0.3;
        let (mut c_r, mut c_c) = ([0f32; 25], [0f32; 25]);
        silk_warped_autocorrelation_flp(&mut c_r, &x, warping, order);
        unsafe { silk_warped_autocorrelation_FLP(c_c.as_mut_ptr(), x.as_ptr(), warping, n as c_int, order as c_int) };
        assert_eq!(bits(&c_r), bits(&c_c), "warped autocorrelation order {order}");
        // The sort, with ties.
        let l = 1 + r.next() as usize % 70;
        let k = 1 + r.next() as usize % l;
        let a: Vec<f32> = (0..l).map(|_| (r.next() % 16) as f32).collect();
        let (mut a_r, mut a_c) = (a.clone(), a.clone());
        let mut i_r = vec![0usize; k];
        let mut i_c = vec![0 as c_int; k];
        silk_insertion_sort_decreasing_flp(&mut a_r, &mut i_r);
        unsafe { silk_insertion_sort_decreasing_FLP(a_c.as_mut_ptr(), i_c.as_mut_ptr(), l as c_int, k as c_int) };
        let i_c: Vec<usize> = i_c.iter().map(|&i| i as usize).collect();
        assert_eq!((bits(&a_r[..k]), i_r), (bits(&a_c[..k]), i_c), "sort {l} {k}");
    }
}
