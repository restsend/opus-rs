//! SILK-only packets of opus-rs with `silk-fixed` against fixed-point libopus,
//! whose SILK encoder (silk/fixed/) that feature ports, over the grid of
//! common/silk_grid.rs. Runs with
//! `cargo test --release --features fixed --test silk_fixed`.
#![cfg(feature = "fixed")]

#[path = "common/silk_grid.rs"]
mod grid;

#[test]
fn silk_packets_match_fixed_point_libopus() {
    grid::silk_packets();
}

#[test]
fn stereo_to_mono_switches_match_fixed_point_libopus() {
    grid::stereo_to_mono_switches();
}

#[test]
fn prediction_disabled_matches_fixed_point_libopus() {
    grid::prediction_disabled();
}

mod sigproc {
    unsafe extern "C" {
        fn silk_schur(rc_q15: *mut i16, c: *const i32, order: i32) -> i32;
    }

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            self.0
        }
    }

    /// silk_schur against libopus's (silk/fixed/schur_FIX.c), also with an
    /// unstable first coefficient, a negative energy and a reflection
    /// buffer that holds old values.
    #[test]
    fn schur_matches_libopus() {
        let mut r = Rng(1);
        for t in 0..20_000 {
            let order = 1 + (r.next() % 16) as usize;
            let mut c = [0i32; 17];
            c[0] = (r.next() >> (r.next() % 31)) as i32;
            for v in &mut c[1..=order] {
                *v = (r.next() as i32) >> (1 + r.next() % 31);
            }
            match t % 4 {
                1 => c[0] = -c[0].max(1),
                2 => c[1 + (r.next() as usize % order)] = c[0],
                _ => {}
            }
            let (mut rc_r, mut rc_c) = ([0x1234i16; 16], [0x1234i16; 16]);
            let nrg_r = opus_rs::silk::silk_schur(&mut rc_r, &c, order);
            let nrg_c = unsafe { silk_schur(rc_c.as_mut_ptr(), c.as_ptr(), order as i32) };
            assert_eq!(
                (nrg_r, rc_r),
                (nrg_c, rc_c),
                "c {:?} order {order}",
                &c[..=order]
            );
        }
    }
}
