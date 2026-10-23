use opus_rs::pvq::pvq_search;

/// An all-zero band ties every score. With K > 4 and a length that is not
/// a multiple of 8, the AVX2 search let a padding lane win the tie and put
/// every pulse past the end of `y`, leaving the band without its pulses.
#[test]
fn a_silent_band_gets_all_its_pulses_inside_the_band() {
    for n in [9usize, 12, 15, 20, 31] {
        let x = vec![0.0f32; n];
        // Guard words after the band show any write past it.
        let mut buf = vec![0i32; n + 16];
        let (y, guard) = buf.split_at_mut(n);
        pvq_search(&x, y, 5, n);
        let pulses: i32 = y.iter().map(|v| v.abs()).sum();
        assert_eq!(pulses, 5, "n = {n}: pulses in the band");
        assert!(
            guard.iter().all(|&v| v == 0),
            "n = {n}: wrote past the band"
        );
    }
}
