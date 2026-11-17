//! The pitch search and the pitch-doubling check of the CELT pre-filter
//! (and, for pitch_search, the decoder's concealment) against libopus.
use opusref::*;

const MAX_PERIOD: usize = 1024;
const MIN_PERIOD: usize = 15;
const N: usize = 960;
const LEN: usize = (MAX_PERIOD + N) >> 1;

/// Downsampled pre-filter inputs: voiced, noisy, loud and quiet stretches,
/// sparse.
fn signals() -> Vec<Vec<f32>> {
    let mut seed = 1u32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5
    };
    let mut out = Vec::new();
    for k in 0..400 {
        let f0 = 40.0 + (k % 37) as f32 * 9.0;
        let loud = 10f32.powi((k % 7) as i32);
        let split = 100 + (k * 13) % 800;
        let x: Vec<f32> = (0..LEN)
            .map(|i| {
                let t = i as f32 / 24000.0;
                let voice: f32 = (1..8)
                    .map(|h| (h as f32 * f0 * t * 6.2832).sin() / h as f32)
                    .sum();
                match k % 5 {
                    0 => voice * 3000.0 + rnd() * 30.0,
                    1 => rnd() * 1000.0,
                    // Loud history, then a quiet stretch.
                    2 => {
                        if i < split {
                            voice * loud * 1000.0
                        } else {
                            voice * 0.01 + rnd() * 1e-3
                        }
                    }
                    // Quiet, then loud: the running energies cancel.
                    3 if k % 10 == 3 => {
                        if i >= split {
                            voice * loud * 1000.0
                        } else {
                            voice * 0.01 + rnd() * 1e-3
                        }
                    }
                    3 => {
                        if i % (60 + k % 50) == 0 {
                            5000.0
                        } else {
                            0.0
                        }
                    }
                    // Quiet noise: correlations of order 1, where the
                    // value outside the candidate windows weighs.
                    4 if k % 10 == 9 => rnd() * 0.05,
                    _ => (voice + rnd()) * 1e-3,
                }
            })
            .collect();
        out.push(x);
    }
    out
}

/// Pitch candidates and doubling results that differ from libopus's, the
/// latter at the search's period or at `long` periods.
fn mismatches(long: bool) -> (usize, usize) {
    let (mut search, mut doubling) = (0, 0);
    for (k, x) in signals().iter().enumerate() {
        let a = opus_rs::pitch::pitch_search(
            &x[MAX_PERIOD >> 1..],
            x,
            N,
            MAX_PERIOD - 3 * MIN_PERIOD,
            opus_rs::Arch::Scalar,
        );
        let mut b = 0;
        let mut y = x.clone();
        unsafe {
            pitch_search(
                x[MAX_PERIOD >> 1..].as_ptr(),
                y.as_mut_ptr(),
                N as i32,
                (MAX_PERIOD - 3 * MIN_PERIOD) as i32,
                &mut b,
                0,
            )
        };
        search += (a != b as usize) as usize;
        // Long periods take their energies from the end of the running sum.
        let t0 = if long {
            962 + k % 60
        } else {
            (MAX_PERIOD - b as usize).min(MAX_PERIOD - 2)
        };
        {
            let (prev_period, prev_gain) = (MIN_PERIOD + k % 300, (k % 5) as f32 * 0.2);
            let mut ta = t0;
            let ga = opus_rs::pitch::remove_doubling(
                x,
                MAX_PERIOD,
                MIN_PERIOD,
                N,
                &mut ta,
                prev_period,
                prev_gain,
            );
            let mut tb = t0 as i32;
            let mut y = x.clone();
            let gb = unsafe {
                remove_doubling(
                    y.as_mut_ptr(),
                    MAX_PERIOD as i32,
                    MIN_PERIOD as i32,
                    N as i32,
                    &mut tb,
                    prev_period as i32,
                    prev_gain,
                    0,
                )
            };
            // A zero gain may differ in sign; the pre-filter treats both as off.
            doubling += ((ta, ga) != (tb as usize, gb)) as usize;
        }
    }
    (search, doubling)
}

/// compute_pitch_gain is xy/sqrt(1+xx*yy) for any sign (pitch.c:447-450);
/// the fine search leaves lags outside the candidate windows at 0
/// (pitch.c:376).
#[test]
fn pitch_search_and_doubling_match_libopus() {
    assert_eq!(mismatches(false), (0, 0));
}

/// remove_doubling clamps only the stored energies at 0 and keeps the
/// running sum (pitch.c:483-487), which cancellation can drive negative.
#[test]
fn doubling_keeps_the_running_energy() {
    assert_eq!(mismatches(true), (0, 0));
}

/// The concealment's pitch search on a decoded history whose best lag
/// lies at a candidate window's edge (lane D's input: concealment after
/// a 2.5 ms CELT packet, mono 24 kHz).
#[test]
fn concealment_pitch_search_matches_libopus() {
    let lp: Vec<f32> = include_bytes!("data/pitch_fine_xcorr_lp.f32")
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let a = opus_rs::pitch::pitch_search(&lp[360..], &lp, 1328, 620, opus_rs::Arch::Scalar);
    let mut b = 0;
    let mut y = lp.clone();
    unsafe { pitch_search(lp[360..].as_ptr(), y.as_mut_ptr(), 1328, 620, &mut b, 0) };
    assert_eq!(a, b as usize);
}
