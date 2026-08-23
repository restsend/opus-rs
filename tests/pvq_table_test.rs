//! PVQ codebook-size verification.
//!
//! Regression coverage for the issue #11 side-findings:
//!   1. `celt_pvq_u`/`celt_pvq_v` panicked with an out-of-bounds index for
//!      `k >= 129` (the fixed-size `compute_u` buffer).
//!   2. `celt_pvq_u`/`celt_pvq_v` could silently wrap around u32 instead of
//!      saturating, yielding plausible-but-wrong codebook sizes.
//!
//! Every `(n, k)` in the supported domain is checked against an exact u128
//! recurrence; out-of-domain results must saturate to `u32::MAX`.

use opus_rs::pvq::{MAX_PVQ_K, celt_pvq_u, celt_pvq_v, ncwrs};

/// Exact U(N,K) via the recurrence, computed in u128.
fn build_u_table(n_max: usize, k_max: usize) -> Vec<Vec<u128>> {
    // U(0,0)=1, U(0,K>0)=0, U(N>0,0)=0
    let mut u = vec![vec![0u128; k_max + 2]; n_max + 1];
    u[0][0] = 1;
    for n in 1..=n_max {
        u[n][0] = 0;
        for k in 1..=k_max + 1 {
            if n == 1 {
                u[1][k] = 1;
            } else {
                u[n][k] = u[n - 1][k]
                    .saturating_add(u[n][k - 1])
                    .saturating_add(u[n - 1][k - 1]);
            }
        }
    }
    u
}

#[test]
fn pvq_u_v_match_exact_recurrence() {
    let n_max = 176;
    let k_max = MAX_PVQ_K; // 128
    let u = build_u_table(n_max, k_max);

    for n in 0..=n_max {
        for k in 0..=k_max {
            let u_exact = u[n][k];
            let expected_u = u_exact.min(u32::MAX as u128) as u32;
            let got_u = celt_pvq_u(n as u32, k as u32);
            assert_eq!(got_u, expected_u, "U({n},{k})");

            let v_exact = u[n][k].saturating_add(u[n][k + 1]);
            let expected_v = v_exact.min(u32::MAX as u128) as u32;
            let got_v = celt_pvq_v(n as u32, k as u32);
            assert_eq!(got_v, expected_v, "V({n},{k})");
        }
    }
}

#[test]
fn pvq_ncwrs_matches_v() {
    let u = build_u_table(64, MAX_PVQ_K);
    for n in 2..=64 {
        for k in 0..=MAX_PVQ_K {
            let v_exact = u[n][k].saturating_add(u[n][k + 1]);
            let expected = v_exact.min(u32::MAX as u128) as u32;
            assert_eq!(ncwrs(n as u32, k as u32), expected, "ncwrs({n},{k})");
        }
    }
}

#[test]
fn pvq_no_panic_out_of_domain() {
    // These used to panic (index out of bounds in compute_u's fixed buffer).
    assert_eq!(celt_pvq_u(8, 129), u32::MAX);
    assert_eq!(celt_pvq_u(16, 200), u32::MAX);
    assert_eq!(celt_pvq_u(32, 129), u32::MAX);
    assert_eq!(celt_pvq_v(8, 128), u32::MAX);
    assert_eq!(celt_pvq_v(8, 129), u32::MAX);
    assert_eq!(celt_pvq_v(64, 16), u32::MAX);
    assert_eq!(celt_pvq_v(32, 64), u32::MAX);
}

#[test]
fn pvq_small_values_exact() {
    // Spot-check against known table values from libopus cwrs.c.
    assert_eq!(celt_pvq_u(2, 1), 1);
    assert_eq!(celt_pvq_u(3, 3), 13);
    assert_eq!(celt_pvq_u(4, 3), 25);
    assert_eq!(celt_pvq_u(5, 4), 129);
    assert_eq!(celt_pvq_v(2, 1), 4);
    assert_eq!(celt_pvq_v(3, 3), 38);
    assert_eq!(celt_pvq_v(4, 3), 88);
    assert_eq!(celt_pvq_v(5, 4), 450);
}
