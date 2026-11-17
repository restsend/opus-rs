//! libopus `op_pvq_search_sse2` (celt/x86/vq_sse2.c), operation for
//! operation. Its pulse search scores positions with `_mm_rsqrt_ps` and the
//! pre-search projects with `_mm_rcp_ps`, approximations whose results the
//! CPU defines, so they come from the same instructions here
//! (fearless_simd's SSE kernel); the rest is plain arithmetic in libopus's
//! order, four lanes at a time.
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use fearless_simd::{Sse2, f32x4, prelude::*};

use crate::compat::with_scratch;

fearless_simd::kernel!(
    /// `_mm_rcp_ps`.
    fn rcp(simd: Sse2, x: [f32; 4]) -> [f32; 4] {
        let r: f32x4<Sse2> = _mm_rcp_ps(f32x4::from_slice(simd, &x).into()).simd_into(simd);
        r.to_array()
    }
);

/// Lane 0 of libopus's horizontal sum (two shuffled adds).
fn hsum(v: [f32; 4]) -> f32 {
    (v[0] + v[2]) + (v[1] + v[3])
}

/// `_mm_cvttps_epi32`: truncation, 0x80000000 out of range.
fn cvtt(v: f32) -> i32 {
    if v >= -2147483648.0 && v < 2147483648.0 {
        v as i32
    } else {
        i32::MIN
    }
}

/// libopus `op_pvq_search_sse2`: the pulses `iy` (`x.len()` of them, `k`
/// in all) whose direction best matches `x`.
pub fn op_pvq_search_sse2(x: &[f32], iy: &mut [i32], k: i32) {
    let n = x.len();
    let simd = fearless_simd::Level::baseline().as_sse2().unwrap();
    // libopus pads X, y and signy by three for whole blocks of four.
    let padded = n + 3;
    with_scratch(padded, |xs: &mut [f32]| {
        with_scratch(padded, |y: &mut [f32]| {
            with_scratch(n.next_multiple_of(4), |iy4: &mut [i32]| {
                search(simd, x, xs, y, iy4, k);
                // Put the original sign back.
                for ((i, &v), &p) in iy.iter_mut().zip(x).zip(iy4.iter()) {
                    let s = -((v < 0.0) as i32);
                    *i = (p + s) ^ s;
                }
            })
        })
    });
}

fn search(simd: Sse2, x: &[f32], xs: &mut [f32], y: &mut [f32], iy: &mut [i32], k: i32) {
    let n = x.len();
    let blocks = n.div_ceil(4);
    let mut sums = [0.0f32; 4];
    for (j, &v) in x.iter().enumerate() {
        xs[j] = v.abs();
    }
    xs[n..].fill(0.0);
    for b in 0..blocks {
        for l in 0..4 {
            sums[l] += xs[4 * b + l];
        }
    }
    let (mut xy, mut yy) = (0.0f32, 0.0f32);
    let mut pulses_left = k;
    if k > (n >> 1) as i32 {
        let mut sum = hsum(sums);
        if !(sum > 1e-15 && sum < 64.0) {
            xs[0] = 1.0;
            xs[1..n].fill(0.0);
            sum = 1.0;
        }
        let rcp = ((k as f64 + 0.8) as f32) * rcp(simd, [sum; 4])[0];
        let (mut xy4, mut yy4, mut pulses) = ([0.0f32; 4], [0.0f32; 4], 0i32);
        for b in 0..blocks {
            for l in 0..4 {
                let j = 4 * b + l;
                let i = cvtt(xs[j] * rcp);
                pulses = pulses.wrapping_add(i);
                iy[j] = i;
                let y4 = i as f32;
                xy4[l] += xs[j] * y4;
                yy4[l] += y4 * y4;
                y[j] = y4 + y4;
            }
        }
        pulses_left -= pulses;
        xy = hsum(xy4);
        yy = hsum(yy4);
    } else {
        y.fill(0.0);
        iy.fill(0);
    }
    xs[n..n + 3].fill(-100.0);
    y[n..n + 3].fill(100.0);

    // This should never happen, but just in case it does (e.g. on silence)
    // we fill the first bin with pulses.
    if pulses_left > n as i32 + 3 {
        let tmp = pulses_left as f32;
        yy += tmp * tmp;
        yy += tmp * y[0];
        iy[0] += pulses_left;
        pulses_left = 0;
    }

    add_pulses(simd, xs, y, iy, xy, yy, pulses_left, blocks);
}

fearless_simd::kernel!(
    /// The pulse loop of `op_pvq_search_sse2`: each pulse goes where
    /// (X + xy) * rsqrt(y + yy) is largest, the largest position among the
    /// lanes that hold the maximum.
    #[allow(clippy::too_many_arguments)]
    fn add_pulses(
        simd: Sse2,
        xs: &[f32],
        y: &mut [f32],
        iy: &mut [i32],
        xy: f32,
        yy: f32,
        pulses: i32,
        blocks: usize,
    ) {
        let (mut xy, mut yy) = (xy, yy);
        let load = |v: &[f32]| -> __m128 { f32x4::from_slice(simd, &v[..4]).into() };
        let fours = _mm_set1_epi32(4);
        for _ in 0..pulses {
            // The squared magnitude term gets added anyway, so add it outside the loop.
            yy += 1.0;
            let xy4 = _mm_set1_ps(xy);
            let yy4 = _mm_set1_ps(yy);
            let mut max = _mm_setzero_ps();
            let mut pos = _mm_setzero_si128();
            let mut count = _mm_set_epi32(3, 2, 1, 0);
            for b in 0..blocks {
                let x4 = _mm_add_ps(load(&xs[4 * b..]), xy4);
                let y4 = _mm_rsqrt_ps(_mm_add_ps(load(&y[4 * b..]), yy4));
                let r4 = _mm_mul_ps(x4, y4);
                let gt = _mm_castps_si128(_mm_cmpgt_ps(r4, max));
                pos = _mm_max_epi16(pos, _mm_and_si128(count, gt));
                max = _mm_max_ps(max, r4);
                count = _mm_add_epi32(count, fours);
            }
            let max2 = _mm_max_ps(max, _mm_shuffle_ps::<0b01_00_11_10>(max, max));
            let max2 = _mm_max_ps(max2, _mm_shuffle_ps::<0b10_11_00_01>(max2, max2));
            pos = _mm_and_si128(pos, _mm_castps_si128(_mm_cmpeq_ps(max, max2)));
            pos = _mm_max_epi16(pos, _mm_unpackhi_epi64(pos, pos));
            pos = _mm_max_epi16(pos, _mm_shufflelo_epi16::<0b01_00_11_10>(pos));
            let best_id = _mm_cvtsi128_si32(pos) as usize;
            xy += xs[best_id];
            yy += y[best_id];
            y[best_id] += 2.0;
            iy[best_id] += 1;
        }
    }
);
