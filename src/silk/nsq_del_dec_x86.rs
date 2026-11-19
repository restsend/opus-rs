//! AVX2 short-term prediction and warped noise-shaping filter of the four
//! delayed-decision states (C `NSQ_del_dec_avx2.c`), through fearless_simd's
//! safe kernel API. State k sits in the low half of 64-bit lane k, so
//! `_mm256_mul_epi32` forms each silk_SMULWB product in one step.
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use fearless_simd::{Avx2, i32x4, prelude::*};

use super::define::NSQ_LPC_BUF_LENGTH;
use super::nsq_del_dec::Lanes;

fearless_simd::kernel!(
    /// Returns the short-term prediction (Q14, shifted as in the scalar
    /// quantizer) and the AR noise-shaping feedback of every state, and
    /// advances `s_ar2_q14` by one sample.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn short_prediction_and_shaping_avx2(
        simd: Avx2,
        s_lpc_q14: &[Lanes; NSQ_LPC_BUF_LENGTH],
        a_q12: &[i16],
        predict_order: usize,
        s_ar2_q14: &mut [Lanes],
        ar_shp_q13: &[i16],
        diff_q14: &Lanes,
        lf_ar_q14: &Lanes,
        warping_q16: i32,
        tilt_q14: i32,
    ) -> [Lanes; 2] {
        let load = |l: &Lanes| _mm256_cvtepi32_epi64(i32x4::from_slice(simd, l).into());
        let low_halves = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        let store = |v: __m256i, l: &mut Lanes| {
            let v = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(v, low_halves));
            let v: i32x4<Avx2> = v.simd_into(simd);
            v.store_slice(l);
        };
        let coef = |c: i32| _mm256_set1_epi64x(c as i16 as i64);
        // silk_SMLAWB(a, b, c) of every lane, with c from `coef`.
        let smlawb = |a: __m256i, b: __m256i, c: __m256i| {
            _mm256_add_epi32(a, _mm256_srli_epi64(_mm256_mul_epi32(b, c), 16))
        };
        let mut out = [[0i32; 4]; 2];

        let mut pred = _mm256_set1_epi64x(predict_order as i64 >> 1);
        let lpc = s_lpc_q14[NSQ_LPC_BUF_LENGTH - predict_order..].iter().rev();
        for (s, &a) in lpc.zip(&a_q12[..predict_order]) {
            pred = smlawb(pred, load(s), coef(a as i32));
        }
        store(_mm256_slli_epi32(pred, 4), &mut out[0]);

        let s_ar2 = s_ar2_q14;
        let ar_shp = ar_shp_q13;
        let order = s_ar2.len();
        let w = coef(warping_q16);
        let mut tmp2 = smlawb(load(diff_q14), load(&s_ar2[0]), w);
        let mut tmp1 = smlawb(load(&s_ar2[0]), _mm256_sub_epi32(load(&s_ar2[1]), tmp2), w);
        store(tmp2, &mut s_ar2[0]);
        let mut n_ar = smlawb(
            _mm256_set1_epi64x(order as i64 >> 1),
            tmp2,
            coef(ar_shp[0] as i32),
        );
        for j in (1..order / 2).map(|h| 2 * h) {
            tmp2 = smlawb(
                load(&s_ar2[j - 1]),
                _mm256_sub_epi32(load(&s_ar2[j]), tmp1),
                w,
            );
            store(tmp1, &mut s_ar2[j - 1]);
            n_ar = smlawb(n_ar, tmp1, coef(ar_shp[j - 1] as i32));
            tmp1 = smlawb(
                load(&s_ar2[j]),
                _mm256_sub_epi32(load(&s_ar2[j + 1]), tmp2),
                w,
            );
            store(tmp2, &mut s_ar2[j]);
            n_ar = smlawb(n_ar, tmp2, coef(ar_shp[j] as i32));
        }
        store(tmp1, &mut s_ar2[order - 1]);
        n_ar = smlawb(n_ar, tmp1, coef(ar_shp[order - 1] as i32));
        n_ar = _mm256_slli_epi32(n_ar, 1);
        n_ar = smlawb(n_ar, load(lf_ar_q14), coef(tilt_q14));
        store(_mm256_slli_epi32(n_ar, 2), &mut out[1]);
        out
    }
);
