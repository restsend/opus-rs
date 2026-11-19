//! NEON short-term prediction and warped noise-shaping filter of the four
//! delayed-decision states (C `NSQ_del_dec_neon_intr.c`), through
//! fearless_simd's safe kernel API. State k sits in lane k; `vqdmulhq_s32`
//! of a value and a coefficient shifted left by 15 is silk_SMULWB.
use core::arch::aarch64::*;

use fearless_simd::{Neon, i32x4, prelude::*};

use super::define::NSQ_LPC_BUF_LENGTH;
use super::nsq_del_dec::Lanes;

fearless_simd::kernel!(
    /// Returns the short-term prediction (Q14, shifted as in the scalar
    /// quantizer) and the AR noise-shaping feedback of every state, and
    /// advances `s_ar2_q14` by one sample.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn short_prediction_and_shaping_neon(
        simd: Neon,
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
        let load = |l: &Lanes| -> int32x4_t { i32x4::from_slice(simd, l).into() };
        let store = |v: int32x4_t, l: &mut Lanes| {
            let v: i32x4<Neon> = v.simd_into(simd);
            v.store_slice(l);
        };
        // The doubling high half of b * (c << 15) is (b * c) >> 16, exact:
        // |c << 15| < 2^31 keeps it from saturating.
        let coef = |c: i32| vdupq_n_s32((c as i16 as i32) << 15);
        // silk_SMLAWB(a, b, c) of every lane, with c from `coef`.
        let smlawb = |a: int32x4_t, b: int32x4_t, c: int32x4_t| vaddq_s32(a, vqdmulhq_s32(b, c));
        let mut out = [[0i32; 4]; 2];

        let mut pred = vdupq_n_s32(predict_order as i32 >> 1);
        let lpc = s_lpc_q14[NSQ_LPC_BUF_LENGTH - predict_order..].iter().rev();
        for (s, &a) in lpc.zip(&a_q12[..predict_order]) {
            pred = smlawb(pred, load(s), coef(a as i32));
        }
        store(vshlq_n_s32::<4>(pred), &mut out[0]);

        let s_ar2 = s_ar2_q14;
        let ar_shp = ar_shp_q13;
        let order = s_ar2.len();
        let w = coef(warping_q16);
        let mut tmp2 = smlawb(load(diff_q14), load(&s_ar2[0]), w);
        let mut tmp1 = smlawb(load(&s_ar2[0]), vsubq_s32(load(&s_ar2[1]), tmp2), w);
        store(tmp2, &mut s_ar2[0]);
        let mut n_ar = smlawb(vdupq_n_s32(order as i32 >> 1), tmp2, coef(ar_shp[0] as i32));
        for j in (1..order / 2).map(|h| 2 * h) {
            tmp2 = smlawb(load(&s_ar2[j - 1]), vsubq_s32(load(&s_ar2[j]), tmp1), w);
            store(tmp1, &mut s_ar2[j - 1]);
            n_ar = smlawb(n_ar, tmp1, coef(ar_shp[j - 1] as i32));
            tmp1 = smlawb(load(&s_ar2[j]), vsubq_s32(load(&s_ar2[j + 1]), tmp2), w);
            store(tmp2, &mut s_ar2[j]);
            n_ar = smlawb(n_ar, tmp2, coef(ar_shp[j] as i32));
        }
        store(tmp1, &mut s_ar2[order - 1]);
        n_ar = smlawb(n_ar, tmp1, coef(ar_shp[order - 1] as i32));
        n_ar = vshlq_n_s32::<1>(n_ar);
        n_ar = smlawb(n_ar, load(lf_ar_q14), coef(tilt_q14));
        store(vshlq_n_s32::<2>(n_ar), &mut out[1]);
        out
    }
);
