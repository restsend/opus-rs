//! WebAssembly simd128 short-term prediction and warped noise-shaping
//! filter of the four delayed-decision states, through fearless_simd's safe
//! kernel API. State k sits in lane k of an i32x4; silk_SMULWB is formed
//! from 32-bit products.
use core::arch::wasm32::*;

use fearless_simd::{WasmSimd128, i32x4, prelude::*};

use super::define::NSQ_LPC_BUF_LENGTH;
use super::nsq_del_dec::Lanes;

fearless_simd::kernel!(
    /// Returns the short-term prediction (Q14, shifted as in the scalar
    /// quantizer) and the AR noise-shaping feedback of every state, and
    /// advances `s_ar2_q14` by one sample.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn short_prediction_and_shaping_wasm(
        simd: WasmSimd128,
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
        let load = |l: &Lanes| -> v128 { i32x4::from_slice(simd, l).into() };
        let store = |v: v128, l: &mut Lanes| {
            let v: i32x4<WasmSimd128> = v.simd_into(simd);
            v.store_slice(l);
        };
        let coef = |c: i32| i32x4_splat(c as i16 as i32);
        // silk_SMLAWB(a, b, c) of every lane, with c from `coef`, in 32-bit
        // halves (silk_SMULWB without a 64-bit product): (b >> 16) * c plus
        // ((b & 0xFFFF) * c) >> 16 is (b * c) >> 16, and neither product
        // overflows.
        let smlawb = |a: v128, b: v128, c: v128| {
            let high = i32x4_mul(i32x4_shr(b, 16), c);
            let low = i32x4_shr(i32x4_mul(v128_and(b, i32x4_splat(0xFFFF)), c), 16);
            i32x4_add(a, i32x4_add(high, low))
        };
        let mut out = [[0i32; 4]; 2];

        let mut pred = i32x4_splat(predict_order as i32 >> 1);
        let lpc = s_lpc_q14[NSQ_LPC_BUF_LENGTH - predict_order..].iter().rev();
        for (s, &a) in lpc.zip(&a_q12[..predict_order]) {
            pred = smlawb(pred, load(s), coef(a as i32));
        }
        store(i32x4_shl(pred, 4), &mut out[0]);

        let s_ar2 = s_ar2_q14;
        let ar_shp = ar_shp_q13;
        let order = s_ar2.len();
        let w = coef(warping_q16);
        let mut tmp2 = smlawb(load(diff_q14), load(&s_ar2[0]), w);
        let mut tmp1 = smlawb(load(&s_ar2[0]), i32x4_sub(load(&s_ar2[1]), tmp2), w);
        store(tmp2, &mut s_ar2[0]);
        let mut n_ar = smlawb(i32x4_splat(order as i32 >> 1), tmp2, coef(ar_shp[0] as i32));
        for j in (1..order / 2).map(|h| 2 * h) {
            tmp2 = smlawb(load(&s_ar2[j - 1]), i32x4_sub(load(&s_ar2[j]), tmp1), w);
            store(tmp1, &mut s_ar2[j - 1]);
            n_ar = smlawb(n_ar, tmp1, coef(ar_shp[j - 1] as i32));
            tmp1 = smlawb(load(&s_ar2[j]), i32x4_sub(load(&s_ar2[j + 1]), tmp2), w);
            store(tmp2, &mut s_ar2[j]);
            n_ar = smlawb(n_ar, tmp2, coef(ar_shp[j] as i32));
        }
        store(tmp1, &mut s_ar2[order - 1]);
        n_ar = smlawb(n_ar, tmp1, coef(ar_shp[order - 1] as i32));
        n_ar = i32x4_shl(n_ar, 1);
        n_ar = smlawb(n_ar, load(lf_ar_q14), coef(tilt_q14));
        store(i32x4_shl(n_ar, 2), &mut out[1]);
        out
    }
);
