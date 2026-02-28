use crate::silk::decoder_structs::{SilkDecoderControl, SilkDecoderState};
use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::tables::SILK_QUANTIZATION_OFFSETS_Q10;

/// Core decoder. Performs inverse NSQ operation LTP + LPC
pub fn silk_decode_core(
    ps_dec: &mut SilkDecoderState,
    ps_dec_ctrl: &SilkDecoderControl,
    xq: &mut [i16],
    pulses: &[i16],
) {
    let nlsf_interpolation_flag = if ps_dec.indices.nlsf_interp_coef_q2 < 4 {
        1
    } else {
        0
    };

    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10[(ps_dec.indices.signal_type >> 1) as usize]
        [ps_dec.indices.quant_offset_type as usize] as i32;

    /* Decode excitation */
    let mut rand_seed = ps_dec.indices.seed as i32;
    for i in 0..ps_dec.frame_length as usize {
        rand_seed = silk_rand(rand_seed);
        ps_dec.exc_q14[i] = (pulses[i] as i32) << 14;
        if ps_dec.exc_q14[i] > 0 {
            ps_dec.exc_q14[i] -= QUANT_LEVEL_ADJUST_Q10 << 4;
        } else if ps_dec.exc_q14[i] < 0 {
            ps_dec.exc_q14[i] += QUANT_LEVEL_ADJUST_Q10 << 4;
        }
        ps_dec.exc_q14[i] += offset_q10 << 4;
        if rand_seed < 0 {
            ps_dec.exc_q14[i] = -ps_dec.exc_q14[i];
        }
        rand_seed = silk_add32_ovflw(rand_seed, pulses[i] as i32);
    }

    /* Copy LPC state */
    let mut s_lpc_q14: [i32; MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER] =
        [0; MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER];
    s_lpc_q14[..MAX_LPC_ORDER].copy_from_slice(&ps_dec.s_lpc_q14_buf);

    let mut pexc_q14_idx: usize = 0;
    let mut pxq_idx: usize = 0;
    let mut s_ltp_buf_idx = ps_dec.ltp_mem_length;

    /* LTP state buffer */
    let mut s_ltp_q15: Vec<i32> =
        vec![0i32; ps_dec.ltp_mem_length as usize + ps_dec.frame_length as usize];
    let mut s_ltp: Vec<i16> = vec![0i16; ps_dec.ltp_mem_length as usize];

    /* Loop over subframes */
    for k in 0..ps_dec.nb_subfr as usize {
        let a_q12 = &ps_dec_ctrl.pred_coef_q12[k >> 1];
        let b_q14 = &ps_dec_ctrl.ltp_coef_q14[k * LTP_ORDER..];
        let signal_type = ps_dec.indices.signal_type;

        let mut inv_gain_q31 = silk_inverse32_varq(ps_dec_ctrl.gains_q16[k], 47);

        /* Calculate gain adjustment factor */
        let gain_adj_q16 = if ps_dec_ctrl.gains_q16[k] != ps_dec.prev_gain_q16 {
            let adj = silk_div32_varq(ps_dec.prev_gain_q16, ps_dec_ctrl.gains_q16[k], 16);
            /* Scale short term state */
            for i in 0..MAX_LPC_ORDER {
                s_lpc_q14[i] = silk_smulww(adj, s_lpc_q14[i]);
            }
            adj
        } else {
            1 << 16
        };

        /* Save inv_gain */
        ps_dec.prev_gain_q16 = ps_dec_ctrl.gains_q16[k];

        /* Avoid abrupt transition from voiced PLC to unvoiced normal decoding */
        let (eff_signal_type, eff_pitch_l) = if ps_dec.loss_cnt > 0
            && ps_dec.prev_signal_type == TYPE_VOICED
            && ps_dec.indices.signal_type as i32 != TYPE_VOICED
            && k < MAX_NB_SUBFR / 2
        {
            // Use modified LTP coefficients
            let mut modified_b = [0i16; LTP_ORDER];
            modified_b[LTP_ORDER / 2] = 8192; // 0.25 in Q14
            (TYPE_VOICED, ps_dec.lag_prev)
        } else {
            (signal_type as i32, ps_dec_ctrl.pitch_l[k])
        };

        let mut lag = 0;
        if eff_signal_type == TYPE_VOICED {
            lag = eff_pitch_l;

            /* Re-whitening */
            if k == 0 || (k == 2 && nlsf_interpolation_flag != 0) {
                /* Rewhiten with new A coefs */
                let start_idx =
                    ps_dec.ltp_mem_length - lag - ps_dec.lpc_order - (LTP_ORDER / 2) as i32;
                debug_assert!(start_idx > 0);

                if k == 2 {
                    let copy_start = ps_dec.ltp_mem_length as usize;
                    let copy_len = 2 * ps_dec.subfr_length as usize;
                    ps_dec.out_buf[copy_start..copy_start + copy_len]
                        .copy_from_slice(&xq[0..copy_len]);
                }

                /* C: silk_LPC_analysis_filter(&sLTP[start_idx], &psDec->outBuf[start_idx + k * subfr_length], ...) */
                let filter_input_offset = start_idx as usize + k * ps_dec.subfr_length as usize;
                let filter_len = (ps_dec.ltp_mem_length - start_idx) as usize;
                silk_lpc_analysis_filter_offset(
                    &mut s_ltp,
                    start_idx as usize,
                    &ps_dec.out_buf,
                    filter_input_offset,
                    a_q12,
                    filter_len,
                    ps_dec.lpc_order as usize,
                );

                /* After rewhitening the LTP state is unscaled */
                if k == 0 {
                    /* Do LTP downscaling to reduce inter-packet dependency */
                    inv_gain_q31 =
                        silk_lshift(silk_smulwb(inv_gain_q31, ps_dec_ctrl.ltp_scale_q14), 2);
                }
                for i in 0..(lag + LTP_ORDER as i32 / 2) as usize {
                    s_ltp_q15[s_ltp_buf_idx as usize - i - 1] = silk_smulwb(
                        inv_gain_q31,
                        s_ltp[ps_dec.ltp_mem_length as usize - i - 1] as i32,
                    );
                }
            } else {
                /* Update LTP state when Gain changes */
                if gain_adj_q16 != (1 << 16) {
                    for i in 0..(lag + LTP_ORDER as i32 / 2) as usize {
                        s_ltp_q15[s_ltp_buf_idx as usize - i - 1] =
                            silk_smulww(gain_adj_q16, s_ltp_q15[s_ltp_buf_idx as usize - i - 1]);
                    }
                }
            }
        }

        /* Res buffer for this subframe */
        let mut res_q14: [i32; MAX_SUB_FRAME_LENGTH] = [0; MAX_SUB_FRAME_LENGTH];

        /* Long-term prediction */
        if eff_signal_type == TYPE_VOICED {
            /* Set up pointer */
            let pred_lag_ptr_start = (s_ltp_buf_idx - lag + LTP_ORDER as i32 / 2) as usize;
            for i in 0..ps_dec.subfr_length as usize {
                /* Unrolled loop - avoids introducing a bias */
                let mut ltp_pred_q13: i32 = 2;
                ltp_pred_q13 = silk_smlawb(
                    ltp_pred_q13,
                    s_ltp_q15[pred_lag_ptr_start + i],
                    b_q14[0] as i32,
                );
                ltp_pred_q13 = silk_smlawb(
                    ltp_pred_q13,
                    s_ltp_q15[pred_lag_ptr_start + i - 1],
                    b_q14[1] as i32,
                );
                ltp_pred_q13 = silk_smlawb(
                    ltp_pred_q13,
                    s_ltp_q15[pred_lag_ptr_start + i - 2],
                    b_q14[2] as i32,
                );
                ltp_pred_q13 = silk_smlawb(
                    ltp_pred_q13,
                    s_ltp_q15[pred_lag_ptr_start + i - 3],
                    b_q14[3] as i32,
                );
                ltp_pred_q13 = silk_smlawb(
                    ltp_pred_q13,
                    s_ltp_q15[pred_lag_ptr_start + i - 4],
                    b_q14[4] as i32,
                );

                /* Generate LPC excitation */
                res_q14[i] = silk_add_lshift32(ps_dec.exc_q14[pexc_q14_idx + i], ltp_pred_q13, 1);

                /* Update states */
                s_ltp_q15[s_ltp_buf_idx as usize] = res_q14[i] << 1;
                s_ltp_buf_idx += 1;
            }
        } else {
            for i in 0..ps_dec.subfr_length as usize {
                res_q14[i] = ps_dec.exc_q14[pexc_q14_idx + i];
            }
        }

        for i in 0..ps_dec.subfr_length as usize {
            /* Short-term prediction */
            let mut lpc_pred_q10: i32 = ps_dec.lpc_order >> 1;

            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + i - 1],
                a_q12[0] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + i - 2],
                a_q12[1] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + i - 3],
                a_q12[2] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + i - 4],
                a_q12[3] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + i - 5],
                a_q12[4] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + i - 6],
                a_q12[5] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + i - 7],
                a_q12[6] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + i - 8],
                a_q12[7] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + i - 9],
                a_q12[8] as i32,
            );
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_lpc_q14[MAX_LPC_ORDER + i - 10],
                a_q12[9] as i32,
            );

            if ps_dec.lpc_order == 16 {
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + i - 11],
                    a_q12[10] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + i - 12],
                    a_q12[11] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + i - 13],
                    a_q12[12] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + i - 14],
                    a_q12[13] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + i - 15],
                    a_q12[14] as i32,
                );
                lpc_pred_q10 = silk_smlawb(
                    lpc_pred_q10,
                    s_lpc_q14[MAX_LPC_ORDER + i - 16],
                    a_q12[15] as i32,
                );
            }

            /* Add prediction to LPC excitation */
            s_lpc_q14[MAX_LPC_ORDER + i] =
                silk_add_sat32(res_q14[i], silk_lshift_sat32(lpc_pred_q10, 4));

            /* Scale with gain */
            /* C: Gain_Q10 = silk_RSHIFT(Gains_Q16[k], 6); */
            /* C: pxq[i] = silk_SAT16(silk_RSHIFT_ROUND(silk_SMULWW(sLPC_Q14[...], Gain_Q10), 8)); */
            let gain_q10 = ps_dec_ctrl.gains_q16[k] >> 6;
            let product = silk_smulww(s_lpc_q14[MAX_LPC_ORDER + i], gain_q10);
            xq[pxq_idx + i] = silk_sat16(silk_rshift_round(product, 8)) as i16;
        }
        /* Update LPC filter state */
        for i in 0..MAX_LPC_ORDER {
            s_lpc_q14[i] = s_lpc_q14[ps_dec.subfr_length as usize + i];
        }
        pexc_q14_idx += ps_dec.subfr_length as usize;
        pxq_idx += ps_dec.subfr_length as usize;
    }

    /* Save LPC state */
    ps_dec
        .s_lpc_q14_buf
        .copy_from_slice(&s_lpc_q14[..MAX_LPC_ORDER]);
}

/// LPC analysis filter with offset support
/// C: silk_LPC_analysis_filter(&sLTP[start_idx], &outBuf[start_idx + k*subfr_length], A_Q12, len, d)
/// out[out_offset..out_offset+len] is written, input[input_offset..input_offset+len] is read
fn silk_lpc_analysis_filter_offset(
    out: &mut [i16],
    out_offset: usize,
    input: &[i16],
    input_offset: usize,
    b: &[i16],
    len: usize,
    d: usize,
) {
    for ix in 0..d {
        out[out_offset + ix] = 0;
    }

    for ix in d..len {
        let mut out32_q12: i32 = 0;
        for j in 0..d {
            out32_q12 = out32_q12.wrapping_add(silk_smulbb(
                input[input_offset + ix - j - 1] as i32,
                b[j] as i32,
            ));
        }

        /* Subtract prediction */
        out32_q12 = ((input[input_offset + ix] as i32) << 12).wrapping_sub(out32_q12);

        /* Scale to Q0 */
        let out32 = silk_rshift_round(out32_q12, 12);

        /* Saturate output */
        out[out_offset + ix] = silk_sat16(out32) as i16;
    }
}
