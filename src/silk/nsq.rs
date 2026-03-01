use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::sigproc_fix::*;
use crate::silk::structs::*;
use crate::silk::tables::*;

pub fn silk_nsq(
    ps_enc_c: &SilkEncoderStateCommon,
    nsq: &mut SilkNSQState,
    ps_indices: &SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltp_coef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) {
    let mut lag: usize;
    let mut start_idx: usize;
    let lsf_interpolation_flag: i32;
    let mut a_q12: &[i16];
    let mut b_q14: &[i16];
    let mut ar_shp_q13: &[i16];
    let mut harm_shape_fir_packed_q14: i32;
    let mut offset_q10: i32;

    let mut s_ltp_q15 = [0i32; LTP_MEM_LENGTH_MS * MAX_FS_KHZ + MAX_FRAME_LENGTH];
    let mut s_ltp = [0i16; LTP_MEM_LENGTH_MS * MAX_FS_KHZ + MAX_FRAME_LENGTH];
    let mut x_sc_q10 = [0i32; MAX_SUB_FRAME_LENGTH];

    nsq.rand_seed = ps_indices.seed as i32;

    /* Set unvoiced lag to the previous one, overwrite later for voiced */
    lag = nsq.lag_prev as usize;

    /* Interpolation status */
    if ps_indices.nlsf_interp_coef_q2 == 4 {
        lsf_interpolation_flag = 0;
    } else {
        lsf_interpolation_flag = 1;
    }

    nsq.s_ltp_shp_buf_idx = ps_enc_c.ltp_mem_length as i32;
    nsq.s_ltp_buf_idx = ps_enc_c.ltp_mem_length as i32;

    let pxq_idx = ps_enc_c.ltp_mem_length as usize;

    for k in 0..ps_enc_c.nb_subfr as usize {
        a_q12 =
            &pred_coef_q12[((k >> 1) | (1 - lsf_interpolation_flag as usize)) * MAX_LPC_ORDER..];
        b_q14 = &ltp_coef_q14[k * LTP_ORDER..];
        ar_shp_q13 = &ar_q13[k * MAX_SHAPE_LPC_ORDER..];

        nsq.rewhite_flag = 0;
        if ps_indices.signal_type == TYPE_VOICED as i8 {
            lag = pitch_l[k] as usize;

            /* Re-whitening */
            if (k & (3 - (lsf_interpolation_flag << 1) as usize)) == 0 {
                /* C: start_idx = ltp_mem_length - lag - predictLPCOrder - LTP_ORDER/2 */
                let start_idx_signed = ps_enc_c.ltp_mem_length
                    - lag as i32
                    - ps_enc_c.predict_lpc_order
                    - (LTP_ORDER as i32 / 2);
                debug_assert!(start_idx_signed > 0);
                start_idx = start_idx_signed as usize;

                silk_lpc_analysis_filter(
                    &mut s_ltp[start_idx..],
                    &nsq.xq[start_idx + k * ps_enc_c.subfr_length as usize..],
                    a_q12,
                    (ps_enc_c.ltp_mem_length as usize) - start_idx,
                    ps_enc_c.predict_lpc_order as usize,
                    0, // arch
                );
                nsq.rewhite_flag = 1;
                nsq.s_ltp_buf_idx = ps_enc_c.ltp_mem_length as i32;
            }
        }

        /* Noise shape parameters */
        harm_shape_fir_packed_q14 = harm_shape_gain_q14[k] >> 2;
        harm_shape_fir_packed_q14 |= (harm_shape_gain_q14[k] >> 1) << 16;

        offset_q10 = SILK_QUANT_OFFSETS_Q10[(ps_indices.signal_type >> 1) as usize]
            [ps_indices.quant_offset_type as usize] as i32;

        silk_nsq_scale_states(
            ps_enc_c,
            nsq,
            &x16[k * ps_enc_c.subfr_length as usize..],
            &mut x_sc_q10,
            &s_ltp,
            &mut s_ltp_q15,
            k,
            ltp_scale_q14,
            gains_q16,
            pitch_l,
            ps_indices.signal_type as i32,
        );

        silk_noise_shape_quantizer(
            nsq,
            ps_indices.signal_type as i32,
            &x_sc_q10,
            &mut pulses[k * ps_enc_c.subfr_length as usize..],
            pxq_idx + k * ps_enc_c.subfr_length as usize,
            &mut s_ltp_q15,
            a_q12,
            b_q14,
            ar_shp_q13,
            lag,
            harm_shape_fir_packed_q14,
            tilt_q14[k],
            lf_shp_q14[k],
            gains_q16[k],
            lambda_q10,
            offset_q10,
            ps_enc_c.subfr_length as usize,
            ps_enc_c.shaping_lpc_order as usize,
            ps_enc_c.predict_lpc_order as usize,
            k,
        );
    }

    nsq.lag_prev = pitch_l[ps_enc_c.nb_subfr as usize - 1];

    /* Save quantized speech and noise shaping signals */
    let frame_length = ps_enc_c.frame_length as usize;
    let ltp_mem_length = ps_enc_c.ltp_mem_length as usize;

    // xq move
    let mut tmp_xq = [0i16; MAX_FRAME_LENGTH * 2];
    tmp_xq[..ltp_mem_length].copy_from_slice(&nsq.xq[frame_length..frame_length + ltp_mem_length]);
    nsq.xq[..ltp_mem_length].copy_from_slice(&tmp_xq[..ltp_mem_length]);

    // sLTP_shp_Q14 move
    let mut tmp_ltp_shp = [0i32; MAX_FRAME_LENGTH * 2];
    tmp_ltp_shp[..ltp_mem_length]
        .copy_from_slice(&nsq.s_ltp_shp_q14[frame_length..frame_length + ltp_mem_length]);
    nsq.s_ltp_shp_q14[..ltp_mem_length].copy_from_slice(&tmp_ltp_shp[..ltp_mem_length]);
}

pub fn silk_nsq_scale_states(
    ps_enc_c: &SilkEncoderStateCommon,
    nsq: &mut SilkNSQState,
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    k: usize,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: i32,
) {
    let subfr_length = ps_enc_c.subfr_length as usize;
    let ltp_mem_length = ps_enc_c.ltp_mem_length as usize;
    let lag = pitch_l[k] as usize;

    let mut inv_gain_q31 = silk_inverse32_varq(if gains_q16[k] > 1 { gains_q16[k] } else { 1 }, 47);

    /* Scale input */
    let inv_gain_q26 = silk_rshift_round(inv_gain_q31, 5);
    for i in 0..subfr_length {
        x_sc_q10[i] = silk_smulww(x16[i] as i32, inv_gain_q26);
    }

    /* After rewhitening the LTP state is un-scaled, so scale with inv_gain_Q31 */
    if nsq.rewhite_flag != 0 {
        if k == 0 {
            /* Do LTP downscaling */
            inv_gain_q31 = silk_lshift(silk_smulwb(inv_gain_q31, ltp_scale_q14), 2);
        }
        let start = (nsq.s_ltp_buf_idx as usize)
            .wrapping_sub(lag)
            .wrapping_sub(LTP_ORDER / 2);
        for i in start..nsq.s_ltp_buf_idx as usize {
            s_ltp_q15[i] = silk_smulwb(inv_gain_q31, s_ltp[i] as i32);
        }
    }

    /* Adjust for changing gain */
    if gains_q16[k] != nsq.prev_gain_q16 {
        let gain_adj_q16 = silk_div32_varq(
            nsq.prev_gain_q16,
            if gains_q16[k] > 1 { gains_q16[k] } else { 1 },
            16,
        );

        /* Scale long-term shaping state */
        let shp_start = (nsq.s_ltp_shp_buf_idx - ltp_mem_length as i32) as usize;
        for i in shp_start..nsq.s_ltp_shp_buf_idx as usize {
            nsq.s_ltp_shp_q14[i] = silk_smulww(gain_adj_q16, nsq.s_ltp_shp_q14[i]);
        }

        /* Scale long-term prediction state */
        if signal_type == TYPE_VOICED as i32 && nsq.rewhite_flag == 0 {
            let ltp_start = (nsq.s_ltp_buf_idx as usize)
                .wrapping_sub(lag)
                .wrapping_sub(LTP_ORDER / 2);
            for i in ltp_start..nsq.s_ltp_buf_idx as usize {
                s_ltp_q15[i] = silk_smulww(gain_adj_q16, s_ltp_q15[i]);
            }
        }

        nsq.s_lf_ar_q14 = silk_smulww(gain_adj_q16, nsq.s_lf_ar_q14);
        nsq.s_diff_shp_q14 = silk_smulww(gain_adj_q16, nsq.s_diff_shp_q14);

        /* Scale short-term prediction and shaping states */
        for i in 0..NSQ_LPC_BUF_LENGTH {
            nsq.s_lpc_q14[i] = silk_smulww(gain_adj_q16, nsq.s_lpc_q14[i]);
        }
        for i in 0..MAX_SHAPE_LPC_ORDER {
            nsq.s_ar2_q14[i] = silk_smulww(gain_adj_q16, nsq.s_ar2_q14[i]);
        }

        /* Save inverse gain */
        nsq.prev_gain_q16 = gains_q16[k];
    }

    /* Save signal type for next subframe */
    nsq.prev_sig_type = signal_type as i8;
}

/// Noise shape feedback loop - matches C silk_NSQ_noise_shape_feedback_loop_c
/// Shifts data0 value into data1 shift register and computes weighted sum

#[inline(always)]
fn silk_nsq_noise_shape_feedback_loop(
    data0_val: i32,
    data1: &mut [i32],
    coef: &[i16],
    order: usize,
) -> i32 {
    // SAFETY: order ≤ MAX_SHAPE_LPC_ORDER ≤ data1.len() and coef.len()
    unsafe {
        let mut tmp2 = data0_val;
        let mut tmp1 = *data1.get_unchecked(0);
        *data1.get_unchecked_mut(0) = tmp2;

        let mut out = (order as i32) >> 1;
        out = silk_smlawb(out, tmp2, *coef.get_unchecked(0) as i32);

        let mut j = 2;
        while j < order {
            tmp2 = *data1.get_unchecked(j - 1);
            *data1.get_unchecked_mut(j - 1) = tmp1;
            out = silk_smlawb(out, tmp1, *coef.get_unchecked(j - 1) as i32);
            tmp1 = *data1.get_unchecked(j);
            *data1.get_unchecked_mut(j) = tmp2;
            out = silk_smlawb(out, tmp2, *coef.get_unchecked(j) as i32);
            j += 2;
        }
        *data1.get_unchecked_mut(order - 1) = tmp1;
        out = silk_smlawb(out, tmp1, *coef.get_unchecked(order - 1) as i32);
        /* Q11 -> Q12 */
        out <<= 1;
        out
    }
}

fn silk_noise_shape_quantizer(
    nsq: &mut SilkNSQState,
    signal_type: i32,
    x_sc_q10: &[i32],
    pulses: &mut [i8],
    xq_offset: usize,
    s_ltp_q15: &mut [i32],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: usize,
    harm_shape_fir_packed_q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    gain_q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    subfr_length: usize,
    shaping_lpc_order: usize,
    predict_lpc_order: usize,
    _subfr_idx: usize,
) {
    let mut ltp_pred_q13: i32;
    let mut lpc_pred_q10: i32;
    let mut n_ar_q12: i32;
    let mut n_lf_q12: i32;
    let mut n_ltp_q13: i32;
    let mut r_q10: i32;
    let mut rr_q10: i32;
    let mut q1_q0: i32;
    let mut q1_q10: i32;
    let mut q2_q10: i32;
    let mut rd1_q20: i32;
    let mut rd2_q20: i32;
    let mut exc_q14: i32;
    let mut lpc_exc_q14: i32;
    let mut xq_q14: i32;
    let mut tmp1: i32;
    let mut tmp2: i32;
    let mut s_lf_ar_shp_q14: i32;

    #[cfg(debug_assertions)]
    let debug_nsq = std::env::var("SILK_DEBUG_NSQ").is_ok();

    let gain_q10 = gain_q16 >> 6;

    /* Compute base indices for lagged access (before loop, as in C) */
    let shp_lag_base = (nsq.s_ltp_shp_buf_idx as usize).wrapping_sub(lag) + HARM_SHAPE_FIR_TAPS / 2;
    let pred_lag_base = (nsq.s_ltp_buf_idx as usize).wrapping_sub(lag) + LTP_ORDER / 2;

    for i in 0..subfr_length {
        /* Generate dither */
        nsq.rand_seed = silk_rand(nsq.rand_seed);

        /* Short-term prediction */
        let ps_lpc_idx = NSQ_LPC_BUF_LENGTH - 1 + i;
        lpc_pred_q10 = (predict_lpc_order as i32) >> 1;
        // SAFETY: ps_lpc_idx - j stays within [NSQ_LPC_BUF_LENGTH-1 .. NSQ_LPC_BUF_LENGTH-1+subfr_length),
        // and a_q12 has at least predict_lpc_order elements (MAX_LPC_ORDER).
        for j in 0..predict_lpc_order {
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                unsafe { *nsq.s_lpc_q14.get_unchecked(ps_lpc_idx - j) },
                unsafe { *a_q12.get_unchecked(j) } as i32,
            );
        }

        /* Long-term prediction */
        if signal_type == TYPE_VOICED as i32 {
            /* Avoids introducing a bias because silk_SMLAWB() always rounds to -inf */
            ltp_pred_q13 = 2;
            ltp_pred_q13 = silk_smlawb(ltp_pred_q13, s_ltp_q15[pred_lag_base + i], b_q14[0] as i32);
            ltp_pred_q13 = silk_smlawb(
                ltp_pred_q13,
                s_ltp_q15[pred_lag_base + i - 1],
                b_q14[1] as i32,
            );
            ltp_pred_q13 = silk_smlawb(
                ltp_pred_q13,
                s_ltp_q15[pred_lag_base + i - 2],
                b_q14[2] as i32,
            );
            ltp_pred_q13 = silk_smlawb(
                ltp_pred_q13,
                s_ltp_q15[pred_lag_base + i - 3],
                b_q14[3] as i32,
            );
            ltp_pred_q13 = silk_smlawb(
                ltp_pred_q13,
                s_ltp_q15[pred_lag_base + i - 4],
                b_q14[4] as i32,
            );
        } else {
            ltp_pred_q13 = 0;
        }

        /* Noise shape feedback */
        n_ar_q12 = silk_nsq_noise_shape_feedback_loop(
            nsq.s_diff_shp_q14,
            &mut nsq.s_ar2_q14,
            ar_shp_q13,
            shaping_lpc_order,
        );
        n_ar_q12 = silk_smlawb(n_ar_q12, nsq.s_lf_ar_q14, tilt_q14);

        n_lf_q12 = silk_smulwb(
            nsq.s_ltp_shp_q14[(nsq.s_ltp_shp_buf_idx - 1) as usize],
            lf_shp_q14,
        );
        n_lf_q12 = silk_smlawt(n_lf_q12, nsq.s_lf_ar_q14, lf_shp_q14);

        /* Combine prediction and noise shaping signals */
        tmp1 = (lpc_pred_q10 << 2).wrapping_sub(n_ar_q12); /* Q12 */
        tmp1 = tmp1.wrapping_sub(n_lf_q12); /* Q12 */
        if lag > 0 {
            /* Symmetric, packed FIR coefficients */
            let shp_idx = shp_lag_base + i;
            n_ltp_q13 = silk_smulwb(
                silk_add_sat32(nsq.s_ltp_shp_q14[shp_idx], nsq.s_ltp_shp_q14[shp_idx - 2]),
                harm_shape_fir_packed_q14,
            );
            n_ltp_q13 = silk_smlawt(
                n_ltp_q13,
                nsq.s_ltp_shp_q14[shp_idx - 1],
                harm_shape_fir_packed_q14,
            );
            n_ltp_q13 <<= 1;

            tmp2 = ltp_pred_q13 - n_ltp_q13; /* Q13 */
            tmp1 = tmp2.wrapping_add(tmp1 << 1); /* Q13 */
            tmp1 = silk_rshift_round(tmp1, 3); /* Q10 */
        } else {
            tmp1 = silk_rshift_round(tmp1, 2); /* Q10 */
        }

        r_q10 = x_sc_q10[i] - tmp1; /* residual error Q10 */

        /* Debug: dump first few samples per subframe */
        #[cfg(debug_assertions)]
        if debug_nsq && i < 5 {
            eprintln!(
                "  nsq[sf={},i={}]: x_sc_q10={} lpc_pred_q10={} n_ar_q12={} n_lf_q12={} tmp1={} r_q10={} rand_seed={}",
                _subfr_idx,
                i,
                x_sc_q10[i],
                lpc_pred_q10,
                n_ar_q12,
                n_lf_q12,
                tmp1,
                r_q10,
                nsq.rand_seed
            );
        }

        /* Flip sign depending on dither */
        if nsq.rand_seed < 0 {
            r_q10 = -r_q10;
        }
        r_q10 = silk_limit_32(r_q10, -(31 << 10), 30 << 10);

        /* Find two quantization level candidates and measure their rate-distortion */
        q1_q10 = r_q10 - offset_q10;
        q1_q0 = q1_q10 >> 10;
        if lambda_q10 > 2048 {
            /* For aggressive RDO, the bias becomes more than one pulse. */
            let rdo_offset = lambda_q10 / 2 - 512;
            if q1_q10 > rdo_offset {
                q1_q0 = (q1_q10 - rdo_offset) >> 10;
            } else if q1_q10 < -rdo_offset {
                q1_q0 = (q1_q10 + rdo_offset) >> 10;
            } else if q1_q10 < 0 {
                q1_q0 = -1;
            } else {
                q1_q0 = 0;
            }
        }

        if q1_q0 > 0 {
            q1_q10 = (q1_q0 << 10) - QUANT_LEVEL_ADJUST_Q10 + offset_q10;
            q2_q10 = q1_q10 + 1024;
            rd1_q20 = silk_smulbb(q1_q10, lambda_q10);
            rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
        } else if q1_q0 == 0 {
            q1_q10 = offset_q10;
            q2_q10 = q1_q10 + 1024 - QUANT_LEVEL_ADJUST_Q10;
            rd1_q20 = silk_smulbb(q1_q10, lambda_q10);
            rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
        } else if q1_q0 == -1 {
            q2_q10 = offset_q10;
            q1_q10 = q2_q10 - (1024 - QUANT_LEVEL_ADJUST_Q10);
            rd1_q20 = silk_smulbb(-q1_q10, lambda_q10);
            rd2_q20 = silk_smulbb(q2_q10, lambda_q10);
        } else {
            /* q1_q0 < -1 */
            q1_q10 = (q1_q0 << 10) + QUANT_LEVEL_ADJUST_Q10 + offset_q10;
            q2_q10 = q1_q10 + 1024;
            rd1_q20 = silk_smulbb(-q1_q10, lambda_q10);
            rd2_q20 = silk_smulbb(-q2_q10, lambda_q10);
        }
        rr_q10 = r_q10 - q1_q10;
        rd1_q20 = silk_smlabb(rd1_q20, rr_q10, rr_q10);
        rr_q10 = r_q10 - q2_q10;
        rd2_q20 = silk_smlabb(rd2_q20, rr_q10, rr_q10);

        if rd2_q20 < rd1_q20 {
            q1_q10 = q2_q10;
        }

        pulses[i] = silk_rshift_round(q1_q10, 10) as i8;

        #[cfg(debug_assertions)]
        if debug_nsq && i < 5 {
            eprintln!(
                "  nsq[sf={},i={}]: pulse={} q1_q10={} offset_q10={} rd1={} rd2={}",
                _subfr_idx, i, pulses[i], q1_q10, offset_q10, rd1_q20, rd2_q20
            );
        }

        /* Excitation */
        exc_q14 = q1_q10 << 4;
        if nsq.rand_seed < 0 {
            exc_q14 = -exc_q14;
        }

        /* Add predictions */
        lpc_exc_q14 = silk_add_lshift32(exc_q14, ltp_pred_q13, 1);
        xq_q14 = lpc_exc_q14.wrapping_add(lpc_pred_q10 << 4);

        /* Scale XQ back to normal level before saving */
        nsq.xq[xq_offset + i] =
            silk_sat16(silk_rshift_round(silk_smulww(xq_q14, gain_q10), 8)) as i16;

        /* Update states */
        nsq.s_lpc_q14[NSQ_LPC_BUF_LENGTH + i] = xq_q14;
        nsq.s_diff_shp_q14 = xq_q14.wrapping_sub(x_sc_q10[i] << 4);
        s_lf_ar_shp_q14 = nsq.s_diff_shp_q14.wrapping_sub(n_ar_q12 << 2);
        nsq.s_lf_ar_q14 = s_lf_ar_shp_q14;

        nsq.s_ltp_shp_q14[nsq.s_ltp_shp_buf_idx as usize] =
            s_lf_ar_shp_q14.wrapping_sub(n_lf_q12 << 2);
        s_ltp_q15[nsq.s_ltp_buf_idx as usize] = lpc_exc_q14 << 1;
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        /* Make dither dependent on quantized signal */
        nsq.rand_seed = nsq.rand_seed.wrapping_add(pulses[i] as i32);
    }

    /* Update LPC synth buffer */
    nsq.s_lpc_q14
        .copy_within(subfr_length..subfr_length + NSQ_LPC_BUF_LENGTH, 0);
}
