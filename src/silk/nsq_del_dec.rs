use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::sigproc_fix::*;
use crate::silk::structs::*;
use crate::silk::tables::*;

#[derive(Copy, Clone)]
pub struct NSQDelDecStruct {
    pub s_lpc_q14: [i32; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    pub rand_state: [i32; DECISION_DELAY],
    pub q_q10: [i32; DECISION_DELAY],
    pub xq_q14: [i32; DECISION_DELAY],
    pub pred_q15: [i32; DECISION_DELAY],
    pub shape_q14: [i32; DECISION_DELAY],
    pub s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER],
    pub lf_ar_q14: i32,
    pub diff_q14: i32,
    pub seed: i32,
    pub seed_init: i32,
    pub rd_q10: i32,
}

impl Default for NSQDelDecStruct {
    fn default() -> Self {
        Self {
            s_lpc_q14: [0; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
            rand_state: [0; DECISION_DELAY],
            q_q10: [0; DECISION_DELAY],
            xq_q14: [0; DECISION_DELAY],
            pred_q15: [0; DECISION_DELAY],
            shape_q14: [0; DECISION_DELAY],
            s_ar2_q14: [0; MAX_SHAPE_LPC_ORDER],
            lf_ar_q14: 0,
            diff_q14: 0,
            seed: 0,
            seed_init: 0,
            rd_q10: 0,
        }
    }
}

#[derive(Copy, Clone, Default)]
pub struct NSQSampleStruct {
    pub q_q10: i32,
    pub rd_q10: i32,
    pub xq_q14: i32,
    pub lf_ar_q14: i32,
    pub diff_q14: i32,
    pub s_ltp_shp_q14: i32,
    pub lpc_exc_q14: i32,
}

pub type NSQSamplePair = [NSQSampleStruct; 2];

#[inline]
fn silk_nsq_del_dec_scale_states(
    ps_enc_c: &SilkEncoderStateCommon,
    nsq: &mut SilkNSQState,
    ps_del_dec: &mut [NSQDelDecStruct],
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: usize,
    n_states_delayed_decision: i32,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: i32,
    decision_delay: i32,
) {
    let lag = pitch_l[subfr] as usize;
    let inv_gain_q31 = silk_inverse32_varq(gains_q16[subfr].max(1), 47);

    let inv_gain_q26 = silk_rshift_round(inv_gain_q31, 5);
    for i in 0..ps_enc_c.subfr_length as usize {
        x_sc_q10[i] = silk_smulww(x16[i] as i32, inv_gain_q26);
    }

    if nsq.rewhite_flag != 0 {
        let mut inv_gain_q31_scaled = inv_gain_q31;
        if subfr == 0 {
            inv_gain_q31_scaled = silk_lshift(silk_smulwb(inv_gain_q31, ltp_scale_q14), 2);
        }
        for i in (nsq.s_ltp_buf_idx as usize - lag - LTP_ORDER / 2)..(nsq.s_ltp_buf_idx as usize) {
            s_ltp_q15[i] = silk_smulwb(inv_gain_q31_scaled, s_ltp[i] as i32);
        }
    }

    if gains_q16[subfr] != nsq.prev_gain_q16 {
        let gain_adj_q16 = silk_div32_varq(nsq.prev_gain_q16, gains_q16[subfr], 16);

        for i in (nsq.s_ltp_shp_buf_idx as usize - ps_enc_c.ltp_mem_length as usize)
            ..(nsq.s_ltp_shp_buf_idx as usize)
        {
            nsq.s_ltp_shp_q14[i] = silk_smulww(gain_adj_q16, nsq.s_ltp_shp_q14[i]);
        }

        if signal_type == TYPE_VOICED as i32 && nsq.rewhite_flag == 0 {
            for i in (nsq.s_ltp_buf_idx as usize - lag - LTP_ORDER / 2)
                ..(nsq.s_ltp_buf_idx as usize - decision_delay as usize)
            {
                s_ltp_q15[i] = silk_smulww(gain_adj_q16, s_ltp_q15[i]);
            }
        }

        for k in 0..n_states_delayed_decision as usize {
            let ps_dd = &mut ps_del_dec[k];
            ps_dd.lf_ar_q14 = silk_smulww(gain_adj_q16, ps_dd.lf_ar_q14);
            ps_dd.diff_q14 = silk_smulww(gain_adj_q16, ps_dd.diff_q14);
            for i in 0..NSQ_LPC_BUF_LENGTH {
                ps_dd.s_lpc_q14[i] = silk_smulww(gain_adj_q16, ps_dd.s_lpc_q14[i]);
            }
            for i in 0..MAX_SHAPE_LPC_ORDER {
                ps_dd.s_ar2_q14[i] = silk_smulww(gain_adj_q16, ps_dd.s_ar2_q14[i]);
            }
            for i in 0..DECISION_DELAY {
                ps_dd.pred_q15[i] = silk_smulww(gain_adj_q16, ps_dd.pred_q15[i]);
                ps_dd.shape_q14[i] = silk_smulww(gain_adj_q16, ps_dd.shape_q14[i]);
            }
        }
        nsq.prev_gain_q16 = gains_q16[subfr];
    }
}

#[inline]
fn silk_noise_shape_quantizer_short_prediction(
    ps_lpc_q14: &[i32],
    idx: usize,
    a_q12: &[i16],
    predict_lpc_order: i32,
) -> i32 {
    let mut out = silk_smulwb(ps_lpc_q14[idx], a_q12[0] as i32);
    for j in 1..predict_lpc_order as usize {
        out = silk_smlawb(out, ps_lpc_q14[idx - j], a_q12[j] as i32);
    }
    out
}

pub fn silk_noise_shape_quantizer_del_dec(
    nsq: &mut SilkNSQState,
    ps_del_dec: &mut [NSQDelDecStruct],
    signal_type: i32,
    x_q10: &[i32],
    pulses: &mut [i8],
    pulses_offset: i32,
    xq_ptr: i32,
    s_ltp_q15: &mut [i32],
    delayed_ga_q10: &mut [i32],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: i32,
    harm_shape_fir_packed_q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    gain_q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    length: i32,
    subfr: i32,
    shaping_lpc_order: i32,
    predict_lpc_order: i32,
    warping_q16: i32,
    n_states_delayed_decision: i32,
    smpl_buf_idx: &mut i32,
    decision_delay: i32,
) {
    let mut ps_sample_state = [[NSQSampleStruct::default(); 2]; NSQ_MAX_STATES_OPERATING];
    let gain_q10 = silk_rshift(gain_q16, 6);

    for i in 0..length {
        let idx = i as usize;
        let mut ltp_pred_q14 = 0;
        if signal_type == TYPE_VOICED as i32 {
            let pred_lag_idx = (nsq.s_ltp_buf_idx - lag + LTP_ORDER as i32 / 2 + i) as usize;
            ltp_pred_q14 = 2;
            ltp_pred_q14 = silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx], b_q14[0] as i32);
            ltp_pred_q14 = silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx - 1], b_q14[1] as i32);
            ltp_pred_q14 = silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx - 2], b_q14[2] as i32);
            ltp_pred_q14 = silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx - 3], b_q14[3] as i32);
            ltp_pred_q14 = silk_smlawb(ltp_pred_q14, s_ltp_q15[pred_lag_idx - 4], b_q14[4] as i32);
            ltp_pred_q14 = silk_lshift(ltp_pred_q14, 1);
        }

        let mut n_ltp_q14 = 0;
        if lag > 0 {
            let shp_lag_idx =
                (nsq.s_ltp_shp_buf_idx - lag + HARM_SHAPE_FIR_TAPS as i32 / 2 + i) as usize;
            n_ltp_q14 = silk_smulwb(
                silk_add_sat32(
                    nsq.s_ltp_shp_q14[shp_lag_idx],
                    nsq.s_ltp_shp_q14[shp_lag_idx - 2],
                ),
                harm_shape_fir_packed_q14,
            );
            n_ltp_q14 = silk_smlawt(
                n_ltp_q14,
                nsq.s_ltp_shp_q14[shp_lag_idx - 1],
                harm_shape_fir_packed_q14,
            );
            n_ltp_q14 = silk_sub_lshift32(ltp_pred_q14, n_ltp_q14, 2);
        }

        for k in 0..n_states_delayed_decision as usize {
            let ps_dd = &mut ps_del_dec[k];
            let ps_ss = &mut ps_sample_state[k];

            ps_dd.seed = silk_rand(ps_dd.seed);
            let ps_lpc_q14_idx = NSQ_LPC_BUF_LENGTH - 1 + idx;
            let lpc_pred_q14 = silk_lshift(
                silk_noise_shape_quantizer_short_prediction(
                    &ps_dd.s_lpc_q14,
                    ps_lpc_q14_idx,
                    a_q12,
                    predict_lpc_order,
                ),
                4,
            );

            let mut tmp2 = silk_smlawb(ps_dd.diff_q14, ps_dd.s_ar2_q14[0], warping_q16);
            let mut tmp1 = silk_smlawb(
                ps_dd.s_ar2_q14[0],
                silk_sub32_ovflw(ps_dd.s_ar2_q14[1], tmp2),
                warping_q16,
            );
            ps_dd.s_ar2_q14[0] = tmp2;
            let mut n_ar_q14 = silk_rshift(shaping_lpc_order, 1);
            n_ar_q14 = silk_smlawb(n_ar_q14, tmp2, ar_shp_q13[0] as i32);
            for j in (2..shaping_lpc_order as usize).step_by(2) {
                tmp2 = silk_smlawb(
                    ps_dd.s_ar2_q14[j - 1],
                    silk_sub32_ovflw(ps_dd.s_ar2_q14[j], tmp1),
                    warping_q16,
                );
                ps_dd.s_ar2_q14[j - 1] = tmp1;
                n_ar_q14 = silk_smlawb(n_ar_q14, tmp1, ar_shp_q13[j - 1] as i32);
                tmp1 = silk_smlawb(
                    ps_dd.s_ar2_q14[j],
                    silk_sub32_ovflw(ps_dd.s_ar2_q14[j + 1], tmp2),
                    warping_q16,
                );
                ps_dd.s_ar2_q14[j] = tmp2;
                n_ar_q14 = silk_smlawb(n_ar_q14, tmp2, ar_shp_q13[j] as i32);
            }
            ps_dd.s_ar2_q14[shaping_lpc_order as usize - 1] = tmp1;
            n_ar_q14 = silk_smlawb(
                n_ar_q14,
                tmp1,
                ar_shp_q13[shaping_lpc_order as usize - 1] as i32,
            );

            n_ar_q14 = silk_lshift(n_ar_q14, 1);
            n_ar_q14 = silk_smlawb(n_ar_q14, ps_dd.lf_ar_q14, tilt_q14);
            n_ar_q14 = silk_lshift(n_ar_q14, 2);

            let mut n_lf_q14 = silk_smulwb(ps_dd.shape_q14[*smpl_buf_idx as usize], lf_shp_q14);
            n_lf_q14 = silk_smlawt(n_lf_q14, ps_dd.lf_ar_q14, lf_shp_q14);
            n_lf_q14 = silk_lshift(n_lf_q14, 2);

            let tmp1_val = silk_sub_sat32(
                silk_add32_ovflw(n_ltp_q14, lpc_pred_q14),
                silk_add_sat32(n_ar_q14, n_lf_q14),
            );
            let r_q10 = x_q10[idx] - silk_rshift_round(tmp1_val, 4);

            let r_q10_signed = if ps_dd.seed < 0 { -r_q10 } else { r_q10 };
            let r_q10_signed = silk_limit_32(r_q10_signed, -(31 << 10), 30 << 10);

            let q1_q10_in = r_q10_signed - offset_q10;
            let mut q1_q0 = silk_rshift(q1_q10_in, 10);
            if lambda_q10 > 2048 {
                let rdo_offset = lambda_q10 / 2 - 512;
                if q1_q10_in > rdo_offset {
                    q1_q0 = silk_rshift(q1_q10_in - rdo_offset, 10);
                } else if q1_q10_in < -rdo_offset {
                    q1_q0 = silk_rshift(q1_q10_in + rdo_offset, 10);
                } else if q1_q10_in < 0 {
                    q1_q0 = -1;
                } else {
                    q1_q0 = 0;
                }
            }

            let (rd1_q10, rd2_q10, q1_q10_val, q2_q10_val);
            if q1_q0 > 0 {
                q1_q10_val =
                    silk_sub32(silk_lshift(q1_q0, 10), QUANT_LEVEL_ADJUST_Q10) + offset_q10;
                q2_q10_val = q1_q10_val + 1024;
                rd1_q10 = silk_smulbb(q1_q10_val, lambda_q10);
                rd2_q10 = silk_smulbb(q2_q10_val, lambda_q10);
            } else if q1_q0 == 0 {
                q1_q10_val = offset_q10;
                q2_q10_val = q1_q10_val + 1024 - QUANT_LEVEL_ADJUST_Q10;
                rd1_q10 = silk_smulbb(q1_q10_val, lambda_q10);
                rd2_q10 = silk_smulbb(q2_q10_val, lambda_q10);
            } else if q1_q0 == -1 {
                q2_q10_val = offset_q10;
                q1_q10_val = q2_q10_val - (1024 - QUANT_LEVEL_ADJUST_Q10);
                rd1_q10 = silk_smulbb(-q1_q10_val, lambda_q10);
                rd2_q10 = silk_smulbb(q2_q10_val, lambda_q10);
            } else {
                q1_q10_val =
                    silk_add32(silk_lshift(q1_q0, 10), QUANT_LEVEL_ADJUST_Q10) + offset_q10;
                q2_q10_val = q1_q10_val + 1024;
                rd1_q10 = silk_smulbb(-q1_q10_val, lambda_q10);
                rd2_q10 = silk_smulbb(-q2_q10_val, lambda_q10);
            }

            let mut rr_q10 = r_q10_signed - q1_q10_val;
            let rd1_q10_final = silk_rshift(silk_smlabb(rd1_q10, rr_q10, rr_q10), 10);
            rr_q10 = r_q10_signed - q2_q10_val;
            let rd2_q10_final = silk_rshift(silk_smlabb(rd2_q10, rr_q10, rr_q10), 10);

            if rd1_q10_final < rd2_q10_final {
                ps_ss[0].rd_q10 = ps_dd.rd_q10 + rd1_q10_final;
                ps_ss[1].rd_q10 = ps_dd.rd_q10 + rd2_q10_final;
                ps_ss[0].q_q10 = q1_q10_val;
                ps_ss[1].q_q10 = q2_q10_val;
            } else {
                ps_ss[0].rd_q10 = ps_dd.rd_q10 + rd2_q10_final;
                ps_ss[1].rd_q10 = ps_dd.rd_q10 + rd1_q10_final;
                ps_ss[0].q_q10 = q2_q10_val;
                ps_ss[1].q_q10 = q1_q10_val;
            }

            for j in 0..2 {
                let mut exc_q14 = silk_lshift(ps_ss[j].q_q10, 4);
                if ps_dd.seed < 0 {
                    exc_q14 = -exc_q14;
                }
                let lpc_exc_q14 = silk_add32(exc_q14, ltp_pred_q14);
                let xq_q14 = silk_add32_ovflw(lpc_exc_q14, lpc_pred_q14);
                ps_ss[j].diff_q14 = silk_sub32_ovflw(xq_q14, silk_lshift(x_q10[idx], 4));
                let s_lf_ar_shp_q14 = silk_sub32_ovflw(ps_ss[j].diff_q14, n_ar_q14);
                ps_ss[j].s_ltp_shp_q14 = silk_sub_sat32(s_lf_ar_shp_q14, n_lf_q14);
                ps_ss[j].lf_ar_q14 = s_lf_ar_shp_q14;
                ps_ss[j].lpc_exc_q14 = lpc_exc_q14;
                ps_ss[j].xq_q14 = xq_q14;
            }
        }

        *smpl_buf_idx = (*smpl_buf_idx - 1 + DECISION_DELAY as i32) % DECISION_DELAY as i32;
        let last_smple_idx = (*smpl_buf_idx + decision_delay) % DECISION_DELAY as i32;

        let mut winner_ind = 0;
        let mut rd_min_q10 = ps_sample_state[0][0].rd_q10;
        for k in 1..n_states_delayed_decision as usize {
            if ps_sample_state[k][0].rd_q10 < rd_min_q10 {
                rd_min_q10 = ps_sample_state[k][0].rd_q10;
                winner_ind = k;
            }
        }

        let winner_rand_state = ps_del_dec[winner_ind].rand_state[last_smple_idx as usize];
        for k in 0..n_states_delayed_decision as usize {
            if ps_del_dec[k].rand_state[last_smple_idx as usize] != winner_rand_state {
                ps_sample_state[k][0].rd_q10 =
                    ps_sample_state[k][0].rd_q10.saturating_add(i32::MAX >> 4);
                ps_sample_state[k][1].rd_q10 =
                    ps_sample_state[k][1].rd_q10.saturating_add(i32::MAX >> 4);
            }
        }

        let mut rd_max_q10 = ps_sample_state[0][0].rd_q10;
        let mut rd_max_ind = 0;
        let mut rd_min_q10_2 = ps_sample_state[0][1].rd_q10;
        let mut rd_min_ind = 0;
        for k in 1..n_states_delayed_decision as usize {
            if ps_sample_state[k][0].rd_q10 > rd_max_q10 {
                rd_max_q10 = ps_sample_state[k][0].rd_q10;
                rd_max_ind = k;
            }
            if ps_sample_state[k][1].rd_q10 < rd_min_q10_2 {
                rd_min_q10_2 = ps_sample_state[k][1].rd_q10;
                rd_min_ind = k;
            }
        }

        if rd_min_q10_2 < rd_max_q10 {
            if rd_min_ind != rd_max_ind {
                let (min_state, max_state) = if rd_min_ind < rd_max_ind {
                    let (left, right) = ps_del_dec.split_at_mut(rd_max_ind);
                    (&left[rd_min_ind], &mut right[0])
                } else {
                    let (left, right) = ps_del_dec.split_at_mut(rd_min_ind);
                    (&right[0], &mut left[rd_max_ind])
                };
                max_state.s_lpc_q14[idx..].copy_from_slice(&min_state.s_lpc_q14[idx..]);
                max_state.rand_state = min_state.rand_state;
                max_state.q_q10 = min_state.q_q10;
                max_state.xq_q14 = min_state.xq_q14;
                max_state.pred_q15 = min_state.pred_q15;
                max_state.shape_q14 = min_state.shape_q14;
                max_state.s_ar2_q14 = min_state.s_ar2_q14;
                max_state.lf_ar_q14 = min_state.lf_ar_q14;
                max_state.diff_q14 = min_state.diff_q14;
                max_state.seed = min_state.seed;
                max_state.seed_init = min_state.seed_init;
                max_state.rd_q10 = min_state.rd_q10;
                ps_sample_state[rd_max_ind][0] = ps_sample_state[rd_min_ind][1];
            }
        }

        let ps_dd = &ps_del_dec[winner_ind];
        if subfr > 0 || i >= decision_delay {
            pulses[(pulses_offset + i - decision_delay) as usize] =
                silk_rshift_round(ps_dd.q_q10[last_smple_idx as usize], 10) as i8;
            nsq.xq[(xq_ptr + i - decision_delay) as usize] = silk_sat16(silk_rshift_round(
                silk_smulww(
                    ps_dd.xq_q14[last_smple_idx as usize],
                    delayed_ga_q10[last_smple_idx as usize],
                ),
                8,
            )) as i16;
            nsq.s_ltp_shp_q14[(nsq.s_ltp_shp_buf_idx - decision_delay) as usize] =
                ps_dd.shape_q14[last_smple_idx as usize];
            s_ltp_q15[(nsq.s_ltp_buf_idx - decision_delay) as usize] =
                ps_dd.pred_q15[last_smple_idx as usize];
        }
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        for k in 0..n_states_delayed_decision as usize {
            let ps_ss = &ps_sample_state[k][0];
            let ps_dd = &mut ps_del_dec[k];
            ps_dd.lf_ar_q14 = ps_ss.lf_ar_q14;
            ps_dd.diff_q14 = ps_ss.diff_q14;
            ps_dd.s_lpc_q14[NSQ_LPC_BUF_LENGTH + idx] = ps_ss.xq_q14;
            ps_dd.xq_q14[*smpl_buf_idx as usize] = ps_ss.xq_q14;
            ps_dd.q_q10[*smpl_buf_idx as usize] = ps_ss.q_q10;
            ps_dd.pred_q15[*smpl_buf_idx as usize] = silk_lshift(ps_ss.lpc_exc_q14, 1);
            ps_dd.shape_q14[*smpl_buf_idx as usize] = ps_ss.s_ltp_shp_q14;
            ps_dd.seed = silk_add32_ovflw(ps_dd.seed, silk_rshift_round(ps_ss.q_q10, 10));
            ps_dd.rand_state[*smpl_buf_idx as usize] = ps_dd.seed;
            ps_dd.rd_q10 = ps_ss.rd_q10;
        }
        delayed_ga_q10[*smpl_buf_idx as usize] = gain_q10;
    }
    for k in 0..n_states_delayed_decision as usize {
        let ps_dd = &mut ps_del_dec[k];
        let mut tmp = [0i32; NSQ_LPC_BUF_LENGTH];
        tmp.copy_from_slice(
            &ps_dd.s_lpc_q14[length as usize..length as usize + NSQ_LPC_BUF_LENGTH],
        );
        ps_dd.s_lpc_q14[..NSQ_LPC_BUF_LENGTH].copy_from_slice(&tmp);
    }
}

pub fn silk_nsq_del_dec(
    ps_common: &SilkEncoderStateCommon,
    ps_nsq: &mut SilkNSQState,
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
    let mut x_sc_q10 = [0i32; MAX_SUB_FRAME_LENGTH];
    let mut delayed_ga_q10 = [0i32; DECISION_DELAY];
    let mut s_ltp_q15 =
        vec![0i32; ps_common.ltp_mem_length as usize + ps_common.frame_length as usize];
    let mut s_ltp = vec![0i16; ps_common.ltp_mem_length as usize + ps_common.frame_length as usize];
    let mut ps_del_dec = [NSQDelDecStruct::default(); NSQ_MAX_STATES_OPERATING];

    let mut lag = ps_nsq.lag_prev;

    for k in 0..ps_common.n_states_delayed_decision as usize {
        let ps_dd = &mut ps_del_dec[k];
        ps_dd.seed = (k as i32 + ps_indices.seed as i32) & 3;
        ps_dd.seed_init = ps_dd.seed;
        ps_dd.rd_q10 = 0;
        ps_dd.lf_ar_q14 = ps_nsq.s_lf_ar_q14;
        ps_dd.diff_q14 = ps_nsq.s_diff_shp_q14;
        ps_dd.shape_q14[0] = ps_nsq.s_ltp_shp_q14[ps_common.ltp_mem_length as usize - 1];
        ps_dd.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]
            .copy_from_slice(&ps_nsq.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]);
        ps_dd.s_ar2_q14.copy_from_slice(&ps_nsq.s_ar2_q14);
    }

    let offset_q10 = SILK_QUANT_OFFSETS_Q10[(ps_indices.signal_type >> 1) as usize]
        [ps_indices.quant_offset_type as usize] as i32;
    let mut smpl_buf_idx = 0i32;
    let mut decision_delay = (DECISION_DELAY as i32).min(ps_common.subfr_length);

    if ps_indices.signal_type as i32 == TYPE_VOICED {
        for k in 0..ps_common.nb_subfr as usize {
            decision_delay = decision_delay.min(pitch_l[k] - LTP_ORDER as i32 / 2 - 1);
        }
    } else if lag > 0 {
        decision_delay = decision_delay.min(lag - LTP_ORDER as i32 / 2 - 1);
    }

    let lsf_interpolation_flag = if ps_indices.nlsf_interp_coef_q2 == 4 {
        0
    } else {
        1
    };

    ps_nsq.s_ltp_shp_buf_idx = ps_common.ltp_mem_length as i32;
    ps_nsq.s_ltp_buf_idx = ps_common.ltp_mem_length as i32;

    let mut x_ptr = 0;
    let mut pulses_ptr = 0;
    let mut xq_ptr = ps_common.ltp_mem_length as usize;
    let mut subfr_nsq = 0;

    for k in 0..ps_common.nb_subfr as usize {
        let a_q12 = &pred_coef_q12
            [((k >> 1) | (1 - lsf_interpolation_flag as usize)) * MAX_LPC_ORDER as usize..];
        let b_q14 = &ltp_coef_q14[k * LTP_ORDER as usize..];
        let ar_shp_q13 = &ar_q13[k * MAX_SHAPE_LPC_ORDER as usize..];

        let harm_shape_gain = harm_shape_gain_q14[k] as i32;
        let mut harm_shape_fir_packed_q14 = silk_rshift(harm_shape_gain, 2);
        harm_shape_fir_packed_q14 |= silk_lshift(silk_rshift(harm_shape_gain, 1), 16);

        ps_nsq.rewhite_flag = 0;
        if ps_indices.signal_type as i32 == TYPE_VOICED {
            lag = pitch_l[k];
            if (k & (3 - silk_lshift(lsf_interpolation_flag, 1) as usize)) == 0 {
                if k == 2 {
                    let mut rd_min_q10 = ps_del_dec[0].rd_q10;
                    let mut winner_ind = 0;
                    for i in 1..ps_common.n_states_delayed_decision as usize {
                        if ps_del_dec[i].rd_q10 < rd_min_q10 {
                            rd_min_q10 = ps_del_dec[i].rd_q10;
                            winner_ind = i;
                        }
                    }
                    for i in 0..ps_common.n_states_delayed_decision as usize {
                        if i != winner_ind {
                            ps_del_dec[i].rd_q10 =
                                ps_del_dec[i].rd_q10.saturating_add(i32::MAX >> 4);
                        }
                    }

                    let ps_dd = &ps_del_dec[winner_ind];
                    let mut last_smple_idx =
                        (smpl_buf_idx + decision_delay) % DECISION_DELAY as i32;
                    for i in 0..decision_delay {
                        last_smple_idx =
                            (last_smple_idx - 1 + DECISION_DELAY as i32) % DECISION_DELAY as i32;
                        pulses[(pulses_ptr as i32 + i - decision_delay) as usize] =
                            silk_rshift_round(ps_dd.q_q10[last_smple_idx as usize], 10) as i8;
                        ps_nsq.xq[(xq_ptr as i32 + i - decision_delay) as usize] =
                            silk_sat16(silk_rshift_round(
                                silk_smulww(ps_dd.xq_q14[last_smple_idx as usize], gains_q16[1]),
                                14,
                            )) as i16;
                        ps_nsq.s_ltp_shp_q14
                            [(ps_nsq.s_ltp_shp_buf_idx + i - decision_delay) as usize] =
                            ps_dd.shape_q14[last_smple_idx as usize];
                    }
                    subfr_nsq = 0;
                }

                let start_idx = (ps_common.ltp_mem_length
                    - lag
                    - ps_common.predict_lpc_order
                    - LTP_ORDER as i32 / 2) as usize;
                silk_lpc_analysis_filter(
                    &mut s_ltp[start_idx..],
                    &ps_nsq.xq[start_idx + k * ps_common.subfr_length as usize..],
                    a_q12,
                    ps_common.ltp_mem_length as usize - start_idx,
                    ps_common.predict_lpc_order as usize,
                    0,
                );
                ps_nsq.s_ltp_buf_idx = ps_common.ltp_mem_length as i32;
                ps_nsq.rewhite_flag = 1;
            }
        }

        silk_nsq_del_dec_scale_states(
            ps_common,
            ps_nsq,
            &mut ps_del_dec,
            &x16[x_ptr..],
            &mut x_sc_q10,
            &s_ltp,
            &mut s_ltp_q15,
            k,
            ps_common.n_states_delayed_decision as i32,
            ltp_scale_q14,
            gains_q16,
            pitch_l,
            ps_indices.signal_type as i32,
            decision_delay,
        );

        silk_noise_shape_quantizer_del_dec(
            ps_nsq,
            &mut ps_del_dec,
            ps_indices.signal_type as i32,
            &x_sc_q10,
            pulses,
            pulses_ptr as i32,
            xq_ptr as i32,
            &mut s_ltp_q15,
            &mut delayed_ga_q10,
            a_q12,
            b_q14,
            ar_shp_q13,
            lag,
            harm_shape_fir_packed_q14,
            tilt_q14[k] as i32,
            lf_shp_q14[k],
            gains_q16[k],
            lambda_q10,
            offset_q10,
            ps_common.subfr_length as i32,
            subfr_nsq,
            ps_common.shaping_lpc_order as i32,
            ps_common.predict_lpc_order as i32,
            ps_common.warping_q16 as i32,
            ps_common.n_states_delayed_decision as i32,
            &mut smpl_buf_idx,
            decision_delay,
        );

        x_ptr += ps_common.subfr_length as usize;
        pulses_ptr += ps_common.subfr_length as usize;
        xq_ptr += ps_common.subfr_length as usize;
        subfr_nsq += 1;
    }

    let mut rd_min_q10 = ps_del_dec[0].rd_q10;
    let mut winner_ind = 0;
    for k in 1..ps_common.n_states_delayed_decision as usize {
        if ps_del_dec[k].rd_q10 < rd_min_q10 {
            rd_min_q10 = ps_del_dec[k].rd_q10;
            winner_ind = k;
        }
    }

    let ps_dd = &ps_del_dec[winner_ind];
    ps_nsq.s_lf_ar_q14 = ps_dd.lf_ar_q14;
    ps_nsq.s_diff_shp_q14 = ps_dd.diff_q14;
    ps_nsq.lag_prev = pitch_l[ps_common.nb_subfr as usize - 1];

    let ltp_mem_len = ps_common.ltp_mem_length as usize;
    let frame_len = ps_common.frame_length as usize;
    ps_nsq.xq.copy_within(frame_len..frame_len + ltp_mem_len, 0);
    ps_nsq
        .s_ltp_shp_q14
        .copy_within(frame_len..frame_len + ltp_mem_len, 0);
}
