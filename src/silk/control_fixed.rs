use crate::silk::define::*;
use crate::silk::gain_quant::*;
use crate::silk::lpc_analysis::*;
use crate::silk::ltp_analysis::*;
use crate::silk::macros::*;
use crate::silk::nlsf::*;
use crate::silk::sigm::*;
use crate::silk::sigproc_fix::*;
use crate::silk::structs::*;
use crate::silk::tables::*;

#[inline(always)]
pub fn silk_find_pred_coefs_fix(
    ps_enc: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControl,
    res_pitch: &[i16],
    res_pitch_frame_idx: usize,
    x: &[i16],
    x_buf_full: &[i16],
    cond_coding: i32,
) {
    let mut inv_gains_q16 = [0i32; MAX_NB_SUBFR];
    let mut local_gains = [0i32; MAX_NB_SUBFR];
    let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
    let mut min_gain_q16: i32;
    let mut min_inv_gain_q30: i32;

    min_gain_q16 = i32::MAX >> 6;
    for i in 0..ps_enc.s_cmn.nb_subfr as usize {
        min_gain_q16 = min_gain_q16.min(ps_enc_ctrl.gains_q16[i]);
    }
    for i in 0..ps_enc.s_cmn.nb_subfr as usize {
        inv_gains_q16[i] = silk_div32_varq(min_gain_q16, ps_enc_ctrl.gains_q16[i], 16 - 2);

        inv_gains_q16[i] = inv_gains_q16[i].max(100);

        local_gains[i] = silk_div32(1 << 16, inv_gains_q16[i]);
    }

    let mut lpc_in_pre = [0i16; MAX_NB_SUBFR * MAX_LPC_ORDER + MAX_FRAME_LENGTH];

    if ps_enc.s_cmn.indices.signal_type == TYPE_VOICED as i8 {
        let mut x_xltp_q17 = [0i32; MAX_NB_SUBFR * LTP_ORDER];
        let mut xxltp_q17 = [0i32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];

        silk_find_ltp_fix(
            &mut xxltp_q17,
            &mut x_xltp_q17,
            res_pitch,
            res_pitch_frame_idx,
            &ps_enc_ctrl.pitch_l,
            ps_enc.s_cmn.subfr_length as usize,
            ps_enc.s_cmn.nb_subfr as usize,
            0,
        );

        silk_quant_ltp_gains(
            &mut ps_enc_ctrl.ltp_coef_q14,
            &mut ps_enc.s_cmn.indices.ltp_index,
            &mut ps_enc.s_cmn.indices.per_index,
            &mut ps_enc.s_cmn.sum_log_gain_q7,
            &mut ps_enc_ctrl.ltp_red_cod_gain_q7,
            &xxltp_q17,
            &x_xltp_q17,
            ps_enc.s_cmn.subfr_length as i32,
            ps_enc.s_cmn.nb_subfr as usize,
            0,
        );

        silk_ltp_scale_ctrl_fix(ps_enc, ps_enc_ctrl, cond_coding);

        let predict_lpc_order = ps_enc.s_cmn.predict_lpc_order as usize;
        let x_base_for_ltp = ps_enc.s_cmn.ltp_mem_length as usize - predict_lpc_order;
        silk_ltp_analysis_filter_fix(
            &mut lpc_in_pre,
            x_buf_full,
            x_base_for_ltp,
            &ps_enc_ctrl.ltp_coef_q14,
            &ps_enc_ctrl.pitch_l,
            &inv_gains_q16,
            ps_enc.s_cmn.subfr_length as usize,
            ps_enc.s_cmn.nb_subfr as usize,
            ps_enc.s_cmn.predict_lpc_order as usize,
        );
    } else {
        let mut x_ptr_idx = 0;
        let mut x_pre_ptr_idx = 0;
        for i in 0..ps_enc.s_cmn.nb_subfr as usize {
            silk_scale_copy_vector16(
                &mut lpc_in_pre[x_pre_ptr_idx..],
                &x[x_ptr_idx..],
                inv_gains_q16[i],
                (ps_enc.s_cmn.subfr_length + ps_enc.s_cmn.predict_lpc_order) as usize,
            );
            x_pre_ptr_idx += (ps_enc.s_cmn.subfr_length + ps_enc.s_cmn.predict_lpc_order) as usize;
            x_ptr_idx += ps_enc.s_cmn.subfr_length as usize;
        }

        ps_enc_ctrl.ltp_coef_q14.fill(0);
        ps_enc_ctrl.ltp_red_cod_gain_q7 = 0;
        ps_enc.s_cmn.sum_log_gain_q7 = 0;
        ps_enc_ctrl.ltp_scale_q14 = 0;
    }

    if ps_enc.s_cmn.first_frame_after_reset != 0 {
        min_inv_gain_q30 =
            (1.0f32 / MAX_PREDICTION_POWER_GAIN_AFTER_RESET * 1073741824.0f32) as i32;
    } else {
        min_inv_gain_q30 = silk_log2lin(silk_smlawb(
            16 << 7,
            ps_enc_ctrl.ltp_red_cod_gain_q7,
            (1.0f32 / 3.0f32 * 65536.0f32) as i32,
        ));
        min_inv_gain_q30 = silk_div32_varq(
            min_inv_gain_q30,
            silk_smulww(
                MAX_PREDICTION_POWER_GAIN as i32,
                silk_smlawb(
                    (0.25f32 * 262144.0f32) as i32,
                    (0.75f32 * 262144.0f32) as i32,
                    ps_enc_ctrl.coding_quality_q14,
                ),
            ),
            14,
        );
    }

    silk_find_lpc_fix(
        &mut ps_enc.s_cmn,
        &mut nlsf_q15,
        &lpc_in_pre,
        min_inv_gain_q30,
    );

    silk_process_nlsfs(ps_enc, ps_enc_ctrl, &mut nlsf_q15);

    silk_residual_energy_fix(
        &mut ps_enc_ctrl.res_nrg,
        &mut ps_enc_ctrl.res_nrg_q,
        &lpc_in_pre,
        &ps_enc_ctrl.pred_coef_q12,
        &local_gains,
        ps_enc.s_cmn.subfr_length,
        ps_enc.s_cmn.nb_subfr,
        ps_enc.s_cmn.predict_lpc_order,
    );
}

pub fn silk_process_gains_fix(
    ps_enc: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControl,
    cond_coding: i32,
) {
    let ps_shape_st = &mut ps_enc.s_shape;
    let s_q15: i32;
    let inv_max_sqr_val_q16: i32;
    let mut gain_q16: i32;
    let mut gain_squared_q16: i32;
    let mut res_nrg_part: i32;

    if ps_enc.s_cmn.indices.signal_type == TYPE_VOICED as i8 {
        s_q15 = -silk_sigm_q15(silk_rshift_round(
            ps_enc_ctrl.ltp_red_cod_gain_q7 - (12.0 * 128.0) as i32,
            4,
        ));
        for k in 0..ps_enc.s_cmn.nb_subfr as usize {
            ps_enc_ctrl.gains_q16[k] =
                silk_smlawb(ps_enc_ctrl.gains_q16[k], ps_enc_ctrl.gains_q16[k], s_q15);
        }
    }

    let log2lin_arg = silk_smulwb(
        ((21.0 + 16.0 / 0.33) * 128.0) as i32 - ps_enc.s_cmn.snr_db_q7,
        21627_i32,
    );
    let log2lin_val = silk_log2lin(log2lin_arg);
    inv_max_sqr_val_q16 = silk_div32_16(log2lin_val, ps_enc.s_cmn.subfr_length);

    for k in 0..ps_enc.s_cmn.nb_subfr as usize {
        let res_nrg = ps_enc_ctrl.res_nrg[k];
        res_nrg_part = silk_smulww(res_nrg, inv_max_sqr_val_q16);
        if ps_enc_ctrl.res_nrg_q[k] > 0 {
            res_nrg_part = silk_rshift_round(res_nrg_part, ps_enc_ctrl.res_nrg_q[k]);
        } else {
            let neg_q = (-ps_enc_ctrl.res_nrg_q[k]).min(30);
            if neg_q == 0 || res_nrg_part >= (i32::MAX >> neg_q) {
                res_nrg_part = if neg_q == 0 { res_nrg_part } else { i32::MAX };
            } else {
                res_nrg_part <<= neg_q;
            }
        }
        gain_q16 = ps_enc_ctrl.gains_q16[k];
        gain_squared_q16 = silk_add_sat32(res_nrg_part, silk_smmul(gain_q16, gain_q16));
        if gain_squared_q16 < i16::MAX as i32 {
            gain_squared_q16 = silk_smlaww(res_nrg_part << 16, gain_q16, gain_q16);

            gain_q16 = silk_sqrt_approx(gain_squared_q16);
            gain_q16 = silk_min_32(gain_q16, i32::MAX >> 8);
            ps_enc_ctrl.gains_q16[k] = silk_lshift_sat32(gain_q16, 8);
        } else {
            gain_q16 = silk_sqrt_approx(gain_squared_q16);
            gain_q16 = silk_min_32(gain_q16, i32::MAX >> 16);
            ps_enc_ctrl.gains_q16[k] = silk_lshift_sat32(gain_q16, 16);
        }
    }

    for k in 0..ps_enc.s_cmn.nb_subfr as usize {
        ps_enc_ctrl.gains_unq_q16[k] = ps_enc_ctrl.gains_q16[k];
    }
    ps_enc_ctrl.last_gain_index_prev = ps_shape_st.last_gain_index;

    silk_gains_quant(
        &mut ps_enc.s_cmn.indices.gains_indices,
        &mut ps_enc_ctrl.gains_q16,
        &mut ps_shape_st.last_gain_index,
        if cond_coding == CODE_CONDITIONALLY {
            1
        } else {
            0
        },
        ps_enc.s_cmn.nb_subfr as usize,
    );

    if ps_enc.s_cmn.indices.signal_type == TYPE_VOICED as i8 {
        if ps_enc_ctrl.ltp_red_cod_gain_q7 + (ps_enc.s_cmn.input_tilt_q15 >> 8) > 128 {
            ps_enc.s_cmn.indices.quant_offset_type = 0;
        } else {
            ps_enc.s_cmn.indices.quant_offset_type = 1;
        }
    }

    let quant_offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10
        [(ps_enc.s_cmn.indices.signal_type >> 1) as usize]
        [ps_enc.s_cmn.indices.quant_offset_type as usize] as i32;

    const LAMBDA_OFFSET_Q10: i32 = 1229;
    const LAMBDA_DELAYED_DECISIONS_Q10: i32 = -50;
    const LAMBDA_SPEECH_ACT_Q18: i32 = -52428;
    const LAMBDA_INPUT_QUALITY_Q12: i32 = -409;
    const LAMBDA_CODING_QUALITY_Q12: i32 = -818;
    const LAMBDA_QUANT_OFFSET_Q16: i32 = 52429;

    ps_enc_ctrl.lambda_q10 = LAMBDA_OFFSET_Q10
        + silk_smulbb(
            LAMBDA_DELAYED_DECISIONS_Q10,
            ps_enc.s_cmn.n_states_delayed_decision,
        )
        + silk_smulwb(LAMBDA_SPEECH_ACT_Q18, ps_enc.s_cmn.speech_activity_q8)
        + silk_smulwb(LAMBDA_INPUT_QUALITY_Q12, ps_enc_ctrl.input_quality_q14)
        + silk_smulwb(LAMBDA_CODING_QUALITY_Q12, ps_enc_ctrl.coding_quality_q14)
        + silk_smulwb(LAMBDA_QUANT_OFFSET_Q16, quant_offset_q10);
}

pub fn silk_ltp_scale_ctrl_fix(
    ps_enc: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControl,
    cond_coding: i32,
) {
    if cond_coding == CODE_INDEPENDENTLY {
        let mut round_loss = ps_enc.s_cmn.packet_loss_perc * ps_enc.s_cmn.n_frames_per_packet;
        if ps_enc.s_cmn.lbrr_flag != 0 {
            round_loss = 2 + silk_smulbb(round_loss, round_loss) / 100;
        }
        ps_enc.s_cmn.indices.ltp_scale_index =
            (silk_smulbb(ps_enc_ctrl.ltp_red_cod_gain_q7, round_loss)
                > silk_log2lin(128 * 7 + 2900 - ps_enc.s_cmn.snr_db_q7)) as i8;
        ps_enc.s_cmn.indices.ltp_scale_index +=
            (silk_smulbb(ps_enc_ctrl.ltp_red_cod_gain_q7, round_loss)
                > silk_log2lin(128 * 7 + 3900 - ps_enc.s_cmn.snr_db_q7)) as i8;
    } else {
        ps_enc.s_cmn.indices.ltp_scale_index = 0;
    }
    ps_enc_ctrl.ltp_scale_q14 =
        SILK_LTP_SCALES_TABLE_Q14[ps_enc.s_cmn.indices.ltp_scale_index as usize] as i32;
}
