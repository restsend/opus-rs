use crate::silk::decoder_structs::{SilkDecoderControl, SilkDecoderState};
use crate::silk::define::*;
use crate::silk::gain_quant::silk_gains_dequant;
use crate::silk::nlsf::silk_nlsf2a;
use crate::silk::nlsf_decode::silk_nlsf_decode;
use crate::silk::sigproc_fix::silk_bwexpander;
use crate::silk::tables::SILK_LTP_SCALES_TABLE_Q14;
use crate::silk::tables::SILK_LTP_VQ_PTRS_Q7;

const BWE_AFTER_LOSS_Q16: i32 = 64738;

fn silk_decode_pitch(
    lag_index: i16,
    contour_index: i8,
    pitch_l: &mut [i32],
    fs_khz: i32,
    nb_subfr: i32,
) {
    let min_lag = PITCH_EST_MIN_LAG_MS * fs_khz;
    let max_lag = PITCH_EST_MAX_LAG_MS * fs_khz;
    let lag = min_lag + lag_index as i32;

    let contour_index = contour_index as usize;

    for k in 0..nb_subfr as usize {
        let delta = if fs_khz == 8 {
            if nb_subfr == PE_MAX_NB_SUBFR as i32 {
                crate::silk::tables::SILK_CB_LAGS_STAGE2[k % 4][contour_index % PE_NB_CBKS_STAGE2_EXT] as i32
            } else {
                crate::silk::tables::SILK_CB_LAGS_STAGE2_10_MS[k % 2][contour_index % PE_NB_CBKS_STAGE2_10MS] as i32
            }
        } else {
            if nb_subfr == PE_MAX_NB_SUBFR as i32 {
                crate::silk::tables::SILK_CB_LAGS_STAGE3[k % 4][contour_index % PE_NB_CBKS_STAGE3_MAX] as i32
            } else {
                crate::silk::tables::SILK_CB_LAGS_STAGE3_10_MS[k % 2][contour_index % PE_NB_CBKS_STAGE3_10MS] as i32
            }
        };
        pitch_l[k] = (lag + delta).max(min_lag).min(max_lag);
    }
}

pub fn silk_decode_parameters(
    ps_dec: &mut SilkDecoderState,
    ps_dec_ctrl: &mut SilkDecoderControl,
    cond_coding: i32,
) {
    let mut p_nlsf_q15: [i16; MAX_LPC_ORDER] = [0; MAX_LPC_ORDER];
    let mut p_nlsf0_q15: [i16; MAX_LPC_ORDER] = [0; MAX_LPC_ORDER];

    silk_gains_dequant(
        &mut ps_dec_ctrl.gains_q16,
        &ps_dec.indices.gains_indices,
        &mut ps_dec.last_gain_index,
        if cond_coding == CODE_CONDITIONALLY { 1 } else { 0 },
        ps_dec.nb_subfr as usize,
    );

    silk_nlsf_decode(
        &mut p_nlsf_q15,
        &ps_dec.indices.nlsf_indices,
        ps_dec.ps_nlsf_cb.unwrap(),
    );

    silk_nlsf2a(
        &mut ps_dec_ctrl.pred_coef_q12[1],
        &p_nlsf_q15,
        ps_dec.lpc_order as usize,
    );

    if ps_dec.first_frame_after_reset == 1 {
        ps_dec.indices.nlsf_interp_coef_q2 = 4;
    }

    if ps_dec.indices.nlsf_interp_coef_q2 < 4 {

        for i in 0..ps_dec.lpc_order as usize {
            p_nlsf0_q15[i] = ps_dec.prev_nlsf_q15[i]
                + (((ps_dec.indices.nlsf_interp_coef_q2 as i32)
                    * (p_nlsf_q15[i] as i32 - ps_dec.prev_nlsf_q15[i] as i32))
                    >> 2) as i16;
        }

        silk_nlsf2a(
            &mut ps_dec_ctrl.pred_coef_q12[0],
            &p_nlsf0_q15,
            ps_dec.lpc_order as usize,
        );
    } else {

        ps_dec_ctrl.pred_coef_q12[0] = ps_dec_ctrl.pred_coef_q12[1];
    }

    ps_dec.prev_nlsf_q15.copy_from_slice(&p_nlsf_q15);

    if ps_dec.loss_cnt > 0 {
        silk_bwexpander(
            &mut ps_dec_ctrl.pred_coef_q12[0],
            ps_dec.lpc_order as usize,
            BWE_AFTER_LOSS_Q16,
        );
        silk_bwexpander(
            &mut ps_dec_ctrl.pred_coef_q12[1],
            ps_dec.lpc_order as usize,
            BWE_AFTER_LOSS_Q16,
        );
    }

    if ps_dec.indices.signal_type == TYPE_VOICED as i8 {

        silk_decode_pitch(
            ps_dec.indices.lag_index,
            ps_dec.indices.contour_index,
            &mut ps_dec_ctrl.pitch_l,
            ps_dec.fs_khz,
            ps_dec.nb_subfr,
        );

        let cbk_ptr_q7 = SILK_LTP_VQ_PTRS_Q7[ps_dec.indices.per_index as usize];

        for k in 0..ps_dec.nb_subfr as usize {
            let ix = ps_dec.indices.ltp_index[k] as usize;
            for i in 0..LTP_ORDER {
                ps_dec_ctrl.ltp_coef_q14[k * LTP_ORDER + i] =
                    ((cbk_ptr_q7[ix][i] as i32) << 7) as i16;
            }
        }

        let ix = ps_dec.indices.ltp_scale_index as usize;
        ps_dec_ctrl.ltp_scale_q14 = SILK_LTP_SCALES_TABLE_Q14[ix] as i32;
    } else {
        for i in 0..ps_dec.nb_subfr as usize {
            ps_dec_ctrl.pitch_l[i] = 0;
        }
        for i in 0..(LTP_ORDER * ps_dec.nb_subfr as usize) {
            ps_dec_ctrl.ltp_coef_q14[i] = 0;
        }
        ps_dec.indices.per_index = 0;
        ps_dec_ctrl.ltp_scale_q14 = 0;
    }
}
