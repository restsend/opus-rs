//! libopus `silk/float/find_pred_coefs_FLP.c`, `LTP_scale_ctrl_FLP.c` and
//! `process_gains_FLP.c`: the LPC and LTP coefficients of a frame and its
//! quantized gains.

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::silk::define::*;
use crate::silk::float::lpc_analysis_flp::{silk_find_lpc_flp, silk_residual_energy_flp};
use crate::silk::float::ltp_analysis_flp::{silk_find_ltp_flp, silk_ltp_analysis_filter_flp};
use crate::silk::float::sigproc_flp::*;
use crate::silk::float::structs_flp::SilkEncoderControlFlp;
use crate::silk::float::wrappers_flp::{silk_process_nlsfs_flp, silk_quant_ltp_gains_flp};
use crate::silk::gain_quant::silk_gains_quant;
use crate::silk::log2lin::silk_log2lin;
use crate::silk::structs::SilkEncoderState;
use crate::silk::tables::{SILK_LTP_SCALES_TABLE_Q14, SILK_QUANTIZATION_OFFSETS_Q10};
use crate::silk::tuning_parameters::*;

/// libopus `silk_find_pred_coefs_FLP`: LTP coefficients (voiced frames),
/// quantized NLSFs with their prediction coefficients, and the residual
/// energies. `res_pitch` is the pitch residual and `x` the input, both from
/// `ltp_mem_length` samples before the frame; `x_frame` is where the frame
/// starts in both.
pub fn silk_find_pred_coefs_flp(
    ps_enc: &mut SilkEncoderState,
    ctrl: &mut SilkEncoderControlFlp,
    res_pitch: &[f32],
    x: &[f32],
    x_frame: usize,
    cond_coding: i32,
) {
    let nb_subfr = ps_enc.s_cmn.nb_subfr as usize;
    let subfr_length = ps_enc.s_cmn.subfr_length as usize;
    let order = ps_enc.s_cmn.predict_lpc_order as usize;
    let mut inv_gains = [0.0f32; MAX_NB_SUBFR];
    for (inv, &g) in inv_gains.iter_mut().zip(&ctrl.gains[..nb_subfr]) {
        *inv = 1.0 / g;
    }

    let mut lpc_in_pre = [0.0f32; MAX_NB_SUBFR * MAX_LPC_ORDER + MAX_FRAME_LENGTH];
    if i32::from(ps_enc.s_cmn.indices.signal_type) == TYPE_VOICED {
        let mut xx_ltp = [0.0f32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];
        let mut x_x_ltp = [0.0f32; MAX_NB_SUBFR * LTP_ORDER];
        silk_find_ltp_flp(&mut xx_ltp, &mut x_x_ltp, res_pitch, x_frame, &ctrl.pitch_l, subfr_length, nb_subfr);

        let cmn = &mut ps_enc.s_cmn;
        ctrl.ltp_red_cod_gain = silk_quant_ltp_gains_flp(
            &mut ctrl.ltp_coef,
            &mut cmn.indices.ltp_index,
            &mut cmn.indices.per_index,
            &mut cmn.sum_log_gain_q7,
            &xx_ltp,
            &x_x_ltp,
            subfr_length as i32,
            nb_subfr,
        );
        silk_ltp_scale_ctrl_flp(ps_enc, ctrl, cond_coding);

        silk_ltp_analysis_filter_flp(
            &mut lpc_in_pre,
            x,
            x_frame - order,
            &ctrl.ltp_coef,
            &ctrl.pitch_l,
            &inv_gains,
            subfr_length,
            nb_subfr,
            order,
        );
    } else {
        // The subframes with their preceding samples, scaled by the inverse
        // gains.
        for (i, &inv) in inv_gains[..nb_subfr].iter().enumerate() {
            let src = x_frame - order + i * subfr_length;
            let dst = i * (subfr_length + order);
            silk_scale_copy_vector_flp(
                &mut lpc_in_pre[dst..dst + subfr_length + order],
                &x[src..src + subfr_length + order],
                inv,
            );
        }
        ctrl.ltp_coef[..nb_subfr * LTP_ORDER].fill(0.0);
        ctrl.ltp_red_cod_gain = 0.0;
        ps_enc.s_cmn.sum_log_gain_q7 = 0;
    }

    // Limit on the total predictive coding gain.
    let min_inv_gain = if ps_enc.s_cmn.first_frame_after_reset != 0 {
        1.0 / MAX_PREDICTION_POWER_GAIN_AFTER_RESET
    } else {
        let g = 2.0f64.powf(f64::from(ctrl.ltp_red_cod_gain / 3.0)) as f32 / MAX_PREDICTION_POWER_GAIN;
        g / (0.25 + 0.75 * ctrl.coding_quality)
    };

    let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
    silk_find_lpc_flp(&mut ps_enc.s_cmn, &mut nlsf_q15, &lpc_in_pre, min_inv_gain);
    silk_process_nlsfs_flp(ps_enc, &mut ctrl.pred_coef, &mut nlsf_q15);
    silk_residual_energy_flp(
        &mut ctrl.res_nrg,
        &lpc_in_pre,
        &ctrl.pred_coef,
        &ctrl.gains,
        subfr_length,
        nb_subfr,
        order,
    );
    // For the interpolation in the next frame.
    ps_enc.s_cmn.prev_nlsf_q15 = nlsf_q15;
}

/// libopus `silk_LTP_scale_ctrl_FLP`: the LTP scaling of a frame coded
/// independently, from the LTP gain and the expected packet loss.
pub fn silk_ltp_scale_ctrl_flp(ps_enc: &mut SilkEncoderState, ctrl: &mut SilkEncoderControlFlp, cond_coding: i32) {
    let cmn = &mut ps_enc.s_cmn;
    if cond_coding == CODE_INDEPENDENTLY {
        let mut round_loss = cmn.packet_loss_perc * cmn.n_frames_per_packet;
        if cmn.lbrr_flag != 0 {
            // LBRR reduces the effective loss, never below 2%.
            round_loss = 2 + round_loss as i16 as i32 * round_loss as i16 as i32 / 100;
        }
        // silk_SMULBB of the float gain: both truncated to 16 bits.
        let g = (ctrl.ltp_red_cod_gain as i16 as i32) * (round_loss as i16 as i32);
        let mut idx = i8::from(g > silk_log2lin(2900 - cmn.snr_db_q7));
        idx += i8::from(g > silk_log2lin(3900 - cmn.snr_db_q7));
        cmn.indices.ltp_scale_index = idx;
    } else {
        cmn.indices.ltp_scale_index = 0;
    }
    ctrl.ltp_scale = f32::from(SILK_LTP_SCALES_TABLE_Q14[cmn.indices.ltp_scale_index as usize]) / 16384.0;
}

/// libopus `silk_process_gains_FLP`: gain reduction for high LTP gains, a
/// soft limit on the residual, the quantized gains, the quantization offset
/// of voiced frames and the rate-distortion trade-off `lambda`.
pub fn silk_process_gains_flp(ps_enc: &mut SilkEncoderState, ctrl: &mut SilkEncoderControlFlp, cond_coding: i32) {
    let cmn = &mut ps_enc.s_cmn;
    let nb_subfr = cmn.nb_subfr as usize;
    let voiced = i32::from(cmn.indices.signal_type) == TYPE_VOICED;

    // Lower gains for a high LTP coding gain.
    if voiced {
        let s = 1.0 - 0.5 * silk_sigmoid(0.25 * (ctrl.ltp_red_cod_gain - 12.0));
        for g in &mut ctrl.gains[..nb_subfr] {
            *g *= s;
        }
    }

    // Soft limit on the ratio of residual energy and squared gains.
    let inv_max_sqr_val = (2.0f64.powf(f64::from(0.33f32 * (21.0 - cmn.snr_db_q7 as f32 * (1.0 / 128.0))))
        / f64::from(cmn.subfr_length)) as f32;
    for k in 0..nb_subfr {
        let gain = ctrl.gains[k];
        let gain = f64::from(gain * gain + ctrl.res_nrg[k] * inv_max_sqr_val).sqrt() as f32;
        ctrl.gains[k] = gain.min(32767.0);
    }

    let mut gains_q16 = [0i32; MAX_NB_SUBFR];
    for (q, &g) in gains_q16.iter_mut().zip(&ctrl.gains[..nb_subfr]) {
        *q = (g * 65536.0) as i32;
    }
    ctrl.gains_unq_q16 = gains_q16;
    ctrl.last_gain_index_prev = ps_enc.s_shape.last_gain_index;
    silk_gains_quant(
        &mut cmn.indices.gains_indices,
        &mut gains_q16,
        &mut ps_enc.s_shape.last_gain_index,
        (cond_coding == CODE_CONDITIONALLY) as i32,
        nb_subfr,
    );
    for (g, &q) in ctrl.gains[..nb_subfr].iter_mut().zip(&gains_q16) {
        *g = q as f32 / 65536.0;
    }

    // A larger offset for voiced frames with a low LTP gain or a high tilt.
    if voiced {
        cmn.indices.quant_offset_type =
            i8::from(!(ctrl.ltp_red_cod_gain + cmn.input_tilt_q15 as f32 * (1.0 / 32768.0) > 1.0));
    }

    let quant_offset = f32::from(
        SILK_QUANTIZATION_OFFSETS_Q10[(cmn.indices.signal_type >> 1) as usize]
            [cmn.indices.quant_offset_type as usize],
    ) / 1024.0;
    ctrl.lambda = LAMBDA_OFFSET
        + LAMBDA_DELAYED_DECISIONS * cmn.n_states_delayed_decision as f32
        + LAMBDA_SPEECH_ACT * cmn.speech_activity_q8 as f32 * (1.0 / 256.0)
        + LAMBDA_INPUT_QUALITY * ctrl.input_quality
        + LAMBDA_CODING_QUALITY * ctrl.coding_quality
        + LAMBDA_QUANT_OFFSET * quant_offset;
}
