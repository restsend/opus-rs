//! libopus `silk/float/wrappers_FLP.c`: the float encoder's calls into the
//! shared fixed-point code (NLSF conversion and quantization, LTP gain
//! quantization, the noise shaping quantizers).

use crate::silk::define::*;
use crate::silk::float::sigproc_flp::silk_float2int;
use crate::silk::float::structs_flp::SilkEncoderControlFlp;
use crate::silk::gain_quant::silk_quant_ltp_gains;
use crate::silk::nlsf::{silk_a2nlsf, silk_nlsf2a, silk_process_nlsfs};
use crate::silk::nsq::silk_nsq;
use crate::silk::nsq_del_dec::silk_nsq_del_dec;
use crate::silk::structs::*;
use crate::silk::tables::SILK_LTP_SCALES_TABLE_Q14;

/// libopus `silk_A2NLSF_FLP`.
pub fn silk_a2nlsf_flp(nlsf_q15: &mut [i16], ar: &[f32]) {
    let mut a_q16 = [0i32; MAX_LPC_ORDER];
    for (q, &a) in a_q16.iter_mut().zip(ar) {
        *q = silk_float2int(a * 65536.0);
    }
    silk_a2nlsf(nlsf_q15, &mut a_q16, ar.len());
}

/// libopus `silk_NLSF2A_FLP`.
pub fn silk_nlsf2a_flp(ar: &mut [f32], nlsf_q15: &[i16]) {
    let order = ar.len();
    let mut a_q12 = [0i16; MAX_LPC_ORDER];
    silk_nlsf2a(&mut a_q12, nlsf_q15, order);
    for (a, &q) in ar.iter_mut().zip(&a_q12) {
        *a = f32::from(q) * (1.0 / 4096.0);
    }
}

/// libopus `silk_process_NLSFs_FLP`: quantizes `nlsf_q15` and sets the
/// interpolated and the final prediction coefficients.
pub fn silk_process_nlsfs_flp(
    ps_enc: &mut SilkEncoderState,
    pred_coef: &mut [[f32; MAX_LPC_ORDER]; 2],
    nlsf_q15: &mut [i16],
) {
    let mut pred_coef_q12 = [[0i16; MAX_LPC_ORDER]; 2];
    silk_process_nlsfs(ps_enc, &mut pred_coef_q12, nlsf_q15);
    let order = ps_enc.s_cmn.predict_lpc_order as usize;
    for (p, q) in pred_coef.iter_mut().zip(&pred_coef_q12) {
        for (a, &c) in p[..order].iter_mut().zip(&q[..order]) {
            *a = f32::from(c) * (1.0 / 4096.0);
        }
    }
}

/// libopus `silk_quant_LTP_gains_FLP`: quantizes the LTP gains `b` and
/// returns the LTP prediction gain in dB.
#[allow(clippy::too_many_arguments)]
pub fn silk_quant_ltp_gains_flp(
    b: &mut [f32; MAX_NB_SUBFR * LTP_ORDER],
    cbk_index: &mut [i8],
    periodicity_index: &mut i8,
    sum_log_gain_q7: &mut i32,
    xx: &[f32],
    x_x: &[f32],
    subfr_len: i32,
    nb_subfr: usize,
) -> f32 {
    let mut b_q14 = [0i16; MAX_NB_SUBFR * LTP_ORDER];
    let mut xx_q17 = [0i32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];
    let mut x_x_q17 = [0i32; MAX_NB_SUBFR * LTP_ORDER];
    for (q, &v) in xx_q17.iter_mut().zip(&xx[..nb_subfr * LTP_ORDER * LTP_ORDER]) {
        *q = silk_float2int(v * 131072.0);
    }
    for (q, &v) in x_x_q17.iter_mut().zip(&x_x[..nb_subfr * LTP_ORDER]) {
        *q = silk_float2int(v * 131072.0);
    }
    let mut pred_gain_db_q7 = 0;
    silk_quant_ltp_gains(
        &mut b_q14,
        cbk_index,
        periodicity_index,
        sum_log_gain_q7,
        &mut pred_gain_db_q7,
        &xx_q17,
        &x_x_q17,
        subfr_len,
        nb_subfr,
        0,
    );
    for (v, &q) in b[..nb_subfr * LTP_ORDER].iter_mut().zip(&b_q14) {
        *v = f32::from(q) * (1.0 / 16384.0);
    }
    pred_gain_db_q7 as f32 * (1.0 / 128.0)
}

/// libopus `silk_NSQ_wrapper_FLP`: runs the noise shaping quantizer on the
/// frame `x` with the control converted to its fixed-point formats;
/// returns the seed of the delayed-decision quantizer's winner, or the
/// seed unchanged.
pub fn silk_nsq_wrapper_flp(
    s_cmn: &SilkEncoderStateCommon,
    nsq: &mut SilkNSQState,
    indices: &SideInfoIndices,
    ctrl: &SilkEncoderControlFlp,
    pulses: &mut [i8],
    x: &[f32],
) -> i8 {
    let nb_subfr = s_cmn.nb_subfr as usize;
    let mut ar_q13 = [0i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER];
    for i in 0..nb_subfr {
        for j in 0..s_cmn.shaping_lpc_order as usize {
            let k = i * MAX_SHAPE_LPC_ORDER + j;
            ar_q13[k] = silk_float2int(ctrl.ar[k] * 8192.0) as i16;
        }
    }
    let mut lf_shp_q14 = [0i32; MAX_NB_SUBFR];
    let mut tilt_q14 = [0i32; MAX_NB_SUBFR];
    let mut harm_shape_gain_q14 = [0i32; MAX_NB_SUBFR];
    for i in 0..nb_subfr {
        lf_shp_q14[i] = (silk_float2int(ctrl.lf_ar_shp[i] * 16384.0) << 16)
            | i32::from(silk_float2int(ctrl.lf_ma_shp[i] * 16384.0) as u16);
        tilt_q14[i] = silk_float2int(ctrl.tilt[i] * 16384.0);
        harm_shape_gain_q14[i] = silk_float2int(ctrl.harm_shape_gain[i] * 16384.0);
    }
    let lambda_q10 = silk_float2int(ctrl.lambda * 1024.0);

    let mut ltp_coef_q14 = [0i16; LTP_ORDER * MAX_NB_SUBFR];
    for (q, &v) in ltp_coef_q14.iter_mut().zip(&ctrl.ltp_coef[..nb_subfr * LTP_ORDER]) {
        *q = silk_float2int(v * 16384.0) as i16;
    }
    // Both halves, as the quantizers take them back to back.
    let mut pred_coef_q12 = [0i16; 2 * MAX_LPC_ORDER];
    for j in 0..2 {
        for i in 0..s_cmn.predict_lpc_order as usize {
            pred_coef_q12[j * MAX_LPC_ORDER + i] = silk_float2int(ctrl.pred_coef[j][i] * 4096.0) as i16;
        }
    }
    let mut gains_q16 = [0i32; MAX_NB_SUBFR];
    for (q, &g) in gains_q16.iter_mut().zip(&ctrl.gains[..nb_subfr]) {
        *q = silk_float2int(g * 65536.0);
    }
    let ltp_scale_q14 = if i32::from(indices.signal_type) == TYPE_VOICED {
        i32::from(SILK_LTP_SCALES_TABLE_Q14[indices.ltp_scale_index as usize])
    } else {
        0
    };

    let frame_length = s_cmn.frame_length as usize;
    let mut x16 = [0i16; MAX_FRAME_LENGTH];
    for (q, &v) in x16[..frame_length].iter_mut().zip(x) {
        *q = silk_float2int(v) as i16;
    }

    if s_cmn.n_states_delayed_decision > 1 || s_cmn.warping_q16 > 0 {
        silk_nsq_del_dec(
            s_cmn,
            nsq,
            indices,
            &x16,
            pulses,
            &pred_coef_q12,
            &ltp_coef_q14,
            &ar_q13,
            &harm_shape_gain_q14,
            &tilt_q14,
            &lf_shp_q14,
            &gains_q16,
            &ctrl.pitch_l,
            lambda_q10,
            ltp_scale_q14,
        ) as i8
    } else {
        silk_nsq(
            s_cmn,
            nsq,
            indices,
            &x16,
            pulses,
            &pred_coef_q12,
            &ltp_coef_q14,
            &ar_q13,
            &harm_shape_gain_q14,
            &tilt_q14,
            &lf_shp_q14,
            &gains_q16,
            &ctrl.pitch_l,
            lambda_q10,
            ltp_scale_q14,
        );
        indices.seed
    }
}
