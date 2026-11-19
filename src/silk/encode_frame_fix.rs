//! libopus `silk/fixed/encode_frame_FIX.c`: the analysis, noise shaping
//! quantization, rate control and LBRR coding of one frame by the
//! fixed-point encoder (the `silk-fixed` feature).

use crate::range_coder::RangeCoder;
use crate::silk::control_fixed::*;
use crate::silk::define::*;
use crate::silk::encode_indices::*;
use crate::silk::encode_pulses::*;
use crate::silk::gain_quant::{silk_gains_dequant, silk_gains_id, silk_gains_quant};
use crate::silk::macros::*;
use crate::silk::noise_shape_analysis::*;
use crate::silk::nsq::*;
use crate::silk::nsq_del_dec::*;
use crate::silk::pitch_analysis::*;
use crate::silk::structs::*;

/// The prefill part of libopus's `silk_encode_frame_FIX`: `frame` goes into
/// the analysis buffer behind the look-ahead and the buffer moves on,
/// coding nothing. `frame_length` is the prefill's.
pub fn silk_prefill_frame_fix(ps_enc: &mut SilkEncoderState, frame: &[i16]) {
    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let la_shape = 5 * ps_enc.s_cmn.fs_khz as usize;
    let x_frame = ps_enc.s_cmn.ltp_mem_length as usize;
    let dst = x_frame + la_shape;
    ps_enc.s_cmn.x_buf[dst..dst + frame_length].copy_from_slice(&frame[..frame_length]);
    ps_enc
        .s_cmn
        .x_buf
        .copy_within(frame_length..frame_length + x_frame + la_shape, 0);
}

pub fn silk_encode_frame_fix(
    ps_enc: &mut SilkEncoderState,
    input: &[i16],
    rc: &mut RangeCoder,
    pn_bytes_out: &mut i32,
    cond_coding: i32,
    max_bits: i32,
    use_cbr: i32,
) -> i32 {
    let mut s_enc_ctrl = SilkEncoderControl::default();

    ps_enc.s_cmn.indices.seed = (ps_enc.s_cmn.frame_counter & 3) as i8;
    ps_enc.s_cmn.frame_counter += 1;

    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let ltp_mem_length = ps_enc.s_cmn.ltp_mem_length as usize;
    let la_shape = ps_enc.s_cmn.la_shape as usize;

    let x_frame_idx = ltp_mem_length;

    let la_shape_max = 5 * ps_enc.s_cmn.fs_khz as usize;
    let new_samples_idx = x_frame_idx + la_shape_max;
    ps_enc.s_cmn.x_buf[new_samples_idx..new_samples_idx + frame_length]
        .copy_from_slice(&input[..frame_length]);

    let x_buf_copy = ps_enc.s_cmn.x_buf;

    let mut res_pitch = [0i16; LA_PITCH_MAX + MAX_FRAME_LENGTH + LTP_MEM_LENGTH_MS * MAX_FS_KHZ];
    let res_pitch_frame_idx = ltp_mem_length;

    silk_find_pitch_lags_fix(ps_enc, &mut s_enc_ctrl, &mut res_pitch, &x_buf_copy, 0);

    let x_tmp = &x_buf_copy[x_frame_idx - la_shape..];
    silk_noise_shape_analysis_fix(
        ps_enc,
        &mut s_enc_ctrl,
        &res_pitch[res_pitch_frame_idx..],
        x_tmp,
    );

    let predict_lpc_order = ps_enc.s_cmn.predict_lpc_order as usize;
    let x_tmp_frame = &x_buf_copy[x_frame_idx - predict_lpc_order..];
    silk_find_pred_coefs_fix(
        ps_enc,
        &mut s_enc_ctrl,
        &res_pitch,
        res_pitch_frame_idx,
        x_tmp_frame,
        &x_buf_copy,
        cond_coding,
    );

    silk_process_gains_fix(ps_enc, &mut s_enc_ctrl, cond_coding);

    /****************************************/
    /* Low Bitrate Redundant Encoding       */
    /****************************************/
    // Port of silk_LBRR_encode_FIX: runs on the frame-entry NSQ state, i.e.
    // after the gains are quantized and before the rate-control loop below.
    silk_lbrr_encode(ps_enc, &s_enc_ctrl, x_frame_idx, cond_coding);

    let max_iter = 6;
    let mut gain_mult_q8: i32 = 256;
    let mut found_lower = false;
    let mut found_upper = false;
    #[allow(unused_assignments)]
    let mut n_bits: i32 = 0;
    let mut n_bits_lower: i32 = 0;
    let mut n_bits_upper: i32 = 0;
    let mut gain_mult_lower: i32 = 0;
    let mut gain_mult_upper: i32 = 0;
    let mut gains_id: i32 =
        silk_gains_id(&ps_enc.s_cmn.indices.gains_indices, ps_enc.s_cmn.nb_subfr);
    let mut gains_id_lower: i32 = -1;
    let mut gains_id_upper: i32 = -1;

    let bits_margin = if use_cbr != 0 { 5 } else { max_bits / 4 };

    let rc_copy = rc.clone();
    let nsq_copy = ps_enc.s_nsq;
    let seed_copy = ps_enc.s_cmn.indices.seed;
    let ec_prev_lag_index_copy = ps_enc.s_cmn.ec_prev_lag_index;
    let ec_prev_signal_type_copy = ps_enc.s_cmn.ec_prev_signal_type;
    let mut rc_copy2: Option<RangeCoder> = None;
    let mut nsq_copy2: Option<SilkNSQState> = None;
    let mut ec_buf_copy = [0u8; 1275];
    let mut last_gain_index_copy2: i8 = 0;

    let mut gain_lock = [false; MAX_NB_SUBFR];
    let mut best_gain_mult = [256i32; MAX_NB_SUBFR];
    let mut best_sum = [i32::MAX; MAX_NB_SUBFR];

    for iter in 0..=max_iter {
        if gains_id == gains_id_lower {
            n_bits = n_bits_lower;
        } else if gains_id == gains_id_upper {
            n_bits = n_bits_upper;
        } else {
            if iter > 0 {
                *rc = rc_copy.clone();
                ps_enc.s_nsq = nsq_copy;
                ps_enc.s_cmn.indices.seed = seed_copy;
                ps_enc.s_cmn.ec_prev_lag_index = ec_prev_lag_index_copy;
                ps_enc.s_cmn.ec_prev_signal_type = ec_prev_signal_type_copy;
            }

            let mut pred_coef_q12_flat = [0i16; 2 * MAX_LPC_ORDER];
            pred_coef_q12_flat[..MAX_LPC_ORDER].copy_from_slice(&s_enc_ctrl.pred_coef_q12[0]);
            pred_coef_q12_flat[MAX_LPC_ORDER..].copy_from_slice(&s_enc_ctrl.pred_coef_q12[1]);

            if ps_enc.s_cmn.n_states_delayed_decision > 1 {
                let winner_seed = silk_nsq_del_dec(
                    &ps_enc.s_cmn,
                    &mut ps_enc.s_nsq,
                    &ps_enc.s_cmn.indices,
                    &ps_enc.s_cmn.x_buf[x_frame_idx..],
                    &mut ps_enc.pulses,
                    &pred_coef_q12_flat,
                    &s_enc_ctrl.ltp_coef_q14,
                    &s_enc_ctrl.ar_q13,
                    &s_enc_ctrl.harm_shape_gain_q14,
                    &s_enc_ctrl.tilt_q14,
                    &s_enc_ctrl.lf_shp_q14,
                    &s_enc_ctrl.gains_q16,
                    &s_enc_ctrl.pitch_l,
                    s_enc_ctrl.lambda_q10,
                    s_enc_ctrl.ltp_scale_q14,
                );

                ps_enc.s_cmn.indices.seed = winner_seed as i8;
            } else {
                silk_nsq(
                    &ps_enc.s_cmn,
                    &mut ps_enc.s_nsq,
                    &ps_enc.s_cmn.indices,
                    &ps_enc.s_cmn.x_buf[x_frame_idx..],
                    &mut ps_enc.pulses,
                    &pred_coef_q12_flat,
                    &s_enc_ctrl.ltp_coef_q14,
                    &s_enc_ctrl.ar_q13,
                    &s_enc_ctrl.harm_shape_gain_q14,
                    &s_enc_ctrl.tilt_q14,
                    &s_enc_ctrl.lf_shp_q14,
                    &s_enc_ctrl.gains_q16,
                    &s_enc_ctrl.pitch_l,
                    s_enc_ctrl.lambda_q10,
                    s_enc_ctrl.ltp_scale_q14,
                );
            }

            if iter == max_iter && !found_lower {
                rc_copy2 = Some(rc.clone());
            }

            silk_encode_indices(
                ps_enc,
                rc,
                ps_enc.s_cmn.n_frames_encoded as usize,
                false,
                cond_coding,
            );

            silk_encode_pulses(
                rc,
                ps_enc.s_cmn.indices.signal_type as i32,
                ps_enc.s_cmn.indices.quant_offset_type as i32,
                &mut ps_enc.pulses,
                ps_enc.s_cmn.frame_length as usize,
            );

            n_bits = rc.tell();

            if iter == max_iter && !found_lower && n_bits > max_bits {
                if let Some(rc_c2) = &rc_copy2 {
                    *rc = rc_c2.clone();
                }

                ps_enc.s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
                for i in 0..ps_enc.s_cmn.nb_subfr as usize {
                    ps_enc.s_cmn.indices.gains_indices[i] = 4;
                }
                if cond_coding != CODE_CONDITIONALLY {
                    ps_enc.s_cmn.indices.gains_indices[0] = s_enc_ctrl.last_gain_index_prev;
                }
                ps_enc.s_cmn.ec_prev_lag_index = ec_prev_lag_index_copy;
                ps_enc.s_cmn.ec_prev_signal_type = ec_prev_signal_type_copy;

                ps_enc.pulses.fill(0);

                silk_encode_indices(
                    ps_enc,
                    rc,
                    ps_enc.s_cmn.n_frames_encoded as usize,
                    false,
                    cond_coding,
                );
                silk_encode_pulses(
                    rc,
                    ps_enc.s_cmn.indices.signal_type as i32,
                    ps_enc.s_cmn.indices.quant_offset_type as i32,
                    &mut ps_enc.pulses,
                    ps_enc.s_cmn.frame_length as usize,
                );

                n_bits = rc.tell();
            }

            if use_cbr == 0 && iter == 0 && n_bits <= max_bits {
                break;
            }
        }

        if iter == max_iter {
            if found_lower && (gains_id == gains_id_lower || n_bits > max_bits) {
                if let Some(rc_c2) = &rc_copy2 {
                    *rc = rc_c2.clone();
                    let offs = rc.offs as usize;
                    rc.buf[..offs].copy_from_slice(&ec_buf_copy[..offs]);
                }
                if let Some(nsq_c2) = &nsq_copy2 {
                    ps_enc.s_nsq = *nsq_c2;
                }
                ps_enc.s_shape.last_gain_index = last_gain_index_copy2;
            }
            break;
        }

        if n_bits > max_bits {
            if !found_lower && iter >= 2 {
                s_enc_ctrl.lambda_q10 =
                    silk_add_rshift32(s_enc_ctrl.lambda_q10, s_enc_ctrl.lambda_q10, 1);
                found_upper = false;
                gains_id_upper = -1;
            } else {
                found_upper = true;
                n_bits_upper = n_bits;
                gain_mult_upper = gain_mult_q8;
                gains_id_upper = gains_id;
            }
        } else if n_bits < max_bits - bits_margin {
            found_lower = true;
            n_bits_lower = n_bits;
            gain_mult_lower = gain_mult_q8;
            if gains_id != gains_id_lower {
                gains_id_lower = gains_id;

                rc_copy2 = Some(rc.clone());
                let offs = rc.offs as usize;
                ec_buf_copy[..offs].copy_from_slice(&rc.buf[..offs]);
                nsq_copy2 = Some(ps_enc.s_nsq);
                last_gain_index_copy2 = ps_enc.s_shape.last_gain_index;
            }
        } else {
            break;
        }

        if !found_lower && n_bits > max_bits {
            let subfr_length = ps_enc.s_cmn.subfr_length as usize;
            for i in 0..ps_enc.s_cmn.nb_subfr as usize {
                let mut sum: i32 = 0;
                for j in (i * subfr_length)..((i + 1) * subfr_length) {
                    sum += ps_enc.pulses[j].abs() as i32;
                }
                if iter == 0 || (sum < best_sum[i] && !gain_lock[i]) {
                    best_sum[i] = sum;
                    best_gain_mult[i] = gain_mult_q8;
                } else {
                    gain_lock[i] = true;
                }
            }
        }

        if !(found_lower && found_upper) {
            if n_bits > max_bits {
                gain_mult_q8 = silk_min_32(1024, (gain_mult_q8 * 3) / 2);
            } else {
                gain_mult_q8 = silk_max_32(64, (gain_mult_q8 * 4) / 5);
            }
        } else {
            let delta = gain_mult_upper - gain_mult_lower;
            gain_mult_q8 = gain_mult_lower
                + silk_div32_16(
                    (gain_mult_upper - gain_mult_lower) * (max_bits - n_bits_lower),
                    n_bits_upper - n_bits_lower,
                );

            let lower_limit = silk_add_rshift32(gain_mult_lower, delta, 2);
            let upper_limit = silk_sub_rshift32(gain_mult_upper, delta, 2);
            if gain_mult_q8 > lower_limit {
                gain_mult_q8 = lower_limit;
            } else if gain_mult_q8 < upper_limit {
                gain_mult_q8 = upper_limit;
            }
        }

        for i in 0..ps_enc.s_cmn.nb_subfr as usize {
            let tmp = if gain_lock[i] {
                best_gain_mult[i]
            } else {
                gain_mult_q8
            };
            s_enc_ctrl.gains_q16[i] =
                silk_lshift_sat32(silk_smulwb(s_enc_ctrl.gains_unq_q16[i], tmp), 8);
        }

        ps_enc.s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
        silk_gains_quant(
            &mut ps_enc.s_cmn.indices.gains_indices,
            &mut s_enc_ctrl.gains_q16,
            &mut ps_enc.s_shape.last_gain_index,
            if cond_coding == CODE_CONDITIONALLY {
                1
            } else {
                0
            },
            ps_enc.s_cmn.nb_subfr as usize,
        );

        gains_id = silk_gains_id(&ps_enc.s_cmn.indices.gains_indices, ps_enc.s_cmn.nb_subfr);
    }

    let move_len = ltp_mem_length + 5 * ps_enc.s_cmn.fs_khz as usize;
    ps_enc
        .s_cmn
        .x_buf
        .copy_within(frame_length..frame_length + move_len, 0);

    ps_enc.s_cmn.prev_lag = s_enc_ctrl.pitch_l[ps_enc.s_cmn.nb_subfr as usize - 1];
    ps_enc.s_cmn.prev_signal_type = ps_enc.s_cmn.indices.signal_type as i32;
    ps_enc.s_cmn.first_frame_after_reset = 0;

    *pn_bytes_out = (rc.tell() + 7) >> 3;

    0
}

/// Low-Bitrate Redundancy (LBRR) encoding: reuse all parameters but encode
/// the excitation at a lower rate. Port of `silk_LBRR_encode_FIX`
/// (encode_frame_FIX.c:391-448), called after the gains are quantized and
/// before the rate-control loop, so the NSQ state is the frame-entry state.
///
/// On success this frame's LBRR payload lives in
/// `s_cmn.indices_lbrr[fi]` / `s_cmn.pulses_lbrr[fi]` and
/// `s_cmn.lbrr_flags[fi]` is set for the *next* packet's LBRR section.
fn silk_lbrr_encode(
    ps_enc: &mut SilkEncoderState,
    s_enc_ctrl: &SilkEncoderControl,
    x16_offset: usize,
    cond_coding: i32,
) {
    // Control use of inband LBRR. LBRR_SPEECH_ACTIVITY_THRES = 0.3
    // (SILK_FIX_CONST(0.3, 8) = 77).
    if ps_enc.s_cmn.lbrr_enabled == 0 || ps_enc.s_cmn.speech_activity_q8 <= 77 {
        return;
    }

    let fi = ps_enc.s_cmn.n_frames_encoded as usize;
    if fi >= MAX_FRAMES_PER_PACKET {
        return;
    }
    ps_enc.s_cmn.lbrr_flags[fi] = 1;

    // Copy noise shaping quantizer state and quantization indices from the
    // regular encoding.
    let mut nsq_lbrr = ps_enc.s_nsq;
    let mut indices_lbrr = ps_enc.s_cmn.indices;

    if fi == 0 || ps_enc.s_cmn.lbrr_flags[fi - 1] == 0 {
        // First frame in packet or previous frame not LBRR coded.
        ps_enc.s_cmn.lbrr_prev_last_gain_index = ps_enc.s_shape.last_gain_index;

        // Increase gains to get the target LBRR rate. This index is coded
        // absolutely here (first frame of an LBRR run), so it must stay
        // within 0..=N_LEVELS_QGAIN-1.
        indices_lbrr.gains_indices[0] = (indices_lbrr.gains_indices[0] as i32
            + ps_enc.s_cmn.lbrr_gain_increases)
            .min(N_LEVELS_QGAIN - 1) as i8;
    }

    // Decode to get gains in sync with the decoder. Works on a scratch copy
    // of the gains; the main frame's gains are left untouched.
    let mut gains_lbrr = s_enc_ctrl.gains_q16;
    silk_gains_dequant(
        &mut gains_lbrr,
        &indices_lbrr.gains_indices,
        &mut ps_enc.s_cmn.lbrr_prev_last_gain_index,
        if cond_coding == CODE_CONDITIONALLY {
            1
        } else {
            0
        },
        ps_enc.s_cmn.nb_subfr as usize,
    );

    let mut pred_coef_q12_flat = [0i16; 2 * MAX_LPC_ORDER];
    pred_coef_q12_flat[..MAX_LPC_ORDER].copy_from_slice(&s_enc_ctrl.pred_coef_q12[0]);
    pred_coef_q12_flat[MAX_LPC_ORDER..].copy_from_slice(&s_enc_ctrl.pred_coef_q12[1]);

    /*****************************************/
    /* Noise shaping quantization            */
    /*****************************************/
    let mut pulses_lbrr = [0i8; MAX_FRAME_LENGTH];
    if ps_enc.s_cmn.n_states_delayed_decision > 1 || ps_enc.s_cmn.warping_q16 > 0 {
        let winner_seed = silk_nsq_del_dec(
            &ps_enc.s_cmn,
            &mut nsq_lbrr,
            &indices_lbrr,
            &ps_enc.s_cmn.x_buf[x16_offset..],
            &mut pulses_lbrr,
            &pred_coef_q12_flat,
            &s_enc_ctrl.ltp_coef_q14,
            &s_enc_ctrl.ar_q13,
            &s_enc_ctrl.harm_shape_gain_q14,
            &s_enc_ctrl.tilt_q14,
            &s_enc_ctrl.lf_shp_q14,
            &gains_lbrr,
            &s_enc_ctrl.pitch_l,
            s_enc_ctrl.lambda_q10,
            s_enc_ctrl.ltp_scale_q14,
        );
        indices_lbrr.seed = winner_seed as i8;
    } else {
        silk_nsq(
            &ps_enc.s_cmn,
            &mut nsq_lbrr,
            &indices_lbrr,
            &ps_enc.s_cmn.x_buf[x16_offset..],
            &mut pulses_lbrr,
            &pred_coef_q12_flat,
            &s_enc_ctrl.ltp_coef_q14,
            &s_enc_ctrl.ar_q13,
            &s_enc_ctrl.harm_shape_gain_q14,
            &s_enc_ctrl.tilt_q14,
            &s_enc_ctrl.lf_shp_q14,
            &gains_lbrr,
            &s_enc_ctrl.pitch_l,
            s_enc_ctrl.lambda_q10,
            s_enc_ctrl.ltp_scale_q14,
        );
    }

    ps_enc.s_cmn.indices_lbrr[fi] = indices_lbrr;
    ps_enc.s_cmn.pulses_lbrr[fi] = pulses_lbrr;
}

