//! libopus `silk/float/encode_frame_FLP.c`: the analysis, noise shaping
//! quantization, rate control and LBRR coding of one frame by the float
//! encoder. The VAD (`silk_encode_do_VAD_FLP`) is the shared
//! [`silk_encode_do_vad`](crate::silk::enc_api::silk_encode_do_vad).

use crate::range_coder::RangeCoder;
use crate::silk::define::*;
use crate::silk::encode_indices::silk_encode_indices;
use crate::silk::encode_pulses::silk_encode_pulses;
use crate::silk::float::control_flp::{silk_find_pred_coefs_flp, silk_process_gains_flp};
use crate::silk::float::noise_shape_analysis_flp::silk_noise_shape_analysis_flp;
use crate::silk::float::find_pitch_lags_flp::silk_find_pitch_lags_flp;
use crate::silk::float::sigproc_flp::silk_short2float_array;
use crate::silk::float::structs_flp::SilkEncoderControlFlp;
use crate::silk::float::wrappers_flp::silk_nsq_wrapper_flp;
use crate::silk::gain_quant::{silk_gains_dequant, silk_gains_id, silk_gains_quant};
use crate::silk::macros::*;
use crate::silk::structs::*;

/// Copies the new `frame` (16-bit, after the variable low-pass) into the
/// float analysis buffer behind the look-ahead and adds the tiny signal
/// libopus adds against denormals.
fn insert_frame(ps_enc: &mut SilkEncoderState, frame: &[i16]) {
    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let start = ps_enc.s_cmn.ltp_mem_length as usize + LA_SHAPE_MS * ps_enc.s_cmn.fs_khz as usize;
    let x = &mut ps_enc.s_cmn.x_buf[start..start + frame_length];
    silk_short2float_array(x, &frame[..frame_length]);
    for i in 0..8 {
        x[i * (frame_length >> 3)] += (1 - (i as i32 & 2)) as f32 * 1e-6;
    }
}

/// Moves the analysis buffer on by one frame.
fn shift_x_buf(ps_enc: &mut SilkEncoderState) {
    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let n = ps_enc.s_cmn.ltp_mem_length as usize + LA_SHAPE_MS * ps_enc.s_cmn.fs_khz as usize;
    ps_enc.s_cmn.x_buf.copy_within(frame_length..frame_length + n, 0);
}

/// The prefill part of libopus's `silk_encode_frame_FLP`: `frame` goes
/// into the analysis buffer and the buffer moves on, coding nothing.
/// `frame_length` is the prefill's.
pub fn silk_prefill_frame_flp(ps_enc: &mut SilkEncoderState, frame: &[i16]) {
    insert_frame(ps_enc, frame);
    shift_x_buf(ps_enc);
}

/// libopus `silk_encode_frame_FLP` without the prefill: codes the frame
/// `input` (16-bit, after the variable low-pass) into `rc` within
/// `max_bits`; `pn_bytes_out` receives the payload size.
pub fn silk_encode_frame_flp(
    ps_enc: &mut SilkEncoderState,
    input: &[i16],
    rc: &mut RangeCoder,
    pn_bytes_out: &mut i32,
    cond_coding: i32,
    max_bits: i32,
    use_cbr: i32,
) -> i32 {
    let mut ctrl = SilkEncoderControlFlp::default();
    let bits_margin = if use_cbr != 0 { 5 } else { max_bits / 4 };

    ps_enc.s_cmn.indices.seed = (ps_enc.s_cmn.frame_counter & 3) as i8;
    ps_enc.s_cmn.frame_counter += 1;

    let x_frame = ps_enc.s_cmn.ltp_mem_length as usize;
    insert_frame(ps_enc, input);
    let x_buf = ps_enc.s_cmn.x_buf;

    // Pitch lags and the initial LPC analysis.
    let mut res_pitch = [0.0f32; 2 * MAX_FRAME_LENGTH + LA_PITCH_MAX];
    silk_find_pitch_lags_flp(ps_enc, &mut ctrl, &mut res_pitch, &x_buf);

    let la_shape = ps_enc.s_cmn.la_shape as usize;
    silk_noise_shape_analysis_flp(ps_enc, &mut ctrl, &res_pitch[x_frame..], &x_buf[x_frame - la_shape..]);

    silk_find_pred_coefs_flp(ps_enc, &mut ctrl, &res_pitch, &x_buf, x_frame, cond_coding);

    silk_process_gains_flp(ps_enc, &mut ctrl, cond_coding);

    silk_lbrr_encode_flp(ps_enc, &mut ctrl, &x_buf[x_frame..], cond_coding);

    // The quantizer and the entropy coder in a loop that controls the rate.
    let max_iter = 6;
    let mut gain_mult_q8: i32 = 256;
    let mut found_lower = false;
    let mut found_upper = false;
    let mut n_bits: i32;
    let mut n_bits_lower = 0;
    let mut n_bits_upper = 0;
    let mut gain_mult_lower = 0;
    let mut gain_mult_upper = 0;
    let nb_subfr = ps_enc.s_cmn.nb_subfr as usize;
    let mut gains_id = silk_gains_id(&ps_enc.s_cmn.indices.gains_indices, ps_enc.s_cmn.nb_subfr);
    let mut gains_id_lower = -1;
    let mut gains_id_upper = -1;

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
    let mut best_sum = [0i32; MAX_NB_SUBFR];

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

            ps_enc.s_cmn.indices.seed = silk_nsq_wrapper_flp(
                &ps_enc.s_cmn,
                &mut ps_enc.s_nsq,
                &ps_enc.s_cmn.indices,
                &ctrl,
                &mut ps_enc.pulses,
                &x_buf[x_frame..],
            );

            if iter == max_iter && !found_lower {
                rc_copy2 = Some(rc.clone());
            }

            let fi = ps_enc.s_cmn.n_frames_encoded as usize;
            silk_encode_indices(ps_enc, rc, fi, false, cond_coding);
            silk_encode_pulses(
                rc,
                i32::from(ps_enc.s_cmn.indices.signal_type),
                i32::from(ps_enc.s_cmn.indices.quant_offset_type),
                &mut ps_enc.pulses,
                ps_enc.s_cmn.frame_length as usize,
            );
            n_bits = rc.tell();

            // Still over the budget after the last iteration: keep the
            // previous frame's gains and code no pulses.
            if iter == max_iter && !found_lower && n_bits > max_bits {
                if let Some(c2) = &rc_copy2 {
                    *rc = c2.clone();
                }
                ps_enc.s_shape.last_gain_index = ctrl.last_gain_index_prev;
                ps_enc.s_cmn.indices.gains_indices[..nb_subfr].fill(4);
                if cond_coding != CODE_CONDITIONALLY {
                    ps_enc.s_cmn.indices.gains_indices[0] = ctrl.last_gain_index_prev;
                }
                ps_enc.s_cmn.ec_prev_lag_index = ec_prev_lag_index_copy;
                ps_enc.s_cmn.ec_prev_signal_type = ec_prev_signal_type_copy;
                ps_enc.pulses.fill(0);
                silk_encode_indices(ps_enc, rc, fi, false, cond_coding);
                silk_encode_pulses(
                    rc,
                    i32::from(ps_enc.s_cmn.indices.signal_type),
                    i32::from(ps_enc.s_cmn.indices.quant_offset_type),
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
                // Back to the iteration that met the budget.
                if let Some(c2) = &rc_copy2 {
                    *rc = c2.clone();
                    let offs = rc.offs as usize;
                    rc.buf[..offs].copy_from_slice(&ec_buf_copy[..offs]);
                }
                if let Some(n) = &nsq_copy2 {
                    ps_enc.s_nsq = *n;
                }
                ps_enc.s_shape.last_gain_index = last_gain_index_copy2;
            }
            break;
        }

        if n_bits > max_bits {
            if !found_lower && iter >= 2 {
                // A higher rate-distortion trade-off and less dithering;
                // the earlier upper results are dropped.
                ctrl.lambda = (ctrl.lambda * 1.5).max(1.5);
                ps_enc.s_cmn.indices.quant_offset_type = 0;
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
            // Close enough.
            break;
        }

        if !found_lower && n_bits > max_bits {
            let subfr_length = ps_enc.s_cmn.subfr_length as usize;
            for i in 0..nb_subfr {
                let sum: i32 = ps_enc.pulses[i * subfr_length..(i + 1) * subfr_length]
                    .iter()
                    .map(|&p| i32::from(p).abs())
                    .sum();
                if iter == 0 || (sum < best_sum[i] && !gain_lock[i]) {
                    best_sum[i] = sum;
                    best_gain_mult[i] = gain_mult_q8;
                } else {
                    gain_lock[i] = true;
                }
            }
        }

        if !(found_lower && found_upper) {
            // The high-rate rate/distortion curve.
            gain_mult_q8 = if n_bits > max_bits {
                (gain_mult_q8 * 3 / 2).min(1024)
            } else {
                (gain_mult_q8 * 4 / 5).max(64)
            };
        } else {
            // Interpolated, within 25% and 75% of the range.
            gain_mult_q8 = gain_mult_lower
                + (gain_mult_upper - gain_mult_lower) * (max_bits - n_bits_lower)
                    / (n_bits_upper - n_bits_lower);
            let delta = gain_mult_upper - gain_mult_lower;
            if gain_mult_q8 > silk_add_rshift32(gain_mult_lower, delta, 2) {
                gain_mult_q8 = silk_add_rshift32(gain_mult_lower, delta, 2);
            } else if gain_mult_q8 < silk_sub_rshift32(gain_mult_upper, delta, 2) {
                gain_mult_q8 = silk_sub_rshift32(gain_mult_upper, delta, 2);
            }
        }

        let mut gains_q16 = [0i32; MAX_NB_SUBFR];
        for i in 0..nb_subfr {
            let tmp = if gain_lock[i] { best_gain_mult[i] } else { gain_mult_q8 };
            gains_q16[i] = silk_lshift_sat32(silk_smulwb(ctrl.gains_unq_q16[i], tmp as i16 as i32), 8);
        }
        ps_enc.s_shape.last_gain_index = ctrl.last_gain_index_prev;
        silk_gains_quant(
            &mut ps_enc.s_cmn.indices.gains_indices,
            &mut gains_q16,
            &mut ps_enc.s_shape.last_gain_index,
            (cond_coding == CODE_CONDITIONALLY) as i32,
            nb_subfr,
        );
        gains_id = silk_gains_id(&ps_enc.s_cmn.indices.gains_indices, ps_enc.s_cmn.nb_subfr);
        for (g, &q) in ctrl.gains[..nb_subfr].iter_mut().zip(&gains_q16) {
            *g = q as f32 / 65536.0;
        }
    }

    shift_x_buf(ps_enc);

    ps_enc.s_cmn.prev_lag = ctrl.pitch_l[nb_subfr - 1];
    ps_enc.s_cmn.prev_signal_type = i32::from(ps_enc.s_cmn.indices.signal_type);
    ps_enc.s_cmn.first_frame_after_reset = 0;
    *pn_bytes_out = (rc.tell() + 7) >> 3;
    0
}

/// libopus `silk_LBRR_encode_FLP`: the frame's redundant copy, quantized
/// with the frame's parameters at higher gains, on a copy of the quantizer
/// state from before the frame.
fn silk_lbrr_encode_flp(
    ps_enc: &mut SilkEncoderState,
    ctrl: &mut SilkEncoderControlFlp,
    xfw: &[f32],
    cond_coding: i32,
) {
    // SILK_FIX_CONST(LBRR_SPEECH_ACTIVITY_THRES, 8).
    if ps_enc.s_cmn.lbrr_enabled == 0 || ps_enc.s_cmn.speech_activity_q8 <= 77 {
        return;
    }
    let fi = ps_enc.s_cmn.n_frames_encoded as usize;
    let nb_subfr = ps_enc.s_cmn.nb_subfr as usize;
    ps_enc.s_cmn.lbrr_flags[fi] = 1;

    let mut nsq_lbrr = ps_enc.s_nsq;
    let mut indices_lbrr = ps_enc.s_cmn.indices;
    let temp_gains = ctrl.gains;

    if fi == 0 || ps_enc.s_cmn.lbrr_flags[fi - 1] == 0 {
        // First frame of an LBRR run: higher gains for the LBRR rate.
        ps_enc.s_cmn.lbrr_prev_last_gain_index = ps_enc.s_shape.last_gain_index;
        indices_lbrr.gains_indices[0] = (i32::from(indices_lbrr.gains_indices[0])
            + ps_enc.s_cmn.lbrr_gain_increases)
            .min(N_LEVELS_QGAIN - 1) as i8;
    }

    // The gains as the decoder sees them.
    let mut gains_q16 = [0i32; MAX_NB_SUBFR];
    silk_gains_dequant(
        &mut gains_q16,
        &indices_lbrr.gains_indices,
        &mut ps_enc.s_cmn.lbrr_prev_last_gain_index,
        (cond_coding == CODE_CONDITIONALLY) as i32,
        nb_subfr,
    );
    for (g, &q) in ctrl.gains[..nb_subfr].iter_mut().zip(&gains_q16) {
        *g = q as f32 * (1.0 / 65536.0);
    }

    let mut pulses_lbrr = [0i8; MAX_FRAME_LENGTH];
    indices_lbrr.seed = silk_nsq_wrapper_flp(
        &ps_enc.s_cmn,
        &mut nsq_lbrr,
        &indices_lbrr,
        ctrl,
        &mut pulses_lbrr,
        xfw,
    );
    ps_enc.s_cmn.indices_lbrr[fi] = indices_lbrr;
    ps_enc.s_cmn.pulses_lbrr[fi] = pulses_lbrr;

    ctrl.gains = temp_gains;
}
