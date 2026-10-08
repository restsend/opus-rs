use crate::range_coder::RangeCoder;
use crate::silk::control_fixed::*;
use crate::silk::control_snr::silk_control_snr;
use crate::silk::define::*;
use crate::silk::encode_indices::*;
use crate::silk::encode_pulses::*;
use crate::silk::gain_quant::{silk_gains_dequant, silk_gains_id, silk_gains_quant};
use crate::silk::hp_variable_cutoff::silk_hp_variable_cutoff;
use crate::silk::lp_variable_cutoff::*;
use crate::silk::macros::*;
use crate::silk::noise_shape_analysis::*;
use crate::silk::nsq::*;
use crate::silk::nsq_del_dec::*;
use crate::silk::pitch_analysis::*;
use crate::silk::stereo_lr_to_ms::silk_stereo_lr_to_ms;
use crate::silk::structs::*;
use crate::silk::tuning_parameters::BITRESERVOIR_DECAY_TIME_MS;
use crate::silk::vad::silk_vad_get_sa_q8;

pub fn silk_encode_do_vad(ps_enc: &mut SilkEncoderState, input: &[i16], activity: i32) {
    let activity_threshold = SPEECH_ACTIVITY_DTX_THRES_Q8;

    let frame_length = ps_enc.s_cmn.frame_length as usize;
    silk_vad_get_sa_q8(ps_enc, input, frame_length);

    if activity == 0 && ps_enc.s_cmn.speech_activity_q8 >= activity_threshold {
        ps_enc.s_cmn.speech_activity_q8 = activity_threshold - 1;
    }

    if ps_enc.s_cmn.speech_activity_q8 < activity_threshold {
        ps_enc.s_cmn.indices.signal_type = TYPE_NO_VOICE_ACTIVITY as i8;
        ps_enc.s_cmn.no_speech_counter += 1;
        if ps_enc.s_cmn.no_speech_counter <= NB_SPEECH_FRAMES_BEFORE_DTX {
            ps_enc.s_cmn.in_dtx = 0;
        } else if ps_enc.s_cmn.no_speech_counter > MAX_CONSECUTIVE_DTX + NB_SPEECH_FRAMES_BEFORE_DTX
        {
            ps_enc.s_cmn.no_speech_counter = NB_SPEECH_FRAMES_BEFORE_DTX;
            ps_enc.s_cmn.in_dtx = 0;
        }
        ps_enc.s_cmn.vad_flags[ps_enc.s_cmn.n_frames_encoded as usize] = 0;
    } else {
        ps_enc.s_cmn.no_speech_counter = 0;
        ps_enc.s_cmn.in_dtx = 0;
        ps_enc.s_cmn.indices.signal_type = TYPE_UNVOICED as i8;
        ps_enc.s_cmn.vad_flags[ps_enc.s_cmn.n_frames_encoded as usize] = 1;
    }
}

/// Internal sample rate (kHz) and 10 ms prefill length of `ps_enc`, or
/// `None` when SILK isn't configured to a supported rate.
fn prefill_geometry(ps_enc: &SilkEncoderState) -> Option<(i32, usize)> {
    let fs_khz = ps_enc.s_cmn.fs_khz;
    if fs_khz != 8 && fs_khz != 12 && fs_khz != 16 {
        return None;
    }
    Some((fs_khz, fs_khz as usize * 10))
}

pub fn silk_encode_prefill(ps_enc: &mut SilkEncoderState, samples: &[i16], _activity: i32) {
    let Some((_, prefill_frame_length)) = prefill_geometry(ps_enc) else {
        return;
    };
    if samples.len() < prefill_frame_length {
        return;
    }

    let mut input_buf = [0i16; MAX_FRAME_LENGTH + 2];
    input_buf[0] = ps_enc.stereo.s_mid[0];
    input_buf[1] = ps_enc.stereo.s_mid[1];
    input_buf[2..2 + prefill_frame_length].copy_from_slice(&samples[..prefill_frame_length]);
    ps_enc.stereo.s_mid[0] = input_buf[prefill_frame_length];
    ps_enc.stereo.s_mid[1] = input_buf[prefill_frame_length + 1];

    prefill_frame(ps_enc, &mut input_buf, prefill_frame_length);
}

/// Prefill a stereo encoder with 10 ms of internal-rate `left` and `right`
/// input, as libopus's `silk_Encode` does with `prefillFlag` set for two
/// internal channels: convert to mid/side (updating the stereo state, writing
/// nothing), run the VADs, and push the frame into the history of the mid
/// and, unless the frame codes the mid only, of the side, without coding it.
/// `total_rate_bps` is the SILK bitrate.
pub fn silk_encode_prefill_stereo(
    mid: &mut SilkEncoderState,
    side: &mut SilkEncoderState,
    left: &[i16],
    right: &[i16],
    total_rate_bps: i32,
    activity: i32,
) {
    let Some((fs_khz, prefill_frame_length)) = prefill_geometry(mid) else {
        return;
    };
    if left.len() < prefill_frame_length || right.len() < prefill_frame_length {
        return;
    }

    let mut x1 = [0i16; MAX_FRAME_LENGTH + 2];
    let mut x2 = [0i16; MAX_FRAME_LENGTH + 2];
    x1[2..2 + prefill_frame_length].copy_from_slice(&left[..prefill_frame_length]);
    x2[2..2 + prefill_frame_length].copy_from_slice(&right[..prefill_frame_length]);
    let (mut ix, mut mid_only, mut rates) = ([[0i8; 3]; 2], 0i8, [0i32; 2]);
    // The encoders were just reset, so the previous speech activity is 0.
    silk_stereo_lr_to_ms(
        &mut mid.stereo,
        &mut x1,
        &mut x2,
        &mut ix,
        &mut mid_only,
        &mut rates,
        total_rate_bps,
        0,
        false,
        fs_khz,
        prefill_frame_length,
    );
    mid.stereo.pred_ix[0] = ix;
    mid.stereo.mid_only_flags[0] = mid_only;

    mid.s_cmn.n_frames_encoded = 0;
    side.s_cmn.n_frames_encoded = 0;
    if mid_only == 0 {
        if mid.stereo.prev_decode_only_middle == 1 {
            reset_side_for_coding(side);
        }
        silk_encode_do_vad(side, &x2[1..1 + prefill_frame_length], activity);
    } else {
        side.s_cmn.vad_flags[0] = 0;
    }
    silk_encode_do_vad(mid, &x1[1..1 + prefill_frame_length], activity);
    prefill_frame(mid, &mut x1, prefill_frame_length);
    if rates[1] > 0 {
        prefill_frame(side, &mut x2, prefill_frame_length);
    }
    mid.stereo.prev_decode_only_middle = mid_only as i32;
}

/// The prefill part of libopus's `silk_encode_frame_FIX`: low-pass the 10 ms
/// frame at `input_buf[1..]` and push it into the encoder's analysis buffer,
/// coding nothing.
fn prefill_frame(
    ps_enc: &mut SilkEncoderState,
    input_buf: &mut [i16; MAX_FRAME_LENGTH + 2],
    prefill_frame_length: usize,
) {
    let fs_khz = ps_enc.s_cmn.fs_khz as usize;
    let real_frame_length = ps_enc.s_cmn.frame_length as usize;
    let real_nb_subfr = ps_enc.s_cmn.nb_subfr;
    let real_subfr_length = ps_enc.s_cmn.subfr_length;

    ps_enc.s_cmn.frame_length = prefill_frame_length as i32;
    ps_enc.s_cmn.nb_subfr = 2;
    ps_enc.s_cmn.subfr_length = (prefill_frame_length / 2) as i32;

    let ltp_mem_length = ps_enc.s_cmn.ltp_mem_length as usize;
    let la_shape_ms_samples = 5 * fs_khz;

    silk_lp_variable_cutoff(
        &mut ps_enc.s_cmn.s_lp,
        &mut input_buf[1..],
        prefill_frame_length,
    );

    let x_frame_idx = ltp_mem_length;
    let dst = x_frame_idx + la_shape_ms_samples;

    if dst + prefill_frame_length <= ps_enc.s_cmn.x_buf.len() {
        ps_enc.s_cmn.x_buf[dst..dst + prefill_frame_length]
            .copy_from_slice(&input_buf[1..1 + prefill_frame_length]);
    }

    let move_len = ltp_mem_length + la_shape_ms_samples;
    if prefill_frame_length + move_len <= ps_enc.s_cmn.x_buf.len() {
        ps_enc
            .s_cmn
            .x_buf
            .copy_within(prefill_frame_length..prefill_frame_length + move_len, 0);
    }

    ps_enc.s_cmn.frame_length = real_frame_length as i32;
    ps_enc.s_cmn.nb_subfr = real_nb_subfr;
    ps_enc.s_cmn.subfr_length = real_subfr_length;
}

pub fn silk_encode_frame(
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
                &ps_enc.pulses,
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
                    &ps_enc.pulses,
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
    if ps_enc.s_cmn.lbrr_enabled == 0
        || ps_enc.s_cmn.speech_activity_q8 <= 77
    {
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
        if cond_coding == CODE_CONDITIONALLY { 1 } else { 0 },
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
        silk_nsq_del_dec(
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

/// Encode the LBRR (in-band FEC) section for a whole packet and return the
/// number of bits written. Matches libopus enc_API.c:365-371: an LBRR symbol
/// for multi-frame packets, then per frame a stereo header (when stereo)
/// followed by indices and pulses.
fn encode_lbrr_section(
    rc: &mut RangeCoder,
    ps_enc: &mut SilkEncoderState,
    lbrr_symbol: i32,
    n_frames_per_packet: i32,
) -> i32 {
    let start_bits = rc.tell();

    if n_frames_per_packet > 1 {
        rc.encode_icdf(lbrr_symbol - 1, lbrr_flags_icdf(n_frames_per_packet), 8);
    }

    for i in 0..n_frames_per_packet as usize {
        if ps_enc.s_cmn.lbrr_flags[i] != 0 {
            let lbrr_cond = if i > 0 && ps_enc.s_cmn.lbrr_flags[i - 1] != 0 {
                CODE_CONDITIONALLY
            } else {
                CODE_INDEPENDENTLY
            };
            // libopus writes the stereo header (pred + conditional mid-only
            // flag) before every LBRR payload; the decoder's LBRR skip path
            // expects it (issue #27, LBRR section).
            if ps_enc.s_cmn.n_channels == 2 {
                silk_encode_stereo(rc, 0, 0, 1);
            }
            silk_encode_indices(ps_enc, rc, i, true, lbrr_cond);
            silk_encode_pulses(
                rc,
                ps_enc.s_cmn.indices_lbrr[i].signal_type as i32,
                ps_enc.s_cmn.indices_lbrr[i].quant_offset_type as i32,
                &ps_enc.s_cmn.pulses_lbrr[i],
                ps_enc.s_cmn.frame_length as usize,
            );
        }
    }

    rc.tell() - start_bits
}

/// Bits frame `frame_idx` of a `tot_blocks`-frame packet may fill the packet
/// up to (libopus enc_API.c). Capping the earlier frames keeps a share of the
/// packet for the later ones; otherwise a 60 ms packet's first two frames can
/// leave the last one less than even its no-pulse fallback costs, and the
/// packet overflows.
fn silk_frame_max_bits(max_bits: i32, tot_blocks: i32, frame_idx: i32) -> i32 {
    match (tot_blocks, frame_idx) {
        (2, 0) => max_bits * 3 / 5,
        (3, 0) => max_bits * 2 / 5,
        (3, 1) => max_bits * 3 / 4,
        _ => max_bits,
    }
}

/// Number of 20 ms blocks (or one 10 ms block) in a packet of `n_samples_in`
/// internal-rate samples: libopus enc_API.c `tot_blocks`.
fn packet_blocks(n_samples_in: usize, fs_khz: i32) -> i32 {
    let n_blocks_of_10ms = (100 * n_samples_in as i32) / (fs_khz * 1000);
    if n_blocks_of_10ms > 1 {
        n_blocks_of_10ms >> 1
    } else {
        1
    }
}

/// The target rate of the packet's next frame, in bits per second
/// (enc_API.c:398-432): its share of `bit_rate`, less the packet's LBRR
/// section, less a 500 ms payback of the bits coded beyond the target, both
/// in earlier packets (`n_bits_exceeded`) and in this packet's earlier frames
/// (`bits_so_far`, the range coder's `tell`).
///
/// `lbrr_bits` is the size of the packet's LBRR section for its first frame
/// and 0 for the others, as libopus's `curr_nBitsUsedLBRR` is: a later frame
/// drops the LBRR average to 0, and the section then counts against it as
/// overshoot.
fn silk_frame_target_rate_bps(
    ps_enc: &mut SilkEncoderState,
    bit_rate: i32,
    lbrr_bits: i32,
    bits_so_far: i32,
) -> i32 {
    let packet_size_ms = ps_enc.s_cmn.packet_size_ms;
    // A moving average of the LBRR usage, except that the first packet with
    // LBRR isn't averaged and a packet without LBRR drops it to 0 at once.
    ps_enc.n_bits_used_lbrr = if lbrr_bits < 10 {
        0
    } else if ps_enc.n_bits_used_lbrr < 10 {
        lbrr_bits
    } else {
        (ps_enc.n_bits_used_lbrr + lbrr_bits) / 2
    };
    let n_bits = (bit_rate * packet_size_ms / 1000 - ps_enc.n_bits_used_lbrr)
        / ps_enc.s_cmn.n_frames_per_packet;
    let mut target = n_bits * if packet_size_ms == 10 { 100 } else { 50 };
    target -= ps_enc.n_bits_exceeded * 1000 / BITRESERVOIR_DECAY_TIME_MS;
    let n_frames_encoded = ps_enc.s_cmn.n_frames_encoded;
    if n_frames_encoded > 0 {
        let bits_balance = bits_so_far - ps_enc.n_bits_used_lbrr - n_bits * n_frames_encoded;
        target -= bits_balance * 1000 / BITRESERVOIR_DECAY_TIME_MS;
    }
    // Never above the input rate. Like silk_LIMIT, the bounds swap when
    // `bit_rate` is below 5000.
    target.clamp(bit_rate.min(5000), bit_rate.max(5000))
}

/// Add the packet's overshoot of `bit_rate` to the bit reservoir that later
/// frame targets pay back (enc_API.c:544-546).
fn silk_update_bit_reservoir(ps_enc: &mut SilkEncoderState, bit_rate: i32, n_bytes_out: i32) {
    let target_bits = bit_rate * ps_enc.s_cmn.packet_size_ms / 1000;
    ps_enc.n_bits_exceeded =
        (ps_enc.n_bits_exceeded + n_bytes_out * 8 - target_bits).clamp(0, 10000);
}

/// iCDF of the per-frame LBRR flags of a 40 or 60 ms packet.
fn lbrr_flags_icdf(n_frames_per_packet: i32) -> &'static [u8] {
    match n_frames_per_packet {
        3 => &crate::silk::tables::SILK_LBRR_FLAGS_3_ICDF[..],
        _ => &crate::silk::tables::SILK_LBRR_FLAGS_2_ICDF[..],
    }
}

pub fn silk_encode(
    ps_enc: &mut SilkEncoderState,
    samples_in: &[i16],
    n_samples_in: usize,
    rc: &mut RangeCoder,
    n_bytes_out: &mut i32,
    target_rate_bps: i32,
    max_bits: i32,
    use_cbr: i32,
    activity: i32,
) -> i32 {
    // The frame loop slices samples_in[..frame_end] based on this declared
    // length; a mismatched declaration would panic mid-encode
    // (issue #27 deep scan).
    assert!(
        n_samples_in <= samples_in.len(),
        "silk_encode: n_samples_in ({}) exceeds samples_in.len() ({})",
        n_samples_in,
        samples_in.len()
    );
    let n_frames_per_packet = ps_enc.s_cmn.n_frames_per_packet;
    let frame_length = ps_enc.s_cmn.frame_length as usize;

    ps_enc.s_cmn.n_frames_encoded = 0;

    let tot_blocks = packet_blocks(n_samples_in, ps_enc.s_cmn.fs_khz);

    let lbrr_possible = ps_enc.s_cmn.use_in_band_fec != 0
        && ps_enc.s_cmn.packet_loss_perc > 0
        && ps_enc.s_cmn.lbrr_enabled != 0;

    // The LBRR flags persisted from the previous packet's frames describe the
    // LBRR payload written at the start of this packet (enc_API.c builds
    // LBRR_symbol from LBRR_flags). A reset drops them (enc_API.c:255-259).
    if ps_enc.s_cmn.first_frame_after_reset != 0 {
        ps_enc.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
    }

    let mut lbrr_symbol: i32 = 0;
    if lbrr_possible {
        for i in 0..n_frames_per_packet as usize {
            lbrr_symbol |= ps_enc.s_cmn.lbrr_flags[i] << i;
        }
    }

    ps_enc.s_cmn.lbrr_flag = if lbrr_symbol > 0 { 1 } else { 0 };

    let mut sample_offset = 0usize;

    // Bits of the LBRR section written at the start of the packet, which come
    // out of the frame target (libopus `curr_nBitsUsedLBRR`).
    let mut lbrr_bits = 0i32;

    for frame_idx in 0..n_frames_per_packet {
        if frame_idx == 0 {
            silk_hp_variable_cutoff(&mut ps_enc.s_cmn);
        }

        let frame_end = (sample_offset + frame_length).min(n_samples_in);
        let raw_frame = &samples_in[sample_offset..frame_end];

        let fs_in_khz = ps_enc.s_cmn.fs_khz as usize;

        if raw_frame.len() < fs_in_khz {
            sample_offset += frame_length;
            continue;
        }

        let n = raw_frame.len();

        if n > MAX_FRAME_LENGTH {
            sample_offset += frame_length;
            continue;
        }

        // The API-rate -> internal-rate conversion (with its delay_matrix_enc
        // input delay) is done by the caller's encoder resampler; frames
        // arrive here at the internal rate.
        let mut resampler_out = [0i16; MAX_FRAME_LENGTH];
        resampler_out[..n].copy_from_slice(raw_frame);

        let mut input_buf = [0i16; MAX_FRAME_LENGTH + 2];
        input_buf[0] = ps_enc.stereo.s_mid[0];
        input_buf[1] = ps_enc.stereo.s_mid[1];
        input_buf[2..2 + n].copy_from_slice(&resampler_out[..n]);

        ps_enc.stereo.s_mid[0] = input_buf[frame_length];
        ps_enc.stereo.s_mid[1] = input_buf[frame_length + 1];

        if frame_idx == 0 {
            let n_channels = ps_enc.s_cmn.n_channels;
            let n_flag_bits = ((n_frames_per_packet + 1) * n_channels) as u32;
            let icdf_val = (256i32 - (256i32 >> n_flag_bits)) as u8;
            let icdf = [icdf_val, 0u8];
            rc.encode_icdf(0, &icdf, 8);

            if lbrr_symbol > 0 {
                // Trial-encode the LBRR section on a scratch range coder. If
                // the packet budget cannot hold the LBRR data plus a minimal
                // main frame, drop LBRR entirely (the decoder sees
                // lbrr_flag = 0 and skips nothing) instead of overflowing the
                // encoder buffer.
                let capacity_bits = (rc.storage as i32) * 8 - 8;
                let saved_ec_prev_signal_type = ps_enc.s_cmn.ec_prev_signal_type;
                let saved_ec_prev_lag_index = ps_enc.s_cmn.ec_prev_lag_index;
                let mut trial = rc.clone();
                let _trial_bits =
                    encode_lbrr_section(&mut trial, ps_enc, lbrr_symbol, n_frames_per_packet);
                ps_enc.s_cmn.ec_prev_signal_type = saved_ec_prev_signal_type;
                ps_enc.s_cmn.ec_prev_lag_index = saved_ec_prev_lag_index;
                if trial.tell() + 64 <= capacity_bits {
                    lbrr_bits = encode_lbrr_section(rc, ps_enc, lbrr_symbol, n_frames_per_packet);
                } else {
                    ps_enc.s_cmn.lbrr_flag = 0;
                    lbrr_symbol = 0;
                }
            }

            // Reset the LBRR flags for the next packet (enc_API.c:379-382);
            // this packet's frames set them again from inside
            // silk_lbrr_encode.
            ps_enc.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];

            // NOTE: the per-frame stereo header is written further below (after
            // VAD), once for every frame of the packet — matching libopus
            // enc_API.c:443-448. Writing it only for frame 0 desynced the
            // bitstream of multi-frame stereo packets (issue #27).
        }

        let frame_rate_bps = silk_frame_target_rate_bps(
            ps_enc,
            target_rate_bps,
            if frame_idx == 0 { lbrr_bits } else { 0 },
            rc.tell(),
        );
        silk_control_snr(&mut ps_enc.s_cmn, frame_rate_bps);

        let vad_frame = &input_buf[1..1 + frame_length];
        silk_encode_do_vad(ps_enc, vad_frame, activity);

        // Per-frame stereo header: mid/side prediction followed by the
        // mid-only flag (written when the side channel carries no VAD, which
        // is always the case for this port's mid-only stereo). The decoder
        // reads these bits for every frame (dec_API.c / dec_api.rs).
        if ps_enc.s_cmn.n_channels == 2 {
            silk_encode_stereo(rc, 0, 0, 1);
        }

        silk_lp_variable_cutoff(&mut ps_enc.s_cmn.s_lp, &mut input_buf[1..], frame_length);

        let frame_samples = &input_buf[1..1 + frame_length];

        let cond_coding = if ps_enc.s_cmn.n_frames_encoded == 0 {
            CODE_INDEPENDENTLY
        } else {
            CODE_CONDITIONALLY
        };

        let mut frame_bytes = 0i32;
        let ret = silk_encode_frame(
            ps_enc,
            frame_samples,
            rc,
            &mut frame_bytes,
            cond_coding,
            // A cap on the packet's bits so far, the LBRR section included,
            // since the rate control measures the range coder's `tell`.
            silk_frame_max_bits(max_bits, tot_blocks, frame_idx),
            if use_cbr != 0 && frame_idx == n_frames_per_packet - 1 {
                1
            } else {
                0
            },
        );
        if ret != 0 {
            return ret;
        }

        ps_enc.s_cmn.n_frames_encoded += 1;
        sample_offset += frame_length;
    }

    let n_channels = ps_enc.s_cmn.n_channels;
    let n_flag_bits = ((n_frames_per_packet + 1) * n_channels) as u32;
    let mut flags = 0u32;
    for i in 0..n_frames_per_packet as usize {
        flags <<= 1;
        flags |= ps_enc.s_cmn.vad_flags[i] as u32;
    }
    flags <<= 1;
    flags |= ps_enc.s_cmn.lbrr_flag as u32;
    if n_channels == 2 {
        flags <<= (n_frames_per_packet + 1) as u32;
    }

    rc.patch_initial_bits(flags, n_flag_bits);

    *n_bytes_out = (rc.tell() + 7) >> 3;
    silk_update_bit_reservoir(ps_enc, target_rate_bps, *n_bytes_out);

    0
}

/// Encode one packet of stereo input as SILK mid/side, the way libopus's
/// `silk_Encode` does for two internal channels (enc_API.c).
///
/// `left` and `right` hold `n_samples_in` samples each, at the internal
/// rate, de-interleaved. Each frame is converted to mid/side by
/// [`silk_stereo_lr_to_ms`], which also picks the predictors, the stereo
/// width and the split of the frame's bits between mid and side. `mid` codes
/// the mid and holds the stereo state; `side` codes the side residual, except
/// in frames that code the mid only (panned mono).
pub fn silk_encode_stereo_packet(
    mid: &mut SilkEncoderState,
    side: &mut SilkEncoderState,
    left: &[i16],
    right: &[i16],
    n_samples_in: usize,
    rc: &mut RangeCoder,
    n_bytes_out: &mut i32,
    target_rate_bps: i32,
    max_bits: i32,
    use_cbr: i32,
    activity: i32,
) -> i32 {
    let n_frames_per_packet = mid.s_cmn.n_frames_per_packet;
    let frame_length = mid.s_cmn.frame_length as usize;
    let fs_khz = mid.s_cmn.fs_khz;
    let packet_length = n_frames_per_packet as usize * frame_length;
    // A frame skipped for want of input would desynchronize the stereo
    // header, so the whole packet must be there.
    if n_samples_in < packet_length || left.len() < packet_length || right.len() < packet_length {
        return SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES;
    }

    mid.s_cmn.n_frames_encoded = 0;
    side.s_cmn.n_frames_encoded = 0;
    let tot_blocks = packet_blocks(n_samples_in, fs_khz);

    // Each channel's LBRR flags left by the previous packet; a reset drops
    // them.
    let lbrr_possible = mid.s_cmn.use_in_band_fec != 0
        && mid.s_cmn.packet_loss_perc > 0
        && mid.s_cmn.lbrr_enabled != 0;
    let mut lbrr_symbols = [0i32; 2];
    for (symbol, ch) in lbrr_symbols.iter_mut().zip([&mut *mid, &mut *side]) {
        if ch.s_cmn.first_frame_after_reset != 0 {
            ch.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
        }
        if lbrr_possible {
            for i in 0..n_frames_per_packet as usize {
                *symbol |= ch.s_cmn.lbrr_flags[i] << i;
            }
        }
        ch.s_cmn.lbrr_flag = (*symbol > 0) as i8;
    }

    let n_flag_bits = ((n_frames_per_packet + 1) * 2) as u32;
    let mut lbrr_bits = 0i32;

    for frame_idx in 0..n_frames_per_packet {
        let fi = frame_idx as usize;
        if frame_idx == 0 {
            silk_hp_variable_cutoff(&mut mid.s_cmn);
        }

        // libopus `inputBuf` layout: two history slots, then the frame.
        let offset = fi * frame_length;
        let mut x1 = [0i16; MAX_FRAME_LENGTH + 2];
        let mut x2 = [0i16; MAX_FRAME_LENGTH + 2];
        x1[2..2 + frame_length].copy_from_slice(&left[offset..offset + frame_length]);
        x2[2..2 + frame_length].copy_from_slice(&right[offset..offset + frame_length]);

        if frame_idx == 0 {
            // Room for both channels' VAD and LBRR flags, patched in below.
            let icdf = [(256i32 - (256i32 >> n_flag_bits)) as u8, 0u8];
            rc.encode_icdf(0, &icdf, 8);

            if lbrr_symbols != [0, 0] {
                // Write the LBRR section only if the packet can hold it plus
                // a minimal frame, as the mono path does.
                let capacity_bits = (rc.storage as i32) * 8 - 8;
                let saved = [&*mid, &*side]
                    .map(|ch| (ch.s_cmn.ec_prev_signal_type, ch.s_cmn.ec_prev_lag_index));
                let mut trial = rc.clone();
                encode_lbrr_section_stereo(
                    &mut trial,
                    mid,
                    side,
                    lbrr_symbols,
                    n_frames_per_packet,
                );
                for (ch, saved) in [&mut *mid, &mut *side].into_iter().zip(saved) {
                    (ch.s_cmn.ec_prev_signal_type, ch.s_cmn.ec_prev_lag_index) = saved;
                }
                if trial.tell() + 64 <= capacity_bits {
                    lbrr_bits = encode_lbrr_section_stereo(
                        rc,
                        mid,
                        side,
                        lbrr_symbols,
                        n_frames_per_packet,
                    );
                } else {
                    mid.s_cmn.lbrr_flag = 0;
                    side.s_cmn.lbrr_flag = 0;
                }
            }
            mid.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
            side.s_cmn.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
        }

        // The frame's target, which the mid/side split shares out.
        let frame_rate_bps = silk_frame_target_rate_bps(
            mid,
            target_rate_bps,
            if frame_idx == 0 { lbrr_bits } else { 0 },
            rc.tell(),
        );
        let (mut ix, mut mid_only, mut rates) = ([[0i8; 3]; 2], 0i8, [0i32; 2]);
        silk_stereo_lr_to_ms(
            &mut mid.stereo,
            &mut x1,
            &mut x2,
            &mut ix,
            &mut mid_only,
            &mut rates,
            frame_rate_bps,
            mid.s_cmn.speech_activity_q8,
            false,
            fs_khz,
            frame_length,
        );
        mid.stereo.pred_ix[fi] = ix;
        mid.stereo.mid_only_flags[fi] = mid_only;
        let side_after_mid_only = mid.stereo.prev_decode_only_middle == 1;
        if mid_only == 0 {
            if side_after_mid_only {
                reset_side_for_coding(side);
            }
            silk_encode_do_vad(side, &x2[1..1 + frame_length], activity);
        } else {
            side.s_cmn.vad_flags[fi] = 0;
        }

        silk_stereo_encode_pred(rc, &ix);
        // A side with voice activity is coded, so the decoder knows the
        // frame isn't mid-only without the flag.
        if side.s_cmn.vad_flags[fi] == 0 {
            silk_stereo_encode_mid_only(rc, mid_only);
        }
        silk_encode_do_vad(mid, &x1[1..1 + frame_length], activity);

        for (n, (ch, buf)) in [(&mut *mid, &mut x1), (&mut *side, &mut x2)]
            .into_iter()
            .enumerate()
        {
            let mut channel_max_bits = silk_frame_max_bits(max_bits, tot_blocks, frame_idx);
            let mut channel_use_cbr = use_cbr != 0 && frame_idx == n_frames_per_packet - 1;
            if n == 0 && rates[1] > 0 {
                // The side fills the frame up to the cap; give the mid up to
                // half of the frame's bits.
                channel_use_cbr = false;
                channel_max_bits -= max_bits / (tot_blocks * 2);
            }
            if rates[n] > 0 {
                silk_control_snr(&mut ch.s_cmn, rates[n]);
                let cond_coding = if frame_idx == 0 {
                    CODE_INDEPENDENTLY
                } else if n == 1 && side_after_mid_only {
                    // The side's previous frame wasn't coded, but its LTP
                    // state is well-defined after the reset.
                    CODE_INDEPENDENTLY_NO_LTP_SCALING
                } else {
                    CODE_CONDITIONALLY
                };
                silk_lp_variable_cutoff(&mut ch.s_cmn.s_lp, &mut buf[1..], frame_length);
                let mut frame_bytes = 0i32;
                let ret = silk_encode_frame(
                    ch,
                    &buf[1..1 + frame_length],
                    rc,
                    &mut frame_bytes,
                    cond_coding,
                    channel_max_bits,
                    channel_use_cbr as i32,
                );
                if ret != 0 {
                    return ret;
                }
            }
            ch.s_cmn.n_frames_encoded += 1;
        }
        mid.stereo.prev_decode_only_middle = mid_only as i32;
    }

    // Each channel's VAD flags then its LBRR flag, mid first.
    let mut flags = 0u32;
    for ch in [&*mid, &*side] {
        for i in 0..n_frames_per_packet as usize {
            flags = (flags << 1) | ch.s_cmn.vad_flags[i] as u32;
        }
        flags = (flags << 1) | ch.s_cmn.lbrr_flag as u32;
    }
    rc.patch_initial_bits(flags, n_flag_bits);

    *n_bytes_out = (rc.tell() + 7) >> 3;
    silk_update_bit_reservoir(mid, target_rate_bps, *n_bytes_out);
    SILK_NO_ERROR
}

/// The LBRR section of a stereo packet (enc_API.c:352-389): each channel's
/// per-frame LBRR symbol for multi-frame packets, then the LBRR frames, mid
/// before side. A mid LBRR frame is preceded by the stereo header of the
/// frame it repeats, written in the previous packet and still in `pred_ix` /
/// `mid_only_flags`; its mid-only flag is implied when the side has LBRR too.
/// Returns the bits written.
fn encode_lbrr_section_stereo(
    rc: &mut RangeCoder,
    mid: &mut SilkEncoderState,
    side: &mut SilkEncoderState,
    lbrr_symbols: [i32; 2],
    n_frames_per_packet: i32,
) -> i32 {
    let start_bits = rc.tell();
    if n_frames_per_packet > 1 {
        for symbol in lbrr_symbols.into_iter().filter(|&s| s > 0) {
            rc.encode_icdf(symbol - 1, lbrr_flags_icdf(n_frames_per_packet), 8);
        }
    }
    for i in 0..n_frames_per_packet as usize {
        if mid.s_cmn.lbrr_flags[i] != 0 {
            silk_stereo_encode_pred(rc, &mid.stereo.pred_ix[i]);
            if side.s_cmn.lbrr_flags[i] == 0 {
                silk_stereo_encode_mid_only(rc, mid.stereo.mid_only_flags[i]);
            }
            encode_lbrr_frame(rc, mid, i);
        }
        if side.s_cmn.lbrr_flags[i] != 0 {
            encode_lbrr_frame(rc, side, i);
        }
    }
    rc.tell() - start_bits
}

/// One channel's LBRR frame `i`: indices, then pulses.
fn encode_lbrr_frame(rc: &mut RangeCoder, ch: &mut SilkEncoderState, i: usize) {
    let cond_coding = if i > 0 && ch.s_cmn.lbrr_flags[i - 1] != 0 {
        CODE_CONDITIONALLY
    } else {
        CODE_INDEPENDENTLY
    };
    silk_encode_indices(ch, rc, i, true, cond_coding);
    silk_encode_pulses(
        rc,
        ch.s_cmn.indices_lbrr[i].signal_type as i32,
        ch.s_cmn.indices_lbrr[i].quant_offset_type as i32,
        &ch.s_cmn.pulses_lbrr[i],
        ch.s_cmn.frame_length as usize,
    );
}

/// Reset the side encoder's memory for its first coded frame after frames
/// that coded the mid only, whose side was never coded (enc_API.c:441-453).
fn reset_side_for_coding(side: &mut SilkEncoderState) {
    side.s_shape = SilkShapeState::default();
    side.s_nsq = SilkNSQState::default();
    side.s_cmn.prev_nlsf_q15 = [0; MAX_LPC_ORDER];
    side.s_cmn.s_lp.in_lp_state = [0; 2];
    side.s_cmn.prev_lag = 100;
    side.s_nsq.lag_prev = 100;
    side.s_shape.last_gain_index = 10;
    side.s_cmn.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
    side.s_nsq.prev_gain_q16 = 65536;
    side.s_cmn.first_frame_after_reset = 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An encoder coding `packet_size_ms` packets of `n_frames` frames, with
    /// `n_frames_encoded` of them coded so far.
    fn encoder(packet_size_ms: i32, n_frames: i32, n_frames_encoded: i32) -> SilkEncoderState {
        let mut enc = SilkEncoderState::default();
        enc.s_cmn.packet_size_ms = packet_size_ms;
        enc.s_cmn.n_frames_per_packet = n_frames;
        enc.s_cmn.n_frames_encoded = n_frames_encoded;
        enc
    }

    #[test]
    fn frame_target_takes_out_the_lbrr_average() {
        let mut enc = encoder(20, 1, 0);
        // No LBRR: the packet's 640 bits.
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 0, 0), 32000);
        // The first LBRR packet isn't averaged: (640 - 200) * 50.
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 200, 0), 22000);
        assert_eq!(enc.n_bits_used_lbrr, 200);
        // Then the average: (200 + 100) / 2 = 150, (640 - 150) * 50.
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 100, 0), 24500);
        assert_eq!(enc.n_bits_used_lbrr, 150);
        // Under 10 bits counts as no LBRR, which drops the average at once.
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 9, 0), 32000);
        assert_eq!(enc.n_bits_used_lbrr, 0);
    }

    #[test]
    fn frame_target_pays_back_the_reservoir() {
        let mut enc = encoder(20, 1, 0);
        enc.n_bits_exceeded = 1000;
        // 1000 bits over 500 ms: 2000 b/s off.
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 0, 0), 30000);
        // 10 ms packets: (320 bits) * 100, less the same payback.
        let mut enc = encoder(10, 1, 0);
        enc.n_bits_exceeded = 1000;
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 0, 0), 30000);
    }

    #[test]
    fn later_frames_count_the_lbrr_section_as_overshoot() {
        // 40 ms at 32 kb/s: 1280 bits in two frames.
        let mut enc = encoder(40, 2, 0);
        // Frame 0: (1280 - 200) / 2 = 540 bits.
        assert_eq!(
            silk_frame_target_rate_bps(&mut enc, 32000, 200, 2 + 200),
            27000
        );
        // Frame 1 drops the LBRR average (libopus resets `curr_nBitsUsedLBRR`
        // per frame): 640 bits a frame, and the 2 flag bits, the 200-bit
        // section and frame 0's 540 bits are 102 over 640: 204 b/s off.
        enc.s_cmn.n_frames_encoded = 1;
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 0, 742), 31796);
        assert_eq!(enc.n_bits_used_lbrr, 0);
    }

    #[test]
    fn frame_target_stays_within_5000_and_the_input_rate() {
        // An LBRR section nearly as large as the packet: floored at 5000.
        let mut enc = encoder(20, 1, 0);
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 600, 0), 5000);
        // Frames under budget so far raise the target, but never above the
        // input rate.
        let mut enc = encoder(40, 2, 1);
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 32000, 0, 100), 32000);
        // Below 5000 the bounds swap, as in silk_LIMIT: 4000 stays 4000, and
        // a payback below the input rate stops at it.
        let mut enc = encoder(20, 1, 0);
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 4000, 0, 0), 4000);
        enc.n_bits_exceeded = 100;
        assert_eq!(silk_frame_target_rate_bps(&mut enc, 4000, 0, 0), 4000);
    }

    #[test]
    fn bit_reservoir_keeps_the_overshoot_within_0_and_10000() {
        // 32 kb/s at 20 ms: 640 bits a packet.
        let mut enc = encoder(20, 1, 0);
        silk_update_bit_reservoir(&mut enc, 32000, 90);
        assert_eq!(enc.n_bits_exceeded, 80);
        silk_update_bit_reservoir(&mut enc, 32000, 70);
        assert_eq!(enc.n_bits_exceeded, 0);
        silk_update_bit_reservoir(&mut enc, 32000, 70);
        assert_eq!(enc.n_bits_exceeded, 0, "an undershoot isn't banked");
        enc.n_bits_exceeded = 9900;
        silk_update_bit_reservoir(&mut enc, 32000, 200);
        assert_eq!(enc.n_bits_exceeded, 10000);
    }
}
