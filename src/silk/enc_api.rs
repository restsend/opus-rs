use crate::range_coder::RangeCoder;
use crate::silk::control_fixed::*;
use crate::silk::control_snr::silk_control_snr;
use crate::silk::define::*;
use crate::silk::encode_indices::*;
use crate::silk::encode_pulses::*;
use crate::silk::gain_quant::{silk_gains_id, silk_gains_quant};
use crate::silk::hp_variable_cutoff::silk_hp_variable_cutoff;
use crate::silk::lp_variable_cutoff::*;
use crate::silk::macros::*;
use crate::silk::noise_shape_analysis::*;
use crate::silk::nsq::*;
use crate::silk::nsq_del_dec::*;
use crate::silk::pitch_analysis::*;
use crate::silk::structs::*;
use crate::silk::vad::silk_vad_get_sa_q8;

pub fn silk_encode_do_vad(
    ps_enc: &mut SilkEncoderState,
    input: &[i16],
    activity: i32,
) {
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

pub fn silk_encode_prefill(
    ps_enc: &mut SilkEncoderState,
    samples: &[i16],
    _activity: i32,
) {

    let fs_khz = ps_enc.s_cmn.fs_khz as usize;

    if fs_khz != 8 && fs_khz != 12 && fs_khz != 16 {
        return;
    }

    let prefill_frame_length = fs_khz * 10;

    if samples.len() < prefill_frame_length {
        return;
    }

    let real_frame_length = ps_enc.s_cmn.frame_length as usize;
    let real_nb_subfr = ps_enc.s_cmn.nb_subfr;
    let real_subfr_length = ps_enc.s_cmn.subfr_length;

    ps_enc.s_cmn.frame_length = prefill_frame_length as i32;
    ps_enc.s_cmn.nb_subfr = 2;
    ps_enc.s_cmn.subfr_length = (prefill_frame_length / 2) as i32;

    let ltp_mem_length = ps_enc.s_cmn.ltp_mem_length as usize;
    let la_shape_ms_samples = 5 * fs_khz;

    let n = prefill_frame_length.min(samples.len());

    let mut input_buf = [0i16; super::define::MAX_FRAME_LENGTH + 2];
    input_buf[0] = ps_enc.stereo.s_mid[0];
    input_buf[1] = ps_enc.stereo.s_mid[1];
    input_buf[2..2 + n].copy_from_slice(&samples[..n]);
    ps_enc.stereo.s_mid[0] = input_buf[prefill_frame_length];
    ps_enc.stereo.s_mid[1] = input_buf[prefill_frame_length + 1];

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
    let nsq_copy = ps_enc.s_nsq.clone();
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
                ps_enc.s_nsq = nsq_copy.clone();
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

            n_bits = rc.tell() as i32;

            if iter == max_iter && !found_lower && n_bits > max_bits {
                if let Some(rc_c2) = &rc_copy2 {
                    *rc = rc_c2.clone();
                }

                ps_enc.s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
                for i in 0..ps_enc.s_cmn.nb_subfr as usize {
                    ps_enc.s_cmn.indices.gains_indices[i] = 4;
                }
                if cond_coding != CODE_CONDITIONALLY {
                    ps_enc.s_cmn.indices.gains_indices[0] = s_enc_ctrl.last_gain_index_prev as i8;
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

                n_bits = rc.tell() as i32;
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
                nsq_copy2 = Some(ps_enc.s_nsq.clone());
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
                ) as i32;

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
    let n_frames_per_packet = ps_enc.s_cmn.n_frames_per_packet;
    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let packet_size_ms = ps_enc.s_cmn.packet_size_ms;

    ps_enc.s_cmn.n_frames_encoded = 0;

    let n_blocks_of_10ms = (100 * n_samples_in as i32) / (ps_enc.s_cmn.fs_khz * 1000);
    let _tot_blocks = if n_blocks_of_10ms > 1 {
        n_blocks_of_10ms >> 1
    } else {
        1
    };

    let n_bits_total = target_rate_bps * packet_size_ms / 1000;
    let n_bits_per_frame = n_bits_total / n_frames_per_packet;
    let frame_rate_bps = if packet_size_ms == 10 {
        n_bits_per_frame * 100
    } else {
        n_bits_per_frame * 50
    };

    let lbrr_possible = ps_enc.s_cmn.use_in_band_fec != 0
        && ps_enc.s_cmn.packet_loss_perc > 0
        && ps_enc.s_cmn.lbrr_enabled != 0;

    let mut lbrr_symbol: i32 = 0;
    if lbrr_possible {
        for i in 0..n_frames_per_packet as usize {
            if ps_enc.s_cmn.indices_lbrr[i].signal_type >= TYPE_UNVOICED as i8 {
                lbrr_symbol |= 1 << i;
            }
        }
    }
    let use_lbrr = lbrr_symbol > 0;

    ps_enc.s_cmn.lbrr_flag = if lbrr_symbol > 0 { 1 } else { 0 };

    for i in 0..n_frames_per_packet as usize {
        ps_enc.s_cmn.lbrr_flags[i] = (lbrr_symbol >> i) & 1;
    }

    let mut sample_offset = 0usize;

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

        let input_delay: usize = match fs_in_khz {
            8 => 6,
            12 => 7,
            16 => 10,
            24 => 6,
            48 => 12,
            _ => 0,
        };
        let n_samp: usize = fs_in_khz - input_delay;

        let n = raw_frame.len();

        if n > MAX_FRAME_LENGTH {
            sample_offset += frame_length;
            continue;
        }

        let mut resampler_out = [0i16; MAX_FRAME_LENGTH];

        let mut delay_buf = ps_enc.resampler_delay_buf;
        delay_buf[input_delay..fs_in_khz].copy_from_slice(&raw_frame[..n_samp]);

        resampler_out[..fs_in_khz].copy_from_slice(&delay_buf[..fs_in_khz]);

        let rest_len = n - fs_in_khz;
        let rest_end = n_samp + rest_len;

        if rest_end <= raw_frame.len() && n <= MAX_FRAME_LENGTH {
            resampler_out[fs_in_khz..n].copy_from_slice(&raw_frame[n_samp..rest_end]);
        }

        if n >= input_delay {
            delay_buf[..input_delay].copy_from_slice(&raw_frame[n - input_delay..]);
        }
        ps_enc.resampler_delay_buf = delay_buf;

        let mut input_buf = [0i16; MAX_FRAME_LENGTH + 2];
        input_buf[0] = ps_enc.stereo.s_mid[0];
        input_buf[1] = ps_enc.stereo.s_mid[1];
        input_buf[2..2 + n].copy_from_slice(&resampler_out[..n]);

        ps_enc.stereo.s_mid[0] = input_buf[frame_length];
        ps_enc.stereo.s_mid[1] = input_buf[frame_length + 1];

        if frame_idx == 0 {

            let n_flag_bits = (n_frames_per_packet + 1) as u32;
            let icdf_val = (256i32 - (256i32 >> n_flag_bits)) as u8;
            let icdf = [icdf_val, 0u8];
            rc.encode_icdf(0, &icdf, 8);

            if lbrr_symbol > 0 {

                let lbrr_icdf = match n_frames_per_packet {
                    2 => &crate::silk::tables::SILK_LBRR_FLAGS_2_ICDF[..],
                    3 => &crate::silk::tables::SILK_LBRR_FLAGS_3_ICDF[..],
                    _ => &crate::silk::tables::SILK_LBRR_FLAGS_2_ICDF[..],
                };
                if n_frames_per_packet > 1 {
                    rc.encode_icdf(lbrr_symbol - 1, lbrr_icdf, 8);
                }

                for i in 0..n_frames_per_packet as usize {
                    if ps_enc.s_cmn.lbrr_flags[i] != 0 {
                        let lbrr_cond = if i > 0 && ps_enc.s_cmn.lbrr_flags[i - 1] != 0 {
                            CODE_CONDITIONALLY
                        } else {
                            CODE_INDEPENDENTLY_NO_LTP_SCALING
                        };
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
            }
        }

        silk_control_snr(&mut ps_enc.s_cmn, frame_rate_bps);

        let vad_frame = &input_buf[1..1 + frame_length];
        silk_encode_do_vad(ps_enc, vad_frame, activity);

        silk_lp_variable_cutoff(&mut ps_enc.s_cmn.s_lp, &mut input_buf[1..], frame_length);

        let frame_samples = &input_buf[1..1 + frame_length];

        let cond_coding = if ps_enc.s_cmn.n_frames_encoded == 0 {
            CODE_INDEPENDENTLY
        } else {
            CODE_CONDITIONALLY
        };

        let frame_max_bits = if _tot_blocks == 2 && frame_idx == 0 {
            max_bits * 3 / 5
        } else {
            max_bits
        };

        let mut frame_bytes = 0i32;
        let ret = silk_encode_frame(
            ps_enc,
            frame_samples,
            rc,
            &mut frame_bytes,
            cond_coding,
            frame_max_bits,
            if use_cbr != 0 && frame_idx == n_frames_per_packet - 1 {
                1
            } else {
                0
            },
        );
        if ret != 0 {
            return ret;
        }

        if use_lbrr || ps_enc.s_cmn.use_in_band_fec != 0 {
            let fi = frame_idx as usize;
            if fi < MAX_FRAMES_PER_PACKET {
                ps_enc.s_cmn.indices_lbrr[fi] = ps_enc.s_cmn.indices;

                let gain_inc = ps_enc.s_cmn.lbrr_gain_increases.max(0).min(16) as i8;
                for g in 0..ps_enc.s_cmn.nb_subfr as usize {
                    let new_gain = (ps_enc.s_cmn.indices_lbrr[fi].gains_indices[g] as i32
                        + gain_inc as i32)
                        .min(63) as i8;
                    ps_enc.s_cmn.indices_lbrr[fi].gains_indices[g] = new_gain;
                }

                ps_enc.s_cmn.pulses_lbrr[fi] = ps_enc.pulses;
            }
        }

        ps_enc.s_cmn.n_frames_encoded += 1;
        sample_offset += frame_length;
    }

    let n_flag_bits = (n_frames_per_packet + 1) as u32;
    let mut flags = 0u32;
    for i in 0..n_frames_per_packet as usize {
        flags <<= 1;
        flags |= ps_enc.s_cmn.vad_flags[i] as u32;
    }
    flags <<= 1;
    flags |= ps_enc.s_cmn.lbrr_flag as u32;

    rc.patch_initial_bits(flags, n_flag_bits);

    *n_bytes_out = (rc.tell() + 7) >> 3;

    0
}
