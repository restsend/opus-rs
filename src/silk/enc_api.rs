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

/// Perform VAD and set signal type flags.
/// Equivalent to C silk_encode_do_VAD_FIX().
/// Must be called before silk_encode_frame().
pub fn silk_encode_do_vad(
    ps_enc: &mut SilkEncoderState,
    input: &[i16],
    activity: i32, // Opus VAD activity decision (0 = inactive, 1 = active)
) {
    let activity_threshold = SPEECH_ACTIVITY_DTX_THRES_Q8;

    /****************************/
    /* Voice Activity Detection */
    /****************************/
    let frame_length = ps_enc.s_cmn.frame_length as usize;
    silk_vad_get_sa_q8(ps_enc, input, frame_length);

    /* If Opus VAD is inactive and Silk VAD is active: lower Silk VAD to just under threshold */
    if activity == 0 && ps_enc.s_cmn.speech_activity_q8 >= activity_threshold {
        ps_enc.s_cmn.speech_activity_q8 = activity_threshold - 1;
    }

    /**************************************************/
    /* Convert speech activity into VAD and DTX flags */
    /**************************************************/
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

/// Prefill the SILK encoder with silence samples to warm up the VAD state.
///
/// This exactly mirrors what C `silk_Encode` does when called with `prefillFlag=1`:
/// 1. Temporarily overrides frame_length to 10ms (80 samples @8kHz) since C forces
///    payloadSize_ms=10 for the prefill regardless of the real packet size.
/// 2. Runs VAD on the silence input (updating speech_activity_q8, input_quality_bands_q15)
/// 3. Runs LP variable cutoff filter
/// 4. Copies the filtered samples into x_buf (at x_frame + LA_SHAPE_MS * fs_kHz)
/// 5. Shifts x_buf left by prefill_frame_length (like C's memmove at end of encode_frame_FIX)
/// 6. Does NOT do pitch analysis, NSQ, or entropy coding.
/// 7. Restores original frame_length.
///
/// After this call the encoder's VAD state matches what C has after its prefill pass,
/// so the first real encoded frame will produce the same noise-shaping parameters.
pub fn silk_encode_prefill(
    ps_enc: &mut SilkEncoderState,
    samples: &[i16], // exactly 10ms of silence (fs_khz * 10 samples)
    activity: i32,   // Opus activity (0=inactive for silence)
) {
    // C forces payloadSize_ms=10 for the prefill, giving frame_length = fs_kHz * 10.
    // Save the real frame_length and override for the duration of the prefill.
    let fs_khz = ps_enc.s_cmn.fs_khz as usize;
    let prefill_frame_length = fs_khz * 10; // 10ms frame (80 samples @8kHz)
    let real_frame_length = ps_enc.s_cmn.frame_length as usize;
    let real_nb_subfr = ps_enc.s_cmn.nb_subfr;
    let real_subfr_length = ps_enc.s_cmn.subfr_length;

    // Temporarily set to 10ms frame parameters (matching C's payloadSize_ms=10 override)
    ps_enc.s_cmn.frame_length = prefill_frame_length as i32;
    // At 10ms we always have 2 subframes (5ms each)
    ps_enc.s_cmn.nb_subfr = 2;
    ps_enc.s_cmn.subfr_length = (prefill_frame_length / 2) as i32;

    let ltp_mem_length = ps_enc.s_cmn.ltp_mem_length as usize;
    // LA_SHAPE_MS is always 5 (a compile-time constant in C), not the
    // complexity-dependent la_shape field.
    let la_shape_ms_samples = 5 * fs_khz; // LA_SHAPE_MS * fs_kHz

    // --- Step 1: VAD on unfiltered data ---
    // C: silk_encode_do_VAD_Fxx(&psEnc->state_Fxx[0], activity)
    //    which calls silk_VAD_GetSA_Q8_FIX on inputBuf+1 (same as our samples slice)
    let n = prefill_frame_length.min(samples.len());
    silk_vad_get_sa_q8(ps_enc, &samples[..n], prefill_frame_length);

    // Silence forced: override VAD if needed (matching silk_encode_do_vad logic)
    let activity_threshold = crate::silk::define::SPEECH_ACTIVITY_DTX_THRES_Q8;
    if activity == 0 && ps_enc.s_cmn.speech_activity_q8 >= activity_threshold {
        ps_enc.s_cmn.speech_activity_q8 = activity_threshold - 1;
    }
    // Update signal_type (C: silk_encode_do_VAD_FIX sets this)
    if ps_enc.s_cmn.speech_activity_q8 < activity_threshold {
        ps_enc.s_cmn.indices.signal_type = crate::silk::define::TYPE_NO_VOICE_ACTIVITY as i8;
        ps_enc.s_cmn.no_speech_counter += 1;
    } else {
        ps_enc.s_cmn.no_speech_counter = 0;
        ps_enc.s_cmn.in_dtx = 0;
        ps_enc.s_cmn.indices.signal_type = crate::silk::define::TYPE_UNVOICED as i8;
    }

    // --- Step 2: LP variable cutoff filter ---
    // Work on a local buffer (C uses inputBuf+1 in-place)
    let mut input_buf = [0i16; super::define::MAX_FRAME_LENGTH + 2];
    // input_buf[0..2] = s_mid (overlap); for prefill these are zero (post-reset)
    input_buf[0] = ps_enc.s_mid[0];
    input_buf[1] = ps_enc.s_mid[1];
    input_buf[2..2 + n].copy_from_slice(&samples[..n]);
    // Save new overlap
    ps_enc.s_mid[0] = input_buf[prefill_frame_length];
    ps_enc.s_mid[1] = input_buf[prefill_frame_length + 1];
    // Apply LP filter (modifies input_buf[1..])
    silk_lp_variable_cutoff(
        &mut ps_enc.s_cmn.s_lp,
        &mut input_buf[1..],
        prefill_frame_length,
    );

    // --- Step 3: Copy filtered samples into x_buf ---
    // C: silk_memcpy(x_frame + LA_SHAPE_MS * fs_kHz, inputBuf + 1, frame_length)
    //    where x_frame = x_buf + ltp_mem_length
    let x_frame_idx = ltp_mem_length;
    let dst = x_frame_idx + la_shape_ms_samples;
    // Only write as many samples as we have (prefill_frame_length)
    if dst + prefill_frame_length <= ps_enc.s_cmn.x_buf.len() {
        ps_enc.s_cmn.x_buf[dst..dst + prefill_frame_length]
            .copy_from_slice(&input_buf[1..1 + prefill_frame_length]);
    }

    // --- Step 4: Update x_buf (shift left by prefill_frame_length) ---
    // C: silk_memmove(x_buf, &x_buf[frame_length], (ltp_mem_length + LA_SHAPE_MS*fs_kHz))
    let move_len = ltp_mem_length + la_shape_ms_samples;
    if prefill_frame_length + move_len <= ps_enc.s_cmn.x_buf.len() {
        ps_enc
            .s_cmn
            .x_buf
            .copy_within(prefill_frame_length..prefill_frame_length + move_len, 0);
    }

    // Update frame counter (C does this in encode_frame_FIX line 116)
    ps_enc.s_cmn.frame_counter += 1;
    // NOTE: In C, first_frame_after_reset is NOT cleared during prefill.
    // In C encode_frame_FIX, `first_frame_after_reset = 0` (line 379) comes
    // AFTER the prefill early return (line 365), so it remains 1 after prefill.
    // The real first encoded frame clears it. Do NOT set it to 0 here.
    // ps_enc.s_cmn.first_frame_after_reset = 0; // <-- wrong during prefill

    // --- Restore original frame parameters ---
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

    // Use persistent frame_counter for seed (not per-packet n_frames_encoded)
    ps_enc.s_cmn.indices.seed = (ps_enc.s_cmn.frame_counter & 3) as i8;
    ps_enc.s_cmn.frame_counter += 1;

    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let ltp_mem_length = ps_enc.s_cmn.ltp_mem_length as usize;
    let la_shape = ps_enc.s_cmn.la_shape as usize;

    /* start of frame to encode */
    let x_frame_idx = ltp_mem_length;

    /*******************************************/
    /* Copy new frame to front of input buffer */
    /*******************************************/
    /* LP variable cutoff is now applied in silk_encode() before this function,
     * matching the C flow where silk_LP_variable_cutoff is in silk_Encode, not
     * in silk_encode_frame_FIX. */
    /* In C: x_frame + LA_SHAPE_MS * psEnc->sCmn.fs_kHz is the start where new samples go */
    /* LA_SHAPE_MS = 5 is a fixed constant, NOT the complexity-dependent la_shape */
    let la_shape_max = 5 * ps_enc.s_cmn.fs_khz as usize; // LA_SHAPE_MS * fs_kHz
    let new_samples_idx = x_frame_idx + la_shape_max;
    ps_enc.s_cmn.x_buf[new_samples_idx..new_samples_idx + frame_length]
        .copy_from_slice(&input[..frame_length]);

    // Stack-copy x_buf before mutable borrows (avoids heap allocations and borrow checker conflicts)
    let x_buf_copy = ps_enc.s_cmn.x_buf; // array value copy (~1440 bytes on stack)

    let mut res_pitch = [0i16; LA_PITCH_MAX + MAX_FRAME_LENGTH + LTP_MEM_LENGTH_MS * MAX_FS_KHZ];
    let res_pitch_frame_idx = ltp_mem_length;

    /*****************************************/
    /* Find pitch lags, initial LPC analysis */
    /*****************************************/
    /* C passes x_frame - ltp_mem_length = x_buf as the x parameter */
    silk_find_pitch_lags_fix(ps_enc, &mut s_enc_ctrl, &mut res_pitch, &x_buf_copy, 0);

    /************************/
    /* Noise shape analysis */
    /************************/
    // C does: x_ptr = x_frame - la_shape inside noise_shape_analysis.
    // Since Rust can't do negative indexing, pass the wider buffer starting
    // la_shape samples before x_frame. noise_shape_analysis expects x[0]
    // to correspond to x_frame - la_shape.
    let x_tmp = &x_buf_copy[x_frame_idx - la_shape..];
    silk_noise_shape_analysis_fix(
        ps_enc,
        &mut s_enc_ctrl,
        &res_pitch[res_pitch_frame_idx..],
        x_tmp,
    );
    #[cfg(debug_assertions)]
    if std::env::var("SILK_DEBUG_NSQ").is_ok() {
        eprintln!(
            "  [ENC] after noise_shape: lf_shp_q14[0]={:#010x}",
            s_enc_ctrl.lf_shp_q14[0]
        );
    }

    /***************************************************/
    /* Find linear prediction coefficients (LPC + LTP) */
    /***************************************************/
    // C uses x = x_frame (x_buf[ltp_mem_length..]), with find_pred_coefs doing x_ptr = x - predict_lpc_order.
    // We pass x_buf[ltp_mem_length - predict_lpc_order..] so x_ptr_idx=0 in find_pred_coefs corresponds to
    // x_buf[ltp_mem_length - predict_lpc_order] = C's (x - predict_lpc_order).
    let predict_lpc_order = ps_enc.s_cmn.predict_lpc_order as usize;
    let x_tmp_frame = &x_buf_copy[x_frame_idx - predict_lpc_order..];
    silk_find_pred_coefs_fix(
        ps_enc,
        &mut s_enc_ctrl,
        &res_pitch[res_pitch_frame_idx..],
        x_tmp_frame,
        &x_buf_copy,
        cond_coding,
    );

    /****************************************/
    /* Process gains                        */
    /****************************************/
    silk_process_gains_fix(ps_enc, &mut s_enc_ctrl, cond_coding);
    #[cfg(debug_assertions)]
    if std::env::var("SILK_DEBUG_NSQ").is_ok() {
        eprintln!(
            "  [ENC] after process_gains: lf_shp_q14[0]={:#010x}",
            s_enc_ctrl.lf_shp_q14[0]
        );
    }

    /****************************************/
    /* Rate control loop                    */
    /****************************************/
    let max_iter = 6;
    let mut gain_mult_q8: i32 = 256; // SILK_FIX_CONST(1, 8)
    let mut found_lower = false;
    let mut found_upper = false;
    let mut n_bits: i32;
    let mut n_bits_lower: i32 = 0;
    let mut n_bits_upper: i32 = 0;
    let mut gain_mult_lower: i32 = 0;
    let mut gain_mult_upper: i32 = 0;
    let mut gains_id: i32 =
        silk_gains_id(&ps_enc.s_cmn.indices.gains_indices, ps_enc.s_cmn.nb_subfr);
    let mut gains_id_lower: i32 = -1;
    let mut gains_id_upper: i32 = -1;

    // For CBR, 5 bits below budget is close enough. For VBR, allow up to 25% below the cap.
    let bits_margin = if use_cbr != 0 { 5 } else { max_bits / 4 };

    // Backup state before loop
    let rc_copy = rc.clone();
    let nsq_copy = ps_enc.s_nsq.clone();
    let seed_copy = ps_enc.s_cmn.indices.seed;
    let ec_prev_lag_index_copy = ps_enc.s_cmn.ec_prev_lag_index;
    let ec_prev_signal_type_copy = ps_enc.s_cmn.ec_prev_signal_type;
    let mut rc_copy2: Option<RangeCoder> = None;
    let mut nsq_copy2: Option<SilkNSQState> = None;
    let mut ec_buf_copy = [0u8; 1275];
    let mut last_gain_index_copy2: i8 = 0;

    // Per-subframe gain locking (stack arrays - no heap needed for 4 elements)
    let mut gain_lock = [false; MAX_NB_SUBFR];
    let mut best_gain_mult = [256i32; MAX_NB_SUBFR];
    let mut best_sum = [i32::MAX; MAX_NB_SUBFR];

    for iter in 0..=max_iter {
        if gains_id == gains_id_lower {
            n_bits = n_bits_lower;
        } else if gains_id == gains_id_upper {
            n_bits = n_bits_upper;
        } else {
            // Restore state if not first iteration
            if iter > 0 {
                *rc = rc_copy.clone();
                ps_enc.s_nsq = nsq_copy.clone();
                ps_enc.s_cmn.indices.seed = seed_copy;
                ps_enc.s_cmn.ec_prev_lag_index = ec_prev_lag_index_copy;
                ps_enc.s_cmn.ec_prev_signal_type = ec_prev_signal_type_copy;
            }

            /****************************************/
            /* NSQ                                  */
            /****************************************/
            let mut pred_coef_q12_flat = [0i16; 2 * MAX_LPC_ORDER];
            pred_coef_q12_flat[..MAX_LPC_ORDER].copy_from_slice(&s_enc_ctrl.pred_coef_q12[0]);
            pred_coef_q12_flat[MAX_LPC_ORDER..].copy_from_slice(&s_enc_ctrl.pred_coef_q12[1]);

            // Debug: dump NSQ input parameters (guarded by env var)
            #[cfg(debug_assertions)]
            if std::env::var("SILK_DEBUG_NSQ").is_ok() && iter == 0 {
                eprintln!(
                    "  [ENC] iter=0 pre-dump: lf_shp_q14[0]={:#010x}",
                    s_enc_ctrl.lf_shp_q14[0]
                );
                eprintln!(
                    "=== NSQ INPUT DUMP (frame_counter={}) ===",
                    ps_enc.s_cmn.frame_counter
                );
                eprintln!(
                    "  signal_type={} quant_offset_type={}",
                    ps_enc.s_cmn.indices.signal_type, ps_enc.s_cmn.indices.quant_offset_type
                );
                eprintln!(
                    "  gains_q16={:?}",
                    &s_enc_ctrl.gains_q16[..ps_enc.s_cmn.nb_subfr as usize]
                );
                eprintln!("  lambda_q10={}", s_enc_ctrl.lambda_q10);
                eprintln!(
                    "  pred_coef_q12[0]={:?}",
                    &s_enc_ctrl.pred_coef_q12[0][..ps_enc.s_cmn.predict_lpc_order as usize]
                );
                eprintln!(
                    "  pred_coef_q12[1]={:?}",
                    &s_enc_ctrl.pred_coef_q12[1][..ps_enc.s_cmn.predict_lpc_order as usize]
                );
                eprintln!(
                    "  ar_q13[0..shaping]={:?}",
                    &s_enc_ctrl.ar_q13[..ps_enc.s_cmn.shaping_lpc_order as usize]
                );
                eprintln!(
                    "  tilt_q14={:?}",
                    &s_enc_ctrl.tilt_q14[..ps_enc.s_cmn.nb_subfr as usize]
                );
                eprintln!(
                    "  lf_shp_q14={:?}",
                    &s_enc_ctrl.lf_shp_q14[..ps_enc.s_cmn.nb_subfr as usize]
                );
                eprintln!(
                    "  harm_shape_gain_q14={:?}",
                    &s_enc_ctrl.harm_shape_gain_q14[..ps_enc.s_cmn.nb_subfr as usize]
                );
                eprintln!(
                    "  pitch_l={:?}",
                    &s_enc_ctrl.pitch_l[..ps_enc.s_cmn.nb_subfr as usize]
                );
                eprintln!(
                    "  x_buf[x_frame..+20]={:?}",
                    &ps_enc.s_cmn.x_buf[x_frame_idx..x_frame_idx + 20]
                );
                eprintln!(
                    "  la_shape={} la_shape_max={} new_samples_at={}",
                    la_shape,
                    la_shape_max,
                    x_frame_idx + la_shape_max
                );
                eprintln!(
                    "  x_buf[new_samples_idx..+20]={:?}",
                    &ps_enc.s_cmn.x_buf
                        [x_frame_idx + la_shape_max..x_frame_idx + la_shape_max + 20]
                );
                eprintln!(
                    "  n_states_delayed_decision={}",
                    ps_enc.s_cmn.n_states_delayed_decision
                );
            }

            if ps_enc.s_cmn.n_states_delayed_decision > 1 {
                silk_nsq_del_dec(
                    &ps_enc.s_cmn,
                    &mut ps_enc.s_nsq,
                    &ps_enc.s_cmn.indices,
                    // C: silk_NSQ_del_dec(psEncC, psEncNSQ, indices, x_frame, ...)
                    // x_frame = x_buf[ltp_mem_length..] — NSQ processes frame_length samples from x_frame
                    &ps_enc.s_cmn.x_buf[x_frame_idx..],
                    // NOTE: x_buf[x_frame_idx..x_frame_idx+LA_SHAPE] = previous frame's look-ahead (shared data)
                    // x_buf[x_frame_idx+LA_SHAPE..] = current frame samples
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
            } else {
                silk_nsq(
                    &ps_enc.s_cmn,
                    &mut ps_enc.s_nsq,
                    &ps_enc.s_cmn.indices,
                    // C: silk_NSQ(psEncC, psEncNSQ, indices, x_frame, ...)
                    // x_frame = x_buf[ltp_mem_length..]
                    &ps_enc.s_cmn.x_buf[x_frame_idx..],
                    // NOTE: x_buf[x_frame_idx..+LA_SHAPE] = previous frame look-ahead; rest = current frame
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

            // Save state at last iteration if we haven't found lower bound yet
            if iter == max_iter && !found_lower {
                rc_copy2 = Some(rc.clone());
            }

            /****************************************/
            /* Encode Indices                       */
            /****************************************/
            silk_encode_indices(
                ps_enc,
                rc,
                ps_enc.s_cmn.n_frames_encoded as usize,
                false,
                cond_coding,
            );

            /****************************************/
            /* Encode Excitation Signal             */
            /****************************************/
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

                // Keep gains the same as last frame
                ps_enc.s_shape.last_gain_index = s_enc_ctrl.last_gain_index_prev;
                for i in 0..ps_enc.s_cmn.nb_subfr as usize {
                    ps_enc.s_cmn.indices.gains_indices[i] = 4;
                }
                if cond_coding != CODE_CONDITIONALLY {
                    ps_enc.s_cmn.indices.gains_indices[0] = s_enc_ctrl.last_gain_index_prev as i8;
                }
                ps_enc.s_cmn.ec_prev_lag_index = ec_prev_lag_index_copy;
                ps_enc.s_cmn.ec_prev_signal_type = ec_prev_signal_type_copy;

                // Clear all pulses
                ps_enc.pulses.fill(0);

                // Re-encode with zero pulses
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

            // Rate control debug
            #[cfg(debug_assertions)]
            if std::env::var("SILK_DEBUG_NSQ").is_ok() && ps_enc.s_cmn.frame_counter <= 3 {
                eprintln!(
                    "  [RC] iter={} n_bits={} max_bits={} found_lower={} found_upper={} gain_mult_q8={}",
                    iter, n_bits, max_bits, found_lower, found_upper, gain_mult_q8
                );
            }

            // VBR: if first iteration and within budget, stop
            if use_cbr == 0 && iter == 0 && n_bits <= max_bits {
                break;
            }
        }

        // Exit after last iteration
        if iter == max_iter {
            if found_lower && (gains_id == gains_id_lower || n_bits > max_bits) {
                // Restore from earlier iteration that met budget
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

        // Adjust strategy based on bit usage
        if n_bits > max_bits {
            if !found_lower && iter >= 2 {
                // Increase lambda to reduce rate
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
                // Save state
                rc_copy2 = Some(rc.clone());
                let offs = rc.offs as usize;
                ec_buf_copy[..offs].copy_from_slice(&rc.buf[..offs]);
                nsq_copy2 = Some(ps_enc.s_nsq.clone());
                last_gain_index_copy2 = ps_enc.s_shape.last_gain_index;
            }
        } else {
            // Close enough
            break;
        }

        // Track best gain per subframe when over budget
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

        // Adjust gain multiplier for next iteration
        // C: if( ( found_lower & found_upper ) == 0 ) — true when either bound missing
        if !(found_lower && found_upper) {
            // Adjust based on high-rate rate/distortion curve
            if n_bits > max_bits {
                gain_mult_q8 = silk_min_32(1024, (gain_mult_q8 * 3) / 2);
            } else {
                gain_mult_q8 = silk_max_32(64, (gain_mult_q8 * 4) / 5);
            }
        } else {
            // Binary search between bounds
            let delta = gain_mult_upper - gain_mult_lower;
            gain_mult_q8 = gain_mult_lower
                + silk_div32_16(
                    (gain_mult_upper - gain_mult_lower) * (max_bits - n_bits_lower),
                    n_bits_upper - n_bits_lower,
                ) as i32;
            // Clamp to 25%-75% of old range
            let lower_limit = silk_add_rshift32(gain_mult_lower, delta, 2);
            let upper_limit = silk_sub_rshift32(gain_mult_upper, delta, 2);
            if gain_mult_q8 > lower_limit {
                gain_mult_q8 = lower_limit;
            }
            if gain_mult_q8 < upper_limit {
                gain_mult_q8 = upper_limit;
            }
        }

        // Apply per-subframe gain multiplier and requantize gains
        for i in 0..ps_enc.s_cmn.nb_subfr as usize {
            let tmp = if gain_lock[i] {
                best_gain_mult[i]
            } else {
                gain_mult_q8
            };
            s_enc_ctrl.gains_q16[i] =
                silk_lshift_sat32(silk_smulwb(s_enc_ctrl.gains_unq_q16[i], tmp), 8);
        }

        // Quantize gains
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

        // Compute unique gains identifier
        gains_id = silk_gains_id(&ps_enc.s_cmn.indices.gains_indices, ps_enc.s_cmn.nb_subfr);
    }

    /* Update input buffer */
    // C: ltp_mem_length + LA_SHAPE_MS * fs_kHz (fixed constant, not complexity-dependent la_shape)
    let move_len = ltp_mem_length + 5 * ps_enc.s_cmn.fs_khz as usize;
    ps_enc
        .s_cmn
        .x_buf
        .copy_within(frame_length..frame_length + move_len, 0);

    /* Parameters needed for next frame */
    ps_enc.s_cmn.prev_lag = s_enc_ctrl.pitch_l[ps_enc.s_cmn.nb_subfr as usize - 1];
    ps_enc.s_cmn.prev_signal_type = ps_enc.s_cmn.indices.signal_type as i32;
    ps_enc.s_cmn.first_frame_after_reset = 0;

    *pn_bytes_out = (rc.tell() + 7) >> 3;
    0
}

/// Top-level SILK encoding function.
/// Equivalent to C `silk_Encode()` in enc_API.c.
///
/// Handles:
/// - Input buffering and resampling (simplified: direct copy for matching sample rates)
/// - VAD/FEC flag preamble encoding
/// - Per-frame encoding loop for multi-frame packets
/// - LBRR flag encoding (stub: no LBRR for now)
/// - SNR control
/// - HP variable cutoff
/// - VAD flag patching at start of bitstream
///
/// # Arguments
/// * `ps_enc` - SILK encoder state
/// * `samples_in` - Input PCM samples (i16)
/// * `n_samples_in` - Number of input samples
/// * `rc` - Range coder for output
/// * `n_bytes_out` - Output: number of bytes produced
/// * `target_rate_bps` - Target bitrate in bits per second
/// * `max_bits` - Maximum bits for this packet
/// * `use_cbr` - Whether to use CBR mode
/// * `activity` - Opus VAD activity decision (0=inactive, 1=active)
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

    // Reset frame counter for this packet
    ps_enc.s_cmn.n_frames_encoded = 0;

    // Compute number of 10ms blocks and total frames
    let n_blocks_of_10ms = (100 * n_samples_in as i32) / (ps_enc.s_cmn.fs_khz * 1000);
    let _tot_blocks = if n_blocks_of_10ms > 1 {
        n_blocks_of_10ms >> 1
    } else {
        1
    };

    // Compute per-frame target rate
    let n_bits_total = target_rate_bps * packet_size_ms / 1000;
    let n_bits_per_frame = n_bits_total / n_frames_per_packet;
    let frame_rate_bps = if packet_size_ms == 10 {
        n_bits_per_frame * 100
    } else {
        n_bits_per_frame * 50
    };

    // Determine if LBRR should be active this packet.
    // LBRR is enabled when use_in_band_fec=1 and packet_loss_perc > 0.
    // The LBRR frames were prepared during the PREVIOUS call by saving the main
    // frame indices into indices_lbrr[]. We use lbrr_enabled from the struct
    // (which the caller sets by storing use_in_band_fec there).
    // IMPORTANT: Only activate LBRR if saved indices have valid voice activity
    // (signal_type >= TYPE_UNVOICED). On the first packet, no previous data exists.
    let lbrr_possible = ps_enc.s_cmn.use_in_band_fec != 0
        && ps_enc.s_cmn.packet_loss_perc > 0
        && ps_enc.s_cmn.lbrr_enabled != 0;

    // Compute lbrr_symbol: which frames have valid saved LBRR data
    // A frame has valid LBRR if its signal_type >= TYPE_UNVOICED (not silence)
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
    // Copy lbrr_flags from the bitmask for later use in encoding and flag patching
    for i in 0..n_frames_per_packet as usize {
        ps_enc.s_cmn.lbrr_flags[i] = (lbrr_symbol >> i) & 1;
    }

    // Iterate over frames in this packet
    let mut sample_offset = 0usize;

    for frame_idx in 0..n_frames_per_packet {
        // --- HP variable cutoff ---
        if frame_idx == 0 {
            silk_hp_variable_cutoff(&mut ps_enc.s_cmn);
        }

        // Get input samples for this frame
        let frame_end = (sample_offset + frame_length).min(n_samples_in);
        let raw_frame = &samples_in[sample_offset..frame_end];

        // --- Input buffering with 2-sample overlap (C: sStereo.sMid) ---
        // C flow (enc_API.c):
        //   silk_resampler (same-rate copy, inputDelay from delay_matrix_enc):
        //     1) delayBuf[inputDelay..Fs_in_kHz] = in[0..nSamples]
        //     2) out[0..Fs_in_kHz] = delayBuf (now updated)
        //     3) out[Fs_out_kHz..n] = in[nSamples..n-Fs_in_kHz+nSamples]
        //     4) delayBuf[0..inputDelay] = in[n-inputDelay..n]
        //   inputBuf[0..2] = sMid (overlap from previous frame)
        //   inputBuf[2..2+N] = resampled out
        //   sMid = inputBuf[frame_length..frame_length+2]
        //   encode_frame uses inputBuf[1..frame_length+1]
        // delay_matrix_enc[rate][rate]: 8kHz=6, 12kHz=7, 16kHz=10
        let fs_in_khz = ps_enc.s_cmn.fs_khz as usize;
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
        let mut resampler_out = [0i16; MAX_FRAME_LENGTH];

        // Step 1: delayBuf[inputDelay..Fs_in_kHz] = in[0..nSamples]
        let mut delay_buf = ps_enc.resampler_delay_buf;
        delay_buf[input_delay..fs_in_khz].copy_from_slice(&raw_frame[..n_samp]);

        // Step 2: out[0..Fs_in_kHz] = delayBuf (now updated)
        resampler_out[..fs_in_khz].copy_from_slice(&delay_buf[..fs_in_khz]);

        // Step 3: out[Fs_out_kHz..n] = in[nSamples..]; length = inLen - Fs_in_kHz
        let rest_len = n - fs_in_khz;
        resampler_out[fs_in_khz..n].copy_from_slice(&raw_frame[n_samp..n_samp + rest_len]);

        // Step 4: delayBuf[0..inputDelay] = in[n-inputDelay..n]
        delay_buf[..input_delay].copy_from_slice(&raw_frame[n - input_delay..]);
        ps_enc.resampler_delay_buf = delay_buf;

        // inputBuf[0..2] = sMid, inputBuf[2..2+n] = resampler_out
        let mut input_buf = [0i16; MAX_FRAME_LENGTH + 2];
        input_buf[0] = ps_enc.s_mid[0];
        input_buf[1] = ps_enc.s_mid[1];
        input_buf[2..2 + n].copy_from_slice(&resampler_out[..n]);

        // Save last 2 samples for next frame's overlap BEFORE LP filter (matching C)
        // C: silk_memcpy(sStereo.sMid, &inputBuf[frame_length], 2)
        // This happens before VAD and LP filter in C.
        ps_enc.s_mid[0] = input_buf[frame_length];
        ps_enc.s_mid[1] = input_buf[frame_length + 1];

        // --- LBRR preamble encoding (first frame only) ---
        if frame_idx == 0 {
            // Create space at start of payload for VAD and FEC flags
            let n_flag_bits = (n_frames_per_packet + 1) as u32; // nFramesPerPacket + 1 for LBRR
            let icdf_val = (256i32 - (256i32 >> n_flag_bits)) as u8;
            let icdf = [icdf_val, 0u8];
            rc.encode_icdf(0, &icdf, 8);

            // Encode LBRR flags
            if lbrr_symbol > 0 {
                // Encode LBRR symbol into range coder
                // C: ec_enc_icdf(psRangeEnc, LBRR_symbol - 1, silk_LBRR_flags_iCDF_ptr[nFramesPerPacket-2], 8)
                let lbrr_icdf = match n_frames_per_packet {
                    2 => &crate::silk::tables::SILK_LBRR_FLAGS_2_ICDF[..],
                    3 => &crate::silk::tables::SILK_LBRR_FLAGS_3_ICDF[..],
                    _ => &crate::silk::tables::SILK_LBRR_FLAGS_2_ICDF[..],
                };
                if n_frames_per_packet > 1 {
                    rc.encode_icdf(lbrr_symbol - 1, lbrr_icdf, 8);
                }
                // For each frame with LBRR, encode its indices and pulses
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

        // --- SNR control ---
        silk_control_snr(&mut ps_enc.s_cmn, frame_rate_bps);

        // --- VAD on unfiltered data (C calls VAD before LP filter) ---
        // C: silk_encode_do_VAD_Fxx(&psEnc->state_Fxx[0], activity) on inputBuf+1
        let vad_frame = &input_buf[1..1 + frame_length];
        silk_encode_do_vad(ps_enc, vad_frame, activity);

        // --- Apply LP variable cutoff filter (C: inside silk_encode_frame_FIX) ---
        // C: silk_LP_variable_cutoff(&psEnc->sCmn.sLP, inputBuf+1, frame_length)
        silk_lp_variable_cutoff(&mut ps_enc.s_cmn.s_lp, &mut input_buf[1..], frame_length);

        // The actual frame data is inputBuf[1..frame_length+1] (160 samples)
        // Now LP-filtered, ready for silk_encode_frame
        let frame_samples = &input_buf[1..1 + frame_length];

        // --- Conditional coding ---
        let cond_coding = if ps_enc.s_cmn.n_frames_encoded == 0 {
            CODE_INDEPENDENTLY
        } else {
            CODE_CONDITIONALLY
        };

        // --- Encode frame ---
        let frame_max_bits = if _tot_blocks == 2 && frame_idx == 0 {
            max_bits * 3 / 5
        } else {
            max_bits
        };
        #[cfg(debug_assertions)]
        if std::env::var("SILK_DEBUG_NSQ").is_ok() {
            eprintln!(
                "  [SILK_ENC] frame_idx={} frame_max_bits={} max_bits={} tot_blocks={} cond_coding={}",
                frame_idx, frame_max_bits, max_bits, _tot_blocks, cond_coding
            );
        }

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

        // --- Save indices for LBRR use in the next packet ---
        // After encoding each frame, save its indices as LBRR data for the next packet.
        // This matches the C silk_LBRR_encode() approach where previous frame
        // data is re-used at lower bitrate in the next packet.
        if use_lbrr || ps_enc.s_cmn.use_in_band_fec != 0 {
            let fi = frame_idx as usize;
            if fi < MAX_FRAMES_PER_PACKET {
                ps_enc.s_cmn.indices_lbrr[fi] = ps_enc.s_cmn.indices;
                // Apply LBRR gain increase: raise gains_indices by lbrr_gain_increases
                // This reduces quality but also reduces bits for the LBRR copy.
                let gain_inc = ps_enc.s_cmn.lbrr_gain_increases.max(0).min(16) as i8;
                for g in 0..ps_enc.s_cmn.nb_subfr as usize {
                    let new_gain = (ps_enc.s_cmn.indices_lbrr[fi].gains_indices[g] as i32
                        + gain_inc as i32)
                        .min(63) as i8;
                    ps_enc.s_cmn.indices_lbrr[fi].gains_indices[g] = new_gain;
                }
                // Store pulses
                ps_enc.s_cmn.pulses_lbrr[fi] = ps_enc.pulses;
            }
        }

        ps_enc.s_cmn.n_frames_encoded += 1;
        sample_offset += frame_length;
    }

    // --- Patch VAD and FEC flags at beginning of bitstream ---
    let n_flag_bits = (n_frames_per_packet + 1) as u32;
    let mut flags = 0u32;
    for i in 0..n_frames_per_packet as usize {
        flags <<= 1;
        flags |= ps_enc.s_cmn.vad_flags[i] as u32;
    }
    flags <<= 1;
    flags |= ps_enc.s_cmn.lbrr_flag as u32;

    rc.patch_initial_bits(flags, n_flag_bits);

    // Output bytes
    *n_bytes_out = (rc.tell() + 7) >> 3;

    0
}
