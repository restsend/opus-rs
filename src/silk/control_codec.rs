use crate::silk::define::*;
use crate::silk::structs::*;
use crate::silk::tables_nlsf::*;

// Constants from C silk/define.h
pub const FIND_PITCH_LPC_WIN_MS: i32 = 20 + (LA_PITCH_MS as i32 * 2);
pub const FIND_PITCH_LPC_WIN_MS_2_SF: i32 = 10 + (LA_PITCH_MS as i32 * 2);
pub const MAX_DEL_DEC_STATES: i32 = 4;
pub const LA_SHAPE_MAX: i32 = LA_SHAPE_MS as i32 * MAX_FS_KHZ as i32;
pub const SHAPE_LPC_WIN_MAX: i32 = 15 * MAX_FS_KHZ as i32;

// Warping multiplier in Q16 (from tuning_parameters: WARPING_MULTIPLIER = 0.015)
// 0.015 * 65536 = 983
pub const WARPING_MULTIPLIER_Q16: i32 = 983;

/// Setup sampling-rate-dependent encoder parameters.
/// Equivalent to C silk_setup_fs().
pub fn silk_setup_fs(ps_enc: &mut SilkEncoderState, fs_khz: i32, packet_size_ms: i32) -> i32 {
    let cmn = &mut ps_enc.s_cmn;

    /* Set packet size params */
    if packet_size_ms <= 10 {
        cmn.n_frames_per_packet = 1;
        cmn.nb_subfr = if packet_size_ms == 10 { 2 } else { 1 };
        cmn.frame_length = packet_size_ms * fs_khz;
        cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS_2_SF * fs_khz;
    } else {
        cmn.n_frames_per_packet = packet_size_ms / MAX_FRAME_LENGTH_MS as i32;
        cmn.nb_subfr = MAX_NB_SUBFR as i32;
        cmn.frame_length = 20 * fs_khz;
        cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS * fs_khz;
    }
    cmn.packet_size_ms = packet_size_ms;

    /* Set internal sampling frequency */
    if cmn.fs_khz != fs_khz {
        /* Reset state on FS change */
        ps_enc.s_nsq = SilkNSQState::default();
        cmn.prev_nlsf_q15 = [0; MAX_LPC_ORDER];

        /* Initialize non-zero parameters */
        cmn.prev_lag = 100;
        cmn.first_frame_after_reset = 1;
        ps_enc.s_shape.last_gain_index = 10;
        ps_enc.s_nsq.lag_prev = 100;
        ps_enc.s_nsq.prev_gain_q16 = 65536;
        cmn.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;

        cmn.fs_khz = fs_khz;

        if fs_khz == 8 || fs_khz == 12 {
            cmn.predict_lpc_order = MIN_LPC_ORDER as i32;
            ps_enc.ps_nlsf_cb = Some(&SILK_NLSF_CB_NB_MB);
        } else {
            cmn.predict_lpc_order = MAX_LPC_ORDER as i32;
            ps_enc.ps_nlsf_cb = Some(&SILK_NLSF_CB_WB);
        }

        cmn.subfr_length = SUB_FRAME_LENGTH_MS as i32 * fs_khz;
        cmn.frame_length = cmn.subfr_length * cmn.nb_subfr;
        cmn.ltp_mem_length = LTP_MEM_LENGTH_MS as i32 * fs_khz;
        cmn.la_pitch = LA_PITCH_MS as i32 * fs_khz;
        // max_pitch_lag not in struct yet, but should be 18 * fs_khz
        if cmn.nb_subfr == MAX_NB_SUBFR as i32 {
            cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS * fs_khz;
        } else {
            cmn.pitch_lpc_win_length = FIND_PITCH_LPC_WIN_MS_2_SF * fs_khz;
        }
    }

    SILK_NO_ERROR
}

/// Setup complexity-dependent encoder parameters.
/// Equivalent to C silk_setup_complexity().
pub fn silk_setup_complexity(ps_enc: &mut SilkEncoderState, complexity: i32) -> i32 {
    let cmn = &mut ps_enc.s_cmn;

    if complexity < 1 {
        cmn.pitch_estimation_complexity = SILK_PE_MIN_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = (0.8f32 * 65536.0) as i32;
        ps_enc.pitch_estimation_lpc_order = 6;
        cmn.shaping_lpc_order = 12;
        cmn.la_shape = 3 * cmn.fs_khz;
        cmn.n_states_delayed_decision = 1;
        cmn.use_interpolated_nlsfs = 0;
        cmn.n_nlsf_survivors = 2;
        cmn.warping_q16 = 0;
    } else if complexity < 2 {
        cmn.pitch_estimation_complexity = SILK_PE_MID_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = (0.76f32 * 65536.0) as i32;
        ps_enc.pitch_estimation_lpc_order = 8;
        cmn.shaping_lpc_order = 14;
        cmn.la_shape = 5 * cmn.fs_khz;
        cmn.n_states_delayed_decision = 1;
        cmn.use_interpolated_nlsfs = 0;
        cmn.n_nlsf_survivors = 3;
        cmn.warping_q16 = 0;
    } else if complexity < 3 {
        cmn.pitch_estimation_complexity = SILK_PE_MIN_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = (0.8f32 * 65536.0) as i32;
        ps_enc.pitch_estimation_lpc_order = 6;
        cmn.shaping_lpc_order = 12;
        cmn.la_shape = 3 * cmn.fs_khz;
        cmn.n_states_delayed_decision = 2;
        cmn.use_interpolated_nlsfs = 0;
        cmn.n_nlsf_survivors = 2;
        cmn.warping_q16 = 0;
    } else if complexity < 4 {
        cmn.pitch_estimation_complexity = SILK_PE_MID_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = (0.76f32 * 65536.0) as i32;
        ps_enc.pitch_estimation_lpc_order = 8;
        cmn.shaping_lpc_order = 14;
        cmn.la_shape = 5 * cmn.fs_khz;
        cmn.n_states_delayed_decision = 2;
        cmn.use_interpolated_nlsfs = 0;
        cmn.n_nlsf_survivors = 4;
        cmn.warping_q16 = 0;
    } else if complexity < 6 {
        cmn.pitch_estimation_complexity = SILK_PE_MID_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = (0.74f32 * 65536.0) as i32;
        ps_enc.pitch_estimation_lpc_order = 10;
        cmn.shaping_lpc_order = 16;
        cmn.la_shape = 5 * cmn.fs_khz;
        cmn.n_states_delayed_decision = 2;
        cmn.use_interpolated_nlsfs = 1;
        cmn.n_nlsf_survivors = 6;
        cmn.warping_q16 = cmn.fs_khz * WARPING_MULTIPLIER_Q16;
    } else if complexity < 8 {
        cmn.pitch_estimation_complexity = SILK_PE_MID_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = (0.72f32 * 65536.0) as i32;
        ps_enc.pitch_estimation_lpc_order = 12;
        cmn.shaping_lpc_order = 20;
        cmn.la_shape = 5 * cmn.fs_khz;
        cmn.n_states_delayed_decision = 3;
        cmn.use_interpolated_nlsfs = 1;
        cmn.n_nlsf_survivors = 8;
        cmn.warping_q16 = cmn.fs_khz * WARPING_MULTIPLIER_Q16;
    } else {
        cmn.pitch_estimation_complexity = SILK_PE_MAX_COMPLEX as i32;
        cmn.pitch_estimation_threshold_q16 = (0.7f32 * 65536.0) as i32;
        ps_enc.pitch_estimation_lpc_order = 16;
        cmn.shaping_lpc_order = 24;
        cmn.la_shape = 5 * cmn.fs_khz;
        cmn.n_states_delayed_decision = MAX_DEL_DEC_STATES;
        cmn.use_interpolated_nlsfs = 1;
        cmn.n_nlsf_survivors = 16;
        cmn.warping_q16 = cmn.fs_khz * WARPING_MULTIPLIER_Q16;
    }

    /* Do not allow higher pitch estimation LPC order than predict LPC order */
    ps_enc.pitch_estimation_lpc_order =
        ps_enc.pitch_estimation_lpc_order.min(cmn.predict_lpc_order);
    cmn.shape_win_length = SUB_FRAME_LENGTH_MS as i32 * cmn.fs_khz + 2 * cmn.la_shape;
    cmn.complexity = complexity;

    SILK_NO_ERROR
}

/// Initialize encoder for a given configuration.
/// Combines silk_setup_fs + silk_setup_complexity + SNR setup.
pub fn silk_control_encoder(
    ps_enc: &mut SilkEncoderState,
    fs_khz: i32,
    packet_size_ms: i32,
    target_rate_bps: i32,
    complexity: i32,
) -> i32 {
    let mut ret = silk_setup_fs(ps_enc, fs_khz, packet_size_ms);
    ret += silk_setup_complexity(ps_enc, complexity);

    ps_enc.s_cmn.target_rate_bps = target_rate_bps;

    ret
}
