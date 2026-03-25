use crate::silk::define::*;

#[derive(Clone)]
pub struct SilkStereoState {
    pub s_mid: [i16; 2],

    pub s_side: [i16; 2],

    pub left: i16,

    pub side: Vec<i16>,
}

impl Default for SilkStereoState {
    fn default() -> Self {
        Self {
            s_mid: [0; 2],
            s_side: [0; 2],
            left: 0,
            side: Vec::new(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct NLSFCodebook {
    pub n_vectors: i16,
    pub order: i16,
    pub quant_step_size_q16: i32,
    pub inv_quant_step_size_q6: i16,
    pub cb1_nlsf_q8: &'static [u8],
    pub cb1_wght_q9: &'static [i16],
    pub cb1_icdf: &'static [u8],
    pub pred_q8: &'static [u8],
    pub ec_sel: &'static [u8],
    pub ec_icdf: &'static [u8],
    pub ec_rates_q5: &'static [u8],
    pub delta_min_q15: &'static [i16],
}

#[derive(Clone, Copy)]
pub struct SideInfoIndices {
    pub gains_indices: [i8; MAX_NB_SUBFR],
    pub ltp_index: [i8; MAX_NB_SUBFR],
    pub nlsf_indices: [i8; MAX_LPC_ORDER + 1],
    pub lag_index: i16,
    pub contour_index: i8,
    pub signal_type: i8,
    pub voicing_idx: i8,
    pub quant_offset_type: i8,
    pub nlsf_interp_coef_q2: i8,
    pub per_index: i8,
    pub ltp_scale_index: i8,
    pub seed: i8,

    pub pred_idx: i8,

    pub side_idx: i8,

    pub only_middle: i8,
}

impl Default for SideInfoIndices {
    fn default() -> Self {
        Self {
            gains_indices: [0; MAX_NB_SUBFR],
            ltp_index: [0; MAX_NB_SUBFR],
            nlsf_indices: [0; MAX_LPC_ORDER + 1],
            lag_index: 0,
            contour_index: 0,
            signal_type: 0,
            voicing_idx: 0,
            quant_offset_type: 0,
            nlsf_interp_coef_q2: 4,
            per_index: 0,
            ltp_scale_index: 0,
            seed: 0,
            pred_idx: 0,
            side_idx: 0,
            only_middle: 0,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct SilkShapeState {
    pub last_gain_index: i8,
    pub harm_boost_smth_q16: i32,
    pub harm_shape_gain_smth_q16: i32,
    pub tilt_smth_q16: i32,
}

#[derive(Clone, Copy)]
pub struct SilkVADState {
    pub ana_state: [i32; 2],
    pub ana_state1: [i32; 2],
    pub ana_state2: [i32; 2],
    pub xnrg_subfr: [i32; VAD_N_BANDS],
    pub nrg_ratio_smth_q8: [i32; VAD_N_BANDS],
    pub hp_state: i16,
    pub nl: [i32; VAD_N_BANDS],
    pub inv_nl: [i32; VAD_N_BANDS],
    pub noise_level_bias: [i32; VAD_N_BANDS],
    pub counter: i32,
}

impl Default for SilkVADState {
    fn default() -> Self {
        Self {
            ana_state: [0; 2],
            ana_state1: [0; 2],
            ana_state2: [0; 2],
            xnrg_subfr: [0; VAD_N_BANDS],
            nrg_ratio_smth_q8: [0; VAD_N_BANDS],
            hp_state: 0,
            nl: [0; VAD_N_BANDS],
            inv_nl: [0; VAD_N_BANDS],
            noise_level_bias: [0; VAD_N_BANDS],
            counter: 0,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct SilkLPState {
    pub in_lp_state: [i32; 2],
    pub transition_frame_no: i32,
    pub mode: i32,
    pub saved_fs_khz: i32,
}

pub struct SilkEncoderStateCommon {
    pub indices: SideInfoIndices,
    pub in_hp_state: [i32; 2],
    pub snr_db_q7: i32,
    pub input_quality_bands_q15: [i32; VAD_N_BANDS],
    pub speech_activity_q8: i32,
    pub use_cbr: i32,
    pub fs_khz: i32,
    pub nb_subfr: i32,
    pub warping_q16: i32,
    pub la_shape: i32,
    pub shape_win_length: i32,
    pub shaping_lpc_order: i32,
    pub predict_lpc_order: i32,
    pub subfr_length: i32,
    pub la_pitch: i32,
    pub frame_length: i32,
    pub ltp_mem_length: i32,
    pub pitch_lpc_win_length: i32,
    pub pitch_estimation_complexity: i32,
    pub pitch_estimation_threshold_q16: i32,
    pub first_frame_after_reset: i32,
    pub prev_signal_type: i32,
    pub input_tilt_q15: i32,
    pub n_states_delayed_decision: i32,
    pub prev_lag: i32,
    pub x_buf: [i16; 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX],
    pub x_buf_idx: i32,
    pub prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    pub nlsf_mu_q20: i32,
    pub n_nlsf_survivors: i32,
    pub variable_hp_smth1_q15: i32,
    pub variable_hp_smth2_q15: i32,
    pub s_lp: SilkLPState,
    pub s_vad: SilkVADState,
    pub lbrr_enabled: i32,
    pub indices_lbrr: [SideInfoIndices; MAX_FRAMES_PER_PACKET],
    pub pulses_lbrr: [[i8; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],
    pub n_frames_encoded: i32,
    pub n_frames_per_packet: i32,
    pub target_rate_bps: i32,
    pub packet_size_ms: i32,
    pub complexity: i32,
    pub sum_log_gain_q7: i32,
    pub packet_loss_perc: i32,
    pub lbrr_flag: i8,
    pub no_speech_counter: i32,
    pub in_dtx: i32,
    pub vad_flags: [i32; MAX_FRAMES_PER_PACKET],

    pub frame_counter: i32,
    pub ec_prev_signal_type: i32,
    pub ec_prev_lag_index: i16,
    pub frames_since_onset: i32,
    pub input_buf: [i16; MAX_FRAME_LENGTH + 2],
    pub input_buf_ix: i32,
    pub controlled_since_last_payload: i32,
    pub use_interpolated_nlsfs: i32,
    pub use_dtx: i32,
    pub use_in_band_fec: i32,
    pub lbrr_gain_increases: i32,
    pub lbrr_flags: [i32; MAX_FRAMES_PER_PACKET],
    pub prefill_flag: i32,

    pub n_channels: i32,
}

impl Default for SilkEncoderStateCommon {
    fn default() -> Self {
        Self {
            indices: SideInfoIndices::default(),
            in_hp_state: [0; 2],
            snr_db_q7: 0,
            input_quality_bands_q15: [0; VAD_N_BANDS],
            speech_activity_q8: 0,
            use_cbr: 0,
            fs_khz: 0,
            nb_subfr: 0,
            warping_q16: 0,
            la_shape: 0,
            shape_win_length: 0,
            shaping_lpc_order: 0,
            predict_lpc_order: 0,
            subfr_length: 0,
            la_pitch: 0,
            frame_length: 0,
            ltp_mem_length: 0,
            pitch_lpc_win_length: 0,
            pitch_estimation_complexity: 0,
            pitch_estimation_threshold_q16: 0,
            first_frame_after_reset: 0,
            prev_signal_type: 0,
            input_tilt_q15: 0,
            n_states_delayed_decision: 0,
            prev_lag: 0,
            x_buf: [0; 2 * MAX_FRAME_LENGTH + LA_SHAPE_MAX],
            x_buf_idx: 0,
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            nlsf_mu_q20: 0,
            n_nlsf_survivors: 0,
            variable_hp_smth1_q15: 0,
            variable_hp_smth2_q15: 0,
            s_lp: SilkLPState::default(),
            s_vad: SilkVADState::default(),
            lbrr_enabled: 0,
            indices_lbrr: [SideInfoIndices::default(); MAX_FRAMES_PER_PACKET],
            pulses_lbrr: [[0; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],
            n_frames_encoded: 0,
            n_frames_per_packet: 0,
            target_rate_bps: 0,
            packet_size_ms: 0,
            complexity: 0,
            sum_log_gain_q7: 0,
            packet_loss_perc: 0,
            lbrr_flag: 0,
            no_speech_counter: 0,
            in_dtx: 0,
            vad_flags: [0; MAX_FRAMES_PER_PACKET],
            frame_counter: 0,
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            frames_since_onset: 0,
            input_buf: [0; MAX_FRAME_LENGTH + 2],
            input_buf_ix: 0,
            controlled_since_last_payload: 0,
            use_interpolated_nlsfs: 0,
            use_dtx: 0,
            use_in_band_fec: 0,
            lbrr_gain_increases: 0,
            lbrr_flags: [0; MAX_FRAMES_PER_PACKET],
            prefill_flag: 0,
            n_channels: 1,
        }
    }
}

pub struct SilkEncoderState {
    pub s_cmn: SilkEncoderStateCommon,
    pub s_shape: SilkShapeState,
    pub pulses: [i8; MAX_FRAME_LENGTH],
    pub s_nsq: SilkNSQState,
    pub ltp_corr_q15: i32,
    pub res_nrg_smth: i32,
    pub pitch_estimation_lpc_order: i32,
    pub ps_nlsf_cb: Option<&'static NLSFCodebook>,

    pub stereo: SilkStereoState,

    pub resampler_delay_buf: [i16; 48],
}

impl Default for SilkEncoderState {
    fn default() -> Self {
        Self {
            s_cmn: SilkEncoderStateCommon::default(),
            s_shape: SilkShapeState::default(),
            pulses: [0; MAX_FRAME_LENGTH],
            s_nsq: SilkNSQState::default(),
            ltp_corr_q15: 0,
            res_nrg_smth: 0,
            pitch_estimation_lpc_order: 0,
            ps_nlsf_cb: None,
            stereo: SilkStereoState::default(),
            resampler_delay_buf: [0; 48],
        }
    }
}

#[derive(Clone, Copy)]
pub struct SilkNSQState {
    pub xq: [i16; 2 * MAX_FRAME_LENGTH],
    pub s_ltp_shp_q14: [i32; 2 * MAX_FRAME_LENGTH],
    pub s_lpc_q14: [i32; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    pub s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER],
    pub s_lf_ar_q14: i32,
    pub s_diff_shp_q14: i32,
    pub lag_prev: i32,
    pub s_ltp_buf_idx: i32,
    pub s_ltp_shp_buf_idx: i32,
    pub rand_seed: i32,
    pub prev_gain_q16: i32,
    pub rewhite_flag: i32,
    pub prev_sig_type: i8,
}

impl Default for SilkNSQState {
    fn default() -> Self {
        Self {
            xq: [0; 2 * MAX_FRAME_LENGTH],
            s_ltp_shp_q14: [0; 2 * MAX_FRAME_LENGTH],
            s_lpc_q14: [0; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
            s_ar2_q14: [0; MAX_SHAPE_LPC_ORDER],
            s_lf_ar_q14: 0,
            s_diff_shp_q14: 0,
            lag_prev: 0,
            s_ltp_buf_idx: 0,
            s_ltp_shp_buf_idx: 0,
            rand_seed: 0,
            prev_gain_q16: 0,
            rewhite_flag: 0,
            prev_sig_type: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct SilkEncoderControl {
    pub input_quality_q14: i32,
    pub coding_quality_q14: i32,
    pub pitch_l: [i32; MAX_NB_SUBFR],
    pub gains_q16: [i32; MAX_NB_SUBFR],
    pub gains_unq_q16: [i32; MAX_NB_SUBFR],
    pub pred_coef_q12: [[i16; MAX_LPC_ORDER]; 2],
    pub ltp_coef_q14: [i16; MAX_NB_SUBFR * LTP_ORDER],
    pub ltp_scale_q14: i32,
    pub ar_q13: [i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    pub lf_shp_q14: [i32; MAX_NB_SUBFR],
    pub tilt_q14: [i32; MAX_NB_SUBFR],
    pub harm_shape_gain_q14: [i32; MAX_NB_SUBFR],
    pub lambda_q10: i32,
    pub pred_gain_q16: i32,
    pub ltp_red_cod_gain_q7: i32,
    pub res_nrg: [i32; MAX_NB_SUBFR],
    pub res_nrg_q: [i32; MAX_NB_SUBFR],
    pub last_gain_index_prev: i8,
}

impl Default for SilkEncoderControl {
    fn default() -> Self {
        Self {
            input_quality_q14: 0,
            coding_quality_q14: 0,
            pitch_l: [0; MAX_NB_SUBFR],
            gains_q16: [0; MAX_NB_SUBFR],
            gains_unq_q16: [0; MAX_NB_SUBFR],
            pred_coef_q12: [[0; MAX_LPC_ORDER]; 2],
            ltp_coef_q14: [0; MAX_NB_SUBFR * LTP_ORDER],
            ltp_scale_q14: 0,
            ar_q13: [0; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
            lf_shp_q14: [0; MAX_NB_SUBFR],
            tilt_q14: [0; MAX_NB_SUBFR],
            harm_shape_gain_q14: [0; MAX_NB_SUBFR],
            lambda_q10: 0,
            pred_gain_q16: 0,
            ltp_red_cod_gain_q7: 0,
            res_nrg: [0; MAX_NB_SUBFR],
            res_nrg_q: [0; MAX_NB_SUBFR],
            last_gain_index_prev: 0,
        }
    }
}
