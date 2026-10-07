pub const ENCODER_NUM_CHANNELS: usize = 2;
pub const DECODER_NUM_CHANNELS: usize = 2;

pub const SILK_NO_ERROR: i32 = 0;
/// The encoder was given fewer samples than a packet holds.
pub const SILK_ENC_INPUT_INVALID_NO_OF_SAMPLES: i32 = -101;

pub const MAX_FRAMES_PER_PACKET: usize = 3;

pub const NB_SPEECH_FRAMES_BEFORE_DTX: i32 = 10;
pub const MAX_CONSECUTIVE_DTX: i32 = 20;

pub const SPEECH_ACTIVITY_DTX_THRES_Q8: i32 = 13;

pub const MIN_TARGET_RATE_BPS: i32 = 5000;
pub const MAX_TARGET_RATE_BPS: i32 = 80000;

pub const MAX_FS_KHZ: usize = 16;

pub const MAX_NB_SUBFR: usize = 4;
pub const LTP_MEM_LENGTH_MS: usize = 20;
pub const SUB_FRAME_LENGTH_MS: usize = 5;
pub const MAX_SUB_FRAME_LENGTH: usize = SUB_FRAME_LENGTH_MS * MAX_FS_KHZ;
pub const MAX_FRAME_LENGTH_MS: usize = SUB_FRAME_LENGTH_MS * MAX_NB_SUBFR;
pub const MAX_FRAME_LENGTH: usize = MAX_FRAME_LENGTH_MS * MAX_FS_KHZ;

pub const LA_PITCH_MS: usize = 2;
pub const LA_PITCH_MAX: usize = LA_PITCH_MS * MAX_FS_KHZ;
pub const LA_SHAPE_MS: usize = 5;
pub const LA_SHAPE_MAX: usize = LA_SHAPE_MS * MAX_FS_KHZ;
pub const MAX_LOOK_AHEAD: usize = (LA_PITCH_MS + LA_SHAPE_MS) * MAX_FS_KHZ;

pub const MAX_FIND_PITCH_LPC_ORDER: usize = 16;
pub const MAX_LPC_ORDER: usize = 16;
pub const MIN_LPC_ORDER: usize = 10;
pub const MAX_SHAPE_LPC_ORDER: usize = 24;

pub const MAX_LPC_STABILIZE_ITERATIONS: usize = 16;

pub const LTP_ORDER: usize = 5;

pub const HARM_SHAPE_FIR_TAPS: usize = 3;

pub const VAD_N_BANDS: usize = 4;
pub const VAD_NOISE_LEVELS_BIAS: i32 = 50;

pub const VAD_INTERNAL_SUBFRAMES_LOG2: usize = 2;
pub const VAD_INTERNAL_SUBFRAMES: usize = 1 << VAD_INTERNAL_SUBFRAMES_LOG2;

pub const VAD_NOISE_LEVEL_SMOOTH_COEF_Q16: i32 = 1024;
pub const VAD_SNR_FACTOR_Q16: i32 = 45000;
pub const VAD_NEGATIVE_OFFSET_Q5: i32 = 128;
pub const VAD_SNR_SMOOTH_COEF_Q18: i32 = 4096;

pub const NSQ_LPC_BUF_LENGTH: usize = MAX_LPC_ORDER;
pub const DECISION_DELAY: usize = 40;
pub const QUANT_LEVEL_ADJUST_Q10: i32 = 80;
pub const NSQ_MAX_STATES_OPERATING: usize = 4;

pub const NLSF_QUANT_MAX_AMPLITUDE: i32 = 4;
pub const NLSF_QUANT_MAX_AMPLITUDE_EXT: i32 = 10;
pub const NLSF_QUANT_LEVEL_ADJ: i32 = 102;
pub const NLSF_QUANT_DEL_DEC_STATES: usize = 4;
pub const NLSF_QUANT_DEL_DEC_STATES_LOG2: usize = 2;

pub const INTERP_NUM_STATES: usize = 5;

pub const TYPE_NO_VOICE_ACTIVITY: i32 = 0;
pub const TYPE_UNVOICED: i32 = 1;
pub const TYPE_VOICED: i32 = 2;

pub const CODE_INDEPENDENTLY_NO_LTP_SCALING: i32 = 2;

/// Length of the stereo predictor and width interpolation at the start of a
/// frame, in ms (libopus `STEREO_INTERP_LEN_MS`).
pub const STEREO_INTERP_LEN_MS: i32 = 8;
/// Size of the stereo predictor quantization table (`STEREO_QUANT_TAB_SIZE`).
pub const STEREO_QUANT_TAB_SIZE: usize = 16;
/// Sub-steps per stereo predictor quantization interval
/// (`STEREO_QUANT_SUB_STEPS`).
pub const STEREO_QUANT_SUB_STEPS: i32 = 5;
/// Half a predictor sub-step, `SILK_FIX_CONST(0.5 / STEREO_QUANT_SUB_STEPS, 16)`
/// in libopus (stereo_decode_pred.c, stereo_quant_pred.c). SILK_FIX_CONST
/// rounds, so this is 6554, not the 6553 of `(1 << 16) / 10`; the truncated
/// value dequantized 45 of the 75 predictor levels a few Q13 units off
/// libopus, and the neutral index to (0, -5) instead of (0, 0).
pub const STEREO_HALF_SUB_STEP_Q16: i32 = 6554;

pub const SILK_PE_MIN_COMPLEX: usize = 0;
pub const SILK_PE_MID_COMPLEX: usize = 1;
pub const SILK_PE_MAX_COMPLEX: usize = 2;

pub const PE_MAX_NB_SUBFR: usize = 4;
pub const PE_NB_CBKS_STAGE2_10MS: usize = 3;
pub const PE_NB_CBKS_STAGE3_10MS: usize = 12;
pub const PE_NB_CBKS_STAGE2_EXT: usize = 11;
pub const PE_NB_CBKS_STAGE2: usize = 3;
pub const PE_NB_CBKS_STAGE3_MAX: usize = 34;
pub const PE_NB_CBKS_STAGE3_MIN: usize = 16;
pub const PE_NB_CBKS_STAGE3_MID: usize = 24;

pub const PITCH_EST_MIN_LAG_MS: i32 = 2;
pub const PITCH_EST_MAX_LAG_MS: i32 = 18;

pub const MAX_PREDICTION_POWER_GAIN: f32 = 1e4f32;
pub const MAX_PREDICTION_POWER_GAIN_AFTER_RESET: f32 = 1e2f32;

pub const PE_MAX_FS_KHZ: usize = 16;
pub const PE_SUBFR_LENGTH_MS: usize = 5;
pub const PE_LTP_MEM_LENGTH_MS: usize = 20;

pub const PE_MAX_FRAME_LENGTH_MS: usize =
    PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS;
pub const PE_MAX_FRAME_LENGTH: usize = PE_MAX_FRAME_LENGTH_MS * PE_MAX_FS_KHZ;
pub const PE_MAX_FRAME_LENGTH_ST_1: usize = PE_MAX_FRAME_LENGTH >> 2;
pub const PE_MAX_FRAME_LENGTH_ST_2: usize = PE_MAX_FRAME_LENGTH >> 1;

pub const PE_MAX_LAG_MS: usize = 18;
pub const PE_MIN_LAG_MS: usize = 2;

pub const TRANSITION_TIME_MS: i32 = 5120;
pub const TRANSITION_NB: usize = 3;
pub const TRANSITION_NA: usize = 2;
pub const TRANSITION_INT_NUM: usize = 5;
pub const TRANSITION_FRAMES: i32 = TRANSITION_TIME_MS / MAX_FRAME_LENGTH_MS as i32;
pub const TRANSITION_INT_STEPS: i32 = TRANSITION_FRAMES / (TRANSITION_INT_NUM as i32 - 1);
pub const PE_MAX_LAG: usize = PE_MAX_LAG_MS * PE_MAX_FS_KHZ;
pub const PE_MIN_LAG: usize = PE_MIN_LAG_MS * PE_MAX_FS_KHZ;

pub const PE_D_SRCH_LENGTH: usize = 24;
pub const PE_NB_STAGE3_LAGS: usize = 5;

pub const PE_SHORTLAG_BIAS_Q13: i32 = 1638;
pub const PE_PREVLAG_BIAS_Q13: i32 = 1638;
pub const PE_FLATCONTOUR_BIAS_Q13: i32 = 410;

pub const COND_ALPHA_MIN_Q15: i32 = 25000;
pub const COND_ALPHA_MAX_Q15: i32 = 31000;

pub const FIND_LPC_COND_FAC_Q31: i32 = 21475;

pub const CODE_INDEPENDENTLY: i32 = 0;
pub const CODE_CONDITIONALLY: i32 = 1;
pub const CODE_INFORMATION: i32 = 2;

pub const N_RATE_LEVELS: usize = 10;
pub const SILK_MAX_PULSES: usize = 16;
pub const SHELL_CODEC_FRAME_LENGTH: usize = 16;

pub const MIN_QGAIN_DB: i32 = 2;
pub const MAX_QGAIN_DB: i32 = 88;
pub const N_LEVELS_QGAIN: i32 = 64;
pub const MAX_DELTA_GAIN_QUANT: i32 = 36;
pub const MIN_DELTA_GAIN_QUANT: i32 = -4;

// PLC (packet loss concealment) constants (silk/PLC.h)
pub const BWE_COEF_Q16: i32 = 64881; // 0.99 in Q16
pub const V_PITCH_GAIN_START_MIN_Q14: i32 = 11469; // 0.7 in Q14
pub const V_PITCH_GAIN_START_MAX_Q14: i32 = 15565; // 0.95 in Q14
pub const MAX_PITCH_LAG_MS: i32 = 18;
pub const RAND_BUF_SIZE: usize = 128;
pub const RAND_BUF_MASK: usize = RAND_BUF_SIZE - 1;
pub const LOG2_INV_LPC_GAIN_HIGH_THRES: i32 = 3; // 2^3 = 8 dB LPC gain
pub const LOG2_INV_LPC_GAIN_LOW_THRES: i32 = 8; // 2^8 = 24 dB LPC gain
pub const PITCH_DRIFT_FAC_Q16: i32 = 655; // 0.01 in Q16
pub const NB_ATT: usize = 2;
pub const HARM_ATT_Q15: [i16; NB_ATT] = [32440, 31130]; // 0.99, 0.95
pub const PLC_RAND_ATTENUATE_V_Q15: [i16; NB_ATT] = [31130, 26214]; // 0.95, 0.8
pub const PLC_RAND_ATTENUATE_UV_Q15: [i16; NB_ATT] = [32440, 29491]; // 0.99, 0.9
