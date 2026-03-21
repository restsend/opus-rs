pub const BITRESERVOIR_DECAY_TIME_MS: i32 = 500;

pub const FIND_PITCH_WHITE_NOISE_FRACTION: f32 = 1e-3;
pub const FIND_PITCH_BANDWIDTH_EXPANSION: f32 = 0.99;

pub const FIND_LPC_COND_FAC: f32 = 1e-5;
#[allow(nonstandard_style)]
pub const MAX_SUM_LOG_GAIN_dB: f32 = 250.0;
pub const LTP_CORR_INV_MAX: f32 = 0.03;

pub const VARIABLE_HP_SMTH_COEF1: f32 = 0.1;
pub const VARIABLE_HP_SMTH_COEF2: f32 = 0.015;
pub const VARIABLE_HP_MAX_DELTA_FREQ: f32 = 0.4;
pub const VARIABLE_HP_MIN_CUTOFF_HZ: i32 = 60;
pub const VARIABLE_HP_MAX_CUTOFF_HZ: i32 = 100;

pub const SPEECH_ACTIVITY_DTX_THRES: f32 = 0.05;
pub const LBRR_SPEECH_ACTIVITY_THRES: f32 = 0.3;

#[allow(nonstandard_style)]
pub const BG_SNR_DECR_dB: f32 = 2.0;

#[allow(nonstandard_style)]
pub const HARM_SNR_INCR_dB: f32 = 2.0;

#[allow(nonstandard_style)]
pub const SPARSE_SNR_INCR_dB: f32 = 2.0;

pub const ENERGY_VARIATION_THRESHOLD_QNT_OFFSET: f32 = 0.6;

pub const WARPING_MULTIPLIER: f32 = 0.015;

pub const SHAPE_WHITE_NOISE_FRACTION: f32 = 3e-5;

pub const BANDWIDTH_EXPANSION: f32 = 0.94;

pub const HARMONIC_SHAPING: f32 = 0.3;

pub const HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING: f32 = 0.2;

pub const HP_NOISE_COEF: f32 = 0.25;

pub const HARM_HP_NOISE_COEF: f32 = 0.35;

pub const INPUT_TILT: f32 = 0.05;

pub const HIGH_RATE_INPUT_TILT: f32 = 0.1;

pub const LOW_FREQ_SHAPING: f32 = 4.0;

pub const LOW_QUALITY_LOW_FREQ_SHAPING_DECR: f32 = 0.5;

pub const SUBFR_SMTH_COEF: f32 = 0.4;

pub const LAMBDA_OFFSET: f32 = 1.2;
pub const LAMBDA_SPEECH_ACT: f32 = -0.2;
pub const LAMBDA_DELAYED_DECISIONS: f32 = -0.05;
pub const LAMBDA_INPUT_QUALITY: f32 = -0.1;
pub const LAMBDA_CODING_QUALITY: f32 = -0.2;
pub const LAMBDA_QUANT_OFFSET: f32 = 0.8;
