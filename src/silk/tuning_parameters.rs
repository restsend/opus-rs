pub const BITRESERVOIR_DECAY_TIME_MS: i32 = 500;

/* Pitch estimator */
pub const FIND_PITCH_WHITE_NOISE_FRACTION: f32 = 1e-3;
pub const FIND_PITCH_BANDWIDTH_EXPANSION: f32 = 0.99;

/* Linear prediction */
pub const FIND_LPC_COND_FAC: f32 = 1e-5;
#[allow(nonstandard_style)]
pub const MAX_SUM_LOG_GAIN_dB: f32 = 250.0;
pub const LTP_CORR_INV_MAX: f32 = 0.03;

/* High pass filtering */
pub const VARIABLE_HP_SMTH_COEF1: f32 = 0.1;
pub const VARIABLE_HP_SMTH_COEF2: f32 = 0.015;
pub const VARIABLE_HP_MAX_DELTA_FREQ: f32 = 0.4;
pub const VARIABLE_HP_MIN_CUTOFF_HZ: i32 = 60;
pub const VARIABLE_HP_MAX_CUTOFF_HZ: i32 = 100;

/* Various */
pub const SPEECH_ACTIVITY_DTX_THRES: f32 = 0.05;
pub const LBRR_SPEECH_ACTIVITY_THRES: f32 = 0.3;

/*************************/
/* Perceptual parameters */
/*************************/

/* reduction in coding SNR during low speech activity */
#[allow(nonstandard_style)]
pub const BG_SNR_DECR_dB: f32 = 2.0;

/* factor for reducing quantization noise during voiced speech */
#[allow(nonstandard_style)]
pub const HARM_SNR_INCR_dB: f32 = 2.0;

/* factor for reducing quantization noise for unvoiced sparse signals */
#[allow(nonstandard_style)]
pub const SPARSE_SNR_INCR_dB: f32 = 2.0;

/* threshold for sparseness measure above which to use lower quantization offset during unvoiced */
pub const ENERGY_VARIATION_THRESHOLD_QNT_OFFSET: f32 = 0.6;

/* warping control */
pub const WARPING_MULTIPLIER: f32 = 0.015;

/* fraction added to first autocorrelation value */
pub const SHAPE_WHITE_NOISE_FRACTION: f32 = 3e-5;

/* noise shaping filter chirp factor */
pub const BANDWIDTH_EXPANSION: f32 = 0.94;

/* harmonic noise shaping */
pub const HARMONIC_SHAPING: f32 = 0.3;

/* extra harmonic noise shaping for high bitrates or noisy input */
pub const HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING: f32 = 0.2;

/* parameter for shaping noise towards higher frequencies */
pub const HP_NOISE_COEF: f32 = 0.25;

/* parameter for shaping noise even more towards higher frequencies during voiced speech */
pub const HARM_HP_NOISE_COEF: f32 = 0.35;

/* parameter for applying a high-pass tilt to the input signal */
pub const INPUT_TILT: f32 = 0.05;

/* parameter for extra high-pass tilt to the input signal at high rates */
pub const HIGH_RATE_INPUT_TILT: f32 = 0.1;

/* parameter for reducing noise at the very low frequencies */
pub const LOW_FREQ_SHAPING: f32 = 4.0;

/* less reduction of noise at the very low frequencies for signals with low SNR at low frequencies */
pub const LOW_QUALITY_LOW_FREQ_SHAPING_DECR: f32 = 0.5;

/* subframe smoothing coefficient for HarmBoost, HarmShapeGain, Tilt (lower -> more smoothing) */
pub const SUBFR_SMTH_COEF: f32 = 0.4;

/* parameters defining the R/D tradeoff in the residual quantizer */
pub const LAMBDA_OFFSET: f32 = 1.2;
pub const LAMBDA_SPEECH_ACT: f32 = -0.2;
pub const LAMBDA_DELAYED_DECISIONS: f32 = -0.05;
pub const LAMBDA_INPUT_QUALITY: f32 = -0.1;
pub const LAMBDA_CODING_QUALITY: f32 = -0.2;
pub const LAMBDA_QUANT_OFFSET: f32 = 0.8;
