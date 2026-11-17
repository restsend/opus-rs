//! Opus speech coding enhancement (OSCE): libopus `dnn/osce.c`,
//! `osce_features.c`, `nndsp.c`.
//!
//! - `features`: the LACE/NoLACE and BBWENet input features and the
//!   cross-fades (`osce_features.c`).

pub mod features;


// osce_config.h
pub const OSCE_FEATURES_MAX_HISTORY: usize = 350;
pub const OSCE_FEATURE_DIM: usize = 93;
pub const OSCE_MAX_FEATURE_FRAMES: usize = 4;
pub const OSCE_CLEAN_SPEC_NUM_BANDS: usize = 64;
pub const OSCE_NOISY_SPEC_NUM_BANDS: usize = 18;
pub const OSCE_NO_PITCH_VALUE: i32 = 7;
pub const OSCE_PREEMPH: f32 = 0.85;
pub const OSCE_PITCH_HANGOVER: i32 = 0;
pub const OSCE_CLEAN_SPEC_START: usize = 0;
pub const OSCE_CLEAN_SPEC_LENGTH: usize = 64;
pub const OSCE_NOISY_CEPSTRUM_START: usize = 64;
pub const OSCE_NOISY_CEPSTRUM_LENGTH: usize = 18;
pub const OSCE_ACORR_START: usize = 82;
pub const OSCE_ACORR_LENGTH: usize = 5;
pub const OSCE_LTP_START: usize = 87;
pub const OSCE_LTP_LENGTH: usize = 5;
pub const OSCE_LOG_GAIN_START: usize = 92;
pub const OSCE_LOG_GAIN_LENGTH: usize = 1;
pub const OSCE_BWE_MAX_INSTAFREQ_BIN: usize = 40;
pub const OSCE_BWE_HALF_WINDOW_SIZE: usize = 160;
pub const OSCE_BWE_WINDOW_SIZE: usize = 2 * OSCE_BWE_HALF_WINDOW_SIZE;
pub const OSCE_BWE_NUM_BANDS: usize = 32;
pub const OSCE_BWE_FEATURE_DIM: usize = 114;
pub const OSCE_BWE_OUTPUT_DELAY: usize = 21;

// osce.h
pub const OSCE_MODE_SILK_ONLY: i32 = 1000;
pub const OSCE_MODE_HYBRID: i32 = 1001;
pub const OSCE_MODE_CELT_ONLY: i32 = 1002;
pub const OSCE_MODE_SILK_BBWE: i32 = 1003;

/// SILK's `TYPE_VOICED`.
pub const TYPE_VOICED: i32 = 2;

/// The enhancement method (`OSCE_METHOD_*`). libopus's decoder picks
/// `Lace` at complexity 6 and `Nolace` from complexity 7.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum OsceMethod {
    None = 0,
    Lace = 1,
    Nolace = 2,
}

/// `OSCE_DEFAULT_METHOD`, the method `silk_reset_decoder` sets.
pub const OSCE_DEFAULT_METHOD: OsceMethod = OsceMethod::Nolace;

/// The SILK decoder data of one decoded frame that `osce_enhance_frame`
/// reads (libopus passes `silk_decoder_state` and `silk_decoder_control`).
#[derive(Clone, Copy, Debug, Default)]
pub struct OsceFrameInput {
    /// `psDec->fs_kHz`: only 16 is enhanced.
    pub fs_khz: i32,
    /// `psDec->nb_subfr`: only 4 (20 ms) is enhanced.
    pub nb_subfr: usize,
    /// `psDec->LPC_order`.
    pub lpc_order: usize,
    /// `psDec->indices.signalType`.
    pub signal_type: i32,
    /// `psDecCtrl->pitchL`: the pitch lags per subframe, at most 300 (the
    /// pitch embedding has 301 rows; SILK's lags are at most 288).
    pub pitch_l: [i32; 4],
    /// `psDecCtrl->Gains_Q16`.
    pub gains_q16: [i32; 4],
    /// `psDecCtrl->PredCoef_Q12`: the LPCs of each half frame.
    pub pred_coef_q12: [[i16; 16]; 2],
    /// `psDecCtrl->LTPCoef_Q14`: `LTP_ORDER` (5) taps per subframe.
    pub ltp_coef_q14: [i16; 20],
    /// The size of the frame's SILK payload in bits: `ec_tell` after
    /// decoding the frame minus `ec_tell` before its indices.
    pub num_bits: i32,
}
