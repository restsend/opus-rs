//! libopus `silk/float/structs_FLP.h`: the per-frame control of the float
//! encoder. Its state (`silk_encoder_state_FLP`) is [`SilkEncoderState`]
//! with the float fields `x_buf`, `ltp_corr` and the float smoothing of
//! `s_shape`.
//!
//! [`SilkEncoderState`]: crate::silk::structs::SilkEncoderState

use crate::silk::define::*;

/// libopus `silk_encoder_control_FLP`.
#[derive(Clone, Copy)]
pub struct SilkEncoderControlFlp {
    /// Prediction and coding parameters.
    pub gains: [f32; MAX_NB_SUBFR],
    /// The interpolated and the final LPC coefficients.
    pub pred_coef: [[f32; MAX_LPC_ORDER]; 2],
    pub ltp_coef: [f32; LTP_ORDER * MAX_NB_SUBFR],
    pub ltp_scale: f32,
    pub pitch_l: [i32; MAX_NB_SUBFR],
    /// Noise shaping parameters.
    pub ar: [f32; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    pub lf_ma_shp: [f32; MAX_NB_SUBFR],
    pub lf_ar_shp: [f32; MAX_NB_SUBFR],
    pub tilt: [f32; MAX_NB_SUBFR],
    pub harm_shape_gain: [f32; MAX_NB_SUBFR],
    pub lambda: f32,
    pub input_quality: f32,
    pub coding_quality: f32,
    /// Measures.
    pub pred_gain: f32,
    pub ltp_red_cod_gain: f32,
    /// Residual energy per subframe.
    pub res_nrg: [f32; MAX_NB_SUBFR],
    /// For the rate control loop.
    pub gains_unq_q16: [i32; MAX_NB_SUBFR],
    pub last_gain_index_prev: i8,
}

impl Default for SilkEncoderControlFlp {
    fn default() -> Self {
        SilkEncoderControlFlp {
            gains: [0.0; MAX_NB_SUBFR],
            pred_coef: [[0.0; MAX_LPC_ORDER]; 2],
            ltp_coef: [0.0; LTP_ORDER * MAX_NB_SUBFR],
            ltp_scale: 0.0,
            pitch_l: [0; MAX_NB_SUBFR],
            ar: [0.0; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
            lf_ma_shp: [0.0; MAX_NB_SUBFR],
            lf_ar_shp: [0.0; MAX_NB_SUBFR],
            tilt: [0.0; MAX_NB_SUBFR],
            harm_shape_gain: [0.0; MAX_NB_SUBFR],
            lambda: 0.0,
            input_quality: 0.0,
            coding_quality: 0.0,
            pred_gain: 0.0,
            ltp_red_cod_gain: 0.0,
            res_nrg: [0.0; MAX_NB_SUBFR],
            gains_unq_q16: [0; MAX_NB_SUBFR],
            last_gain_index_prev: 0,
        }
    }
}
