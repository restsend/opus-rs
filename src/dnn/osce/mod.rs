//! Opus speech coding enhancement (OSCE): libopus `dnn/osce.c`,
//! `osce_features.c`, `nndsp.c`.
//!
//! - [`osce_enhance_frame`] runs LACE or NoLACE on a decoded 20 ms SILK
//!   wideband frame, in place; [`osce_reset`] selects the method.
//! - [`osce_bwe`] extends a 16 kHz SILK signal to 48 kHz with BBWENet;
//!   [`osce_bwe_reset`] resets it.
//!
//! The SILK decoder hands each frame's parameters over in an
//! [`OsceFrameInput`]. Its per-channel state holds an [`Osce`] (libopus
//! `silk_OSCE_struct`) and an [`OsceBwe`] (`silk_OSCE_BWE_struct`); the
//! decoder holds one [`OsceModel`].

#[cfg(not(feature = "std"))]
use crate::compat::Math;

pub mod bbwenet;
pub mod features;
pub mod lace;
pub mod nndsp;
pub mod nolace;

use crate::dnn::builtin;
use crate::dnn::weights::{WeightError, WeightList};

use bbwenet::{bbwenet_process_frames, Bbwenet, BbwenetState};
use features::{osce_bwe_calculate_features, osce_calculate_features, osce_cross_fade_10ms, OsceBweFeatureState, OsceFeatureState};
use lace::{lace_process_20ms_frame, Lace, LaceState};
use nolace::{nolace_process_20ms_frame, Nolace, NolaceState};

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

/// The OSCE models (libopus `OSCEModel` with `loaded` set): the layers
/// and overlap windows of LACE, NoLACE and BBWENet.
#[derive(Clone, Debug)]
pub struct OsceModel<'a> {
    pub lace: Lace<'a>,
    pub nolace: Nolace<'a>,
    pub bbwenet: Bbwenet<'a>,
}

impl<'a> OsceModel<'a> {
    /// `osce_load_models` from a weight blob with the arrays of all three
    /// models.
    pub fn init(list: &WeightList<'a>) -> Result<Self, WeightError> {
        Ok(OsceModel {
            lace: Lace::new(crate::dnn::models::lace::LaceLayers::init(list)?),
            nolace: Nolace::new(crate::dnn::models::nolace::NolaceLayers::init(list)?),
            bbwenet: Bbwenet::new(crate::dnn::models::bbwenet::BbwenetLayers::init(list)?),
        })
    }
}

impl OsceModel<'static> {
    /// `osce_load_models(model, NULL, 0)`: the built-in weights.
    pub fn builtin() -> Result<Self, WeightError> {
        Ok(OsceModel {
            lace: Lace::new(builtin::lace()?),
            nolace: Nolace::new(builtin::nolace()?),
            bbwenet: Bbwenet::new(builtin::bbwenet()?),
        })
    }
}

/// The OSCE state of one SILK channel (libopus `silk_OSCE_struct`). libopus
/// keeps the LACE and NoLACE states in a union; [`osce_reset`] resets the
/// one of the selected method, so separate fields behave the same.
#[derive(Clone, Debug)]
pub struct Osce {
    pub features: OsceFeatureState,
    pub lace: LaceState,
    pub nolace: NolaceState,
    pub method: OsceMethod,
}

impl Osce {
    /// The state after `silk_init_decoder`: zero, then
    /// `osce_reset(OSCE_DEFAULT_METHOD)`.
    pub fn new() -> Self {
        let mut st = Osce {
            features: OsceFeatureState::new(),
            lace: LaceState::new(),
            nolace: NolaceState::new(),
            method: OsceMethod::None,
        };
        osce_reset(&mut st, OSCE_DEFAULT_METHOD);
        st
    }
}

impl Default for Osce {
    fn default() -> Self {
        Self::new()
    }
}

/// The bandwidth extension state of one SILK channel (libopus
/// `silk_OSCE_BWE_struct`).
#[derive(Clone, Debug)]
pub struct OsceBwe {
    pub features: OsceBweFeatureState,
    pub bbwenet: BbwenetState,
}

impl OsceBwe {
    /// The state after `silk_init_decoder`: all zero. libopus calls
    /// [`osce_bwe_reset`] before the first extended frame.
    pub fn new() -> Self {
        OsceBwe { features: OsceBweFeatureState::new(), bbwenet: BbwenetState::new() }
    }
}

impl Default for OsceBwe {
    fn default() -> Self {
        Self::new()
    }
}

/// `float2int` (lrintf): round to nearest, ties to even.
#[inline]
fn float2int(x: f32) -> i32 {
    let f = x.floor();
    let d = x - f;
    let r = if d > 0.5 || (d == 0.5 && (f as i64) % 2 != 0) { f + 1.0 } else { f };
    r as i32
}

/// The output conversion of `osce_enhance_frame` and `osce_bwe`:
/// `float2int` of `32768 x` clamped to +-32767.
#[inline]
fn to_i16(x: f32) -> i16 {
    float2int((32768.0f32 * x).clamp(-32767.0, 32767.0)) as i16
}

/// `osce_reset`: clears the feature state, resets the state of `method`
/// and selects it; the next frame passes through unchanged and the one
/// after fades into the enhanced signal.
pub fn osce_reset(st: &mut Osce, method: OsceMethod) {
    st.features = OsceFeatureState::new();
    match method {
        OsceMethod::None => {}
        OsceMethod::Lace => st.lace = LaceState::new(),
        OsceMethod::Nolace => st.nolace = NolaceState::new(),
    }
    st.method = method;
    st.features.reset = 2;
}

/// `osce_bwe_reset`.
pub fn osce_bwe_reset(st: &mut OsceBwe) {
    st.features = OsceBweFeatureState::new();
    // libopus: "weird python initialization: Fix eventually!"
    for k in 0..=OSCE_BWE_MAX_INSTAFREQ_BIN {
        st.features.last_spec[2 * k] = 1e-9f64 as f32;
    }
    st.bbwenet = BbwenetState::new();
}

/// `osce_enhance_frame`: enhances the decoded SILK frame `xq` in place with
/// `st.method`, or resets `st` when the frame is not 20 ms at 16 kHz.
/// `model` is `None` when no model is loaded (libopus `model->loaded`
/// false): the features still update and the signal passes through.
pub fn osce_enhance_frame(model: Option<&OsceModel<'_>>, st: &mut Osce, input: &OsceFrameInput, xq: &mut [i16]) {
    let mut in_buffer = [0.0f32; 320];
    let mut out_buffer = [0.0f32; 320];
    let mut features = [0.0f32; 4 * OSCE_FEATURE_DIM];
    let mut numbits = [0.0f32; 2];
    let mut periods = [0i32; 4];

    if input.fs_khz != 16 || input.nb_subfr != 4 {
        let method = st.method;
        osce_reset(st, method);
        return;
    }

    osce_calculate_features(&mut st.features, input, &mut features, &mut numbits, &mut periods, xq);

    for (b, &x) in in_buffer.iter_mut().zip(&xq[..320]) {
        *b = x as f32 * (1.0f32 / 32768.0f32);
    }

    let method = if model.is_some() { st.method } else { OsceMethod::None };
    match (method, model) {
        (OsceMethod::Lace, Some(m)) => {
            lace_process_20ms_frame(&m.lace, &mut st.lace, &mut out_buffer, &in_buffer, &features, &numbits, &periods)
        }
        (OsceMethod::Nolace, Some(m)) => nolace_process_20ms_frame(
            &m.nolace,
            &mut st.nolace,
            &mut out_buffer,
            &in_buffer,
            &features,
            &numbits,
            &periods,
        ),
        _ => out_buffer.copy_from_slice(&in_buffer),
    }

    if st.features.reset > 1 {
        out_buffer.copy_from_slice(&in_buffer);
        st.features.reset -= 1;
    } else if st.features.reset != 0 {
        osce_cross_fade_10ms(&mut out_buffer, &in_buffer);
        st.features.reset = 0;
    }

    for (x, &v) in xq[..320].iter_mut().zip(&out_buffer) {
        *x = to_i16(v);
    }
}

/// `osce_bwe`: extends `xq16` (10 or 20 ms at 16 kHz) to `xq48` (three
/// times as many samples at 48 kHz) with BBWENet. The output is delayed by
/// `OSCE_BWE_OUTPUT_DELAY` samples.
pub fn osce_bwe(model: &OsceModel<'_>, st: &mut OsceBwe, xq48: &mut [i16], xq16: &[i16]) {
    let mut in_buffer = [0.0f32; 320];
    let mut out_buffer = [0.0f32; 3 * 320];
    let mut features = [0.0f32; 2 * OSCE_BWE_FEATURE_DIM];
    let len = xq16.len();
    assert!(len == 160 || len == 320);
    let num_frames = len / 160;

    for (b, &x) in in_buffer.iter_mut().zip(xq16) {
        *b = x as f32 * (1.0f32 / 32768.0f32);
    }

    osce_bwe_calculate_features(&mut st.features, &mut features, xq16);

    bbwenet_process_frames(&model.bbwenet, &mut st.bbwenet, &mut out_buffer, &in_buffer, &features, num_frames);

    const D: usize = OSCE_BWE_OUTPUT_DELAY;
    xq48[..D].copy_from_slice(&st.bbwenet.output_buffer);
    for i in 0..3 * len - D {
        xq48[i + D] = to_i16(out_buffer[i]);
    }
    for i in 0..D {
        st.bbwenet.output_buffer[i] = to_i16(out_buffer[3 * len - D + i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The built-in models load, and a NoLACE frame and a BWE frame run on
    /// silence and stay silent.
    #[test]
    fn builtin_models_run() {
        let model = OsceModel::builtin().unwrap();
        let mut st = Osce::new();
        let input = OsceFrameInput { fs_khz: 16, nb_subfr: 4, lpc_order: 16, num_bits: 200, ..Default::default() };
        let mut xq = [0i16; 320];
        for _ in 0..3 {
            osce_enhance_frame(Some(&model), &mut st, &input, &mut xq);
        }
        assert_eq!(st.features.reset, 0);
        assert!(xq.iter().all(|&x| x.abs() <= 1));
        let mut bwe = OsceBwe::new();
        osce_bwe_reset(&mut bwe);
        let mut xq48 = [0i16; 960];
        osce_bwe(&model, &mut bwe, &mut xq48, &[0i16; 320]);
        assert!(xq48.iter().all(|&x| x.abs() <= 1));
    }
}
