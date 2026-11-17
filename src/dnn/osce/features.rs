//! The OSCE input features (libopus `dnn/osce_features.c`): the LACE and
//! NoLACE features from the SILK decoder's parameters and output, the
//! BBWENet features from the wideband signal, and the cross-fades.

#[cfg(not(feature = "std"))]
use crate::compat::Math;

use super::{
    OsceFrameInput, OSCE_ACORR_START, OSCE_BWE_FEATURE_DIM, OSCE_BWE_HALF_WINDOW_SIZE, OSCE_BWE_MAX_INSTAFREQ_BIN,
    OSCE_BWE_NUM_BANDS, OSCE_BWE_WINDOW_SIZE, OSCE_CLEAN_SPEC_LENGTH, OSCE_CLEAN_SPEC_NUM_BANDS,
    OSCE_CLEAN_SPEC_START, OSCE_FEATURES_MAX_HISTORY, OSCE_FEATURE_DIM, OSCE_LOG_GAIN_START, OSCE_LTP_LENGTH,
    OSCE_LTP_START, OSCE_MAX_FEATURE_FRAMES, OSCE_NOISY_CEPSTRUM_LENGTH, OSCE_NOISY_CEPSTRUM_START,
    OSCE_NOISY_SPEC_NUM_BANDS, OSCE_NO_PITCH_VALUE, TYPE_VOICED,
};
use crate::dnn::fft::Cpx;
use crate::dnn::freq::{dct, forward_transform, NB_BANDS};
use crate::dnn::vec::c_int;

const OSCE_SPEC_WINDOW_SIZE: usize = 320;
const OSCE_SPEC_NUM_FREQS: usize = 161;

/// libopus `OSCEFeatureState`.
#[derive(Clone, Debug)]
pub struct OsceFeatureState {
    pub numbits_smooth: f32,
    pub pitch_hangover_count: i32,
    pub last_lag: i32,
    pub last_type: i32,
    pub signal_history: [f32; OSCE_FEATURES_MAX_HISTORY],
    /// Frames left until the enhanced signal replaces the decoded one:
    /// 2 after a reset (one frame passed through, one cross-faded).
    pub reset: i32,
}

impl OsceFeatureState {
    /// All zero (`OPUS_CLEAR`).
    pub const fn new() -> Self {
        OsceFeatureState {
            numbits_smooth: 0.0,
            pitch_hangover_count: 0,
            last_lag: 0,
            last_type: 0,
            signal_history: [0.0; OSCE_FEATURES_MAX_HISTORY],
            reset: 0,
        }
    }
}

impl Default for OsceFeatureState {
    fn default() -> Self {
        Self::new()
    }
}

/// libopus `OSCEBWEFeatureState`.
#[derive(Clone, Debug)]
pub struct OsceBweFeatureState {
    pub signal_history: [f32; OSCE_BWE_HALF_WINDOW_SIZE],
    pub last_spec: [f32; 2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2],
}

impl OsceBweFeatureState {
    /// All zero (`OPUS_CLEAR`).
    pub const fn new() -> Self {
        OsceBweFeatureState {
            signal_history: [0.0; OSCE_BWE_HALF_WINDOW_SIZE],
            last_spec: [0.0; 2 * OSCE_BWE_MAX_INSTAFREQ_BIN + 2],
        }
    }
}

impl Default for OsceBweFeatureState {
    fn default() -> Self {
        Self::new()
    }
}

const CENTER_BINS_CLEAN: [usize; 64] = [
    0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 33, 35, 38, 40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70,
    73, 75, 78, 80, 82, 85, 88, 90, 92, 95, 98, 100, 102, 105, 108, 110, 112, 115, 118, 120, 122, 125, 128, 130, 132,
    135, 138, 140, 142, 145, 148, 150, 152, 155, 160,
];

const CENTER_BINS_NOISY: [usize; 18] = [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160];

const CENTER_BINS_BWE: [usize; 32] = [
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130,
    135, 140, 145, 150, 160,
];

const BAND_WEIGHTS_CLEAN: [f32; 64] = [
    0.666666666667f32, 0.400000000000f32, 0.333333333333f32, 0.400000000000f32,
    0.500000000000f32, 0.400000000000f32, 0.333333333333f32, 0.400000000000f32,
    0.500000000000f32, 0.400000000000f32, 0.333333333333f32, 0.400000000000f32,
    0.400000000000f32, 0.400000000000f32, 0.400000000000f32, 0.400000000000f32,
    0.500000000000f32, 0.400000000000f32, 0.333333333333f32, 0.400000000000f32,
    0.500000000000f32, 0.400000000000f32, 0.333333333333f32, 0.400000000000f32,
    0.500000000000f32, 0.400000000000f32, 0.333333333333f32, 0.400000000000f32,
    0.400000000000f32, 0.400000000000f32, 0.400000000000f32, 0.400000000000f32,
    0.500000000000f32, 0.400000000000f32, 0.333333333333f32, 0.400000000000f32,
    0.500000000000f32, 0.400000000000f32, 0.333333333333f32, 0.400000000000f32,
    0.500000000000f32, 0.400000000000f32, 0.333333333333f32, 0.400000000000f32,
    0.500000000000f32, 0.400000000000f32, 0.333333333333f32, 0.400000000000f32,
    0.500000000000f32, 0.400000000000f32, 0.333333333333f32, 0.400000000000f32,
    0.500000000000f32, 0.400000000000f32, 0.333333333333f32, 0.400000000000f32,
    0.500000000000f32, 0.400000000000f32, 0.333333333333f32, 0.400000000000f32,
    0.500000000000f32, 0.400000000000f32, 0.250000000000f32, 0.333333333333f32,
];

const BAND_WEIGHTS_NOISY: [f32; 18] = [
    0.400000000000f32, 0.250000000000f32, 0.250000000000f32, 0.250000000000f32,
    0.250000000000f32, 0.250000000000f32, 0.250000000000f32, 0.250000000000f32,
    0.166666666667f32, 0.125000000000f32, 0.125000000000f32, 0.125000000000f32,
    0.083333333333f32, 0.062500000000f32, 0.062500000000f32, 0.050000000000f32,
    0.041666666667f32, 0.080000000000f32,
];

// Double literals in a float array: each is rounded from its double value.
const BAND_WEIGHTS_BWE: [f32; 32] = [
    0.333333333f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32,
    0.200000000f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32,
    0.200000000f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32,
    0.200000000f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32,
    0.200000000f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32,
    0.200000000f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32,
    0.200000000f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32, 0.200000000f64 as f32,
    0.200000000f64 as f32, 0.200000000f64 as f32, 0.133333333f64 as f32, 0.181818182f64 as f32,
];

/// `osce_window`: a sine window of 320 samples.
#[rustfmt::skip]
pub const OSCE_WINDOW: [f32; OSCE_SPEC_WINDOW_SIZE] = [
    0.004908718808f32, 0.014725683311f32, 0.024541228523f32, 0.034354408400f32, 0.044164277127f32,
    0.053969889210f32, 0.063770299562f32, 0.073564563600f32, 0.083351737332f32, 0.093130877450f32,
    0.102901041421f32, 0.112661287575f32, 0.122410675199f32, 0.132148264628f32, 0.141873117332f32,
    0.151584296010f32, 0.161280864678f32, 0.170961888760f32, 0.180626435180f32, 0.190273572448f32,
    0.199902370753f32, 0.209511902052f32, 0.219101240157f32, 0.228669460829f32, 0.238215641862f32,
    0.247738863176f32, 0.257238206902f32, 0.266712757475f32, 0.276161601717f32, 0.285583828929f32,
    0.294978530977f32, 0.304344802381f32, 0.313681740399f32, 0.322988445118f32, 0.332264019538f32,
    0.341507569661f32, 0.350718204573f32, 0.359895036535f32, 0.369037181064f32, 0.378143757022f32,
    0.387213886697f32, 0.396246695891f32, 0.405241314005f32, 0.414196874117f32, 0.423112513073f32,
    0.431987371563f32, 0.440820594212f32, 0.449611329655f32, 0.458358730621f32, 0.467061954019f32,
    0.475720161014f32, 0.484332517110f32, 0.492898192230f32, 0.501416360796f32, 0.509886201809f32,
    0.518306898929f32, 0.526677640552f32, 0.534997619887f32, 0.543266035038f32, 0.551482089078f32,
    0.559644990127f32, 0.567753951426f32, 0.575808191418f32, 0.583806933818f32, 0.591749407690f32,
    0.599634847523f32, 0.607462493302f32, 0.615231590581f32, 0.622941390558f32, 0.630591150148f32,
    0.638180132051f32, 0.645707604824f32, 0.653172842954f32, 0.660575126926f32, 0.667913743292f32,
    0.675187984742f32, 0.682397150168f32, 0.689540544737f32, 0.696617479953f32, 0.703627273726f32,
    0.710569250438f32, 0.717442741007f32, 0.724247082951f32, 0.730981620454f32, 0.737645704427f32,
    0.744238692572f32, 0.750759949443f32, 0.757208846506f32, 0.763584762206f32, 0.769887082016f32,
    0.776115198508f32, 0.782268511401f32, 0.788346427627f32, 0.794348361383f32, 0.800273734191f32,
    0.806121974951f32, 0.811892519997f32, 0.817584813152f32, 0.823198305781f32, 0.828732456844f32,
    0.834186732948f32, 0.839560608398f32, 0.844853565250f32, 0.850065093356f32, 0.855194690420f32,
    0.860241862039f32, 0.865206121757f32, 0.870086991109f32, 0.874883999665f32, 0.879596685080f32,
    0.884224593137f32, 0.888767277786f32, 0.893224301196f32, 0.897595233788f32, 0.901879654283f32,
    0.906077149740f32, 0.910187315596f32, 0.914209755704f32, 0.918144082372f32, 0.921989916403f32,
    0.925746887127f32, 0.929414632439f32, 0.932992798835f32, 0.936481041442f32, 0.939879024058f32,
    0.943186419177f32, 0.946402908026f32, 0.949528180593f32, 0.952561935658f32, 0.955503880820f32,
    0.958353732530f32, 0.961111216112f32, 0.963776065795f32, 0.966348024735f32, 0.968826845041f32,
    0.971212287799f32, 0.973504123096f32, 0.975702130039f32, 0.977806096779f32, 0.979815820533f32,
    0.981731107599f32, 0.983551773378f32, 0.985277642389f32, 0.986908548290f32, 0.988444333892f32,
    0.989884851171f32, 0.991229961288f32, 0.992479534599f32, 0.993633450666f32, 0.994691598273f32,
    0.995653875433f32, 0.996520189401f32, 0.997290456679f32, 0.997964603026f32, 0.998542563469f32,
    0.999024282300f32, 0.999409713092f32, 0.999698818696f32, 0.999891571247f32, 0.999987952167f32,
    0.999987952167f32, 0.999891571247f32, 0.999698818696f32, 0.999409713092f32, 0.999024282300f32,
    0.998542563469f32, 0.997964603026f32, 0.997290456679f32, 0.996520189401f32, 0.995653875433f32,
    0.994691598273f32, 0.993633450666f32, 0.992479534599f32, 0.991229961288f32, 0.989884851171f32,
    0.988444333892f32, 0.986908548290f32, 0.985277642389f32, 0.983551773378f32, 0.981731107599f32,
    0.979815820533f32, 0.977806096779f32, 0.975702130039f32, 0.973504123096f32, 0.971212287799f32,
    0.968826845041f32, 0.966348024735f32, 0.963776065795f32, 0.961111216112f32, 0.958353732530f32,
    0.955503880820f32, 0.952561935658f32, 0.949528180593f32, 0.946402908026f32, 0.943186419177f32,
    0.939879024058f32, 0.936481041442f32, 0.932992798835f32, 0.929414632439f32, 0.925746887127f32,
    0.921989916403f32, 0.918144082372f32, 0.914209755704f32, 0.910187315596f32, 0.906077149740f32,
    0.901879654283f32, 0.897595233788f32, 0.893224301196f32, 0.888767277786f32, 0.884224593137f32,
    0.879596685080f32, 0.874883999665f32, 0.870086991109f32, 0.865206121757f32, 0.860241862039f32,
    0.855194690420f32, 0.850065093356f32, 0.844853565250f32, 0.839560608398f32, 0.834186732948f32,
    0.828732456844f32, 0.823198305781f32, 0.817584813152f32, 0.811892519997f32, 0.806121974951f32,
    0.800273734191f32, 0.794348361383f32, 0.788346427627f32, 0.782268511401f32, 0.776115198508f32,
    0.769887082016f32, 0.763584762206f32, 0.757208846506f32, 0.750759949443f32, 0.744238692572f32,
    0.737645704427f32, 0.730981620454f32, 0.724247082951f32, 0.717442741007f32, 0.710569250438f32,
    0.703627273726f32, 0.696617479953f32, 0.689540544737f32, 0.682397150168f32, 0.675187984742f32,
    0.667913743292f32, 0.660575126926f32, 0.653172842954f32, 0.645707604824f32, 0.638180132051f32,
    0.630591150148f32, 0.622941390558f32, 0.615231590581f32, 0.607462493302f32, 0.599634847523f32,
    0.591749407690f32, 0.583806933818f32, 0.575808191418f32, 0.567753951426f32, 0.559644990127f32,
    0.551482089078f32, 0.543266035038f32, 0.534997619887f32, 0.526677640552f32, 0.518306898929f32,
    0.509886201809f32, 0.501416360796f32, 0.492898192230f32, 0.484332517110f32, 0.475720161014f32,
    0.467061954019f32, 0.458358730621f32, 0.449611329655f32, 0.440820594212f32, 0.431987371563f32,
    0.423112513073f32, 0.414196874117f32, 0.405241314005f32, 0.396246695891f32, 0.387213886697f32,
    0.378143757022f32, 0.369037181064f32, 0.359895036535f32, 0.350718204573f32, 0.341507569661f32,
    0.332264019538f32, 0.322988445118f32, 0.313681740399f32, 0.304344802381f32, 0.294978530977f32,
    0.285583828929f32, 0.276161601717f32, 0.266712757475f32, 0.257238206902f32, 0.247738863176f32,
    0.238215641862f32, 0.228669460829f32, 0.219101240157f32, 0.209511902052f32, 0.199902370753f32,
    0.190273572448f32, 0.180626435180f32, 0.170961888760f32, 0.161280864678f32, 0.151584296010f32,
    0.141873117332f32, 0.132148264628f32, 0.122410675199f32, 0.112661287575f32, 0.102901041421f32,
    0.093130877450f32, 0.083351737332f32, 0.073564563600f32, 0.063770299562f32, 0.053969889210f32,
    0.044164277127f32, 0.034354408400f32, 0.024541228523f32, 0.014725683311f32, 0.004908718808f32,
];

/// `apply_filterbank`: triangular bands centered on `center_bins`, each
/// weighted by `band_weights`.
fn apply_filterbank(x_out: &mut [f32], x_in: &[f32], center_bins: &[usize], band_weights: &[f32]) {
    let num_bands = center_bins.len();
    x_out[0] = 0.0;
    for b in 0..num_bands - 1 {
        x_out[b + 1] = 0.0;
        for i in center_bins[b]..center_bins[b + 1] {
            let frac = (center_bins[b + 1] - i) as f32 / (center_bins[b + 1] - center_bins[b]) as f32;
            x_out[b] += band_weights[b] * frac * x_in[i];
            x_out[b + 1] += band_weights[b + 1] * (1.0 - frac) * x_in[i];
        }
    }
    x_out[num_bands - 1] += band_weights[num_bands - 1] * x_in[center_bins[num_bands - 1]];
}

/// The magnitudes of the first `OSCE_SPEC_NUM_FREQS` bins of a transform,
/// times 320 (`sqrt` in double).
fn magnitudes(out: &mut [f32], spec: &[Cpx]) {
    for (o, c) in out[..OSCE_SPEC_NUM_FREQS].iter_mut().zip(spec) {
        *o = (OSCE_SPEC_WINDOW_SIZE as f64 * ((c.r * c.r + c.i * c.i) as f64).sqrt()) as f32;
    }
}

/// `mag_spec_320_onesided`.
fn mag_spec_320_onesided(out: &mut [f32], input: &[f32]) {
    let mut buffer = [Cpx::default(); OSCE_SPEC_WINDOW_SIZE];
    forward_transform(&mut buffer, input);
    magnitudes(out, &buffer);
}

/// `calculate_log_spectrum_from_lpc`: the log filterbank energies of the
/// LPC synthesis filter `1/A(z)`.
fn calculate_log_spectrum_from_lpc(spec: &mut [f32], a_q12: &[i16], lpc_order: usize) {
    let mut buffer = [0.0f32; OSCE_SPEC_WINDOW_SIZE];
    buffer[0] = 1.0;
    for i in 0..lpc_order {
        buffer[i + 1] = -(a_q12[i] as f32) / 4096.0;
    }
    let mut mag = [0.0f32; OSCE_SPEC_NUM_FREQS];
    mag_spec_320_onesided(&mut mag, &buffer);
    for m in mag.iter_mut() {
        *m = 1.0 / (*m + 1e-9f32);
    }
    apply_filterbank(spec, &mag, &CENTER_BINS_CLEAN, &BAND_WEIGHTS_CLEAN);
    for s in spec[..OSCE_CLEAN_SPEC_NUM_BANDS].iter_mut() {
        *s = (0.3f32 as f64 * ((*s + 1e-9f32) as f64).ln()) as f32;
    }
}

/// `calculate_cepstrum`: the DCT of the log filterbank magnitudes of the
/// windowed `signal[..320]`.
fn calculate_cepstrum(cepstrum: &mut [f32], signal: &[f32]) {
    let mut buffer = [0.0f32; OSCE_SPEC_WINDOW_SIZE];
    for ((b, &w), &s) in buffer.iter_mut().zip(&OSCE_WINDOW).zip(signal) {
        *b = w * s;
    }
    let mut mag = [0.0f32; OSCE_SPEC_NUM_FREQS];
    mag_spec_320_onesided(&mut mag, &buffer);
    let mut spec = [0.0f32; OSCE_NOISY_SPEC_NUM_BANDS];
    apply_filterbank(&mut spec, &mag, &CENTER_BINS_NOISY, &BAND_WEIGHTS_NOISY);
    for s in spec.iter_mut() {
        *s = ((*s + 1e-9f32) as f64).ln() as f32;
    }
    const _: () = assert!(OSCE_NOISY_SPEC_NUM_BANDS == NB_BANDS);
    dct(cepstrum, &spec);
}

/// `calculate_acorr`: the normalized correlations of the 80 samples at
/// `signal[pos..]` with those `lag - 2 ..= lag + 2` samples earlier.
fn calculate_acorr(acorr: &mut [f32], signal: &[f32], pos: usize, lag: usize) {
    for k in 0..5 {
        let d = pos + k - 2 - lag;
        let mut xx = 0.0f32;
        let mut xy = 0.0f32;
        let mut yy = 0.0f32;
        for n in 0..80 {
            let x = signal[pos + n];
            let y = signal[d + n];
            xx += x * x;
            yy += y * y;
            xy += x * y;
        }
        acorr[k] = (xy as f64 / ((xx * yy + 1e-9f32) as f64).sqrt()) as f32;
    }
}

/// `pitch_postprocessing` with libopus's settings (OSCE_PITCH_HANGOVER 0,
/// OSCE_HANGOVER_BUGFIX undefined): the lag of a voiced frame, else
/// `OSCE_NO_PITCH_VALUE`.
fn pitch_postprocessing(st: &mut OsceFeatureState, lag: i32, signal_type: i32) -> i32 {
    let new_lag;
    if signal_type != TYPE_VOICED {
        new_lag = OSCE_NO_PITCH_VALUE;
        st.pitch_hangover_count = 0;
    } else {
        new_lag = lag;
        st.last_lag = lag;
        st.pitch_hangover_count = 0;
    }
    st.last_type = signal_type;
    new_lag
}

/// `osce_calculate_features`: `OSCE_FEATURE_DIM` features per 5 ms
/// subframe of the SILK frame `xq` (`input.nb_subfr` subframes of 80
/// samples), the bit count and its smoothed value, and the pitch lag per
/// subframe.
pub fn osce_calculate_features(
    st: &mut OsceFeatureState,
    input: &OsceFrameInput,
    features: &mut [f32; OSCE_MAX_FEATURE_FRAMES * OSCE_FEATURE_DIM],
    numbits: &mut [f32; 2],
    periods: &mut [i32; OSCE_MAX_FEATURE_FRAMES],
    xq: &[i16],
) {
    const H: usize = OSCE_FEATURES_MAX_HISTORY;
    let mut buffer = [0.0f32; H + OSCE_MAX_FEATURE_FRAMES * 80];
    let num_subframes = input.nb_subfr;
    assert!(num_subframes <= OSCE_MAX_FEATURE_FRAMES);
    let num_samples = num_subframes * 80;

    st.numbits_smooth = 0.9f32 * st.numbits_smooth + 0.1f32 * input.num_bits as f32;
    numbits[0] = input.num_bits as f32;
    numbits[1] = st.numbits_smooth;

    for (b, &x) in buffer[H..H + num_samples].iter_mut().zip(xq) {
        *b = x as f32 / 32768.0;
    }
    buffer[..H].copy_from_slice(&st.signal_history);

    for k in 0..num_subframes {
        let (prev, cur) = features.split_at_mut(k * OSCE_FEATURE_DIM);
        let pf = &mut cur[..OSCE_FEATURE_DIM];
        let frame = H + k * 80;

        // Clean spectrum from the LPCs and noisy cepstrum from the signal,
        // each updated every other subframe.
        if k % 2 == 0 {
            calculate_log_spectrum_from_lpc(
                &mut pf[OSCE_CLEAN_SPEC_START..],
                &input.pred_coef_q12[k >> 1],
                input.lpc_order,
            );
            calculate_cepstrum(&mut pf[OSCE_NOISY_CEPSTRUM_START..], &buffer[frame - 160..]);
        } else {
            let last = &prev[(k - 1) * OSCE_FEATURE_DIM..];
            pf[OSCE_CLEAN_SPEC_START..OSCE_CLEAN_SPEC_START + OSCE_CLEAN_SPEC_LENGTH]
                .copy_from_slice(&last[OSCE_CLEAN_SPEC_START..OSCE_CLEAN_SPEC_START + OSCE_CLEAN_SPEC_LENGTH]);
            pf[OSCE_NOISY_CEPSTRUM_START..OSCE_NOISY_CEPSTRUM_START + OSCE_NOISY_CEPSTRUM_LENGTH].copy_from_slice(
                &last[OSCE_NOISY_CEPSTRUM_START..OSCE_NOISY_CEPSTRUM_START + OSCE_NOISY_CEPSTRUM_LENGTH],
            );
        }

        periods[k] = pitch_postprocessing(st, input.pitch_l[k], input.signal_type);
        calculate_acorr(&mut pf[OSCE_ACORR_START..], &buffer, frame, periods[k] as usize);

        for i in 0..OSCE_LTP_LENGTH {
            pf[OSCE_LTP_START + i] = input.ltp_coef_q14[k * OSCE_LTP_LENGTH + i] as f32 / 16384.0;
        }
        pf[OSCE_LOG_GAIN_START] = ((input.gains_q16[k] as f32 / 65536.0 + 1e-9f32) as f64).ln() as f32;
    }

    st.signal_history.copy_from_slice(&buffer[num_samples..num_samples + H]);
}

/// `osce_bwe_calculate_features`: `OSCE_BWE_FEATURE_DIM` features per 10 ms
/// of the 16 kHz signal `xq` (a multiple of 160 samples, at most 320): 32
/// log band magnitudes, then the cosines and sines of the phase advance of
/// bins 0 to 40.
pub fn osce_bwe_calculate_features(st: &mut OsceBweFeatureState, features: &mut [f32], xq: &[i16]) {
    const HALF: usize = OSCE_BWE_HALF_WINDOW_SIZE;
    const NBIN: usize = OSCE_BWE_MAX_INSTAFREQ_BIN + 1;
    const _: () = assert!(OSCE_BWE_WINDOW_SIZE == OSCE_SPEC_WINDOW_SIZE);
    assert!(xq.len().is_multiple_of(HALF));
    let num_frames = xq.len() / HALF;
    let mut fft_buffer = [Cpx::default(); OSCE_BWE_WINDOW_SIZE];
    let mut spec = [0.0f32; 2 * NBIN];
    let mut buffer = [0.0f32; OSCE_BWE_WINDOW_SIZE];
    let mut mag_spec = [0.0f32; OSCE_SPEC_NUM_FREQS];

    for frame in 0..num_frames {
        let feat = &mut features[frame * OSCE_BWE_FEATURE_DIM..(frame + 1) * OSCE_BWE_FEATURE_DIM];
        feat.fill(0.0);
        let (lmspec, instafreq) = feat.split_at_mut(OSCE_BWE_NUM_BANDS);
        let x = &xq[frame * HALF..(frame + 1) * HALF];

        buffer[..HALF].copy_from_slice(&st.signal_history);
        for (b, &v) in buffer[HALF..].iter_mut().zip(x) {
            *b = v as f32 / 32768.0;
        }
        st.signal_history.copy_from_slice(&buffer[HALF..]);
        for (b, &w) in buffer.iter_mut().zip(&OSCE_WINDOW) {
            *b *= w;
        }

        forward_transform(&mut fft_buffer, &buffer);

        for k in 0..NBIN {
            spec[2 * k] = ((OSCE_BWE_WINDOW_SIZE as f32 * fft_buffer[k].r) as f64 + 1e-9) as f32;
            spec[2 * k + 1] = OSCE_BWE_WINDOW_SIZE as f32 * fft_buffer[k].i;
            let (re1, im1) = (spec[2 * k], spec[2 * k + 1]);
            let (re2, im2) = (st.last_spec[2 * k], st.last_spec[2 * k + 1]);
            let aux_r = re1 * re2 + im1 * im2;
            let aux_i = im1 * re2 - re1 * im2;
            let aux_abs = ((aux_r * aux_r + aux_i * aux_i) as f64).sqrt() as f32;
            instafreq[k] = (aux_r as f64 / (aux_abs as f64 + 1e-9)) as f32;
            instafreq[k + NBIN] = (aux_i as f64 / (aux_abs as f64 + 1e-9)) as f32;
        }

        magnitudes(&mut mag_spec, &fft_buffer);
        apply_filterbank(lmspec, &mag_spec, &CENTER_BINS_BWE, &BAND_WEIGHTS_BWE);
        for v in lmspec.iter_mut() {
            *v = ((*v as f64) + 1e-9).ln() as f32;
        }

        st.last_spec.copy_from_slice(&spec);
    }
}

/// `osce_cross_fade_10ms`: fades the first 160 samples from `x_in` into
/// `x_enhanced` with the rising half of `OSCE_WINDOW`.
pub fn osce_cross_fade_10ms(x_enhanced: &mut [f32], x_in: &[f32]) {
    assert!(x_enhanced.len() >= 160);
    for i in 0..160 {
        x_enhanced[i] = OSCE_WINDOW[i] * x_enhanced[i] + (1.0 - OSCE_WINDOW[i]) * x_in[i];
    }
}

/// `osce_bwe_cross_fade_10ms`: fades the first 480 samples (48 kHz) from
/// `x_fadeout` into `x_fadein`, with `OSCE_WINDOW` interpolated to 48 kHz.
pub fn osce_bwe_cross_fade_10ms(x_fadein: &mut [i16], x_fadeout: &[i16]) {
    assert!(x_fadein.len() >= 480 && x_fadeout.len() >= 480);
    let f = 1.0f32 / 3.0;
    for i in 0..160 {
        let diff = if i == 159 { 0.0 } else { OSCE_WINDOW[i + 1] - OSCE_WINDOW[i] };
        let mut w_curr = OSCE_WINDOW[i];
        for j in 0..3 {
            let n = 3 * i + j;
            let v = w_curr * x_fadein[n] as f32 + (1.0 - w_curr) * x_fadeout[n] as f32;
            x_fadein[n] = c_int(v as f64 + 0.5) as i16;
            if j < 2 {
                w_curr += diff * f;
            }
        }
    }
}
