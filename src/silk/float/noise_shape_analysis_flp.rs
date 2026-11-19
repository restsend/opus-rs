//! libopus `silk/float/noise_shape_analysis_FLP.c`: noise shaping filters,
//! initial gains, tilt and harmonic shaping of a frame.

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::silk::define::*;
use crate::silk::float::sigproc_flp::*;
use crate::silk::float::structs_flp::SilkEncoderControlFlp;
use crate::silk::structs::SilkEncoderState;
use crate::silk::tuning_parameters::*;

/// `SHAPE_LPC_WIN_MAX` (define.h): 15 ms at 16 kHz.
const SHAPE_LPC_WIN_MAX: usize = 15 * MAX_FS_KHZ;
/// `MIN_QGAIN_DB` (define.h).
const MIN_QGAIN_DB: i32 = 2;

/// The gain that gives the warped filter `coefs` a zero mean log frequency
/// response on a non-warped frequency scale.
fn warped_gain(coefs: &[f32], lambda: f32) -> f32 {
    let lambda = -lambda;
    let order = coefs.len();
    let mut gain = coefs[order - 1];
    for &c in coefs[..order - 1].iter().rev() {
        gain = lambda * gain + c;
    }
    1.0 / (1.0 - lambda * gain)
}

/// Converts warped coefficients to monic pseudo-warped ones, limiting their
/// largest magnitude to `limit` by bandwidth expansion of the true ones.
fn warped_true2monic_coefs(coefs: &mut [f32], lambda: f32, limit: f32) {
    let order = coefs.len();
    let to_monic = |coefs: &mut [f32]| {
        for i in (1..order).rev() {
            coefs[i - 1] -= lambda * coefs[i];
        }
        let gain = (1.0 - lambda * lambda) / (1.0 + lambda * coefs[0]);
        for c in coefs.iter_mut() {
            *c *= gain;
        }
        gain
    };
    let mut gain = to_monic(coefs);
    for iter in 0..10 {
        let (ind, maxabs) = max_abs(coefs);
        if maxabs <= limit {
            return;
        }
        // Back to true warped coefficients.
        for i in 1..order {
            coefs[i - 1] += lambda * coefs[i];
        }
        gain = 1.0 / gain;
        for c in coefs.iter_mut() {
            *c *= gain;
        }
        let chirp =
            0.99 - (0.8 + 0.1 * iter as f32) * (maxabs - limit) / (maxabs * (ind + 1) as f32);
        silk_bwexpander_flp(coefs, chirp);
        gain = to_monic(coefs);
    }
}

/// Limits the largest magnitude of `coefs` to `limit` by bandwidth
/// expansion.
fn limit_coefs(coefs: &mut [f32], limit: f32) {
    for iter in 0..10 {
        let (ind, maxabs) = max_abs(coefs);
        if maxabs <= limit {
            return;
        }
        let chirp =
            0.99 - (0.8 + 0.1 * iter as f32) * (maxabs - limit) / (maxabs * (ind + 1) as f32);
        silk_bwexpander_flp(coefs, chirp);
    }
}

/// The index and value of the first largest magnitude.
fn max_abs(coefs: &[f32]) -> (usize, f32) {
    let mut maxabs = -1.0f32;
    let mut ind = 0;
    for (i, &c) in coefs.iter().enumerate() {
        let tmp = c.abs();
        if tmp > maxabs {
            maxabs = tmp;
            ind = i;
        }
    }
    (ind, maxabs)
}

/// libopus `silk_noise_shape_analysis_FLP`: the noise shaping AR filters,
/// gains, tilt, low-frequency and harmonic shaping of the frame, and the
/// quantization offset type of an unvoiced frame. `pitch_res` is the pitch
/// analysis residual of the frame; `x` the input from `la_shape` samples
/// before the frame (`x - la_shape` in libopus).
pub fn silk_noise_shape_analysis_flp(
    ps_enc: &mut SilkEncoderState,
    ctrl: &mut SilkEncoderControlFlp,
    pitch_res: &[f32],
    x: &[f32],
) {
    let cmn = &mut ps_enc.s_cmn;
    let fs_khz = cmn.fs_khz as usize;
    let nb_subfr = cmn.nb_subfr as usize;
    let shaping_order = cmn.shaping_lpc_order as usize;
    let signal_type = i32::from(cmn.indices.signal_type);

    // Gain control.
    let mut snr_adj_db = cmn.snr_db_q7 as f32 * (1.0 / 128.0);
    ctrl.input_quality = 0.5
        * (cmn.input_quality_bands_q15[0] + cmn.input_quality_bands_q15[1]) as f32
        * (1.0 / 32768.0);
    ctrl.coding_quality = silk_sigmoid(0.25 * (snr_adj_db - 20.0));
    if cmn.use_cbr == 0 {
        // Less coding SNR during low speech activity.
        let b = 1.0 - cmn.speech_activity_q8 as f32 * (1.0 / 256.0);
        snr_adj_db -=
            BG_SNR_DECR_dB * ctrl.coding_quality * (0.5 + 0.5 * ctrl.input_quality) * b * b;
    }
    if signal_type == TYPE_VOICED {
        snr_adj_db += HARM_SNR_INCR_dB * ps_enc.ltp_corr;
    } else {
        snr_adj_db += (-0.4 * cmn.snr_db_q7 as f32 * (1.0 / 128.0) + 6.0) * (1.0 - ctrl.input_quality);
    }

    // Sparseness: the quantization offset of unvoiced frames.
    if signal_type == TYPE_VOICED {
        cmn.indices.quant_offset_type = 0;
    } else {
        let n_samples = 2 * fs_khz;
        let n_segs = SUB_FRAME_LENGTH_MS * nb_subfr / 2;
        let mut energy_variation = 0.0f32;
        let mut log_energy_prev = 0.0f32;
        for k in 0..n_segs {
            let seg = &pitch_res[k * n_samples..(k + 1) * n_samples];
            let nrg = n_samples as f32 + silk_energy_flp(seg) as f32;
            let log_energy = silk_log2(f64::from(nrg));
            if k > 0 {
                energy_variation += (log_energy - log_energy_prev).abs();
            }
            log_energy_prev = log_energy;
        }
        cmn.indices.quant_offset_type =
            if energy_variation > ENERGY_VARIATION_THRESHOLD_QNT_OFFSET * (n_segs - 1) as f32 {
                0
            } else {
                1
            };
    }

    // More bandwidth expansion for signals with high prediction gain.
    let strength = FIND_PITCH_WHITE_NOISE_FRACTION * ctrl.pred_gain;
    let bw_exp = BANDWIDTH_EXPANSION / (1.0 + strength * strength);
    // More warping moves quantization noise up in frequency.
    let warping = cmn.warping_q16 as f32 / 65536.0 + 0.01 * ctrl.coding_quality;

    // Noise shaping AR filters and gains.
    let shape_win_length = cmn.shape_win_length as usize;
    let subfr_length = cmn.subfr_length as usize;
    let mut x_windowed = [0.0f32; SHAPE_LPC_WIN_MAX];
    let mut auto_corr = [0.0f32; MAX_SHAPE_LPC_ORDER + 1];
    let mut rc = [0.0f32; MAX_SHAPE_LPC_ORDER + 1];
    for k in 0..nb_subfr {
        // Sine slope, flat part, cosine slope.
        let x_ptr = &x[k * subfr_length..];
        let flat_part = fs_khz * 3;
        let slope_part = (shape_win_length - flat_part) / 2;
        silk_apply_sine_window_flp(&mut x_windowed, x_ptr, 1, slope_part);
        let mut shift = slope_part;
        x_windowed[shift..shift + flat_part].copy_from_slice(&x_ptr[shift..shift + flat_part]);
        shift += flat_part;
        silk_apply_sine_window_flp(&mut x_windowed[shift..], &x_ptr[shift..], 2, slope_part);

        let win = &x_windowed[..shape_win_length];
        if cmn.warping_q16 > 0 {
            silk_warped_autocorrelation_flp(&mut auto_corr, win, warping, shaping_order);
        } else {
            silk_autocorrelation_flp(&mut auto_corr[..shaping_order + 1], win, cmn.arch);
        }
        // White noise as a fraction of the energy.
        auto_corr[0] += auto_corr[0] * SHAPE_WHITE_NOISE_FRACTION + 1.0;

        let nrg = silk_schur_flp(&mut rc, &auto_corr, shaping_order);
        let ar = &mut ctrl.ar[k * MAX_SHAPE_LPC_ORDER..k * MAX_SHAPE_LPC_ORDER + shaping_order];
        silk_k2a_flp(ar, &rc[..shaping_order]);
        ctrl.gains[k] = f64::from(nrg).sqrt() as f32;
        if cmn.warping_q16 > 0 {
            ctrl.gains[k] *= warped_gain(ar, warping);
        }
        silk_bwexpander_flp(ar, bw_exp);
        if cmn.warping_q16 > 0 {
            warped_true2monic_coefs(ar, warping, 3.999);
        } else {
            limit_coefs(ar, 3.999);
        }
    }

    // Higher gains during low speech activity.
    let gain_mult = 2.0f64.powf(f64::from(-0.16f32 * snr_adj_db)) as f32;
    let gain_add = 2.0f64.powf(f64::from(0.16f32 * MIN_QGAIN_DB as f32)) as f32;
    for g in &mut ctrl.gains[..nb_subfr] {
        *g *= gain_mult;
        *g += gain_add;
    }

    // Low-frequency shaping and noise tilt; less for noisy inputs.
    let mut strength = LOW_FREQ_SHAPING
        * (1.0
            + LOW_QUALITY_LOW_FREQ_SHAPING_DECR
                * (cmn.input_quality_bands_q15[0] as f32 * (1.0 / 32768.0) - 1.0));
    strength *= cmn.speech_activity_q8 as f32 * (1.0 / 256.0);
    let tilt;
    if signal_type == TYPE_VOICED {
        // Less low-frequency quantization noise for periodic signals,
        // depending on the pitch lag.
        for k in 0..nb_subfr {
            let b = 0.2 / fs_khz as f32 + 3.0 / ctrl.pitch_l[k] as f32;
            ctrl.lf_ma_shp[k] = -1.0 + b;
            ctrl.lf_ar_shp[k] = 1.0 - b - b * strength;
        }
        tilt = -HP_NOISE_COEF
            - (1.0 - HP_NOISE_COEF)
                * HARM_HP_NOISE_COEF
                * cmn.speech_activity_q8 as f32
                * (1.0 / 256.0);
    } else {
        let b = 1.3 / fs_khz as f32;
        ctrl.lf_ma_shp[0] = -1.0 + b;
        ctrl.lf_ar_shp[0] = 1.0 - b - b * strength * 0.6;
        for k in 1..nb_subfr {
            ctrl.lf_ma_shp[k] = ctrl.lf_ma_shp[0];
            ctrl.lf_ar_shp[k] = ctrl.lf_ar_shp[0];
        }
        tilt = -HP_NOISE_COEF;
    }

    // Harmonic shaping (USE_HARM_SHAPING).
    let harm_shape_gain = if signal_type == TYPE_VOICED {
        // More for high bitrates or noisy input, less for less periodic
        // signals.
        let mut g = HARMONIC_SHAPING;
        g += HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING
            * (1.0 - (1.0 - ctrl.coding_quality) * ctrl.input_quality);
        g * f64::from(ps_enc.ltp_corr).sqrt() as f32
    } else {
        0.0
    };

    // Smoothing over subframes.
    let shape = &mut ps_enc.s_shape;
    for k in 0..nb_subfr {
        shape.harm_shape_gain_smth += SUBFR_SMTH_COEF * (harm_shape_gain - shape.harm_shape_gain_smth);
        ctrl.harm_shape_gain[k] = shape.harm_shape_gain_smth;
        shape.tilt_smth += SUBFR_SMTH_COEF * (tilt - shape.tilt_smth);
        ctrl.tilt[k] = shape.tilt_smth;
    }
}
