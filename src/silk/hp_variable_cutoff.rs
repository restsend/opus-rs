use crate::silk::define::TYPE_VOICED;
/// High-pass filter with cutoff frequency adaptation based on pitch lag statistics.
/// Port of silk/HP_variable_cutoff.c
///
/// This filter adapts its cutoff frequency based on the detected pitch frequency,
/// providing better noise rejection for voiced speech.
use crate::silk::lin2log::silk_lin2log;
use crate::silk::macros::*;
use crate::silk::structs::SilkEncoderStateCommon;
use crate::silk::tuning_parameters::*;

/// Adaptive high-pass filter cutoff frequency update.
/// Updates the smoothed cutoff frequency in the encoder state based on pitch lag.
/// The actual filtering is done by the biquad filter applied to the input.
pub fn silk_hp_variable_cutoff(ps_enc: &mut SilkEncoderStateCommon) {
    if ps_enc.prev_signal_type == TYPE_VOICED {
        // Compute pitch frequency in Hz (Q16)
        let pitch_freq_hz_q16 =
            ((ps_enc.fs_khz as i64 * 1000 * (1 << 16)) / ps_enc.prev_lag as i64) as i32;
        let pitch_freq_log_q7 = silk_lin2log(pitch_freq_hz_q16) - (16 << 7);

        // Adjustment based on quality
        let quality_q15 = ps_enc.input_quality_bands_q15[0];
        let min_cutoff_log_q7 = silk_lin2log(fix_const_q16(VARIABLE_HP_MIN_CUTOFF_HZ)) - (16 << 7);
        let pitch_freq_log_q7 = silk_smlawb(
            pitch_freq_log_q7,
            silk_smulwb((-quality_q15) << 2, quality_q15),
            pitch_freq_log_q7 - min_cutoff_log_q7,
        );

        // delta_freq = pitch_freq_log - smoothed value
        let mut delta_freq_q7 = pitch_freq_log_q7 - silk_rshift(ps_enc.variable_hp_smth1_q15, 8);
        if delta_freq_q7 < 0 {
            // Less smoothing for decreasing pitch frequency
            delta_freq_q7 *= 3;
        }

        // Limit delta to reduce impact of outliers
        let max_delta = fix_const_q7(VARIABLE_HP_MAX_DELTA_FREQ);
        delta_freq_q7 = silk_limit_32(delta_freq_q7, -max_delta, max_delta);

        // Update smoother
        let smth_coef1 = fix_const_q16(VARIABLE_HP_SMTH_COEF1);
        ps_enc.variable_hp_smth1_q15 = silk_smlawb(
            ps_enc.variable_hp_smth1_q15,
            silk_smulbb(ps_enc.speech_activity_q8, delta_freq_q7),
            smth_coef1,
        );

        // Limit frequency range
        let min_q15 = silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
        let max_q15 = silk_lin2log(VARIABLE_HP_MAX_CUTOFF_HZ) << 8;
        ps_enc.variable_hp_smth1_q15 =
            silk_limit_32(ps_enc.variable_hp_smth1_q15, min_q15, max_q15);
    }
}

/// SILK_FIX_CONST(x, 16) - convert float constant to Q16 fixed point
fn fix_const_q16(x: impl Into<f64>) -> i32 {
    (x.into() * 65536.0) as i32
}

/// SILK_FIX_CONST(x, 7) - convert float constant to Q7 fixed point
fn fix_const_q7(x: impl Into<f64>) -> i32 {
    (x.into() * 128.0) as i32
}
