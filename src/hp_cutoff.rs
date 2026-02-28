/// High-pass filter for input signal
/// Port of opus_encoder.c hp_cutoff()
use crate::silk::biquad_alt::{silk_biquad_alt_stride1, silk_biquad_alt_stride2};
use crate::silk::macros::*;

// SILK_FIX_CONST(1.5 * 3.14159 / 1000, 19) = round(1.5 * 3.14159 / 1000 * 2^19)
// C macro: (int32)((C) * (1<<Q) + 0.5), so we must add 0.5 before truncation
const SILK_FIX_CONST_19: i32 = ((1.5 * 3.14159 / 1000.0) * (1 << 19) as f64 + 0.5) as i32;

/// Apply high-pass filter to input signal
///
/// # Arguments
/// * `input` - Input PCM samples (interleaved if stereo)
/// * `cutoff_hz` - Cutoff frequency in Hz
/// * `output` - Output buffer for filtered samples
/// * `hp_mem` - Filter memory state [4] for stereo, [2] for mono
/// * `len` - Number of samples per channel
/// * `channels` - Number of channels (1 or 2)
/// * `fs` - Sampling rate in Hz
pub fn hp_cutoff(
    input: &[f32],
    cutoff_hz: i32,
    output: &mut [i16],
    hp_mem: &mut [i32],
    len: usize,
    channels: usize,
    fs: i32,
) {
    let mut b_q28 = [0i32; 3];
    let mut a_q28 = [0i32; 2];

    // Fc_Q19 = (1.5 * pi / 1000 * cutoff_Hz) / (Fs/1000)
    let fc_q19 = silk_div32_16(silk_smulbb(SILK_FIX_CONST_19, cutoff_hz), fs / 1000);

    // r = 1.0 - 0.92 * Fc
    // SILK_FIX_CONST(0.92, 9) = round(0.92 * 512) = 471
    let r_q28 = (1i32 << 28) - silk_mul(471, fc_q19);

    // Biquad coefficients: b = r * [1; -2; 1]
    b_q28[0] = r_q28;
    b_q28[1] = -silk_lshift(r_q28, 1);
    b_q28[2] = r_q28;

    // a = [1; -2*r*(1 - 0.5*Fc^2); r^2]
    let r_q22 = silk_rshift(r_q28, 6);
    a_q28[0] = silk_smulww(r_q22, silk_smulww(fc_q19, fc_q19) - (2i32 << 22));
    a_q28[1] = silk_smulww(r_q22, r_q22);

    // Convert f32 input to i16 for biquad processing
    // Matching C's FLOAT2INT16: floor(0.5 + x * 32768.0)
    let mut input_i16 = vec![0i16; input.len()];
    for i in 0..input.len() {
        let sample = (input[i] * 32768.0 + 0.5).floor().clamp(-32768.0, 32767.0);
        input_i16[i] = sample as i16;
    }

    // Apply biquad filter
    if channels == 1 {
        let s = &mut [hp_mem[0], hp_mem[1]];
        silk_biquad_alt_stride1(&input_i16, &b_q28, &a_q28, s, output);
        hp_mem[0] = s[0];
        hp_mem[1] = s[1];
    } else {
        let s = &mut [hp_mem[0], hp_mem[1], hp_mem[2], hp_mem[3]];
        silk_biquad_alt_stride2(&input_i16, &b_q28, &a_q28, s, output, len);
        hp_mem[0] = s[0];
        hp_mem[1] = s[1];
        hp_mem[2] = s[2];
        hp_mem[3] = s[3];
    }
}
