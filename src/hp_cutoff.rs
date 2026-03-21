use crate::silk::biquad_alt::{silk_biquad_alt_stride1, silk_biquad_alt_stride2};
use crate::silk::macros::*;

const SILK_FIX_CONST_19: i32 = ((1.5 * std::f64::consts::PI / 1000.0) * (1 << 19) as f64 + 0.5) as i32;

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

    let fc_q19 = silk_div32_16(silk_smulbb(SILK_FIX_CONST_19, cutoff_hz), fs / 1000);

    let r_q28 = (1i32 << 28) - silk_mul(471, fc_q19);

    b_q28[0] = r_q28;
    b_q28[1] = -silk_lshift(r_q28, 1);
    b_q28[2] = r_q28;

    let r_q22 = silk_rshift(r_q28, 6);
    a_q28[0] = silk_smulww(r_q22, silk_smulww(fc_q19, fc_q19) - (2i32 << 22));
    a_q28[1] = silk_smulww(r_q22, r_q22);

    let mut input_i16 = vec![0i16; input.len()];
    for i in 0..input.len() {
        let sample = (input[i] * 32768.0 + 0.5).floor().clamp(-32768.0, 32767.0);
        input_i16[i] = sample as i16;
    }

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
