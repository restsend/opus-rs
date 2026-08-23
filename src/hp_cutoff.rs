use crate::silk::biquad_alt::{silk_biquad_alt_stride1, silk_biquad_alt_stride2};
use crate::silk::macros::*;

#[cfg(not(feature = "std"))]
use crate::compat::Math;

const SILK_FIX_CONST_19: i32 =
    ((1.5 * core::f64::consts::PI / 1000.0) * (1 << 19) as f64 + 0.5) as i32;

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

    const MAX_HP_INPUT: usize = 11520;
    debug_assert!(input.len() <= MAX_HP_INPUT);
    let mut input_i16_buf = [0i16; MAX_HP_INPUT];
    let input_i16 = &mut input_i16_buf[..input.len()];
    for i in 0..input.len() {
        let sample = (input[i] * 32768.0 + 0.5).floor().clamp(-32768.0, 32767.0);
        input_i16[i] = sample as i16;
    }

    if channels == 1 {
        let s = &mut [hp_mem[0], hp_mem[1]];
        silk_biquad_alt_stride1(input_i16, &b_q28, &a_q28, s, output);
        hp_mem[0] = s[0];
        hp_mem[1] = s[1];
    } else {
        let s = &mut [hp_mem[0], hp_mem[1], hp_mem[2], hp_mem[3]];
        silk_biquad_alt_stride2(input_i16, &b_q28, &a_q28, s, output, len);
        hp_mem[0] = s[0];
        hp_mem[1] = s[1];
        hp_mem[2] = s[2];
        hp_mem[3] = s[3];
    }
}

/// Direct-form-II-transposed second-order ARMA filter with **float** in/out,
/// mirroring the float build of C `silk_biquad_res()` (opus_encoder.c). Used by
/// the float `hp_cutoff_float` path so CELT sees the exact same pre-emphasis
/// samples as libopus 1.6.
///
/// `B_Q28`/`A_Q28` are Q28 coefficients; the state `S` is 2 elements per
/// channel (stride pass). Output is written with the same `stride` as the input.
fn silk_biquad_res_float(
    input: &[f32],
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    s: &mut [f32; 2],
    output: &mut [f32],
    len: usize,
    stride: usize,
) {
    let inv = 1.0f32 / ((1i32 << 28) as f32);
    let a0 = a_q28[0] as f32 * inv;
    let a1 = a_q28[1] as f32 * inv;
    let b0 = b_q28[0] as f32 * inv;
    let b1 = b_q28[1] as f32 * inv;
    let b2 = b_q28[2] as f32 * inv;

    for k in 0..len {
        let inval = input[k * stride];
        let vout = s[0] + b0 * inval;
        s[0] = s[1] - vout * a0 + b1 * inval;
        s[1] = -vout * a1 + b2 * inval + 1e-30;
        output[k * stride] = vout;
    }
}

/// Floating-point high-pass `hp_cutoff` (C `hp_cutoff()` float path,
/// opus_encoder.c:441). Writes float output; state is 4 floats.
pub fn hp_cutoff_float(
    input: &[f32],
    cutoff_hz: i32,
    output: &mut [f32],
    hp_mem: &mut [f32; 4],
    len: usize,
    channels: usize,
    fs: i32,
) {
    let fc_q19 = silk_div32_16(silk_smulbb(SILK_FIX_CONST_19, cutoff_hz), fs / 1000);

    let r_q28 = (1i32 << 28) - silk_mul(471, fc_q19);

    let b_q28 = [r_q28, -silk_lshift(r_q28, 1), r_q28];

    let r_q22 = silk_rshift(r_q28, 6);
    let a_q28 = [
        silk_smulww(r_q22, silk_smulww(fc_q19, fc_q19) - (2i32 << 22)),
        silk_smulww(r_q22, r_q22),
    ];

    if channels == 1 {
        let mut s = [hp_mem[0], hp_mem[1]];
        silk_biquad_res_float(input, &b_q28, &a_q28, &mut s, output, len, 1);
        hp_mem[0] = s[0];
        hp_mem[1] = s[1];
    } else {
        let mut s0 = [hp_mem[0], hp_mem[1]];
        silk_biquad_res_float(input, &b_q28, &a_q28, &mut s0, output, len, channels);
        hp_mem[0] = s0[0];
        hp_mem[1] = s0[1];
        let mut s1 = [hp_mem[2], hp_mem[3]];
        silk_biquad_res_float(
            &input[1..],
            &b_q28,
            &a_q28,
            &mut s1,
            &mut output[1..],
            len,
            channels,
        );
        hp_mem[2] = s1[0];
        hp_mem[3] = s1[1];
    }
}

/// DC-reject first-order high-pass filter (C `dc_reject()` float path,
/// opus_encoder.c:507). `VERY_SMALL` matches `arch.h` float builds (1e-30f).
/// Used for the CELT/hybrid input when the application is not VOIP.
pub fn dc_reject_float(
    input: &[f32],
    cutoff_hz: i32,
    output: &mut [f32],
    hp_mem: &mut [f32; 4],
    len: usize,
    channels: usize,
    fs: i32,
) {
    let coef = 6.3f32 * cutoff_hz as f32 / fs as f32;
    let coef2 = 1.0 - coef;
    if channels == 2 {
        let mut m0 = hp_mem[0];
        let mut m2 = hp_mem[2];
        for i in 0..len {
            let x0 = input[2 * i];
            let x1 = input[2 * i + 1];
            let out0 = x0 - m0;
            let out1 = x1 - m2;
            m0 = coef * x0 + 1e-30 + coef2 * m0;
            m2 = coef * x1 + 1e-30 + coef2 * m2;
            output[2 * i] = out0;
            output[2 * i + 1] = out1;
        }
        hp_mem[0] = m0;
        hp_mem[2] = m2;
    } else {
        let mut m0 = hp_mem[0];
        for i in 0..len {
            let x = input[i];
            let y = x - m0;
            m0 = coef * x + 1e-30 + coef2 * m0;
            output[i] = y;
        }
        hp_mem[0] = m0;
    }
}
