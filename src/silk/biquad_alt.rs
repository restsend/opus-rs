use crate::silk::macros::*;

/// `input` is read through `sample`, which yields the 16-bit input value.
pub fn silk_biquad_alt_stride1<T: Copy>(
    input: &[T],
    sample: impl Fn(T) -> i16,
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    s: &mut [i32; 2],
    output: &mut [i16],
) {
    let len = input.len().min(output.len());

    let a0_l_q28 = (-a_q28[0]) & 0x00003FFF;
    let a0_u_q28 = (-a_q28[0]) >> 14;
    let a1_l_q28 = (-a_q28[1]) & 0x00003FFF;
    let a1_u_q28 = (-a_q28[1]) >> 14;

    for k in 0..len {
        let inval = sample(input[k]) as i32;
        let out32_q14 = silk_smlawb(s[0], b_q28[0], inval) << 2;

        s[0] = s[1] + silk_rshift_round(silk_smulwb(out32_q14, a0_l_q28), 14);
        s[0] = silk_smlawb(s[0], out32_q14, a0_u_q28);
        s[0] = silk_smlawb(s[0], b_q28[1], inval);

        s[1] = silk_rshift_round(silk_smulwb(out32_q14, a1_l_q28), 14);
        s[1] = silk_smlawb(s[1], out32_q14, a1_u_q28);
        s[1] = silk_smlawb(s[1], b_q28[2], inval);

        let out_val = silk_sat16(silk_rshift(out32_q14 + (1 << 14) - 1, 14)) as i16;
        output[k] = out_val;
    }
}

/// `input` is read through `sample`, which yields the 16-bit input value.
pub fn silk_biquad_alt_stride2<T: Copy>(
    input: &[T],
    sample: impl Fn(T) -> i16,
    b_q28: &[i32; 3],
    a_q28: &[i32; 2],
    s: &mut [i32; 4],
    output: &mut [i16],
    len: usize,
) {
    let a0_l_q28 = (-a_q28[0]) & 0x00003FFF;
    let a0_u_q28 = (-a_q28[0]) >> 14;
    let a1_l_q28 = (-a_q28[1]) & 0x00003FFF;
    let a1_u_q28 = (-a_q28[1]) >> 14;

    for k in 0..len {
        let in0 = sample(input[2 * k]) as i32;
        let in1 = sample(input[2 * k + 1]) as i32;
        let out32_q14_0 = silk_smlawb(s[0], b_q28[0], in0) << 2;
        let out32_q14_1 = silk_smlawb(s[2], b_q28[0], in1) << 2;

        s[0] = s[1] + silk_rshift_round(silk_smulwb(out32_q14_0, a0_l_q28), 14);
        s[2] = s[3] + silk_rshift_round(silk_smulwb(out32_q14_1, a0_l_q28), 14);
        s[0] = silk_smlawb(s[0], out32_q14_0, a0_u_q28);
        s[2] = silk_smlawb(s[2], out32_q14_1, a0_u_q28);
        s[0] = silk_smlawb(s[0], b_q28[1], in0);
        s[2] = silk_smlawb(s[2], b_q28[1], in1);

        s[1] = silk_rshift_round(silk_smulwb(out32_q14_0, a1_l_q28), 14);
        s[3] = silk_rshift_round(silk_smulwb(out32_q14_1, a1_l_q28), 14);
        s[1] = silk_smlawb(s[1], out32_q14_0, a1_u_q28);
        s[3] = silk_smlawb(s[3], out32_q14_1, a1_u_q28);
        s[1] = silk_smlawb(s[1], b_q28[2], in0);
        s[3] = silk_smlawb(s[3], b_q28[2], in1);

        output[2 * k] = silk_sat16(silk_rshift(out32_q14_0 + (1 << 14) - 1, 14)) as i16;
        output[2 * k + 1] = silk_sat16(silk_rshift(out32_q14_1 + (1 << 14) - 1, 14)) as i16;
    }
}
