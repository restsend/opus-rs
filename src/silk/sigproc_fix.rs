use crate::silk::define::*;
use crate::silk::macros::*;

#[inline]
pub fn silk_k2a_q16(a_q24: &mut [i32], rc_q16: &[i32], order: usize) {
    for k in 0..order {
        let rc = rc_q16[k];
        for n in 0..((k + 1) >> 1) {
            let tmp1 = a_q24[n];
            let tmp2 = a_q24[k - n - 1];
            a_q24[n] = silk_smlaww(tmp1, tmp2, rc);
            a_q24[k - n - 1] = silk_smlaww(tmp2, tmp1, rc);
        }
        a_q24[k] = -(rc << 8);
    }
}

pub fn silk_schur64(rc_q16: &mut [i32], c: &[i32], order: usize) -> i32 {
    let mut c_matrix = [[0i32; 2]; MAX_SHAPE_LPC_ORDER + 1];
    let mut c_tmp1_q30: i32;
    let mut c_tmp2_q30: i32;
    let mut rc_tmp_q31: i32;

    /* Check for invalid input */
    if c[0] <= 0 {
        for i in 0..order {
            rc_q16[i] = 0;
        }
        return 0;
    }

    for k in 0..=order {
        c_matrix[k][0] = c[k];
        c_matrix[k][1] = c[k];
    }

    let mut k = 0;
    while k < order {
        /* Check that we won't be getting an unstable rc, otherwise stop here. */
        if c_matrix[k + 1][0].abs() >= c_matrix[0][1] {
            if c_matrix[k + 1][0] > 0 {
                rc_q16[k] = -64880; // -0.99 in Q16
            } else {
                rc_q16[k] = 64880; // 0.99 in Q16
            }
            k += 1;
            break;
        }

        /* Get reflection coefficient: divide two Q30 values and get result in Q31 */
        rc_tmp_q31 = silk_div32_varq(-c_matrix[k + 1][0], c_matrix[0][1], 31);

        /* Save the output */
        rc_q16[k] = silk_rshift_round(rc_tmp_q31, 15);

        /* Update correlations */
        for n in 0..(order - k) {
            c_tmp1_q30 = c_matrix[n + k + 1][0];
            c_tmp2_q30 = c_matrix[n][1];

            /* Multiply and add the highest int32 */
            c_matrix[n + k + 1][0] =
                c_tmp1_q30.wrapping_add(silk_smmul(c_tmp2_q30 << 1, rc_tmp_q31));
            c_matrix[n][1] = c_tmp2_q30.wrapping_add(silk_smmul(c_tmp1_q30 << 1, rc_tmp_q31));
        }
        k += 1;
    }

    while k < order {
        rc_q16[k] = 0;
        k += 1;
    }

    c_matrix[0][1].max(1)
}

pub fn silk_biquad_alt_stride1(
    input_output: &mut [i16],
    b_q28: &[i32],
    a_q28: &[i32],
    s: &mut [i32],
    len: usize,
) {
    /* Negate A_Q28 values and split in two parts */
    let a0_l_q28 = (-a_q28[0]) & 0x00003FFF; /* lower part */
    let a0_u_q28 = -a_q28[0] >> 14; /* upper part */
    let a1_l_q28 = (-a_q28[1]) & 0x00003FFF; /* lower part */
    let a1_u_q28 = -a_q28[1] >> 14; /* upper part */

    for k in 0..len {
        /* S[ 0 ], S[ 1 ]: Q12 */
        let inval = input_output[k] as i32;
        let out32_q14 = silk_smlawb(s[0], b_q28[0], inval) << 2;

        s[0] = s[1] + silk_rshift_round(silk_smulwb(out32_q14, a0_l_q28), 14);
        s[0] = silk_smlawb(s[0], out32_q14, a0_u_q28);
        s[0] = silk_smlawb(s[0], b_q28[1], inval);

        s[1] = silk_rshift_round(silk_smulwb(out32_q14, a1_l_q28), 14);
        s[1] = silk_smlawb(s[1], out32_q14, a1_u_q28);
        s[1] = silk_smlawb(s[1], b_q28[2], inval);

        /* Scale back to Q0 and saturate */
        input_output[k] = silk_sat16(silk_rshift(out32_q14 + (1 << 14) - 1, 14)) as i16;
    }
}

pub fn silk_biquad_alt_stride2(
    input_output: &mut [i16],
    b_q28: &[i32],
    a_q28: &[i32],
    s: &mut [i32],
    len: usize,
) {
    /* Negate A_Q28 values and split in two parts */
    let a0_l_q28 = (-a_q28[0]) & 0x00003FFF; /* lower part */
    let a0_u_q28 = -a_q28[0] >> 14; /* upper part */
    let a1_l_q28 = (-a_q28[1]) & 0x00003FFF; /* lower part */
    let a1_u_q28 = -a_q28[1] >> 14; /* upper part */

    for k in 0..len {
        /* S[ 0 ], S[ 1 ], S[ 2 ], S[ 3 ]: Q12 */
        let out32_q14_0 = silk_smlawb(s[0], b_q28[0], input_output[2 * k] as i32) << 2;
        let out32_q14_1 = silk_smlawb(s[2], b_q28[0], input_output[2 * k + 1] as i32) << 2;

        s[0] = s[1] + silk_rshift_round(silk_smulwb(out32_q14_0, a0_l_q28), 14);
        s[2] = s[3] + silk_rshift_round(silk_smulwb(out32_q14_1, a0_l_q28), 14);
        s[0] = silk_smlawb(s[0], out32_q14_0, a0_u_q28);
        s[2] = silk_smlawb(s[2], out32_q14_1, a0_u_q28);
        s[0] = silk_smlawb(s[0], b_q28[1], input_output[2 * k] as i32);
        s[2] = silk_smlawb(s[2], b_q28[1], input_output[2 * k + 1] as i32);

        s[1] = silk_rshift_round(silk_smulwb(out32_q14_0, a1_l_q28), 14);
        s[3] = silk_rshift_round(silk_smulwb(out32_q14_1, a1_l_q28), 14);
        s[1] = silk_smlawb(s[1], out32_q14_0, a1_u_q28);
        s[3] = silk_smlawb(s[3], out32_q14_1, a1_u_q28);
        s[1] = silk_smlawb(s[1], b_q28[2], input_output[2 * k] as i32);
        s[3] = silk_smlawb(s[3], b_q28[2], input_output[2 * k + 1] as i32);

        /* Scale back to Q0 and saturate */
        input_output[2 * k] = silk_sat16(silk_rshift(out32_q14_0 + (1 << 14) - 1, 14)) as i16;
        input_output[2 * k + 1] = silk_sat16(silk_rshift(out32_q14_1 + (1 << 14) - 1, 14)) as i16;
    }
}

/// Port of xcorr_kernel_c from celt/pitch.h
/// Computes sum[k] = sum_{j=0}^{len-1} x[j] * y[j+k] for k=0..3
/// using 32-bit wrapping (MAC16_16) arithmetic.
#[inline]
fn xcorr_kernel_c(x: &[i16], y: &[i16], sum: &mut [i32; 4], len: usize) {
    let mut j = 0;
    let mut y_0 = y[0];
    let mut y_1 = y[1];
    let mut y_2 = y[2];
    let mut y_3: i16 = 0;
    let mut yi = 3; /* next y index to read */
    while j + 3 < len {
        let tmp = x[j];
        y_3 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_0);
        sum[1] = mac16_16(sum[1], tmp, y_1);
        sum[2] = mac16_16(sum[2], tmp, y_2);
        sum[3] = mac16_16(sum[3], tmp, y_3);
        let tmp = x[j + 1];
        y_0 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_1);
        sum[1] = mac16_16(sum[1], tmp, y_2);
        sum[2] = mac16_16(sum[2], tmp, y_3);
        sum[3] = mac16_16(sum[3], tmp, y_0);
        let tmp = x[j + 2];
        y_1 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_2);
        sum[1] = mac16_16(sum[1], tmp, y_3);
        sum[2] = mac16_16(sum[2], tmp, y_0);
        sum[3] = mac16_16(sum[3], tmp, y_1);
        let tmp = x[j + 3];
        y_2 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_3);
        sum[1] = mac16_16(sum[1], tmp, y_0);
        sum[2] = mac16_16(sum[2], tmp, y_1);
        sum[3] = mac16_16(sum[3], tmp, y_2);
        j += 4;
    }
    /* Handle remainder (0-3 samples) */
    if j < len {
        let tmp = x[j];
        j += 1;
        y_3 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_0);
        sum[1] = mac16_16(sum[1], tmp, y_1);
        sum[2] = mac16_16(sum[2], tmp, y_2);
        sum[3] = mac16_16(sum[3], tmp, y_3);
    }
    if j < len {
        let tmp = x[j];
        j += 1;
        y_0 = y[yi];
        yi += 1;
        sum[0] = mac16_16(sum[0], tmp, y_1);
        sum[1] = mac16_16(sum[1], tmp, y_2);
        sum[2] = mac16_16(sum[2], tmp, y_3);
        sum[3] = mac16_16(sum[3], tmp, y_0);
    }
    if j < len {
        let tmp = x[j];
        y_1 = y[yi];
        sum[0] = mac16_16(sum[0], tmp, y_2);
        sum[1] = mac16_16(sum[1], tmp, y_3);
        sum[2] = mac16_16(sum[2], tmp, y_0);
        sum[3] = mac16_16(sum[3], tmp, y_1);
    }
    let _ = (y_0, y_1, y_2, y_3); /* suppress warnings */
}

/// MAC16_16: Multiply-accumulate for 16-bit values
/// Matches C: #define MAC16_16(a,b,c) ((a)+(b)*(c))
#[inline(always)]
fn mac16_16(a: i32, b: i16, c: i16) -> i32 {
    a.wrapping_add((b as i32).wrapping_mul(c as i32))
}

pub fn silk_autocorr(
    results: &mut [i32],
    scale: &mut i32,
    input_data: &[i16],
    input_data_size: usize,
    correlation_count: usize,
) {
    #[inline]
    fn ec_ilog(x: u32) -> i32 {
        if x == 0 {
            0
        } else {
            32 - x.leading_zeros() as i32
        }
    }

    let n = input_data_size;
    let mut shift: i32;

    /* Replicate _celt_autocorr FIXED_POINT energy estimation */
    let xptr = input_data;

    /* ac0_shift = celt_ilog2(n + (n>>4)) = EC_ILOG(x) - 1 */
    let ac0_shift = ec_ilog((n + (n >> 4)) as u32) - 1;

    let mut ac0: i32 = 1 + ((n as i32) << 7);
    let mut i = n & 1;
    if n & 1 != 0 {
        ac0 += ((xptr[0] as i16 as i32) * (xptr[0] as i16 as i32)) >> ac0_shift;
    }
    while i < n {
        ac0 += ((xptr[i] as i16 as i32) * (xptr[i] as i16 as i32)) >> ac0_shift;
        ac0 += ((xptr[i + 1] as i16 as i32) * (xptr[i + 1] as i16 as i32)) >> ac0_shift;
        i += 2;
    }
    /* Consider the effect of rounding-to-nearest when scaling the signal. */
    ac0 += ac0 >> 7;

    let ac0_log2 = ec_ilog(ac0 as u32) - 1; /* celt_ilog2(ac0) = EC_ILOG - 1 */
    shift = ac0_log2 - 30 + ac0_shift + 1;
    shift = shift / 2;

    // Stack buffer: max n = pitch_lpc_win_length ≤ (20+4)*16 = 384; shape_win ≤ 240.
    // Use PE_MAX_FRAME_LENGTH (640) to also cover direct benchmark calls with n=640.
    let mut xx_buf = [0i16; PE_MAX_FRAME_LENGTH];
    let xptr: &[i16];

    if shift > 0 {
        /* PSHR32: rounding shift */
        for j in 0..n {
            xx_buf[j] = silk_rshift_round(input_data[j] as i32, shift) as i16;
        }
        xptr = &xx_buf[..n];
    } else {
        shift = 0;
        xptr = input_data;
    }

    /* Compute autocorrelation matching C's _celt_autocorr exactly.
     * C uses celt_pitch_xcorr_c: lags 0..max_pitch-4 via xcorr_kernel_c (4-at-a-time),
     * remaining lags via celt_inner_prod (sequential), then tail. */
    let lag = correlation_count - 1;
    let fast_n = n - lag;
    let max_pitch = lag + 1; /* = correlation_count */

    /* First, process lags 0..max_pitch-4 via xcorr_kernel_c (4 at a time) */
    let mut lag_idx = 0;
    while lag_idx + 3 < max_pitch {
        let mut sum = [0i32; 4];
        xcorr_kernel_c(xptr, &xptr[lag_idx..], &mut sum, fast_n);
        results[lag_idx] = sum[0];
        results[lag_idx + 1] = sum[1];
        results[lag_idx + 2] = sum[2];
        results[lag_idx + 3] = sum[3];
        lag_idx += 4;
    }
    /* Remaining lags via celt_inner_prod (sequential MAC16_16) */
    while lag_idx < max_pitch {
        let mut sum = 0i32;
        for j in 0..fast_n {
            sum = sum.wrapping_add((xptr[j] as i32).wrapping_mul(xptr[j + lag_idx] as i32));
        }
        results[lag_idx] = sum;
        lag_idx += 1;
    }
    if input_data_size == 88 && correlation_count == 13 {}

    /* Add tail: for each lag k, sum samples k+fastN..n */
    for k in 0..correlation_count {
        let mut d: i32 = 0;
        for i in (k + fast_n)..n {
            d = d.wrapping_add((xptr[i] as i32).wrapping_mul(xptr[i - k] as i32));
        }
        results[k] = results[k].wrapping_add(d);
    }

    if input_data_size == 88 && correlation_count == 13 {}

    /* Post-computation normalization, matching C FIXED_POINT */
    shift = 2 * shift;
    if shift <= 0 {
        let add_shift = (-shift).min(30);
        results[0] += 1i32 << add_shift;
    }
    if results[0] > 0 && results[0] < 268435456 {
        /* ac[0] < 2^28: upshift */
        let shift2 = 29 - ec_ilog(results[0] as u32);
        for j in 0..correlation_count {
            results[j] <<= shift2;
        }
        shift -= shift2;
    } else if results[0] >= 536870912 {
        /* ac[0] >= 2^29: downshift */
        let mut shift2 = 1;
        if results[0] >= 1073741824 {
            shift2 += 1;
        }
        for j in 0..correlation_count {
            results[j] >>= shift2;
        }
        shift += shift2;
    }

    *scale = shift;
}

pub fn silk_sum_sqr_shift(energy: &mut i32, shift: &mut i32, x: &[i16], len: usize) {
    let mut i: usize;
    let mut shft: i32;
    let mut nrg_tmp: u32;
    let mut nrg: i32;

    /* Do a first run with the maximum shift we could have. */
    shft = 31 - silk_clz32(len as i32);
    /* Let's be conservative with rounding and start with nrg=len. */
    nrg = len as i32;
    i = 0;
    while i < len - 1 {
        nrg_tmp = silk_smulbb(x[i] as i32, x[i] as i32) as u32;
        nrg_tmp = nrg_tmp.wrapping_add(silk_smulbb(x[i + 1] as i32, x[i + 1] as i32) as u32);
        nrg = nrg.wrapping_add((nrg_tmp >> shft) as i32);
        i += 2;
    }
    if i < len {
        nrg_tmp = silk_smulbb(x[i] as i32, x[i] as i32) as u32;
        nrg = nrg.wrapping_add((nrg_tmp >> shft) as i32);
    }

    /* Make sure the result will fit in a 32-bit signed integer with two bits of headroom. */
    shft = (shft + 3 - silk_clz32(nrg)).max(0);
    nrg = 0;
    i = 0;
    while i < len - 1 {
        nrg_tmp = silk_smulbb(x[i] as i32, x[i] as i32) as u32;
        nrg_tmp = nrg_tmp.wrapping_add(silk_smulbb(x[i + 1] as i32, x[i + 1] as i32) as u32);
        nrg = nrg.wrapping_add((nrg_tmp >> shft) as i32);
        i += 2;
    }
    if i < len {
        nrg_tmp = silk_smulbb(x[i] as i32, x[i] as i32) as u32;
        nrg = nrg.wrapping_add((nrg_tmp >> shft) as i32);
    }

    *shift = shft;
    *energy = nrg;
}

#[inline(always)]
pub fn silk_inner_prod_aligned(ptr1: &[i16], ptr2: &[i16], len: usize) -> i32 {
    let ptr1 = &ptr1[..len];
    let ptr2 = &ptr2[..len];
    let mut i = 0;
    let mut sum0 = 0i32;
    let mut sum1 = 0i32;
    let len4 = (len / 4) * 4;

    // 2-way loop unrolling
    while i < len4 {
        sum0 = sum0.wrapping_add((ptr1[i] as i32).wrapping_mul(ptr2[i] as i32));
        sum1 = sum1.wrapping_add((ptr1[i + 1] as i32).wrapping_mul(ptr2[i + 1] as i32));
        i += 2;
    }

    while i < len {
        sum0 = sum0.wrapping_add((ptr1[i] as i32).wrapping_mul(ptr2[i] as i32));
        i += 1;
    }

    sum0.wrapping_add(sum1)
}

pub fn silk_corr_vector_fix(
    x: &[i16], /* I    x vector [L + order - 1] used to form data matrix X                         */
    t: &[i16], /* I    Target vector [L]                                                           */
    l: usize, /* I    Length of vectors                                                           */
    order: usize, /* I    Max lag for correlation                                                     */
    xt: &mut [i32], /* O    Pointer to X'*t correlation vector [order]                                  */
    rshifts: i32, /* I    Right shifts of correlations                                                */
) {
    let mut ptr1_idx = order - 1;
    if rshifts > 0 {
        for lag in 0..order {
            let mut inner_prod: i32 = 0;
            for i in 0..l {
                inner_prod = silk_add_rshift32(
                    inner_prod,
                    silk_smulbb(x[ptr1_idx + i] as i32, t[i] as i32),
                    rshifts,
                );
            }
            xt[lag] = inner_prod;
            if ptr1_idx > 0 {
                ptr1_idx -= 1;
            }
        }
    } else {
        for lag in 0..order {
            xt[lag] = silk_inner_prod_aligned(&x[ptr1_idx..], t, l);
            if ptr1_idx > 0 {
                ptr1_idx -= 1;
            }
        }
    }
}

pub fn silk_corr_matrix_fix(
    x: &[i16], /* I    x vector [L + order - 1] used to form data matrix X                         */
    l: usize, /* I    Length of vectors                                                           */
    order: usize, /* I    Max lag for correlation                                                     */
    xx: &mut [i32], /* O    Pointer to X'*X correlation matrix [ order x order ]                        */
    nrg: &mut i32, /* O    Energy of x vector                                                            */
    rshifts: &mut i32, /* O    Right shifts of correlations and energy                                     */
) {
    /* Calculate energy to find shift used to fit in 32 bits */
    silk_sum_sqr_shift(nrg, rshifts, x, l + order - 1);
    let mut energy = *nrg;

    /* Calculate energy of first column (0) of X: X[:,0]'*X[:,0] */
    /* Remove contribution of first order - 1 samples */
    for i in 0..(order - 1) {
        energy -= silk_rshift32(silk_smulbb(x[i] as i32, x[i] as i32), *rshifts);
    }

    /* Calculate energy of remaining columns of X: X[:,j]'*X[:,j] */
    /* Fill out the diagonal of the correlation matrix */
    xx[0 * order + 0] = energy;
    let ptr1_start_idx = order - 1;
    for j in 1..order {
        energy = silk_sub32(
            energy,
            silk_rshift32(
                silk_smulbb(
                    x[ptr1_start_idx + l - j] as i32,
                    x[ptr1_start_idx + l - j] as i32,
                ),
                *rshifts,
            ),
        );
        energy = silk_add32(
            energy,
            silk_rshift32(
                silk_smulbb(x[ptr1_start_idx - j] as i32, x[ptr1_start_idx - j] as i32),
                *rshifts,
            ),
        );
        xx[j * order + j] = energy;
    }

    /* Fill out the off-diagonal elements */
    for lag in 1..order {
        /* Calculate row 0 and column 0 */
        let ptr1_idx = ptr1_start_idx;
        let ptr2_idx = ptr1_start_idx - lag;
        let mut inner_prod: i32 = 0;
        if *rshifts > 0 {
            for i in 0..l {
                inner_prod = silk_add_rshift32(
                    inner_prod,
                    silk_smulbb(x[ptr1_idx + i] as i32, x[ptr2_idx + i] as i32),
                    *rshifts,
                );
            }
        } else {
            inner_prod = silk_inner_prod_aligned(&x[ptr1_idx..], &x[ptr2_idx..], l);
        }
        xx[0 * order + lag] = inner_prod;
        xx[lag * order + 0] = inner_prod;

        /* Use property that matrix is almost a Toelpitz matrix to fill out the rest of the elements */
        for j in 1..(order - lag) {
            inner_prod = silk_sub32(
                inner_prod,
                silk_rshift32(
                    silk_smulbb(x[ptr1_idx + l - j] as i32, x[ptr2_idx + l - j] as i32),
                    *rshifts,
                ),
            );
            inner_prod = silk_add32(
                inner_prod,
                silk_rshift32(
                    silk_smulbb(x[ptr1_idx - j] as i32, x[ptr2_idx - j] as i32),
                    *rshifts,
                ),
            );
            xx[j * order + (lag + j)] = inner_prod;
            xx[(lag + j) * order + j] = inner_prod;
        }
    }
}

const FREQ_TABLE_Q16: [i16; 27] = [
    12111, 9804, 8235, 7100, 6239, 5565, 5022, 4575, 4202, 3885, 3612, 3375, 3167, 2984, 2820,
    2674, 2542, 2422, 2313, 2214, 2123, 2038, 1961, 1889, 1822, 1760, 1702,
];

pub fn silk_apply_sine_window(px_win: &mut [i16], px: &[i16], win_type: i32, length: usize) {
    let f_q16: i32;
    let c_q16: i32;
    let mut s0_q16: i32;
    let mut s1_q16: i32;

    /* Length must be in a range from 16 to 120 and a multiple of 4 */
    let idx = (length >> 2) - 4;
    f_q16 = FREQ_TABLE_Q16[idx] as i32;

    /* Factor used for cosine approximation */
    c_q16 = silk_smulwb(f_q16, -f_q16 as i32);

    /* initialize state */
    if win_type == 1 {
        s0_q16 = 0;
        s1_q16 = f_q16 + (length as i32 >> 3);
    } else {
        s0_q16 = 1 << 16;
        s1_q16 = (1 << 16) + (c_q16 >> 1) + (length as i32 >> 4);
    }

    for k in (0..length).step_by(4) {
        px_win[k] = silk_smulwb((s0_q16 + s1_q16) >> 1, px[k] as i32) as i16;
        px_win[k + 1] = silk_smulwb(s1_q16, px[k + 1] as i32) as i16;
        s0_q16 = silk_smulwb(s1_q16, c_q16 as i32) + (s1_q16 << 1) - s0_q16 + 1;
        s0_q16 = s0_q16.min(1 << 16);

        px_win[k + 2] = silk_smulwb((s0_q16 + s1_q16) >> 1, px[k + 2] as i32) as i16;
        px_win[k + 3] = silk_smulwb(s0_q16, px[k + 3] as i32) as i16;
        s1_q16 = silk_smulwb(s0_q16, c_q16 as i32) + (s0_q16 << 1) - s1_q16;
        s1_q16 = s1_q16.min(1 << 16);
    }
}

#[inline(always)]
pub fn silk_pitch_xcorr(x: &[i16], y: &[i16], xcorr: &mut [i32], len: usize, max_pitch: usize) {
    // Ensure y has enough length
    let y_len = y.len();
    let effective_len = len.min(y_len.saturating_sub(max_pitch));

    if effective_len < len {
        for i in 0..max_pitch {
            let mut sum: i32 = 0;
            for j in 0..effective_len {
                sum = silk_smlabb(sum, x[j] as i32, y[i + j] as i32);
            }
            xcorr[i] = sum;
        }
        return;
    }

    // Main path with 2-way loop unrolling
    for i in 0..max_pitch {
        let mut sum0: i32 = 0;
        let mut sum1: i32 = 0;
        let y_offset = i;
        let mut j = 0;
        let len4 = (len / 4) * 4;

        while j < len4 {
            sum0 = silk_smlabb(sum0, x[j] as i32, y[y_offset + j] as i32);
            sum1 = silk_smlabb(sum1, x[j + 1] as i32, y[y_offset + j + 1] as i32);
            j += 2;
        }

        while j < len {
            sum0 = silk_smlabb(sum0, x[j] as i32, y[y_offset + j] as i32);
            j += 1;
        }

        xcorr[i] = sum0.wrapping_add(sum1);
    }
}

pub fn silk_warped_autocorrelation_fix(
    corr: &mut [i32], // O    Result [order + 1]
    scale: &mut i32,  // O    Scaling of the correlation vector
    input: &[i16],    // I    Input data to correlate
    warping_q16: i32, // I    Warping coefficient
    length: usize,    // I    Length of input
    order: usize,     // I    Correlation order (even)
) {
    // Constants matching C's main_FIX.h
    const QC: i32 = 10;
    const QS: i32 = 13;

    let mut tmp1_qs: i32;
    let mut tmp2_qs: i32;
    let mut state_qs = [0i32; MAX_SHAPE_LPC_ORDER + 1];
    let mut corr_qc = [0i64; MAX_SHAPE_LPC_ORDER + 1];

    /* Order must be even */
    debug_assert!((order & 1) == 0);

    /* Loop over samples */
    for n in 0..length {
        tmp1_qs = (input[n] as i32) << QS;
        /* Loop over allpass sections */
        let mut i = 0;
        while i < order {
            /* Output of allpass section - use SMULWW to match ARM NEON vqdmulhq_s32 */
            tmp2_qs = silk_smlaww(state_qs[i], state_qs[i + 1] - tmp1_qs, warping_q16);
            state_qs[i] = tmp1_qs;
            corr_qc[i] += silk_rshift64(silk_smull(tmp1_qs, state_qs[0]), 2 * QS - QC);
            /* Output of allpass section */
            tmp1_qs = silk_smlaww(state_qs[i + 1], state_qs[i + 2] - tmp2_qs, warping_q16);
            state_qs[i + 1] = tmp2_qs;
            corr_qc[i + 1] += silk_rshift64(silk_smull(tmp2_qs, state_qs[0]), 2 * QS - QC);
            i += 2;
        }
        state_qs[order] = tmp1_qs;
        corr_qc[order] += silk_rshift64(silk_smull(tmp1_qs, state_qs[0]), 2 * QS - QC);
    }

    let mut lsh = silk_clz64(corr_qc[0]) - 35;
    lsh = silk_limit_32(lsh, -12 - QC, 30 - QC);
    *scale = -(QC + lsh);
    if lsh >= 0 {
        for i in 0..=order {
            corr[i] = (corr_qc[i] << lsh) as i32;
        }
    } else {
        for i in 0..=order {
            corr[i] = (corr_qc[i] >> (-lsh)) as i32;
        }
    }
}

pub fn silk_schur(
    rc_q15: &mut [i16], // O reflection coefficients [order] Q15
    c: &[i32],          // I correlations [order+1]
    order: usize,       // I prediction order
) -> i32 {
    let mut c_inner = [[0i32; 2]; MAX_LPC_ORDER + 1];
    let mut ctmp1: i32;
    let mut ctmp2: i32;
    let mut rc_tmp_q15: i32;

    assert!(order <= MAX_LPC_ORDER);

    /* Get number of leading zeros */
    let lz = c[0].leading_zeros() as i32;

    /* Copy correlations and adjust level to Q30 */
    if lz < 2 {
        /* lz must be 1, so shift one to the right */
        for i in 0..=order {
            c_inner[i][0] = c[i] >> 1;
            c_inner[i][1] = c[i] >> 1;
        }
    } else if lz > 2 {
        /* Shift to the left */
        let lz_adj = lz - 2;
        for i in 0..=order {
            c_inner[i][0] = c[i] << lz_adj;
            c_inner[i][1] = c[i] << lz_adj;
        }
    } else {
        /* No need to shift */
        for i in 0..=order {
            c_inner[i][0] = c[i];
            c_inner[i][1] = c[i];
        }
    }

    for k in 0..order {
        /* Check that we won't be getting an unstable rc, otherwise stop here. */
        if c_inner[k + 1][0].abs() >= c_inner[0][1] {
            if c_inner[k + 1][0] > 0 {
                rc_q15[k] = -32440; // -0.99 in Q15
            } else {
                rc_q15[k] = 32440; // 0.99 in Q15
            }
            return c_inner[0][1];
        }

        /* Get reflection coefficient */
        rc_tmp_q15 = -silk_div32_16(c_inner[k + 1][0], (c_inner[0][1] >> 15).max(1));

        /* Clip */
        rc_tmp_q15 = silk_sat16(rc_tmp_q15);

        /* Store */
        rc_q15[k] = rc_tmp_q15 as i16;

        /* Update correlations */
        for n in 0..order - k {
            ctmp1 = c_inner[n + k + 1][0];
            ctmp2 = c_inner[n][1];
            c_inner[n + k + 1][0] = silk_smlawb(ctmp1, ctmp2 << 1, rc_tmp_q15 as i32);
            c_inner[n][1] = silk_smlawb(ctmp2, ctmp1 << 1, rc_tmp_q15 as i32);
        }
    }

    c_inner[0][1]
}

pub fn silk_k2a(
    a_q24: &mut [i32], // O Prediction coefficients [order] Q24
    rc_q15: &[i16],    // I Reflection coefficients [order] Q15
    order: usize,      // I Prediction order
) {
    for k in 0..order {
        let rc = rc_q15[k] as i32;
        for n in 0..(k + 1) >> 1 {
            let tmp1 = a_q24[n];
            let tmp2 = a_q24[k - n - 1];
            a_q24[n] = silk_smlawb(tmp1, tmp2 << 1, rc as i32);
            a_q24[k - n - 1] = silk_smlawb(tmp2, tmp1 << 1, rc as i32);
        }
        a_q24[k] = -(rc << 9);
    }
}

pub fn silk_bwexpander(
    ar: &mut [i16],     // I/O AR filter to be expanded (Q12)
    d: usize,           // I Order
    mut chirp_q16: i32, // I Chirp factor (Q16)
) {
    let chirp_minus_one_q16 = chirp_q16 - 65536;

    /* x_exp[i] = x[i] * chirp^(i+1) */
    for i in 0..d - 1 {
        ar[i] = silk_smulww(ar[i] as i32, chirp_q16) as i16;
        chirp_q16 += silk_rshift_round(silk_smulww(chirp_q16, chirp_minus_one_q16), 16);
    }
    ar[d - 1] = silk_smulww(ar[d - 1] as i32, chirp_q16) as i16;
}

pub fn silk_lpc_analysis_filter(
    out: &mut [i16], // O Output signal
    input: &[i16],   // I Input signal
    b: &[i16],       // I MA prediction coefficients, Q12 [order]
    len: usize,      // I Signal length
    d: usize,        // I Filter order
    _arch: i32,      // I Run-time architecture
) {
    assert!(d >= 6);
    assert!((d & 1) == 0);
    assert!(d <= len);

    for ix in 0..d {
        out[ix] = 0;
    }

    // SAFETY: ix ranges d..len, and we access input[ix-1..ix-d-1], all within input[0..len+d-1].
    // b has exactly d elements. out has at least len elements.
    unsafe {
        for ix in d..len {
            let mut out32_q12: i32;
            out32_q12 = silk_smulbb(
                *input.get_unchecked(ix - 1) as i32,
                *b.get_unchecked(0) as i32,
            );
            out32_q12 = out32_q12.wrapping_add(silk_smulbb(
                *input.get_unchecked(ix - 2) as i32,
                *b.get_unchecked(1) as i32,
            ));
            out32_q12 = out32_q12.wrapping_add(silk_smulbb(
                *input.get_unchecked(ix - 3) as i32,
                *b.get_unchecked(2) as i32,
            ));
            out32_q12 = out32_q12.wrapping_add(silk_smulbb(
                *input.get_unchecked(ix - 4) as i32,
                *b.get_unchecked(3) as i32,
            ));
            out32_q12 = out32_q12.wrapping_add(silk_smulbb(
                *input.get_unchecked(ix - 5) as i32,
                *b.get_unchecked(4) as i32,
            ));
            out32_q12 = out32_q12.wrapping_add(silk_smulbb(
                *input.get_unchecked(ix - 6) as i32,
                *b.get_unchecked(5) as i32,
            ));
            let mut j = 6;
            while j < d {
                out32_q12 = out32_q12.wrapping_add(silk_smulbb(
                    *input.get_unchecked(ix - j - 1) as i32,
                    *b.get_unchecked(j) as i32,
                ));
                out32_q12 = out32_q12.wrapping_add(silk_smulbb(
                    *input.get_unchecked(ix - j - 2) as i32,
                    *b.get_unchecked(j + 1) as i32,
                ));
                j += 2;
            }

            /* Subtract prediction */
            out32_q12 = ((*input.get_unchecked(ix) as i32) << 12).wrapping_sub(out32_q12);

            /* Scale to Q0 */
            let out32 = silk_rshift_round(out32_q12, 12);

            /* Saturate output */
            *out.get_unchecked_mut(ix) = silk_sat16(out32) as i16;
        }
    }
}

pub fn silk_scale_copy_vector16(
    data_out: &mut [i16],
    data_in: &[i16],
    gain_q16: i32,
    data_size: usize,
) {
    for i in 0..data_size {
        let tmp32 = silk_smulwb(gain_q16, data_in[i] as i32);
        data_out[i] = silk_sat16(tmp32) as i16;
    }
}

/// PSHR32: Rounding right shift (matches C's PSHR32 macro)
/// PSHR32(a,shift) = SHR32((a)+((EXTEND32(1)<<((shift))>>1)),shift)
#[inline(always)]
pub fn silk_pshr32(a: i32, shift: i32) -> i32 {
    if shift <= 0 {
        return a;
    }
    let round = 1i32 << (shift - 1);
    (a + round) >> shift
}

/// SHR32: Arithmetic right shift (matches C's SHR32 macro)
#[inline(always)]
pub fn silk_shr32(a: i32, shift: i32) -> i32 {
    if shift <= 0 {
        return a;
    }
    a >> shift
}
