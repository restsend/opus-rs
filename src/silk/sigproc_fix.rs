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

    if c[0] <= 0 {
        for v in rc_q16.iter_mut().take(order) {
            *v = 0;
        }
        return 0;
    }

    for k in 0..=order {
        c_matrix[k][0] = c[k];
        c_matrix[k][1] = c[k];
    }

    let mut k = 0;
    while k < order {
        if c_matrix[k + 1][0].wrapping_abs() >= c_matrix[0][1] {
            if c_matrix[k + 1][0] > 0 {
                rc_q16[k] = -64881;
            } else {
                rc_q16[k] = 64881;
            }
            k += 1;
            break;
        }

        rc_tmp_q31 = silk_div32_varq(-c_matrix[k + 1][0], c_matrix[0][1], 31);

        rc_q16[k] = silk_rshift_round(rc_tmp_q31, 15);

        for n in 0..(order - k) {
            c_tmp1_q30 = c_matrix[n + k + 1][0];
            c_tmp2_q30 = c_matrix[n][1];

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
    let a0_l_q28 = (-a_q28[0]) & 0x00003FFF;
    let a0_u_q28 = -a_q28[0] >> 14;
    let a1_l_q28 = (-a_q28[1]) & 0x00003FFF;
    let a1_u_q28 = -a_q28[1] >> 14;

    for item in input_output.iter_mut().take(len) {
        let inval = *item as i32;
        let out32_q14 = silk_smlawb(s[0], b_q28[0], inval) << 2;

        s[0] = s[1] + silk_rshift_round(silk_smulwb(out32_q14, a0_l_q28), 14);
        s[0] = silk_smlawb(s[0], out32_q14, a0_u_q28);
        s[0] = silk_smlawb(s[0], b_q28[1], inval);

        s[1] = silk_rshift_round(silk_smulwb(out32_q14, a1_l_q28), 14);
        s[1] = silk_smlawb(s[1], out32_q14, a1_u_q28);
        s[1] = silk_smlawb(s[1], b_q28[2], inval);

        *item = silk_sat16(silk_rshift(out32_q14 + (1 << 14) - 1, 14)) as i16;
    }
}

pub fn silk_biquad_alt_stride2(
    input_output: &mut [i16],
    b_q28: &[i32],
    a_q28: &[i32],
    s: &mut [i32],
    len: usize,
) {
    let a0_l_q28 = (-a_q28[0]) & 0x00003FFF;
    let a0_u_q28 = -a_q28[0] >> 14;
    let a1_l_q28 = (-a_q28[1]) & 0x00003FFF;
    let a1_u_q28 = -a_q28[1] >> 14;

    for k in 0..len {
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

        input_output[2 * k] = silk_sat16(silk_rshift(out32_q14_0 + (1 << 14) - 1, 14)) as i16;
        input_output[2 * k + 1] = silk_sat16(silk_rshift(out32_q14_1 + (1 << 14) - 1, 14)) as i16;
    }
}

/// `sum[k] += x[j] * y[j + k]` over `j < len` for the four lags `k < 4`
/// (libopus `xcorr_kernel_c`; the integer sums wrap, so their order is free).
#[inline]
fn xcorr_kernel_c(x: &[i16], y: &[i16], sum: &mut [i32; 4], len: usize) {
    let x = &x[..len];
    let y = &y[..len + 3];
    for (k, s) in sum.iter_mut().enumerate() {
        *s = s.wrapping_add(inner_prod_i16(x, &y[k..k + len]));
    }
}

/// Wrapping sum of `a[i] * b[i]` over the shorter length, written so that the
/// compiler vectorizes it (`pmaddwd` on x86, `smlal` on aarch64).
#[inline(always)]
fn inner_prod_i16(a: &[i16], b: &[i16]) -> i32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0i32; 16];
    let (a16, b16) = (a.chunks_exact(16), b.chunks_exact(16));
    let mut tail = 0i32;
    for (&x, &y) in a16.remainder().iter().zip(b16.remainder()) {
        tail = tail.wrapping_add(x as i32 * y as i32);
    }
    for (x, y) in a16.zip(b16) {
        for l in 0..16 {
            acc[l] = acc[l].wrapping_add(x[l] as i32 * y[l] as i32);
        }
    }
    acc.iter().fold(tail, |t, &s| t.wrapping_add(s))
}

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

    let xptr = input_data;

    let ac0_shift = ec_ilog((n + (n >> 4)) as u32) - 1;

    let mut ac0: i32 = 1 + ((n as i32) << 7);
    let mut i = n & 1;
    if n & 1 != 0 {
        ac0 += ((xptr[0] as i32) * (xptr[0] as i32)) >> ac0_shift;
    }
    while i < n {
        ac0 += ((xptr[i] as i32) * (xptr[i] as i32)) >> ac0_shift;
        ac0 += ((xptr[i + 1] as i32) * (xptr[i + 1] as i32)) >> ac0_shift;
        i += 2;
    }

    ac0 += ac0 >> 7;

    let ac0_log2 = ec_ilog(ac0 as u32) - 1;
    shift = ac0_log2 - 30 + ac0_shift + 1;
    shift /= 2;

    let mut xx_buf = [0i16; PE_MAX_FRAME_LENGTH];
    let xptr: &[i16];

    if shift > 0 {
        for j in 0..n {
            xx_buf[j] = silk_rshift_round(input_data[j] as i32, shift) as i16;
        }
        xptr = &xx_buf[..n];
    } else {
        shift = 0;
        xptr = input_data;
    }

    let lag = correlation_count - 1;
    let fast_n = n - lag;
    let max_pitch = lag + 1;

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

    while lag_idx < max_pitch {
        let mut sum = 0i32;
        for j in 0..fast_n {
            sum = sum.wrapping_add((xptr[j] as i32).wrapping_mul(xptr[j + lag_idx] as i32));
        }
        results[lag_idx] = sum;
        lag_idx += 1;
    }

    for k in 0..correlation_count {
        let mut d: i32 = 0;
        for i in (k + fast_n)..n {
            d = d.wrapping_add((xptr[i] as i32).wrapping_mul(xptr[i - k] as i32));
        }
        results[k] = results[k].wrapping_add(d);
    }

    shift *= 2;
    if shift <= 0 {
        let add_shift = (-shift).min(30);
        results[0] += 1i32 << add_shift;
    }
    if results[0] > 0 && results[0] < 268435456 {
        let shift2 = 29 - ec_ilog(results[0] as u32);
        for v in results[..correlation_count].iter_mut() {
            *v <<= shift2;
        }
        shift -= shift2;
    } else if results[0] >= 536870912 {
        let mut shift2 = 1;
        if results[0] >= 1073741824 {
            shift2 += 1;
        }
        for v in results[..correlation_count].iter_mut() {
            *v >>= shift2;
        }
        shift += shift2;
    }

    *scale = shift;
}

pub fn silk_sum_sqr_shift(energy: &mut i32, shift: &mut i32, x: &[i16], len: usize) {
    // sum_sqr_shift.c: a first pass finds the shift, a second sums with it.
    let mut shft = 31 - silk_clz32(len as i32);
    let nrg = sum_sqr_pairs_shifted(x, len, len as i32, shft);
    shft = (shft + 3 - silk_clz32(nrg)).max(0);
    *energy = sum_sqr_pairs_shifted(x, len, 0, shft);
    *shift = shft;
}

/// One pass of libopus's `silk_sum_sqr_shift`: squares summed in pairs, each
/// pair's sum shifted right by `shft` (as unsigned) and added to `init`.
#[inline(always)]
fn sum_sqr_pairs_shifted(x: &[i16], len: usize, init: i32, shft: i32) -> i32 {
    let mut nrg = init;
    let mut i = 0;
    while i + 1 < len {
        let mut nrg_tmp = silk_smulbb(x[i] as i32, x[i] as i32) as u32;
        nrg_tmp = nrg_tmp.wrapping_add(silk_smulbb(x[i + 1] as i32, x[i + 1] as i32) as u32);
        nrg = nrg.wrapping_add((nrg_tmp >> shft) as i32);
        i += 2;
    }
    if i < len {
        let nrg_tmp = silk_smulbb(x[i] as i32, x[i] as i32) as u32;
        nrg = nrg.wrapping_add((nrg_tmp >> shft) as i32);
    }
    nrg
}

/// Inner product with each product shifted right by `scale` before it is
/// accumulated (libopus `silk_inner_prod_aligned_scale`).
pub fn silk_inner_prod_aligned_scale(
    in_vec1: &[i16],
    in_vec2: &[i16],
    scale: i32,
    len: usize,
) -> i32 {
    let mut sum: i32 = 0;
    for (&a, &b) in in_vec1[..len].iter().zip(&in_vec2[..len]) {
        sum = sum.wrapping_add(silk_smulbb(a as i32, b as i32) >> scale);
    }
    sum
}

#[inline(always)]
pub fn silk_inner_prod_aligned(ptr1: &[i16], ptr2: &[i16], len: usize) -> i32 {
    inner_prod_i16(&ptr1[..len], &ptr2[..len])
}

pub fn silk_corr_vector_fix(
    x: &[i16],
    t: &[i16],
    l: usize,
    order: usize,
    xt: &mut [i32],
    rshifts: i32,
) {
    let mut ptr1_idx = order - 1;
    if rshifts > 0 {
        for xt_val in xt[..order].iter_mut() {
            let mut inner_prod: i32 = 0;
            for i in 0..l {
                inner_prod = silk_add_rshift32(
                    inner_prod,
                    silk_smulbb(x[ptr1_idx + i] as i32, t[i] as i32),
                    rshifts,
                );
            }
            *xt_val = inner_prod;
            ptr1_idx = ptr1_idx.saturating_sub(1);
        }
    } else {
        for xt_val in xt[..order].iter_mut() {
            *xt_val = silk_inner_prod_aligned(&x[ptr1_idx..], t, l);
            ptr1_idx = ptr1_idx.saturating_sub(1);
        }
    }
}

pub fn silk_corr_matrix_fix(
    x: &[i16],
    l: usize,
    order: usize,
    xx: &mut [i32],
    nrg: &mut i32,
    rshifts: &mut i32,
) {
    silk_sum_sqr_shift(nrg, rshifts, x, l + order - 1);
    let mut energy = *nrg;

    for xi in x.iter().take(order - 1) {
        energy -= silk_rshift32(silk_smulbb(*xi as i32, *xi as i32), *rshifts);
    }

    xx[0] = energy;
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

    for lag in 1..order {
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
        xx[lag] = inner_prod;
        xx[lag * order] = inner_prod;

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
    let mut s0_q16: i32;
    let mut s1_q16: i32;

    let idx = (length >> 2) - 4;
    let f_q16: i32 = FREQ_TABLE_Q16[idx] as i32;

    let c_q16: i32 = silk_smulwb(f_q16, -f_q16);

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
        s0_q16 = silk_smulwb(s1_q16, c_q16) + (s1_q16 << 1) - s0_q16 + 1;
        s0_q16 = s0_q16.min(1 << 16);

        px_win[k + 2] = silk_smulwb((s0_q16 + s1_q16) >> 1, px[k + 2] as i32) as i16;
        px_win[k + 3] = silk_smulwb(s0_q16, px[k + 3] as i32) as i16;
        s1_q16 = silk_smulwb(s0_q16, c_q16) + (s0_q16 << 1) - s1_q16;
        s1_q16 = s1_q16.min(1 << 16);
    }
}

#[inline(always)]
pub fn silk_pitch_xcorr(x: &[i16], y: &[i16], xcorr: &mut [i32], len: usize, max_pitch: usize) {
    debug_assert!(max_pitch > 0);
    debug_assert!(x.len() >= len);
    debug_assert!(xcorr.len() >= max_pitch);

    let y_len = y.len();

    let mut i = 0;
    while i + 3 < max_pitch {
        if y_len >= i + len + 3 {
            let mut sum = [0i32; 4];
            xcorr_kernel_c(x, &y[i..], &mut sum, len);
            xcorr[i] = sum[0];
            xcorr[i + 1] = sum[1];
            xcorr[i + 2] = sum[2];
            xcorr[i + 3] = sum[3];
            i += 4;
        } else {
            for k in i..i.saturating_add(4).min(max_pitch) {
                let avail = len.min(y_len.saturating_sub(k));
                let mut sum = 0i32;
                for j in 0..avail {
                    sum = mac16_16(sum, x[j], y[k + j]);
                }
                xcorr[k] = sum;
            }
            i = i.saturating_add(4).min(max_pitch);
        }
    }

    while i < max_pitch {
        let avail = len.min(y_len.saturating_sub(i));
        let mut sum = 0i32;
        for j in 0..avail {
            sum = mac16_16(sum, x[j], y[i + j]);
        }
        xcorr[i] = sum;
        i += 1;
    }
}

pub fn silk_warped_autocorrelation_fix(
    corr: &mut [i32],
    scale: &mut i32,
    input: &[i16],
    warping_q16: i32,
    length: usize,
    order: usize,
) {
    const QC: i32 = 10;
    const QS: i32 = 13;

    let mut state_qs = [0i32; MAX_SHAPE_LPC_ORDER + 1];
    let mut corr_qc = [0i64; MAX_SHAPE_LPC_ORDER + 1];

    debug_assert!((order & 1) == 0);
    let state_qs = &mut state_qs[..=order];
    let corr_qc = &mut corr_qc[..=order];
    let input = &input[..length];

    // K samples per pass: sample m's allpass section i reads sample m - 1's
    // outputs of sections i - 1 and i, so the K chains run one section apart.
    const K: usize = 4;
    let mut blocks = input.chunks_exact(K);
    for block in &mut blocks {
        let x: [i32; K] = core::array::from_fn(|m| (block[m] as i32) << QS);
        let mut t = x;
        for i in 0..order {
            let mut next = [0i32; K];
            next[0] = silk_smlaww(state_qs[i], state_qs[i + 1] - t[0], warping_q16);
            for m in 1..K {
                next[m] = silk_smlaww(t[m - 1], next[m - 1] - t[m], warping_q16);
            }
            let mut c = 0i64;
            for m in 0..K {
                c += silk_rshift64(silk_smull(t[m], x[m]), 2 * QS - QC);
            }
            corr_qc[i] += c;
            state_qs[i] = t[K - 1];
            t = next;
        }
        state_qs[order] = t[K - 1];
        let mut c = 0i64;
        for m in 0..K {
            c += silk_rshift64(silk_smull(t[m], x[m]), 2 * QS - QC);
        }
        corr_qc[order] += c;
    }
    for &input_n in blocks.remainder() {
        let x_qs = (input_n as i32) << QS;
        let mut tmp1_qs = x_qs;
        for i in 0..order {
            let tmp2_qs = silk_smlaww(state_qs[i], state_qs[i + 1] - tmp1_qs, warping_q16);
            state_qs[i] = tmp1_qs;
            corr_qc[i] += silk_rshift64(silk_smull(tmp1_qs, x_qs), 2 * QS - QC);
            tmp1_qs = tmp2_qs;
        }
        state_qs[order] = tmp1_qs;
        corr_qc[order] += silk_rshift64(silk_smull(tmp1_qs, x_qs), 2 * QS - QC);
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

pub fn silk_schur(rc_q15: &mut [i16], c: &[i32], order: usize) -> i32 {
    let mut c_inner = [[0i32; 2]; MAX_LPC_ORDER + 1];
    let mut ctmp1: i32;
    let mut ctmp2: i32;
    let mut rc_tmp_q15: i32;

    assert!(order <= MAX_LPC_ORDER);

    let lz = c[0].leading_zeros() as i32;

    if lz < 2 {
        for i in 0..=order {
            c_inner[i][0] = c[i] >> 1;
            c_inner[i][1] = c[i] >> 1;
        }
    } else if lz > 2 {
        let lz_adj = lz - 2;
        for i in 0..=order {
            c_inner[i][0] = c[i] << lz_adj;
            c_inner[i][1] = c[i] << lz_adj;
        }
    } else {
        for i in 0..=order {
            c_inner[i][0] = c[i];
            c_inner[i][1] = c[i];
        }
    }

    for k in 0..order {
        if c_inner[k + 1][0].abs() >= c_inner[0][1] {
            if c_inner[k + 1][0] > 0 {
                rc_q15[k] = -32440;
            } else {
                rc_q15[k] = 32440;
            }
            return c_inner[0][1];
        }

        rc_tmp_q15 = -silk_div32_16(c_inner[k + 1][0], (c_inner[0][1] >> 15).max(1));

        rc_tmp_q15 = silk_sat16(rc_tmp_q15);

        rc_q15[k] = rc_tmp_q15 as i16;

        for n in 0..order - k {
            ctmp1 = c_inner[n + k + 1][0];
            ctmp2 = c_inner[n][1];
            c_inner[n + k + 1][0] = silk_smlawb(ctmp1, ctmp2 << 1, rc_tmp_q15);
            c_inner[n][1] = silk_smlawb(ctmp2, ctmp1 << 1, rc_tmp_q15);
        }
    }

    c_inner[0][1]
}

pub fn silk_k2a(a_q24: &mut [i32], rc_q15: &[i16], order: usize) {
    for k in 0..order {
        let rc = rc_q15[k] as i32;
        for n in 0..(k + 1) >> 1 {
            let tmp1 = a_q24[n];
            let tmp2 = a_q24[k - n - 1];
            a_q24[n] = silk_smlawb(tmp1, tmp2 << 1, rc);
            a_q24[k - n - 1] = silk_smlawb(tmp2, tmp1 << 1, rc);
        }
        a_q24[k] = -(rc << 9);
    }
}

pub fn silk_bwexpander(ar: &mut [i16], d: usize, mut chirp_q16: i32) {
    let chirp_minus_one_q16 = chirp_q16 - 65536;

    for ar_val in ar[..d - 1].iter_mut() {
        *ar_val = silk_rshift_round((*ar_val as i32).wrapping_mul(chirp_q16), 16) as i16;
        chirp_q16 += silk_rshift_round(chirp_q16.wrapping_mul(chirp_minus_one_q16), 16);
    }
    ar[d - 1] = silk_rshift_round((ar[d - 1] as i32).wrapping_mul(chirp_q16), 16) as i16;
}

pub fn silk_lpc_analysis_filter(
    out: &mut [i16],
    input: &[i16],
    b: &[i16],
    len: usize,
    d: usize,
    _arch: i32,
) {
    assert!(d >= 6);
    assert!((d & 1) == 0);
    assert!(d <= len);
    match d {
        10 => lpc_analysis_filter::<10>(out, input, b, len),
        16 => lpc_analysis_filter::<16>(out, input, b, len),
        _ => {
            let mut b_rev = [0i16; MAX_LPC_ORDER];
            let b_rev = &mut b_rev[..d];
            for (r, &c) in b_rev.iter_mut().rev().zip(&b[..d]) {
                *r = c;
            }
            lpc_analysis_filter_rev(out, input, b_rev, len);
        }
    }
}

#[inline(always)]
fn lpc_analysis_filter<const D: usize>(out: &mut [i16], input: &[i16], b: &[i16], len: usize) {
    let mut b_rev = [0i16; D];
    for (r, &c) in b_rev.iter_mut().rev().zip(&b[..D]) {
        *r = c;
    }
    lpc_analysis_filter_rev(out, input, &b_rev, len);
}

/// libopus `silk_LPC_analysis_filter` with the taps reversed (`b_rev[d - 1 - k]
/// = b[k]`), so each output is an inner product over the `d` inputs before it.
#[inline(always)]
fn lpc_analysis_filter_rev(out: &mut [i16], input: &[i16], b_rev: &[i16], len: usize) {
    let d = b_rev.len();
    let out = &mut out[..len];
    let input = &input[..len];
    out[..d].fill(0);
    for ix in d..len {
        let s = inner_prod_i16(&input[ix - d..ix], b_rev);
        let out32_q12 = ((input[ix] as i32) << 12).wrapping_sub(s);
        out[ix] = silk_sat16(silk_rshift_round(out32_q12, 12)) as i16;
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

#[inline(always)]
pub fn silk_pshr32(a: i32, shift: i32) -> i32 {
    if shift <= 0 {
        return a;
    }
    let round = 1i32 << (shift - 1);
    (a + round) >> shift
}

#[inline(always)]
pub fn silk_shr32(a: i32, shift: i32) -> i32 {
    if shift <= 0 {
        return a;
    }
    a >> shift
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn lpc_analysis_filter_ref(out: &mut [i16], input: &[i16], b: &[i16], len: usize, d: usize) {
        out[..d].fill(0);
        for ix in d..len {
            let mut s = 0i32;
            for k in 0..d {
                s = s.wrapping_add((input[ix - 1 - k] as i32).wrapping_mul(b[k] as i32));
            }
            let out32_q12 = ((input[ix] as i32) << 12).wrapping_sub(s);
            out[ix] = silk_sat16(silk_rshift_round(out32_q12, 12)) as i16;
        }
    }

    fn make_signal(len: usize, seed: u32) -> Vec<i16> {
        let mut s = seed;
        let mut out = vec![0i16; len];
        for v in &mut out {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (s >> 16) as i16;
        }
        out
    }

    #[test]
    fn test_silk_lpc_analysis_filter_matches_reference() {
        for d in (6usize..=16).step_by(2) {
            for &len in &[d + 8, d + 31, 160, 320] {
                let input = make_signal(len, (d as u32) * 17 + len as u32);
                let b = make_signal(d, 0x1234_5678 ^ d as u32);

                let mut out_opt = vec![0i16; len];
                let mut out_ref = vec![0i16; len];

                silk_lpc_analysis_filter(&mut out_opt, &input, &b, len, d, 0);
                lpc_analysis_filter_ref(&mut out_ref, &input, &b, len, d);

                assert_eq!(out_opt, out_ref, "mismatch for d={d}, len={len}");
            }
        }
    }
}
