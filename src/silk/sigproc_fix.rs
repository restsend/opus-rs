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
        if c_matrix[k + 1][0].abs() >= c_matrix[0][1] {
            if c_matrix[k + 1][0] > 0 {
                rc_q16[k] = -64880;
            } else {
                rc_q16[k] = 64880;
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
        silk_biquad_alt_stride2_neon(input_output, b_q28, a_q28, s, len);
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
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
}

#[inline]
fn xcorr_kernel_c(x: &[i16], y: &[i16], sum: &mut [i32; 4], len: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            xcorr_kernel_neon_s16(x, y, sum, len);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        unsafe { xcorr_kernel_avx2(x, y, sum, len) };
        return;
    }
    xcorr_kernel_scalar(x, y, sum, len);
}

/// Scalar C implementation of xcorr_kernel (4 lags simultaneously).
/// Kept separate so it is always compiled and can be used as reference.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
fn xcorr_kernel_scalar(x: &[i16], y: &[i16], sum: &mut [i32; 4], len: usize) {
    let mut j = 0;
    let mut y_0 = y[0];
    let mut y_1 = y[1];
    let mut y_2 = y[2];
    let mut y_3: i16 = 0;
    let mut yi = 3;
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
    let _ = (y_0, y_1, y_2, y_3);
}

/// NEON-accelerated xcorr_kernel for i16 SILK data.
///
/// Computes sum[k] = Σ_{i=0}^{len-1} x[i] * y[i+k] for k=0..3 simultaneously,
/// adding onto the existing values in `sum`.
///
/// Requires y.len() >= len + 3 (same contract as xcorr_kernel_c).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn xcorr_kernel_neon_s16(x: &[i16], y: &[i16], sum: &mut [i32; 4], mut len: usize) {
    use std::arch::aarch64::*;

    debug_assert!(x.len() >= len);
    debug_assert!(y.len() >= len + 3);

    // Each acc lane k accumulates: Σ x[i] * y[i+k] for the vectorised portion.
    // We use a single int32×4 accumulator where lane 0 = lag-0, lane 1 = lag-1, etc.
    let mut acc = vld1q_s32(sum.as_ptr()); // [sum[0], sum[1], sum[2], sum[3]]

    let mut xi = x.as_ptr();
    let mut yi = y.as_ptr();

    // Load initial 8 y-values (need y[0..len+3])
    let mut yy = vld1q_s16(yi); // [y[0]..y[7]]

    // Main loop: 4 x-samples per iteration.
    // Each iteration multiplies x[j+0..3] against the appropriate y-window
    // and accumulates into the single acc vector (lane k = lag k).
    //
    // For x[j]:  acc[k] += x[j] * y[j+k]  → multiply [x[j]]*4 by [y[j]..y[j+3]]
    // For x[j+1]: acc[k] += x[j+1] * y[j+1+k] → [x[j+1]]*4 by [y[j+1]..y[j+4]]
    // etc.
    while len > 4 {
        yi = yi.add(4);
        let yy1 = vld1q_s16(yi); // [y[j+4]..y[j+11]]

        // x[j+0] * y[j+0..j+3]
        let xj0 = vld1_dup_s16(xi);
        acc = vmlal_s16(acc, vget_low_s16(yy), xj0);

        // x[j+1] * y[j+1..j+4]
        let xj1 = vld1_dup_s16(xi.add(1));
        let ye1 = vextq_s16(yy, yy1, 1);
        acc = vmlal_s16(acc, vget_low_s16(ye1), xj1);

        // x[j+2] * y[j+2..j+5]
        let xj2 = vld1_dup_s16(xi.add(2));
        let ye2 = vextq_s16(yy, yy1, 2);
        acc = vmlal_s16(acc, vget_low_s16(ye2), xj2);

        // x[j+3] * y[j+3..j+6]
        let xj3 = vld1_dup_s16(xi.add(3));
        let ye3 = vextq_s16(yy, yy1, 3);
        acc = vmlal_s16(acc, vget_low_s16(ye3), xj3);

        xi = xi.add(4);
        yy = yy1;
        len -= 4;
    }

    // Write vectorised results back (each lane = one lag's partial sum)
    vst1q_s32(sum.as_mut_ptr(), acc);

    // Scalar tail (0..3 remaining x-samples).
    // yi now points at y[j], where j is the next unprocessed index.
    for k in 0..len {
        let xv = *xi.add(k) as i32;
        // y for this x[j+k] at lag 0..3 = yi[k], yi[k+1], yi[k+2], yi[k+3]
        sum[0] = sum[0].wrapping_add(xv * (*yi.add(k) as i32));
        sum[1] = sum[1].wrapping_add(xv * (*yi.add(k + 1) as i32));
        sum[2] = sum[2].wrapping_add(xv * (*yi.add(k + 2) as i32));
        sum[3] = sum[3].wrapping_add(xv * (*yi.add(k + 3) as i32));
    }
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
    let mut i: usize;
    let mut shft: i32;
    let mut nrg_tmp: u32;
    let mut nrg: i32;

    shft = 31 - silk_clz32(len as i32);

    // Pass 1: estimate energy scale with initial shft
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

    shft = (shft + 3 - silk_clz32(nrg)).max(0);

    // Pass 2: compute energy with final shft
    #[cfg(target_arch = "aarch64")]
    {
        nrg = unsafe { silk_sum_sqr_shift_neon(x, len, shft) };
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        nrg = unsafe { silk_sum_sqr_shift_avx2(x, len, shft) };
    } else {
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
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
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
    }

    *shift = shft;
    *energy = nrg;
}

/// NEON-accelerated squared-sum with right-shift for aarch64.
/// Computes Σ (x[i]^2 >> shft) for i=0..len-1.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn silk_sum_sqr_shift_neon(x: &[i16], len: usize, shft: i32) -> i32 {
    use std::arch::aarch64::*;

    // Use i64 accumulator to avoid overflow before the shift reduces values
    let mut acc = vdupq_n_s64(0i64);
    let mut i = 0;

    // Process 8 elements at a time: square to i32 then accumulate into i64
    while i + 8 <= len {
        let v = vld1q_s16(x.as_ptr().add(i));
        // Widen to i32 and square
        let lo = vget_low_s16(v);
        let hi = vget_high_s16(v);
        let sq_lo = vmull_s16(lo, lo); // i32×4
        let sq_hi = vmull_s16(hi, hi); // i32×4
        // Right-shift each i32 by shft (vshlq_s32 with negative value = right shift)
        let shift_vec = vdupq_n_s32(-shft);
        let sq_lo_sh = vshlq_s32(sq_lo, shift_vec);
        let sq_hi_sh = vshlq_s32(sq_hi, shift_vec);
        acc = vaddq_s64(acc, vpaddlq_s32(sq_lo_sh));
        acc = vaddq_s64(acc, vpaddlq_s32(sq_hi_sh));
        i += 8;
    }

    let mut nrg = vaddvq_s64(acc) as i32;

    // Scalar tail
    while i < len {
        let v = x[i] as i32;
        let sq = (v * v) as u32;
        nrg = nrg.wrapping_add((sq >> shft) as i32);
        i += 1;
    }
    nrg
}

#[inline(always)]
pub fn silk_inner_prod_aligned(ptr1: &[i16], ptr2: &[i16], len: usize) -> i32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return silk_inner_prod_aligned_neon(ptr1, ptr2, len);
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { silk_inner_prod_aligned_avx2(ptr1, ptr2, len) };
    }
    silk_inner_prod_aligned_scalar(ptr1, ptr2, len)
}

#[inline(always)]
fn silk_inner_prod_aligned_scalar(ptr1: &[i16], ptr2: &[i16], len: usize) -> i32 {
    let ptr1 = &ptr1[..len];
    let ptr2 = &ptr2[..len];
    let mut i = 0;
    let mut sum0 = 0i32;
    let mut sum1 = 0i32;
    let mut sum2 = 0i32;
    let mut sum3 = 0i32;
    let len4 = (len / 4) * 4;

    while i < len4 {
        sum0 = sum0.wrapping_add((ptr1[i] as i32).wrapping_mul(ptr2[i] as i32));
        sum1 = sum1.wrapping_add((ptr1[i + 1] as i32).wrapping_mul(ptr2[i + 1] as i32));
        sum2 = sum2.wrapping_add((ptr1[i + 2] as i32).wrapping_mul(ptr2[i + 2] as i32));
        sum3 = sum3.wrapping_add((ptr1[i + 3] as i32).wrapping_mul(ptr2[i + 3] as i32));
        i += 4;
    }

    while i < len {
        sum0 = sum0.wrapping_add((ptr1[i] as i32).wrapping_mul(ptr2[i] as i32));
        i += 1;
    }

    sum0.wrapping_add(sum1)
        .wrapping_add(sum2)
        .wrapping_add(sum3)
}

/// NEON-accelerated i16 dot-product for aarch64.
/// Uses 4 × accumulators for maximum instruction-level parallelism.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn silk_inner_prod_aligned_neon(ptr1: &[i16], ptr2: &[i16], len: usize) -> i32 {
    use std::arch::aarch64::*;

    let mut acc0 = vdupq_n_s32(0i32);
    let mut acc1 = vdupq_n_s32(0i32);
    let mut acc2 = vdupq_n_s32(0i32);
    let mut acc3 = vdupq_n_s32(0i32);

    let mut i = 0;

    // Process 32 elements per iteration for maximum ILP
    while i + 32 <= len {
        let a0 = vld1q_s16(ptr1.as_ptr().add(i));
        let b0 = vld1q_s16(ptr2.as_ptr().add(i));
        let a1 = vld1q_s16(ptr1.as_ptr().add(i + 8));
        let b1 = vld1q_s16(ptr2.as_ptr().add(i + 8));
        let a2 = vld1q_s16(ptr1.as_ptr().add(i + 16));
        let b2 = vld1q_s16(ptr2.as_ptr().add(i + 16));
        let a3 = vld1q_s16(ptr1.as_ptr().add(i + 24));
        let b3 = vld1q_s16(ptr2.as_ptr().add(i + 24));

        acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
        acc0 = vmlal_high_s16(acc0, a0, b0);
        acc1 = vmlal_s16(acc1, vget_low_s16(a1), vget_low_s16(b1));
        acc1 = vmlal_high_s16(acc1, a1, b1);
        acc2 = vmlal_s16(acc2, vget_low_s16(a2), vget_low_s16(b2));
        acc2 = vmlal_high_s16(acc2, a2, b2);
        acc3 = vmlal_s16(acc3, vget_low_s16(a3), vget_low_s16(b3));
        acc3 = vmlal_high_s16(acc3, a3, b3);

        i += 32;
    }

    // Process 8 elements per iteration
    while i + 8 <= len {
        let a0 = vld1q_s16(ptr1.as_ptr().add(i));
        let b0 = vld1q_s16(ptr2.as_ptr().add(i));
        acc0 = vmlal_s16(acc0, vget_low_s16(a0), vget_low_s16(b0));
        acc0 = vmlal_high_s16(acc0, a0, b0);
        i += 8;
    }

    // Horizontal reduce: sum all four accumulators into one i64
    let sum01 = vpaddq_s32(acc0, acc1);
    let sum23 = vpaddq_s32(acc2, acc3);
    let sum = vpaddq_s32(sum01, sum23);
    let wide = vpaddlq_s32(sum);
    let mut result = vaddvq_s64(wide);

    // Scalar tail
    while i < len {
        result += (ptr1[i] as i64) * (ptr2[i] as i64);
        i += 1;
    }

    result as i32
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

/// Compute `max_pitch` cross-correlations between `x[0..len]` and `y[i..i+len]`
/// for i = 0..max_pitch.
///
/// Matches libopus `celt_pitch_xcorr_c`: processes 4 lags at once using
/// `xcorr_kernel_c` to share y-vector loads across accumulators, then
/// handles the tail lags one at a time.
///
/// Callers must ensure `x.len() >= len` and `y.len() >= max_pitch + len`.
/// When y is shorter than required, individual lag calculations are bounded
/// by the available slice.
#[inline(always)]
pub fn silk_pitch_xcorr(x: &[i16], y: &[i16], xcorr: &mut [i32], len: usize, max_pitch: usize) {
    debug_assert!(max_pitch > 0);
    debug_assert!(x.len() >= len);
    debug_assert!(xcorr.len() >= max_pitch);

    // xcorr_kernel_c needs y[i..] to be at least len+3 long so it can pre-load
    // y_0..y_2 and then read y_3 in the loop. Check that there is enough room
    // for at least one kernel invocation.
    let y_len = y.len();

    // Process 4 lags at a time using xcorr_kernel_c (same as celt_pitch_xcorr_c).
    // This shares y-vector loads between the 4 accumulators, reducing memory
    // bandwidth compared to computing each lag independently.
    let mut i = 0;
    while i + 3 < max_pitch {
        // xcorr_kernel_c requires y[i..].len() >= len + 3
        if y_len >= i + len + 3 {
            let mut sum = [0i32; 4];
            xcorr_kernel_c(x, &y[i..], &mut sum, len);
            xcorr[i] = sum[0];
            xcorr[i + 1] = sum[1];
            xcorr[i + 2] = sum[2];
            xcorr[i + 3] = sum[3];
            i += 4;
        } else {
            // Fall back to scalar for this group
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
    // Scalar tail for remaining lags
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

    let mut tmp1_qs: i32;
    let mut tmp2_qs: i32;
    let mut state_qs = [0i32; MAX_SHAPE_LPC_ORDER + 1];
    let mut corr_qc = [0i64; MAX_SHAPE_LPC_ORDER + 1];

    debug_assert!((order & 1) == 0);

    for &input_n in input.iter().take(length) {
        tmp1_qs = (input_n as i32) << QS;

        let mut i = 0;
        while i < order {
            tmp2_qs = silk_smlaww(state_qs[i], state_qs[i + 1] - tmp1_qs, warping_q16);
            state_qs[i] = tmp1_qs;
            corr_qc[i] += silk_rshift64(silk_smull(tmp1_qs, state_qs[0]), 2 * QS - QC);

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

    // Use AVX2 path for d <= 16 (all practical SILK LPC orders)
    #[cfg(target_arch = "x86_64")]
    if d <= 16 && is_x86_feature_detected!("avx2") {
        unsafe {
            silk_lpc_analysis_filter_avx2(out, input, b, len, d);
        }
        return;
    }

    for out_val in out[..d].iter_mut() {
        *out_val = 0;
    }

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

            out32_q12 = ((*input.get_unchecked(ix) as i32) << 12).wrapping_sub(out32_q12);

            let out32 = silk_rshift_round(out32_q12, 12);

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

// ============================================================================
// x86_64 AVX2 Optimizations (runtime-dispatched)
// ============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2-accelerated i16 dot-product for x86_64.
/// Computes Σ (ptr1[i] * ptr2[i]) for i=0..len-1, wrapping.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_inner_prod_aligned_avx2(ptr1: &[i16], ptr2: &[i16], len: usize) -> i32 {
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut i = 0;

    // 32 elements per iteration (two 256-bit loads)
    while i + 32 <= len {
        let a0 = _mm256_loadu_si256(ptr1.as_ptr().add(i) as *const __m256i);
        let b0 = _mm256_loadu_si256(ptr2.as_ptr().add(i) as *const __m256i);
        let a1 = _mm256_loadu_si256(ptr1.as_ptr().add(i + 16) as *const __m256i);
        let b1 = _mm256_loadu_si256(ptr2.as_ptr().add(i + 16) as *const __m256i);
        // _mm256_madd_epi16: horizontally sum adjacent pairs → 8 × i32
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a0, b0));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a1, b1));
        i += 32;
    }
    while i + 16 <= len {
        let a = _mm256_loadu_si256(ptr1.as_ptr().add(i) as *const __m256i);
        let b = _mm256_loadu_si256(ptr2.as_ptr().add(i) as *const __m256i);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a, b));
        i += 16;
    }

    // Combine two accumulators and reduce to scalar
    let acc = _mm256_add_epi32(acc0, acc1);
    let hi = _mm256_extracti128_si256(acc, 1);
    let lo = _mm256_castsi256_si128(acc);
    let sum4 = _mm_add_epi32(lo, hi);
    let sum2 = _mm_add_epi32(sum4, _mm_srli_si128(sum4, 8));
    let sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
    let mut result = _mm_cvtsi128_si32(sum1);

    // Handle 8-element remainder (SSE madd)
    if i + 8 <= len {
        let a = _mm_loadu_si128(ptr1.as_ptr().add(i) as *const __m128i);
        let b = _mm_loadu_si128(ptr2.as_ptr().add(i) as *const __m128i);
        let p = _mm_madd_epi16(a, b);
        let p2 = _mm_add_epi32(p, _mm_srli_si128(p, 8));
        let p1 = _mm_add_epi32(p2, _mm_srli_si128(p2, 4));
        result = result.wrapping_add(_mm_cvtsi128_si32(p1));
        i += 8;
    }

    // Scalar tail
    while i < len {
        result = result.wrapping_add((ptr1[i] as i32).wrapping_mul(ptr2[i] as i32));
        i += 1;
    }

    result
}

/// AVX2-accelerated xcorr_kernel for i16 data.
/// Computes sum[k] += Σ x[i] * y[i+k] for k=0..3 simultaneously.
/// Contract: x.len() >= len, y.len() >= len + 3.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn xcorr_kernel_avx2(x: &[i16], y: &[i16], sum: &mut [i32; 4], len: usize) {
    debug_assert!(x.len() >= len);
    debug_assert!(y.len() >= len + 3);

    // Algorithm: for each 4-element group of x[i..i+4], compute contributions to
    // all 4 lags simultaneously.
    //
    // xdup = [x[i], x[i+1], x[i+2], x[i+3], x[i], x[i+1], x[i+2], x[i+3]]
    //
    // y4_01 = [y[i], y[i+1], y[i+2], y[i+3],   <- for lag 0
    //          y[i+1], y[i+2], y[i+3], y[i+4]]  <- for lag 1
    //
    // y4_23 = [y[i+2], y[i+3], y[i+4], y[i+5],  <- for lag 2
    //          y[i+3], y[i+4], y[i+5], y[i+6]]   <- for lag 3
    //
    // p01 = madd(xdup, y4_01):
    //   lane 0 = x[i]*y[i+0] + x[i+1]*y[i+1]
    //   lane 1 = x[i+2]*y[i+2] + x[i+3]*y[i+3]
    //   lane 2 = x[i]*y[i+1] + x[i+1]*y[i+2]
    //   lane 3 = x[i+2]*y[i+3] + x[i+3]*y[i+4]
    //
    // hadd(p01, p23)[0] = p01[0]+p01[1] = sum[0] partial contribution ✓
    // hadd(p01, p23)[1] = p01[2]+p01[3] = sum[1] partial contribution ✓
    // hadd(p01, p23)[2] = p23[0]+p23[1] = sum[2] partial contribution ✓
    // hadd(p01, p23)[3] = p23[2]+p23[3] = sum[3] partial contribution ✓

    let mut acc = _mm_setzero_si128();
    let mut i = 0;

    while i + 4 <= len {
        // Load x[i..i+4] into low 64 bits, duplicate to upper 64 bits
        let x4 = _mm_loadl_epi64(x.as_ptr().add(i) as *const __m128i);
        let xdup = _mm_unpacklo_epi64(x4, x4);

        // Load y[i..i+8] once; shift for each lag using byte-shift
        let y8 = _mm_loadu_si128(y.as_ptr().add(i) as *const __m128i);
        let y1 = _mm_bsrli_si128(y8, 2); // y[i+1..i+7], zeros MSB
        let y2 = _mm_bsrli_si128(y8, 4);
        let y3 = _mm_bsrli_si128(y8, 6);

        // y4_01: low half = y[i+0..3], high half = y[i+1..4]
        let y4_01 = _mm_unpacklo_epi64(y8, y1);
        // y4_23: low half = y[i+2..5], high half = y[i+3..6]
        let y4_23 = _mm_unpacklo_epi64(y2, y3);

        let p01 = _mm_madd_epi16(xdup, y4_01);
        let p23 = _mm_madd_epi16(xdup, y4_23);

        // hadd gives [sum0_partial, sum1_partial, sum2_partial, sum3_partial]
        acc = _mm_add_epi32(acc, _mm_hadd_epi32(p01, p23));

        i += 4;
    }

    // Extract accumulated partial sums
    let mut tmp = [0i32; 4];
    _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, acc);
    sum[0] = sum[0].wrapping_add(tmp[0]);
    sum[1] = sum[1].wrapping_add(tmp[1]);
    sum[2] = sum[2].wrapping_add(tmp[2]);
    sum[3] = sum[3].wrapping_add(tmp[3]);

    // Scalar tail (0..3 remaining elements)
    let mut y_0 = y[i];
    let mut y_1 = y[i + 1];
    let mut y_2 = y[i + 2];
    for j in i..len {
        let y_3 = y[j + 3];
        let xv = x[j] as i32;
        sum[0] = sum[0].wrapping_add(xv.wrapping_mul(y_0 as i32));
        sum[1] = sum[1].wrapping_add(xv.wrapping_mul(y_1 as i32));
        sum[2] = sum[2].wrapping_add(xv.wrapping_mul(y_2 as i32));
        sum[3] = sum[3].wrapping_add(xv.wrapping_mul(y_3 as i32));
        y_0 = y_1;
        y_1 = y_2;
        y_2 = y_3;
    }
}

/// AVX2-accelerated squared-sum with right-shift for x86_64.
/// Computes Σ ((x[i]*x[i]) >> shft) for i=0..len-1 wrapping into i32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_sum_sqr_shift_avx2(x: &[i16], len: usize, shft: i32) -> i32 {
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;

    // Process 16 elements per iteration
    while i + 16 <= len {
        let v = _mm256_loadu_si256(x.as_ptr().add(i) as *const __m256i);
        // madd(v, v) = v[0]*v[0]+v[1]*v[1] per i32 lane → 8 × i32
        let sq = _mm256_madd_epi16(v, v);
        // srlv_epi32 performs per-lane variable shift (same count for all lanes)
        let vshift = _mm256_set1_epi32(shft);
        let shifted = if shft > 0 {
            _mm256_srlv_epi32(sq, vshift)
        } else {
            sq
        };
        acc = _mm256_add_epi32(acc, shifted);
        i += 16;
    }

    // Horizontal reduce
    let hi = _mm256_extracti128_si256(acc, 1);
    let lo = _mm256_castsi256_si128(acc);
    let sum4 = _mm_add_epi32(lo, hi);
    let sum2 = _mm_add_epi32(sum4, _mm_srli_si128(sum4, 8));
    let sum1 = _mm_add_epi32(sum2, _mm_srli_si128(sum2, 4));
    let mut nrg = _mm_cvtsi128_si32(sum1);

    // Scalar tail
    while i < len {
        let v = x[i] as i32;
        let sq = (v * v) as u32;
        nrg = nrg.wrapping_add((sq >> shft) as i32);
        i += 1;
    }

    nrg
}

/// AVX2-accelerated LPC analysis filter for i16 data, d <= 16.
///
/// Computes out[ix] = clamp16( round((input[ix]<<12 - Σ input[ix-1-k]*b[k]), 12) )
/// for ix in d..len.  Exactly matches the scalar wrapping arithmetic by widening
/// inputs to i32 before multiplying (avoids _mm256_madd_epi16 saturation).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn silk_lpc_analysis_filter_avx2(
    out: &mut [i16],
    input: &[i16],
    b: &[i16],
    len: usize,
    d: usize,
) {
    debug_assert!(d >= 6 && (d & 1) == 0 && d <= 16);

    // Build b_rev_pad[16]: b_rev_pad[j] = b[15-j] for j >= 16-d, else 0.
    // Then input[ix-16+j] * b_rev_pad[j] = input[ix-1-(15-j)] * b[15-j]
    //   (setting k=15-j) = input[ix-1-k] * b[k]  ✓
    // For j < 16-d, b_rev_pad[j]=0 so those products contribute 0.
    let mut b_rev_pad = [0i16; 16];
    for k in 0..d {
        b_rev_pad[15 - k] = b[k];
    }

    // Zero first d outputs
    for v in out[..d].iter_mut() {
        *v = 0;
    }

    // Scalar for ix in d..16 (insufficient history for 16-element window)
    let avx_start = 16usize.max(d);
    for ix in d..avx_start.min(len) {
        let mut s: i32 = 0;
        for k in 0..d {
            s = s.wrapping_add((input[ix - 1 - k] as i32).wrapping_mul(b[k] as i32));
        }
        let out32_q12 = ((input[ix] as i32) << 12).wrapping_sub(s);
        out[ix] = silk_sat16(silk_rshift_round(out32_q12, 12)) as i16;
    }

    if avx_start >= len {
        return;
    }

    // Widen b_rev_pad to i32 (in two 8-element halves)
    let b_lo = _mm256_cvtepi16_epi32(_mm_loadu_si128(b_rev_pad.as_ptr() as *const __m128i));
    let b_hi = _mm256_cvtepi16_epi32(_mm_loadu_si128(b_rev_pad.as_ptr().add(8) as *const __m128i));

    for ix in avx_start..len {
        // Load 16 consecutive input samples ending at input[ix-1]
        let base = input.as_ptr().add(ix - 16);
        let inp_lo = _mm256_cvtepi16_epi32(_mm_loadu_si128(base as *const __m128i));
        let inp_hi = _mm256_cvtepi16_epi32(_mm_loadu_si128(base.add(8) as *const __m128i));

        // i16*i16 products fit exactly in i32; _mm256_mullo_epi32 gives wrapping result
        let prod_lo = _mm256_mullo_epi32(inp_lo, b_lo);
        let prod_hi = _mm256_mullo_epi32(inp_hi, b_hi);

        // Wrapping horizontal sum of 16 i32 values
        let sum256 = _mm256_add_epi32(prod_lo, prod_hi);
        let hi128 = _mm256_extracti128_si256(sum256, 1);
        let lo128 = _mm256_castsi256_si128(sum256);
        let sum128 = _mm_add_epi32(lo128, hi128);
        let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
        let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
        let s = _mm_cvtsi128_si32(sum32);

        let out32_q12 = ((input[ix] as i32) << 12).wrapping_sub(s);
        out[ix] = silk_sat16(silk_rshift_round(out32_q12, 12)) as i16;
    }
}


/// NEON-accelerated biquad filter (stride 2) for aarch64.
/// Based on silk_biquad_alt_stride2_neon from C code.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn silk_biquad_alt_stride2_neon(
    input_output: &mut [i16],
    b_q28: &[i32],
    a_q28: &[i32],
    s: &mut [i32],
    len: usize,
) {
    use std::arch::aarch64::*;

    // Offset for rounding: (1 << 14) - 1
    let offset_s32x2 = vdup_n_s32((1 << 14) - 1);
    let offset_s32x4 = vcombine_s32(offset_s32x2, offset_s32x2);

    // Negate A_Q28 values and split in two parts (matching C implementation)
    let a_q28_s32x2 = vneg_s32(vld1_s32(a_q28.as_ptr()));
    let a_l_s32x2 = vreinterpret_s32_u32(vshr_n_u32(
        vreinterpret_u32_s32(vshl_n_s32(a_q28_s32x2, 18)),
        3,
    ));
    let mut a_u_s32x2 = vshr_n_s32(a_q28_s32x2, 14);
    a_u_s32x2 = vshr_n_s32(vshl_n_s32(a_u_s32x2, 16), 1);

    let b_q28_s32x2 = vld1_s32(b_q28.as_ptr());
    let t_s32x2 = vld1_s32(b_q28.as_ptr().add(1));

    // Zip coefficients for vectorized use
    let t0_s32x2x2 = vzip_s32(a_l_s32x2, a_l_s32x2);
    let t1_s32x2x2 = vzip_s32(a_u_s32x2, a_u_s32x2);
    let t2_s32x2x2 = vzip_s32(t_s32x2, t_s32x2);

    let a_l_s32x4 = vcombine_s32(t0_s32x2x2.0, t0_s32x2x2.1);
    let a_u_s32x4 = vcombine_s32(t1_s32x2x2.0, t1_s32x2x2.1);
    let b_q28_s32x4 = vcombine_s32(t2_s32x2x2.0, t2_s32x2x2.1);

    // Load and transpose state vector
    let mut s_s32x4 = vld1q_s32(s.as_ptr());
    let s_s32x2x2 = vtrn_s32(vget_low_s32(s_s32x4), vget_high_s32(s_s32x4));
    s_s32x4 = vcombine_s32(s_s32x2x2.0, s_s32x2x2.1);

    let mut k = 0;
    while k + 1 < len {
        // Load 4 input samples: in[2*k + 0,1,2,3]
        let in_s16x4 = vld1_s16(input_output.as_ptr().add(2 * k));
        let in_s32x4 = vshll_n_s16(in_s16x4, 15);

        // t = vqdmulhq_lane_s32(in << 15, B_Q28_s32x2, 0) = SMULWB(B[0], in)
        let t_s32x4 = vqdmulhq_lane_s32(in_s32x4, b_q28_s32x2, 0);

        // Process first pair
        let in_low = vcombine_s32(vget_low_s32(in_s32x4), vget_low_s32(in_s32x4));
        let t_low = vget_low_s32(t_s32x4);

        // out32_Q14 = (S[0,1] + SMULWB(B[0], in[0,1])) << 2
        let mut out32_q14_s32x2 = vadd_s32(vget_low_s32(s_s32x4), t_low);
        out32_q14_s32x2 = vshl_n_s32(out32_q14_s32x2, 2);

        // S[0,1] = S[2,3]; S[2,3] = 0
        s_s32x4 = vcombine_s32(vget_high_s32(s_s32x4), vdup_n_s32(0));

        // out32_Q14 broadcast
        let out32_q14_s32x4 = vcombine_s32(out32_q14_s32x2, out32_q14_s32x2);

        // S += vqdmulhq_s32(out32_Q14, A_L) >> 14 (rounded)
        let tmp_s32x4 = vqdmulhq_s32(out32_q14_s32x4, a_l_s32x4);
        s_s32x4 = vrsraq_n_s32(s_s32x4, tmp_s32x4, 14);

        // S += vqdmulhq_s32(out32_Q14, A_U)
        let tmp_s32x4 = vqdmulhq_s32(out32_q14_s32x4, a_u_s32x4);
        s_s32x4 = vaddq_s32(s_s32x4, tmp_s32x4);

        // S += vqdmulhq_s32(in, B_Q28[1,1,2,2])
        let tmp_s32x4 = vqdmulhq_s32(in_low, b_q28_s32x4);
        s_s32x4 = vaddq_s32(s_s32x4, tmp_s32x4);

        // Process second pair
        let in_high = vcombine_s32(vget_high_s32(in_s32x4), vget_high_s32(in_s32x4));
        let t_high = vget_high_s32(t_s32x4);

        let mut out32_q14_1_s32x2 = vadd_s32(vget_low_s32(s_s32x4), t_high);
        out32_q14_1_s32x2 = vshl_n_s32(out32_q14_1_s32x2, 2);

        s_s32x4 = vcombine_s32(vget_high_s32(s_s32x4), vdup_n_s32(0));

        let out32_q14_1_s32x4 = vcombine_s32(out32_q14_1_s32x2, out32_q14_1_s32x2);

        let tmp_s32x4 = vqdmulhq_s32(out32_q14_1_s32x4, a_l_s32x4);
        s_s32x4 = vrsraq_n_s32(s_s32x4, tmp_s32x4, 14);

        let tmp_s32x4 = vqdmulhq_s32(out32_q14_1_s32x4, a_u_s32x4);
        s_s32x4 = vaddq_s32(s_s32x4, tmp_s32x4);

        let tmp_s32x4 = vqdmulhq_s32(in_high, b_q28_s32x4);
        s_s32x4 = vaddq_s32(s_s32x4, tmp_s32x4);

        // Combine outputs and saturate
        let out32_q14_combined = vcombine_s32(out32_q14_s32x2, out32_q14_1_s32x2);
        let out_s32x4 = vaddq_s32(out32_q14_combined, offset_s32x4);
        let out_s16x4 = vqshrn_n_s32(out_s32x4, 14);

        vst1_s16(input_output.as_mut_ptr().add(2 * k), out_s16x4);

        k += 2;
    }

    // Process leftover (if len is odd)
    if k < len {
        let in0 = input_output[2 * k] as i32;
        let in1 = input_output[2 * k + 1] as i32;

        let out32_q14_0 = silk_smlawb(vgetq_lane_s32(s_s32x4, 0), b_q28[0], in0) << 2;
        let out32_q14_1 = silk_smlawb(vgetq_lane_s32(s_s32x4, 2), b_q28[0], in1) << 2;

        // Scalar fallback for last iteration
        let a0_l = (-a_q28[0]) & 0x00003FFF;
        let a0_u = -a_q28[0] >> 14;
        let a1_l = (-a_q28[1]) & 0x00003FFF;
        let a1_u = -a_q28[1] >> 14;

        let s0_new = vgetq_lane_s32(s_s32x4, 1)
            + silk_rshift_round(silk_smulwb(out32_q14_0, a0_l), 14);
        let s2_new = vgetq_lane_s32(s_s32x4, 3)
            + silk_rshift_round(silk_smulwb(out32_q14_1, a0_l), 14);

        let s0_new = silk_smlawb(s0_new, out32_q14_0, a0_u);
        let s2_new = silk_smlawb(s2_new, out32_q14_1, a0_u);
        let s0_new = silk_smlawb(s0_new, b_q28[1], in0);
        let s2_new = silk_smlawb(s2_new, b_q28[1], in1);

        let s1_new = silk_rshift_round(silk_smulwb(out32_q14_0, a1_l), 14);
        let s3_new = silk_rshift_round(silk_smulwb(out32_q14_1, a1_l), 14);
        let s1_new = silk_smlawb(s1_new, out32_q14_0, a1_u);
        let s3_new = silk_smlawb(s3_new, out32_q14_1, a1_u);
        let s1_new = silk_smlawb(s1_new, b_q28[2], in0);
        let s3_new = silk_smlawb(s3_new, b_q28[2], in1);

        input_output[2 * k] = silk_sat16(silk_rshift(out32_q14_0 + (1 << 14) - 1, 14)) as i16;
        input_output[2 * k + 1] = silk_sat16(silk_rshift(out32_q14_1 + (1 << 14) - 1, 14)) as i16;

        s[0] = s0_new;
        s[1] = s1_new;
        s[2] = s2_new;
        s[3] = s3_new;
    } else {
        // Transpose state back
        vst1q_lane_s32(&mut s[0], s_s32x4, 0);
        vst1q_lane_s32(&mut s[1], s_s32x4, 2);
        vst1q_lane_s32(&mut s[2], s_s32x4, 1);
        vst1q_lane_s32(&mut s[3], s_s32x4, 3);
    }
}
