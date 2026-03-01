use crate::silk::define::*;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::macros::*;
use crate::silk::resampler::*;
use crate::silk::sigproc_fix::*;
use crate::silk::sort::silk_insertion_sort_decreasing_int16;
use crate::silk::structs::*;
use crate::silk::tables::*;

const SCRATCH_SIZE: usize = 22;
const SF_LENGTH_4KHZ: usize = PE_SUBFR_LENGTH_MS * 4;
const SF_LENGTH_8KHZ: usize = PE_SUBFR_LENGTH_MS * 8;
const MIN_LAG_4KHZ: usize = PE_MIN_LAG_MS * 4;
const MIN_LAG_8KHZ: usize = PE_MIN_LAG_MS * 8;
const MAX_LAG_4KHZ: usize = PE_MAX_LAG_MS * 4;
const MAX_LAG_8KHZ: usize = PE_MAX_LAG_MS * 8 - 1;
const CSTRIDE_4KHZ: usize = MAX_LAG_4KHZ + 1 - MIN_LAG_4KHZ;
const CSTRIDE_8KHZ: usize = MAX_LAG_8KHZ + 3 - (MIN_LAG_8KHZ - 2);
const D_COMP_MIN: isize = (MIN_LAG_8KHZ as isize) - 3;
const D_COMP_MAX: isize = (MAX_LAG_8KHZ as isize) + 4;
const D_COMP_STRIDE: usize = (D_COMP_MAX - D_COMP_MIN) as usize;

static SILK_NB_CBK_SEARCHS_STAGE3: [usize; SILK_PE_MAX_COMPLEX + 1] = [
    PE_NB_CBKS_STAGE3_MIN,
    PE_NB_CBKS_STAGE3_MID,
    PE_NB_CBKS_STAGE3_MAX,
];

pub fn silk_find_pitch_lags_fix(
    ps_enc: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControl,
    res: &mut [i16],
    x: &[i16],
    _arch: i32,
) {
    let la_pitch = ps_enc.s_cmn.la_pitch as usize;
    let frame_length = ps_enc.s_cmn.frame_length as usize;
    let ltp_mem_length = ps_enc.s_cmn.ltp_mem_length as usize;
    let pitch_lpc_win_length = ps_enc.s_cmn.pitch_lpc_win_length as usize;
    let pitch_lpc_order = ps_enc.pitch_estimation_lpc_order as usize;

    let buf_len = la_pitch + frame_length + ltp_mem_length;

    /*************************************/
    /* Estimate LPC AR coefficients      */
    /*************************************/

    /* Calculate windowed signal */
    let mut wsig = [0i16; PE_MAX_FRAME_LENGTH];
    let x_ptr_start = buf_len - pitch_lpc_win_length;

    /* First la_pitch samples: sine-rise window */
    silk_apply_sine_window(&mut wsig[..], &x[x_ptr_start..], 1, la_pitch);

    /* Middle un-windowed samples */
    let mid_len = pitch_lpc_win_length - 2 * la_pitch;
    wsig[la_pitch..la_pitch + mid_len]
        .copy_from_slice(&x[x_ptr_start + la_pitch..x_ptr_start + la_pitch + mid_len]);

    /* Last la_pitch samples: sine-fall window */
    let last_start = pitch_lpc_win_length - la_pitch;
    silk_apply_sine_window(
        &mut wsig[last_start..],
        &x[x_ptr_start + last_start..],
        2,
        la_pitch,
    );

    /* Calculate autocorrelation sequence */
    let mut auto_corr = [0i32; MAX_FIND_PITCH_LPC_ORDER + 1];
    let mut scale = 0i32;
    silk_autocorr(
        &mut auto_corr,
        &mut scale,
        &wsig,
        pitch_lpc_win_length,
        pitch_lpc_order + 1,
    );

    /* Add white noise, as fraction of energy */
    // FIND_PITCH_WHITE_NOISE_FRACTION = 1e-3 in Q16: SILK_FIX_CONST(0.001, 16) = round(0.001 * 65536 + 0.5) = 66
    auto_corr[0] = silk_smlawb(auto_corr[0], auto_corr[0], 66) + 1;

    /* Calculate the reflection coefficients using Schur */
    let mut rc_q15 = [0i16; MAX_FIND_PITCH_LPC_ORDER];
    let res_nrg = silk_schur(&mut rc_q15, &auto_corr, pitch_lpc_order);

    /* Prediction gain */
    ps_enc_ctrl.pred_gain_q16 = silk_div32_varq(auto_corr[0], res_nrg.max(1), 16);

    /* Convert reflection coefficients to prediction coefficients */
    let mut a_q24 = [0i32; MAX_FIND_PITCH_LPC_ORDER];
    silk_k2a(&mut a_q24, &rc_q15, pitch_lpc_order);

    /* Convert from 32 bit Q24 to 16 bit Q12 coefs */
    let mut a_q12 = [0i16; MAX_FIND_PITCH_LPC_ORDER];
    for i in 0..pitch_lpc_order {
        a_q12[i] = silk_sat16(a_q24[i] >> 12) as i16;
    }

    /* Do BWE */
    // FIND_PITCH_BANDWIDTH_EXPANSION = 0.99, Q16 = 64881
    silk_bwexpander(&mut a_q12, pitch_lpc_order, 64881);

    /*****************************************/
    /* LPC analysis filtering                */
    /*****************************************/
    silk_lpc_analysis_filter(res, x, &a_q12, buf_len, pitch_lpc_order, 0);

    if ps_enc.s_cmn.indices.signal_type != TYPE_NO_VOICE_ACTIVITY as i8
        && ps_enc.s_cmn.first_frame_after_reset == 0
    {
        /* Threshold for pitch estimator */
        // thrhld_Q13 = SILK_FIX_CONST(0.6, 13) = 4915
        let mut thrhld_q13: i32 = 4915;
        // SILK_FIX_CONST(-0.004, 13) = -33
        thrhld_q13 = silk_smlabb(thrhld_q13, -33, pitch_lpc_order as i32);
        // SILK_FIX_CONST(-0.1, 21) = -209715
        thrhld_q13 = silk_smlawb(thrhld_q13, -209715, ps_enc.s_cmn.speech_activity_q8);
        // SILK_FIX_CONST(-0.15, 13) = -1229
        thrhld_q13 = silk_smlabb(thrhld_q13, -1229, ps_enc.s_cmn.prev_signal_type >> 1);
        // SILK_FIX_CONST(-0.1, 14) = -1638
        thrhld_q13 = silk_smlawb(thrhld_q13, -1638, ps_enc.s_cmn.input_tilt_q15);
        thrhld_q13 = silk_sat16(thrhld_q13) as i32;

        /*****************************************/
        /* Call pitch estimator                  */
        /*****************************************/
        let mut ltp_corr_q15 = ps_enc.ltp_corr_q15;
        if silk_pitch_analysis_core(
            res,
            &mut ps_enc_ctrl.pitch_l,
            &mut ps_enc.s_cmn.indices.lag_index,
            &mut ps_enc.s_cmn.indices.contour_index,
            &mut ltp_corr_q15,
            ps_enc.s_cmn.prev_lag,
            ps_enc.s_cmn.pitch_estimation_threshold_q16,
            thrhld_q13,
            ps_enc.s_cmn.fs_khz,
            ps_enc.s_cmn.pitch_estimation_complexity as usize,
            ps_enc.s_cmn.nb_subfr as usize,
        ) == 0
        {
            ps_enc.s_cmn.indices.signal_type = TYPE_VOICED as i8;
        } else {
            ps_enc.s_cmn.indices.signal_type = TYPE_UNVOICED as i8;
        }
        ps_enc.ltp_corr_q15 = ltp_corr_q15;
    } else {
        ps_enc_ctrl.pitch_l = [0; MAX_NB_SUBFR];
        ps_enc.s_cmn.indices.lag_index = 0;
        ps_enc.s_cmn.indices.contour_index = 0;
        ps_enc.ltp_corr_q15 = 0;
    }
}

pub fn silk_pitch_analysis_core(
    frame_unscaled: &[i16],
    pitch_out: &mut [i32],
    lag_index: &mut i16,
    contour_index: &mut i8,
    ltp_corr_q15: &mut i32,
    prev_lag: i32,
    search_thres1_q16: i32,
    search_thres2_q13: i32,
    fs_khz: i32,
    complexity: usize,
    nb_subfr: usize,
) -> i32 {
    let mut filt_state = [0i32; 6];
    let mut frame_8khz_buf =
        [0i16; (PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS) * 8];
    let mut frame_4khz = [0i16; (PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS) * 4];
    let mut frame_scaled_buf =
        [0i16; (PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS) * PE_MAX_FS_KHZ];

    let frame_length = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_khz as usize;
    let frame_length_8khz = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 8;
    let frame_length_4khz = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 4;
    let sf_length = PE_SUBFR_LENGTH_MS * fs_khz as usize;
    let min_lag = PE_MIN_LAG_MS * fs_khz as usize;
    let max_lag = PE_MAX_LAG_MS * fs_khz as usize;

    /* Downscale input if necessary */
    let mut energy = 0i32;
    let mut shift = 0i32;
    silk_sum_sqr_shift(&mut energy, &mut shift, frame_unscaled, frame_length);
    shift += 3 - (31 - energy.leading_zeros() as i32); // at least two bits headroom

    let frame: &[i16];
    if shift > 0 {
        let s = (shift + 1) >> 1;
        for i in 0..frame_length {
            frame_scaled_buf[i] = (frame_unscaled[i] >> s) as i16;
        }
        frame = &frame_scaled_buf[..frame_length];
    } else {
        frame = &frame_unscaled[..frame_length];
    }

    /* Resample from input sampled at Fs_kHz to 8 kHz */
    let frame_8khz: &[i16];
    if fs_khz == 16 {
        filt_state[0..2].fill(0);
        let output = unsafe {
            std::slice::from_raw_parts_mut(frame_8khz_buf.as_mut_ptr(), frame_length_8khz)
        };
        silk_resampler_down2(&mut filt_state[..2], output, frame, frame_length as i32);
        frame_8khz = output;
    } else if fs_khz == 12 {
        filt_state[0..6].fill(0);
        let output = unsafe {
            std::slice::from_raw_parts_mut(frame_8khz_buf.as_mut_ptr(), frame_length_8khz)
        };
        silk_resampler_down2_3(&mut filt_state[..6], output, frame, frame_length as i32);
        frame_8khz = output;
    } else {
        frame_8khz = frame;
    }

    /* Decimate again to 4 kHz */
    filt_state[0..2].fill(0);
    let frame_4khz_sub =
        unsafe { std::slice::from_raw_parts_mut(frame_4khz.as_mut_ptr(), frame_length_4khz) };
    silk_resampler_down2(
        &mut filt_state[..2],
        frame_4khz_sub,
        frame_8khz,
        frame_length_8khz as i32,
    );

    /* Low-pass filter */
    for i in (1..frame_length_4khz).rev() {
        frame_4khz[i] = silk_add_sat16(frame_4khz[i], frame_4khz[i - 1]);
    }

    /******************************************************************************
     * FIRST STAGE, operating in 4 khz
     ******************************************************************************/
    let mut c = [0i16; PE_MAX_NB_SUBFR * CSTRIDE_4KHZ];
    let mut xcorr32 = [0i32; MAX_LAG_4KHZ - MIN_LAG_4KHZ + 1];

    let mut target_ptr_idx = SF_LENGTH_4KHZ << 2;
    for k in 0..(nb_subfr >> 1) {
        let basis_ptr_idx = target_ptr_idx - MIN_LAG_4KHZ;
        silk_pitch_xcorr(
            &frame_4khz[target_ptr_idx..],
            &frame_4khz[(target_ptr_idx - MAX_LAG_4KHZ)..],
            &mut xcorr32,
            SF_LENGTH_8KHZ,
            MAX_LAG_4KHZ - MIN_LAG_4KHZ + 1,
        );

        let mut cross_corr = xcorr32[MAX_LAG_4KHZ - MIN_LAG_4KHZ];
        let mut normalizer = silk_inner_prod_aligned(
            &frame_4khz[target_ptr_idx..],
            &frame_4khz[target_ptr_idx..],
            SF_LENGTH_8KHZ,
        );
        normalizer = normalizer.wrapping_add(silk_inner_prod_aligned(
            &frame_4khz[basis_ptr_idx..],
            &frame_4khz[basis_ptr_idx..],
            SF_LENGTH_8KHZ,
        ));
        normalizer = normalizer.wrapping_add(silk_smulbb(SF_LENGTH_8KHZ as i32, 4000));

        c[k * CSTRIDE_4KHZ] = silk_div32_varq(cross_corr, normalizer, 14) as i16;

        let mut current_basis_ptr_idx = basis_ptr_idx;
        for d in (MIN_LAG_4KHZ + 1)..=MAX_LAG_4KHZ {
            current_basis_ptr_idx -= 1;
            cross_corr = xcorr32[MAX_LAG_4KHZ - d];
            normalizer = normalizer.wrapping_add(
                silk_smulbb(
                    frame_4khz[current_basis_ptr_idx] as i32,
                    frame_4khz[current_basis_ptr_idx] as i32,
                ) - silk_smulbb(
                    frame_4khz[current_basis_ptr_idx + SF_LENGTH_8KHZ] as i32,
                    frame_4khz[current_basis_ptr_idx + SF_LENGTH_8KHZ] as i32,
                ),
            );
            c[k * CSTRIDE_4KHZ + (d - MIN_LAG_4KHZ)] =
                silk_div32_varq(cross_corr, normalizer, 14) as i16;
        }
        target_ptr_idx += SF_LENGTH_8KHZ;
    }

    let mut c_combined = [0i16; CSTRIDE_4KHZ];
    if nb_subfr == PE_MAX_NB_SUBFR {
        for i in (MIN_LAG_4KHZ..=MAX_LAG_4KHZ).rev() {
            let mut sum = c[0 * CSTRIDE_4KHZ + (i - MIN_LAG_4KHZ)] as i32
                + c[1 * CSTRIDE_4KHZ + (i - MIN_LAG_4KHZ)] as i32;
            sum = silk_smlawb(sum, sum, (-(i as i32) << 4) as i32);
            c_combined[i - MIN_LAG_4KHZ] = sum as i16;
        }
    } else {
        for i in (MIN_LAG_4KHZ..=MAX_LAG_4KHZ).rev() {
            let mut sum = (c[0 * CSTRIDE_4KHZ + (i - MIN_LAG_4KHZ)] as i32) << 1;
            sum = silk_smlawb(sum, sum, (-(i as i32) << 4) as i32);
            c_combined[i - MIN_LAG_4KHZ] = sum as i16;
        }
    }

    let mut d_srch = [0i32; PE_D_SRCH_LENGTH];
    let length_d_srch_orig = (complexity << 1) + 4;
    let mut length_d_srch = length_d_srch_orig;
    silk_insertion_sort_decreasing_int16(
        &mut c_combined[..CSTRIDE_4KHZ],
        &mut d_srch[..length_d_srch],
        CSTRIDE_4KHZ,
        length_d_srch,
    );

    if (c_combined[0] as i32) < 3277 {
        // SILK_FIX_CONST( 0.2, 14 )
        pitch_out[0..nb_subfr].fill(0);
        *ltp_corr_q15 = 0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    let threshold = silk_smulwb(search_thres1_q16, c_combined[0] as i32);
    for i in 0..length_d_srch {
        if (c_combined[i] as i32) > threshold {
            d_srch[i] = (d_srch[i] + MIN_LAG_4KHZ as i32) << 1;
        } else {
            length_d_srch = i;
            break;
        }
    }

    let mut d_comp = [0i16; D_COMP_STRIDE];
    for i in 0..length_d_srch {
        d_comp[(d_srch[i] as isize - D_COMP_MIN) as usize] = 1;
    }

    for i in (3..D_COMP_STRIDE).rev() {
        d_comp[i] += d_comp[i - 1] + d_comp[i - 2] + d_comp[i - 3];
    }

    let mut length_d_comp = 0;
    let mut d_comp_indices = [0i32; D_COMP_STRIDE];
    for i in MIN_LAG_8KHZ..D_COMP_MAX as usize {
        if d_comp[i - D_COMP_MIN as usize] > 0 {
            d_comp_indices[length_d_comp] = (i - 2) as i32;
            length_d_comp += 1;
        }
    }

    /**********************************************************************************
     ** SECOND STAGE, operating at 8 kHz, on lag sections with high correlation
     *************************************************************************************/
    let mut c_8khz = [0i16; PE_MAX_NB_SUBFR * CSTRIDE_8KHZ];
    let mut target_ptr_8khz_idx = PE_LTP_MEM_LENGTH_MS * 8;
    for k in 0..nb_subfr {
        let energy_target = silk_inner_prod_aligned(
            &frame_8khz[target_ptr_8khz_idx..],
            &frame_8khz[target_ptr_8khz_idx..],
            SF_LENGTH_8KHZ,
        )
        .wrapping_add(1);
        for j in 0..length_d_comp {
            let d = d_comp_indices[j];
            let basis_ptr_idx = target_ptr_8khz_idx as i32 - d;
            let cross_corr = silk_inner_prod_aligned(
                &frame_8khz[target_ptr_8khz_idx..],
                &frame_8khz[basis_ptr_idx as usize..],
                SF_LENGTH_8KHZ,
            );
            if cross_corr > 0 {
                let energy_basis = silk_inner_prod_aligned(
                    &frame_8khz[basis_ptr_idx as usize..],
                    &frame_8khz[basis_ptr_idx as usize..],
                    SF_LENGTH_8KHZ,
                );
                c_8khz[k * CSTRIDE_8KHZ + (d - (MIN_LAG_8KHZ as i32 - 2)) as usize] =
                    silk_div32_varq(cross_corr, energy_target.wrapping_add(energy_basis), 14)
                        as i16;
            } else {
                c_8khz[k * CSTRIDE_8KHZ + (d - (MIN_LAG_8KHZ as i32 - 2)) as usize] = 0;
            }
        }
        target_ptr_8khz_idx += SF_LENGTH_8KHZ;
    }

    let mut cc_max = i32::MIN;
    let mut cc_max_b = i32::MIN;
    let mut cb_i_max = 0usize;
    let mut lag = -1i32;

    let mut prev_lag_8khz = prev_lag;
    if prev_lag_8khz > 0 {
        if fs_khz == 12 {
            prev_lag_8khz = silk_div32_16(prev_lag_8khz << 1, 3);
        } else if fs_khz == 16 {
            prev_lag_8khz >>= 1;
        }
    }
    let prev_lag_log2_q7 = if prev_lag_8khz > 0 {
        silk_lin2log(prev_lag_8khz)
    } else {
        0
    };

    let nb_cbk_search: usize = if nb_subfr == PE_MAX_NB_SUBFR {
        if fs_khz == 8 && complexity > SILK_PE_MIN_COMPLEX {
            PE_NB_CBKS_STAGE2_EXT
        } else {
            PE_NB_CBKS_STAGE2
        }
    } else {
        PE_NB_CBKS_STAGE2_10MS
    };

    for k in 0..length_d_srch {
        let d = d_srch[k];
        let mut cc = [0i32; PE_NB_CBKS_STAGE2_EXT];
        for j in 0..nb_cbk_search {
            cc[j] = 0;
            for i in 0..nb_subfr {
                let lag_cb_ptr_curr: &[i8] = if nb_subfr == PE_MAX_NB_SUBFR {
                    &SILK_CB_LAGS_STAGE2[i]
                } else {
                    &SILK_CB_LAGS_STAGE2_10_MS[i]
                };
                let d_subfr = d + lag_cb_ptr_curr[j] as i32;
                // Boundary check: ensure d_subfr is within valid range for c_8khz
                // Valid range for c_8khz index is [0, CSTRIDE_8KHZ-1]
                // Index = d_subfr - (MIN_LAG_8KHZ - 2)
                // So d_subfr must be in [MIN_LAG_8KHZ - 2, MIN_LAG_8KHZ - 2 + CSTRIDE_8KHZ - 1]
                // = [MIN_LAG_8KHZ - 2, MIN_LAG_8KHZ + CSTRIDE_8KHZ - 3]
                let min_lag = MIN_LAG_8KHZ as i32 - 2;
                let max_lag = MIN_LAG_8KHZ as i32 + CSTRIDE_8KHZ as i32 - 3;
                if d_subfr >= min_lag && d_subfr <= max_lag {
                    let idx = (d_subfr - min_lag) as usize;
                    if i * CSTRIDE_8KHZ + idx < c_8khz.len() {
                        cc[j] += c_8khz[i * CSTRIDE_8KHZ + idx] as i32;
                    }
                }
                // If out of range, skip (C code may also skip or use default values)
            }
        }

        let mut cc0_max_new = i32::MIN;
        let mut cb_i_max_new = 0usize;
        for i in 0..nb_cbk_search {
            if cc[i] > cc0_max_new {
                cc0_max_new = cc[i];
                cb_i_max_new = i;
            }
        }

        let lag_log2_q7 = silk_lin2log(d);
        let mut cc_max_new_b =
            cc0_max_new - silk_rshift(silk_smulbb((nb_subfr * 410) as i32, lag_log2_q7), 7); // 410 is SILK_FIX_CONST( PE_SHORTLAG_BIAS, 13 )

        if prev_lag_8khz > 0 {
            let mut delta_lag_log2_sqr_q7 = lag_log2_q7 - prev_lag_log2_q7;
            delta_lag_log2_sqr_q7 =
                silk_rshift(silk_smulbb(delta_lag_log2_sqr_q7, delta_lag_log2_sqr_q7), 7);
            let mut prev_lag_bias_q13 =
                silk_rshift(silk_smulbb((nb_subfr * 1638) as i32, *ltp_corr_q15), 15); // 1638 is SILK_FIX_CONST( PE_PREVLAG_BIAS, 13 )
            prev_lag_bias_q13 = silk_div32(
                silk_mul(prev_lag_bias_q13, delta_lag_log2_sqr_q7),
                delta_lag_log2_sqr_q7 + 64,
            ); // 64 is SILK_FIX_CONST( 0.5, 7 )
            cc_max_new_b -= prev_lag_bias_q13;
        }

        let lag_cb_ptr_curr_0: &[i8] = if nb_subfr == PE_MAX_NB_SUBFR {
            &SILK_CB_LAGS_STAGE2[0]
        } else {
            &SILK_CB_LAGS_STAGE2_10_MS[0]
        };
        if cc_max_new_b > cc_max_b
            && cc0_max_new > silk_smulbb(nb_subfr as i32, search_thres2_q13)
            && (d + lag_cb_ptr_curr_0[cb_i_max_new] as i32) >= MIN_LAG_8KHZ as i32
        {
            cc_max_b = cc_max_new_b;
            cc_max = cc0_max_new;
            lag = d;
            cb_i_max = cb_i_max_new;
        }
    }

    if lag == -1 {
        pitch_out[0..nb_subfr].fill(0);
        *ltp_corr_q15 = 0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    *ltp_corr_q15 = silk_lshift(silk_div32_16(cc_max, nb_subfr as i32), 2);

    if fs_khz > 8 {
        let cb_i_max_old = cb_i_max;
        if fs_khz == 12 {
            lag = silk_rshift(silk_smulbb(lag, 3), 1);
        } else if fs_khz == 16 {
            lag <<= 1;
        } else {
            lag = silk_smulbb(lag, 3);
        }

        lag = silk_limit_32(lag, min_lag as i32, max_lag as i32);
        let start_lag = (lag - 2).max(min_lag as i32);
        let end_lag = (lag + 2).min(max_lag as i32);
        let mut lag_new = lag;
        cb_i_max = 0;

        cc_max = i32::MIN;
        for k in 0..nb_subfr {
            let lag_cb_ptr_curr: &[i8] = if nb_subfr == PE_MAX_NB_SUBFR {
                &SILK_CB_LAGS_STAGE2[k]
            } else {
                &SILK_CB_LAGS_STAGE2_10_MS[k]
            };
            pitch_out[k] = lag + 2 * lag_cb_ptr_curr[cb_i_max_old] as i32;
        }

        let nb_cbk_search_st3: usize = if nb_subfr == PE_MAX_NB_SUBFR {
            SILK_NB_CBK_SEARCHS_STAGE3[complexity]
        } else {
            PE_NB_CBKS_STAGE3_10MS
        };

        let mut energies_st3 =
            [[[0i32; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]; PE_MAX_NB_SUBFR];
        let mut cross_corr_st3 =
            [[[0i32; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]; PE_MAX_NB_SUBFR];

        silk_p_ana_calc_corr_st3(
            &mut cross_corr_st3,
            frame,
            start_lag as usize,
            sf_length,
            nb_subfr,
            complexity,
            fs_khz as usize,
        );
        silk_p_ana_calc_energy_st3(
            &mut energies_st3,
            frame,
            start_lag as usize,
            sf_length,
            nb_subfr,
            complexity,
            fs_khz as usize,
        );

        let mut lag_counter = 0;
        let contour_bias_q15 = silk_div32_16(1638, (lag as i16).into()); // 1638 is SILK_FIX_CONST( PE_FLATCONTOUR_BIAS, 15 )

        let energy_target = silk_inner_prod_aligned(
            &frame[PE_LTP_MEM_LENGTH_MS * fs_khz as usize..],
            &frame[PE_LTP_MEM_LENGTH_MS * fs_khz as usize..],
            nb_subfr * sf_length,
        )
        .wrapping_add(1);
        for d in start_lag..=end_lag {
            for j in 0..nb_cbk_search_st3 {
                let mut cross_corr = 0i32;
                let mut energy = energy_target;
                for k in 0..nb_subfr {
                    cross_corr = silk_add_sat32(cross_corr, cross_corr_st3[k][j][lag_counter]);
                    energy = silk_add_sat32(energy, energies_st3[k][j][lag_counter]);
                }
                if cross_corr > 0 {
                    let mut cc_max_new = silk_div32_varq(cross_corr, energy, 14);
                    let diff = i16::MAX as i32 - silk_mul(contour_bias_q15, j as i32);
                    cc_max_new = silk_smulwb(cc_max_new, diff as i32);
                    let lag_cb_ptr_st3_0: &[i8] = if nb_subfr == PE_MAX_NB_SUBFR {
                        &SILK_CB_LAGS_STAGE3[0]
                    } else {
                        &SILK_CB_LAGS_STAGE3_10_MS[0]
                    };
                    if cc_max_new > cc_max && (d + lag_cb_ptr_st3_0[j] as i32) <= max_lag as i32 {
                        cc_max = cc_max_new;
                        lag_new = d;
                        cb_i_max = j;
                    }
                }
            }
            lag_counter += 1;
        }

        for k in 0..nb_subfr {
            let lag_cb_ptr_st3_k: &[i8] = if nb_subfr == PE_MAX_NB_SUBFR {
                &SILK_CB_LAGS_STAGE3[k]
            } else {
                &SILK_CB_LAGS_STAGE3_10_MS[k]
            };
            pitch_out[k] = lag_new + lag_cb_ptr_st3_k[cb_i_max] as i32;
            pitch_out[k] = silk_limit(
                pitch_out[k],
                min_lag as i32,
                (PE_MAX_LAG_MS * fs_khz as usize) as i32,
            );
        }
        *lag_index = (lag_new - min_lag as i32) as i16;
        *contour_index = cb_i_max as i8;
    } else {
        /* fs_khz == 8 */
        for k in 0..nb_subfr {
            let lag_cb_ptr_curr: &[i8] = if nb_subfr == PE_MAX_NB_SUBFR {
                &SILK_CB_LAGS_STAGE2[k]
            } else {
                &SILK_CB_LAGS_STAGE2_10_MS[k]
            };
            pitch_out[k] = lag + lag_cb_ptr_curr[cb_i_max] as i32;
            pitch_out[k] = silk_limit(
                pitch_out[k],
                MIN_LAG_8KHZ as i32,
                (PE_MAX_LAG_MS * 8) as i32,
            );
        }
        *lag_index = (lag - MIN_LAG_8KHZ as i32) as i16;
        *contour_index = cb_i_max as i8;
    }

    0
}

fn silk_p_ana_calc_corr_st3(
    cross_corr_st3: &mut [[[i32; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]; PE_MAX_NB_SUBFR],
    frame: &[i16],
    start_lag: usize,
    sf_length: usize,
    nb_subfr: usize,
    complexity: usize,
    fs_khz: usize,
) {
    let mut xcorr32 = [0i32; SCRATCH_SIZE];
    let mut scratch_mem = [0i32; SCRATCH_SIZE];

    let mut target_ptr_idx = PE_LTP_MEM_LENGTH_MS * (if fs_khz > 8 { fs_khz as usize } else { 8 });
    for k in 0..nb_subfr {
        let (lag_cb_ptr_curr, lag_range_ptr_curr, nb_cbk_search): (&[i8], &[i8; 2], usize) =
            if nb_subfr == PE_MAX_NB_SUBFR {
                (
                    &SILK_CB_LAGS_STAGE3[k],
                    &SILK_LAG_RANGE_STAGE3[complexity][k],
                    SILK_NB_CBK_SEARCHS_STAGE3[complexity],
                )
            } else {
                static LAG_RANGE_10MS: [[i8; 2]; 2] = [[-3, 7], [-2, 7]];
                (
                    &SILK_CB_LAGS_STAGE3_10_MS[k],
                    &LAG_RANGE_10MS[k],
                    PE_NB_CBKS_STAGE3_10MS,
                )
            };

        let lag_low = lag_range_ptr_curr[0] as i32;
        let lag_high = lag_range_ptr_curr[1] as i32;
        let lag_counter_total = (lag_high - lag_low + 1) as usize;
        assert!(lag_counter_total <= SCRATCH_SIZE);
        silk_pitch_xcorr(
            &frame[target_ptr_idx..],
            &frame[(target_ptr_idx as i32 - start_lag as i32 - lag_high) as usize..],
            &mut xcorr32,
            sf_length,
            lag_counter_total,
        );
        for j in 0..lag_counter_total {
            scratch_mem[j] = xcorr32[lag_counter_total - 1 - j];
        }

        let delta = lag_range_ptr_curr[0];
        for i in 0..nb_cbk_search {
            let idx = (lag_cb_ptr_curr[i] - delta) as usize;
            for j in 0..PE_NB_STAGE3_LAGS {
                cross_corr_st3[k][i][j] = scratch_mem[idx + j];
            }
        }
        target_ptr_idx += sf_length;
    }
}

fn silk_p_ana_calc_energy_st3(
    energies_st3: &mut [[[i32; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]; PE_MAX_NB_SUBFR],
    frame: &[i16],
    start_lag: usize,
    sf_length: usize,
    nb_subfr: usize,
    complexity: usize,
    fs_khz: usize,
) {
    let mut scratch_mem = [0i32; SCRATCH_SIZE];

    let mut target_ptr_idx = PE_LTP_MEM_LENGTH_MS * (if fs_khz > 8 { fs_khz as usize } else { 8 });
    for k in 0..nb_subfr {
        let (lag_cb_ptr_curr, lag_range_ptr_curr, nb_cbk_search): (&[i8], &[i8; 2], usize) =
            if nb_subfr == PE_MAX_NB_SUBFR {
                (
                    &SILK_CB_LAGS_STAGE3[k],
                    &SILK_LAG_RANGE_STAGE3[complexity][k],
                    SILK_NB_CBK_SEARCHS_STAGE3[complexity],
                )
            } else {
                static LAG_RANGE_10MS: [[i8; 2]; 2] = [[-3, 7], [-2, 7]];
                (
                    &SILK_CB_LAGS_STAGE3_10_MS[k],
                    &LAG_RANGE_10MS[k],
                    PE_NB_CBKS_STAGE3_10MS,
                )
            };

        let basis_ptr_idx =
            (target_ptr_idx as i32 - (start_lag as i32 + lag_range_ptr_curr[0] as i32)) as usize;
        let mut energy =
            silk_inner_prod_aligned(&frame[basis_ptr_idx..], &frame[basis_ptr_idx..], sf_length);
        scratch_mem[0] = energy;

        let lag_diff = (lag_range_ptr_curr[1] - lag_range_ptr_curr[0] + 1) as usize;
        for i in 1..lag_diff {
            energy = energy.wrapping_sub(silk_smulbb(
                frame[basis_ptr_idx + sf_length - i] as i32,
                frame[basis_ptr_idx + sf_length - i] as i32,
            ));
            let back_idx = basis_ptr_idx - i;
            energy = silk_add_sat32(
                energy,
                silk_smulbb(frame[back_idx] as i32, frame[back_idx] as i32),
            );
            scratch_mem[i] = energy;
        }

        let delta = lag_range_ptr_curr[0];
        for i in 0..nb_cbk_search {
            let idx = (lag_cb_ptr_curr[i] - delta) as usize;
            for j in 0..PE_NB_STAGE3_LAGS {
                energies_st3[k][i][j] = scratch_mem[idx + j];
            }
        }
        target_ptr_idx += sf_length;
    }
}
