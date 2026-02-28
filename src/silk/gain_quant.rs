use crate::silk::define::*;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::log2lin::silk_log2lin;
use crate::silk::macros::*;
use crate::silk::tables::*;
use crate::silk::tuning_parameters::MAX_SUM_LOG_GAIN_dB;
use crate::silk::vq_wmat_ec::silk_vq_wmat_ec;

pub const OFFSET: i32 = (MIN_QGAIN_DB * 128) / 6 + 16 * 128;
pub const SCALE_Q16: i32 =
    (65536 * (N_LEVELS_QGAIN - 1)) / (((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128) / 6);
// INV_SCALE_Q16 = 65536 * ((MAX_QGAIN_DB - MIN_QGAIN_DB) * 128 / 6) / (N_LEVELS_QGAIN - 1)
// = 65536 * 1834 / 63 = 1907825
pub const INV_SCALE_Q16: i32 = 1907825;

/* Gain scalar quantization with hysteresis, uniform on log scale */
pub fn silk_gains_quant(
    ind: &mut [i8; MAX_NB_SUBFR], /* O    gain indices                                */
    gain_q16: &mut [i32; MAX_NB_SUBFR], /* I/O  gains (quantized out)                       */
    prev_ind: &mut i8,            /* I/O  last index in previous frame                */
    conditional: i32,             /* I    first gain is delta coded if 1              */
    nb_subfr: usize,              /* I    number of subframes                         */
) {
    let mut double_step_size_threshold: i32;

    for k in 0..nb_subfr {
        /* Convert to log scale, scale, floor() */
        ind[k] = silk_smulwb(SCALE_Q16, silk_lin2log(gain_q16[k]) - OFFSET) as i8;

        /* Round towards previous quantized gain (hysteresis) */
        if ind[k] < *prev_ind {
            ind[k] += 1;
        }
        ind[k] = silk_limit_int(ind[k] as i32, 0, N_LEVELS_QGAIN - 1) as i8;

        /* Compute delta indices and limit */
        if k == 0 && conditional == 0 {
            /* Full index */
            ind[k] = silk_limit_int(
                ind[k] as i32,
                (*prev_ind as i32) + MIN_DELTA_GAIN_QUANT,
                N_LEVELS_QGAIN - 1,
            ) as i8;
            *prev_ind = ind[k];
        } else {
            /* Delta index */
            ind[k] = ind[k] - *prev_ind;

            /* Double the quantization step size for large gain increases, so that the max gain level can be reached */
            double_step_size_threshold =
                2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + (*prev_ind as i32);
            if (ind[k] as i32) > double_step_size_threshold {
                ind[k] = (double_step_size_threshold
                    + silk_rshift((ind[k] as i32) - double_step_size_threshold + 1, 1))
                    as i8;
            }

            ind[k] =
                silk_limit_int(ind[k] as i32, MIN_DELTA_GAIN_QUANT, MAX_DELTA_GAIN_QUANT) as i8;

            /* Accumulate deltas */
            if (ind[k] as i32) > double_step_size_threshold {
                *prev_ind = ((*prev_ind as i32) + silk_lshift(ind[k] as i32, 1)
                    - double_step_size_threshold) as i8;
                *prev_ind = silk_min_int(*prev_ind as i32, N_LEVELS_QGAIN - 1) as i8;
            } else {
                *prev_ind = ((*prev_ind as i32) + (ind[k] as i32)) as i8;
            }

            /* Shift to make non-negative */
            ind[k] -= MIN_DELTA_GAIN_QUANT as i8;
        }

        /* Scale and convert to linear scale */
        gain_q16[k] = silk_log2lin(silk_min_32(
            silk_smulwb(INV_SCALE_Q16, *prev_ind as i32) + OFFSET,
            3967,
        )); /* 3967 = 31 in Q7 */
    }
}

/* Gains scalar dequantization, uniform on log scale */
pub fn silk_gains_dequant(
    gain_q16: &mut [i32; MAX_NB_SUBFR], /* O    quantized gains                             */
    ind: &[i8; MAX_NB_SUBFR],           /* I    gain indices                                */
    prev_ind: &mut i8,                  /* I/O  last index in previous frame                */
    conditional: i32,                   /* I    first gain is delta coded if 1              */
    nb_subfr: usize,                    /* I    number of subframes                         */
) {
    let mut double_step_size_threshold: i32;
    let mut ind_tmp: i32;

    for k in 0..nb_subfr {
        if k == 0 && conditional == 0 {
            /* Gain index is not allowed to go down more than 16 steps (~21.8 dB) */
            *prev_ind = std::cmp::max(ind[k] as i32, (*prev_ind as i32) - 16) as i8;
        } else {
            /* Delta index: add MIN_DELTA_GAIN_QUANT to convert back from stored index to delta value */
            ind_tmp = (ind[k] as i32) + MIN_DELTA_GAIN_QUANT;

            /* Double the quantization step size for large gain increases, so that the max gain level can be reached */
            double_step_size_threshold =
                2 * MAX_DELTA_GAIN_QUANT - N_LEVELS_QGAIN + (*prev_ind as i32);
            if ind_tmp > double_step_size_threshold {
                *prev_ind = ((*prev_ind as i32) + silk_lshift(ind_tmp, 1)
                    - double_step_size_threshold) as i8;
            } else {
                *prev_ind = ((*prev_ind as i32) + ind_tmp) as i8;
            }
        }
        *prev_ind = silk_limit_int(*prev_ind as i32, 0, N_LEVELS_QGAIN - 1) as i8;

        /* Scale and convert to linear scale */
        gain_q16[k] = silk_log2lin(silk_min_32(
            silk_smulwb(INV_SCALE_Q16, *prev_ind as i32) + OFFSET,
            3967,
        )); /* 3967 = 31 in Q7 */
    }
}

pub fn silk_quant_ltp_gains(
    b_q14: &mut [i16],          /* O    Quantized LTP gains             */
    cbk_index: &mut [i8],       /* O    Codebook Index                  */
    periodicity_index: &mut i8, /* O    Periodicity Index               */
    sum_log_gain_q7: &mut i32,  /* I/O  Cumulative max prediction gain  */
    pred_gain_db_q7: &mut i32,  /* O    LTP prediction gain             */
    xx_q17: &[i32],             /* I    Correlation matrix in Q18       */
    xx_in_q17: &[i32],          /* I    Correlation vector in Q18       */
    subfr_len: i32,             /* I    Number of samples per subframe  */
    nb_subfr: usize,            /* I    Number of subframes             */
    _arch: i32,                 /* I    Run-time architecture           */
) {
    let mut temp_idx = [0i8; MAX_NB_SUBFR];
    let mut res_nrg_q15: i32 = 0;
    let mut res_nrg_total_q15: i32;
    let mut best_res_nrg_total_q15: i32 = 0;
    let mut rate_dist_total_q7: i32;
    let mut rate_dist_subfr_q7: i32 = 0;
    let mut min_rate_dist_q7: i32;
    let mut sum_log_gain_tmp_q7: i32;
    let mut best_sum_log_gain_q7: i32 = 0;
    let mut max_gain_q7: i32;
    let mut gain_q7: i32 = 0;

    min_rate_dist_q7 = i32::MAX;
    for k in 0..3 {
        /* Safety margin for pitch gain control, to take into account factors
        such as state rescaling/rewhitening. */
        let gain_safety = 51; // SILK_FIX_CONST( 0.4, 7 )

        let cl_ptr = SILK_LTP_GAIN_BITS_Q5_PTRS[k];
        let cbk_ptr = SILK_LTP_VQ_PTRS_Q7[k];
        let cbk_gain_ptr = SILK_LTP_VQ_GAIN_PTRS_Q7[k];
        let cbk_size = SILK_LTP_VQ_SIZES[k as usize];

        res_nrg_total_q15 = 0;
        rate_dist_total_q7 = 0;
        sum_log_gain_tmp_q7 = *sum_log_gain_q7;
        for j in 0..nb_subfr {
            max_gain_q7 = silk_log2lin(
                (silk_float_to_fixed_q7(MAX_SUM_LOG_GAIN_dB / 6.0) - sum_log_gain_tmp_q7) + 896, // SILK_FIX_CONST( 7, 7 )
            ) - gain_safety;

            silk_vq_wmat_ec(
                &mut temp_idx[j],
                &mut res_nrg_q15,
                &mut rate_dist_subfr_q7,
                &mut gain_q7,
                xx_q17,
                j * LTP_ORDER * LTP_ORDER,
                xx_in_q17,
                j * LTP_ORDER,
                cbk_ptr,
                cbk_gain_ptr,
                cl_ptr,
                subfr_len,
                max_gain_q7,
                cbk_size,
            );

            res_nrg_total_q15 = silk_add_pos_sat32(res_nrg_total_q15, res_nrg_q15);
            rate_dist_total_q7 = silk_add_pos_sat32(rate_dist_total_q7, rate_dist_subfr_q7);
            sum_log_gain_tmp_q7 = silk_add_pos_sat32(
                sum_log_gain_tmp_q7,
                silk_lin2log(gain_q7 + gain_safety) - 1920, // SILK_FIX_CONST( 15, 7 )
            );
        }

        if rate_dist_total_q7 <= min_rate_dist_q7 {
            min_rate_dist_q7 = rate_dist_total_q7;
            best_res_nrg_total_q15 = res_nrg_total_q15;
            best_sum_log_gain_q7 = sum_log_gain_tmp_q7;
            *periodicity_index = k as i8;
            for i in 0..nb_subfr {
                cbk_index[i] = temp_idx[i];
            }
        }
    }

    *sum_log_gain_q7 = best_sum_log_gain_q7;

    /* Fill out b_Q14 */
    for j in 0..nb_subfr {
        let cbk_ptr = &SILK_LTP_VQ_PTRS_Q7[*periodicity_index as usize][cbk_index[j] as usize];
        for i in 0..LTP_ORDER {
            b_q14[j * LTP_ORDER + i] = silk_lshift(cbk_ptr[i] as i32, 7) as i16;
        }
    }

    /* Prediction gain */
    *pred_gain_db_q7 = silk_smulbb(
        -3,
        silk_lin2log(best_res_nrg_total_q15) - 1920, // SILK_FIX_CONST( 15, 7 )
    );
}

fn silk_float_to_fixed_q7(f: f32) -> i32 {
    (f * 128.0 + 0.5) as i32
}

/// Compute unique identifier of gains vector.
/// Equivalent to C `silk_gains_ID()` in gain_quant.c.
pub fn silk_gains_id(ind: &[i8; MAX_NB_SUBFR], nb_subfr: i32) -> i32 {
    let mut gains_id: i32 = 0;
    for k in 0..nb_subfr as usize {
        gains_id = (ind[k] as i32) + (gains_id << 8);
    }
    gains_id
}
