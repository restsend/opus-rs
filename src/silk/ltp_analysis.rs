use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::sigproc_fix::*;

pub fn silk_find_ltp_fix(
    xxltp_q17_matrix: &mut [i32],
    xxltp_q17_vector: &mut [i32],
    r_buf: &[i16],
    r_frame_idx: usize,
    lag: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
    _arch: i32,
) {
    let mut xx: i32 = 0;
    let mut nrg: i32 = 0;
    let mut temp: i32;
    let mut xx_shifts: i32 = 0;
    let mut xx_shifts_matrix: i32 = 0;
    let mut xx_shifts_vector: i32;
    let mut extra_shifts: i32;
    let mut r_ptr_idx = r_frame_idx;

    for k in 0..nb_subfr {
        let lag_ptr_idx = r_ptr_idx - (lag[k] as usize + LTP_ORDER / 2);

        silk_sum_sqr_shift(
            &mut xx,
            &mut xx_shifts,
            &r_buf[r_ptr_idx..],
            subfr_length + LTP_ORDER,
        );

        let xxlp_ptr = &mut xxltp_q17_matrix[k * LTP_ORDER * LTP_ORDER..];
        silk_corr_matrix_fix(
            &r_buf[lag_ptr_idx..],
            subfr_length,
            LTP_ORDER,
            xxlp_ptr,
            &mut nrg,
            &mut xx_shifts_matrix,
        );

        extra_shifts = xx_shifts - xx_shifts_matrix;
        if extra_shifts > 0 {
            xx_shifts_vector = xx_shifts;
            for i in 0..(LTP_ORDER * LTP_ORDER) {
                xxlp_ptr[i] = silk_rshift32(xxlp_ptr[i], extra_shifts);
            }
            nrg = silk_rshift32(nrg, extra_shifts);
        } else if extra_shifts < 0 {
            xx_shifts_vector = xx_shifts_matrix;
            xx = silk_rshift32(xx, -extra_shifts);
        } else {
            xx_shifts_vector = xx_shifts;
        }

        let xxlp_vec_ptr = &mut xxltp_q17_vector[k * LTP_ORDER..];
        silk_corr_vector_fix(
            &r_buf[lag_ptr_idx..],
            &r_buf[r_ptr_idx..],
            subfr_length,
            LTP_ORDER,
            xxlp_vec_ptr,
            xx_shifts_vector,
        );

        temp = silk_smlawb(1, nrg, 1966);
        temp = temp.max(1).max(xx);

        for i in 0..(LTP_ORDER * LTP_ORDER) {
            xxlp_ptr[i] = (((xxlp_ptr[i] as i64) << 17) / temp as i64) as i32;
        }
        for i in 0..LTP_ORDER {
            xxlp_vec_ptr[i] = (((xxlp_vec_ptr[i] as i64) << 17) / temp as i64) as i32;
        }

        r_ptr_idx += subfr_length;
    }
}

pub fn silk_ltp_analysis_filter_fix(
    ltp_res: &mut [i16],
    x: &[i16],
    x_base_idx: usize,
    b_q14: &[i16],
    pitch_l: &[i32],
    inv_gain_q16: &[i32],
    subfr_length: usize,
    nb_subfr: usize,
    pre_length: usize,
) {
    let mut x_ptr_idx = x_base_idx;
    let mut ltp_res_ptr_idx = 0;

    for k in 0..nb_subfr {
        let valid_pitch = pitch_l[k].max(2);
        let x_lag_ptr_idx = x_ptr_idx as isize - valid_pitch as isize;

        let btmp_q14 = [
            b_q14[k * LTP_ORDER],
            b_q14[k * LTP_ORDER + 1],
            b_q14[k * LTP_ORDER + 2],
            b_q14[k * LTP_ORDER + 3],
            b_q14[k * LTP_ORDER + 4],
        ];

        for i in 0..(subfr_length + pre_length) {
            let idx = x_ptr_idx + i;
            let res_val = if idx < x.len() { x[idx] } else { 0 };

            let get_x = |offset: isize| -> i16 {
                let idx = x_lag_ptr_idx + i as isize + offset;
                if idx >= 0 && (idx as usize) < x.len() {
                    x[idx as usize]
                } else {
                    0
                }
            };
            let mut ltp_est: i32 = silk_smulbb(get_x(2) as i32, btmp_q14[0] as i32);
            ltp_est = silk_smlabb(ltp_est, get_x(1) as i32, btmp_q14[1] as i32);
            ltp_est = silk_smlabb(ltp_est, get_x(0) as i32, btmp_q14[2] as i32);
            ltp_est = silk_smlabb(ltp_est, get_x(-1) as i32, btmp_q14[3] as i32);
            ltp_est = silk_smlabb(ltp_est, get_x(-2) as i32, btmp_q14[4] as i32);

            let ltp_est_q0 = (ltp_est + 8192) >> 14;

            let res_q0 = silk_sat16(res_val as i32 - ltp_est_q0) as i16;

            ltp_res[ltp_res_ptr_idx + i] = silk_smulwb(inv_gain_q16[k], res_q0 as i32) as i16;
        }

        ltp_res_ptr_idx += subfr_length + pre_length;
        x_ptr_idx += subfr_length;
    }
}
