use crate::silk::define::*;
use crate::silk::macros::*;

pub fn silk_vq_wmat_ec(
    ind: &mut i8,
    res_nrg_q15: &mut i32,
    rate_dist_q7: &mut i32,
    gain_q7_out: &mut i32,
    xx_q17: &[i32],
    xx_q17_ptr: usize,
    xx_q17_in: &[i32],
    xx_q17_in_ptr: usize,
    cb_q7: &[[i8; LTP_ORDER]],
    cb_gain_q7: &[u8],
    cl_q5: &[u8],
    subfr_len: i32,
    max_gain_q7: i32,
    l: i32,
) {
    let mut neg_xx_q24 = [0i32; 5];
    let mut sum1_q15: i32;
    let mut sum2_q24: i32;

    neg_xx_q24[0] = -silk_lshift(xx_q17_in[xx_q17_in_ptr + 0], 7);
    neg_xx_q24[1] = -silk_lshift(xx_q17_in[xx_q17_in_ptr + 1], 7);
    neg_xx_q24[2] = -silk_lshift(xx_q17_in[xx_q17_in_ptr + 2], 7);
    neg_xx_q24[3] = -silk_lshift(xx_q17_in[xx_q17_in_ptr + 3], 7);
    neg_xx_q24[4] = -silk_lshift(xx_q17_in[xx_q17_in_ptr + 4], 7);

    *rate_dist_q7 = i32::MAX;
    *res_nrg_q15 = i32::MAX;
    *ind = 0;

    for k in 0..l {
        let gain_tmp_q7 = cb_gain_q7[k as usize] as i32;

        sum1_q15 = 32801;

        let penalty = silk_lshift((gain_tmp_q7 - max_gain_q7).max(0), 11);

        let cb_row = &cb_q7[k as usize];

        sum2_q24 = silk_mla(neg_xx_q24[0], xx_q17[xx_q17_ptr + 1], cb_row[1] as i32);
        sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 2], cb_row[2] as i32);
        sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 3], cb_row[3] as i32);
        sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 4], cb_row[4] as i32);
        sum2_q24 = silk_lshift(sum2_q24, 1);
        sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 0], cb_row[0] as i32);
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[0] as i32);

        sum2_q24 = silk_mla(neg_xx_q24[1], xx_q17[xx_q17_ptr + 7], cb_row[2] as i32);
        sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 8], cb_row[3] as i32);
        sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 9], cb_row[4] as i32);
        sum2_q24 = silk_lshift(sum2_q24, 1);
        sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 6], cb_row[1] as i32);
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[1] as i32);

        sum2_q24 = silk_mla(neg_xx_q24[2], xx_q17[xx_q17_ptr + 13], cb_row[3] as i32);
        sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 14], cb_row[4] as i32);
        sum2_q24 = silk_lshift(sum2_q24, 1);
        sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 12], cb_row[2] as i32);
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[2] as i32);

        sum2_q24 = silk_mla(neg_xx_q24[3], xx_q17[xx_q17_ptr + 19], cb_row[4] as i32);
        sum2_q24 = silk_lshift(sum2_q24, 1);
        sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 18], cb_row[3] as i32);
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[3] as i32);

        sum2_q24 = silk_lshift(neg_xx_q24[4], 1);
        sum2_q24 = silk_mla(sum2_q24, xx_q17[xx_q17_ptr + 24], cb_row[4] as i32);
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[4] as i32);

        if sum1_q15 >= 0 {
            let bits_res_q8 = silk_smulbb(subfr_len, silk_lin2log(sum1_q15 + penalty) - (15 << 7));

            let bits_tot_q8 = bits_res_q8 + ((cl_q5[k as usize] as i32) << (3 - 1));
            if bits_tot_q8 <= *rate_dist_q7 {
                *rate_dist_q7 = bits_tot_q8;
                *res_nrg_q15 = sum1_q15 + penalty;
                *ind = k as i8;
                *gain_q7_out = gain_tmp_q7;
            }
        }
    }
}
