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
    let mut sum1_q15: i32;
    let mut sum2_q24: i32;

    let neg_xx_q24: [i32; LTP_ORDER] =
        core::array::from_fn(|i| -silk_lshift(xx_q17_in[xx_q17_in_ptr + i], 7));
    let xx: &[i32; LTP_ORDER * LTP_ORDER] = xx_q17[xx_q17_ptr..xx_q17_ptr + LTP_ORDER * LTP_ORDER]
        .try_into()
        .unwrap();

    *rate_dist_q7 = i32::MAX;
    *res_nrg_q15 = i32::MAX;
    *ind = 0;

    let l = l as usize;
    let rows = cb_q7[..l].iter().zip(&cb_gain_q7[..l]).zip(&cl_q5[..l]);
    for (k, ((cb_row, &gain_tmp_q7), &cl_q5)) in rows.enumerate() {
        let gain_tmp_q7 = gain_tmp_q7 as i32;

        sum1_q15 = 32801;

        let penalty = silk_lshift((gain_tmp_q7 - max_gain_q7).max(0), 11);

        sum2_q24 = silk_mla(neg_xx_q24[0], xx[1], cb_row[1] as i32);
        sum2_q24 = silk_mla(sum2_q24, xx[2], cb_row[2] as i32);
        sum2_q24 = silk_mla(sum2_q24, xx[3], cb_row[3] as i32);
        sum2_q24 = silk_mla(sum2_q24, xx[4], cb_row[4] as i32);
        sum2_q24 = silk_lshift(sum2_q24, 1);
        sum2_q24 = silk_mla(sum2_q24, xx[0], cb_row[0] as i32);
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[0] as i32);

        sum2_q24 = silk_mla(neg_xx_q24[1], xx[7], cb_row[2] as i32);
        sum2_q24 = silk_mla(sum2_q24, xx[8], cb_row[3] as i32);
        sum2_q24 = silk_mla(sum2_q24, xx[9], cb_row[4] as i32);
        sum2_q24 = silk_lshift(sum2_q24, 1);
        sum2_q24 = silk_mla(sum2_q24, xx[6], cb_row[1] as i32);
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[1] as i32);

        sum2_q24 = silk_mla(neg_xx_q24[2], xx[13], cb_row[3] as i32);
        sum2_q24 = silk_mla(sum2_q24, xx[14], cb_row[4] as i32);
        sum2_q24 = silk_lshift(sum2_q24, 1);
        sum2_q24 = silk_mla(sum2_q24, xx[12], cb_row[2] as i32);
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[2] as i32);

        sum2_q24 = silk_mla(neg_xx_q24[3], xx[19], cb_row[4] as i32);
        sum2_q24 = silk_lshift(sum2_q24, 1);
        sum2_q24 = silk_mla(sum2_q24, xx[18], cb_row[3] as i32);
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[3] as i32);

        sum2_q24 = silk_lshift(neg_xx_q24[4], 1);
        sum2_q24 = silk_mla(sum2_q24, xx[24], cb_row[4] as i32);
        sum1_q15 = silk_smlawb(sum1_q15, sum2_q24, cb_row[4] as i32);

        if sum1_q15 >= 0 {
            let bits_res_q8 = silk_smulbb(subfr_len, silk_lin2log(sum1_q15 + penalty) - (15 << 7));

            let bits_tot_q8 = bits_res_q8 + ((cl_q5 as i32) << (3 - 1));
            if bits_tot_q8 <= *rate_dist_q7 {
                *rate_dist_q7 = bits_tot_q8;
                *res_nrg_q15 = sum1_q15 + penalty;
                *ind = k as i8;
                *gain_q7_out = gain_tmp_q7;
            }
        }
    }
}
