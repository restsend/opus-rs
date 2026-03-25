use crate::silk::define::{
    MAX_LPC_ORDER, NLSF_QUANT_DEL_DEC_STATES, NLSF_QUANT_DEL_DEC_STATES_LOG2, NLSF_QUANT_LEVEL_ADJ,
    NLSF_QUANT_MAX_AMPLITUDE, NLSF_QUANT_MAX_AMPLITUDE_EXT,
};
use crate::silk::macros::{silk_limit, silk_mla, silk_smlabb, silk_smulbb};

pub fn silk_nlsf_del_dec_quant(
    indices: &mut [i8],
    x_q10: &[i16],
    w_q5: &[i16],
    pred_coef_q8: &[u8],
    ec_ix: &[i16],
    ec_rates_q5: &[u8],
    quant_step_size_q16: i32,
    inv_quant_step_size_q6: i16,
    mu_q20: i32,
    order: i16,
) -> i32 {
    let mut n_states: usize = 1;
    let mut ind_tmp: i32;
    let mut ind_min_max: usize;
    let mut ind_max_min: usize;
    let mut in_q10: i32;
    let mut res_q10: i32;
    let mut pred_q10: i32;
    let mut diff_q10: i32;
    let mut rate0_q5: i32;
    let mut rate1_q5: i32;
    let mut out0_q10: i32;
    let mut out1_q10: i32;
    let mut rd_tmp_q25: i32;
    let mut min_q25: i32;
    let mut min_max_q25: i32;
    let mut max_min_q25: i32;

    let mut ind_sort = [0usize; NLSF_QUANT_DEL_DEC_STATES];
    let mut ind = [[0i8; MAX_LPC_ORDER]; NLSF_QUANT_DEL_DEC_STATES];
    let mut prev_out_q10 = [0i16; 2 * NLSF_QUANT_DEL_DEC_STATES];
    let mut rd_q25 = [0i32; 2 * NLSF_QUANT_DEL_DEC_STATES];
    let mut rd_min_q25 = [0i32; NLSF_QUANT_DEL_DEC_STATES];
    let mut rd_max_q25 = [0i32; NLSF_QUANT_DEL_DEC_STATES];

    let mut out0_q10_table = [0i32; 2 * NLSF_QUANT_MAX_AMPLITUDE_EXT as usize];
    let mut out1_q10_table = [0i32; 2 * NLSF_QUANT_MAX_AMPLITUDE_EXT as usize];

    for i in -(NLSF_QUANT_MAX_AMPLITUDE_EXT)..NLSF_QUANT_MAX_AMPLITUDE_EXT {
        let mut tmp_out0_q10 = i << 10;
        let mut tmp_out1_q10 = tmp_out0_q10 + 1024;
        if i > 0 {
            tmp_out0_q10 -= NLSF_QUANT_LEVEL_ADJ;
            tmp_out1_q10 -= NLSF_QUANT_LEVEL_ADJ;
        } else if i == 0 {
            tmp_out1_q10 -= NLSF_QUANT_LEVEL_ADJ;
        } else if i == -1 {
            tmp_out0_q10 += NLSF_QUANT_LEVEL_ADJ;
        } else {
            tmp_out0_q10 += NLSF_QUANT_LEVEL_ADJ;
            tmp_out1_q10 += NLSF_QUANT_LEVEL_ADJ;
        }
        out0_q10_table[(i + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize] =
            (silk_smulbb(tmp_out0_q10, quant_step_size_q16)) >> 16;
        out1_q10_table[(i + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize] =
            (silk_smulbb(tmp_out1_q10, quant_step_size_q16)) >> 16;
    }

    rd_q25[0] = 0;
    prev_out_q10[0] = 0;

    for i in (0..order as usize).rev() {
        let rates_q5_ptr = &ec_rates_q5[ec_ix[i] as usize..];
        in_q10 = x_q10[i] as i32;

        for j in 0..n_states {
            pred_q10 = silk_smulbb(pred_coef_q8[i] as i32, prev_out_q10[j] as i32) >> 8;
            res_q10 = in_q10 - pred_q10;
            ind_tmp = (silk_smulbb(inv_quant_step_size_q6 as i32, res_q10)) >> 16;
            ind_tmp = silk_limit(
                ind_tmp,
                -NLSF_QUANT_MAX_AMPLITUDE_EXT,
                NLSF_QUANT_MAX_AMPLITUDE_EXT - 1,
            );
            ind[j][i] = ind_tmp as i8;

            out0_q10 = out0_q10_table[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize];
            out1_q10 = out1_q10_table[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE_EXT) as usize];

            out0_q10 += pred_q10;
            out1_q10 += pred_q10;
            prev_out_q10[j] = out0_q10 as i16;
            prev_out_q10[j + n_states] = out1_q10 as i16;

            if ind_tmp + 1 >= NLSF_QUANT_MAX_AMPLITUDE {
                if ind_tmp + 1 == NLSF_QUANT_MAX_AMPLITUDE {
                    rate0_q5 = rates_q5_ptr[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                    rate1_q5 = 280;
                } else {
                    rate0_q5 = silk_smlabb(280 - 43 * NLSF_QUANT_MAX_AMPLITUDE, 43, ind_tmp as i32);
                    rate1_q5 = rate0_q5 + 43;
                }
            } else if ind_tmp <= -NLSF_QUANT_MAX_AMPLITUDE {
                if ind_tmp == -NLSF_QUANT_MAX_AMPLITUDE {
                    rate0_q5 = 280;
                    rate1_q5 =
                        rates_q5_ptr[(ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                } else {
                    rate0_q5 =
                        silk_smlabb(280 - 43 * NLSF_QUANT_MAX_AMPLITUDE, -43, ind_tmp as i32);
                    rate1_q5 = rate0_q5 - 43;
                }
            } else {
                rate0_q5 = rates_q5_ptr[(ind_tmp + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
                rate1_q5 = rates_q5_ptr[(ind_tmp + 1 + NLSF_QUANT_MAX_AMPLITUDE) as usize] as i32;
            }

            rd_tmp_q25 = rd_q25[j];
            diff_q10 = in_q10 - out0_q10;
            rd_q25[j] = silk_smlabb(
                silk_mla(rd_tmp_q25, silk_smulbb(diff_q10, diff_q10), w_q5[i] as i32),
                mu_q20 as i32,
                rate0_q5,
            );
            diff_q10 = in_q10 - out1_q10;
            rd_q25[j + n_states] = silk_smlabb(
                silk_mla(rd_tmp_q25, silk_smulbb(diff_q10, diff_q10), w_q5[i] as i32),
                mu_q20 as i32,
                rate1_q5,
            );
        }

        if n_states <= NLSF_QUANT_DEL_DEC_STATES / 2 {
            for j in 0..n_states {
                ind[j + n_states][i] = ind[j][i] + 1;
            }
            n_states <<= 1;
            for j in n_states..NLSF_QUANT_DEL_DEC_STATES {
                ind[j][i] = ind[j - n_states][i];
            }
        } else {
            for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                if rd_q25[j] > rd_q25[j + NLSF_QUANT_DEL_DEC_STATES] {
                    rd_max_q25[j] = rd_q25[j];
                    rd_min_q25[j] = rd_q25[j + NLSF_QUANT_DEL_DEC_STATES];
                    rd_q25[j] = rd_min_q25[j];
                    rd_q25[j + NLSF_QUANT_DEL_DEC_STATES] = rd_max_q25[j];

                    let tmp = prev_out_q10[j];
                    prev_out_q10[j] = prev_out_q10[j + NLSF_QUANT_DEL_DEC_STATES];
                    prev_out_q10[j + NLSF_QUANT_DEL_DEC_STATES] = tmp;
                    ind_sort[j] = j + NLSF_QUANT_DEL_DEC_STATES;
                } else {
                    rd_min_q25[j] = rd_q25[j];
                    rd_max_q25[j] = rd_q25[j + NLSF_QUANT_DEL_DEC_STATES];
                    ind_sort[j] = j;
                }
            }

            loop {
                min_max_q25 = i32::MAX;
                max_min_q25 = 0;
                ind_min_max = 0;
                ind_max_min = 0;
                for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                    if min_max_q25 > rd_max_q25[j] {
                        min_max_q25 = rd_max_q25[j];
                        ind_min_max = j;
                    }
                    if max_min_q25 < rd_min_q25[j] {
                        max_min_q25 = rd_min_q25[j];
                        ind_max_min = j;
                    }
                }
                if min_max_q25 >= max_min_q25 {
                    break;
                }

                ind_sort[ind_max_min] = ind_sort[ind_min_max] ^ NLSF_QUANT_DEL_DEC_STATES;
                rd_q25[ind_max_min] = rd_q25[ind_min_max + NLSF_QUANT_DEL_DEC_STATES];
                prev_out_q10[ind_max_min] = prev_out_q10[ind_min_max + NLSF_QUANT_DEL_DEC_STATES];
                rd_min_q25[ind_max_min] = 0;
                rd_max_q25[ind_min_max] = i32::MAX;
                ind[ind_max_min] = ind[ind_min_max];
            }

            for j in 0..NLSF_QUANT_DEL_DEC_STATES {
                ind[j][i] += (ind_sort[j] >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;
            }
        }
    }

    ind_tmp = 0;
    min_q25 = i32::MAX;
    for j in 0..2 * NLSF_QUANT_DEL_DEC_STATES {
        if min_q25 > rd_q25[j] {
            min_q25 = rd_q25[j];
            ind_tmp = j as i32;
        }
    }
    for i in 0..order as usize {
        indices[i] = ind[(ind_tmp & (NLSF_QUANT_DEL_DEC_STATES as i32 - 1)) as usize][i];
    }
    indices[0] += (ind_tmp >> NLSF_QUANT_DEL_DEC_STATES_LOG2) as i8;

    min_q25
}
