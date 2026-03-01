use crate::silk::define::MAX_LPC_ORDER;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::macros::{silk_div32_varq, silk_rshift, silk_smlabb, silk_smulbb};
use crate::silk::nlsf::{silk_nlsf_stabilize, silk_nlsf_vq};
use crate::silk::nlsf_decode::silk_nlsf_decode;
use crate::silk::nlsf_del_dec_quant::silk_nlsf_del_dec_quant;
use crate::silk::nlsf_unpack::silk_nlsf_unpack;
use crate::silk::sort::silk_insertion_sort_increasing;
use crate::silk::structs::NLSFCodebook;

/// NLSF vector encoder
pub fn silk_nlsf_encode(
    nlsf_indices: &mut [i8], /* O    Codebook path vector [ LPC_ORDER + 1 ]      */
    p_nlsf_q15: &mut [i16],  /* I/O  (Un)quantized NLSF vector [ LPC_ORDER ]     */
    ps_nlsf_cb: &NLSFCodebook, /* I    Codebook object                             */
    p_w_q2: &[i16],          /* I    NLSF weight vector [ LPC_ORDER ]            */
    nlsf_mu_q20: i32,        /* I    Rate weight for the RD optimization         */
    n_survivors: usize,      /* I    Max survivors after first stage             */
    signal_type: i32,        /* I    Signal type: 0/1/2                          */
) -> i32 {
    let order = ps_nlsf_cb.order as usize;
    // Stack buffers: n_vectors ≤ 32, n_survivors ≤ 16 (see control_codec.rs).
    let mut err_q24 = [0i32; 32];
    let mut temp_indices1 = [0i32; 16];
    let mut rd_q25 = [0i32; 16];
    let mut temp_indices2 = [0i8; 16 * MAX_LPC_ORDER];

    let mut res_q10 = [0i16; MAX_LPC_ORDER];
    let mut nlsf_tmp_q15: [i16; MAX_LPC_ORDER] = [0; MAX_LPC_ORDER];
    let mut w_adj_q5 = [0i16; MAX_LPC_ORDER];
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    let mut ec_ix = [0i16; MAX_LPC_ORDER];

    /* NLSF stabilization */
    silk_nlsf_stabilize(
        p_nlsf_q15,
        ps_nlsf_cb.delta_min_q15,
        ps_nlsf_cb.order as usize,
    );

    /* First stage: VQ */
    silk_nlsf_vq(
        &mut err_q24,
        p_nlsf_q15,
        ps_nlsf_cb.cb1_nlsf_q8,
        ps_nlsf_cb.cb1_wght_q9,
        ps_nlsf_cb.n_vectors as usize,
        ps_nlsf_cb.order as usize,
    );

    /* Sort the quantization errors */
    silk_insertion_sort_increasing(
        &mut err_q24,
        &mut temp_indices1,
        ps_nlsf_cb.n_vectors as usize,
        n_survivors,
    );

    /* Loop over survivors */
    for s in 0..n_survivors {
        let ind1 = temp_indices1[s] as usize;

        /* Residual after first stage */
        let p_cb_element = &ps_nlsf_cb.cb1_nlsf_q8[ind1 * order..];
        let p_cb_wght_q9 = &ps_nlsf_cb.cb1_wght_q9[ind1 * order..];
        for i in 0..order {
            nlsf_tmp_q15[i] = (p_cb_element[i] as i16) << 7;
            let w_tmp_q9 = p_cb_wght_q9[i] as i32;
            res_q10[i] =
                (silk_smulbb(p_nlsf_q15[i] as i32 - nlsf_tmp_q15[i] as i32, w_tmp_q9) >> 14) as i16;
            w_adj_q5[i] = silk_div32_varq(p_w_q2[i] as i32, w_tmp_q9 * w_tmp_q9, 21) as i16;
        }

        /* Unpack entropy table indices and predictor for current CB1 index */
        silk_nlsf_unpack(&mut ec_ix, &mut pred_q8, ps_nlsf_cb, ind1);

        /* Trellis quantizer */
        rd_q25[s] = silk_nlsf_del_dec_quant(
            &mut temp_indices2[s * MAX_LPC_ORDER..(s + 1) * MAX_LPC_ORDER],
            &res_q10,
            &w_adj_q5,
            &pred_q8,
            &ec_ix,
            ps_nlsf_cb.ec_rates_q5,
            ps_nlsf_cb.quant_step_size_q16,
            ps_nlsf_cb.inv_quant_step_size_q6,
            nlsf_mu_q20,
            ps_nlsf_cb.order,
        );

        /* Add rate for first stage */
        let i_cdf_ptr =
            &ps_nlsf_cb.cb1_icdf[((signal_type >> 1) as usize) * ps_nlsf_cb.n_vectors as usize..];
        let prob_q8 = if ind1 == 0 {
            256 - i_cdf_ptr[ind1] as i32
        } else {
            i_cdf_ptr[ind1 - 1] as i32 - i_cdf_ptr[ind1] as i32
        };
        let bits_q7 = (8 << 7) - silk_lin2log(prob_q8);
        rd_q25[s] = silk_smlabb(rd_q25[s], bits_q7, silk_rshift(nlsf_mu_q20, 2));
    }

    /* Find the lowest rate-distortion error */
    let mut best_index = [0i32; 1];
    silk_insertion_sort_increasing(&mut rd_q25, &mut best_index, n_survivors, 1);
    let best_idx = best_index[0] as usize;

    nlsf_indices[0] = temp_indices1[best_idx] as i8;
    nlsf_indices[1..1 + order].copy_from_slice(
        &temp_indices2[best_idx * MAX_LPC_ORDER..best_idx * MAX_LPC_ORDER + order],
    );

    /* Decode */
    silk_nlsf_decode(p_nlsf_q15, nlsf_indices, ps_nlsf_cb);

    rd_q25[0]
}
