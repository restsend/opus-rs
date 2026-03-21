use crate::silk::define::{MAX_LPC_ORDER, NLSF_QUANT_LEVEL_ADJ};
use crate::silk::macros::{silk_div32_16, silk_limit, silk_smlawb, silk_smulbb};
use crate::silk::nlsf::silk_nlsf_stabilize;
use crate::silk::nlsf_unpack::silk_nlsf_unpack;
use crate::silk::structs::NLSFCodebook;

fn silk_nlsf_residual_dequant(
    x_q10: &mut [i16],
    indices: &[i8],
    pred_coef_q8: &[u8],
    quant_step_size_q16: i32,
    order: i16,
) {
    let mut out_q10: i32 = 0;
    for i in (0..order as usize).rev() {
        let pred_q10 = silk_smulbb(out_q10, pred_coef_q8[i] as i32) >> 8;
        let mut current_out_q10 = (indices[i] as i32) << 10;
        if current_out_q10 > 0 {
            current_out_q10 -= NLSF_QUANT_LEVEL_ADJ;
        } else if current_out_q10 < 0 {
            current_out_q10 += NLSF_QUANT_LEVEL_ADJ;
        }

        out_q10 = silk_smlawb(pred_q10, current_out_q10, quant_step_size_q16);
        x_q10[i] = out_q10 as i16;
    }
}

pub fn silk_nlsf_decode(p_nlsf_q15: &mut [i16], nlsf_indices: &[i8], ps_nlsf_cb: &NLSFCodebook) {
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    let mut ec_ix = [0i16; MAX_LPC_ORDER];
    let mut res_q10 = [0i16; MAX_LPC_ORDER];
    let order_usize = ps_nlsf_cb.order as usize;

    silk_nlsf_unpack(
        &mut ec_ix,
        &mut pred_q8,
        ps_nlsf_cb,
        nlsf_indices[0] as usize,
    );

    silk_nlsf_residual_dequant(
        &mut res_q10,
        &nlsf_indices[1..],
        &pred_q8,
        ps_nlsf_cb.quant_step_size_q16,
        ps_nlsf_cb.order,
    );

    let cb1_index = nlsf_indices[0] as usize;
    let p_cb_element = &ps_nlsf_cb.cb1_nlsf_q8[cb1_index * order_usize..];
    let p_cb_wght_q9 = &ps_nlsf_cb.cb1_wght_q9[cb1_index * order_usize..];

    for i in 0..order_usize {
        let nlsf_q15_tmp = (silk_div32_16((res_q10[i] as i32) << 14, p_cb_wght_q9[i] as i32))
            + ((p_cb_element[i] as i32) << 7);
        p_nlsf_q15[i] = silk_limit(nlsf_q15_tmp, 0, 32767) as i16;
    }

    silk_nlsf_stabilize(
        p_nlsf_q15,
        ps_nlsf_cb.delta_min_q15,
        ps_nlsf_cb.order as usize,
    );
}
