use crate::silk::define::NLSF_QUANT_MAX_AMPLITUDE;
use crate::silk::macros::silk_smulbb;
use crate::silk::structs::NLSFCodebook;

/// Unpack predictor values and indices for entropy coding tables
pub fn silk_nlsf_unpack(
    ec_ix: &mut [i16],
    pred_q8: &mut [u8],
    ps_nlsf_cb: &NLSFCodebook,
    cb1_index: usize,
) {
    let order = ps_nlsf_cb.order as usize;
    let ec_sel_ptr = &ps_nlsf_cb.ec_sel[cb1_index * order / 2..];

    for i in (0..order).step_by(2) {
        let entry = ec_sel_ptr[i / 2];
        ec_ix[i] = silk_smulbb(
            ((entry >> 1) & 7) as i32,
            (2 * NLSF_QUANT_MAX_AMPLITUDE + 1) as i32,
        ) as i16;
        pred_q8[i] = ps_nlsf_cb.pred_q8[i + ((entry & 1) as usize) * (order - 1)];

        ec_ix[i + 1] = silk_smulbb(
            ((entry >> 5) & 7) as i32,
            (2 * NLSF_QUANT_MAX_AMPLITUDE + 1) as i32,
        ) as i16;
        pred_q8[i + 1] = ps_nlsf_cb.pred_q8[i + (((entry >> 4) & 1) as usize) * (order - 1) + 1];
    }
}
