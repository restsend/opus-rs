use crate::silk::macros::*;

const SIGM_LUT_SLOPE_Q10: [i32; 6] = [237, 153, 73, 30, 12, 7];
const SIGM_LUT_POS_Q15: [i32; 6] = [16384, 23955, 28861, 31213, 32178, 32548];
const SIGM_LUT_NEG_Q15: [i32; 6] = [16384, 8812, 3906, 1554, 589, 219];

pub fn silk_sigm_q15(in_q5: i32) -> i32 {
    let ind: usize;

    if in_q5 < 0 {
        let in_q5_abs = -in_q5;
        if in_q5_abs >= 6 * 32 {
            0
        } else {
            ind = silk_rshift(in_q5_abs, 5) as usize;
            SIGM_LUT_NEG_Q15[ind] - silk_smulbb(SIGM_LUT_SLOPE_Q10[ind], in_q5_abs & 0x1F)
        }
    } else {
        if in_q5 >= 6 * 32 {
            32767
        } else {
            ind = silk_rshift(in_q5, 5) as usize;
            SIGM_LUT_POS_Q15[ind] + silk_smulbb(SIGM_LUT_SLOPE_Q10[ind], in_q5 & 0x1F)
        }
    }
}
