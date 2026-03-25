use crate::silk::macros::*;

pub fn silk_log2lin(in_log_q7: i32) -> i32 {
    let out: i32;
    let frac_q7: i32;

    if in_log_q7 < 0 {
        return 0;
    } else if in_log_q7 >= 3967 {
        return i32::MAX;
    }

    out = 1 << (in_log_q7 >> 7);
    frac_q7 = in_log_q7 & 0x7F;
    if in_log_q7 < 2048 {
        let tmp = silk_smulbb(frac_q7, 128 - frac_q7);
        let val = silk_smlawb(frac_q7, tmp, -174);
        out.wrapping_add(((out as i64 * val as i64) >> 7) as i32)
    } else {
        let tmp = silk_smulbb(frac_q7, 128 - frac_q7);
        let val = silk_smlawb(frac_q7, tmp, -174);
        out.wrapping_add(silk_mul(out >> 7, val))
    }
}
