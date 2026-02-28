use crate::silk::macros::*;
use crate::silk::nlsf::silk_bwexpander_32;

pub fn silk_lpc_fit(
    a_qout: &mut [i16],
    a_qin: &mut [i32],
    qout: i32,
    qin: i32,
    d: usize,
) {
    let mut idx = 0;
    let mut maxabs: i32;
    let mut absval: i32;
    let mut chirp_q16: i32;

    /* Limit the maximum absolute value of the prediction coefficients, so that they'll fit in int16 */
    let mut i = 0;
    while i < 10 {
        /* Find maximum absolute value and its index */
        maxabs = 0;
        for k in 0..d {
            absval = a_qin[k].abs();
            if absval > maxabs {
                maxabs = absval;
                idx = k;
            }
        }
        maxabs = silk_rshift_round(maxabs, qin - qout);

        if maxabs > i16::MAX as i32 {
            /* Reduce magnitude of prediction coefficients */
            maxabs = maxabs.min(163838); /* ( silk_int32_MAX >> 14 ) + silk_int16_MAX = 163838 */
            let num = (maxabs - i16::MAX as i32) << 14;
            let den = (maxabs as i64 * (idx + 1) as i64) >> 2;
            chirp_q16 = 65470 - (num as i64 / den) as i32; // 0.999 in Q16 is 65470
            silk_bwexpander_32(a_qin, d, chirp_q16);
        } else {
            break;
        }
        i += 1;
    }

    if i == 10 {
        /* Reached the last iteration, clip the coefficients */
        for k in 0..d {
            a_qout[k] = silk_sat16(silk_rshift_round(a_qin[k], qin - qout)) as i16;
            a_qin[k] = (a_qout[k] as i32) << (qin - qout);
        }
    } else {
        for k in 0..d {
            a_qout[k] = silk_rshift_round(a_qin[k], qin - qout) as i16;
        }
    }
}
