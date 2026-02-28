use crate::silk::macros::*;

/* Coefficients for 2-band filter bank based on first-order allpass filters */
const A_FB1_20: i32 = 5394 << 1;
const A_FB1_21: i32 = -24290; /* (opus_int16)(20623 << 1) */

/* Split signal into two decimated bands using first-order allpass filters */
pub fn silk_ana_filt_bank_1(
    input: &[i16],          /* I    Input signal [N]                                            */
    s: &mut [i32],          /* I/O  State vector [2]                                            */
    out_l: &mut [i16],      /* O    Low band [N/2]                                              */
    out_h: &mut [i16],      /* O    High band [N/2]                                             */
    n: usize                /* I    Number of input samples                                     */
) {
    let n2 = n >> 1;
    
    /* Internal variables and state are in Q10 format */
    for k in 0..n2 {
        /* Convert to Q10 */
        let in32_even = (input[2 * k] as i32) << 10;

        /* All-pass section for even input sample */
        let y_even = in32_even - s[0];
        let x_even = silk_smlawb(y_even, y_even, A_FB1_21);
        let out_1 = s[0] + x_even;
        s[0] = in32_even + x_even;

        /* Convert to Q10 */
        let in32_odd = (input[2 * k + 1] as i32) << 10;

        /* All-pass section for odd input sample, and add to output of previous section */
        let y_odd = in32_odd - s[1];
        let x_odd = silk_smulwb(y_odd, A_FB1_20);
        let out_2 = s[1] + x_odd;
        s[1] = in32_odd + x_odd;

        /* Add/subtract, convert back to int16 and store to output */
        out_l[k] = silk_sat16(((out_2 + out_1) + 1024) >> 11) as i16;
        out_h[k] = silk_sat16(((out_2 - out_1) + 1024) >> 11) as i16;
    }
}
