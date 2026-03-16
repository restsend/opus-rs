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
    /* Validate input sizes to prevent out-of-bounds access */
    /* Need at least 2 samples per iteration (2 * k and 2 * k + 1) */
    /* Also limit n to a reasonable maximum to prevent overflow */
    /* Check n is valid and won't cause index overflow: max index is n-1 */
    if n < 2 || n > 4096 || n > input.len() {
        return;
    }
    /* Additional check: ensure n-1 is valid (won't underflow) */
    if let Some(max_idx) = n.checked_sub(1) {
        if max_idx >= input.len() {
            return;
        }
    } else {
        return;
    }

    let n2 = n >> 1;

    /* Validate output buffers have enough space */
    if n2 > out_l.len() || n2 > out_h.len() {
        return;
    }

    /* Internal variables and state are in Q10 format */
    for k in 0..n2 {
        /* Use get() to safely access elements and prevent panic */
        let idx_even = 2 * k;
        let idx_odd = idx_even + 1;

        /* Skip if indices are out of bounds */
        if idx_even >= input.len() || idx_odd >= input.len() {
            break;
        }

        /* Convert to Q10 */
        let in32_even = (input[idx_even] as i32) << 10;

        /* All-pass section for even input sample */
        let y_even = in32_even - s[0];
        let x_even = silk_smlawb(y_even, y_even, A_FB1_21);
        let out_1 = s[0] + x_even;
        s[0] = in32_even + x_even;

        /* Convert to Q10 */
        let in32_odd = (input[idx_odd] as i32) << 10;

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
