use crate::silk::macros::*;

pub const SILK_RESAMPLER_DOWN2_0: i32 = 9872;
pub const SILK_RESAMPLER_DOWN2_1: i32 = 39809 - 65536; // -25727

pub const SILK_RESAMPLER_2_3_COEFS_LQ: [i16; 6] = [-2797, -651, 4690, 15537, 23281, 26844];

pub fn silk_resampler_down2(
    s: &mut [i32],   // I/O State vector [2]
    out: &mut [i16], // O Output signal [floor(len/2)]
    input: &[i16],   // I Input signal [len]
    in_len: i32,     // I Number of input samples
) {
    let len2 = in_len >> 1;
    let mut in32: i32;
    let mut out32: i32;
    let mut y: i32;
    let mut x: i32;

    /* Internal variables and state are in Q10 format */
    for k in 0..len2 as usize {
        /* Convert to Q10 */
        in32 = (input[2 * k] as i32) << 10;

        /* All-pass section for even input sample */
        y = in32.wrapping_sub(s[0]);
        x = silk_smlawb(y, y, SILK_RESAMPLER_DOWN2_1 as i32);
        out32 = s[0].wrapping_add(x);
        s[0] = in32.wrapping_add(x);

        /* Convert to Q10 */
        in32 = (input[2 * k + 1] as i32) << 10;

        /* All-pass section for odd input sample, and add to output of previous section */
        y = in32.wrapping_sub(s[1]);
        x = silk_smulwb(y, SILK_RESAMPLER_DOWN2_0 as i32);
        out32 = out32.wrapping_add(s[1]);
        out32 = out32.wrapping_add(x);
        s[1] = in32.wrapping_add(x);

        /* Add, convert back to int16 and store to output */
        out[k] = silk_sat16(silk_rshift_round(out32, 11)) as i16;
    }
}

pub fn silk_resampler_private_ar2(
    s: &mut [i32],      // I/O State vector [2]
    out_q8: &mut [i32], // O Output signal [len]
    input: &[i16],      // I Input signal [len]
    a_q14: &[i16],      // I AR coefficients [2]
    len: i32,           // I Number of samples
) {
    let mut out32: i32;
    for k in 0..len as usize {
        out32 = s[0].wrapping_add((input[k] as i32) << 8);
        s[0] = s[1].wrapping_add(silk_smlawb(out32, out32, a_q14[0] as i32));
        s[1] = silk_smlawb(0, out32, a_q14[1] as i32);
        out_q8[k] = out32;
    }
}

const RESAMPLER_MAX_BATCH_SIZE_IN: i32 = 480;
const ORDER_FIR: usize = 4;

pub fn silk_resampler_down2_3(
    s: &mut [i32],   // I/O State vector [6]
    out: &mut [i16], // O Output signal [floor(2*len/3)]
    input: &[i16],   // I Input signal [len]
    in_len: i32,     // I Number of input samples
) {
    let mut n_samples_in: i32;
    let mut counter: i32;
    let mut res_q6: i32;
    let mut buf = [0i32; (RESAMPLER_MAX_BATCH_SIZE_IN as usize) + ORDER_FIR];
    let mut in_idx = 0;
    let mut out_idx = 0;
    let mut remaining_len = in_len;

    /* Copy buffered samples to start of buffer */
    buf[0..ORDER_FIR].copy_from_slice(&s[0..ORDER_FIR]);

    while remaining_len > 0 {
        n_samples_in = remaining_len.min(RESAMPLER_MAX_BATCH_SIZE_IN);

        /* Second-order AR filter (output in Q8) */
        silk_resampler_private_ar2(
            &mut s[ORDER_FIR..ORDER_FIR + 2],
            &mut buf[ORDER_FIR..ORDER_FIR + n_samples_in as usize],
            &input[in_idx..in_idx + n_samples_in as usize],
            &SILK_RESAMPLER_2_3_COEFS_LQ,
            n_samples_in,
        );

        /* Interpolate filtered signal */
        let mut buf_ptr = 0;
        counter = n_samples_in;
        while counter > 2 {
            /* Inner product */
            res_q6 = silk_smulwb(buf[buf_ptr], SILK_RESAMPLER_2_3_COEFS_LQ[2] as i32);
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_ptr + 1],
                SILK_RESAMPLER_2_3_COEFS_LQ[3] as i32,
            );
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_ptr + 2],
                SILK_RESAMPLER_2_3_COEFS_LQ[5] as i32,
            );
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_ptr + 3],
                SILK_RESAMPLER_2_3_COEFS_LQ[4] as i32,
            );

            /* Scale down, saturate and store in output array */
            out[out_idx] = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;
            out_idx += 1;

            res_q6 = silk_smulwb(buf[buf_ptr + 1], SILK_RESAMPLER_2_3_COEFS_LQ[4] as i32);
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_ptr + 2],
                SILK_RESAMPLER_2_3_COEFS_LQ[5] as i32,
            );
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_ptr + 3],
                SILK_RESAMPLER_2_3_COEFS_LQ[3] as i32,
            );
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_ptr + 4],
                SILK_RESAMPLER_2_3_COEFS_LQ[2] as i32,
            );

            /* Scale down, saturate and store in output array */
            out[out_idx] = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;
            out_idx += 1;

            buf_ptr += 3;
            counter -= 3;
        }

        in_idx += n_samples_in as usize;
        remaining_len -= n_samples_in;

        if remaining_len > 0 {
            /* More iterations to do; copy last part of filtered signal to beginning of buffer */
            for i in 0..ORDER_FIR {
                buf[i] = buf[n_samples_in as usize + i];
            }
        } else {
            /* Copy last part of filtered signal to the state for the next call */
            s[0..ORDER_FIR]
                .copy_from_slice(&buf[n_samples_in as usize..n_samples_in as usize + ORDER_FIR]);
            break;
        }
    }
}
