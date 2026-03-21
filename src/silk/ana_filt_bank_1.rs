use crate::silk::macros::*;

const A_FB1_20: i32 = 5394 << 1;
const A_FB1_21: i32 = -24290;

pub fn silk_ana_filt_bank_1(
    input: &[i16],
    s: &mut [i32],
    out_l: &mut [i16],
    out_h: &mut [i16],
    n: usize
) {

    if n < 2 || n > 4096 || n > input.len() {
        return;
    }

    if let Some(max_idx) = n.checked_sub(1) {
        if max_idx >= input.len() {
            return;
        }
    } else {
        return;
    }

    let n2 = n >> 1;

    if n2 > out_l.len() || n2 > out_h.len() {
        return;
    }

    for k in 0..n2 {

        let idx_even = 2 * k;
        let idx_odd = idx_even + 1;

        if idx_even >= input.len() || idx_odd >= input.len() {
            break;
        }

        let in32_even = (input[idx_even] as i32) << 10;

        let y_even = in32_even - s[0];
        let x_even = silk_smlawb(y_even, y_even, A_FB1_21);
        let out_1 = s[0] + x_even;
        s[0] = in32_even + x_even;

        let in32_odd = (input[idx_odd] as i32) << 10;

        let y_odd = in32_odd - s[1];
        let x_odd = silk_smulwb(y_odd, A_FB1_20);
        let out_2 = s[1] + x_odd;
        s[1] = in32_odd + x_odd;

        out_l[k] = silk_sat16(((out_2 + out_1) + 1024) >> 11) as i16;
        out_h[k] = silk_sat16(((out_2 - out_1) + 1024) >> 11) as i16;
    }
}
