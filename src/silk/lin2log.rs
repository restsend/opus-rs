use crate::silk::macros::{silk_mul, silk_smlawb};

/// Approximation of 128 * log2() (very close inverse of silk_log2lin())
/// Convert input to a log scale
pub fn silk_lin2log(in_lin: i32) -> i32 {
    if in_lin <= 0 {
        return 0;
    }

    let lz = in_lin.leading_zeros() as i32;
    // C: frac_Q7 = silk_ROR32(in, 24 - lzeros) & 0x7f;
    // silk_ROR32 handles negative rot by doing left rotation
    let rot = 24 - lz;
    let x = in_lin as u32;
    let frac_q7 = if rot == 0 {
        x & 0x7f
    } else if rot < 0 {
        let m = (-rot) as u32;
        ((x << m) | (x >> (32 - m))) & 0x7f
    } else {
        let r = rot as u32;
        ((x << (32 - r)) | (x >> r)) & 0x7f
    } as i32;

    // Piece-wise parabolic approximation
    let res = silk_smlawb(frac_q7, silk_mul(frac_q7, 128 - frac_q7), 179);
    res + ((31 - lz) << 7)
}
