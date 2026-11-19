//! The quantizer schedule of the DRED latents (libopus `dnn/dred_coding.c`).

/// `compute_quantizer`: the quantizer level of latent chunk `i` for the
/// first level `q0`, the step index `dq` (0 to 7) and the largest level
/// `qmax`.
pub fn compute_quantizer(q0: i32, dq: i32, qmax: i32, i: i32) -> i32 {
    const DQ_TABLE: [i32; 8] = [0, 2, 3, 4, 6, 8, 12, 16];
    let quant = q0 + (DQ_TABLE[dq as usize] * i + 8) / 16;
    if quant > qmax { qmax } else { quant }
}
