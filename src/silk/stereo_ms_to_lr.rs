use crate::silk::macros::{
    silk_add_lshift32, silk_div32_16, silk_lshift, silk_rshift_round, silk_sat16, silk_smlawb,
    silk_smulbb,
};

const STEREO_INTERP_LEN_MS: i32 = 8;

/// Persistent stereo decoder state (mirrors libopus `stereo_dec_state`).
pub struct StereoDecState {
    pub pred_prev_q13: [i32; 2],
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
}

impl Default for StereoDecState {
    fn default() -> Self {
        Self {
            pred_prev_q13: [0; 2],
            s_mid: [0; 2],
            s_side: [0; 2],
        }
    }
}

/// Convert adaptive Mid/Side representation to Left/Right stereo signal.
/// Ported from libopus `silk/stereo_MS_to_LR.c`.
///
/// `x1` / `x2` are per-channel buffers of `frame_length + 2` samples.
/// Decoded samples occupy indices `[2..=frame_length+1]`; indices `[0..2]`
/// hold the overlap from the previous frame (`s_mid`/`s_side`).
/// After the call, the L/R output is at `x1[1..=frame_length]` and
/// `x2[1..=frame_length]`.
pub fn silk_stereo_ms_to_lr(
    state: &mut StereoDecState,
    x1: &mut [i16],
    x2: &mut [i16],
    pred_q13: &[i32; 2],
    fs_khz: i32,
    frame_length: i32,
) {
    let fl = frame_length as usize;

    // Buffering: swap overlap samples between state and buffer head.
    x1[0] = state.s_mid[0];
    x1[1] = state.s_mid[1];
    x2[0] = state.s_side[0];
    x2[1] = state.s_side[1];

    // Save the last two decoded samples for the next frame's overlap.
    state.s_mid[0] = x1[fl];
    state.s_mid[1] = x1[fl + 1];
    state.s_side[0] = x2[fl];
    state.s_side[1] = x2[fl + 1];

    // Interpolate predictors and add prediction to the side channel.
    let mut pred0_q13 = state.pred_prev_q13[0];
    let mut pred1_q13 = state.pred_prev_q13[1];
    let denom_q16 = silk_div32_16(1 << 16, STEREO_INTERP_LEN_MS * fs_khz);
    let delta0_q13 = silk_rshift_round(
        silk_smulbb(pred_q13[0] - state.pred_prev_q13[0], denom_q16),
        16,
    );
    let delta1_q13 = silk_rshift_round(
        silk_smulbb(pred_q13[1] - state.pred_prev_q13[1], denom_q16),
        16,
    );

    let interp_len = (STEREO_INTERP_LEN_MS * fs_khz) as usize;

    for n in 0..interp_len {
        pred0_q13 += delta0_q13;
        pred1_q13 += delta1_q13;
        // sum = (x1[n] + 2*x1[n+1] + x1[n+2]) << 9   [Q11]
        let sum = silk_lshift(
            silk_add_lshift32(x1[n] as i32 + x1[n + 2] as i32, x1[n + 1] as i32, 1),
            9,
        );
        // sum = x2[n+1]<<8 + sum * pred0_q13 >> 16   [Q8]
        let sum = silk_smlawb(silk_lshift(x2[n + 1] as i32, 8), sum, pred0_q13);
        let sum = silk_smlawb(sum, silk_lshift(x1[n + 1] as i32, 11), pred1_q13);
        x2[n + 1] = silk_sat16(silk_rshift_round(sum, 8)) as i16;
    }

    // Steady-state predictor for the remainder of the frame.
    pred0_q13 = pred_q13[0];
    pred1_q13 = pred_q13[1];
    for n in interp_len..fl {
        let sum = silk_lshift(
            silk_add_lshift32(x1[n] as i32 + x1[n + 2] as i32, x1[n + 1] as i32, 1),
            9,
        );
        let sum = silk_smlawb(silk_lshift(x2[n + 1] as i32, 8), sum, pred0_q13);
        let sum = silk_smlawb(sum, silk_lshift(x1[n + 1] as i32, 11), pred1_q13);
        x2[n + 1] = silk_sat16(silk_rshift_round(sum, 8)) as i16;
    }

    state.pred_prev_q13[0] = pred_q13[0];
    state.pred_prev_q13[1] = pred_q13[1];

    // Convert mid/side to left/right.
    for n in 0..fl {
        let mid = x1[n + 1] as i32;
        let side = x2[n + 1] as i32;
        x1[n + 1] = silk_sat16(mid + side) as i16; // Left
        x2[n + 1] = silk_sat16(mid - side) as i16; // Right
    }
}
