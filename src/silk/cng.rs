//! SILK comfort noise generation (CNG), ported from libopus `silk/CNG.c`.
//!
//! In frames without voice activity the decoder tracks the background's
//! smoothed spectrum and level. While packets are lost it adds noise shaped
//! like that background on top of the concealment, so a loss in background
//! noise continues the background instead of fading it out.

use crate::silk::decoder_structs::{SilkDecoderControl, SilkDecoderState};
use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::nlsf::silk_nlsf2a;

const CNG_BUF_MASK_MAX: usize = 255;
const CNG_GAIN_SMTH_Q16: i32 = 4634;
const CNG_GAIN_SMTH_THRESHOLD_Q16: i32 = 46396;
const CNG_NLSF_SMTH_Q16: i32 = 16348;

/// Reset the comfort noise state (silk_CNG_Reset).
fn silk_cng_reset(ps_dec: &mut SilkDecoderState) {
    let order = ps_dec.lpc_order as usize;
    let step_q15 = silk_div32_16(i16::MAX as i32, order as i32 + 1);
    let mut acc_q15 = 0;
    for nlsf in ps_dec.s_cng.cng_smth_nlsf_q15.iter_mut().take(order) {
        acc_q15 += step_q15;
        *nlsf = acc_q15 as i16;
    }
    ps_dec.s_cng.cng_smth_gain_q16 = 0;
    ps_dec.s_cng.rand_seed = 3176576;
}

/// Update the background estimate, and add comfort noise to a lost frame (silk_CNG).
pub fn silk_cng(
    ps_dec: &mut SilkDecoderState,
    ps_dec_ctrl: &SilkDecoderControl,
    frame: &mut [i16],
    length: usize,
) {
    if ps_dec.fs_khz != ps_dec.s_cng.fs_khz {
        silk_cng_reset(ps_dec);
        ps_dec.s_cng.fs_khz = ps_dec.fs_khz;
    }
    let order = ps_dec.lpc_order as usize;
    let nb_subfr = ps_dec.nb_subfr as usize;
    let subfr_length = ps_dec.subfr_length as usize;
    let cng = &mut ps_dec.s_cng;

    if ps_dec.loss_cnt == 0 && ps_dec.prev_signal_type == TYPE_NO_VOICE_ACTIVITY {
        // Smooth the NLSFs.
        for i in 0..order {
            let delta = ps_dec.prev_nlsf_q15[i] as i32 - cng.cng_smth_nlsf_q15[i] as i32;
            cng.cng_smth_nlsf_q15[i] =
                (cng.cng_smth_nlsf_q15[i] as i32 + silk_smulwb(delta, CNG_NLSF_SMTH_Q16)) as i16;
        }
        // Keep the excitation of the subframe with the highest gain.
        let mut max_gain_q16 = 0;
        let mut subfr = 0;
        for i in 0..nb_subfr {
            if ps_dec_ctrl.gains_q16[i] > max_gain_q16 {
                max_gain_q16 = ps_dec_ctrl.gains_q16[i];
                subfr = i;
            }
        }
        cng.cng_exc_buf_q14
            .copy_within(0..(nb_subfr - 1) * subfr_length, subfr_length);
        cng.cng_exc_buf_q14[..subfr_length]
            .copy_from_slice(&ps_dec.exc_q14[subfr * subfr_length..(subfr + 1) * subfr_length]);
        // Smooth the gains, following a drop of more than 3 dB at once.
        for i in 0..nb_subfr {
            let gain = ps_dec_ctrl.gains_q16[i];
            cng.cng_smth_gain_q16 += silk_smulwb(gain - cng.cng_smth_gain_q16, CNG_GAIN_SMTH_Q16);
            if silk_smulww(cng.cng_smth_gain_q16, CNG_GAIN_SMTH_THRESHOLD_Q16) > gain {
                cng.cng_smth_gain_q16 = gain;
            }
        }
    }

    if ps_dec.loss_cnt == 0 {
        cng.cng_synth_state[..order].fill(0);
        return;
    }

    // The noise level: what the background had beyond the concealment's noise.
    let smultt = |a: i32, b: i32| (a >> 16).wrapping_mul(b >> 16);
    let mut gain_q16 = silk_smulww(
        ps_dec.s_plc.rand_scale_q14 as i32,
        ps_dec.s_plc.prev_gain_q16[1],
    );
    if gain_q16 >= (1 << 21) || cng.cng_smth_gain_q16 > (1 << 23) {
        gain_q16 = smultt(gain_q16, gain_q16);
        gain_q16 = silk_sub_lshift32(
            smultt(cng.cng_smth_gain_q16, cng.cng_smth_gain_q16),
            gain_q16,
            5,
        );
        gain_q16 = silk_sqrt_approx(gain_q16).wrapping_shl(16);
    } else {
        gain_q16 = silk_smulww(gain_q16, gain_q16);
        gain_q16 = silk_sub_lshift32(
            silk_smulww(cng.cng_smth_gain_q16, cng.cng_smth_gain_q16),
            gain_q16,
            5,
        );
        gain_q16 = silk_sqrt_approx(gain_q16).wrapping_shl(8);
    }
    let gain_q10 = silk_rshift(gain_q16, 6);

    // Excitation: random picks from the stored background excitation.
    let mut sig_q14 = [0i32; MAX_FRAME_LENGTH + MAX_LPC_ORDER];
    let mut exc_mask = CNG_BUF_MASK_MAX;
    while exc_mask > length {
        exc_mask >>= 1;
    }
    let mut seed = cng.rand_seed;
    for sample in sig_q14[MAX_LPC_ORDER..MAX_LPC_ORDER + length].iter_mut() {
        seed = silk_rand(seed);
        *sample = cng.cng_exc_buf_q14[(silk_rshift(seed, 24) as usize) & exc_mask];
    }
    cng.rand_seed = seed;

    let mut a_q12 = [0i16; MAX_LPC_ORDER];
    silk_nlsf2a(&mut a_q12, &cng.cng_smth_nlsf_q15[..order], order);

    // Shape it with the background's spectrum and add it.
    sig_q14[..MAX_LPC_ORDER].copy_from_slice(&cng.cng_synth_state);
    for i in 0..length {
        let mut lpc_pred_q10 = silk_rshift(order as i32, 1);
        for j in 0..order {
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                sig_q14[MAX_LPC_ORDER + i - j - 1],
                a_q12[j] as i32,
            );
        }
        sig_q14[MAX_LPC_ORDER + i] = silk_add_sat32(
            sig_q14[MAX_LPC_ORDER + i],
            silk_lshift_sat32(lpc_pred_q10, 4),
        );
        let noise = silk_sat16(silk_rshift_round(
            silk_smulww(sig_q14[MAX_LPC_ORDER + i], gain_q10),
            8,
        )) as i16;
        frame[i] = silk_add_sat16(frame[i], noise);
    }
    cng.cng_synth_state
        .copy_from_slice(&sig_q14[length..length + MAX_LPC_ORDER]);
}
