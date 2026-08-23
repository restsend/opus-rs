//! SILK packet loss concealment (PLC), ported from libopus `silk/PLC.c`.
//!
//! Generates pitch-based extrapolation audio for lost frames and for
//! SILK↔CELT mode-transition bridging.

use crate::silk::decoder_structs::{SilkDecoderControl, SilkDecoderState};
use crate::silk::define::*;
use crate::silk::lpc_analysis::silk_lpc_inverse_pred_gain;
use crate::silk::macros::*;
use crate::silk::sigproc_fix::{silk_bwexpander, silk_lpc_analysis_filter, silk_sum_sqr_shift};

/// Reset PLC state (silk_PLC_Reset).
pub fn silk_plc_reset(ps_dec: &mut SilkDecoderState) {
    ps_dec.s_plc.pitch_l_q8 = silk_lshift(ps_dec.frame_length, 8 - 1);
    ps_dec.s_plc.prev_gain_q16[0] = 1 << 16;
    ps_dec.s_plc.prev_gain_q16[1] = 1 << 16;
    ps_dec.s_plc.subfr_length = 20;
    ps_dec.s_plc.nb_subfr = 2;
}

/// PLC control function (silk_PLC).
pub fn silk_plc(
    ps_dec: &mut SilkDecoderState,
    ps_dec_ctrl: &mut SilkDecoderControl,
    frame: &mut [i16],
    lost: i32,
) {
    if ps_dec.fs_khz != ps_dec.s_plc.fs_khz {
        silk_plc_reset(ps_dec);
        ps_dec.s_plc.fs_khz = ps_dec.fs_khz;
    }

    if lost != 0 {
        // Generate signal
        silk_plc_conceal(ps_dec, ps_dec_ctrl, frame);
        ps_dec.loss_cnt += 1;
    } else {
        // Update state
        silk_plc_update(ps_dec, ps_dec_ctrl);
    }
}

/// Update PLC state from a successfully decoded frame (silk_PLC_update).
fn silk_plc_update(ps_dec: &mut SilkDecoderState, ps_dec_ctrl: &SilkDecoderControl) {
    let mut ltp_gain_q14: i32 = 0;
    let mut temp_ltp_gain_q14: i32;

    ps_dec.prev_signal_type = ps_dec.indices.signal_type as i32;

    if ps_dec.indices.signal_type as i32 == TYPE_VOICED {
        // Find the parameters for the last subframe which contains a pitch pulse.
        let mut j = 0i32;
        while (j * ps_dec.subfr_length) < ps_dec_ctrl.pitch_l[(ps_dec.nb_subfr - 1) as usize] {
            if j == ps_dec.nb_subfr {
                break;
            }
            temp_ltp_gain_q14 = 0;
            for i in 0..LTP_ORDER {
                temp_ltp_gain_q14 += ps_dec_ctrl.ltp_coef_q14
                    [(ps_dec.nb_subfr - 1 - j) as usize * LTP_ORDER + i]
                    as i32;
            }
            if temp_ltp_gain_q14 > ltp_gain_q14 {
                ltp_gain_q14 = temp_ltp_gain_q14;
                for i in 0..LTP_ORDER {
                    ps_dec.s_plc.ltp_coef_q14[i] = ps_dec_ctrl.ltp_coef_q14
                        [(ps_dec.nb_subfr - 1 - j) as usize * LTP_ORDER + i];
                }
                ps_dec.s_plc.pitch_l_q8 =
                    silk_lshift(ps_dec_ctrl.pitch_l[(ps_dec.nb_subfr - 1 - j) as usize], 8);
            }
            j += 1;
        }

        ps_dec.s_plc.ltp_coef_q14.fill(0);
        ps_dec.s_plc.ltp_coef_q14[LTP_ORDER / 2] = ltp_gain_q14 as i16;

        // Limit LT coefs
        if ltp_gain_q14 < V_PITCH_GAIN_START_MIN_Q14 {
            let tmp = silk_lshift(V_PITCH_GAIN_START_MIN_Q14, 10);
            let scale_q10 = silk_div32(tmp, ltp_gain_q14.max(1));
            for i in 0..LTP_ORDER {
                ps_dec.s_plc.ltp_coef_q14[i] = silk_rshift(
                    silk_smulbb(ps_dec.s_plc.ltp_coef_q14[i] as i32, scale_q10),
                    10,
                ) as i16;
            }
        } else if ltp_gain_q14 > V_PITCH_GAIN_START_MAX_Q14 {
            let tmp = silk_lshift(V_PITCH_GAIN_START_MAX_Q14, 14);
            let scale_q14 = silk_div32(tmp, ltp_gain_q14.max(1));
            for i in 0..LTP_ORDER {
                ps_dec.s_plc.ltp_coef_q14[i] = silk_rshift(
                    silk_smulbb(ps_dec.s_plc.ltp_coef_q14[i] as i32, scale_q14),
                    14,
                ) as i16;
            }
        }
    } else {
        ps_dec.s_plc.pitch_l_q8 = silk_lshift(silk_smulbb(ps_dec.fs_khz, 18), 8);
        ps_dec.s_plc.ltp_coef_q14.fill(0);
    }

    // Save LPC coefficients
    for i in 0..ps_dec.lpc_order as usize {
        ps_dec.s_plc.prev_lpc_q12[i] = ps_dec_ctrl.pred_coef_q12[1][i];
    }
    ps_dec.s_plc.prev_ltp_scale_q14 = ps_dec_ctrl.ltp_scale_q14 as i16;

    // Save last two gains
    for i in 0..2 {
        ps_dec.s_plc.prev_gain_q16[i] = ps_dec_ctrl.gains_q16[(ps_dec.nb_subfr - 2) as usize + i];
    }

    ps_dec.s_plc.subfr_length = ps_dec.subfr_length;
    ps_dec.s_plc.nb_subfr = ps_dec.nb_subfr;
}

/// Compute energies of the last two scaled subframes (silk_PLC_energy).
fn silk_plc_energy(
    energy1: &mut i32,
    shift1: &mut i32,
    energy2: &mut i32,
    shift2: &mut i32,
    exc_q14: &[i32],
    prev_gain_q10: &[i32; 2],
    subfr_length: i32,
    nb_subfr: i32,
) {
    let mut exc_buf = [0i16; 2 * 320];
    let subfr = subfr_length as usize;
    for k in 0..2usize {
        let offset = (k + (nb_subfr - 2) as usize) * subfr;
        for i in 0..subfr {
            let v = silk_sat16(silk_rshift(
                silk_smulww(exc_q14[offset + i], prev_gain_q10[k]),
                8,
            ));
            exc_buf[k * subfr + i] = v as i16;
        }
    }
    silk_sum_sqr_shift(energy1, shift1, &exc_buf[..subfr], subfr);
    silk_sum_sqr_shift(energy2, shift2, &exc_buf[subfr..], subfr);
}

/// Generate concealment audio for a lost frame (silk_PLC_conceal).
fn silk_plc_conceal(
    ps_dec: &mut SilkDecoderState,
    ps_dec_ctrl: &mut SilkDecoderControl,
    frame: &mut [i16],
) {
    let mut s_ltp_q14 = [0i32; MAX_FRAME_LENGTH + MAX_LTP_MEM];
    let mut s_ltp = [0i16; MAX_LTP_MEM];
    let mut a_q12 = [0i16; MAX_LPC_ORDER];

    let ps_plc = &mut ps_dec.s_plc;
    let prev_gain_q10: [i32; 2] = [
        silk_rshift(ps_plc.prev_gain_q16[0], 6),
        silk_rshift(ps_plc.prev_gain_q16[1], 6),
    ];

    if ps_dec.first_frame_after_reset != 0 {
        ps_plc.prev_lpc_q12.fill(0);
    }

    let mut energy1 = 0;
    let mut shift1 = 0;
    let mut energy2 = 0;
    let mut shift2 = 0;
    silk_plc_energy(
        &mut energy1,
        &mut shift1,
        &mut energy2,
        &mut shift2,
        &ps_dec.exc_q14,
        &prev_gain_q10,
        ps_dec.subfr_length,
        ps_dec.nb_subfr,
    );

    // Pick the subframe with lowest energy as random noise source.
    let rand_idx: usize = if silk_rshift(energy1, shift2) < silk_rshift(energy2, shift1) {
        ((ps_plc.nb_subfr - 1) * ps_plc.subfr_length - RAND_BUF_SIZE as i32).max(0) as usize
    } else {
        (ps_plc.nb_subfr * ps_plc.subfr_length - RAND_BUF_SIZE as i32).max(0) as usize
    };
    let rand_src = &ps_dec.exc_q14[rand_idx..rand_idx + RAND_BUF_SIZE];

    let mut b_q14 = ps_plc.ltp_coef_q14;
    let mut rand_scale_q14 = ps_plc.rand_scale_q14 as i32;

    // Set up attenuation gains.
    let loss_idx = (NB_ATT - 1).min(ps_dec.loss_cnt as usize);
    let harm_gain_q15 = HARM_ATT_Q15[loss_idx] as i32;
    let mut rand_gain_q15 = if ps_dec.prev_signal_type == TYPE_VOICED {
        PLC_RAND_ATTENUATE_V_Q15[loss_idx] as i32
    } else {
        PLC_RAND_ATTENUATE_UV_Q15[loss_idx] as i32
    };

    // LPC concealment: apply BWE to previous LPC.
    silk_bwexpander(
        &mut ps_plc.prev_lpc_q12,
        ps_dec.lpc_order as usize,
        BWE_COEF_Q16,
    );
    a_q12[..ps_dec.lpc_order as usize]
        .copy_from_slice(&ps_plc.prev_lpc_q12[..ps_dec.lpc_order as usize]);

    // First lost frame.
    if ps_dec.loss_cnt == 0 {
        rand_scale_q14 = 1 << 14;
        if ps_dec.prev_signal_type == TYPE_VOICED {
            for i in 0..LTP_ORDER {
                rand_scale_q14 -= b_q14[i] as i32;
            }
            rand_scale_q14 = rand_scale_q14.max(3277); // 0.2
            rand_scale_q14 = silk_rshift(
                silk_smulbb(rand_scale_q14, ps_plc.prev_ltp_scale_q14 as i32),
                14,
            );
        } else {
            // Reduce random noise for unvoiced frames with high LPC gain.
            let inv_gain_q30 = silk_lpc_inverse_pred_gain(&a_q12, ps_dec.lpc_order as usize);
            let mut down_scale_q30 =
                silk_rshift(1 << 30, LOG2_INV_LPC_GAIN_HIGH_THRES).min(inv_gain_q30);
            down_scale_q30 = silk_rshift(1 << 30, LOG2_INV_LPC_GAIN_LOW_THRES).max(down_scale_q30);
            down_scale_q30 = silk_lshift(down_scale_q30, LOG2_INV_LPC_GAIN_HIGH_THRES);
            rand_gain_q15 = silk_rshift(silk_smulwb(down_scale_q30, rand_gain_q15), 14);
        }
    }

    let mut rand_seed = ps_plc.rand_seed;
    let mut lag = silk_rshift_round(ps_plc.pitch_l_q8, 8);
    let ltp_mem_length = ps_dec.ltp_mem_length as usize;
    let frame_length = ps_dec.frame_length as usize;
    let mut s_ltp_buf_idx = ltp_mem_length;

    // Rewhiten LTP state.
    let idx = (ltp_mem_length as i32 - lag - ps_dec.lpc_order - LTP_ORDER as i32 / 2) as usize;
    // silk_LPC_analysis_filter(&sLTP[idx], &psDec->outBuf[idx], A_Q12, ltp_mem_length - idx, LPC_order)
    silk_lpc_analysis_filter(
        &mut s_ltp[idx..],
        &ps_dec.out_buf[idx..],
        &a_q12,
        ltp_mem_length - idx,
        ps_dec.lpc_order as usize,
        0,
    );
    // Scale LTP state.
    let inv_gain_q30 = silk_inverse32_varq(ps_plc.prev_gain_q16[1], 46).min(i32::MAX >> 1);
    for i in (idx + ps_dec.lpc_order as usize)..ltp_mem_length {
        s_ltp_q14[i] = silk_smulwb(inv_gain_q30, s_ltp[i] as i32);
    }

    // LTP synthesis filtering.
    for _k in 0..ps_dec.nb_subfr as usize {
        let pred_lag_base = s_ltp_buf_idx + LTP_ORDER / 2 - lag as usize;
        for i in 0..ps_dec.subfr_length as usize {
            // Unrolled: sum B_Q14[j] * sLTP_Q14[pred_lag + i - j]
            let mut ltp_pred_q12 = 2i32;
            for j in 0..LTP_ORDER {
                let v = s_ltp_q14[pred_lag_base + i - j];
                ltp_pred_q12 = silk_smlawb(ltp_pred_q12, v, b_q14[j] as i32);
            }

            // Generate LPC excitation.
            rand_seed = silk_rand(rand_seed);
            let ridx = (silk_rshift(rand_seed, 25) as usize) & RAND_BUF_MASK;
            let rv = rand_src[ridx];
            s_ltp_q14[s_ltp_buf_idx] =
                silk_lshift_sat32(silk_smlawb(ltp_pred_q12, rv, rand_scale_q14), 2);
            s_ltp_buf_idx += 1;
        }

        // Gradually reduce LTP gain.
        for j in 0..LTP_ORDER {
            b_q14[j] = silk_rshift(silk_smulbb(harm_gain_q15, b_q14[j] as i32), 15) as i16;
        }
        // Gradually reduce excitation gain.
        rand_scale_q14 = silk_rshift(silk_smulbb(rand_scale_q14, rand_gain_q15), 15);
        // Slowly increase pitch lag.
        ps_plc.pitch_l_q8 = silk_smlawb(ps_plc.pitch_l_q8, ps_plc.pitch_l_q8, PITCH_DRIFT_FAC_Q16);
        ps_plc.pitch_l_q8 = ps_plc
            .pitch_l_q8
            .min(silk_lshift(silk_smulbb(MAX_PITCH_LAG_MS, ps_dec.fs_khz), 8));
        lag = silk_rshift_round(ps_plc.pitch_l_q8, 8);
    }

    // LPC synthesis filtering.
    let lpc_mem_start = ltp_mem_length - MAX_LPC_ORDER;
    s_ltp_q14[lpc_mem_start..lpc_mem_start + MAX_LPC_ORDER].copy_from_slice(&ps_dec.s_lpc_q14_buf);

    let order = ps_dec.lpc_order as usize;
    for i in 0..frame_length {
        let mut lpc_pred_q10 = silk_rshift(order as i32, 1);
        for j in 0..order {
            lpc_pred_q10 = silk_smlawb(
                lpc_pred_q10,
                s_ltp_q14[lpc_mem_start + MAX_LPC_ORDER + i - j - 1],
                a_q12[j] as i32,
            );
        }
        // Add prediction to LPC excitation.
        s_ltp_q14[lpc_mem_start + MAX_LPC_ORDER + i] = silk_add_sat32(
            s_ltp_q14[lpc_mem_start + MAX_LPC_ORDER + i],
            silk_lshift_sat32(lpc_pred_q10, 4),
        );
        // Scale with gain.
        frame[i] = silk_sat16(silk_sat16(silk_rshift_round(
            silk_smulww(
                s_ltp_q14[lpc_mem_start + MAX_LPC_ORDER + i],
                prev_gain_q10[1],
            ),
            8,
        ))) as i16;
    }

    // Save LPC state.
    ps_dec.s_lpc_q14_buf.copy_from_slice(
        &s_ltp_q14[lpc_mem_start + frame_length..lpc_mem_start + frame_length + MAX_LPC_ORDER],
    );

    // Update states.
    ps_plc.rand_seed = rand_seed;
    ps_plc.rand_scale_q14 = rand_scale_q14 as i16;
    for i in 0..MAX_NB_SUBFR {
        ps_dec_ctrl.pitch_l[i] = lag;
    }
}

/// Glue concealed frames with new good received frames (silk_PLC_glue_frames).
pub fn silk_plc_glue_frames(ps_dec: &mut SilkDecoderState, frame: &mut [i16], length: usize) {
    if ps_dec.loss_cnt != 0 {
        let mut energy_shift = 0;
        silk_sum_sqr_shift(
            &mut ps_dec.s_plc.conc_energy,
            &mut energy_shift,
            frame,
            length,
        );
        ps_dec.s_plc.conc_energy_shift = energy_shift;
        ps_dec.s_plc.last_frame_lost = 1;
    } else {
        if ps_dec.s_plc.last_frame_lost != 0 {
            // Calculate residual in decoded signal if last frame was lost.
            let mut energy = 0;
            let mut energy_shift = 0;
            silk_sum_sqr_shift(&mut energy, &mut energy_shift, frame, length);

            // Normalize energies.
            if energy_shift > ps_dec.s_plc.conc_energy_shift {
                ps_dec.s_plc.conc_energy = silk_rshift(
                    ps_dec.s_plc.conc_energy,
                    energy_shift - ps_dec.s_plc.conc_energy_shift,
                );
            } else if energy_shift < ps_dec.s_plc.conc_energy_shift {
                energy = silk_rshift(energy, ps_dec.s_plc.conc_energy_shift - energy_shift);
            }

            // Fade in the energy difference.
            if energy > ps_dec.s_plc.conc_energy {
                let frac_q24;
                let lz = silk_clz32(ps_dec.s_plc.conc_energy) - 1;
                ps_dec.s_plc.conc_energy = silk_lshift(ps_dec.s_plc.conc_energy, lz);
                energy = silk_rshift(energy, (24 - lz).max(0));

                frac_q24 = silk_div32(ps_dec.s_plc.conc_energy, energy.max(1));

                let mut gain_q16 = silk_lshift(silk_sqrt_approx(frac_q24), 4);
                let slope_q16 = silk_div32_16((1 << 16) - gain_q16, length as i32);
                // Make slope 4x steeper to avoid missing onsets after DTX.
                let slope_q16 = silk_lshift(slope_q16, 2);
                for i in 0..length {
                    frame[i] = silk_smulwb(gain_q16, frame[i] as i32) as i16;
                    gain_q16 += slope_q16;
                    if gain_q16 > (1 << 16) {
                        break;
                    }
                }
            }
        }
        ps_dec.s_plc.last_frame_lost = 0;
    }
}

const MAX_LTP_MEM: usize = 320;
