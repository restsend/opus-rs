use crate::silk::define::*;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::log2lin::silk_log2lin;
use crate::silk::lpc_fit::silk_lpc_fit;
use crate::silk::macros::*;
use crate::silk::nlsf::silk_bwexpander_32;
use crate::silk::sigm::*;
use crate::silk::sigproc_fix::*;
use crate::silk::structs::*;

/* Compute gain to make warped filter coefficients have a zero mean log frequency response on a   */
/* non-warped frequency scale. (So that it can be implemented with a minimum-phase monic filter.) */
pub fn warped_gain(coefs_q24: &[i32], mut lambda_q16: i32, order: usize) -> i32 {
    let mut gain_q24: i32;

    lambda_q16 = -lambda_q16;
    gain_q24 = coefs_q24[order - 1];
    for i in (0..order - 1).rev() {
        gain_q24 = silk_smlawb(coefs_q24[i], gain_q24, lambda_q16 as i32);
    }
    gain_q24 = silk_smlawb(1 << 24, gain_q24, -lambda_q16 as i32);
    silk_inverse32_varq(gain_q24, 40)
}

/* Convert warped filter coefficients to monic pseudo-warped coefficients and limit maximum     */
/* amplitude of monic warped coefficients by using bandwidth expansion on the true coefficients */
pub fn limit_warped_coefs(
    coefs_q24: &mut [i32],
    mut lambda_q16: i32,
    limit_q24: i32,
    order: usize,
) {
    let mut tmp: i32;
    let mut maxabs_q24: i32;
    let mut chirp_q16: i32;
    let mut gain_q16: i32 = 0;
    let mut nom_q16: i32;
    let mut den_q24: i32;
    let limit_q20: i32;
    let mut maxabs_q20: i32;
    let mut ind: usize = 0;

    /* Convert to monic coefficients */
    lambda_q16 = -lambda_q16;
    for i in (1..order).rev() {
        coefs_q24[i - 1] = silk_smlawb(coefs_q24[i - 1], coefs_q24[i], lambda_q16 as i32);
    }
    lambda_q16 = -lambda_q16;
    nom_q16 = silk_smlawb(1 << 16, -lambda_q16, lambda_q16 as i32);
    den_q24 = silk_smlawb(1 << 24, coefs_q24[0], lambda_q16 as i32);
    if den_q24 != 0 {
        gain_q16 = silk_div32_varq(nom_q16, den_q24, 24);
    }
    for i in 0..order {
        coefs_q24[i] = silk_smulww(gain_q16, coefs_q24[i]);
    }

    limit_q20 = limit_q24 >> 4;
    for iter in 0..10 {
        /* Find maximum absolute value */
        maxabs_q24 = -1;
        for i in 0..order {
            tmp = coefs_q24[i].abs();
            if tmp > maxabs_q24 {
                maxabs_q24 = tmp;
                ind = i;
            }
        }
        /* Use Q20 to avoid any overflow when multiplying by (ind + 1) later. */
        maxabs_q20 = maxabs_q24 >> 4;
        if maxabs_q20 <= limit_q20 {
            /* Coefficients are within range - done */
            return;
        }

        /* Convert back to true warped coefficients */
        for i in 1..order {
            coefs_q24[i - 1] = silk_smlawb(coefs_q24[i - 1], coefs_q24[i], lambda_q16 as i32);
        }
        if gain_q16 != 0 {
            gain_q16 = silk_inverse32_varq(gain_q16, 32);
        }
        for i in 0..order {
            coefs_q24[i] = silk_smulww(gain_q16, coefs_q24[i]);
        }

        /* Apply bandwidth expansion */
        // chirp_Q16 = SILK_FIX_CONST(0.99, 16) - silk_DIV32_varQ(
        //     silk_SMULWB(maxabs_Q20 - limit_Q20, silk_SMLABB(0.8_Q10, 0.1_Q10, iter)),
        //     silk_MUL(maxabs_Q20, ind + 1), 22);
        let weight = silk_smlabb(819, 102 as i32, iter as i32); // Q10
        let numerator = silk_smulwb(maxabs_q20 - limit_q20, weight as i32); // Q14
        let denominator = maxabs_q20 * ((ind + 1) as i32); // Q20
        chirp_q16 = 64881 - silk_div32_varq(numerator, denominator, 22);

        silk_bwexpander_32(coefs_q24, order, chirp_q16);

        /* Convert to monic warped coefficients */
        lambda_q16 = -lambda_q16;
        for i in (1..order).rev() {
            coefs_q24[i - 1] = silk_smlawb(coefs_q24[i - 1], coefs_q24[i], lambda_q16 as i32);
        }
        lambda_q16 = -lambda_q16;
        nom_q16 = silk_smlawb(1 << 16, -lambda_q16, lambda_q16 as i32);
        den_q24 = silk_smlawb(1 << 24, coefs_q24[0], lambda_q16 as i32);
        if den_q24 != 0 {
            gain_q16 = silk_div32_varq(nom_q16, den_q24, 24);
        }
        for i in 0..order {
            coefs_q24[i] = silk_smulww(gain_q16, coefs_q24[i]);
        }
    }
}

pub fn silk_noise_shape_analysis_fix(
    ps_enc: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControl,
    pitch_res: &[i16],
    x: &[i16],
) {
    let n_samples: usize;
    let n_segs: usize;
    let mut q_nrg: i32;
    let warping_q16: i32;
    let mut scale: i32 = 0;
    let mut snr_adj_db_q7: i32;
    let mut harm_shape_gain_q16: i32;
    let tilt_q16: i32;
    let mut tmp32: i32;
    let mut nrg: i32 = 0;
    let mut log_energy_q7: i32;
    let mut log_energy_prev_q7: i32 = 0;
    let mut energy_variation_q7: i32;
    let bw_exp_q16: i32;
    let mut strength_q16: i32;
    let mut b_q8: i32;
    let mut auto_corr = [0i32; MAX_SHAPE_LPC_ORDER + 1];
    let mut refl_coef_q16 = [0i32; MAX_SHAPE_LPC_ORDER];
    let mut ar_q24 = [0i32; MAX_SHAPE_LPC_ORDER];
    // Stack buffer: shape_win_length ≤ SUB_FRAME_LENGTH_MS*MAX_FS_KHZ + 2*LA_SHAPE_MAX = 80+160 = 240.
    let mut x_windowed = [0i16; MAX_SUB_FRAME_LENGTH + 2 * LA_SHAPE_MAX];

    /* Point to start of first LPC analysis block */
    // x should be the buffer starting from x[0] - ps_enc.s_cmn.la_shape.
    // We'll use x_ptr_idx to track current position in x.
    let mut x_ptr_idx: usize = 0;

    /****************/
    /* GAIN CONTROL */
    /****************/
    snr_adj_db_q7 = ps_enc.s_cmn.snr_db_q7;

    /* Input quality is the average of the quality in the lowest two VAD bands */
    ps_enc_ctrl.input_quality_q14 =
        (ps_enc.s_cmn.input_quality_bands_q15[0] + ps_enc.s_cmn.input_quality_bands_q15[1]) >> 2;

    /* Coding quality level, between 0.0_Q0 and 1.0_Q0, but in Q14 */
    ps_enc_ctrl.coding_quality_q14 =
        silk_rshift(silk_sigm_q15(silk_rshift_round(snr_adj_db_q7 - 2560, 4)), 1);

    /* Reduce coding SNR during low speech activity */
    if ps_enc.s_cmn.use_cbr == 0 {
        b_q8 = 256 - ps_enc.s_cmn.speech_activity_q8;
        b_q8 = silk_smulwb(b_q8 << 8, b_q8 as i32);
        snr_adj_db_q7 = silk_smlawb(
            snr_adj_db_q7,
            silk_smulbb(-256 >> 5, b_q8),
            silk_smulwb(
                16384 + ps_enc_ctrl.input_quality_q14,
                ps_enc_ctrl.coding_quality_q14,
            ),
        );
    }

    if ps_enc.s_cmn.indices.signal_type == TYPE_VOICED as i8 {
        /* Reduce gains for periodic signals */
        snr_adj_db_q7 = silk_smlawb(snr_adj_db_q7, 512, ps_enc.ltp_corr_q15 as i32);
    } else {
        /* For unvoiced signals and low-quality input, adjust the quality slower than SNR_dB setting */
        snr_adj_db_q7 = silk_smlawb(
            snr_adj_db_q7,
            silk_smlawb(3072, -104858, ps_enc.s_cmn.snr_db_q7 as i32),
            16384 - ps_enc_ctrl.input_quality_q14,
        );
    }

    /*************************/
    /* SPARSENESS PROCESSING */
    /*************************/
    /* Set quantizer offset */
    if ps_enc.s_cmn.indices.signal_type == TYPE_VOICED as i8 {
        ps_enc.s_cmn.indices.quant_offset_type = 0;
    } else {
        /* Sparseness measure, based on relative fluctuations of energy per 2 milliseconds */
        n_samples = (ps_enc.s_cmn.fs_khz << 1) as usize;
        energy_variation_q7 = 0;
        let mut pitch_res_idx = 0;
        n_segs = (SUB_FRAME_LENGTH_MS * ps_enc.s_cmn.nb_subfr as usize) / 2;
        for k in 0..n_segs {
            silk_sum_sqr_shift(
                &mut nrg,
                &mut scale,
                &pitch_res[pitch_res_idx..pitch_res_idx + n_samples],
                n_samples,
            );
            nrg += (n_samples as i32) >> scale;

            log_energy_q7 = silk_lin2log(nrg);
            if k > 0 {
                energy_variation_q7 += (log_energy_q7 - log_energy_prev_q7).abs();
            }
            log_energy_prev_q7 = log_energy_q7;
            pitch_res_idx += n_samples;
        }

        /* Set quantization offset depending on sparseness measure */
        if energy_variation_q7 > 77 * (n_segs as i32 - 1) {
            // 0.6 in Q7 is 77
            ps_enc.s_cmn.indices.quant_offset_type = 0;
        } else {
            ps_enc.s_cmn.indices.quant_offset_type = 1;
        }
    }

    /*******************************/
    /* Control bandwidth expansion */
    /*******************************/
    strength_q16 = silk_smulwb(ps_enc_ctrl.pred_gain_q16, 66);
    bw_exp_q16 = silk_div32_varq(61604, silk_smlaww(65536, strength_q16, strength_q16), 16);

    if ps_enc.s_cmn.warping_q16 > 0 {
        warping_q16 = silk_smlawb(
            ps_enc.s_cmn.warping_q16,
            ps_enc_ctrl.coding_quality_q14,
            2621,
        );
    } else {
        warping_q16 = 0;
    }

    /********************************************/
    /* Compute noise shaping AR coefs and gains */
    /********************************************/
    for k in 0..ps_enc.s_cmn.nb_subfr as usize {
        let flat_part = (ps_enc.s_cmn.fs_khz * 3) as usize;
        let slope_part = (ps_enc.s_cmn.shape_win_length as usize - flat_part) >> 1;

        silk_apply_sine_window(
            &mut x_windowed[0..slope_part],
            &x[x_ptr_idx..],
            1,
            slope_part,
        );
        x_windowed[slope_part..slope_part + flat_part]
            .copy_from_slice(&x[x_ptr_idx + slope_part..x_ptr_idx + slope_part + flat_part]);
        silk_apply_sine_window(
            &mut x_windowed[slope_part + flat_part..slope_part + flat_part + slope_part],
            &x[x_ptr_idx + slope_part + flat_part..],
            2,
            slope_part,
        );

        x_ptr_idx += ps_enc.s_cmn.subfr_length as usize;

        if ps_enc.s_cmn.warping_q16 > 0 {
            silk_warped_autocorrelation_fix(
                &mut auto_corr,
                &mut scale,
                &x_windowed,
                warping_q16,
                ps_enc.s_cmn.shape_win_length as usize,
                ps_enc.s_cmn.shaping_lpc_order as usize,
            );
        } else {
            silk_autocorr(
                &mut auto_corr,
                &mut scale,
                &x_windowed,
                ps_enc.s_cmn.shape_win_length as usize,
                ps_enc.s_cmn.shaping_lpc_order as usize + 1,
            );
        }

        auto_corr[0] = auto_corr[0].wrapping_add((silk_smulwb(auto_corr[0] >> 4, 31)).max(1));

        nrg = silk_schur64(
            &mut refl_coef_q16,
            &auto_corr,
            ps_enc.s_cmn.shaping_lpc_order as usize,
        );

        silk_k2a_q16(
            &mut ar_q24,
            &refl_coef_q16,
            ps_enc.s_cmn.shaping_lpc_order as usize,
        );

        q_nrg = -scale;

        if (q_nrg & 1) != 0 {
            q_nrg -= 1;
            nrg >>= 1;
        }

        tmp32 = silk_sqrt_approx(nrg);
        q_nrg >>= 1; /* range: -6...15 */

        /* silk_LSHIFT_SAT32(tmp32, 16 - Qnrg) */
        let shift = 16 - q_nrg;
        if shift >= 0 {
            let max_val = i32::MAX >> shift;
            if tmp32 > max_val {
                ps_enc_ctrl.gains_q16[k] = i32::MAX;
            } else {
                ps_enc_ctrl.gains_q16[k] = tmp32 << shift;
            }
        } else {
            ps_enc_ctrl.gains_q16[k] = tmp32 >> (-shift);
        }

        if ps_enc.s_cmn.warping_q16 > 0 {
            /* Adjust gain for warping */
            let gain_mult_q16 = warped_gain(
                &ar_q24,
                warping_q16,
                ps_enc.s_cmn.shaping_lpc_order as usize,
            );
            if ps_enc_ctrl.gains_q16[k] < (1 << 14) {
                /* SILK_FIX_CONST(0.25, 16) = 16384 */
                ps_enc_ctrl.gains_q16[k] = silk_smulww(ps_enc_ctrl.gains_q16[k], gain_mult_q16);
            } else {
                ps_enc_ctrl.gains_q16[k] = silk_smulww(
                    silk_rshift_round(ps_enc_ctrl.gains_q16[k], 1),
                    gain_mult_q16,
                );
                if ps_enc_ctrl.gains_q16[k] >= (i32::MAX >> 1) {
                    ps_enc_ctrl.gains_q16[k] = i32::MAX;
                } else {
                    ps_enc_ctrl.gains_q16[k] = ps_enc_ctrl.gains_q16[k] << 1;
                }
            }
        }

        silk_bwexpander_32(
            &mut ar_q24,
            ps_enc.s_cmn.shaping_lpc_order as usize,
            bw_exp_q16,
        );

        if ps_enc.s_cmn.warping_q16 > 0 {
            limit_warped_coefs(
                &mut ar_q24,
                warping_q16,
                67092087,
                ps_enc.s_cmn.shaping_lpc_order as usize,
            );
            for i in 0..ps_enc.s_cmn.shaping_lpc_order as usize {
                ps_enc_ctrl.ar_q13[k * MAX_SHAPE_LPC_ORDER + i] =
                    silk_sat16(silk_rshift_round(ar_q24[i], 11)) as i16;
            }
        } else {
            silk_lpc_fit(
                &mut ps_enc_ctrl.ar_q13[k * MAX_SHAPE_LPC_ORDER
                    ..k * MAX_SHAPE_LPC_ORDER + ps_enc.s_cmn.shaping_lpc_order as usize],
                &mut ar_q24,
                13,
                24,
                ps_enc.s_cmn.shaping_lpc_order as usize,
            );
        }
    }

    /*****************/
    /* Gain tweaking */
    /*****************/
    // gain_mult_Q16 = silk_log2lin(-silk_SMLAWB(-SILK_FIX_CONST(16.0, 7), SNR_adj_dB_Q7, SILK_FIX_CONST(0.16, 16)))
    let gain_mult_q16_val = silk_log2lin(-silk_smlawb(-2048, snr_adj_db_q7, 10486));
    // gain_add_Q16 = silk_log2lin(silk_SMLAWB(SILK_FIX_CONST(16.0, 7), SILK_FIX_CONST(MIN_QGAIN_DB, 7), SILK_FIX_CONST(0.16, 16)))
    // MIN_QGAIN_DB = 2, SILK_FIX_CONST(2, 7) = 256
    let gain_add_q16_val = silk_log2lin(silk_smlawb(2048, 256, 10486));

    for k in 0..ps_enc.s_cmn.nb_subfr as usize {
        ps_enc_ctrl.gains_q16[k] = silk_smulww(ps_enc_ctrl.gains_q16[k], gain_mult_q16_val);
        ps_enc_ctrl.gains_q16[k] = silk_add_pos_sat32(ps_enc_ctrl.gains_q16[k], gain_add_q16_val);
    }

    /************************************************/
    /* Control low-frequency shaping and noise tilt */
    /************************************************/
    strength_q16 = silk_mul(
        64,
        silk_smlawb(4096, 4096, ps_enc.s_cmn.input_quality_bands_q15[0] - 32768),
    );
    strength_q16 = silk_rshift(silk_mul(strength_q16, ps_enc.s_cmn.speech_activity_q8), 8);

    if ps_enc.s_cmn.indices.signal_type == TYPE_VOICED as i8 {
        let fs_khz_inv = silk_div32_16(3277, ps_enc.s_cmn.fs_khz);
        for k in 0..ps_enc.s_cmn.nb_subfr as usize {
            let b_q14 = fs_khz_inv + silk_div32_16(49152, ps_enc_ctrl.pitch_l[k]);
            ps_enc_ctrl.lf_shp_q14[k] =
                (16384 - b_q14 - silk_smulwb(strength_q16, b_q14 as i32)) << 16;
            ps_enc_ctrl.lf_shp_q14[k] |= ((b_q14 - 16384) & 0xFFFF) as i32;
        }
        tilt_q16 = -16384
            - silk_smulwb(
                65536 - 16384,
                silk_smulwb(5872025, ps_enc.s_cmn.speech_activity_q8 as i32) as i32,
            );
    } else {
        let b_q14 = silk_div32_16(21299, ps_enc.s_cmn.fs_khz);
        let lf_high = 16384 - b_q14 - silk_smulwb(strength_q16, silk_smulwb(39322, b_q14 as i32) as i32);
        let lf_low_raw = b_q14 - 16384;
        let lf_low_masked = lf_low_raw & 0xFFFF;
        #[cfg(debug_assertions)]
        if std::env::var("SILK_DEBUG_NSQ").is_ok() {
            eprintln!("  [NSA] unvoiced: b_q14={} strength_q16={}", b_q14, strength_q16);
            eprintln!("  [NSA] lf_high={} lf_low_raw={} lf_low_masked={} packed={:#010x}",
                lf_high, lf_low_raw, lf_low_masked, (lf_high << 16) | lf_low_masked);
        }
        ps_enc_ctrl.lf_shp_q14[0] = lf_high << 16;
        ps_enc_ctrl.lf_shp_q14[0] |= lf_low_masked;
        #[cfg(debug_assertions)]
        if std::env::var("SILK_DEBUG_NSQ").is_ok() {
            eprintln!("  [NSA] lf_shp_q14[0] after assignment = {:#010x}", ps_enc_ctrl.lf_shp_q14[0]);
        }
        for k in 1..ps_enc.s_cmn.nb_subfr as usize {
            ps_enc_ctrl.lf_shp_q14[k] = ps_enc_ctrl.lf_shp_q14[0];
        }
        tilt_q16 = -16384;
    }

    /****************************/
    /* HARMONIC SHAPING CONTROL */
    /****************************/
    if ps_enc.s_cmn.indices.signal_type == TYPE_VOICED as i8 {
        harm_shape_gain_q16 = silk_smlawb(
            19661,
            65536
                - silk_smulwb(
                    262144 - (ps_enc_ctrl.coding_quality_q14 << 4),
                    ps_enc_ctrl.input_quality_q14,
                ),
            13107,
        );
        harm_shape_gain_q16 = silk_smulwb(
            harm_shape_gain_q16 << 1,
            silk_sqrt_approx(ps_enc.ltp_corr_q15 << 15),
        );
    } else {
        harm_shape_gain_q16 = 0;
    }

    /*************************/
    /* Smooth over subframes */
    /*************************/
    // C iterates MAX_NB_SUBFR (4) times always, not nb_subfr
    for k in 0..MAX_NB_SUBFR {
        ps_enc.s_shape.harm_shape_gain_smth_q16 = silk_smlawb(
            ps_enc.s_shape.harm_shape_gain_smth_q16,
            harm_shape_gain_q16 - ps_enc.s_shape.harm_shape_gain_smth_q16,
            26214,
        );
        ps_enc.s_shape.tilt_smth_q16 = silk_smlawb(
            ps_enc.s_shape.tilt_smth_q16,
            tilt_q16 - ps_enc.s_shape.tilt_smth_q16,
            26214,
        );

        ps_enc_ctrl.harm_shape_gain_q14[k] =
            silk_rshift_round(ps_enc.s_shape.harm_shape_gain_smth_q16, 2);
        ps_enc_ctrl.tilt_q14[k] = silk_rshift_round(ps_enc.s_shape.tilt_smth_q16, 2);
    }
}
