use crate::silk::define::*;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::log2lin::silk_log2lin;
use crate::silk::lpc_fit::silk_lpc_fit;
use crate::silk::macros::*;
use crate::silk::nlsf::silk_bwexpander_32;
use crate::silk::sigm::*;
use crate::silk::sigproc_fix::*;
use crate::silk::structs::*;

pub fn warped_gain(coefs_q24: &[i32], mut lambda_q16: i32, order: usize) -> i32 {
    let mut gain_q24: i32;

    lambda_q16 = -lambda_q16;
    gain_q24 = coefs_q24[order - 1];
    for i in (0..order - 1).rev() {
        gain_q24 = silk_smlawb(coefs_q24[i], gain_q24, lambda_q16);
    }
    gain_q24 = silk_smlawb(1 << 24, gain_q24, -lambda_q16);
    silk_inverse32_varq(gain_q24, 40)
}

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
    let mut maxabs_q20: i32;
    let mut ind: usize = 0;

    lambda_q16 = -lambda_q16;
    for i in (1..order).rev() {
        coefs_q24[i - 1] = silk_smlawb(coefs_q24[i - 1], coefs_q24[i], lambda_q16);
    }
    lambda_q16 = -lambda_q16;
    nom_q16 = silk_smlawb(1 << 16, -lambda_q16, lambda_q16);
    den_q24 = silk_smlawb(1 << 24, coefs_q24[0], lambda_q16);
    if den_q24 != 0 {
        gain_q16 = silk_div32_varq(nom_q16, den_q24, 24);
    }
    for v in coefs_q24[..order].iter_mut() {
        *v = silk_smulww(gain_q16, *v);
    }

    let limit_q20 = limit_q24 >> 4;
    for iter in 0..10 {

        maxabs_q24 = -1;
        for (i, &v) in coefs_q24[..order].iter().enumerate() {
            tmp = v.abs();
            if tmp > maxabs_q24 {
                maxabs_q24 = tmp;
                ind = i;
            }
        }

        maxabs_q20 = maxabs_q24 >> 4;
        if maxabs_q20 <= limit_q20 {

            return;
        }

        for i in 1..order {
            coefs_q24[i - 1] = silk_smlawb(coefs_q24[i - 1], coefs_q24[i], lambda_q16);
        }
        if gain_q16 != 0 {
            gain_q16 = silk_inverse32_varq(gain_q16, 32);
        }
        for v in coefs_q24[..order].iter_mut() {
            *v = silk_smulww(gain_q16, *v);
        }

        let weight = silk_smlabb(819, 102_i32, iter);
        let numerator = silk_smulwb(maxabs_q20 - limit_q20, weight);
        let denominator = maxabs_q20 * ((ind + 1) as i32);
        chirp_q16 = 64881 - silk_div32_varq(numerator, denominator, 22);

        silk_bwexpander_32(coefs_q24, order, chirp_q16);

        lambda_q16 = -lambda_q16;
        for i in (1..order).rev() {
            coefs_q24[i - 1] = silk_smlawb(coefs_q24[i - 1], coefs_q24[i], lambda_q16);
        }
        lambda_q16 = -lambda_q16;
        nom_q16 = silk_smlawb(1 << 16, -lambda_q16, lambda_q16);
        den_q24 = silk_smlawb(1 << 24, coefs_q24[0], lambda_q16);
        if den_q24 != 0 {
            gain_q16 = silk_div32_varq(nom_q16, den_q24, 24);
        }
        for v in coefs_q24[..order].iter_mut() {
            *v = silk_smulww(gain_q16, *v);
        }
    }
}

#[inline(always)]
pub fn silk_noise_shape_analysis_fix(
    ps_enc: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControl,
    pitch_res: &[i16],
    x: &[i16],
) {
    let n_samples: usize;
    let n_segs: usize;
    let mut q_nrg: i32;
    let mut scale: i32 = 0;
    let mut snr_adj_db_q7: i32;
    let mut harm_shape_gain_q16: i32;
    let tilt_q16: i32;
    let mut tmp32: i32;
    let mut nrg: i32 = 0;
    let mut log_energy_q7: i32;
    let mut log_energy_prev_q7: i32 = 0;
    let mut energy_variation_q7: i32;
    let mut strength_q16: i32;
    let mut b_q8: i32;
    let mut auto_corr = [0i32; MAX_SHAPE_LPC_ORDER + 1];
    let mut refl_coef_q16 = [0i32; MAX_SHAPE_LPC_ORDER];
    let mut ar_q24 = [0i32; MAX_SHAPE_LPC_ORDER];

    let mut x_windowed = [0i16; MAX_SUB_FRAME_LENGTH + 2 * LA_SHAPE_MAX];

    let mut x_ptr_idx: usize = 0;

    snr_adj_db_q7 = ps_enc.s_cmn.snr_db_q7;

    ps_enc_ctrl.input_quality_q14 =
        (ps_enc.s_cmn.input_quality_bands_q15[0] + ps_enc.s_cmn.input_quality_bands_q15[1]) >> 2;

    ps_enc_ctrl.coding_quality_q14 =
        silk_rshift(silk_sigm_q15(silk_rshift_round(snr_adj_db_q7 - 2560, 4)), 1);

    if ps_enc.s_cmn.use_cbr == 0 {
        b_q8 = 256 - ps_enc.s_cmn.speech_activity_q8;
        b_q8 = silk_smulwb(b_q8 << 8, b_q8);
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

        snr_adj_db_q7 = silk_smlawb(snr_adj_db_q7, 512, ps_enc.ltp_corr_q15);
    } else {

        snr_adj_db_q7 = silk_smlawb(
            snr_adj_db_q7,
            silk_smlawb(3072, -104858, ps_enc.s_cmn.snr_db_q7),
            16384 - ps_enc_ctrl.input_quality_q14,
        );
    }

    if ps_enc.s_cmn.indices.signal_type == TYPE_VOICED as i8 {
        ps_enc.s_cmn.indices.quant_offset_type = 0;
    } else {

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

        if energy_variation_q7 > 77 * (n_segs as i32 - 1) {

            ps_enc.s_cmn.indices.quant_offset_type = 0;
        } else {
            ps_enc.s_cmn.indices.quant_offset_type = 1;
        }
    }

    strength_q16 = silk_smulwb(ps_enc_ctrl.pred_gain_q16, 66);
    let bw_exp_q16 = silk_div32_varq(61604, silk_smlaww(65536, strength_q16, strength_q16), 16);

    let warping_q16 = if ps_enc.s_cmn.warping_q16 > 0 {
        silk_smlawb(
            ps_enc.s_cmn.warping_q16,
            ps_enc_ctrl.coding_quality_q14,
            2621,
        )
    } else {
        0
    };

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
        q_nrg >>= 1;

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

            let gain_mult_q16 = warped_gain(
                &ar_q24,
                warping_q16,
                ps_enc.s_cmn.shaping_lpc_order as usize,
            );
            if ps_enc_ctrl.gains_q16[k] < (1 << 14) {

                ps_enc_ctrl.gains_q16[k] = silk_smulww(ps_enc_ctrl.gains_q16[k], gain_mult_q16);
            } else {
                ps_enc_ctrl.gains_q16[k] = silk_smulww(
                    silk_rshift_round(ps_enc_ctrl.gains_q16[k], 1),
                    gain_mult_q16,
                );
                if ps_enc_ctrl.gains_q16[k] >= (i32::MAX >> 1) {
                    ps_enc_ctrl.gains_q16[k] = i32::MAX;
                } else {
                    ps_enc_ctrl.gains_q16[k] <<= 1;
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
            for (i, ar_val) in ar_q24[..ps_enc.s_cmn.shaping_lpc_order as usize].iter().enumerate() {
                ps_enc_ctrl.ar_q13[k * MAX_SHAPE_LPC_ORDER + i] =
                    silk_sat16(silk_rshift_round(*ar_val, 11)) as i16;
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

    let gain_mult_q16_val = silk_log2lin(-silk_smlawb(-2048, snr_adj_db_q7, 10486));

    let gain_add_q16_val = silk_log2lin(silk_smlawb(2048, 256, 10486));

    for k in 0..ps_enc.s_cmn.nb_subfr as usize {
        ps_enc_ctrl.gains_q16[k] = silk_smulww(ps_enc_ctrl.gains_q16[k], gain_mult_q16_val);
        ps_enc_ctrl.gains_q16[k] = silk_add_pos_sat32(ps_enc_ctrl.gains_q16[k], gain_add_q16_val);
    }

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
                (16384 - b_q14 - silk_smulwb(strength_q16, b_q14)) << 16;
            ps_enc_ctrl.lf_shp_q14[k] |= (b_q14 - 16384) & 0xFFFF;
        }
        tilt_q16 = -16384
            - silk_smulwb(
                65536 - 16384,
                silk_smulwb(5872026, ps_enc.s_cmn.speech_activity_q8),
            );
    } else {
        let b_q14 = silk_div32_16(21299, ps_enc.s_cmn.fs_khz);
        let lf_high =
            16384 - b_q14 - silk_smulwb(strength_q16, silk_smulwb(39322, b_q14));
        let lf_low_raw = b_q14 - 16384;
        let lf_low_masked = lf_low_raw & 0xFFFF;

        ps_enc_ctrl.lf_shp_q14[0] = lf_high << 16;
        ps_enc_ctrl.lf_shp_q14[0] |= lf_low_masked;
        for k in 1..ps_enc.s_cmn.nb_subfr as usize {
            ps_enc_ctrl.lf_shp_q14[k] = ps_enc_ctrl.lf_shp_q14[0];
        }
        tilt_q16 = -16384;
    }

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
