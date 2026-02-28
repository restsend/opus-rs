use crate::silk::define::*;
use crate::silk::macros::*;
use crate::silk::nlsf_encode::silk_nlsf_encode;
use crate::silk::structs::*;
use crate::silk::tables::*;
use crate::silk::tables_nlsf::*;

use crate::silk::lpc_analysis::*;
use crate::silk::lpc_fit::*;

pub fn silk_bwexpander_32(ar: &mut [i32], d: usize, mut chirp_q16: i32) {
    let chirp_minus_one_q16 = chirp_q16 - 65536;

    for i in 0..d - 1 {
        ar[i] = silk_smulww(chirp_q16, ar[i]);
        chirp_q16 += silk_rshift_round(
            ((chirp_q16 as i64) * (chirp_minus_one_q16 as i64)) as i32,
            16,
        );
    }
    ar[d - 1] = silk_smulww(chirp_q16, ar[d - 1]);
}

fn silk_a2nlsf_trans_poly(p: &mut [i32], dd: usize) {
    for k in 2..=dd {
        for n in (k + 1..=dd).rev() {
            p[n - 2] -= p[n];
        }
        p[k - 2] -= p[k] << 1;
    }
}

fn silk_a2nlsf_eval_poly(p: &[i32], x: i32, dd: usize) -> i32 {
    let mut y32 = p[dd];
    let x_q16 = x << 4;

    for n in (0..dd).rev() {
        y32 = silk_smlaww(p[n], y32, x_q16);
    }
    y32
}

fn silk_a2nlsf_init(a_q16: &[i32], p: &mut [i32], q: &mut [i32], dd: usize) {
    p[dd] = 1 << 16;
    q[dd] = 1 << 16;
    for k in 0..dd {
        p[k] = -a_q16[dd - k - 1] - a_q16[dd + k];
        q[k] = -a_q16[dd - k - 1] + a_q16[dd + k];
    }

    for k in (1..=dd).rev() {
        p[k - 1] -= p[k];
        q[k - 1] += q[k];
    }

    silk_a2nlsf_trans_poly(p, dd);
    silk_a2nlsf_trans_poly(q, dd);
}

pub fn silk_a2nlsf(nlsf: &mut [i16], a_q16: &mut [i32], d: usize) {
    let mut p = [0i32; MAX_LPC_ORDER / 2 + 1];
    let mut q = [0i32; MAX_LPC_ORDER / 2 + 1];
    let dd = d >> 1;

    silk_a2nlsf_init(a_q16, &mut p, &mut q, dd);

    let mut xlo = SILK_LSF_COS_TAB_FIX_Q12[0] as i32;
    let mut ylo = silk_a2nlsf_eval_poly(&p, xlo, dd);

    let mut root_ix: usize;
    let mut poly_idx: usize;
    if ylo < 0 {
        nlsf[0] = 0;
        poly_idx = 1;
        ylo = silk_a2nlsf_eval_poly(&q, xlo, dd);
        root_ix = 1;
    } else {
        poly_idx = 0;
        root_ix = 0;
    }

    let mut k = 1;
    let mut i = 0;
    let mut thr = 0;

    const BIN_DIV_STEPS_A2NLSF_FIX: i32 = 3;
    const MAX_ITERATIONS_A2NLSF_FIX: i32 = 16;

    loop {
        let p_ptr = if poly_idx == 0 { &p } else { &q };
        let xhi = SILK_LSF_COS_TAB_FIX_Q12[k] as i32;
        let yhi = silk_a2nlsf_eval_poly(p_ptr, xhi, dd);

        if (ylo <= 0 && yhi >= thr) || (ylo >= 0 && yhi <= -thr) {
            if yhi == 0 {
                thr = 1;
            } else {
                thr = 0;
            }

            let mut ffrac = -256;
            let mut xlo_div = xlo;
            let mut xhi_div = xhi;
            let mut ylo_div = ylo;
            let mut yhi_div = yhi;

            for m in 0..BIN_DIV_STEPS_A2NLSF_FIX {
                let xmid = (xlo_div + xhi_div + 1) >> 1;
                let ymid = silk_a2nlsf_eval_poly(p_ptr, xmid, dd);

                if (ylo_div <= 0 && ymid >= 0) || (ylo_div >= 0 && ymid <= 0) {
                    xhi_div = xmid;
                    yhi_div = ymid;
                } else {
                    xlo_div = xmid;
                    ylo_div = ymid;
                    ffrac = silk_add_rshift(ffrac, 128, m);
                }
            }

            let den: i32;
            let nom: i32;
            if ylo_div.abs() < 65536 {
                den = ylo_div - yhi_div;
                nom = (ylo_div << (8 - BIN_DIV_STEPS_A2NLSF_FIX)) + (den >> 1);
                if den != 0 {
                    ffrac += nom / den;
                }
            } else {
                ffrac += ylo_div / ((ylo_div - yhi_div) >> (8 - BIN_DIV_STEPS_A2NLSF_FIX));
            }

            nlsf[root_ix] = ((k as i32) << 8).wrapping_add(ffrac).min(i16::MAX as i32) as i16;

            root_ix += 1;
            if root_ix >= d {
                break;
            }

            poly_idx = root_ix & 1;
            xlo = SILK_LSF_COS_TAB_FIX_Q12[k - 1] as i32;
            ylo = (1 - (root_ix as i32 & 2)) << 12;
        } else {
            k += 1;
            xlo = xhi;
            ylo = yhi;
            thr = 0;

            if k > LSF_COS_TAB_SZ_FIX {
                i += 1;
                if i > MAX_ITERATIONS_A2NLSF_FIX {
                    let val = (1i32 << 15) / (d as i32 + 1);
                    nlsf[0] = val as i16;
                    for j in 1..d {
                        nlsf[j] = nlsf[j - 1].wrapping_add(val as i16);
                    }
                    return;
                }

                silk_bwexpander_32(a_q16, d, 65536 - (1 << i));
                silk_a2nlsf_init(a_q16, &mut p, &mut q, dd);
                xlo = SILK_LSF_COS_TAB_FIX_Q12[0] as i32;
                ylo = silk_a2nlsf_eval_poly(&p, xlo, dd);
                if ylo < 0 {
                    nlsf[0] = 0;
                    poly_idx = 1;
                    ylo = silk_a2nlsf_eval_poly(&q, xlo, dd);
                    root_ix = 1;
                } else {
                    poly_idx = 0;
                    root_ix = 0;
                }
                k = 1;
            }
        }
    }
}

pub fn silk_nlsf_stabilize(nlsf_q15: &mut [i16], n_delta_min_q15: &[i16], l: usize) {
    const MAX_LOOPS: i32 = 20;

    let mut loops = 0;
    while loops < MAX_LOOPS {
        let mut min_diff_q15 = nlsf_q15[0] as i32 - n_delta_min_q15[0] as i32;
        let mut i_idx = 0;

        for i in 1..l {
            let diff_q15 =
                nlsf_q15[i] as i32 - (nlsf_q15[i - 1] as i32 + n_delta_min_q15[i] as i32);
            if diff_q15 < min_diff_q15 {
                min_diff_q15 = diff_q15;
                i_idx = i;
            }
        }

        let diff_q15 = (1 << 15) - (nlsf_q15[l - 1] as i32 + n_delta_min_q15[l] as i32);
        if diff_q15 < min_diff_q15 {
            min_diff_q15 = diff_q15;
            i_idx = l;
        }

        if min_diff_q15 >= 0 {
            return;
        }

        if i_idx == 0 {
            nlsf_q15[0] = n_delta_min_q15[0];
        } else if i_idx == l {
            nlsf_q15[l - 1] = ((1i32 << 15) - n_delta_min_q15[l] as i32) as i16;
        } else {
            let mut min_center_q15 = 0;
            for k in 0..i_idx {
                min_center_q15 += n_delta_min_q15[k] as i32;
            }
            min_center_q15 += (n_delta_min_q15[i_idx] as i32) >> 1;

            let mut max_center_q15 = 1 << 15;
            for k in (i_idx + 1..=l).rev() {
                max_center_q15 -= n_delta_min_q15[k] as i32;
            }
            max_center_q15 -= (n_delta_min_q15[i_idx] as i32) >> 1;

            let center_freq_q15 = silk_limit_32(
                silk_rshift_round(nlsf_q15[i_idx - 1] as i32 + nlsf_q15[i_idx] as i32, 1),
                min_center_q15,
                max_center_q15,
            ) as i16;
            nlsf_q15[i_idx - 1] = center_freq_q15 - ((n_delta_min_q15[i_idx] as i32) >> 1) as i16;
            nlsf_q15[i_idx] = nlsf_q15[i_idx - 1] + n_delta_min_q15[i_idx];
        }
        loops += 1;
    }

    if loops == MAX_LOOPS {
        nlsf_q15[..l].sort();

        nlsf_q15[0] = nlsf_q15[0].max(n_delta_min_q15[0]);

        for i in 1..l {
            nlsf_q15[i] = nlsf_q15[i].max(nlsf_q15[i - 1].saturating_add(n_delta_min_q15[i]));
        }

        nlsf_q15[l - 1] = nlsf_q15[l - 1].min(((1i32 << 15) - n_delta_min_q15[l] as i32) as i16);

        for i in (0..l - 1).rev() {
            nlsf_q15[i] = nlsf_q15[i].min(nlsf_q15[i + 1].wrapping_sub(n_delta_min_q15[i + 1]));
        }
    }
}

pub fn silk_nlsf_vq(
    err_q24: &mut [i32],
    in_q15: &[i16],
    pcb_q8: &[u8],
    pwght_q9: &[i16],
    k: usize,
    lpc_order: usize,
) {
    for i in 0..k {
        let mut sum_error_q24 = 0i32;
        let mut pred_q24 = 0i32;
        let cb_ptr = &pcb_q8[i * lpc_order..];
        let w_ptr = &pwght_q9[i * lpc_order..];

        for m in (0..lpc_order).step_by(2).rev() {
            // Index m + 1
            let diff_q15 = (in_q15[m + 1] as i32) - ((cb_ptr[m + 1] as i32) << 7);
            let diffw_q24 = silk_smulbb(diff_q15, w_ptr[m + 1] as i32);
            sum_error_q24 = sum_error_q24.wrapping_add((diffw_q24 - (pred_q24 >> 1)).abs());
            pred_q24 = diffw_q24;

            // Index m
            let diff_q15_m = (in_q15[m] as i32) - ((cb_ptr[m] as i32) << 7);
            let diffw_q24_m = silk_smulbb(diff_q15_m, w_ptr[m] as i32);
            sum_error_q24 = sum_error_q24.wrapping_add((diffw_q24_m - (pred_q24 >> 1)).abs());
            pred_q24 = diffw_q24;
        }
        err_q24[i] = sum_error_q24;
    }
}

fn silk_nlsf2a_find_poly(out: &mut [i32], clsf: &[i32], dd: usize) {
    out[0] = 1 << 16;
    out[1] = -clsf[0];
    for k in 1..dd {
        let ftmp = clsf[2 * k];
        out[k + 1] = (out[k - 1] << 1) - (silk_rshift_round64(silk_smull(ftmp, out[k]), 16) as i32);
        for n in (2..=k).rev() {
            out[n] += out[n - 2] - (silk_rshift_round64(silk_smull(ftmp, out[n - 1]), 16) as i32);
        }
        out[1] -= ftmp;
    }
}

pub fn silk_nlsf2a(a_q12: &mut [i16], nlsf: &[i16], d: usize) {
    const ORDERING16: [u8; 16] = [0, 15, 8, 7, 4, 11, 12, 3, 2, 13, 10, 5, 6, 9, 14, 1];
    const ORDERING10: [u8; 10] = [0, 9, 6, 3, 4, 5, 8, 1, 2, 7];

    let ordering = if d == 16 {
        &ORDERING16[..]
    } else {
        &ORDERING10[..]
    };
    let mut cos_lsf_qa = [0i32; MAX_LPC_ORDER];
    for k in 0..d {
        let f_int = (nlsf[k] >> (15 - 7)) as usize;
        let f_frac = nlsf[k] - ((f_int as i16) << (15 - 7));
        let cos_val = SILK_LSF_COS_TAB_FIX_Q12[f_int] as i32;
        let delta = SILK_LSF_COS_TAB_FIX_Q12[f_int + 1] as i32 - cos_val;
        cos_lsf_qa[ordering[k] as usize] =
            silk_rshift_round((cos_val << 8) + (delta as i32 * f_frac as i32), 20 - 16);
    }

    let dd = d >> 1;
    let mut p = [0i32; MAX_LPC_ORDER / 2 + 1];
    let mut q = [0i32; MAX_LPC_ORDER / 2 + 1];

    silk_nlsf2a_find_poly(&mut p, &cos_lsf_qa[0..], dd);
    silk_nlsf2a_find_poly(&mut q, &cos_lsf_qa[1..], dd);

    let mut a32_qa1 = [0i32; MAX_LPC_ORDER];
    for k in 0..dd {
        let ptmp = p[k + 1].wrapping_add(p[k]);
        let qtmp = q[k + 1].wrapping_sub(q[k]);
        a32_qa1[k] = -qtmp.wrapping_add(ptmp);
        a32_qa1[d - k - 1] = qtmp.wrapping_sub(ptmp);
    }

    silk_lpc_fit(a_q12, &mut a32_qa1, 12, 16 + 1, d);

    for i in 0..MAX_LPC_STABILIZE_ITERATIONS {
        if silk_lpc_inverse_pred_gain(a_q12, d) != 0 {
            break;
        }
        silk_bwexpander_32(&mut a32_qa1, d, 65536 - (2 << i));
        for k in 0..d {
            a_q12[k] = silk_rshift_round(a32_qa1[k], 16 + 1 - 12) as i16;
        }
    }
}

pub fn silk_nlsf_vq_weights_laroia(p_w_q5: &mut [i16], p_nlsf_q15: &[i16], d: usize) {
    /* NLSF_W_Q = 2, so 1 << (15 + NLSF_W_Q) = 1 << 17 = 131072 */
    const NUMER: i32 = 1 << 17;

    /* First value: weight[0] = 1/nlsf[0] + 1/(nlsf[1]-nlsf[0]) */
    let mut tmp1_int = (p_nlsf_q15[0] as i32).max(1);
    tmp1_int = silk_div32_16(NUMER, tmp1_int);
    let mut tmp2_int = ((p_nlsf_q15[1] - p_nlsf_q15[0]) as i32).max(1);
    tmp2_int = silk_div32_16(NUMER, tmp2_int);
    p_w_q5[0] = silk_limit(tmp1_int + tmp2_int, 0, i16::MAX as i32) as i16;

    /* Main loop: handle pairs (step by 2) */
    let mut k = 1;
    while k < d - 1 {
        tmp1_int = ((p_nlsf_q15[k + 1] - p_nlsf_q15[k]) as i32).max(1);
        tmp1_int = silk_div32_16(NUMER, tmp1_int);
        p_w_q5[k] = silk_limit(tmp2_int + tmp1_int, 0, i16::MAX as i32) as i16;

        tmp2_int = ((p_nlsf_q15[k + 2] - p_nlsf_q15[k + 1]) as i32).max(1);
        tmp2_int = silk_div32_16(NUMER, tmp2_int);
        p_w_q5[k + 1] = silk_limit(tmp1_int + tmp2_int, 0, i16::MAX as i32) as i16;

        k += 2;
    }

    /* Last value: weight[D-1] = 1/(32768-nlsf[D-1]) + prev_term */
    tmp1_int = ((32767 - p_nlsf_q15[d - 1]) as i32).max(1);
    tmp1_int = silk_div32_16(NUMER, tmp1_int);
    p_w_q5[d - 1] = silk_limit(tmp2_int + tmp1_int, 0, i16::MAX as i32) as i16;
}

pub fn silk_process_nlsfs(
    ps_enc: &mut SilkEncoderState,
    ps_enc_ctrl: &mut SilkEncoderControl,
    nlsf_q15: &mut [i16],
) {
    let do_interp: i32;
    let mut nlsf_interp_q15 = [0i16; MAX_LPC_ORDER];
    let mut p_cb = &SILK_NLSF_CB_WB;
    if ps_enc.s_cmn.fs_khz == 8 {
        p_cb = &SILK_NLSF_CB_NB_MB;
    }

    let order = ps_enc.s_cmn.predict_lpc_order as usize;
    /* NLSF_mu = 0.003 - 0.001 * speech_activity (matching C process_NLSFs.c) */
    /* SILK_FIX_CONST(0.003, 20) = 3145, SILK_FIX_CONST(-0.001, 28) = -268435 */
    let mut nlsf_mu_q20 = silk_smlawb(3145, -268435, ps_enc.s_cmn.speech_activity_q8);
    /* Multiply by 1.5 for 10 ms packets (nb_subfr == 2) */
    if ps_enc.s_cmn.nb_subfr == 2 {
        nlsf_mu_q20 = nlsf_mu_q20 + silk_rshift(nlsf_mu_q20, 1);
    }
    let interp_coef_q2 = ps_enc.s_cmn.indices.nlsf_interp_coef_q2 as i32;
    let use_interp = ps_enc.s_cmn.use_interpolated_nlsfs != 0;

    /* NLSF stabilization */
    silk_nlsf_stabilize(nlsf_q15, p_cb.delta_min_q15, order);

    /* Calculate weights */
    let mut w_q5 = [0i16; MAX_LPC_ORDER];
    silk_nlsf_vq_weights_laroia(&mut w_q5, nlsf_q15, order);

    /* Update NLSF weights for interpolated NLSFs */
    do_interp = (use_interp && interp_coef_q2 < 4) as i32;
    if do_interp != 0 {
        /* Calculate the interpolated NLSF vector for the first half */
        let prev_nlsf_q15 = ps_enc.s_cmn.prev_nlsf_q15;
        for i in 0..order {
            nlsf_interp_q15[i] = prev_nlsf_q15[i]
                + silk_rshift(
                    silk_mul((nlsf_q15[i] - prev_nlsf_q15[i]) as i32, interp_coef_q2),
                    2,
                ) as i16;
        }

        /* Calculate first half NLSF weights for the interpolated NLSFs */
        let mut w0_q5 = [0i16; MAX_LPC_ORDER];
        silk_nlsf_vq_weights_laroia(&mut w0_q5, &nlsf_interp_q15, order);

        /* Update NLSF weights with contribution from first half */
        let i_sqr_q15 = silk_lshift(silk_smulbb(interp_coef_q2, interp_coef_q2), 11);
        for i in 0..order {
            w_q5[i] = silk_rshift(w_q5[i] as i32, 1) as i16
                + silk_rshift(silk_smulbb(w0_q5[i] as i32, i_sqr_q15), 16) as i16;
        }
    }

    /* NLSF quantization - nlsf_q15 is both input and output */
    /* w_q5 already contains Q2 weights (output of silk_NLSF_VQ_weights_laroia where NLSF_W_Q=2) */
    silk_nlsf_encode(
        &mut ps_enc.s_cmn.indices.nlsf_indices,
        nlsf_q15,
        p_cb,
        &w_q5, /* Q2 weights, matching C's pNLSFW_QW passed directly */
        nlsf_mu_q20,
        ps_enc.s_cmn.n_nlsf_survivors as usize,
        ps_enc.s_cmn.indices.signal_type as i32,
    );

    /* Convert quantized NLSFs to LPC */
    silk_nlsf2a(&mut ps_enc_ctrl.pred_coef_q12[1], nlsf_q15, order);

    if do_interp != 0 {
        /* Calculate the interpolated, quantized LSF vector for the first half */
        let prev_nlsf_q15 = ps_enc.s_cmn.prev_nlsf_q15;
        for i in 0..order {
            nlsf_interp_q15[i] = prev_nlsf_q15[i]
                + silk_rshift(
                    silk_mul((nlsf_q15[i] - prev_nlsf_q15[i]) as i32, interp_coef_q2),
                    2,
                ) as i16;
        }
        /* Convert back to LPC coefficients */
        silk_nlsf2a(&mut ps_enc_ctrl.pred_coef_q12[0], &nlsf_interp_q15, order);
    } else {
        /* Copy LPC coefficients for first half from second half */
        let (first, second) = ps_enc_ctrl.pred_coef_q12.split_at_mut(1);
        first[0][..order].copy_from_slice(&second[0][..order]);
    }

    /* Copy quantized NLSFs to previous for next frame */
    ps_enc.s_cmn.prev_nlsf_q15[..order].copy_from_slice(&nlsf_q15[..order]);
}
