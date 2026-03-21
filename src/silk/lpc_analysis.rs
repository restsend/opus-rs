use crate::silk::define::*;
use crate::silk::interpolate::silk_interpolate;
use crate::silk::macros::*;
use crate::silk::nlsf::*;
use crate::silk::sigproc_fix::*;
use crate::silk::structs::*;
use crate::silk::tuning_parameters::*;

pub fn silk_find_lpc_fix(
    ps_enc_c: &mut SilkEncoderStateCommon,
    nlsf_q15: &mut [i16],
    x: &[i16],
    min_inv_gain_q30: i32,
) {
    let mut res_nrg: i32 = 0;
    let mut res_nrg_q: i32 = 0;
    let mut a_q16 = [0i32; MAX_LPC_ORDER];
    let d = ps_enc_c.predict_lpc_order as usize;
    let subfr_length = (ps_enc_c.subfr_length + ps_enc_c.predict_lpc_order) as usize;

    ps_enc_c.indices.nlsf_interp_coef_q2 = 4;

    silk_burg_modified_fix(
        &mut res_nrg,
        &mut res_nrg_q,
        &mut a_q16,
        x,
        min_inv_gain_q30,
        subfr_length,
        ps_enc_c.nb_subfr as usize,
        d,
    );

    if ps_enc_c.use_interpolated_nlsfs != 0
        && ps_enc_c.first_frame_after_reset == 0
        && ps_enc_c.nb_subfr == MAX_NB_SUBFR as i32
    {

        let mut res_tmp_nrg: i32 = 0;
        let mut res_tmp_nrg_q: i32 = 0;
        let mut a_tmp_q16 = [0i32; MAX_LPC_ORDER];
        silk_burg_modified_fix(
            &mut res_tmp_nrg,
            &mut res_tmp_nrg_q,
            &mut a_tmp_q16,
            &x[2 * subfr_length..],
            min_inv_gain_q30,
            subfr_length,
            2,
            d,
        );

        let shift = res_tmp_nrg_q - res_nrg_q;
        if shift >= 0 {
            if shift < 32 {
                res_nrg = res_nrg - silk_rshift(res_tmp_nrg, shift);
            }

        } else {
            debug_assert!(shift > -32);
            res_nrg = silk_rshift(res_nrg, -shift) - res_tmp_nrg;
            res_nrg_q = res_tmp_nrg_q;
        }

        silk_a2nlsf(nlsf_q15, &mut a_tmp_q16, d);

        let lpc_res_len = 2 * subfr_length;

        let mut lpc_res = [0i16; 2 * (MAX_SUB_FRAME_LENGTH + MAX_LPC_ORDER)];

        for k in (0..=3).rev() {

            let nlsf0_q15 = silk_interpolate(&ps_enc_c.prev_nlsf_q15, nlsf_q15, k, d);

            let mut a_tmp_q12 = [0i16; MAX_LPC_ORDER];
            silk_nlsf2a(&mut a_tmp_q12, &nlsf0_q15, d);

            silk_lpc_analysis_filter(
                &mut lpc_res,
                x,
                &a_tmp_q12,
                lpc_res_len,
                d,
                0,
            );

            let mut res_nrg0: i32 = 0;
            let mut rshift0: i32 = 0;
            silk_sum_sqr_shift(
                &mut res_nrg0,
                &mut rshift0,
                &lpc_res[d..d + (subfr_length - d)],
                subfr_length - d,
            );

            let mut res_nrg1: i32 = 0;
            let mut rshift1: i32 = 0;
            silk_sum_sqr_shift(
                &mut res_nrg1,
                &mut rshift1,
                &lpc_res[d + subfr_length..d + subfr_length + (subfr_length - d)],
                subfr_length - d,
            );

            let res_nrg_interp_q: i32;
            let shift = rshift0 - rshift1;
            if shift >= 0 {
                res_nrg1 = silk_rshift(res_nrg1, shift);
                res_nrg_interp_q = -rshift0;
            } else {
                res_nrg0 = silk_rshift(res_nrg0, -shift);
                res_nrg_interp_q = -rshift1;
            }
            let res_nrg_interp = res_nrg0.wrapping_add(res_nrg1);

            let shift = res_nrg_interp_q - res_nrg_q;
            let is_interp_lower = if shift >= 0 {
                if shift < 32 {
                    silk_rshift(res_nrg_interp, shift) < res_nrg
                } else {
                    false
                }
            } else {
                if -shift < 32 {
                    res_nrg_interp < silk_rshift(res_nrg, -shift)
                } else {
                    false
                }
            };

            if is_interp_lower {

                res_nrg = res_nrg_interp;
                res_nrg_q = res_nrg_interp_q;
                ps_enc_c.indices.nlsf_interp_coef_q2 = k as i8;
            }
        }
    }

    if ps_enc_c.indices.nlsf_interp_coef_q2 == 4 {

        silk_a2nlsf(nlsf_q15, &mut a_q16, d);
    }
}

pub fn silk_residual_energy_fix(
    nrgs: &mut [i32],
    nrgs_q: &mut [i32],
    x: &[i16],
    a_q12: &[[i16; MAX_LPC_ORDER]; 2],
    gains: &[i32],
    subfr_length: i32,
    nb_subfr: i32,
    lpc_order: i32,
) {
    let offset: usize;
    let mut rshift: i32 = 0;
    let mut lz1: i32;
    let mut lz2: i32;
    let mut x_ptr_idx: usize = 0;
    let mut tmp32: i32;

    offset = (lpc_order + subfr_length) as usize;

    if (nb_subfr >> 1) * (MAX_NB_SUBFR as i32 >> 1) != nb_subfr {
        return;
    }

    let mut lpc_res = [0i16; (MAX_NB_SUBFR / 2) * (MAX_LPC_ORDER + MAX_SUB_FRAME_LENGTH)];

    for i in 0..(nb_subfr as usize >> 1) {

        silk_lpc_analysis_filter(
            &mut lpc_res,
            &x[x_ptr_idx..],
            &a_q12[i],
            (MAX_NB_SUBFR >> 1) * offset,
            lpc_order as usize,
            0,
        );

        let mut lpc_res_idx = lpc_order as usize;
        for j in 0..(MAX_NB_SUBFR >> 1) {

            silk_sum_sqr_shift(
                &mut nrgs[i * (MAX_NB_SUBFR >> 1) + j],
                &mut rshift,
                &lpc_res[lpc_res_idx..lpc_res_idx + subfr_length as usize],
                subfr_length as usize,
            );

            nrgs_q[i * (MAX_NB_SUBFR >> 1) + j] = -rshift;

            lpc_res_idx += offset;
        }

        x_ptr_idx += (MAX_NB_SUBFR >> 1) * offset;
    }

    for i in 0..nb_subfr as usize {

        lz1 = silk_clz32(nrgs[i]) - 1;
        lz2 = silk_clz32(gains[i]) - 1;

        tmp32 = silk_lshift(gains[i], lz2);

        tmp32 = silk_smmul(tmp32, tmp32);

        nrgs[i] = silk_smmul(tmp32, silk_lshift(nrgs[i], lz1));
        nrgs_q[i] += lz1 + 2 * lz2 - 32 - 32;
    }
}

pub fn silk_burg_modified_fix(
    res_nrg: &mut i32,
    res_nrg_q: &mut i32,
    a_q16: &mut [i32],
    x: &[i16],
    min_inv_gain_q30: i32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) {
    const QA: i32 = 25;
    const N_BITS_HEAD_ROOM: i32 = 3;
    const MIN_RSHIFTS: i32 = -16;
    const MAX_RSHIFTS: i32 = 32 - QA;

    const FIND_LPC_COND_FAC_Q32: i32 = 42950;

    assert!(d <= MAX_LPC_ORDER);

    let mut c_first_row = [0i32; MAX_LPC_ORDER];
    let mut c_last_row = [0i32; MAX_LPC_ORDER];
    let mut af_qa = [0i32; MAX_LPC_ORDER];
    let mut ca_f = [0i32; MAX_LPC_ORDER + 1];
    let mut ca_b = [0i32; MAX_LPC_ORDER + 1];

    let total_len = subfr_length * nb_subfr;
    let mut c0_64: i64 = 0;
    for i in 0..total_len {
        c0_64 += (x[i] as i64) * (x[i] as i64);
    }

    let lz = silk_clz64(c0_64);
    let mut rshifts = 32 + 1 + N_BITS_HEAD_ROOM - lz;
    if rshifts > MAX_RSHIFTS {
        rshifts = MAX_RSHIFTS;
    }
    if rshifts < MIN_RSHIFTS {
        rshifts = MIN_RSHIFTS;
    }

    let c0: i32;
    if rshifts > 0 {
        c0 = silk_rshift64(c0_64, rshifts) as i32;
    } else {
        c0 = ((c0_64 as i32) << (-rshifts)) as i32;
    }

    ca_f[0] = c0 + silk_smmul(FIND_LPC_COND_FAC_Q32, c0) + 1;
    ca_b[0] = ca_f[0];

    if rshifts > 0 {
        for s in 0..nb_subfr {
            let x_ptr = s * subfr_length;
            for n in 1..d + 1 {
                let mut sum: i64 = 0;
                for i in 0..subfr_length - n {
                    sum += (x[x_ptr + i] as i64) * (x[x_ptr + i + n] as i64);
                }
                c_first_row[n - 1] =
                    c_first_row[n - 1].wrapping_add(silk_rshift64(sum, rshifts) as i32);
            }
        }
    } else {
        for s in 0..nb_subfr {
            let x_ptr = s * subfr_length;
            for n in 1..d + 1 {
                let mut sum: i64 = 0;
                for i in 0..subfr_length - n {
                    sum += (x[x_ptr + i] as i64) * (x[x_ptr + i + n] as i64);
                }
                c_first_row[n - 1] =
                    c_first_row[n - 1].wrapping_add(((sum as i32) << (-rshifts)) as i32);
            }
        }
    }
    c_last_row[..d].copy_from_slice(&c_first_row[..d]);

    ca_f[0] = c0 + silk_smmul(FIND_LPC_COND_FAC_Q32, c0) + 1;
    ca_b[0] = ca_f[0];

    let mut inv_gain_q30: i32 = 1 << 30;
    let mut reached_max_gain = false;

    for n in 0..d {

        if rshifts > -2 {
            for s in 0..nb_subfr {
                let x_ptr = s * subfr_length;
                let x1 = -((x[x_ptr + n] as i32) << (16 - rshifts));
                let x2 = -((x[x_ptr + subfr_length - n - 1] as i32) << (16 - rshifts));
                let mut tmp1 = (x[x_ptr + n] as i32) << (QA - 16);
                let mut tmp2 = (x[x_ptr + subfr_length - n - 1] as i32) << (QA - 16);
                for k in 0..n {
                    c_first_row[k] = silk_smlawb(c_first_row[k], x1, x[x_ptr + n - k - 1] as i32);
                    c_last_row[k] =
                        silk_smlawb(c_last_row[k], x2, x[x_ptr + subfr_length - n + k] as i32);
                    let atmp_qa = af_qa[k];
                    tmp1 = silk_smlawb(tmp1, atmp_qa, x[x_ptr + n - k - 1] as i32);
                    tmp2 = silk_smlawb(tmp2, atmp_qa, x[x_ptr + subfr_length - n + k] as i32);
                }
                tmp1 = (-tmp1) << (32 - QA - rshifts);
                tmp2 = (-tmp2) << (32 - QA - rshifts);
                for k in 0..=n {
                    ca_f[k] = silk_smlawb(ca_f[k], tmp1, x[x_ptr + n - k] as i32);
                    ca_b[k] =
                        silk_smlawb(ca_b[k], tmp2, x[x_ptr + subfr_length - n + k - 1] as i32);
                }
            }
        } else {
            for s in 0..nb_subfr {
                let x_ptr = s * subfr_length;
                let x1 = -((x[x_ptr + n] as i32) << (-rshifts));
                let x2 = -((x[x_ptr + subfr_length - n - 1] as i32) << (-rshifts));
                let mut tmp1 = (x[x_ptr + n] as i32) << 17;
                let mut tmp2 = (x[x_ptr + subfr_length - n - 1] as i32) << 17;
                for k in 0..n {
                    c_first_row[k] = silk_mla(c_first_row[k], x1, x[x_ptr + n - k - 1] as i32);
                    c_last_row[k] =
                        silk_mla(c_last_row[k], x2, x[x_ptr + subfr_length - n + k] as i32);
                    let atmp1 = silk_rshift_round(af_qa[k], QA - 17);

                    tmp1 = (tmp1 as u32).wrapping_add(
                        (x[x_ptr + n - k - 1] as i32 as u32).wrapping_mul(atmp1 as u32),
                    ) as i32;
                    tmp2 = (tmp2 as u32).wrapping_add(
                        (x[x_ptr + subfr_length - n + k] as i32 as u32).wrapping_mul(atmp1 as u32),
                    ) as i32;
                }
                tmp1 = -tmp1;
                tmp2 = -tmp2;
                for k in 0..=n {
                    ca_f[k] =
                        silk_smlaww(ca_f[k], tmp1, (x[x_ptr + n - k] as i32) << (-rshifts - 1));
                    ca_b[k] = silk_smlaww(
                        ca_b[k],
                        tmp2,
                        (x[x_ptr + subfr_length - n + k - 1] as i32) << (-rshifts - 1),
                    );
                }
            }
        }

        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        let mut num: i32 = 0;
        let mut nrg: i32 = ca_b[0].wrapping_add(ca_f[0]);
        for k in 0..n {
            let atmp_qa = af_qa[k];
            let lz = (silk_clz32(atmp_qa.abs()) - 1).min(32 - QA);
            let atmp1 = atmp_qa << lz;
            let shift = 32 - QA - lz;

            tmp1 = tmp1.wrapping_add(
                (silk_smmul(c_last_row[n - k - 1], atmp1) as u32 as i64 >> 0 << shift) as i32,
            );
            tmp2 = tmp2.wrapping_add(
                (silk_smmul(c_first_row[n - k - 1], atmp1) as u32 as i64 >> 0 << shift) as i32,
            );
            num = num.wrapping_add(((silk_smmul(ca_b[n - k], atmp1) as i64) << shift) as i32);
            nrg = nrg.wrapping_add(
                ((silk_smmul(ca_b[k + 1].wrapping_add(ca_f[k + 1]), atmp1) as i64) << shift) as i32,
            );
        }
        ca_f[n + 1] = tmp1;
        ca_b[n + 1] = tmp2;
        num = num.wrapping_add(tmp2);
        num = (-num) << 1;

        let mut rc_q31: i32;
        if num.abs() < nrg {
            rc_q31 = silk_div32_varq(num, nrg, 31);
        } else {
            rc_q31 = if num > 0 { i32::MAX } else { i32::MIN };
        }

        tmp1 = (1 << 30) - silk_smmul(rc_q31, rc_q31);
        tmp1 = silk_smmul(inv_gain_q30, tmp1) << 2;
        if tmp1 <= min_inv_gain_q30 {

            tmp2 = (1 << 30) - silk_div32_varq(min_inv_gain_q30, inv_gain_q30, 30);
            rc_q31 = silk_sqrt_approx(tmp2);
            if rc_q31 > 0 {

                rc_q31 = (rc_q31 + silk_div32(tmp2, rc_q31)) >> 1;
                rc_q31 = rc_q31 << 16;
                if num < 0 {
                    rc_q31 = -rc_q31;
                }
            }
            inv_gain_q30 = min_inv_gain_q30;
            reached_max_gain = true;
        } else {
            inv_gain_q30 = tmp1;
        }

        for k in 0..(n + 1) >> 1 {
            tmp1 = af_qa[k];
            tmp2 = af_qa[n - k - 1];
            af_qa[k] = tmp1.wrapping_add((silk_smmul(tmp2, rc_q31) as i64 * 2) as i32);
            af_qa[n - k - 1] = tmp2.wrapping_add((silk_smmul(tmp1, rc_q31) as i64 * 2) as i32);
        }
        af_qa[n] = rc_q31 >> (31 - QA);

        if reached_max_gain {

            for k in n + 1..d {
                af_qa[k] = 0;
            }
            break;
        }

        for k in 0..=n + 1 {
            let idx = n + 1 - k;
            tmp1 = ca_f[k];
            tmp2 = ca_b[idx];
            ca_f[k] = tmp1.wrapping_add((silk_smmul(tmp2, rc_q31) as i64 * 2) as i32);
            ca_b[idx] = tmp2.wrapping_add((silk_smmul(tmp1, rc_q31) as i64 * 2) as i32);
        }
    }

    if reached_max_gain {
        for k in 0..d {
            a_q16[k] = -silk_rshift_round(af_qa[k], QA - 16);
        }

        let mut c0_adj = c0;
        if rshifts > 0 {
            for s in 0..nb_subfr {
                let x_ptr = s * subfr_length;
                let mut sum: i64 = 0;
                for i in 0..d {
                    sum += (x[x_ptr + i] as i64) * (x[x_ptr + i] as i64);
                }
                c0_adj = c0_adj.wrapping_sub(silk_rshift64(sum, rshifts) as i32);
            }
        } else {
            for s in 0..nb_subfr {
                let x_ptr = s * subfr_length;
                let mut sum: i32 = 0;
                for i in 0..d {
                    sum = sum.wrapping_add((x[x_ptr + i] as i32).wrapping_mul(x[x_ptr + i] as i32));
                }
                c0_adj = c0_adj.wrapping_sub(sum << (-rshifts));
            }
        }
        *res_nrg = silk_smmul(inv_gain_q30, c0_adj) << 2;
        *res_nrg_q = -rshifts;
    } else {

        let mut nrg = ca_f[0];
        let mut tmp1_q16: i32 = 1 << 16;
        for k in 0..d {
            let atmp1 = silk_rshift_round(af_qa[k], QA - 16);
            nrg = silk_smlaww(nrg, ca_f[k + 1], atmp1);
            tmp1_q16 = silk_smlaww(tmp1_q16, atmp1, atmp1);
            a_q16[k] = -atmp1;
        }
        *res_nrg = silk_smlaww(nrg, silk_smmul(FIND_LPC_COND_FAC_Q32, c0), -tmp1_q16);
        *res_nrg_q = -rshifts;
    }
}

pub fn energy_flp(x: &[f32]) -> f64 {
    let mut sum = 0.0;
    for &val in x {
        sum += val as f64 * val as f64;
    }
    sum
}

pub fn inner_product_flp(x1: &[f32], x2: &[f32]) -> f64 {
    let len = x1.len().min(x2.len());
    let mut sum = 0.0;
    for i in 0..len {
        sum += x1[i] as f64 * x2[i] as f64;
    }
    sum
}

pub fn silk_lpc_analysis_filter_flp(
    r_lpc: &mut [f32],
    pred_coef: &[f32],
    s: &[f32],
    length: usize,
    order: usize,
) {
    assert!(order <= MAX_LPC_ORDER);

    for ix in order..length {
        let mut lpc_pred = 0.0f32;
        for j in 0..order {
            lpc_pred += s[ix - 1 - j] * pred_coef[j];
        }
        r_lpc[ix] = s[ix] - lpc_pred;
    }
}

pub fn silk_burg_modified_flp(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: usize,
    nb_subfr: usize,
    d: usize,
) -> f32 {
    let mut c_first_row = [0.0f64; MAX_LPC_ORDER];
    let mut c_last_row = [0.0f64; MAX_LPC_ORDER];
    let mut caf = [0.0f64; MAX_LPC_ORDER + 1];
    let mut cab = [0.0f64; MAX_LPC_ORDER + 1];
    let mut af = [0.0f64; MAX_LPC_ORDER];

    let c0 = energy_flp(&x[..nb_subfr * subfr_length]);

    for s in 0..nb_subfr {
        let x_ptr = &x[s * subfr_length..];
        for n in 1..d + 1 {
            c_first_row[n - 1] +=
                inner_product_flp(&x_ptr[..subfr_length - n], &x_ptr[n..subfr_length]);
        }
    }
    c_last_row[..d].copy_from_slice(&c_first_row[..d]);

    caf[0] = c0 + (FIND_LPC_COND_FAC as f64) * c0 + 1e-9;
    cab[0] = caf[0];

    let mut inv_gain = 1.0f64;
    let mut reached_max_gain = false;

    for n in 0..d {
        for s in 0..nb_subfr {
            let x_ptr = &x[s * subfr_length..];
            let mut tmp1 = x_ptr[n] as f64;
            let mut tmp2 = x_ptr[subfr_length - n - 1] as f64;
            for k in 0..n {
                let atmp = af[k];
                tmp1 += x_ptr[n - k - 1] as f64 * atmp;
                tmp2 += x_ptr[subfr_length - n + k] as f64 * atmp;
            }
            for k in 0..=n {
                caf[k] -= tmp1 * x_ptr[n - k] as f64;
                cab[k] -= tmp2 * x_ptr[subfr_length - n + k - 1] as f64;
            }
        }

        let mut tmp1 = c_first_row[n];
        let mut tmp2 = c_last_row[n];
        for k in 0..n {
            let atmp = af[k];
            tmp1 += c_last_row[n - k - 1] * atmp;
            tmp2 += c_first_row[n - k - 1] * atmp;
        }

        caf[n + 1] = tmp1;
        cab[n + 1] = tmp2;

        let mut num = cab[n + 1];
        let mut nrg_b = cab[0];
        let mut nrg_f = caf[0];
        for k in 0..n {
            let atmp = af[k];
            num += cab[n - k] * atmp;
            nrg_b += cab[k + 1] * atmp;
            nrg_f += caf[k + 1] * atmp;
        }

        let mut rc = -2.0 * num / (nrg_f + nrg_b);

        let tmp1_rc = inv_gain * (1.0 - rc * rc);
        if tmp1_rc <= min_inv_gain as f64 {
            rc = (1.0 - min_inv_gain as f64 / inv_gain).sqrt();
            if num > 0.0 {
                rc = -rc;
            }
            inv_gain = min_inv_gain as f64;
            reached_max_gain = true;
        } else {
            inv_gain = tmp1_rc;
        }

        for k in 0..((n + 1) / 2) {
            let tmp1 = af[k];
            let tmp2 = af[n - k];
            af[k] = tmp1 + rc * tmp2;
            af[n - k] = tmp2 + rc * tmp1;
        }
        af[n] = rc;

        if reached_max_gain {
            for k in n + 1..d {
                af[k] = 0.0;
            }
            break;
        }

        for k in 0..=n + 1 {
            let tmp1 = caf[k];
            caf[k] += rc * cab[n - k + 1];
            cab[n - k + 1] += rc * tmp1;
        }
    }

    let mut final_nrg_f: f64;
    if reached_max_gain {
        for k in 0..d {
            a[k] = (-af[k]) as f32;
        }
        let mut c0_mod = c0;
        for s in 0..nb_subfr {
            c0_mod -= energy_flp(&x[s * subfr_length..s * subfr_length + d]);
        }
        final_nrg_f = c0_mod * inv_gain;
    } else {
        final_nrg_f = caf[0];
        let mut tmp1 = 1.0f64;
        for k in 0..d {
            let atmp = af[k];
            final_nrg_f += caf[k + 1] * atmp;
            tmp1 += atmp * atmp;
            a[k] = (-atmp) as f32;
        }
        final_nrg_f -= (FIND_LPC_COND_FAC as f64) * c0 * tmp1;
    }

    final_nrg_f as f32
}

const QA_INV: i32 = 24;
const A_LIMIT: i32 = 16773043;

fn lpc_inverse_pred_gain_qa(a_qa_in: &mut [i32], order: usize) -> i32 {
    let mut inv_gain_q30 = 1 << 30;
    for k in (1..order).rev() {
        if a_qa_in[k] > A_LIMIT || a_qa_in[k] < -A_LIMIT {
            return 0;
        }

        let rc_q31 = -(a_qa_in[k] << (31 - QA_INV));
        let rc_mult1_q30 = (1 << 30) - silk_smmul(rc_q31, rc_q31);

        inv_gain_q30 = silk_smmul(inv_gain_q30, rc_mult1_q30) << 2;
        if inv_gain_q30 < (1 << 30) / 10000 {

            return 0;
        }

        let mult2q = 32 - silk_clz32(rc_mult1_q30.abs());
        let rc_mult2 = silk_inverse32_varq(rc_mult1_q30, mult2q + 30);

        for n in 0..(k + 1) / 2 {
            let tmp1 = a_qa_in[n];
            let tmp2 = a_qa_in[k - n - 1];

            let mul_q = |a: i32, b: i32, q: i32| -> i32 {
                (silk_rshift_round64(silk_smull(a, b), q)) as i32
            };

            let tmp64_1 = silk_rshift_round64(
                silk_smull(tmp1.wrapping_sub(mul_q(tmp2, rc_q31, 31)), rc_mult2),
                mult2q,
            );
            if tmp64_1 > i32::MAX as i64 || tmp64_1 < i32::MIN as i64 {
                return 0;
            }
            a_qa_in[n] = tmp64_1 as i32;

            let tmp64_2 = silk_rshift_round64(
                silk_smull(tmp2.wrapping_sub(mul_q(tmp1, rc_q31, 31)), rc_mult2),
                mult2q,
            );
            if tmp64_2 > i32::MAX as i64 || tmp64_2 < i32::MIN as i64 {
                return 0;
            }
            a_qa_in[k - n - 1] = tmp64_2 as i32;
        }
    }

    if a_qa_in[0] > A_LIMIT || a_qa_in[0] < -A_LIMIT {
        return 0;
    }

    let rc_q31 = -(a_qa_in[0] << (31 - QA_INV));
    let rc_mult1_q30 = (1 << 30) - silk_smmul(rc_q31, rc_q31);
    inv_gain_q30 = silk_smmul(inv_gain_q30, rc_mult1_q30) << 2;

    if inv_gain_q30 < (1 << 30) / 10000 {
        return 0;
    }

    inv_gain_q30
}

pub fn silk_lpc_inverse_pred_gain(a_q12: &[i16], order: usize) -> i32 {
    let mut dc_resp = 0i32;
    let mut atmp_qa = [0i32; MAX_LPC_ORDER];
    for k in 0..order {
        dc_resp += a_q12[k] as i32;
        atmp_qa[k] = (a_q12[k] as i32) << (QA_INV - 12);
    }
    if dc_resp >= 4096 {
        return 0;
    }
    lpc_inverse_pred_gain_qa(&mut atmp_qa, order)
}
