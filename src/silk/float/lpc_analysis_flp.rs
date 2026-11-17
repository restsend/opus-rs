//! libopus `silk/float/find_LPC_FLP.c` and `residual_energy_FLP.c`: the LPC
//! analysis of a frame and the residual energies of its subframes.

use crate::silk::define::*;
use crate::silk::float::sigproc_flp::*;
use crate::silk::float::wrappers_flp::{silk_a2nlsf_flp, silk_nlsf2a_flp};
use crate::silk::interpolate::silk_interpolate;
use crate::silk::structs::SilkEncoderStateCommon;

/// libopus `silk_find_LPC_FLP`: the NLSFs of the frame `x` (`nb_subfr`
/// subframes, each with its `predict_lpc_order` preceding samples), and the
/// NLSF interpolation index that gives the first half the lowest residual
/// energy.
pub fn silk_find_lpc_flp(
    ps_enc_c: &mut SilkEncoderStateCommon,
    nlsf_q15: &mut [i16; MAX_LPC_ORDER],
    x: &[f32],
    min_inv_gain: f32,
) {
    let order = ps_enc_c.predict_lpc_order as usize;
    let subfr_length = ps_enc_c.subfr_length as usize + order;
    let nb_subfr = ps_enc_c.nb_subfr as usize;
    let mut a = [0.0f32; MAX_LPC_ORDER];

    // No interpolation by default.
    ps_enc_c.indices.nlsf_interp_coef_q2 = 4;

    // Burg AR analysis of the full frame.
    let arch = ps_enc_c.arch;
    let mut res_nrg = silk_burg_modified_flp(&mut a, x, min_inv_gain, subfr_length, nb_subfr, order, arch);

    if ps_enc_c.use_interpolated_nlsfs != 0 && ps_enc_c.first_frame_after_reset == 0 && nb_subfr == MAX_NB_SUBFR {
        // The optimum for the last 10 ms; its residual energy is taken out
        // here, so the search compares the first halves.
        let mut a_tmp = [0.0f32; MAX_LPC_ORDER];
        res_nrg -= silk_burg_modified_flp(
            &mut a_tmp,
            &x[(MAX_NB_SUBFR / 2) * subfr_length..],
            min_inv_gain,
            subfr_length,
            MAX_NB_SUBFR / 2,
            order,
            arch,
        );
        silk_a2nlsf_flp(&mut nlsf_q15[..order], &a_tmp[..order]);

        // The interpolation index with the lowest residual energy.
        let mut lpc_res = [0.0f32; MAX_FRAME_LENGTH + MAX_NB_SUBFR * MAX_LPC_ORDER];
        let mut res_nrg_2nd = f32::MAX;
        for k in (0..4).rev() {
            let nlsf0_q15 = silk_interpolate(&ps_enc_c.prev_nlsf_q15, &nlsf_q15[..], k, order);
            silk_nlsf2a_flp(&mut a_tmp[..order], &nlsf0_q15[..order]);
            silk_lpc_analysis_filter_flp(&mut lpc_res, &a_tmp[..order], x, 2 * subfr_length);
            let res_nrg_interp = (silk_energy_flp(&lpc_res[order..subfr_length])
                + silk_energy_flp(&lpc_res[order + subfr_length..2 * subfr_length]))
                as f32;
            if res_nrg_interp < res_nrg {
                res_nrg = res_nrg_interp;
                ps_enc_c.indices.nlsf_interp_coef_q2 = k as i8;
            } else if res_nrg_interp > res_nrg_2nd {
                // The residual energies only rise from here.
                break;
            }
            res_nrg_2nd = res_nrg_interp;
        }
    }

    if ps_enc_c.indices.nlsf_interp_coef_q2 == 4 {
        // No interpolation: the NLSFs of the full frame.
        silk_a2nlsf_flp(&mut nlsf_q15[..order], &a[..order]);
    }
}

/// libopus `silk_residual_energy_FLP`: `nrgs[k]` is the energy of subframe
/// `k`'s residual with the LPC coefficients of its frame half, times its
/// squared gain.
pub fn silk_residual_energy_flp(
    nrgs: &mut [f32; MAX_NB_SUBFR],
    x: &[f32],
    a: &[[f32; MAX_LPC_ORDER]; 2],
    gains: &[f32; MAX_NB_SUBFR],
    subfr_length: usize,
    nb_subfr: usize,
    order: usize,
) {
    let mut lpc_res = [0.0f32; (MAX_FRAME_LENGTH + MAX_NB_SUBFR * MAX_LPC_ORDER) / 2];
    let shift = order + subfr_length;
    for half in 0..nb_subfr / 2 {
        silk_lpc_analysis_filter_flp(&mut lpc_res, &a[half][..order], &x[2 * half * shift..], 2 * shift);
        for s in 0..2 {
            let k = 2 * half + s;
            let res = &lpc_res[order + s * shift..order + s * shift + subfr_length];
            nrgs[k] = (f64::from(gains[k] * gains[k]) * silk_energy_flp(res)) as f32;
        }
    }
}
