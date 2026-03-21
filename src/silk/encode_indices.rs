use crate::range_coder::RangeCoder;
use crate::silk::define::*;
use crate::silk::structs::*;
use crate::silk::tables::*;
use crate::silk::nlsf_unpack::silk_nlsf_unpack;

pub fn silk_encode_indices(
    ps_enc_c: &mut SilkEncoderState,
    ps_range_enc: &mut RangeCoder,
    frame_index: usize,
    encode_lbrr: bool,
    cond_coding: i32,
) {

    let ps_indices = if encode_lbrr {
        ps_enc_c.s_cmn.indices_lbrr[frame_index]
    } else {
        ps_enc_c.s_cmn.indices
    };
    let ps_indices = &ps_indices;

    let type_offset = 2 * ps_indices.signal_type + ps_indices.quant_offset_type;
    if encode_lbrr || type_offset >= 2 {
        ps_range_enc.encode_icdf((type_offset - 2) as i32, &SILK_TYPE_OFFSET_VAD_ICDF, 8);
    } else {
        ps_range_enc.encode_icdf(type_offset as i32, &SILK_TYPE_OFFSET_NO_VAD_ICDF, 8);
    }

    if cond_coding == CODE_CONDITIONALLY {

        ps_range_enc.encode_icdf(ps_indices.gains_indices[0] as i32, &SILK_DELTA_GAIN_ICDF, 8);
    } else {

        ps_range_enc.encode_icdf((ps_indices.gains_indices[0] >> 3) as i32, &SILK_GAIN_ICDF[ps_indices.signal_type as usize], 8);
        ps_range_enc.encode_icdf((ps_indices.gains_indices[0] & 7) as i32, &SILK_UNIFORM8_ICDF, 8);
    }

    for i in 1..ps_enc_c.s_cmn.nb_subfr as usize {
        ps_range_enc.encode_icdf(
            ps_indices.gains_indices[i] as i32,
            &SILK_DELTA_GAIN_ICDF,
            8,
        );
    }

    let cb = ps_enc_c.ps_nlsf_cb.expect("NLSF codebook not initialized");
    ps_range_enc.encode_icdf(ps_indices.nlsf_indices[0] as i32, &cb.cb1_icdf[((ps_indices.signal_type >> 1) as usize * cb.n_vectors as usize)..], 8);

    let mut ec_ix = [0i16; MAX_LPC_ORDER];
    let mut pred_q8 = [0u8; MAX_LPC_ORDER];
    silk_nlsf_unpack(
        &mut ec_ix,
        &mut pred_q8,
        cb,
        ps_indices.nlsf_indices[0] as usize,
    );

    for i in 0..cb.order as usize {
        if ps_indices.nlsf_indices[i + 1] >= NLSF_QUANT_MAX_AMPLITUDE as i8 {
            ps_range_enc.encode_icdf(2 * NLSF_QUANT_MAX_AMPLITUDE, &cb.ec_icdf[ec_ix[i] as usize..], 8);
            ps_range_enc.encode_icdf((ps_indices.nlsf_indices[i + 1] - NLSF_QUANT_MAX_AMPLITUDE as i8) as i32, &SILK_NLSF_EXT_ICDF, 8);
        } else if ps_indices.nlsf_indices[i + 1] <= - (NLSF_QUANT_MAX_AMPLITUDE as i8) {
            ps_range_enc.encode_icdf(0, &cb.ec_icdf[ec_ix[i] as usize..], 8);
            ps_range_enc.encode_icdf((-ps_indices.nlsf_indices[i + 1] - NLSF_QUANT_MAX_AMPLITUDE as i8) as i32, &SILK_NLSF_EXT_ICDF, 8);
        } else {
            ps_range_enc.encode_icdf((ps_indices.nlsf_indices[i + 1] + NLSF_QUANT_MAX_AMPLITUDE as i8) as i32, &cb.ec_icdf[ec_ix[i] as usize..], 8);
        }
    }

    if ps_enc_c.s_cmn.nb_subfr == MAX_NB_SUBFR as i32 {
        ps_range_enc.encode_icdf(
            ps_indices.nlsf_interp_coef_q2 as i32,
            &SILK_NLSF_INTERPOLATION_FACTOR_ICDF,
            8,
        );
    }

    if ps_indices.signal_type == TYPE_VOICED as i8 {

        let mut encode_absolute_lag_index = true;
        if cond_coding == CODE_CONDITIONALLY && ps_enc_c.s_cmn.ec_prev_signal_type == TYPE_VOICED {

            let mut delta_lag_index = ps_indices.lag_index as i32 - ps_enc_c.s_cmn.ec_prev_lag_index as i32;
            if delta_lag_index < -8 || delta_lag_index > 11 {
                delta_lag_index = 0;
            } else {
                delta_lag_index += 9;
                encode_absolute_lag_index = false;
            }
            ps_range_enc.encode_icdf(delta_lag_index, &SILK_PITCH_DELTA_ICDF, 8);
        }
        if encode_absolute_lag_index {
            let half_fs = ps_enc_c.s_cmn.fs_khz / 2;
            // Clamp lag_index so pitch_high_bits fits in SILK_PITCH_LAG_ICDF (32 entries)
            // and pitch_low_bits fits in the low-bits table (fs_khz/2 entries).
            let max_lag_index = (SILK_PITCH_LAG_ICDF.len() as i32 - 1) * half_fs + (half_fs - 1);
            let lag_index = (ps_indices.lag_index as i32).min(max_lag_index);

            let pitch_high_bits = lag_index / half_fs;
            let pitch_low_bits = lag_index - pitch_high_bits * half_fs;
            ps_range_enc.encode_icdf(pitch_high_bits, &SILK_PITCH_LAG_ICDF, 8);

            let low_bits_icdf = match ps_enc_c.s_cmn.fs_khz {
                8 => &SILK_UNIFORM4_ICDF[..],
                12 => &SILK_UNIFORM6_ICDF[..],
                16 => &SILK_UNIFORM8_ICDF[..],
                _ => &SILK_UNIFORM8_ICDF[..],
            };
            ps_range_enc.encode_icdf(pitch_low_bits, low_bits_icdf, 8);
        }
        ps_enc_c.s_cmn.ec_prev_lag_index = ps_indices.lag_index;

        let contour_icdf = if ps_enc_c.s_cmn.nb_subfr == 2 {
            if ps_enc_c.s_cmn.fs_khz == 8 {
                &SILK_PITCH_CONTOUR_10_MS_NB_ICDF[..]
            } else {
                &SILK_PITCH_CONTOUR_10_MS_ICDF[..]
            }
        } else {
            if ps_enc_c.s_cmn.fs_khz == 8 {
                &SILK_PITCH_CONTOUR_NB_ICDF[..]
            } else {
                &SILK_PITCH_CONTOUR_ICDF[..]
            }
        };
        ps_range_enc.encode_icdf(ps_indices.contour_index as i32, contour_icdf, 8);

        ps_range_enc.encode_icdf(ps_indices.per_index as i32, &SILK_LTP_PER_INDEX_ICDF, 8);

        for k in 0..ps_enc_c.s_cmn.nb_subfr as usize {
            ps_range_enc.encode_icdf(
                ps_indices.ltp_index[k] as i32,
                SILK_LTP_GAIN_ICDF_PTRS[ps_indices.per_index as usize],
                8,
            );
        }

        if cond_coding == CODE_INDEPENDENTLY {
            ps_range_enc.encode_icdf(ps_indices.ltp_scale_index as i32, &SILK_LTPSCALE_ICDF, 8);
        }
    }

    ps_enc_c.s_cmn.ec_prev_signal_type = ps_indices.signal_type as i32;

    ps_range_enc.encode_icdf(ps_indices.seed as i32, &SILK_UNIFORM4_ICDF, 8);
}

pub fn silk_encode_stereo(
    ps_range_enc: &mut RangeCoder,
    side_idx: i8,
    pred_idx: i8,
    only_middle: i8,
) {

    ps_range_enc.encode_icdf(
        only_middle as i32,
        &SILK_STEREO_ONLY_CODE_MID_ICDF,
        8,
    );

    if only_middle == 0 {

        let i = (side_idx as i32).min(4);
        let j = (pred_idx as i32) >> 2;
        let joint_idx = i * 5 + j;

        ps_range_enc.encode_icdf(joint_idx, &SILK_STEREO_PRED_JOINT_ICDF, 8);
    }
}
