use crate::range_coder::RangeCoder;
use crate::silk::decoder_structs::SilkDecoderState;
use crate::silk::define::*;
use crate::silk::nlsf_unpack::silk_nlsf_unpack;
use crate::silk::tables::*;

/// Decode side-information parameters from payload
pub fn silk_decode_indices(
    ps_dec: &mut SilkDecoderState,
    ps_range_dec: &mut RangeCoder,
    frame_index: i32,
    decode_lbrr: i32,
    cond_coding: i32,
) {
    let mut ix: i32;

    /* Decode signal type and quantizer offset */
    if decode_lbrr != 0 || ps_dec.vad_flags[frame_index as usize] != 0 {
        ix = ps_range_dec.decode_icdf(&SILK_TYPE_OFFSET_VAD_ICDF, 8) + 2;
    } else {
        ix = ps_range_dec.decode_icdf(&SILK_TYPE_OFFSET_NO_VAD_ICDF, 8);
    }
    ps_dec.indices.signal_type = (ix >> 1) as i8;
    ps_dec.indices.quant_offset_type = (ix & 1) as i8;

    /* Decode gains */
    /* First subframe */
    if cond_coding == CODE_CONDITIONALLY {
        /* Conditional coding */
        ps_dec.indices.gains_indices[0] = ps_range_dec.decode_icdf(&SILK_DELTA_GAIN_ICDF, 8) as i8;
    } else {
        /* Independent coding, in two stages: MSB bits followed by 3 LSBs */
        ps_dec.indices.gains_indices[0] = (ps_range_dec
            .decode_icdf(&SILK_GAIN_ICDF[ps_dec.indices.signal_type as usize], 8)
            << 3) as i8;
        ps_dec.indices.gains_indices[0] += ps_range_dec.decode_icdf(&SILK_UNIFORM8_ICDF, 8) as i8;
    }

    /* Remaining subframes */
    for i in 1..ps_dec.nb_subfr as usize {
        ps_dec.indices.gains_indices[i] = ps_range_dec.decode_icdf(&SILK_DELTA_GAIN_ICDF, 8) as i8;
    }

    /* Decode LSF Indices */
    let nlsf_cb = ps_dec.ps_nlsf_cb.unwrap();
    ps_dec.indices.nlsf_indices[0] = ps_range_dec.decode_icdf(
        &nlsf_cb.cb1_icdf
            [((ps_dec.indices.signal_type >> 1) as usize) * (nlsf_cb.n_vectors as usize)..],
        8,
    ) as i8;

    let mut ec_ix: [i16; MAX_LPC_ORDER] = [0; MAX_LPC_ORDER];
    let mut pred_q8: [u8; MAX_LPC_ORDER] = [0; MAX_LPC_ORDER];
    silk_nlsf_unpack(
        &mut ec_ix,
        &mut pred_q8,
        nlsf_cb,
        ps_dec.indices.nlsf_indices[0] as usize,
    );

    for i in 0..(nlsf_cb.order as usize) {
        ix = ps_range_dec.decode_icdf(&nlsf_cb.ec_icdf[ec_ix[i] as usize..], 8);
        if ix == 0 {
            ix -= ps_range_dec.decode_icdf(&SILK_NLSF_EXT_ICDF, 8);
        } else if ix == 2 * NLSF_QUANT_MAX_AMPLITUDE {
            ix += ps_range_dec.decode_icdf(&SILK_NLSF_EXT_ICDF, 8);
        }
        ps_dec.indices.nlsf_indices[i + 1] = (ix - NLSF_QUANT_MAX_AMPLITUDE) as i8;
    }

    /* Decode LSF interpolation factor */
    if ps_dec.nb_subfr == MAX_NB_SUBFR as i32 {
        ps_dec.indices.nlsf_interp_coef_q2 =
            ps_range_dec.decode_icdf(&SILK_NLSF_INTERPOLATION_FACTOR_ICDF, 8) as i8;
    } else {
        ps_dec.indices.nlsf_interp_coef_q2 = 4;
    }

    if ps_dec.indices.signal_type == TYPE_VOICED as i8 {
        /* Decode pitch lags */
        let mut decode_absolute_lag_index = 1;

        if cond_coding == CODE_CONDITIONALLY && ps_dec.ec_prev_signal_type == TYPE_VOICED {
            /* Decode Delta index */
            let delta_lag_index = ps_range_dec.decode_icdf(&SILK_PITCH_DELTA_ICDF, 8) as i16;
            if delta_lag_index > 0 {
                ps_dec.indices.lag_index = ps_dec.ec_prev_lag_index + delta_lag_index - 9;
                decode_absolute_lag_index = 0;
            }
        }
        if decode_absolute_lag_index != 0 {
            /* Absolute decoding */
            ps_dec.indices.lag_index = ((ps_range_dec.decode_icdf(&SILK_PITCH_LAG_ICDF, 8) as i32)
                * (ps_dec.fs_khz >> 1)) as i16;
            ps_dec.indices.lag_index +=
                ps_range_dec.decode_icdf(ps_dec.pitch_lag_low_bits_icdf, 8) as i16;
        }
        ps_dec.ec_prev_lag_index = ps_dec.indices.lag_index;

        /* Get contour index */
        ps_dec.indices.contour_index = ps_range_dec.decode_icdf(ps_dec.pitch_contour_icdf, 8) as i8;

        /* Decode LTP gains */
        /* Decode PERIndex value */
        ps_dec.indices.per_index = ps_range_dec.decode_icdf(&SILK_LTP_PER_INDEX_ICDF, 8) as i8;

        for k in 0..ps_dec.nb_subfr as usize {
            ps_dec.indices.ltp_index[k] = ps_range_dec.decode_icdf(
                SILK_LTP_GAIN_ICDF_PTRS[ps_dec.indices.per_index as usize],
                8,
            ) as i8;
        }

        /* Decode LTP scaling */
        if cond_coding == CODE_INDEPENDENTLY {
            ps_dec.indices.ltp_scale_index = ps_range_dec.decode_icdf(&SILK_LTPSCALE_ICDF, 8) as i8;
        } else {
            ps_dec.indices.ltp_scale_index = 0;
        }
    }
    ps_dec.ec_prev_signal_type = ps_dec.indices.signal_type as i32;

    /* Decode seed */
    ps_dec.indices.seed = ps_range_dec.decode_icdf(&SILK_UNIFORM4_ICDF, 8) as i8;
}

/// Decode stereo prediction parameters
pub fn silk_decode_stereo(
    ps_range_dec: &mut RangeCoder,
) -> (i8, i8, i8) {
    // Decode whether we're only decoding mid (mono)
    let only_middle = ps_range_dec.decode_icdf(&SILK_STEREO_ONLY_CODE_MID_ICDF, 8) as i8;

    if only_middle == 0 {
        // Decode the joint stereo index
        let joint_idx = ps_range_dec.decode_icdf(&SILK_STEREO_PRED_JOINT_ICDF, 8) as i8;

        // Extract side_idx and pred_idx from joint index
        // joint_idx = side_idx * 5 + (pred_idx >> 2)
        let side_idx = joint_idx / 5;
        let pred_idx = (joint_idx % 5) * 4; // Scale back to 0-12 (multiply by 4)

        (side_idx, pred_idx, only_middle)
    } else {
        (0, 0, only_middle)
    }
}
