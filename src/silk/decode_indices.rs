use crate::range_coder::RangeCoder;
use crate::silk::decoder_structs::SilkDecoderState;
use crate::silk::define::*;
use crate::silk::macros::{silk_smlabb, silk_smulwb};
use crate::silk::nlsf_unpack::silk_nlsf_unpack;
use crate::silk::tables::*;

pub fn silk_decode_indices(
    ps_dec: &mut SilkDecoderState,
    ps_range_dec: &mut RangeCoder,
    frame_index: i32,
    decode_lbrr: i32,
    cond_coding: i32,
) {
    let mut ix: i32;

    if decode_lbrr != 0 || ps_dec.vad_flags[frame_index as usize] != 0 {
        ix = ps_range_dec.decode_icdf(&SILK_TYPE_OFFSET_VAD_ICDF, 8) + 2;
    } else {
        ix = ps_range_dec.decode_icdf(&SILK_TYPE_OFFSET_NO_VAD_ICDF, 8);
    }
    ps_dec.indices.signal_type = (ix >> 1) as i8;
    ps_dec.indices.quant_offset_type = (ix & 1) as i8;

    if cond_coding == CODE_CONDITIONALLY {
        ps_dec.indices.gains_indices[0] = ps_range_dec.decode_icdf(&SILK_DELTA_GAIN_ICDF, 8) as i8;
    } else {
        ps_dec.indices.gains_indices[0] = (ps_range_dec
            .decode_icdf(&SILK_GAIN_ICDF[ps_dec.indices.signal_type as usize], 8)
            << 3) as i8;
        ps_dec.indices.gains_indices[0] += ps_range_dec.decode_icdf(&SILK_UNIFORM8_ICDF, 8) as i8;
    }

    for i in 1..ps_dec.nb_subfr as usize {
        ps_dec.indices.gains_indices[i] = ps_range_dec.decode_icdf(&SILK_DELTA_GAIN_ICDF, 8) as i8;
    }

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

    if ps_dec.nb_subfr == MAX_NB_SUBFR as i32 {
        ps_dec.indices.nlsf_interp_coef_q2 =
            ps_range_dec.decode_icdf(&SILK_NLSF_INTERPOLATION_FACTOR_ICDF, 8) as i8;
    } else {
        ps_dec.indices.nlsf_interp_coef_q2 = 4;
    }

    if ps_dec.indices.signal_type == TYPE_VOICED as i8 {
        let mut decode_absolute_lag_index = 1;

        if cond_coding == CODE_CONDITIONALLY && ps_dec.ec_prev_signal_type == TYPE_VOICED {
            let delta_lag_index = ps_range_dec.decode_icdf(&SILK_PITCH_DELTA_ICDF, 8) as i16;
            if delta_lag_index > 0 {
                ps_dec.indices.lag_index = ps_dec.ec_prev_lag_index + delta_lag_index - 9;
                decode_absolute_lag_index = 0;
            }
        }
        if decode_absolute_lag_index != 0 {
            ps_dec.indices.lag_index =
                ((ps_range_dec.decode_icdf(&SILK_PITCH_LAG_ICDF, 8)) * (ps_dec.fs_khz >> 1)) as i16;
            ps_dec.indices.lag_index +=
                ps_range_dec.decode_icdf(ps_dec.pitch_lag_low_bits_icdf, 8) as i16;
        }
        ps_dec.ec_prev_lag_index = ps_dec.indices.lag_index;

        ps_dec.indices.contour_index = ps_range_dec.decode_icdf(ps_dec.pitch_contour_icdf, 8) as i8;

        ps_dec.indices.per_index = ps_range_dec.decode_icdf(&SILK_LTP_PER_INDEX_ICDF, 8) as i8;

        for k in 0..ps_dec.nb_subfr as usize {
            ps_dec.indices.ltp_index[k] = ps_range_dec.decode_icdf(
                SILK_LTP_GAIN_ICDF_PTRS[ps_dec.indices.per_index as usize],
                8,
            ) as i8;
        }

        if cond_coding == CODE_INDEPENDENTLY {
            ps_dec.indices.ltp_scale_index = ps_range_dec.decode_icdf(&SILK_LTPSCALE_ICDF, 8) as i8;
        } else {
            ps_dec.indices.ltp_scale_index = 0;
        }
    }
    ps_dec.ec_prev_signal_type = ps_dec.indices.signal_type as i32;

    ps_dec.indices.seed = ps_range_dec.decode_icdf(&SILK_UNIFORM4_ICDF, 8) as i8;
}

pub fn silk_stereo_decode_pred(ps_range_dec: &mut RangeCoder) -> [i32; 2] {
    let mut pred_q13 = [0i32; 2];

    // Entropy decoding
    let n = ps_range_dec.decode_icdf(&SILK_STEREO_PRED_JOINT_ICDF, 8);
    let mut ix = [[0i32; 3]; 2];
    ix[0][2] = n / 5;
    ix[1][2] = n - 5 * ix[0][2];
    for i in 0..2 {
        ix[i][0] = ps_range_dec.decode_icdf(&SILK_UNIFORM3_ICDF, 8);
        ix[i][1] = ps_range_dec.decode_icdf(&SILK_UNIFORM5_ICDF, 8);
    }

    // Dequantize
    for i in 0..2 {
        ix[i][0] += 3 * ix[i][2];
        let low_q13 = SILK_STEREO_PRED_QUANT_Q13[ix[i][0] as usize] as i32;
        let step_q13 = silk_smulwb(
            SILK_STEREO_PRED_QUANT_Q13[(ix[i][0] + 1) as usize] as i32 - low_q13,
            STEREO_HALF_SUB_STEP_Q16,
        );
        pred_q13[i] = silk_smlabb(low_q13, step_q13, 2 * ix[i][1] + 1);
    }

    // Subtract second from first predictor
    pred_q13[0] -= pred_q13[1];
    pred_q13
}

pub fn silk_stereo_decode_mid_only(ps_range_dec: &mut RangeCoder) -> bool {
    ps_range_dec.decode_icdf(&SILK_STEREO_ONLY_CODE_MID_ICDF, 8) != 0
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    /// Every stereo predictor level, from libopus's quantizer
    /// (stereo_quant_pred.c): `LEVELS_Q13[i][j]` is quantization interval
    /// `i`, sub-step `j`.
    const LEVELS_Q13: [[i32; 5]; 15] = [
        [-13364, -12628, -11892, -11156, -10420],
        [-9872, -9516, -9160, -8804, -8448],
        [-8192, -8044, -7896, -7748, -7600],
        [-7424, -7220, -7016, -6812, -6608],
        [-6350, -6050, -5750, -5450, -5150],
        [-4795, -4385, -3975, -3565, -3155],
        [-2737, -2311, -1885, -1459, -1033],
        [-656, -328, 0, 328, 656],
        [1033, 1459, 1885, 2311, 2737],
        [3155, 3565, 3975, 4385, 4795],
        [5150, 5450, 5750, 6050, 6350],
        [6602, 6806, 7010, 7214, 7418],
        [7600, 7748, 7896, 8044, 8192],
        [8444, 8800, 9156, 9512, 9868],
        [10418, 11154, 11890, 12626, 13362],
    ];

    /// Range-code two predictors as libopus's `silk_stereo_encode_pred`
    /// does, each given as (interval, sub-step), and decode them back.
    fn decode(pred: [(usize, usize); 2]) -> [i32; 2] {
        let mut enc = RangeCoder::new_encoder(64);
        let joint = 5 * (pred[0].0 / 3) + pred[1].0 / 3;
        enc.encode_icdf(joint as i32, &SILK_STEREO_PRED_JOINT_ICDF, 8);
        for (i, j) in pred {
            enc.encode_icdf((i % 3) as i32, &SILK_UNIFORM3_ICDF, 8);
            enc.encode_icdf(j as i32, &SILK_UNIFORM5_ICDF, 8);
        }
        let bytes = enc.finish();
        silk_stereo_decode_pred(&mut RangeCoder::new_decoder(&bytes))
    }

    #[test]
    fn stereo_pred_levels_match_libopus() {
        for (i, row) in LEVELS_Q13.iter().enumerate() {
            for (j, &level) in row.iter().enumerate() {
                // The second predictor is subtracted from the first; keep it
                // at the zero level so the first reads back unchanged.
                assert_eq!(decode([(i, j), (7, 2)]), [level, 0], "level ({i}, {j})");
            }
        }
    }

    #[test]
    fn stereo_pred_zero_level_is_neutral() {
        assert_eq!(decode([(7, 2), (7, 2)]), [0, 0]);
    }
}
