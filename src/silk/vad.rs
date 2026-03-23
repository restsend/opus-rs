use crate::silk::ana_filt_bank_1::silk_ana_filt_bank_1;
use crate::silk::define::{
    MAX_FRAME_LENGTH, SILK_NO_ERROR, VAD_INTERNAL_SUBFRAMES, VAD_INTERNAL_SUBFRAMES_LOG2,
    VAD_N_BANDS, VAD_NEGATIVE_OFFSET_Q5, VAD_NOISE_LEVEL_SMOOTH_COEF_Q16, VAD_NOISE_LEVELS_BIAS,
    VAD_SNR_FACTOR_Q16, VAD_SNR_SMOOTH_COEF_Q18,
};
use crate::silk::lin2log::silk_lin2log;
use crate::silk::macros::{
    silk_add_pos_sat32, silk_div32, silk_div32_16, silk_max_int, silk_min_int, silk_smlabb,
    silk_smlawb, silk_smulwb, silk_smulww, silk_sqrt_approx,
};
use crate::silk::sigm::silk_sigm_q15;
use crate::silk::structs::{SilkEncoderState, SilkVADState};
use std::cmp::{max, min};

const TILT_WEIGHTS: [i32; VAD_N_BANDS] = [30000, 6000, -12000, -12000];

fn silk_vad_get_noise_levels(p_x: &[i32; VAD_N_BANDS], ps_silk_vad: &mut SilkVADState) {
    let min_coef;

    if ps_silk_vad.counter < 1000 {

        min_coef = silk_div32_16(i16::MAX as i32, (ps_silk_vad.counter >> 4) + 1);

        ps_silk_vad.counter += 1;
    } else {
        min_coef = 0;
    }

    for k in 0..VAD_N_BANDS {

        let mut nl = ps_silk_vad.nl[k];

        let nrg = silk_add_pos_sat32(p_x[k], ps_silk_vad.noise_level_bias[k]);

        let inv_nrg = silk_div32(i32::MAX, nrg);

        let coef = if nrg > (nl << 3) {
            VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 >> 3
        } else if nrg < nl {
            VAD_NOISE_LEVEL_SMOOTH_COEF_Q16
        } else {
            let tmp = silk_smulww(inv_nrg, nl);
            silk_smulwb(tmp, (VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 as i32) << 1)
        };

        let coef = silk_max_int(coef, min_coef);

        ps_silk_vad.inv_nl[k] =
            silk_smlawb(ps_silk_vad.inv_nl[k], inv_nrg - ps_silk_vad.inv_nl[k], coef);

        nl = silk_div32(i32::MAX, ps_silk_vad.inv_nl[k]);

        nl = min(nl, 0x00FFFFFF);

        ps_silk_vad.nl[k] = nl;
    }
}

pub fn silk_vad_get_sa_q8(ps_enc: &mut SilkEncoderState, p_in: &[i16], _n_in: usize) -> i32 {
    let ps_silk_vad = &mut ps_enc.s_cmn.s_vad;

    let mut x_nrg = [0i32; VAD_N_BANDS];
    let mut nrg_to_noise_ratio_q8 = [0i32; VAD_N_BANDS];

    let fs_khz = ps_enc.s_cmn.fs_khz as usize;

    if fs_khz != 8 && fs_khz != 12 && fs_khz != 16 {

        return 0;
    }

    let frame_length = p_in.len();

    let decimated_framelength1 = frame_length >> 1;
    let decimated_framelength2 = frame_length >> 2;
    let decimated_framelength = frame_length >> 3;

    let x_offset = [
        0,
        decimated_framelength + decimated_framelength2,
        decimated_framelength + decimated_framelength2 + decimated_framelength,
        decimated_framelength + decimated_framelength2 + decimated_framelength + decimated_framelength2,
    ];
    let x_offset_1 = x_offset[1];
    let x_offset_2 = x_offset[2];
    let x_offset_3 = x_offset[3];

    let alloc_size = x_offset_3 + decimated_framelength1;

    // MAX_FRAME_LENGTH = 320, so alloc_size ≤ (320/8 + 320/4)*2 + 320/2 = 400
    const MAX_VAD_X_SIZE: usize = 400;
    debug_assert!(alloc_size <= MAX_VAD_X_SIZE);
    let mut x_buf = [0i16; MAX_VAD_X_SIZE];
    let x = &mut x_buf[..alloc_size];

    let (x_low, x_high) = x.split_at_mut(x_offset_3);
    silk_ana_filt_bank_1(
        p_in,
        &mut ps_silk_vad.ana_state,
        x_low,
        x_high,
        frame_length,
    );

    let (x_part1, x_part2) = x.split_at_mut(x_offset_2);

    let mut x_in_buf1 = [0i16; MAX_FRAME_LENGTH / 2];
    x_in_buf1[..decimated_framelength1].copy_from_slice(&x_part1[..decimated_framelength1]);
    silk_ana_filt_bank_1(
        &x_in_buf1[..decimated_framelength1],
        &mut ps_silk_vad.ana_state1,
        x_part1,
        x_part2,
        decimated_framelength1,
    );

    let (x_part1, x_part2) = x.split_at_mut(x_offset_1);

    let mut x_in_buf2 = [0i16; MAX_FRAME_LENGTH / 4];
    x_in_buf2[..decimated_framelength2].copy_from_slice(&x_part1[..decimated_framelength2]);
    silk_ana_filt_bank_1(
        &x_in_buf2[..decimated_framelength2],
        &mut ps_silk_vad.ana_state2,
        x_part1,
        x_part2,
        decimated_framelength2,
    );

    x[decimated_framelength - 1] >>= 1;
    let hp_state_tmp = x[decimated_framelength - 1];
    for i in (1..decimated_framelength).rev() {
        x[i - 1] >>= 1;
        x[i] -= x[i - 1];
    }
    x[0] -= ps_silk_vad.hp_state;
    ps_silk_vad.hp_state = hp_state_tmp;

    for b in 0..VAD_N_BANDS {

        let dec_framelength_band = frame_length >> min(VAD_N_BANDS - b, VAD_N_BANDS - 1);

        let dec_subframe_length = dec_framelength_band >> VAD_INTERNAL_SUBFRAMES_LOG2;
        let mut dec_subframe_offset = 0;

        x_nrg[b] = ps_silk_vad.xnrg_subfr[b];

        let mut sum_squared: i32 = 0;
        for s in 0..VAD_INTERNAL_SUBFRAMES {
            sum_squared = 0;
            for i in 0..dec_subframe_length {

                let x_tmp = (x[x_offset[b] + i + dec_subframe_offset] >> 3) as i32;
                sum_squared = silk_smlabb(sum_squared, x_tmp as i32, x_tmp as i32);
            }

            if s < VAD_INTERNAL_SUBFRAMES - 1 {
                x_nrg[b] = silk_add_pos_sat32(x_nrg[b], sum_squared);
            } else {

                x_nrg[b] = silk_add_pos_sat32(x_nrg[b], sum_squared >> 1);
            }

            dec_subframe_offset += dec_subframe_length;
        }
        ps_silk_vad.xnrg_subfr[b] = sum_squared;
    }

    silk_vad_get_noise_levels(&x_nrg, ps_silk_vad);

    let mut sum_squared: i32 = 0;
    let mut input_tilt: i32 = 0;

    for b in 0..VAD_N_BANDS {
        let speech_nrg = x_nrg[b] - ps_silk_vad.nl[b];
        if speech_nrg > 0 {

            if (x_nrg[b] & 0xFF800000_u32 as i32) == 0 {
                nrg_to_noise_ratio_q8[b] = silk_div32(x_nrg[b] << 8, ps_silk_vad.nl[b] + 1);
            } else {
                nrg_to_noise_ratio_q8[b] = silk_div32(x_nrg[b], (ps_silk_vad.nl[b] >> 8) + 1);
            }

            let mut snr_q7 = silk_lin2log(nrg_to_noise_ratio_q8[b]) - 8 * 128;

            sum_squared = silk_smlabb(sum_squared, snr_q7 as i32, snr_q7 as i32);

            if speech_nrg < (1 << 20) {

                snr_q7 = silk_smulwb(silk_sqrt_approx(speech_nrg) << 6, snr_q7 as i32);
            }
            input_tilt = silk_smlawb(input_tilt, TILT_WEIGHTS[b], snr_q7 as i32);
        } else {
            nrg_to_noise_ratio_q8[b] = 256;
        }
    }

    sum_squared = silk_div32_16(sum_squared, VAD_N_BANDS as i32);

    let p_snr_db_q7 = (3 * silk_sqrt_approx(sum_squared)) as i16;

    let mut sa_q15 =
        silk_sigm_q15(silk_smulwb(VAD_SNR_FACTOR_Q16, p_snr_db_q7 as i32) - VAD_NEGATIVE_OFFSET_Q5);

    ps_enc.s_cmn.input_tilt_q15 = (silk_sigm_q15(input_tilt) - 16384) << 1;

    let mut speech_nrg = 0;
    for b in 0..VAD_N_BANDS {

        speech_nrg += (b as i32 + 1) * ((x_nrg[b] - ps_silk_vad.nl[b]) >> 4);
    }

    if ps_enc.s_cmn.frame_length == 20 * ps_enc.s_cmn.fs_khz {
        speech_nrg >>= 1;
    }

    if speech_nrg <= 0 {
        sa_q15 >>= 1;
    } else if speech_nrg < 16384 {
        speech_nrg <<= 16;

        speech_nrg = silk_sqrt_approx(speech_nrg);
        sa_q15 = silk_smulwb(32768 + speech_nrg, sa_q15 as i32);
    }

    ps_enc.s_cmn.speech_activity_q8 = silk_min_int(sa_q15 >> 7, u8::MAX as i32);

    let mut smooth_coef_q16;
    let inner = silk_smulwb(sa_q15, sa_q15 as i32);
    smooth_coef_q16 = silk_smulwb(VAD_SNR_SMOOTH_COEF_Q18, inner as i32);

    if ps_enc.s_cmn.frame_length == 10 * ps_enc.s_cmn.fs_khz {
        smooth_coef_q16 >>= 1;
    }

    for b in 0..VAD_N_BANDS {

        ps_silk_vad.nrg_ratio_smth_q8[b] = silk_smlawb(
            ps_silk_vad.nrg_ratio_smth_q8[b],
            nrg_to_noise_ratio_q8[b] - ps_silk_vad.nrg_ratio_smth_q8[b],
            smooth_coef_q16,
        );

        let snr_q7 = 3 * (silk_lin2log(ps_silk_vad.nrg_ratio_smth_q8[b]) - 8 * 128);

        ps_enc.s_cmn.input_quality_bands_q15[b] = silk_sigm_q15((snr_q7 - 16 * 128) >> 4);
    }

    SILK_NO_ERROR
}

pub fn silk_vad_init(s_vad: &mut SilkVADState) -> i32 {
    let ret = SILK_NO_ERROR;

    *s_vad = SilkVADState {
        ana_state: [0; 2],
        ana_state1: [0; 2],
        ana_state2: [0; 2],
        xnrg_subfr: [0; VAD_N_BANDS],
        nrg_ratio_smth_q8: [0; VAD_N_BANDS],
        hp_state: 0,
        nl: [0; VAD_N_BANDS],
        inv_nl: [0; VAD_N_BANDS],
        noise_level_bias: [0; VAD_N_BANDS],
        counter: 0,
    };

    for b in 0..VAD_N_BANDS {
        s_vad.noise_level_bias[b] = max(VAD_NOISE_LEVELS_BIAS / (b as i32 + 1), 1);
    }

    for b in 0..VAD_N_BANDS {
        s_vad.nl[b] = 100 * s_vad.noise_level_bias[b];
        s_vad.inv_nl[b] = i32::MAX / s_vad.nl[b];
    }
    s_vad.counter = 15;

    for b in 0..VAD_N_BANDS {
        s_vad.nrg_ratio_smth_q8[b] = 100 * 256;
    }

    ret
}
