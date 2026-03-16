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

/* Weighting factors for tilt measure */
const TILT_WEIGHTS: [i32; VAD_N_BANDS] = [30000, 6000, -12000, -12000];

fn silk_vad_get_noise_levels(p_x: &[i32; VAD_N_BANDS], ps_silk_vad: &mut SilkVADState) {
    let min_coef;

    /* Initially faster smoothing */
    if ps_silk_vad.counter < 1000 {
        /* 1000 = 20 sec */
        min_coef = silk_div32_16(i16::MAX as i32, (ps_silk_vad.counter >> 4) + 1);
        /* Increment frame counter */
        ps_silk_vad.counter += 1;
    } else {
        min_coef = 0;
    }

    for k in 0..VAD_N_BANDS {
        /* Get old noise level estimate for current band */
        let mut nl = ps_silk_vad.nl[k];
        // silk_assert( nl >= 0 );

        /* Add bias */
        let nrg = silk_add_pos_sat32(p_x[k], ps_silk_vad.noise_level_bias[k]);
        // silk_assert( nrg > 0 );

        /* Invert energies */
        let inv_nrg = silk_div32(i32::MAX, nrg);
        // silk_assert( inv_nrg >= 0 );

        /* Less update when subband energy is high */
        let coef = if nrg > (nl << 3) {
            VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 >> 3
        } else if nrg < nl {
            VAD_NOISE_LEVEL_SMOOTH_COEF_Q16
        } else {
            let tmp = silk_smulww(inv_nrg, nl);
            silk_smulwb(tmp, (VAD_NOISE_LEVEL_SMOOTH_COEF_Q16 as i32) << 1)
        };

        /* Initially faster smoothing */
        let coef = silk_max_int(coef, min_coef);

        /* Smooth inverse energies */
        ps_silk_vad.inv_nl[k] =
            silk_smlawb(ps_silk_vad.inv_nl[k], inv_nrg - ps_silk_vad.inv_nl[k], coef);
        // silk_assert( psSilk_VAD->inv_NL[ k ] >= 0 );

        /* Compute noise level by inverting again */
        nl = silk_div32(i32::MAX, ps_silk_vad.inv_nl[k]);
        // silk_assert( nl >= 0 );

        /* Limit noise levels (guarantee 7 bits of head room) */
        nl = min(nl, 0x00FFFFFF);

        /* Store as part of state */
        ps_silk_vad.nl[k] = nl;
    }
}

pub fn silk_vad_get_sa_q8(ps_enc: &mut SilkEncoderState, p_in: &[i16], _n_in: usize) -> i32 {
    let ps_silk_vad = &mut ps_enc.s_cmn.s_vad;

    // Safety check for buffer sizes
    // In C, scratch memory is allocated on stack or via VarDecl.
    // Here we need to manage temporary buffers.
    // The max frame length is MAX_FRAME_LENGTH.

    let mut x_nrg = [0i32; VAD_N_BANDS];
    let mut nrg_to_noise_ratio_q8 = [0i32; VAD_N_BANDS];

    /* Filter and Decimate */
    // Decimate into 4 bands: 0-8k -> 0-4k (L) & 4-8k (H) -> 0-2k & 2-4k -> 0-1k & 1-2k
    // Based on C VAD.c:
    // dec_framelength1 = sr/2
    // dec_framelength2 = sr/4
    // dec_framelength = sr/8

    // Use fs_khz to derive frame_length for VAD consistency.
    // VAD operates on the internal SILK sample rate, and frame_length should
    // correspond to the current frame duration (typically 10ms for prefill).
    let fs_khz = ps_enc.s_cmn.fs_khz as usize;

    // Validate fs_khz is reasonable (8, 12, or 16 kHz for SILK)
    if fs_khz != 8 && fs_khz != 12 && fs_khz != 16 {
        // Invalid sample rate for VAD
        return 0;
    }

    // For VAD prefill, we use 10ms frame (fs_khz * 10 samples)
    // This matches the prefill behavior in silk_encode_prefill
    let frame_length = fs_khz * 10;

    // Cache all derived values at the start to ensure consistency
    let decimated_framelength1 = frame_length >> 1;
    let decimated_framelength2 = frame_length >> 2;
    let decimated_framelength = frame_length >> 3;

    // Offset calculation (store in array for later indexing)
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

    let mut x = vec![0i16; alloc_size];

    // Note: We need to handle slices carefully to mimic C pointer arithmetic
    // X is used as input and output.
    // ana_filt_bank_1(in, state, outL, outH, N)

    // 0-8 kHz to 0-4 kHz and 4-8 kHz
    // outL goes to X[0] (temporarily), outH goes to X[X_offset_3]
    // But input is p_in which is separate.
    let (x_low, x_high) = x.split_at_mut(x_offset_3);
    silk_ana_filt_bank_1(
        p_in,
        &mut ps_silk_vad.ana_state,
        x_low,
        x_high,
        frame_length,
    );

    // 0-4 kHz to 0-2 kHz and 2-4 kHz
    // Input is X[0..decimated_framelength1].
    // outL goes to X[0] (temporarily), outH goes to X[X_offset_2]
    // Rust borrow checker requires split.
    let (x_part1, x_part2) = x.split_at_mut(x_offset_2);
    // Stack copy to avoid simultaneous mutable + immutable borrow on x_part1.
    // Max decimated_framelength1 = MAX_FRAME_LENGTH/2 = 320 (when frame_length=640, i.e. 40ms@16kHz).
    let mut x_in_buf1 = [0i16; MAX_FRAME_LENGTH / 2];
    x_in_buf1[..decimated_framelength1].copy_from_slice(&x_part1[..decimated_framelength1]);
    silk_ana_filt_bank_1(
        &x_in_buf1[..decimated_framelength1],
        &mut ps_silk_vad.ana_state1,
        x_part1,
        x_part2,
        decimated_framelength1,
    );

    // 0-2 kHz to 0-1 kHz and 1-2 kHz
    // Input is X[0..decimated_framelength2].
    // outL goes to X[0] (temporarily), outH goes to X[X_offset_1]
    let (x_part1, x_part2) = x.split_at_mut(x_offset_1);
    // Stack copy to avoid simultaneous mutable + immutable borrow on x_part1.
    // Max decimated_framelength2 = MAX_FRAME_LENGTH/4 = 160 (when frame_length=640, i.e. 40ms@16kHz).
    let mut x_in_buf2 = [0i16; MAX_FRAME_LENGTH / 4];
    x_in_buf2[..decimated_framelength2].copy_from_slice(&x_part1[..decimated_framelength2]);
    silk_ana_filt_bank_1(
        &x_in_buf2[..decimated_framelength2],
        &mut ps_silk_vad.ana_state2,
        x_part1,
        x_part2,
        decimated_framelength2,
    );

    /*********************************************/
    /* HP filter on lowest band (differentiator) */
    /*********************************************/
    // Lowest band is at X[0...decimated_framelength]
    x[decimated_framelength - 1] >>= 1;
    let hp_state_tmp = x[decimated_framelength - 1];
    for i in (1..decimated_framelength).rev() {
        x[i - 1] >>= 1;
        x[i] -= x[i - 1];
    }
    x[0] -= ps_silk_vad.hp_state;
    ps_silk_vad.hp_state = hp_state_tmp;

    /*************************************/
    /* Calculate the energy in each band */
    /*************************************/
    for b in 0..VAD_N_BANDS {
        /* Find the decimated framelength in the non-uniformly divided bands */
        let dec_framelength_band = frame_length >> min(VAD_N_BANDS - b, VAD_N_BANDS - 1);

        /* Split length into subframe lengths */
        let dec_subframe_length = dec_framelength_band >> VAD_INTERNAL_SUBFRAMES_LOG2;
        let mut dec_subframe_offset = 0;

        /* Compute energy per sub-frame */
        /* initialize with summed energy of last subframe */
        x_nrg[b] = ps_silk_vad.xnrg_subfr[b];

        let mut sum_squared: i32 = 0;
        for s in 0..VAD_INTERNAL_SUBFRAMES {
            sum_squared = 0;
            for i in 0..dec_subframe_length {
                // The energy will be less than dec_subframe_length * ( silk_int16_MIN / 8 ) ^ 2.
                // Therefore we can accumulate with no risk of overflow (unless dec_subframe_length > 128)
                let x_tmp = (x[x_offset[b] + i + dec_subframe_offset] >> 3) as i32;
                sum_squared = silk_smlabb(sum_squared, x_tmp as i32, x_tmp as i32);
            }

            /* Add/saturate summed energy of current subframe */
            if s < VAD_INTERNAL_SUBFRAMES - 1 {
                x_nrg[b] = silk_add_pos_sat32(x_nrg[b], sum_squared);
            } else {
                /* Look-ahead subframe */
                x_nrg[b] = silk_add_pos_sat32(x_nrg[b], sum_squared >> 1);
            }

            dec_subframe_offset += dec_subframe_length;
        }
        ps_silk_vad.xnrg_subfr[b] = sum_squared; // Store last subframe energy for next frame
    }

    // In C, ps_silk_vad.XnrgSubfr[b] stores sumSquared of the last subframe.
    // The code above calculates sumSquared inside the loop. After the loop, sumSquared holds the value for the last subframe (s == VAD_INTERNAL_SUBFRAMES - 1).
    // The previous implementation used sum_squared from the last iteration, which is correct.

    /********************/
    /* Noise estimation */
    /********************/
    silk_vad_get_noise_levels(&x_nrg, ps_silk_vad);

    /***********************************************/
    /* Signal-plus-noise to noise ratio estimation */
    /***********************************************/
    let mut sum_squared: i32 = 0;
    let mut input_tilt: i32 = 0;

    for b in 0..VAD_N_BANDS {
        let speech_nrg = x_nrg[b] - ps_silk_vad.nl[b];
        if speech_nrg > 0 {
            /* Divide, with sufficient resolution */
            if (x_nrg[b] & 0xFF800000_u32 as i32) == 0 {
                nrg_to_noise_ratio_q8[b] = silk_div32(x_nrg[b] << 8, ps_silk_vad.nl[b] + 1);
            } else {
                nrg_to_noise_ratio_q8[b] = silk_div32(x_nrg[b], (ps_silk_vad.nl[b] >> 8) + 1);
            }

            /* Convert to log domain */
            let mut snr_q7 = silk_lin2log(nrg_to_noise_ratio_q8[b]) - 8 * 128;

            /* Sum-of-squares */
            sum_squared = silk_smlabb(sum_squared, snr_q7 as i32, snr_q7 as i32); /* Q14 */

            /* Tilt measure */
            if speech_nrg < (1 << 20) {
                /* Scale down SNR value for small subband speech energies */
                snr_q7 = silk_smulwb(silk_sqrt_approx(speech_nrg) << 6, snr_q7 as i32);
            }
            input_tilt = silk_smlawb(input_tilt, TILT_WEIGHTS[b], snr_q7 as i32);
        } else {
            nrg_to_noise_ratio_q8[b] = 256;
        }
    }

    /* Mean-of-squares */
    sum_squared = silk_div32_16(sum_squared, VAD_N_BANDS as i32); /* Q14 */

    /* Root-mean-square approximation, scale to dBs, and write to output pointer */
    let p_snr_db_q7 = (3 * silk_sqrt_approx(sum_squared)) as i16; /* Q7 */

    /*********************************/
    /* Speech Probability Estimation */
    /*********************************/
    let mut sa_q15 =
        silk_sigm_q15(silk_smulwb(VAD_SNR_FACTOR_Q16, p_snr_db_q7 as i32) - VAD_NEGATIVE_OFFSET_Q5);

    /**************************/
    /* Frequency Tilt Measure */
    /**************************/
    ps_enc.s_cmn.input_tilt_q15 = (silk_sigm_q15(input_tilt) - 16384) << 1;

    /**************************************************/
    /* Scale the sigmoid output based on power levels */
    /**************************************************/
    let mut speech_nrg = 0;
    for b in 0..VAD_N_BANDS {
        /* Accumulate signal-without-noise energies, higher frequency bands have more weight */
        speech_nrg += (b as i32 + 1) * ((x_nrg[b] - ps_silk_vad.nl[b]) >> 4);
    }

    // Check frame length for 20ms frames at specific fs?
    // In C: if( psEncC->frame_length == 20 * psEncC->fs_kHz )
    // frame_length / fs_khz = 20
    if ps_enc.s_cmn.frame_length == 20 * ps_enc.s_cmn.fs_khz {
        speech_nrg >>= 1;
    }

    /* Power scaling */
    if speech_nrg <= 0 {
        sa_q15 >>= 1;
    } else if speech_nrg < 16384 {
        speech_nrg <<= 16;
        /* square-root */
        speech_nrg = silk_sqrt_approx(speech_nrg);
        sa_q15 = silk_smulwb(32768 + speech_nrg, sa_q15 as i32);
    }

    /* Copy the resulting speech activity in Q8 */
    ps_enc.s_cmn.speech_activity_q8 = silk_min_int(sa_q15 >> 7, u8::MAX as i32);

    /***********************************/
    /* Energy Level and SNR estimation */
    /***********************************/
    /* Smoothing coefficient */
    let mut smooth_coef_q16;
    let inner = silk_smulwb(sa_q15, sa_q15 as i32);
    smooth_coef_q16 = silk_smulwb(VAD_SNR_SMOOTH_COEF_Q18, inner as i32);

    if ps_enc.s_cmn.frame_length == 10 * ps_enc.s_cmn.fs_khz {
        smooth_coef_q16 >>= 1;
    }

    for b in 0..VAD_N_BANDS {
        /* compute smoothed energy-to-noise ratio per band */
        ps_silk_vad.nrg_ratio_smth_q8[b] = silk_smlawb(
            ps_silk_vad.nrg_ratio_smth_q8[b],
            nrg_to_noise_ratio_q8[b] - ps_silk_vad.nrg_ratio_smth_q8[b],
            smooth_coef_q16,
        );

        /* signal to noise ratio in dB per band */
        let snr_q7 = 3 * (silk_lin2log(ps_silk_vad.nrg_ratio_smth_q8[b]) - 8 * 128);
        /* quality = sigmoid( 0.25 * ( SNR_dB - 16 ) ); */
        ps_enc.s_cmn.input_quality_bands_q15[b] = silk_sigm_q15((snr_q7 - 16 * 128) >> 4);
    }

    SILK_NO_ERROR
}

pub fn silk_vad_init(s_vad: &mut SilkVADState) -> i32 {
    let ret = SILK_NO_ERROR;

    // reset state memory
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

    /* init noise levels */
    /* Initialize array with approx pink noise levels (psd proportional to inverse of frequency) */
    for b in 0..VAD_N_BANDS {
        s_vad.noise_level_bias[b] = max(VAD_NOISE_LEVELS_BIAS / (b as i32 + 1), 1);
    }

    /* Initialize state */
    for b in 0..VAD_N_BANDS {
        s_vad.nl[b] = 100 * s_vad.noise_level_bias[b];
        s_vad.inv_nl[b] = i32::MAX / s_vad.nl[b];
    }
    s_vad.counter = 15;

    /* init smoothed energy-to-noise ratio*/
    for b in 0..VAD_N_BANDS {
        s_vad.nrg_ratio_smth_q8[b] = 100 * 256; /* 100 * 256 --> 20 dB SNR */
    }

    ret
}
