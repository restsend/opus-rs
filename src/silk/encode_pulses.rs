use crate::range_coder::RangeCoder;
use crate::silk::define::*;
use crate::silk::shell_coder::silk_shell_encoder;
use crate::silk::tables::*;

fn combine_and_check(
    pulses_comb: &mut [i32],
    pulses_in: &[i32],
    max_pulses: i32,
    len: usize,
) -> i32 {
    for k in 0..len {
        let sum = pulses_in[2 * k] + pulses_in[2 * k + 1];
        if sum > max_pulses {
            return 1;
        }
        pulses_comb[k] = sum;
    }
    0
}

pub fn silk_encode_pulses(
    ps_range_enc: &mut RangeCoder,
    signal_type: i32,
    quant_offset_type: i32,
    pulses: &[i8],
    frame_length: usize,
) {
    let mut pulses_comb = [0i32; 8];
    // Stack-allocated fixed-size buffers: frame_length ≤ MAX_FRAME_LENGTH (640).
    let mut abs_pulses = [0i32; MAX_FRAME_LENGTH + SHELL_CODEC_FRAME_LENGTH];

    let iter = (frame_length + SHELL_CODEC_FRAME_LENGTH - 1) / SHELL_CODEC_FRAME_LENGTH;

    for i in 0..frame_length {
        abs_pulses[i] = pulses[i].abs() as i32;
    }

    // iter ≤ MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH = 640 / 16 = 40.
    let mut sum_pulses = [0i32; MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH];
    let mut n_rshifts = [0i32; MAX_FRAME_LENGTH / SHELL_CODEC_FRAME_LENGTH];

    for i in 0..iter {
        let abs_pulses_ptr = &mut abs_pulses[i * SHELL_CODEC_FRAME_LENGTH..];
        n_rshifts[i] = 0;

        loop {
            let mut scale_down = combine_and_check(
                &mut pulses_comb,
                abs_pulses_ptr,
                SILK_MAX_PULSES_TABLE[0] as i32,
                8,
            );

            let pulses_temp8 = pulses_comb;
            scale_down += combine_and_check(
                &mut pulses_comb,
                &pulses_temp8,
                SILK_MAX_PULSES_TABLE[1] as i32,
                4,
            );

            let pulses_temp4 = pulses_comb;
            scale_down += combine_and_check(
                &mut pulses_comb,
                &pulses_temp4,
                SILK_MAX_PULSES_TABLE[2] as i32,
                2,
            );

            let pulses_temp2 = pulses_comb;
            let mut sum_pulse_val = [sum_pulses[i]];
            scale_down += combine_and_check(
                &mut sum_pulse_val,
                &pulses_temp2,
                SILK_MAX_PULSES_TABLE[3] as i32,
                1,
            );
            sum_pulses[i] = sum_pulse_val[0];

            if scale_down > 0 {
                n_rshifts[i] += 1;
                for k in 0..SHELL_CODEC_FRAME_LENGTH {
                    abs_pulses_ptr[k] >>= 1;
                }
            } else {
                break;
            }
        }
    }

    /* Rate level */
    let mut min_sum_bits_q5 = i32::MAX;
    let mut rate_level_index = 0;
    for k in 0..N_RATE_LEVELS - 1 {
        let n_bits_ptr = &SILK_PULSES_PER_BLOCK_BITS_Q5[k];
        let mut sum_bits_q5 = SILK_RATE_LEVELS_BITS_Q5[(signal_type >> 1) as usize][k] as i32;
        for i in 0..iter {
            if n_rshifts[i] > 0 {
                sum_bits_q5 += n_bits_ptr[SILK_MAX_PULSES + 1] as i32;
            } else {
                sum_bits_q5 += n_bits_ptr[sum_pulses[i] as usize] as i32;
            }
        }
        if sum_bits_q5 < min_sum_bits_q5 {
            min_sum_bits_q5 = sum_bits_q5;
            rate_level_index = k;
        }
    }
    ps_range_enc.encode_icdf(
        rate_level_index as i32,
        &SILK_RATE_LEVELS_ICDF[(signal_type >> 1) as usize],
        8,
    );

    /* Sum-Weighted-Pulses Encoding */
    let cdf_ptr = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index];
    for i in 0..iter {
        if n_rshifts[i] == 0 {
            ps_range_enc.encode_icdf(sum_pulses[i], cdf_ptr, 8);
        } else {
            ps_range_enc.encode_icdf(SILK_MAX_PULSES as i32 + 1, cdf_ptr, 8);
            for _ in 0..n_rshifts[i] - 1 {
                ps_range_enc.encode_icdf(
                    SILK_MAX_PULSES as i32 + 1,
                    &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1],
                    8,
                );
            }
            ps_range_enc.encode_icdf(
                sum_pulses[i],
                &SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1],
                8,
            );
        }
    }

    /* Shell Encoding */
    for i in 0..iter {
        if sum_pulses[i] > 0 {
            silk_shell_encoder(
                ps_range_enc,
                &abs_pulses[i * SHELL_CODEC_FRAME_LENGTH..(i + 1) * SHELL_CODEC_FRAME_LENGTH],
            );
        }
    }

    /* LSB Encoding */
    for i in 0..iter {
        if n_rshifts[i] > 0 {
            let n_ls = n_rshifts[i] - 1;
            for k in 0..SHELL_CODEC_FRAME_LENGTH {
                let abs_q = pulses[i * SHELL_CODEC_FRAME_LENGTH + k].abs() as i32;
                for j in (1..=n_ls).rev() {
                    let bit = (abs_q >> j) & 1;
                    ps_range_enc.encode_icdf(bit, &SILK_LSB_ICDF, 8);
                }
                let bit = abs_q & 1;
                ps_range_enc.encode_icdf(bit, &SILK_LSB_ICDF, 8);
            }
        }
    }

    /* Encode Signs */
    // Select ICDF table offset based on signalType and quantOffsetType:
    // i = 7 * (quantOffsetType + signalType * 2)
    let icdf_offset = (7 * (quant_offset_type + signal_type * 2)) as usize;
    for i in 0..iter {
        let p = sum_pulses[i];
        if p > 0 {
            // Select probability from table using clamped (p & 0x1F) as index
            let icdf0 = SILK_SIGN_ICDF[icdf_offset + ((p & 0x1F) as usize).min(6)];
            let icdf = [icdf0, 0u8];
            for j in 0..SHELL_CODEC_FRAME_LENGTH {
                let pulse = pulses[i * SHELL_CODEC_FRAME_LENGTH + j];
                if pulse != 0 {
                    // silk_enc_map: positive → 1, negative → 0
                    let mapped = if pulse > 0 { 1i32 } else { 0i32 };
                    ps_range_enc.encode_icdf(mapped, &icdf, 8);
                }
            }
        }
    }
}
