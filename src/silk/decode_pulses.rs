use crate::range_coder::RangeCoder;
use crate::silk::define::*;
use crate::silk::tables::*;

#[inline]
fn decode_split(ps_range_dec: &mut RangeCoder, p: i32, shell_table: &[u8]) -> (i16, i16) {
    if p > 0 {
        let idx = SILK_SHELL_CODE_TABLE_OFFSETS[p as usize] as usize;
        let child1 = ps_range_dec.decode_icdf(&shell_table[idx..], 8) as i16;
        let child2 = (p - child1 as i32) as i16;
        (child1, child2)
    } else {
        (0, 0)
    }
}

fn silk_shell_decoder(pulses0: &mut [i16], ps_range_dec: &mut RangeCoder, pulses4: i32) {
    let mut pulses3: [i16; 2] = [0; 2];
    let mut pulses2: [i16; 4] = [0; 4];
    let mut pulses1: [i16; 8] = [0; 8];

    debug_assert!(SHELL_CODEC_FRAME_LENGTH == 16);

    let (p3_0, p3_1) = decode_split(ps_range_dec, pulses4, &SILK_SHELL_CODE_TABLE3);
    pulses3[0] = p3_0;
    pulses3[1] = p3_1;

    let (p2_0, p2_1) = decode_split(ps_range_dec, pulses3[0] as i32, &SILK_SHELL_CODE_TABLE2);
    pulses2[0] = p2_0;
    pulses2[1] = p2_1;

    let (p1_0, p1_1) = decode_split(ps_range_dec, pulses2[0] as i32, &SILK_SHELL_CODE_TABLE1);
    pulses1[0] = p1_0;
    pulses1[1] = p1_1;

    let (p0_0, p0_1) = decode_split(ps_range_dec, pulses1[0] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[0] = p0_0;
    pulses0[1] = p0_1;

    let (p0_2, p0_3) = decode_split(ps_range_dec, pulses1[1] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[2] = p0_2;
    pulses0[3] = p0_3;

    let (p1_2, p1_3) = decode_split(ps_range_dec, pulses2[1] as i32, &SILK_SHELL_CODE_TABLE1);
    pulses1[2] = p1_2;
    pulses1[3] = p1_3;

    let (p0_4, p0_5) = decode_split(ps_range_dec, pulses1[2] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[4] = p0_4;
    pulses0[5] = p0_5;

    let (p0_6, p0_7) = decode_split(ps_range_dec, pulses1[3] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[6] = p0_6;
    pulses0[7] = p0_7;

    let (p2_2, p2_3) = decode_split(ps_range_dec, pulses3[1] as i32, &SILK_SHELL_CODE_TABLE2);
    pulses2[2] = p2_2;
    pulses2[3] = p2_3;

    let (p1_4, p1_5) = decode_split(ps_range_dec, pulses2[2] as i32, &SILK_SHELL_CODE_TABLE1);
    pulses1[4] = p1_4;
    pulses1[5] = p1_5;

    let (p0_8, p0_9) = decode_split(ps_range_dec, pulses1[4] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[8] = p0_8;
    pulses0[9] = p0_9;

    let (p0_10, p0_11) = decode_split(ps_range_dec, pulses1[5] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[10] = p0_10;
    pulses0[11] = p0_11;

    let (p1_6, p1_7) = decode_split(ps_range_dec, pulses2[3] as i32, &SILK_SHELL_CODE_TABLE1);
    pulses1[6] = p1_6;
    pulses1[7] = p1_7;

    let (p0_12, p0_13) = decode_split(ps_range_dec, pulses1[6] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[12] = p0_12;
    pulses0[13] = p0_13;

    let (p0_14, p0_15) = decode_split(ps_range_dec, pulses1[7] as i32, &SILK_SHELL_CODE_TABLE0);
    pulses0[14] = p0_14;
    pulses0[15] = p0_15;
}

fn silk_decode_signs(
    ps_range_dec: &mut RangeCoder,
    pulses: &mut [i16],
    frame_length: i32,
    signal_type: i32,
    quant_offset_type: i32,
    sum_pulses: &[i32],
) {
    let n_blocks =
        (frame_length + SHELL_CODEC_FRAME_LENGTH as i32 / 2) >> LOG2_SHELL_CODEC_FRAME_LENGTH;

    let i_base = 7 * (quant_offset_type + (signal_type << 1));

    for i in 0..n_blocks as usize {
        let p = sum_pulses[i];
        if p > 0 {

            let idx = (p & 0x1F).min(6) as usize;
            let icdf_0 = SILK_SIGN_ICDF[i_base as usize + idx];

            let icdf = [icdf_0, 0u8];

            let start = i * SHELL_CODEC_FRAME_LENGTH;
            for j in 0..SHELL_CODEC_FRAME_LENGTH {
                if pulses[start + j] > 0 {

                    let sign = ps_range_dec.decode_icdf(&icdf, 8);
                    if sign == 0 {
                        pulses[start + j] = -pulses[start + j];
                    }
                }
            }
        }
    }
}

pub fn silk_decode_pulses(
    ps_range_dec: &mut RangeCoder,
    pulses: &mut [i16],
    signal_type: i32,
    quant_offset_type: i32,
    frame_length: i32,
) {
    let mut sum_pulses: [i32; MAX_NB_SHELL_BLOCKS] = [0; MAX_NB_SHELL_BLOCKS];
    let mut n_lshifts: [i32; MAX_NB_SHELL_BLOCKS] = [0; MAX_NB_SHELL_BLOCKS];

    let rate_level_index =
        ps_range_dec.decode_icdf(&SILK_RATE_LEVELS_ICDF[(signal_type >> 1) as usize], 8) as usize;

    let mut iter = (frame_length as usize) >> LOG2_SHELL_CODEC_FRAME_LENGTH;
    if iter * SHELL_CODEC_FRAME_LENGTH < frame_length as usize {
        iter += 1;
    }

    let cdf_ptr = &SILK_PULSES_PER_BLOCK_ICDF[rate_level_index];
    for i in 0..iter {
        n_lshifts[i] = 0;
        sum_pulses[i] = ps_range_dec.decode_icdf(cdf_ptr, 8) as i32;

        while sum_pulses[i] == (SILK_MAX_PULSES as i32 + 1) {
            n_lshifts[i] += 1;

            if n_lshifts[i] == 10 {
                sum_pulses[i] = ps_range_dec
                    .decode_icdf(&SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1][1..], 8)
                    as i32;
            } else {
                sum_pulses[i] = ps_range_dec
                    .decode_icdf(&SILK_PULSES_PER_BLOCK_ICDF[N_RATE_LEVELS - 1], 8)
                    as i32;
            }
        }
    }

    for i in 0..iter {
        let start = i * SHELL_CODEC_FRAME_LENGTH;
        if sum_pulses[i] > 0 {
            silk_shell_decoder(&mut pulses[start..], ps_range_dec, sum_pulses[i]);
        } else {
            for j in 0..SHELL_CODEC_FRAME_LENGTH {
                pulses[start + j] = 0;
            }
        }
    }

    for i in 0..iter {
        if n_lshifts[i] > 0 {
            let n_ls = n_lshifts[i];
            let start = i * SHELL_CODEC_FRAME_LENGTH;
            for k in 0..SHELL_CODEC_FRAME_LENGTH {
                let mut abs_q = pulses[start + k] as i32;
                for _ in 0..n_ls {
                    abs_q <<= 1;
                    abs_q += ps_range_dec.decode_icdf(&SILK_LSB_ICDF, 8);
                }
                pulses[start + k] = abs_q as i16;
            }

            sum_pulses[i] |= n_ls << 5;
        }
    }

    silk_decode_signs(
        ps_range_dec,
        pulses,
        frame_length,
        signal_type,
        quant_offset_type,
        &sum_pulses,
    );
}

const LOG2_SHELL_CODEC_FRAME_LENGTH: usize = 4;
const MAX_NB_SHELL_BLOCKS: usize =
    (MAX_FRAME_LENGTH + SHELL_CODEC_FRAME_LENGTH - 1) / SHELL_CODEC_FRAME_LENGTH;
