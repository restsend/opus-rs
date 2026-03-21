use crate::range_coder::RangeCoder;
use crate::silk::tables::*;

#[inline]
fn combine_pulses(out: &mut [i32], input: &[i32], len: usize) {
    for k in 0..len {
        out[k] = input[2 * k] + input[2 * k + 1];
    }
}

#[inline]
fn encode_split(ps_range_enc: &mut RangeCoder, p_child1: i32, p: i32, shell_table: &[u8]) {
    if p > 0 {
        ps_range_enc.encode_icdf(
            p_child1,
            &shell_table[SILK_SHELL_CODE_TABLE_OFFSETS[p as usize] as usize..],
            8,
        );
    }
}

pub fn silk_shell_encoder(ps_range_enc: &mut RangeCoder, pulses0: &[i32]) {
    let mut pulses1 = [0i32; 8];
    let mut pulses2 = [0i32; 4];
    let mut pulses3 = [0i32; 2];
    let mut pulses4 = [0i32; 1];

    combine_pulses(&mut pulses1, pulses0, 8);
    combine_pulses(&mut pulses2, &pulses1, 4);
    combine_pulses(&mut pulses3, &pulses2, 2);
    combine_pulses(&mut pulses4, &pulses3, 1);

    encode_split(ps_range_enc, pulses3[0], pulses4[0], &SILK_SHELL_CODE_TABLE3);

    encode_split(ps_range_enc, pulses2[0], pulses3[0], &SILK_SHELL_CODE_TABLE2);

    encode_split(ps_range_enc, pulses1[0], pulses2[0], &SILK_SHELL_CODE_TABLE1);
    encode_split(ps_range_enc, pulses0[0], pulses1[0], &SILK_SHELL_CODE_TABLE0);
    encode_split(ps_range_enc, pulses0[2], pulses1[1], &SILK_SHELL_CODE_TABLE0);

    encode_split(ps_range_enc, pulses1[2], pulses2[1], &SILK_SHELL_CODE_TABLE1);
    encode_split(ps_range_enc, pulses0[4], pulses1[2], &SILK_SHELL_CODE_TABLE0);
    encode_split(ps_range_enc, pulses0[6], pulses1[3], &SILK_SHELL_CODE_TABLE0);

    encode_split(ps_range_enc, pulses2[2], pulses3[1], &SILK_SHELL_CODE_TABLE2);

    encode_split(ps_range_enc, pulses1[4], pulses2[2], &SILK_SHELL_CODE_TABLE1);
    encode_split(ps_range_enc, pulses0[8], pulses1[4], &SILK_SHELL_CODE_TABLE0);
    encode_split(ps_range_enc, pulses0[10], pulses1[5], &SILK_SHELL_CODE_TABLE0);

    encode_split(ps_range_enc, pulses1[6], pulses2[3], &SILK_SHELL_CODE_TABLE1);
    encode_split(ps_range_enc, pulses0[12], pulses1[6], &SILK_SHELL_CODE_TABLE0);
    encode_split(ps_range_enc, pulses0[14], pulses1[7], &SILK_SHELL_CODE_TABLE0);
}
