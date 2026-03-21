use crate::range_coder::RangeCoder;
use crate::silk::decode_core::silk_decode_core;
use crate::silk::decode_indices::silk_decode_indices;
use crate::silk::decode_parameters::silk_decode_parameters;
use crate::silk::decode_pulses::silk_decode_pulses;
use crate::silk::decoder_structs::{SilkDecoderControl, SilkDecoderState};
use crate::silk::define::*;

pub const FLAG_DECODE_NORMAL: i32 = 0;
pub const FLAG_PACKET_LOST: i32 = 1;
pub const FLAG_DECODE_LBRR: i32 = 2;

pub fn silk_decode_frame(
    ps_dec: &mut SilkDecoderState,
    ps_range_dec: &mut RangeCoder,
    p_out: &mut [i16],
    p_n: &mut i32,
    lost_flag: i32,
    cond_coding: i32,
) -> i32 {
    let l = ps_dec.frame_length as usize;
    let mut ps_dec_ctrl = SilkDecoderControl::new();
    ps_dec_ctrl.ltp_scale_q14 = 0;

    debug_assert!(l > 0 && l <= MAX_FRAME_LENGTH);

    if lost_flag == FLAG_DECODE_NORMAL
        || (lost_flag == FLAG_DECODE_LBRR
            && ps_dec.lbrr_flags[ps_dec.n_frames_decoded as usize] == 1)
    {
        let mut pulses: [i16; MAX_FRAME_LENGTH] = [0; MAX_FRAME_LENGTH];

        silk_decode_indices(
            ps_dec,
            ps_range_dec,
            ps_dec.n_frames_decoded,
            lost_flag,
            cond_coding,
        );

        silk_decode_pulses(
            ps_range_dec,
            &mut pulses,
            ps_dec.indices.signal_type as i32,
            ps_dec.indices.quant_offset_type as i32,
            ps_dec.frame_length,
        );

        silk_decode_parameters(ps_dec, &mut ps_dec_ctrl, cond_coding);

        silk_decode_core(ps_dec, &ps_dec_ctrl, p_out, &pulses);

        let mv_len = ps_dec.ltp_mem_length - ps_dec.frame_length;
        ps_dec.out_buf.rotate_left(ps_dec.frame_length as usize);
        ps_dec.out_buf[mv_len as usize..mv_len as usize + l].copy_from_slice(&p_out[..l]);

        ps_dec.loss_cnt = 0;
        ps_dec.prev_signal_type = ps_dec.indices.signal_type as i32;

        ps_dec.first_frame_after_reset = 0;
    } else {

        for i in 0..l {
            p_out[i] = 0;
        }

        let mv_len = ps_dec.ltp_mem_length - ps_dec.frame_length;
        ps_dec.out_buf.rotate_left(ps_dec.frame_length as usize);
        ps_dec.out_buf[mv_len as usize..mv_len as usize + l].copy_from_slice(&p_out[..l]);

        ps_dec.loss_cnt += 1;
    }

    ps_dec.lag_prev = ps_dec_ctrl.pitch_l[ps_dec.nb_subfr as usize - 1];

    *p_n = l as i32;

    0
}
