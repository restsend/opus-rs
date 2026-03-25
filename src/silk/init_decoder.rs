use crate::silk::decoder_structs::SilkDecoderState;
use crate::silk::define::*;
use crate::silk::tables_nlsf::*;

pub fn silk_decoder_set_fs(dec: &mut SilkDecoderState, fs_khz: i32, fs_api_hz: i32) -> i32 {
    if fs_khz != 8 && fs_khz != 12 && fs_khz != 16 {
        return -1;
    }

    dec.fs_khz = fs_khz;
    dec.fs_api_hz = fs_api_hz;

    dec.nb_subfr = MAX_NB_SUBFR as i32;
    dec.subfr_length = (SUB_FRAME_LENGTH_MS as i32) * fs_khz;
    dec.frame_length = dec.subfr_length * dec.nb_subfr;
    dec.ltp_mem_length = (LTP_MEM_LENGTH_MS as i32) * fs_khz;
    dec.lpc_order = if fs_khz == 8 {
        MIN_LPC_ORDER as i32
    } else {
        MAX_LPC_ORDER as i32
    };

    if fs_khz == 8 {
        if dec.nb_subfr == MAX_NB_SUBFR as i32 {
            dec.pitch_contour_icdf = &crate::silk::tables::SILK_PITCH_CONTOUR_NB_ICDF;
        } else {
            dec.pitch_contour_icdf = &crate::silk::tables::SILK_PITCH_CONTOUR_10_MS_NB_ICDF;
        }
    } else {
        if dec.nb_subfr == MAX_NB_SUBFR as i32 {
            dec.pitch_contour_icdf = &crate::silk::tables::SILK_PITCH_CONTOUR_ICDF;
        } else {
            dec.pitch_contour_icdf = &crate::silk::tables::SILK_PITCH_CONTOUR_10_MS_ICDF;
        }
    }

    dec.pitch_lag_low_bits_icdf = match fs_khz {
        8 => &crate::silk::tables::SILK_UNIFORM4_ICDF,
        12 => &crate::silk::tables::SILK_UNIFORM6_ICDF,
        16 => &crate::silk::tables::SILK_UNIFORM8_ICDF,
        _ => &crate::silk::tables::SILK_UNIFORM8_ICDF,
    };

    dec.ps_nlsf_cb = match fs_khz {
        8 => Some(&SILK_NLSF_CB_NB_MB),
        12 => Some(&SILK_NLSF_CB_NB_MB),
        _ => Some(&SILK_NLSF_CB_WB),
    };

    0
}

pub fn silk_reset_decoder(ps_dec: &mut SilkDecoderState) -> i32 {
    ps_dec.prev_gain_q16 = 1 << 16;
    ps_dec.exc_q14.fill(0);
    ps_dec.s_lpc_q14_buf.fill(0);
    ps_dec.out_buf.fill(0);
    ps_dec.lag_prev = 100;
    ps_dec.last_gain_index = 10;
    ps_dec.first_frame_after_reset = 1;
    ps_dec.ec_prev_signal_type = 0;
    ps_dec.ec_prev_lag_index = 0;
    ps_dec.vad_flags.fill(0);
    ps_dec.lbrr_flag = 0;
    ps_dec.lbrr_flags.fill(0);
    ps_dec.prev_nlsf_q15.fill(0);
    ps_dec.loss_cnt = 0;
    ps_dec.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
    ps_dec.indices = Default::default();

    0
}

pub fn silk_init_decoder(ps_dec: &mut SilkDecoderState) -> i32 {
    silk_reset_decoder(ps_dec);

    ps_dec.s_cng = Default::default();

    ps_dec.s_plc = Default::default();

    0
}

pub fn silk_create_decoder() -> SilkDecoderState {
    let mut dec = SilkDecoderState::default();
    silk_init_decoder(&mut dec);
    dec
}
