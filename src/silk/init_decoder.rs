use crate::silk::decoder_structs::SilkDecoderState;
use crate::silk::define::*;
use crate::silk::tables_nlsf::*;

pub fn silk_decoder_set_fs(dec: &mut SilkDecoderState, fs_khz: i32, fs_api_hz: i32) -> i32 {
    if fs_khz != 8 && fs_khz != 12 && fs_khz != 16 {
        return -1;
    }

    // Matches the previous behaviour of this port (and C's effective flow):
    // nb_subfr is forced to MAX before the geometry is (re)computed; the
    // 10 ms path in dec_api.rs overrides nb_subfr/frame_length afterwards.
    dec.nb_subfr = MAX_NB_SUBFR as i32;
    dec.subfr_length = (SUB_FRAME_LENGTH_MS as i32) * fs_khz;
    let frame_length = dec.subfr_length * dec.nb_subfr;

    if dec.fs_khz != fs_khz || frame_length != dec.frame_length {
        dec.pitch_contour_icdf = if fs_khz == 8 {
            if dec.nb_subfr == MAX_NB_SUBFR as i32 {
                &crate::silk::tables::SILK_PITCH_CONTOUR_NB_ICDF
            } else {
                &crate::silk::tables::SILK_PITCH_CONTOUR_10_MS_NB_ICDF
            }
        } else if dec.nb_subfr == MAX_NB_SUBFR as i32 {
            &crate::silk::tables::SILK_PITCH_CONTOUR_ICDF
        } else {
            &crate::silk::tables::SILK_PITCH_CONTOUR_10_MS_ICDF
        };

        if dec.fs_khz != fs_khz {
            // Bandwidth switch (libopus decoder_set_fs.c): reset the state that
            // is tied to the old rate. Without this, a stale `lag_prev` (e.g.
            // 288 at 16 kHz) survives a switch to 8 kHz and drives
            // out-of-bounds indexing in silk_decode_core (issue #27 deep scan).
            dec.ltp_mem_length = (LTP_MEM_LENGTH_MS as i32) * fs_khz;
            dec.lpc_order = if fs_khz == 8 {
                MIN_LPC_ORDER as i32
            } else {
                MAX_LPC_ORDER as i32
            };
            dec.pitch_lag_low_bits_icdf = match fs_khz {
                8 => &crate::silk::tables::SILK_UNIFORM4_ICDF,
                12 => &crate::silk::tables::SILK_UNIFORM6_ICDF,
                _ => &crate::silk::tables::SILK_UNIFORM8_ICDF,
            };
            dec.ps_nlsf_cb = Some(if fs_khz == 8 || fs_khz == 12 {
                &SILK_NLSF_CB_NB_MB
            } else {
                &SILK_NLSF_CB_WB
            });
            dec.first_frame_after_reset = 1;
            dec.lag_prev = 100;
            dec.last_gain_index = 10;
            dec.prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
            dec.out_buf.fill(0);
            dec.s_lpc_q14_buf.fill(0);
        }

        dec.fs_khz = fs_khz;
        dec.frame_length = frame_length;
    }

    dec.fs_api_hz = fs_api_hz;

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
