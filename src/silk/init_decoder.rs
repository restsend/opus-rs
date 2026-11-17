use crate::silk::decoder_structs::SilkDecoderState;
use crate::silk::define::*;
use crate::silk::tables_nlsf::*;
use crate::silk::cng::silk_cng_reset;
use crate::silk::plc::silk_plc_reset;

pub fn silk_decoder_set_fs(dec: &mut SilkDecoderState, fs_khz: i32, fs_api_hz: i32) -> i32 {
    if fs_khz != 8 && fs_khz != 12 && fs_khz != 16 {
        return -1;
    }
    let mut ret = 0;

    // The caller sets nb_subfr for the packet's frame size first, as in
    // libopus: 10 ms frames read their pitch contour with the 10 ms table.
    // Anything but 2 (an unset state) means 4.
    if dec.nb_subfr != 2 {
        dec.nb_subfr = MAX_NB_SUBFR as i32;
    }
    dec.subfr_length = (SUB_FRAME_LENGTH_MS as i32) * fs_khz;
    let frame_length = dec.subfr_length * dec.nb_subfr;

    // Initialize the resampler when the internal or API rate changes.
    if dec.fs_khz != fs_khz || dec.fs_api_hz != fs_api_hz {
        ret += dec.resampler_state.init(fs_khz * 1000, fs_api_hz);
        dec.fs_api_hz = fs_api_hz;
    }

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
            dec.ltp_mem_length = (LTP_MEM_LENGTH_MS as i32) * fs_khz;
            let (lpc_order, nlsf_cb) = if fs_khz == 8 || fs_khz == 12 {
                (MIN_LPC_ORDER, &SILK_NLSF_CB_NB_MB)
            } else {
                (MAX_LPC_ORDER, &SILK_NLSF_CB_WB)
            };
            dec.lpc_order = lpc_order as i32;
            dec.ps_nlsf_cb = Some(nlsf_cb);
            dec.pitch_lag_low_bits_icdf = match fs_khz {
                8 => &crate::silk::tables::SILK_UNIFORM4_ICDF,
                12 => &crate::silk::tables::SILK_UNIFORM6_ICDF,
                _ => &crate::silk::tables::SILK_UNIFORM8_ICDF,
            };
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

    ret
}

/// libopus `silk_reset_decoder`: clears the whole state (rates, frame
/// geometry and resampler included), then resets CNG and PLC.
pub fn silk_reset_decoder(ps_dec: &mut SilkDecoderState) -> i32 {
    *ps_dec = SilkDecoderState::DEFAULT;
    ps_dec.lag_prev = 0;
    ps_dec.last_gain_index = 0;

    // Used to deactivate LSF interpolation.
    ps_dec.first_frame_after_reset = 1;
    ps_dec.prev_gain_q16 = 65536;

    silk_cng_reset(ps_dec);
    silk_plc_reset(ps_dec);

    0
}

/// libopus `silk_init_decoder`.
pub fn silk_init_decoder(ps_dec: &mut SilkDecoderState) -> i32 {
    silk_reset_decoder(ps_dec)
}

pub fn silk_create_decoder() -> SilkDecoderState {
    let mut dec = SilkDecoderState::default();
    silk_init_decoder(&mut dec);
    dec
}
