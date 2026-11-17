//! libopus `silk/float/find_pitch_lags_FLP.c`: the whitened residual of the
//! frame and its pitch lags.

use crate::silk::define::*;
use crate::silk::float::pitch_analysis_core_flp::silk_pitch_analysis_core_flp;
use crate::silk::float::sigproc_flp::*;
use crate::silk::float::structs_flp::SilkEncoderControlFlp;
use crate::silk::structs::SilkEncoderState;
use crate::silk::tuning_parameters::*;

/// `FIND_PITCH_LPC_WIN_MAX` (define.h).
const FIND_PITCH_LPC_WIN_MAX: usize = (20 + (LA_PITCH_MS << 1)) * MAX_FS_KHZ;

/// libopus `silk_find_pitch_lags_FLP`: `res` receives the LPC residual of
/// `x_buf` (the input from `ltp_mem_length` samples before the frame, `x -
/// ltp_mem_length` in libopus), and the signal type and pitch lags are set
/// for a frame with voice activity.
pub fn silk_find_pitch_lags_flp(
    ps_enc: &mut SilkEncoderState,
    ctrl: &mut SilkEncoderControlFlp,
    res: &mut [f32],
    x_buf: &[f32],
) {
    let cmn = &mut ps_enc.s_cmn;
    let la_pitch = cmn.la_pitch as usize;
    let buf_len = la_pitch + (cmn.frame_length + cmn.ltp_mem_length) as usize;
    let win_length = cmn.pitch_lpc_win_length as usize;
    let order = ps_enc.pitch_estimation_lpc_order as usize;

    // Windowed signal: sine slopes over the first and the last la_pitch
    // samples.
    let mut wsig = [0.0f32; FIND_PITCH_LPC_WIN_MAX];
    let x_win = &x_buf[buf_len - win_length..buf_len];
    silk_apply_sine_window_flp(&mut wsig, x_win, 1, la_pitch);
    let mid = win_length - (la_pitch << 1);
    wsig[la_pitch..la_pitch + mid].copy_from_slice(&x_win[la_pitch..la_pitch + mid]);
    let tail = la_pitch + mid;
    silk_apply_sine_window_flp(&mut wsig[tail..], &x_win[tail..], 2, la_pitch);

    let mut auto_corr = [0.0f32; MAX_FIND_PITCH_LPC_ORDER + 1];
    silk_autocorrelation_flp(&mut auto_corr[..order + 1], &wsig[..win_length], cmn.arch);
    // White noise as a fraction of the energy.
    auto_corr[0] += auto_corr[0] * FIND_PITCH_WHITE_NOISE_FRACTION + 1.0;

    let mut refl_coef = [0.0f32; MAX_FIND_PITCH_LPC_ORDER];
    let res_nrg = silk_schur_flp(&mut refl_coef, &auto_corr, order);
    ctrl.pred_gain = auto_corr[0] / res_nrg.max(1.0);

    let mut a = [0.0f32; MAX_FIND_PITCH_LPC_ORDER];
    silk_k2a_flp(&mut a[..order], &refl_coef[..order]);
    silk_bwexpander_flp(&mut a[..order], FIND_PITCH_BANDWIDTH_EXPANSION);

    silk_lpc_analysis_filter_flp(res, &a[..order], x_buf, buf_len);

    if i32::from(cmn.indices.signal_type) != TYPE_NO_VOICE_ACTIVITY && cmn.first_frame_after_reset == 0 {
        let mut thrhld = 0.6f32;
        thrhld -= 0.004 * order as f32;
        thrhld -= 0.1 * cmn.speech_activity_q8 as f32 * (1.0 / 256.0);
        thrhld -= 0.15 * (cmn.prev_signal_type >> 1) as f32;
        thrhld -= 0.1 * cmn.input_tilt_q15 as f32 * (1.0 / 32768.0);

        let est = silk_pitch_analysis_core_flp(
            res,
            &mut ctrl.pitch_l,
            &mut ps_enc.ltp_corr,
            cmn.prev_lag,
            cmn.pitch_estimation_threshold_q16 as f32 / 65536.0,
            thrhld,
            cmn.fs_khz as usize,
            cmn.pitch_estimation_complexity as usize,
            cmn.nb_subfr as usize,
            cmn.arch,
        );
        match est {
            Some(e) => {
                cmn.indices.lag_index = e.lag_index;
                cmn.indices.contour_index = e.contour_index;
                cmn.indices.signal_type = TYPE_VOICED as i8;
            }
            None => {
                cmn.indices.lag_index = 0;
                cmn.indices.contour_index = 0;
                cmn.indices.signal_type = TYPE_UNVOICED as i8;
            }
        }
    } else {
        ctrl.pitch_l = [0; MAX_NB_SUBFR];
        cmn.indices.lag_index = 0;
        cmn.indices.contour_index = 0;
        ps_enc.ltp_corr = 0.0;
    }
}
