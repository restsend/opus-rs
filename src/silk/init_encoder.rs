use crate::silk::define::SILK_NO_ERROR;
use crate::silk::lin2log::silk_lin2log;
use crate::silk::structs::SilkEncoderState;
use crate::silk::tuning_parameters::VARIABLE_HP_MIN_CUTOFF_HZ;
use crate::silk::vad::silk_vad_init;

/// Initialize Silk Encoder state
pub fn silk_init_encoder(ps_enc: &mut SilkEncoderState, _arch: i32) -> i32 {
    let mut ret = SILK_NO_ERROR;

    // Clear the entire encoder state
    // Rust Note: We assume ps_enc is initialized to defaults by caller or reset here manually if needed.
    // Since we don't have a full clear function, we only reset what C init_encoder resets explicitly besides memset.

    // ps_enc.s_cmn.arch = arch; // Not in struct yet

    // C: silk_LSHIFT( silk_lin2log( SILK_FIX_CONST( VARIABLE_HP_MIN_CUTOFF_HZ, 16 ) ) - ( 16 << 7 ), 8 )
    // SILK_FIX_CONST(x, 16) = x << 16, so silk_lin2log(x<<16) - 16*128 = silk_lin2log(x)
    ps_enc.s_cmn.variable_hp_smth1_q15 = silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ) << 8;
    ps_enc.s_cmn.variable_hp_smth2_q15 = ps_enc.s_cmn.variable_hp_smth1_q15;

    // Used to deactivate LSF interpolation, pitch prediction
    ps_enc.s_cmn.first_frame_after_reset = 1;

    // Initialize Silk VAD
    ret += silk_vad_init(&mut ps_enc.s_cmn.s_vad);
    // if ret != SILK_NO_ERROR { return ret; }

    ret
}
