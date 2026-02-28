use crate::silk::define::{
    TRANSITION_FRAMES, TRANSITION_INT_NUM, TRANSITION_INT_STEPS, TRANSITION_NA, TRANSITION_NB,
};
use crate::silk::macros::{silk_lshift, silk_rshift, silk_limit, silk_smlawb};
use crate::silk::sigproc_fix::silk_biquad_alt_stride1;
use crate::silk::structs::SilkLPState;
use crate::silk::tables::{SILK_TRANSITION_LP_A_Q28, SILK_TRANSITION_LP_B_Q28};

/* Helper function, interpolates the filter taps */
fn silk_lp_interpolate_filter_taps(
    b_q28: &mut [i32; TRANSITION_NB],
    a_q28: &mut [i32; TRANSITION_NA],
    ind: usize,
    fac_q16: i32,
) {
    if ind < TRANSITION_INT_NUM - 1 {
        if fac_q16 > 0 {
            if fac_q16 < 32768 {
                /* fac_Q16 is in range of a 16-bit int */
                /* Piece-wise linear interpolation of B and A */
                for nb in 0..TRANSITION_NB {
                    b_q28[nb] = silk_smlawb(
                        SILK_TRANSITION_LP_B_Q28[ind][nb],
                        SILK_TRANSITION_LP_B_Q28[ind + 1][nb] - SILK_TRANSITION_LP_B_Q28[ind][nb],
                        fac_q16,
                    );
                }
                for na in 0..TRANSITION_NA {
                    a_q28[na] = silk_smlawb(
                        SILK_TRANSITION_LP_A_Q28[ind][na],
                        SILK_TRANSITION_LP_A_Q28[ind + 1][na] - SILK_TRANSITION_LP_A_Q28[ind][na],
                        fac_q16,
                    );
                }
            } else {
                /* ( fac_Q16 - ( 1 << 16 ) ) is in range of a 16-bit int */
                // assert!(fac_q16 - (1 << 16) == silk_SAT16(fac_q16 - (1 << 16)) as i32);
                /* Piece-wise linear interpolation of B and A */
                for nb in 0..TRANSITION_NB {
                    b_q28[nb] = silk_smlawb(
                        SILK_TRANSITION_LP_B_Q28[ind + 1][nb],
                        SILK_TRANSITION_LP_B_Q28[ind + 1][nb] - SILK_TRANSITION_LP_B_Q28[ind][nb],
                        fac_q16 - (1 << 16),
                    );
                }
                for na in 0..TRANSITION_NA {
                    a_q28[na] = silk_smlawb(
                        SILK_TRANSITION_LP_A_Q28[ind + 1][na],
                        SILK_TRANSITION_LP_A_Q28[ind + 1][na] - SILK_TRANSITION_LP_A_Q28[ind][na],
                        fac_q16 - (1 << 16),
                    );
                }
            }
        } else {
            *b_q28 = SILK_TRANSITION_LP_B_Q28[ind];
            *a_q28 = SILK_TRANSITION_LP_A_Q28[ind];
        }
    } else {
        *b_q28 = SILK_TRANSITION_LP_B_Q28[TRANSITION_INT_NUM - 1];
        *a_q28 = SILK_TRANSITION_LP_A_Q28[TRANSITION_INT_NUM - 1];
    }
}

pub fn silk_lp_variable_cutoff(ps_lp: &mut SilkLPState, frame: &mut [i16], frame_length: usize) {
    let mut b_q28 = [0i32; TRANSITION_NB];
    let mut a_q28 = [0i32; TRANSITION_NA];
    let mut fac_q16: i32;
    let ind: usize;

    // assert!(ps_lp.transition_frame_no >= 0 && ps_lp.transition_frame_no <= TRANSITION_FRAMES);

    /* Run filter if needed */
    if ps_lp.mode != 0 {
        /* Calculate index and interpolation factor for interpolation */
        if TRANSITION_INT_STEPS == 64 {
            fac_q16 = silk_lshift(TRANSITION_FRAMES - ps_lp.transition_frame_no, 16 - 6);
        } else {
            fac_q16 = (silk_lshift(TRANSITION_FRAMES - ps_lp.transition_frame_no, 16))
                / TRANSITION_FRAMES;
        }

        ind = silk_rshift(fac_q16, 16) as usize;
        fac_q16 -= silk_lshift(ind as i32, 16);

        // assert!(ind >= 0);
        // assert!(ind < TRANSITION_INT_NUM);

        /* Interpolate filter coefficients */
        silk_lp_interpolate_filter_taps(&mut b_q28, &mut a_q28, ind, fac_q16);

        /* Update transition frame number for next frame */
        ps_lp.transition_frame_no =
            silk_limit(ps_lp.transition_frame_no + ps_lp.mode, 0, TRANSITION_FRAMES);

        /* ARMA low-pass filtering */
        // assert!(TRANSITION_NB == 3 && TRANSITION_NA == 2);
        silk_biquad_alt_stride1(frame, &b_q28, &a_q28, &mut ps_lp.in_lp_state, frame_length);
    }
}
