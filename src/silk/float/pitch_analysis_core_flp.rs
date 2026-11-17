//! libopus `silk/float/pitch_analysis_core_FLP.c`: the three-stage pitch
//! estimator of the float encoder (4 kHz, 8 kHz and the input rate).

use crate::silk::define::*;
use crate::silk::float::sigproc_flp::*;
use crate::silk::resampler::{silk_resampler_down2, silk_resampler_down2_3};
use crate::silk::tables::*;

const SCRATCH_SIZE: usize = 22;
const PE_SHORTLAG_BIAS: f32 = 0.2;
const PE_PREVLAG_BIAS: f32 = 0.2;
const PE_FLATCONTOUR_BIAS: f32 = 0.05;
const C_STRIDE: usize = (PE_MAX_LAG >> 1) + 5;
/// `silk_nb_cbk_searchs_stage3`.
const NB_CBK_SEARCHS_STAGE3: [usize; SILK_PE_MAX_COMPLEX + 1] =
    [PE_NB_CBKS_STAGE3_MIN, PE_NB_CBKS_STAGE3_MID, PE_NB_CBKS_STAGE3_MAX];
/// `silk_Lag_range_stage3_10_ms`.
const LAG_RANGE_STAGE3_10_MS: [[i8; 2]; PE_MAX_NB_SUBFR >> 1] = [[-3, 7], [-2, 7]];

type Stage3 = [[[f32; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]; PE_MAX_NB_SUBFR];

/// The lag codebook entry of subframe `k`, codebook vector `j`.
fn cb_lag(nb_subfr: usize, stage3: bool, k: usize, j: usize) -> i32 {
    i32::from(match (nb_subfr == PE_MAX_NB_SUBFR, stage3) {
        (true, false) => SILK_CB_LAGS_STAGE2[k][j],
        (false, false) => SILK_CB_LAGS_STAGE2_10_MS[k][j],
        (true, true) => SILK_CB_LAGS_STAGE3[k][j],
        (false, true) => SILK_CB_LAGS_STAGE3_10_MS[k][j],
    })
}

/// The stage 3 lag range of subframe `k`.
fn lag_range(nb_subfr: usize, complexity: usize, k: usize) -> [i8; 2] {
    if nb_subfr == PE_MAX_NB_SUBFR {
        SILK_LAG_RANGE_STAGE3[complexity][k]
    } else {
        LAG_RANGE_STAGE3_10_MS[k]
    }
}

fn nb_cbk_stage3(nb_subfr: usize, complexity: usize) -> usize {
    if nb_subfr == PE_MAX_NB_SUBFR {
        NB_CBK_SEARCHS_STAGE3[complexity]
    } else {
        PE_NB_CBKS_STAGE3_10MS
    }
}

/// The pitch estimate of [`silk_pitch_analysis_core_flp`].
pub struct PitchEstimate {
    pub lag_index: i16,
    pub contour_index: i8,
}

/// libopus `silk_pitch_analysis_core_FLP`: `Some` pitch lags in
/// `pitch_out` with their indices for a voiced `frame` (of
/// `(PE_LTP_MEM_LENGTH_MS + nb_subfr * 5) * fs_khz` samples), `None` for an
/// unvoiced one. `ltp_corr` holds the previous frame's normalized
/// correlation and receives this one's.
#[allow(clippy::too_many_arguments)]
pub fn silk_pitch_analysis_core_flp(
    frame: &[f32],
    pitch_out: &mut [i32; MAX_NB_SUBFR],
    ltp_corr: &mut f32,
    prev_lag: i32,
    search_thres1: f32,
    search_thres2: f32,
    fs_khz: usize,
    complexity: usize,
    nb_subfr: usize,
) -> Option<PitchEstimate> {
    let frame_length = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_khz;
    let frame_length_4khz = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 4;
    let frame_length_8khz = (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 8;
    let sf_length = PE_SUBFR_LENGTH_MS * fs_khz;
    let sf_length_4khz = PE_SUBFR_LENGTH_MS * 4;
    let sf_length_8khz = PE_SUBFR_LENGTH_MS * 8;
    let min_lag = PE_MIN_LAG_MS * fs_khz;
    let min_lag_4khz = PE_MIN_LAG_MS * 4;
    let min_lag_8khz = PE_MIN_LAG_MS * 8;
    let max_lag = PE_MAX_LAG_MS * fs_khz - 1;
    let max_lag_4khz = PE_MAX_LAG_MS * 4;
    let max_lag_8khz = PE_MAX_LAG_MS * 8 - 1;

    // Resample to 8 kHz.
    let mut frame_8_fix = [0i16; PE_MAX_FRAME_LENGTH_MS * 8];
    let mut frame_8khz = [0.0f32; PE_MAX_FRAME_LENGTH_MS * 8];
    let mut filt_state = [0i32; 6];
    let mut frame_fix = [0i16; PE_MAX_FRAME_LENGTH];
    if fs_khz == 16 || fs_khz == 12 {
        silk_float2short_array(&mut frame_fix[..frame_length], &frame[..frame_length]);
        if fs_khz == 16 {
            silk_resampler_down2(
                &mut filt_state[..2],
                &mut frame_8_fix,
                &frame_fix[..frame_length],
                frame_length as i32,
            );
        } else {
            silk_resampler_down2_3(
                &mut filt_state,
                &mut frame_8_fix,
                &frame_fix[..frame_length],
                frame_length as i32,
            );
        }
        silk_short2float_array(&mut frame_8khz[..frame_length_8khz], &frame_8_fix[..frame_length_8khz]);
    } else {
        silk_float2short_array(&mut frame_8_fix[..frame_length_8khz], &frame[..frame_length_8khz]);
    }

    // Decimate again to 4 kHz, then low-pass.
    let mut frame_4_fix = [0i16; PE_MAX_FRAME_LENGTH_MS * 4];
    let mut filt_state = [0i32; 2];
    silk_resampler_down2(
        &mut filt_state,
        &mut frame_4_fix,
        &frame_8_fix[..frame_length_8khz],
        frame_length_8khz as i32,
    );
    let mut frame_4khz = [0.0f32; PE_MAX_FRAME_LENGTH_MS * 4];
    silk_short2float_array(&mut frame_4khz[..frame_length_4khz], &frame_4_fix[..frame_length_4khz]);
    for i in (1..frame_length_4khz).rev() {
        // silk_ADD_SAT16 of the two values.
        let s = frame_4khz[i] as i32 + frame_4khz[i - 1] as i32;
        frame_4khz[i] = s.clamp(i16::MIN as i32, i16::MAX as i32) as f32;
    }

    // First stage, at 4 kHz.
    let mut c = [[0.0f32; C_STRIDE]; PE_MAX_NB_SUBFR];
    let mut xcorr = [0.0f32; PE_MAX_LAG_MS * 4 - PE_MIN_LAG_MS * 4 + 1];
    let mut target = sf_length_4khz << 2;
    for _ in 0..nb_subfr >> 1 {
        let target_ptr = &frame_4khz[target..target + sf_length_8khz];
        crate::pitch::pitch_xcorr(
            target_ptr,
            &frame_4khz[target - max_lag_4khz..],
            &mut xcorr,
            sf_length_8khz,
            max_lag_4khz - min_lag_4khz + 1,
            crate::Arch::Scalar,
        );
        let mut basis = target - min_lag_4khz;
        let cross_corr = f64::from(xcorr[max_lag_4khz - min_lag_4khz]);
        let mut normalizer = silk_energy_flp(target_ptr)
            + silk_energy_flp(&frame_4khz[basis..basis + sf_length_8khz])
            + f64::from(sf_length_8khz as f32 * 4000.0);
        c[0][min_lag_4khz] += (2.0 * cross_corr / normalizer) as f32;
        // The normalizer recursively from here on.
        for d in min_lag_4khz + 1..=max_lag_4khz {
            basis -= 1;
            let cross_corr = f64::from(xcorr[max_lag_4khz - d]);
            let b0 = f64::from(frame_4khz[basis]);
            let b1 = f64::from(frame_4khz[basis + sf_length_8khz]);
            normalizer += b0 * b0 - b1 * b1;
            c[0][d] += (2.0 * cross_corr / normalizer) as f32;
        }
        target += sf_length_8khz;
    }

    // Short-lag bias.
    for i in (min_lag_4khz..=max_lag_4khz).rev() {
        c[0][i] -= c[0][i] * i as f32 / 4096.0;
    }

    let mut length_d_srch = 4 + 2 * complexity;
    let mut d_srch = [0usize; PE_D_SRCH_LENGTH];
    silk_insertion_sort_decreasing_flp(
        &mut c[0][min_lag_4khz..=max_lag_4khz],
        &mut d_srch[..length_d_srch],
    );

    // Escape if the correlation is very low already here.
    let cmax = c[0][min_lag_4khz];
    if cmax < 0.2 {
        pitch_out[..nb_subfr].fill(0);
        *ltp_corr = 0.0;
        return None;
    }

    let threshold = search_thres1 * cmax;
    for i in 0..length_d_srch {
        // 8 kHz indices of the sorted correlations above the threshold.
        if c[0][min_lag_4khz + i] > threshold {
            d_srch[i] = (d_srch[i] + min_lag_4khz) << 1;
        } else {
            length_d_srch = i;
            break;
        }
    }

    let mut d_comp = [0i16; C_STRIDE];
    for &d in &d_srch[..length_d_srch] {
        d_comp[d] = 1;
    }
    // Convolution.
    for i in (min_lag_8khz..=max_lag_8khz + 3).rev() {
        d_comp[i] += d_comp[i - 1] + d_comp[i - 2];
    }
    length_d_srch = 0;
    for i in min_lag_8khz..max_lag_8khz + 1 {
        if d_comp[i + 1] > 0 {
            d_srch[length_d_srch] = i;
            length_d_srch += 1;
        }
    }
    // Convolution.
    for i in (min_lag_8khz..=max_lag_8khz + 3).rev() {
        d_comp[i] += d_comp[i - 1] + d_comp[i - 2] + d_comp[i - 3];
    }
    let mut length_d_comp = 0;
    for i in min_lag_8khz..max_lag_8khz + 4 {
        if d_comp[i] > 0 {
            d_comp[length_d_comp] = (i - 2) as i16;
            length_d_comp += 1;
        }
    }

    // Second stage, at 8 kHz, on the lags with high correlation: the
    // energy of each subframe projected onto its history.
    let mut c = [[0.0f32; C_STRIDE]; PE_MAX_NB_SUBFR];
    let frame_8: &[f32] = if fs_khz == 8 { frame } else { &frame_8khz };
    let mut target = PE_LTP_MEM_LENGTH_MS * 8;
    for ck in c.iter_mut().take(nb_subfr) {
        let target_ptr = &frame_8[target..target + sf_length_8khz];
        let energy_tmp = silk_energy_flp(target_ptr) + 1.0;
        for &d in &d_comp[..length_d_comp] {
            let d = d as usize;
            let basis_ptr = &frame_8[target - d..target - d + sf_length_8khz];
            let cross_corr = silk_inner_product_flp(basis_ptr, target_ptr);
            ck[d] = if cross_corr > 0.0 {
                let energy = silk_energy_flp(basis_ptr);
                (2.0 * cross_corr / (energy + energy_tmp)) as f32
            } else {
                0.0
            };
        }
        target += sf_length_8khz;
    }

    // Search over the lags and the lag codebook.
    let mut cc_max = 0.0f32;
    let mut cc_max_b = -1000.0f32;
    let mut cb_imax = 0;
    let mut lag = -1i32;
    let mut prev_lag = prev_lag;
    let prev_lag_log2 = if prev_lag > 0 {
        if fs_khz == 12 {
            prev_lag = (prev_lag << 1) / 3;
        } else if fs_khz == 16 {
            prev_lag >>= 1;
        }
        silk_log2(f64::from(prev_lag as f32))
    } else {
        0.0
    };

    let nb_cbk_search = if nb_subfr == PE_MAX_NB_SUBFR {
        if fs_khz == 8 && complexity > SILK_PE_MIN_COMPLEX {
            // 8 kHz input: a larger codebook, as this is the last stage.
            PE_NB_CBKS_STAGE2_EXT
        } else {
            PE_NB_CBKS_STAGE2
        }
    } else {
        PE_NB_CBKS_STAGE2_10MS
    };

    let mut cc = [0.0f32; PE_NB_CBKS_STAGE2_EXT];
    for &d in &d_srch[..length_d_srch] {
        for (j, ccj) in cc.iter_mut().enumerate().take(nb_cbk_search) {
            *ccj = 0.0;
            for (i, ci) in c.iter().enumerate().take(nb_subfr) {
                *ccj += ci[(d as i32 + cb_lag(nb_subfr, false, i, j)) as usize];
            }
        }
        // The best codebook vector.
        let mut cc_max_new = -1000.0f32;
        let mut cb_imax_new = 0;
        for (i, &v) in cc.iter().enumerate().take(nb_cbk_search) {
            if v > cc_max_new {
                cc_max_new = v;
                cb_imax_new = i;
            }
        }

        // Bias towards shorter lags and the previous lag.
        let lag_log2 = silk_log2(f64::from(d as f32));
        let mut cc_max_new_b = cc_max_new - PE_SHORTLAG_BIAS * nb_subfr as f32 * lag_log2;
        if prev_lag > 0 {
            let mut delta_lag_log2_sqr = lag_log2 - prev_lag_log2;
            delta_lag_log2_sqr *= delta_lag_log2_sqr;
            cc_max_new_b -= PE_PREVLAG_BIAS * nb_subfr as f32 * *ltp_corr * delta_lag_log2_sqr
                / (delta_lag_log2_sqr + 0.5);
        }

        if cc_max_new_b > cc_max_b && cc_max_new > nb_subfr as f32 * search_thres2 {
            cc_max_b = cc_max_new_b;
            cc_max = cc_max_new;
            lag = d as i32;
            cb_imax = cb_imax_new;
        }
    }

    if lag == -1 {
        // No candidate.
        *pitch_out = [0; MAX_NB_SUBFR];
        *ltp_corr = 0.0;
        return None;
    }

    *ltp_corr = cc_max / nb_subfr as f32;

    if fs_khz > 8 {
        // Search the input signal around the lag, compensating for the
        // decimation.
        let min_lag = min_lag as i32;
        let max_lag = max_lag as i32;
        lag = if fs_khz == 12 {
            ((lag * 3) >> 1) + ((lag * 3) & 1)
        } else {
            lag << 1
        };
        lag = lag.clamp(min_lag, max_lag);
        let start_lag = (lag - 2).max(min_lag);
        let end_lag = (lag + 2).min(max_lag);
        let mut lag_new = lag;
        let mut cb_imax = 0;
        let mut cc_max = -1000.0f32;

        let mut cross_corr_st3: Stage3 = [[[0.0; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]; PE_MAX_NB_SUBFR];
        let mut energies_st3: Stage3 = [[[0.0; PE_NB_STAGE3_LAGS]; PE_NB_CBKS_STAGE3_MAX]; PE_MAX_NB_SUBFR];
        calc_corr_st3(&mut cross_corr_st3, frame, start_lag as usize, sf_length, nb_subfr, complexity);
        calc_energy_st3(&mut energies_st3, frame, start_lag as usize, sf_length, nb_subfr, complexity);

        let contour_bias = PE_FLATCONTOUR_BIAS / lag as f32;
        let nb_cbk_search = nb_cbk_stage3(nb_subfr, complexity);

        let target = PE_LTP_MEM_LENGTH_MS * fs_khz;
        let energy_tmp = silk_energy_flp(&frame[target..target + nb_subfr * sf_length]) + 1.0;
        for (lag_counter, d) in (start_lag..=end_lag).enumerate() {
            for j in 0..nb_cbk_search {
                let mut cross_corr = 0.0f64;
                let mut energy = energy_tmp;
                for k in 0..nb_subfr {
                    cross_corr += f64::from(cross_corr_st3[k][j][lag_counter]);
                    energy += f64::from(energies_st3[k][j][lag_counter]);
                }
                let cc_max_new = if cross_corr > 0.0 {
                    // Less for a less flat contour.
                    (2.0 * cross_corr / energy) as f32 * (1.0 - contour_bias * j as f32)
                } else {
                    0.0
                };
                if cc_max_new > cc_max && d + i32::from(SILK_CB_LAGS_STAGE3[0][j]) <= max_lag {
                    cc_max = cc_max_new;
                    lag_new = d;
                    cb_imax = j;
                }
            }
        }

        for (k, p) in pitch_out.iter_mut().enumerate().take(nb_subfr) {
            *p = (lag_new + cb_lag(nb_subfr, true, k, cb_imax))
                .clamp(min_lag, (PE_MAX_LAG_MS * fs_khz) as i32);
        }
        Some(PitchEstimate {
            lag_index: (lag_new - min_lag) as i16,
            contour_index: cb_imax as i8,
        })
    } else {
        for (k, p) in pitch_out.iter_mut().enumerate().take(nb_subfr) {
            *p = (lag + cb_lag(nb_subfr, false, k, cb_imax))
                .clamp(min_lag_8khz as i32, (PE_MAX_LAG_MS * 8) as i32);
        }
        Some(PitchEstimate {
            lag_index: (lag - min_lag_8khz as i32) as i16,
            contour_index: cb_imax as i8,
        })
    }
}

/// libopus `silk_P_Ana_calc_corr_st3`: the stage 3 correlations of each
/// subframe for every codebook vector and the lags from `start_lag`.
fn calc_corr_st3(
    cross_corr_st3: &mut Stage3,
    frame: &[f32],
    start_lag: usize,
    sf_length: usize,
    nb_subfr: usize,
    complexity: usize,
) {
    let nb_cbk_search = nb_cbk_stage3(nb_subfr, complexity);
    let mut target = sf_length << 2;
    let mut xcorr = [0.0f32; SCRATCH_SIZE];
    let mut scratch_mem = [0.0f32; SCRATCH_SIZE];
    for k in 0..nb_subfr {
        let [lag_low, lag_high] = lag_range(nb_subfr, complexity, k).map(i32::from);
        let n = (lag_high - lag_low + 1) as usize;
        let y_start = (target as i32 - start_lag as i32 - lag_high) as usize;
        crate::pitch::pitch_xcorr(
            &frame[target..target + sf_length],
            &frame[y_start..],
            &mut xcorr,
            sf_length,
            n,
            crate::Arch::Scalar,
        );
        for (lag_counter, j) in (lag_low..=lag_high).enumerate() {
            scratch_mem[lag_counter] = xcorr[(lag_high - j) as usize];
        }
        for i in 0..nb_cbk_search {
            let idx = (cb_lag(nb_subfr, true, k, i) - lag_low) as usize;
            cross_corr_st3[k][i].copy_from_slice(&scratch_mem[idx..idx + PE_NB_STAGE3_LAGS]);
        }
        target += sf_length;
    }
}

/// libopus `silk_P_Ana_calc_energy_st3`: the stage 3 energies, recursively
/// over the lags.
fn calc_energy_st3(
    energies_st3: &mut Stage3,
    frame: &[f32],
    start_lag: usize,
    sf_length: usize,
    nb_subfr: usize,
    complexity: usize,
) {
    let nb_cbk_search = nb_cbk_stage3(nb_subfr, complexity);
    let mut target = sf_length << 2;
    let mut scratch_mem = [0.0f32; SCRATCH_SIZE];
    for k in 0..nb_subfr {
        let [lag_low, lag_high] = lag_range(nb_subfr, complexity, k).map(i32::from);
        let basis = (target as i32 - (start_lag as i32 + lag_low)) as usize;
        let mut energy = silk_energy_flp(&frame[basis..basis + sf_length]) + 1e-3;
        scratch_mem[0] = energy as f32;
        let lag_diff = (lag_high - lag_low + 1) as usize;
        for i in 1..lag_diff {
            // Remove the sample leaving the window, add the one entering.
            let out = f64::from(frame[basis + sf_length - i]);
            energy -= out * out;
            let inc = f64::from(frame[basis - i]);
            energy += inc * inc;
            scratch_mem[i] = energy as f32;
        }
        for i in 0..nb_cbk_search {
            let idx = (cb_lag(nb_subfr, true, k, i) - lag_low) as usize;
            energies_st3[k][i].copy_from_slice(&scratch_mem[idx..idx + PE_NB_STAGE3_LAGS]);
        }
        target += sf_length;
    }
}
