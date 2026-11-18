use crate::bands::{
    SPREAD_AGGRESSIVE, SPREAD_NONE, SPREAD_NORMAL, compute_band_energies, denormalise_bands, haar1,
    log2amp, normalise_bands, quant_all_bands, spreading_decision,
};
use crate::mathops::{celt_exp2, celt_log2};
use crate::modes::{CeltMode, SPREAD_ICDF, TAPSET_ICDF, TF_SELECT_TABLE, TRIM_ICDF};
use crate::quant_bands::{
    quant_coarse_energy_advanced, quant_energy_finalise, quant_fine_energy, unquant_coarse_energy,
    unquant_energy_finalise, unquant_fine_energy,
};
use crate::range_coder::RangeCoder;
use crate::rate::{BITRES, clt_compute_allocation};
use crate::silk::define::TYPE_VOICED;

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::fixedvec::FixedVec;

mod plc;
#[inline]
fn bitrate_to_bits(bitrate: i32, fs: i32, frame_size: i32) -> i32 {
    bitrate * 6 / (6 * fs / frame_size)
}

pub(crate) const OPUS_BITRATE_MAX: i32 = -1;
#[allow(clippy::too_many_arguments)]
fn compute_vbr(
    mode: &CeltMode,
    analysis: &AnalysisInfo,
    base_target: i32,
    lm: i32,
    bitrate: i32,
    last_coded_bands: i32,
    c: i32,
    intensity: i32,
    constrained_vbr: bool,
    stereo_saving: f32,
    tot_boost: i32,
    tf_estimate: f32,
    pitch_change: i32,
    max_depth: f32,
    lfe: bool,
    has_surround_mask: bool,
    surround_masking: f32,
    temporal_vbr: f32,
) -> i32 {
    let nb_ebands = mode.nb_ebands as i32;
    let mut coded_bands = if last_coded_bands != 0 {
        last_coded_bands
    } else {
        nb_ebands
    };
    coded_bands = coded_bands.clamp(0, nb_ebands);
    let mut coded_bins = mode.e_bands[coded_bands as usize] as i32 * (1 << lm);
    if c == 2 {
        let idx = intensity.min(coded_bands) as usize;
        coded_bins += mode.e_bands[idx] as i32 * (1 << lm);
    }
    let mut target = base_target;
    if analysis.valid && analysis.activity < 0.4 {
        target -= ((coded_bins << BITRES) as f32 * (0.4 - analysis.activity)) as i32;
    }
    if c == 2 {
        let coded_stereo_bands = intensity.min(coded_bands);
        let coded_stereo_dof =
            mode.e_bands[coded_stereo_bands as usize] as i32 * (1 << lm) - coded_stereo_bands;
        let max_frac = (0.8 * coded_stereo_dof as f32) / coded_bins as f32;
        let ss = stereo_saving.min(1.0);
        let save = max_frac * target as f32;
        let adjust = (ss - 0.1) * (coded_stereo_dof << BITRES) as f32;
        target -= (if save < adjust { save } else { adjust }) as i32;
    }
    target += tot_boost - (19 << lm);
    // tf boost: libopus SHL32(MULT16_32_Q15(tf-0.044, target),1) collapses to (tf-0.044)*target in float
    target += ((tf_estimate - 0.044) * target as f32) as i32;
    if analysis.valid && !lfe {
        let tonal = (analysis.tonality - 0.15).max(0.0) - 0.12;
        let mut tonal_target = target + ((coded_bins << BITRES) as f32 * 1.2 * tonal) as i32;
        if pitch_change != 0 {
            tonal_target += ((coded_bins << BITRES) as f32 * 0.8) as i32;
        }
        target = tonal_target;
    }
    if has_surround_mask && !lfe {
        let surround_target =
            target + (surround_masking * coded_bins as f32 * (1 << BITRES) as f32) as i32;
        target = (target / 4).max(surround_target);
    }
    // No more than the signal's depth above the noise floor allows.
    {
        let bins = mode.e_bands[(nb_ebands - 2) as usize] as i32 * (1 << lm);
        let floor_depth = ((c * bins * (1 << BITRES)) as f32 * max_depth) as i32;
        let floor_depth = floor_depth.max(target >> 2);
        target = target.min(floor_depth);
    }
    if (!has_surround_mask || lfe) && constrained_vbr {
        target = base_target + ((target - base_target) as f32 * 0.67) as i32;
    }
    if !has_surround_mask && tf_estimate < 0.2 {
        let amount = 0.0000031 * (32000.min((96000 - bitrate).max(0)) as f32);
        let tvbr_factor = temporal_vbr * amount;
        target += (tvbr_factor * target as f32) as i32;
    }
    target.min(2 * base_target)
}

// --- Heap-free buffer capacity constants (all sized for the worst case:
//     2 channels, 48 kHz, max frame). Runtime construction uses smaller logical
//     lengths when fewer channels / bands are active; the FixedVec tracks that. ---
/// Maximum channel count supported by the API.
const CELT_MAX_CHANNELS: usize = 2;
/// Number of energy bands (`nb_ebands`) for the 48 kHz / 120-overlap mode.
const CELT_NB_EBANDS: usize = 21;
/// `nb_ebands * max_channels` = worst-case per-band-per-channel element count.
const CELT_NB_X_CH: usize = CELT_NB_EBANDS * CELT_MAX_CHANNELS;
/// MDCT overlap for the default mode.
const CELT_OVERLAP: usize = 120;
/// CELT synthesis / decode memory window per channel.
const CELT_CHANNEL_MEM: usize = 2048 + CELT_OVERLAP;
/// `channels * channel_mem` worst case.
const CELT_SYN_MEM: usize = CELT_MAX_CHANNELS * CELT_CHANNEL_MEM;
/// Prefilter comb-filter memory: `channels * COMBFILTER_MAXPERIOD`.
const CELT_PREFILTER_MEM: usize = CELT_MAX_CHANNELS * COMBFILTER_MAXPERIOD;
/// Encoder input buffer stride: `(MAX_FRAME_SIZE + overlap) * channels`.
const CELT_BUFSTRIDE: usize = (MAX_FRAME_SIZE + CELT_OVERLAP) * CELT_MAX_CHANNELS;
/// `MAX_FRAME_SIZE * channels`.
const CELT_FRAME_X_CH: usize = MAX_FRAME_SIZE * CELT_MAX_CHANNELS;
/// Frequency scratch (+4 padding for NEON pre-rotation over-read).
const CELT_W_FREQ: usize = CELT_FRAME_X_CH + 4;
/// PVQ stride-access scratch: `frame_x_ch + STRIDE_ACCESS_PAD`.
const CELT_W_X_ENC: usize = CELT_FRAME_X_CH + STRIDE_ACCESS_PAD;
/// Prefilter pre-scratch: `channels * (COMBFILTER_MAXPERIOD + MAX_FRAME_SIZE)`.
const CELT_PREFILTER_PRE: usize = CELT_MAX_CHANNELS * (COMBFILTER_MAXPERIOD + MAX_FRAME_SIZE);
/// Prefilter pitch buffer: `(COMBFILTER_MAXPERIOD + MAX_FRAME_SIZE) >> 1`.
const CELT_PREFILTER_PITCH: usize = (COMBFILTER_MAXPERIOD + MAX_FRAME_SIZE) >> 1;
/// Decoder `w_x`: `DECODE_BUFFER_SIZE * channels + STRIDE_ACCESS_PAD`.
const CELT_W_X_DEC: usize = DECODE_BUFFER_SIZE * CELT_MAX_CHANNELS + STRIDE_ACCESS_PAD;
/// Decoder `w_freq`: `DECODE_BUFFER_SIZE * channels + 4`.
const CELT_W_FREQ_DEC: usize = DECODE_BUFFER_SIZE * CELT_MAX_CHANNELS + 4;
/// Decoder `decode_mem`: `channels * (DECODE_BUFFER_SIZE + overlap)`.
const CELT_DECODE_MEM: usize = CELT_MAX_CHANNELS * (DECODE_BUFFER_SIZE + CELT_OVERLAP);

/// CELT internal-to-API decimation factor (port of libopus `resampling_factor`).
/// The CELT decoder always runs at 48 kHz internally; the output is decimated
/// by this factor to reach the API sampling rate.
fn resampling_factor(sampling_rate: i32) -> usize {
    match sampling_rate {
        48000 => 1,
        24000 => 2,
        16000 => 3,
        12000 => 4,
        8000 => 6,
        _ => 1,
    }
}

/// libopus `compute_mdcts` with `upsample != 1`: keep only the first
/// `N/upsample` coefficients (the API Nyquist band), scale them by `upsample`
/// to preserve energy after zero-stuffing, and zero the rest.
fn upsample_scale_zero(freq: &mut [f32], frame_size: usize, upsample: usize, channels: usize) {
    if upsample <= 1 {
        return;
    }
    let bound = frame_size / upsample;
    for c in 0..channels {
        let base = c * frame_size;
        for v in freq[base..base + bound].iter_mut() {
            *v *= upsample as f32;
        }
        freq[base + bound..base + frame_size].fill(0.0);
    }
}

#[inline(always)]
fn sum_abs(x: &[f32]) -> f32 {
    x.iter().map(|&v| v.abs()).sum()
}

const MAX_FRAME_SIZE: usize = 2880;

/// Largest CELT-internal synthesis buffer. The CELT decoder writes `n + overlap`
/// MDCT samples per channel where `n ≤ frame_size ≤ 960` (20 ms @ 48 kHz); 2048
/// is the libopus reference value (`DECODE_BUFFER_SIZE`) and gives ample margin.
/// (Reduced from a previous 3072 to shrink the heap-free footprint.)
const DECODE_BUFFER_SIZE: usize = 2048;

/// libopus transient_analysis inv_table (celt_encoder.c:286-294).
const INV_TABLE: [u8; 128] = [
    255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18, 17,
    16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8, 8,
    8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
];

const MAX_TRANSIENT_LEN: usize = 3000;

#[derive(Debug, Clone, Copy)]
pub struct AnalysisInfo {
    pub valid: bool,
    pub tonality: f32,
    pub tonality_slope: f32,
    pub noisiness: f32,
    pub activity: f32,
    pub music_prob: f32,
    pub music_prob_min: f32,
    pub music_prob_max: f32,
    pub bandwidth: i32,
    pub activity_probability: f32,
    pub max_pitch_ratio: f32,
    pub leak_boost: [u8; 19], // LEAK_BANDS = 19
}

impl AnalysisInfo {
    /// The all-zero analysis of a new or reset libopus analysis state.
    pub(crate) const fn zeroed() -> Self {
        Self {
            valid: false,
            tonality: 0.0,
            tonality_slope: 0.0,
            noisiness: 0.0,
            activity: 0.0,
            music_prob: 0.0,
            music_prob_min: 0.0,
            music_prob_max: 0.0,
            bandwidth: 0,
            activity_probability: 0.0,
            max_pitch_ratio: 0.0,
            leak_boost: [0; 19],
        }
    }
}

impl Default for AnalysisInfo {
    fn default() -> Self {
        Self {
            valid: false,
            tonality: 0.0,
            tonality_slope: 0.0,
            noisiness: 0.0,
            activity: 0.0,
            music_prob: 0.0,
            music_prob_min: 0.0,
            music_prob_max: 0.0,
            bandwidth: 0,
            activity_probability: 0.0,
            max_pitch_ratio: 1.0,
            leak_boost: [0; 19],
        }
    }
}

/// libopus compute_mdcts (celt_encoder.c:511-555): each input channel's
/// long MDCT or `1 << lm` short ones, interleaved, their average for a mono
/// stream of stereo input, scaled for the upsampling.
#[allow(clippy::too_many_arguments)]
fn compute_mdcts(
    mode: &CeltMode,
    short_blocks: bool,
    in_buf: &[f32],
    freq: &mut [f32],
    channels: usize,
    cc: usize,
    frame_size: usize,
    overlap: usize,
    lm: usize,
    upsample: usize,
) {
    let buf_stride = frame_size + overlap;
    let (b, shift) = if short_blocks {
        (1 << lm, mode.max_lm)
    } else {
        (1, mode.max_lm - lm)
    };
    let n = frame_size / b;
    for c in 0..cc {
        for i in 0..b {
            mode.mdct.forward(
                &in_buf[c * buf_stride + i * n..c * buf_stride + buf_stride],
                &mut freq[c * frame_size + i..],
                mode.window,
                overlap,
                shift,
                b,
            );
        }
    }
    if cc == 2 && channels == 1 {
        for i in 0..frame_size {
            freq[i] = 0.5 * freq[i] + 0.5 * freq[frame_size + i];
        }
    }
    upsample_scale_zero(freq, frame_size, upsample, channels);
}

/// libopus patch_transient_decision (celt_encoder.c:473-508): whether the
/// band energies rose by more than 1 on average above the previous frame's,
/// spread by -1 per band.
fn patch_transient_decision(
    new_e: &[f32],
    old_e: &[f32],
    nb_ebands: usize,
    start: usize,
    end: usize,
    channels: usize,
) -> bool {
    let mut spread_old = [0.0f32; 26];
    let old = |i: usize| {
        if channels == 1 {
            old_e[i]
        } else {
            old_e[i].max(old_e[i + nb_ebands])
        }
    };
    spread_old[start] = old(start);
    for i in start + 1..end {
        spread_old[i] = (spread_old[i - 1] - 1.0).max(old(i));
    }
    for i in (start..end - 1).rev() {
        spread_old[i] = spread_old[i].max(spread_old[i + 1] - 1.0);
    }
    let first = start.max(2);
    let mut mean_diff = 0.0f32;
    for c in 0..channels {
        for i in first..end - 1 {
            let x1 = new_e[i + c * nb_ebands].max(0.0);
            let x2 = spread_old[i].max(0.0);
            mean_diff += (x1 - x2).max(0.0);
        }
    }
    mean_diff /= (channels as i32 * (end as i32 - 1 - first as i32)) as f32;
    mean_diff > 1.0
}

/// transient_analysis's high-pass filter, forward (post-echo) and backward
/// (pre-echo) passes for `C` channels side by side. Each channel's chains
/// keep libopus's order; the filter and the forward pass run in one loop and
/// the channels in step, so the serial chains overlap. `tmp[c * len / 2..]`
/// gets channel c's threshold; returns each channel's energy sum and peak.
#[inline(always)]
fn transient_thresholds<const C: usize>(
    input: &[f32],
    len: usize,
    forward_decay: f32,
    tmp: &mut [f32],
) -> [(f32, f32); C] {
    let len2 = len / 2;
    let mut hp = [[0.0f32; 2]; C];
    let mut fwd = [0.0f32; C];
    let mut out = [(0.0f32, 0.0f32); C];
    for i in 0..len2 {
        for c in 0..C {
            let mut y = [0.0f32; 2];
            for (k, y) in y.iter_mut().enumerate() {
                // High-pass filter: (1 - 2*z^-1 + z^-2) / (1 - z^-1 + .5*z^-2).
                let x = input[c * len + 2 * i + k];
                let [mem0, mem1] = hp[c];
                *y = mem0 + x;
                hp[c] = [mem0 - x + 0.5 * mem1, x - mem0];
            }
            // The first 12 outputs are cleared: the filter has no memory.
            if i < 6 {
                y = [0.0; 2];
            }
            // C float build: PSHR32 is an identity, so x2 is the plain sum of
            // squares and mean += x2.
            let x2 = y[0] * y[0] + y[1] * y[1];
            out[c].0 += x2;
            fwd[c] = x2 + (1.0 - forward_decay) * fwd[c];
            tmp[c * len2 + i] = forward_decay * fwd[c];
        }
    }
    let mut bwd = [0.0f32; C];
    for i in (0..len2).rev() {
        for c in 0..C {
            let t = &mut tmp[c * len2 + i];
            bwd[c] = *t + 0.875 * bwd[c];
            *t = 0.125 * bwd[c];
            if *t > out[c].1 {
                out[c].1 = *t;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn transient_analysis(
    input: &[f32],
    len: usize,
    channels: usize,
    tf_estimate: &mut f32,
    tf_chan: &mut usize,
    allow_weak_transients: bool,
    weak_transient: &mut bool,
    _tone_freq: f32,
    toneishness: f32,
    tmp: &mut [f32],
) -> bool {
    let mut mask_metric = 0.0f32;
    let mut forward_decay = 0.0625f32;

    *weak_transient = false;
    if allow_weak_transients {
        forward_decay = 0.03125f32;
    }

    let len2 = len / 2;
    debug_assert!(len <= MAX_TRANSIENT_LEN);
    let tmp = &mut tmp[..channels * len2];
    let mut energy = [(0.0f32, 0.0f32); 2];
    if channels == 2 {
        energy = transient_thresholds::<2>(input, len, forward_decay, tmp);
    } else {
        [energy[0]] = transient_thresholds::<1>(input, len, forward_decay, tmp);
    }

    for (c, (threshold, &(mean, max_e))) in tmp.chunks_exact(len2).zip(&energy).enumerate() {
        // C promotes to double here (`.5` is a double literal) and takes a
        // double sqrt: `celt_sqrt(mean * maxE*.5*len2)`.
        let mean = ((mean * max_e) as f64 * 0.5 * len2 as f64).sqrt() as f32;
        // C (float build): norm = len2 / (EPSILON + mean)  [EPSILON = 1e-15f]
        let norm = (len2 as f32) / (1e-15 + mean);

        // C computes the harmonic mean with INTEGER arithmetic (mask_metric and
        // unmask are opus_int32; the 64/6 scaling is an integer division).
        let mut unmask: i32 = 0;
        for i in (12..(len2 - 5)).step_by(4) {
            // The cast truncates toward zero; it equals floor() after the
            // clamp to 0..=127.
            let id = (64.0 * norm * (threshold[i] + 1e-15)) as i32;
            let id = id.clamp(0, 127) as usize;
            unmask += INV_TABLE[id] as i32;
        }

        unmask = 64 * unmask * 4 / (6 * (len2 as i32 - 17));
        if (unmask as f32) > mask_metric {
            *tf_chan = c;
            mask_metric = unmask as f32;
        }
    }

    let mut is_transient = mask_metric > 200.0;

    if toneishness > 0.98 && _tone_freq < 0.026 {
        is_transient = false;
        mask_metric = 0.0;
    }
    // Weak transients at low hybrid rates (celt_encoder.c:453-456).
    if allow_weak_transients && is_transient && mask_metric < 600.0 {
        is_transient = false;
        *weak_transient = true;
    }

    // C (celt_encoder.c:460-461): tf_max = max(0, sqrt(27*mask_metric) - 42);
    // tf_estimate = sqrt(max(0, 0.0069*min(163, tf_max) - 0.139))  [float build]
    let tf_max = ((27.0 * mask_metric).sqrt() - 42.0).max(0.0);
    // The product is float, the 0.139, the max and the sqrt are double.
    *tf_estimate = (f64::from(0.0069 * tf_max.min(163.0)) - 0.139)
        .max(0.0)
        .sqrt() as f32;

    is_transient
}

const MAX_NB_EBANDS: usize = 21;

const MAX_TF_TMP: usize = 176;
/// Most haar1 steps tf_analysis tries (lm + 1 with lm <= 3).
const MAX_TF_LEVELS: usize = 4;

#[allow(clippy::too_many_arguments)]
fn tf_analysis(
    mode: &CeltMode,
    len: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lambda: i32,
    x: &[f32],
    n0: usize,
    lm: i32,
    tf_estimate: f32,
    tf_chan: usize,
    importance: &[i32],
) -> i32 {
    debug_assert!(len <= MAX_NB_EBANDS);
    let mut metric = [0i32; MAX_NB_EBANDS];
    let bias = 0.04 * (-0.25f32).max(0.5 - tf_estimate);
    // The haar1 levels of one band (see below).
    let mut levels = [[0.0f32; MAX_TF_TMP]; 2 + MAX_TF_LEVELS];

    for (i, metric_i) in metric[..len].iter_mut().enumerate() {
        let n = ((mode.e_bands[i + 1] - mode.e_bands[i]) as usize) << lm;
        let narrow = (mode.e_bands[i + 1] - mode.e_bands[i]) == 1;
        let offset = tf_chan * n0 + ((mode.e_bands[i] as usize) << lm);
        // The levels depend on each other only through haar1, so all of them
        // are transformed first and their (sequential) abs sums run side by
        // side; the decisions then follow in libopus's order.
        // levels[0]: the band; [1]: haar1 at the shortest blocks (transient,
        // wide bands); [2 + k]: after the k+1 progressive haar1 steps.
        levels[0][..n].copy_from_slice(&x[offset..offset + n]);
        let try_short = is_transient && !narrow;
        if try_short {
            levels[1][..n].copy_from_slice(&x[offset..offset + n]);
            haar1(&mut levels[1][..n], n >> lm, 1 << lm);
        }
        let steps = (lm + if is_transient || narrow { 0 } else { 1 }) as usize;
        for k in 0..steps {
            let (done, next) = levels.split_at_mut(2 + k);
            let prev = if k == 0 { &done[0] } else { &done[1 + k] };
            next[0][..n].copy_from_slice(&prev[..n]);
            haar1(&mut next[0][..n], n >> k, 1 << k);
        }
        let mut sums = [0.0f32; 2 + MAX_TF_LEVELS];
        for j in 0..n {
            for (sum, level) in sums.iter_mut().zip(&levels) {
                *sum += level[j].abs();
            }
        }
        let l1 = |sum: f32, lm: i32| sum + (lm as f32) * bias * sum;

        let mut best_l1 = l1(sums[0], if is_transient { lm } else { 0 });
        let mut best_level = 0;

        if try_short {
            let l1 = l1(sums[1], lm + 1);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = -1;
            }
        }

        for k in 0..steps as i32 {
            let b = if is_transient { lm - k - 1 } else { k + 1 };
            let l1 = l1(sums[2 + k as usize], b);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = k + 1;
            }
        }

        if is_transient {
            *metric_i = 2 * best_level;
        } else {
            *metric_i = -2 * best_level;
        }

        if narrow && (*metric_i == 0 || *metric_i == -2 * lm) {
            *metric_i -= 1;
        }
    }

    let mut tf_select = 0;
    let mut selcost = [0i32; 2];

    for sel in 0..2 {
        let mut cost0 = importance[0]
            * (metric[0]
                - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel] as i32)
                .abs();
        let mut cost1 = importance[0]
            * (metric[0]
                - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel + 1]
                    as i32)
                .abs()
            + if is_transient { 0 } else { lambda };

        for i in 1..len {
            let curr0 = cost0.min(cost1 + lambda);
            let curr1 = (cost0 + lambda).min(cost1);
            cost0 = curr0
                + importance[i]
                    * (metric[i]
                        - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel]
                            as i32)
                        .abs();
            cost1 = curr1
                + importance[i]
                    * (metric[i]
                        - 2 * TF_SELECT_TABLE[lm as usize]
                            [4 * (is_transient as usize) + 2 * sel + 1]
                            as i32)
                        .abs();
        }
        selcost[sel] = cost0.min(cost1);
    }

    // C: only allow tf_select=1 for transients (celt_encoder.c:787)
    if selcost[1] < selcost[0] && is_transient {
        tf_select = 1;
    }

    // Viterbi forward pass with path storage, then backward pass to
    // reconstruct the optimal tf_res (C celt_encoder.c:789-822).
    let sel = tf_select as usize;
    let mut cost0 = importance[0]
        * (metric[0]
            - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel] as i32)
            .abs();
    let mut cost1 = importance[0]
        * (metric[0]
            - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel + 1] as i32)
            .abs()
        + if is_transient { 0 } else { lambda };

    let mut path0 = [0i32; MAX_NB_EBANDS];
    let mut path1 = [0i32; MAX_NB_EBANDS];
    for i in 1..len {
        let from0 = cost0;
        let from1 = cost1 + lambda;
        let curr0 = if from0 < from1 {
            path0[i] = 0;
            from0
        } else {
            path0[i] = 1;
            from1
        };

        let from0 = cost0 + lambda;
        let from1 = cost1;
        let curr1 = if from0 < from1 {
            path1[i] = 0;
            from0
        } else {
            path1[i] = 1;
            from1
        };

        cost0 = curr0
            + importance[i]
                * (metric[i]
                    - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel]
                        as i32)
                    .abs();
        cost1 = curr1
            + importance[i]
                * (metric[i]
                    - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel + 1]
                        as i32)
                    .abs();
    }
    tf_res[len - 1] = if cost0 < cost1 { 0 } else { 1 };
    for i in (0..len - 1).rev() {
        tf_res[i] = if tf_res[i + 1] == 1 {
            path1[i + 1]
        } else {
            path0[i + 1]
        };
    }

    tf_select
}

fn tf_encode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    mut tf_select: i32,
    rc: &mut RangeCoder,
) -> i32 {
    let mut curr = 0;
    let mut tf_changed = 0;
    let mut logp = if is_transient { 2 } else { 4 };
    let mut budget = rc.storage as i32 * 8;
    let mut tell = rc.tell();

    let tf_select_rsv = if lm > 0 && tell + logp < budget { 1 } else { 0 };
    budget -= tf_select_rsv;

    for tf_res_i in tf_res[start..end].iter_mut() {
        if tell + logp <= budget {
            rc.encode_bit_logp(*tf_res_i ^ curr != 0, logp as u32);
            tell = rc.tell();
            curr = *tf_res_i;
            tf_changed |= curr;
        } else {
            *tf_res_i = curr;
        }
        logp = if is_transient { 4 } else { 5 };
    }

    if tf_select_rsv != 0
        && TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + (tf_changed as usize)]
            != TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 + (tf_changed as usize)]
    {
        rc.encode_bit_logp(tf_select != 0, 1);
    } else {
        tf_select = 0;
    }

    for tf_res_i in tf_res[start..end].iter_mut() {
        *tf_res_i = TF_SELECT_TABLE[lm as usize]
            [4 * (is_transient as usize) + 2 * (tf_select as usize) + (*tf_res_i as usize)]
            as i32;
    }

    tf_changed
}

fn tf_decode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    rc: &mut RangeCoder,
) {
    let mut curr = 0;
    let mut tf_changed = 0;
    let mut logp = if is_transient { 2 } else { 4 };
    let budget = rc.storage as i32 * 8;
    let mut tell = rc.tell();

    let tf_select_rsv = if lm > 0 && tell + logp < budget { 1 } else { 0 };
    let budget = budget - tf_select_rsv;

    for tf_res_i in tf_res[start..end].iter_mut() {
        if tell + logp <= budget {
            curr ^= if rc.decode_bit_logp(logp as u32) {
                1
            } else {
                0
            };
            tell = rc.tell();
            tf_changed |= curr;
        }
        *tf_res_i = curr;
        logp = if is_transient { 4 } else { 5 };
    }

    let mut tf_select = 0;
    let _budget = budget + tf_select_rsv;
    if tf_select_rsv > 0
        && TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + (tf_changed as usize)]
            != TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 + (tf_changed as usize)]
    {
        tf_select = if rc.decode_bit_logp(1) { 1 } else { 0 };
    }

    for tf_res_i in tf_res[start..end].iter_mut() {
        *tf_res_i = TF_SELECT_TABLE[lm as usize]
            [4 * (is_transient as usize) + 2 * (tf_select as usize) + (*tf_res_i as usize)]
            as i32;
    }
}

fn stereo_analysis(m: &CeltMode, x: &[f32], lm: i32, n0: usize) -> bool {
    let mut sum_lr = 1e-15f32;
    let mut sum_ms = 1e-15f32;

    for i in 0..13 {
        let start = (m.e_bands[i] as usize) << lm;
        let end = (m.e_bands[i + 1] as usize) << lm;
        for j in start..end {
            let l = x[j];
            let r = x[n0 + j];
            let m_val = l + r;
            let s_val = l - r;
            sum_lr += l.abs() + r.abs();
            sum_ms += m_val.abs() + s_val.abs();
        }
    }

    sum_ms *= 0.707107;
    let mut thetas = 13;
    if lm <= 1 {
        thetas -= 8;
    }

    let left = (((m.e_bands[13] as usize) << (lm + 1)) + thetas) as f32 * sum_ms;
    let right = ((m.e_bands[13] as usize) << (lm + 1)) as f32 * sum_lr;

    left > right
}

const COMBFILTER_MINPERIOD: usize = 15;
const COMBFILTER_MAXPERIOD: usize = 1024;

// The digits of celt.c:246-249.
#[allow(clippy::excessive_precision)]
const PREFILTER_GAINS: [[f32; 3]; 3] = [
    [0.306_640_625, 0.217_041_015_6, 0.129_638_671_9],
    [0.463_867_187_5, 0.268_066_406_2, 0.0],
    [0.799_804_687_5, 0.100_097_656_2, 0.0],
];

/// `y[y_idx..][..n]`: the comb filter with constant period and gains.
#[allow(clippy::too_many_arguments)]
fn comb_filter_const(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
    arch: crate::Arch,
) {
    let y = &mut y[y_idx..y_idx + n];
    let xs = &x[x_idx..x_idx + n];
    // x[x_idx + i - t + k - 2] is xk[i].
    let past = &x[x_idx - t - 2..x_idx - t + 2 + n];
    let (x4, x3, x2, x1, x0) = (
        &past[..n],
        &past[1..n + 1],
        &past[2..n + 2],
        &past[3..n + 3],
        &past[4..n + 4],
    );
    let sse = sse_groups(n, arch);
    for i in 0..n {
        y[i] = comb_tap(
            i < sse,
            xs[i],
            [x0[i], x1[i], x2[i], x3[i], x4[i]],
            g10,
            g11,
            g12,
        );
    }
}

/// The samples of a constant comb filter of `n` samples that libopus's
/// `comb_filter_const_sse` computes at the x86 levels: whole groups of
/// four (the rest, which only custom modes have, as the scalar code).
fn sse_groups(n: usize, arch: crate::Arch) -> usize {
    if arch == crate::Arch::Scalar {
        0
    } else {
        n & !3
    }
}

/// One sample of the constant comb filter from `x` and the taps
/// `[x0, x1, x2, x3, x4]` (x[i - T + 2] down to x[i - T - 2]): in
/// `comb_filter_const_c`'s order, or with `sse` in
/// `comb_filter_const_sse`'s partial sums.
#[inline(always)]
fn comb_tap(sse: bool, x: f32, t: [f32; 5], g10: f32, g11: f32, g12: f32) -> f32 {
    let [x0, x1, x2, x3, x4] = t;
    if sse {
        (x + g10 * x2) + (g11 * (x3 + x1) + g12 * (x4 + x0))
    } else {
        x + g10 * x2 + g11 * (x1 + x3) + g12 * (x0 + x4)
    }
}

#[cfg(feature = "fuzzing")]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn fuzz_comb_filter(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: i32,
    tapset1: i32,
    window: &[f32],
    overlap: usize,
) {
    comb_filter(
        y,
        x,
        y_idx,
        x_idx,
        t0,
        t1,
        n,
        g0,
        g1,
        tapset0,
        tapset1,
        window,
        overlap,
        crate::Arch::select(),
    )
}

#[allow(clippy::too_many_arguments)]
fn comb_filter(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: i32,
    tapset1: i32,
    window: &[f32],
    overlap: usize,
    arch: crate::Arch,
) {
    if g0 == 0.0 && g1 == 0.0 {
        if x_idx != y_idx || !core::ptr::eq(x.as_ptr(), y.as_ptr()) {
            y[y_idx..y_idx + n].copy_from_slice(&x[x_idx..x_idx + n]);
        }
        return;
    }

    let t0 = t0.clamp(
        COMBFILTER_MINPERIOD,
        x_idx.saturating_sub(2).max(COMBFILTER_MINPERIOD),
    );
    let t1 = t1.clamp(
        COMBFILTER_MINPERIOD,
        x_idx.saturating_sub(2).max(COMBFILTER_MINPERIOD),
    );

    let g00 = g0 * PREFILTER_GAINS[tapset0 as usize][0];
    let g01 = g0 * PREFILTER_GAINS[tapset0 as usize][1];
    let g02 = g0 * PREFILTER_GAINS[tapset0 as usize][2];

    let g10 = g1 * PREFILTER_GAINS[tapset1 as usize][0];
    let g11 = g1 * PREFILTER_GAINS[tapset1 as usize][1];
    let g12 = g1 * PREFILTER_GAINS[tapset1 as usize][2];

    let mut x1 = x[x_idx - t1 + 1];
    let mut x2 = x[x_idx - t1];
    let mut x3 = x[x_idx - t1 - 1];
    let mut x4 = x[x_idx - t1 - 2];

    let mut inner_overlap = overlap;
    if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        inner_overlap = 0;
    }

    let mut i = 0;
    while i < inner_overlap && i < n {
        let x0 = x[x_idx + i - t1 + 2];
        let f = window[i] * window[i];
        // celt.c:283-289: each gain is scaled by the window before use.
        y[y_idx + i] = x[x_idx + i]
            + ((1.0 - f) * g00) * x[x_idx + i - t0]
            + ((1.0 - f) * g01) * (x[x_idx + i - t0 + 1] + x[x_idx + i - t0 - 1])
            + ((1.0 - f) * g02) * (x[x_idx + i - t0 + 2] + x[x_idx + i - t0 - 2])
            + (f * g10) * x2
            + (f * g11) * (x1 + x3)
            + (f * g12) * (x0 + x4);

        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
        i += 1;
    }

    if i < n {
        if g1 == 0.0 {
            y[y_idx + i..y_idx + n].copy_from_slice(&x[x_idx + i..x_idx + n]);
        } else {
            comb_filter_const(y, x, y_idx + i, x_idx + i, t1, n - i, g10, g11, g12, arch);
        }
    }
}

/// comb_filter (celt.c) in place on `buf[y_idx..][..n]`, as the decoder
/// calls it: the reads at `y_idx + i - T` see the history before `y_idx` and
/// the samples already filtered.
#[allow(clippy::too_many_arguments)]
fn comb_filter_inplace(
    buf: &mut [f32],
    y_idx: usize,
    t0: usize,
    t1: usize,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: i32,
    tapset1: i32,
    window: &[f32],
    overlap: usize,
    arch: crate::Arch,
) {
    if g0 == 0.0 && g1 == 0.0 {
        return;
    }

    let t0 = t0.clamp(COMBFILTER_MINPERIOD, y_idx - 2);
    let t1 = t1.clamp(COMBFILTER_MINPERIOD, y_idx - 2);

    let g00 = g0 * PREFILTER_GAINS[tapset0 as usize][0];
    let g01 = g0 * PREFILTER_GAINS[tapset0 as usize][1];
    let g02 = g0 * PREFILTER_GAINS[tapset0 as usize][2];

    let g10 = g1 * PREFILTER_GAINS[tapset1 as usize][0];
    let g11 = g1 * PREFILTER_GAINS[tapset1 as usize][1];
    let g12 = g1 * PREFILTER_GAINS[tapset1 as usize][2];

    // The filter did not change: no cross-fade.
    let overlap = if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        0
    } else {
        overlap.min(n)
    };
    for i in 0..overlap {
        let idx = y_idx + i;
        let f = window[i] * window[i];
        // celt.c:283-289: each gain is scaled by the window before use.
        buf[idx] = buf[idx]
            + ((1.0 - f) * g00) * buf[idx - t0]
            + ((1.0 - f) * g01) * (buf[idx - t0 + 1] + buf[idx - t0 - 1])
            + ((1.0 - f) * g02) * (buf[idx - t0 + 2] + buf[idx - t0 - 2])
            + (f * g10) * buf[idx - t1]
            + (f * g11) * (buf[idx - t1 + 1] + buf[idx - t1 - 1])
            + (f * g12) * (buf[idx - t1 + 2] + buf[idx - t1 - 2]);
    }
    if g1 == 0.0 {
        return;
    }

    // comb_filter_const_c: x[i - t1 + k - 2] is buf[start + i + k] here.
    let start = y_idx + overlap - t1 - 2;
    let x = &mut buf[start..y_idx + n];
    // The SSE groups of four read taps at least T >= 15 samples back, all
    // written before the group, so a group is four independent samples.
    let sse = sse_groups(n - overlap, arch);
    for i in (0..sse).step_by(4) {
        let tap = |k: usize| -> [f32; 4] { x[i + 4 - k..i + 8 - k].try_into().unwrap() };
        let [x0, x1, x2, x3, x4] = [tap(0), tap(1), tap(2), tap(3), tap(4)];
        let y: &mut [f32; 4] = (&mut x[i + t1 + 2..i + t1 + 6]).try_into().unwrap();
        for l in 0..4 {
            y[l] = comb_tap(
                true,
                y[l],
                [x0[l], x1[l], x2[l], x3[l], x4[l]],
                g10,
                g11,
                g12,
            );
        }
    }
    let (mut x4, mut x3, mut x2, mut x1) = (x[sse], x[sse + 1], x[sse + 2], x[sse + 3]);
    for i in sse..n - overlap {
        let x0 = x[i + 4];
        let y = &mut x[i + t1 + 2];
        *y = comb_tap(false, *y, [x0, x1, x2, x3, x4], g10, g11, g12);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

// 3.1416/3.141593 literals mirror celt_encoder.c; keep them verbatim.
#[allow(clippy::approx_constant)]
fn run_prefilter(
    in_buf: &mut [f32],
    prefilter_mem: &mut [f32],
    prefilter_period: usize,
    prefilter_gain: f32,
    prefilter_tapset: i32,
    tapset_decision: i32,
    window: &[f32],
    channels: usize,
    frame_size: usize,
    overlap: usize,

    pre: &mut [f32],
    pitch_buf: &mut [f32],
    before: &mut [f32],
    after: &mut [f32],

    analysis: &AnalysisInfo,
    loss_rate: i32,
    tf_estimate: f32,
    enabled: bool,
    nb_available_bytes: usize,
    tone_freq: f32,
    toneishness: f32,
    complexity: i32,
    arch: crate::Arch,
) -> (bool, f32, usize) {
    let max_period = COMBFILTER_MAXPERIOD;
    let min_period = COMBFILTER_MINPERIOD;
    let buf_stride = frame_size + overlap;
    let pre_size = max_period + frame_size;

    for c in 0..channels {
        pre[c * pre_size..c * pre_size + max_period]
            .copy_from_slice(&prefilter_mem[c * max_period..(c + 1) * max_period]);
        pre[c * pre_size + max_period..c * pre_size + pre_size].copy_from_slice(
            &in_buf[c * buf_stride + overlap..c * buf_stride + overlap + frame_size],
        );
    }

    let mut pitch_index: usize;
    let mut gain1: f32;
    if enabled && toneishness > 0.99 {
        // Tone path (C 1449-1473): bypass the pitch search for pure tones.
        let mut multiple = 1.0f32;
        let mut tone_freq = tone_freq;
        if tone_freq >= 3.1416 {
            tone_freq = 3.141593 - tone_freq;
        }
        while tone_freq >= multiple * 0.39 {
            multiple += 1.0;
        }
        if tone_freq > 0.006148 {
            // In double: M_PI and .5 are double literals (C 1452).
            let period =
                0.5 + 2.0 * core::f64::consts::PI * f64::from(multiple) / f64::from(tone_freq);
            pitch_index = (period.floor() as usize).min(COMBFILTER_MAXPERIOD - 2);
        } else {
            pitch_index = COMBFILTER_MINPERIOD;
        }
        gain1 = 0.75;
    } else if enabled && complexity >= 5 {
        let pitch_buf_len = (max_period + frame_size) >> 1;
        {
            let mut pre_slices: FixedVec<&[f32], 2> = FixedVec::new();
            for c in 0..channels {
                pre_slices.push(&pre[c * pre_size..c * pre_size + pre_size]);
            }
            crate::pitch::pitch_downsample(
                &pre_slices,
                pitch_buf,
                pitch_buf_len,
                channels,
                2,
                arch,
            );
        }

        let search_max = max_period - 3 * min_period;
        let pitch_result = crate::pitch::pitch_search(
            &pitch_buf[max_period >> 1..],
            pitch_buf,
            frame_size,
            search_max,
            arch,
        );
        pitch_index = (max_period - pitch_result).min(max_period - 2);

        gain1 = crate::pitch::remove_doubling(
            pitch_buf,
            max_period,
            min_period,
            frame_size,
            &mut pitch_index,
            prefilter_period,
            prefilter_gain,
            arch,
        );
        if pitch_index > max_period - 2 {
            pitch_index = max_period - 2;
        }
        gain1 *= 0.7;
        // C: halve at >2% loss, halve again at >4%, zero at >8%
        if loss_rate > 2 {
            gain1 *= 0.5;
        }
        if loss_rate > 4 {
            gain1 *= 0.5;
        }
        if loss_rate > 8 {
            gain1 = 0.0;
        }
    } else {
        gain1 = 0.0;
        pitch_index = COMBFILTER_MINPERIOD;
    }

    if analysis.valid {
        gain1 *= analysis.max_pitch_ratio;
    }

    let mut pf_threshold = 0.2f32;
    if (pitch_index as i32 - prefilter_period as i32).unsigned_abs() as usize * 10 > pitch_index {
        pf_threshold += 0.2;
        // Completely disable the prefilter on strong transients without continuity.
        if tf_estimate > 0.98 {
            gain1 = 0.0;
        }
    }
    if nb_available_bytes < 25 {
        pf_threshold += 0.1;
    }
    if nb_available_bytes < 35 {
        pf_threshold += 0.1;
    }
    if prefilter_gain > 0.4 {
        pf_threshold -= 0.1;
    }
    if prefilter_gain > 0.55 {
        pf_threshold -= 0.1;
    }
    pf_threshold = pf_threshold.max(0.2);

    let pf_on;
    if gain1 < pf_threshold {
        gain1 = 0.0;
        pf_on = false;
    } else {
        if (gain1 - prefilter_gain).abs() < 0.1 {
            gain1 = prefilter_gain;
        }
        let qg = ((gain1 * 32.0 / 3.0 + 0.5).floor() as i32 - 1).clamp(0, 7);
        gain1 = 0.09375 * (qg + 1) as f32;
        pf_on = true;
    }

    let before = &mut before[..channels];
    for c in 0..channels {
        let start = c * buf_stride + overlap;
        before[c] = sum_abs(&in_buf[start..start + frame_size]);
    }

    let offset = 0usize;
    let prev_period = prefilter_period.clamp(COMBFILTER_MINPERIOD, max_period - 2);

    for c in 0..channels {
        if offset > 0 {
            let pre_c = &pre[c * pre_size..];
            comb_filter(
                in_buf,
                pre_c,
                c * buf_stride + overlap,
                max_period,
                prev_period,
                prev_period,
                offset,
                -prefilter_gain,
                -prefilter_gain,
                prefilter_tapset,
                prefilter_tapset,
                window,
                0,
                arch,
            );
        }

        {
            let pre_c = &pre[c * pre_size..];
            comb_filter(
                in_buf,
                pre_c,
                c * buf_stride + overlap + offset,
                max_period + offset,
                prev_period,
                pitch_index,
                frame_size - offset,
                -prefilter_gain,
                -gain1,
                prefilter_tapset,
                tapset_decision,
                window,
                overlap,
                arch,
            );
        }
    }

    let after = &mut after[..channels];
    for c in 0..channels {
        let start = c * buf_stride + overlap;
        after[c] = sum_abs(&in_buf[start..start + frame_size]);
    }

    let cancel_pitch = if channels == 2 {
        // Stereo (celt_encoder.c:1555-1563): cancel if one channel gets
        // significantly worse, or neither gets significantly better.
        let thresh = [
            0.25 * gain1 * before[0] + 0.01 * before[1],
            0.25 * gain1 * before[1] + 0.01 * before[0],
        ];
        after[0] - before[0] > thresh[0]
            || after[1] - before[1] > thresh[1]
            || (before[0] - after[0] < thresh[0] && before[1] - after[1] < thresh[1])
    } else {
        after[0] > before[0]
    };

    // C 1572-1584: revert to a gain of zero, but still fade the previous
    // frame's filter out over the overlap. The decoder's post-filter always
    // cross-fades from the old gain, so dropping the fade here leaves 120
    // samples it post-filters that were never pre-filtered (issue #54).
    if cancel_pitch {
        for c in 0..channels {
            let body = c * buf_stride + overlap;
            in_buf[body..body + frame_size].copy_from_slice(
                &pre[c * pre_size + max_period..c * pre_size + max_period + frame_size],
            );
            comb_filter(
                in_buf,
                &pre[c * pre_size..],
                body + offset,
                max_period + offset,
                prev_period,
                pitch_index,
                overlap,
                -prefilter_gain,
                0.0,
                prefilter_tapset,
                tapset_decision,
                window,
                overlap,
                arch,
            );
        }
    }

    for c in 0..channels {
        push_prefilter_history(
            &mut prefilter_mem[c * max_period..(c + 1) * max_period],
            &pre[c * pre_size + max_period..c * pre_size + pre_size],
        );
    }

    if cancel_pitch {
        (false, 0.0, pitch_index)
    } else {
        (pf_on, gain1, pitch_index)
    }
}

/// Append one frame of pre-emphasized, unfiltered input to a channel's
/// `prefilter_mem` history (libopus `run_prefilter`, which runs every frame
/// whether or not the comb filter is enabled).
fn push_prefilter_history(mem: &mut [f32], frame: &[f32]) {
    let (n, max_period) = (frame.len(), mem.len());
    if n >= max_period {
        mem.copy_from_slice(&frame[n - max_period..]);
    } else {
        mem.copy_within(n.., 0);
        mem[max_period - n..].copy_from_slice(frame);
    }
}

/// Padding appended to `w_x` to absorb any SIMD over-shoot past the last band.
/// Sized to `MAX_PVQ_N` (352) — far larger than the widest SIMD access (16),
/// which is always loop-guarded (`while i + width <= n`). (Previously
/// `MAX_PVQ_N * 8`, which had no access pattern justifying it.)
const STRIDE_ACCESS_PAD: usize = crate::pvq::MAX_PVQ_N;

pub struct CeltEncoder {
    mode: &'static CeltMode,
    channels: usize,
    /// libopus `CELTEncoder.arch`: the level whose arithmetic the kernels
    /// reproduce, chosen at init.
    arch: crate::Arch,
    /// libopus `CELTEncoder.upsample`: the API input is zero-stuffed by this
    /// factor so the encoder always works at the mode's 48 kHz geometry.
    upsample: usize,
    /// End band the packet is coded to, from the Opus bandwidth (libopus
    /// `CELT_SET_END_BAND`): NB 13, MB/WB 17, SWB 19, FB 21. The decoder
    /// derives the same value from the TOC, so coding even one band more
    /// desynchronises the range coder (issue #37).
    end_band: usize,
    pub complexity: i32,
    syn_mem: FixedVec<f32, CELT_SYN_MEM>,
    old_band_e: FixedVec<f32, CELT_NB_X_CH>,
    preemph_mem: FixedVec<f32, CELT_MAX_CHANNELS>,
    tonal_average: i32,
    hf_average: i32,
    tapset_decision: i32,
    spread_decision: i32,
    intensity: i32,
    last_coded_bands: i32,
    prefilter_mem: FixedVec<f32, CELT_PREFILTER_MEM>,
    prefilter_period: usize,
    prefilter_gain: f32,
    prefilter_tapset: i32,
    old_band_e2: FixedVec<f32, CELT_NB_X_CH>,
    old_band_e3: FixedVec<f32, CELT_NB_X_CH>,
    last_band_log_e: FixedVec<f32, CELT_NB_X_CH>,
    delayed_intra: f32,
    /// libopus `energyError`: the last frame's fine-quantisation error,
    /// clamped to +-0.5, which biases stable bands' coarse energy.
    energy_error: [f32; CELT_NB_X_CH],
    /// libopus `CELTEncoder.rng`: the last frame's final range, which seeds
    /// the resynthesis's folding noise.
    rng: u32,
    /// libopus `CELT_SET_CHANNELS`: the channels coded, 1 for a mono stream
    /// of stereo input.
    stream_channels: usize,
    lsb_depth: i32,
    overlap_max: f32,
    bitrate: i32,
    vbr: bool,
    constrained_vbr: bool,
    vbr_reservoir: i32,
    vbr_drift: i32,
    vbr_offset: i32,
    vbr_count: i32,

    w_in_buf: FixedVec<f32, CELT_BUFSTRIDE>,
    w_freq: FixedVec<f32, CELT_W_FREQ>,
    w_band_e: FixedVec<f32, CELT_NB_X_CH>,
    w_band_e2: FixedVec<f32, CELT_NB_X_CH>,
    w_x: FixedVec<f32, CELT_W_X_ENC>,
    w_band_log_e: FixedVec<f32, CELT_NB_X_CH>,
    w_band_log_e2: FixedVec<f32, CELT_NB_X_CH>,
    w_error: FixedVec<f32, CELT_NB_X_CH>,
    w_tf_res: FixedVec<i32, CELT_NB_EBANDS>,
    w_cap: FixedVec<i32, CELT_NB_EBANDS>,
    w_offsets: FixedVec<i32, CELT_NB_EBANDS>,
    w_pulses: FixedVec<i32, CELT_NB_EBANDS>,
    w_ebits: FixedVec<i32, CELT_NB_X_CH>,
    w_fine_priority: FixedVec<i32, CELT_NB_X_CH>,
    w_collapse_masks: FixedVec<u32, CELT_NB_X_CH>,
    consec_transient: i32,

    w_prefilter_pre: FixedVec<f32, CELT_PREFILTER_PRE>,
    w_prefilter_pitch_buf: FixedVec<f32, CELT_PREFILTER_PITCH>,
    w_prefilter_before: FixedVec<f32, CELT_MAX_CHANNELS>,
    w_prefilter_after: FixedVec<f32, CELT_MAX_CHANNELS>,

    w_transient_tmp: FixedVec<f32, MAX_TRANSIENT_LEN>,

    analysis: AnalysisInfo,
    /// libopus `stereo_saving`: the bits the stereo correlation saves, which
    /// the trim analysis smooths across frames and the VBR target uses.
    stereo_saving: f32,
    /// libopus `spec_avg`: the average spectral level, for temporal VBR.
    spec_avg: f32,
    loss_rate: i32,
    /// libopus `SILKInfo`: the signal type and quantization offset of the
    /// SILK layer, which steer the hybrid CELT layer.
    silk_signal_type: i32,
    silk_offset: i32,
    /// libopus `CELT_SET_PREDICTION`: below 2 the pitch pre-filter is off,
    /// at 0 the energy is coded intra.
    disable_pf: bool,
    force_intra: bool,
    /// No phase inversion in intensity stereo (libopus `disable_inv`,
    /// `OPUS_SET_PHASE_INVERSION_DISABLED`).
    pub(crate) disable_inv: bool,
    /// The stream is a surround LFE channel (libopus `lfe`,
    /// `OPUS_SET_LFE`).
    pub(crate) lfe: bool,
    /// The surround masking of the multistream encoder per band and
    /// channel (libopus `energy_mask`, `OPUS_SET_ENERGY_MASK`).
    pub(crate) energy_mask: Option<[f32; 2 * CELT_NB_EBANDS]>,
    /// The last frame's allocation decisions, which unit tests inspect.
    #[cfg(test)]
    decisions: FrameDecisions,
}

#[cfg(test)]
#[derive(Clone, Copy, Default)]
struct FrameDecisions {
    alloc_trim: i32,
    tf_select: i32,
    tf_res: [i32; MAX_NB_EBANDS],
    is_transient: bool,
    weak_transient: bool,
}

const INTEN_THRESHOLDS: [i32; 21] = [
    1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 36, 44, 50, 56, 62, 67, 72, 79, 88, 106, 134,
];
const INTEN_HYSTERESIS: [i32; 21] = [
    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 8, 8,
];

fn hysteresis_decision(val: i32, thresholds: &[i32], hysteresis: &[i32], prev: i32) -> i32 {
    let mut i = 0;
    while i < thresholds.len() {
        if val < thresholds[i] {
            break;
        }
        i += 1;
    }
    let mut res = i as i32;
    if res > prev && val < thresholds[prev as usize] + hysteresis[prev as usize] {
        res = prev;
    }
    if res < prev && res > 0 && val > thresholds[prev as usize - 1] - hysteresis[prev as usize - 1]
    {
        res = prev;
    }
    res
}

#[allow(clippy::too_many_arguments)]
fn alloc_trim_analysis(
    mode: &CeltMode,
    x: &[f32],
    band_log_e: &[f32],
    end: usize,
    lm: i32,
    channels: usize,
    n0: usize,
    analysis: &AnalysisInfo,
    stereo_saving: &mut f32,
    tf_estimate: f32,
    intensity: i32,
    surround_trim: f32,
    equiv_rate: i32,
    arch: crate::Arch,
) -> i32 {
    let mut trim = 5.0f32;
    if equiv_rate < 64000 {
        trim = 4.0;
    } else if equiv_rate < 80000 {
        let frac = (equiv_rate - 64000) >> 10;
        trim = 4.0 + (1.0 / 16.0) * frac as f32;
    }

    if channels == 2 {
        let mut sum = 0.0f32;
        for i in 0..8 {
            let offset = (mode.e_bands[i] as usize) << lm;
            let n = ((mode.e_bands[i + 1] - mode.e_bands[i]) as usize) << lm;
            sum += crate::pitch::inner_prod(&x[offset..], &x[n0 + offset..], n, arch);
        }
        sum = (sum / 8.0).abs().min(1.0);
        let mut min_xc = sum;
        for i in 8..intensity as usize {
            let offset = (mode.e_bands[i] as usize) << lm;
            let n = ((mode.e_bands[i + 1] - mode.e_bands[i]) as usize) << lm;
            let partial = crate::pitch::inner_prod(&x[offset..], &x[n0 + offset..], n, arch);
            min_xc = min_xc.min(partial.abs());
        }
        min_xc = min_xc.min(1.0);

        let log_xc = celt_log2(1.001 - sum * sum);
        let log_xc2 = (log_xc * 0.5).max(celt_log2(1.001 - min_xc * min_xc));

        trim += (-4.0f32).max(0.75 * log_xc);
        *stereo_saving = (*stereo_saving + 0.25).min(-0.5 * log_xc2);
    }

    let mut diff = 0.0f32;
    for c in 0..channels {
        for i in 0..end - 1 {
            diff += band_log_e[c * mode.nb_ebands + i] * (2 + 2 * i as i32 - end as i32) as f32;
        }
    }
    diff /= (channels * (end - 1)) as f32;
    trim -= (-2.0f32).max(2.0f32.min((diff + 1.0) / 6.0));
    trim -= surround_trim;
    trim -= 2.0 * tf_estimate;
    // A tilt of the tonality towards high frequencies lowers the trim.
    if analysis.valid {
        trim -= (-2.0f32).max(2.0f32.min(2.0 * (analysis.tonality_slope + 0.05)));
    }

    let trim_index = (trim + 0.5).floor() as i32;
    trim_index.clamp(0, 10)
}

#[inline(always)]
fn median3(a: f32, b: f32, c: f32) -> f32 {
    let mut v = [a, b, c];
    crate::compat::sort_by(&mut v[..], |x, y| {
        x.partial_cmp(y).unwrap_or(core::cmp::Ordering::Equal)
    });
    v[1]
}

#[inline(always)]
fn median5(v: &[f32]) -> f32 {
    let mut x = [v[0], v[1], v[2], v[3], v[4]];
    crate::compat::sort_by(&mut x[..], |a, b| {
        a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal)
    });
    x[2]
}

/// The largest of `init` and the magnitudes in `x`. `f32::max` skips NaN
/// and otherwise gives the same result in any order, so eight running
/// maxima equal the sequential fold.
fn max_abs(x: &[f32], init: f32) -> f32 {
    let mut m = [init; 8];
    let blocks = x.chunks_exact(8);
    let rest = blocks.remainder();
    for b in blocks {
        for (m, v) in m.iter_mut().zip(b) {
            *m = m.max(v.abs());
        }
    }
    rest.iter().chain(&m).fold(init, |m, v| m.max(v.abs()))
}

const TONE_LAGS: [usize; 8] = [0, 1, 2, 4, 8, 16, 32, 64];

/// Adds x[i]*x[i+TONE_LAGS[j]] to acc[j] for i in start..end, j < K.
#[inline(always)]
fn tone_sums<const K: usize>(acc: &mut [f32; 8], x: &[f32], start: usize, end: usize) {
    for i in start..end {
        let w = &x[i..=i + TONE_LAGS[K - 1]];
        for j in 0..K {
            acc[j] += w[0] * w[TONE_LAGS[j]];
        }
    }
}

/// tone_lpc's forward correlations [r00, r01, r02] for the delays 1 << t,
/// t < nd: the sums of x[i]*x[i], x[i]*x[i+d] and x[i]*x[i+2d] over
/// i < len-2d. The sum at lag k for delay k and the one for delay k/2 are
/// prefixes of one sum, so one pass in ascending i gives every delay, each
/// sum added in tone_lpc's order. Needs nd <= 6 and len > 2 << (nd - 1).
fn tone_correlations(x: &[f32], nd: usize) -> [[f32; 3]; 6] {
    let mut acc = [0.0f32; 8];
    let mut r = [[0.0; 3]; 6];
    let mut start = 0;
    for t in (0..nd).rev() {
        let end = x.len() - (2 << t);
        // The sums at lags up to 2 << t.
        match t {
            0 => tone_sums::<3>(&mut acc, x, start, end),
            1 => tone_sums::<4>(&mut acc, x, start, end),
            2 => tone_sums::<5>(&mut acc, x, start, end),
            3 => tone_sums::<6>(&mut acc, x, start, end),
            4 => tone_sums::<7>(&mut acc, x, start, end),
            _ => tone_sums::<8>(&mut acc, x, start, end),
        }
        r[t] = [acc[0], acc[t + 1], acc[t + 2]];
        start = end;
    }
    r
}

// Port of celt_encoder.c tone_lpc / tone_detect (float path, no FIXED_POINT shifts).
// `r` holds the first loop's r00, r01, r02 (tone_correlations).
fn tone_lpc(x: &[f32], len: usize, delay: usize, r: [f32; 3], lpc: &mut [f32; 2]) -> bool {
    // returns true on fail
    if len <= 2 * delay {
        return true;
    }
    let [mut r00, mut r01, mut r02] = r;
    let mut r11: f32;
    let mut r12: f32;
    let mut edges = 0.0f32;
    for i in 0..delay {
        edges += x[len + i - 2 * delay] * x[len + i - 2 * delay] - x[i] * x[i];
    }
    r11 = r00 + edges;
    edges = 0.0;
    for i in 0..delay {
        edges += x[len + i - delay] * x[len + i - delay] - x[i + delay] * x[i + delay];
    }
    let r22: f32 = r11 + edges;
    edges = 0.0;
    for i in 0..delay {
        edges += x[len + i - 2 * delay] * x[len + i - delay] - x[i] * x[i + delay];
    }
    r12 = r01 + edges;
    // Reverse and sum to get backward contribution (float: simple sums)
    let (rr00, rr01, rr11, rr02, rr12) = (r00 + r22, r01 + r12, 2.0 * r11, 2.0 * r02, r12 + r01);
    r00 = rr00;
    r01 = rr01;
    r11 = rr11;
    r02 = rr02;
    r12 = rr12;
    let den = r00 * r11 - r01 * r01;
    if den < 0.001 * (r00 * r11) {
        return true;
    }
    // celt_encoder.c:1349-1356: the limits apply where the quotient
    // reaches 1 or 2.
    let num1 = r02 * r11 - r01 * r12;
    lpc[1] = if num1 >= den {
        1.0
    } else if num1 <= -den {
        -1.0
    } else {
        num1 / den
    };
    let num0 = r00 * r12 - r02 * r01;
    lpc[0] = if 0.5 * num0 >= den {
        1.999999
    } else if 0.5 * num0 <= -den {
        -1.999999
    } else {
        num0 / den
    };
    false
}

/// Whether 1 - lpc[0] z^-1 - lpc[1] z^-2 has complex roots: the float
/// square plus the double product QCONST32(3.999999,29)*lpc[1] below 0,
/// summed in double (celt_encoder.c:1383).
fn tone_has_complex_roots(lpc: [f32; 2]) -> bool {
    f64::from(lpc[0] * lpc[0]) + 3.999999 * f64::from(lpc[1]) < 0.0
}

/// The tone's frequency, acos(.5f*lpc[0])/delay (celt_encoder.c:1394):
/// a double acos and division, rounded once to float.
fn tone_freq(lpc0: f32, delay: usize) -> f32 {
    let x = f64::from(0.5 * lpc0);
    #[cfg(feature = "std")]
    let a = x.acos();
    #[cfg(not(feature = "std"))]
    let a = crate::compat::Math::acos(x);
    (a / delay as f64) as f32
}

fn tone_detect(in_buf: &[f32], cc: usize, n: usize, fs: i32) -> (f32, f32) {
    // in_buf holds n = frame + overlap samples per channel. Stereo sums
    // the channels (celt_encoder.c:1374; SHR32 is an identity in float).
    let mut sum;
    let x = if cc == 2 {
        sum = [0.0f32; 2048];
        for i in 0..n {
            sum[i] = in_buf[i] + in_buf[i + n];
        }
        &sum[..n]
    } else {
        &in_buf[..n]
    };
    // The delays the loop below can reach, 1 << t for t < nd, with len > 2*delay.
    let mut nd = 1;
    while 1 << (nd - 1) <= fs as usize / 3000 {
        nd += 1;
    }
    while nd > 0 && n <= 2 << (nd - 1) {
        nd -= 1;
    }
    let r = tone_correlations(x, nd);
    let mut delay = 1usize;
    let mut lpc = [0.0f32; 2];
    let mut fail = tone_lpc(x, n, delay, r[0], &mut lpc);
    while delay <= (fs as usize / 3000) && (fail || (lpc[0] > 1.0 && lpc[1] < 0.0)) {
        delay *= 2;
        fail = tone_lpc(x, n, delay, r[delay.trailing_zeros() as usize], &mut lpc);
    }
    if !fail && tone_has_complex_roots(lpc) {
        (tone_freq(lpc[0], delay), -lpc[1])
    } else {
        (-1.0, 0.0)
    }
}

#[allow(clippy::too_many_arguments)]
fn dynalloc_analysis_simple(
    mode: &CeltMode,
    band_log_e: &[f32],
    band_log_e2: Option<&[f32]>,
    old_band_e: &[f32],
    start: usize,
    end: usize,
    channels: usize,
    lm: usize,
    effective_bytes: usize,
    is_transient: bool,
    offsets: &mut [i32],
    lsb_depth: i32,
    vbr: bool,
    constrained_vbr: bool,
    analysis: &AnalysisInfo,
    lfe: bool,
    surround_dynalloc: &[f32],
    tone_freq: f32,
    toneishness: f32,
    importance: &mut [i32],
    spread_weight: &mut [i32],
) -> (f32, i32) {
    offsets.fill(0);
    let nb = mode.nb_ebands;
    // celt_encoder.c:1063-1117: the noise floor, the deepest band above it,
    // and from a simple masking model the weights of the spreading decision.
    let mut noise_floor = [0.0f32; CELT_NB_EBANDS];
    let mut max_depth = -31.9f32;
    for i in 0..end {
        noise_floor[i] = 0.0625 * mode.log_n[i] as f32 + 0.5 + (9 - lsb_depth) as f32
            - mode.e_means[i]
            + 0.0062 * (i + 5) as f32 * (i + 5) as f32;
    }
    for c in 0..channels {
        for i in 0..end {
            max_depth = max_depth.max(band_log_e[c * nb + i] - noise_floor[i]);
        }
    }
    let mut mask = [0.0f32; CELT_NB_EBANDS];
    for i in 0..end {
        mask[i] = band_log_e[i] - noise_floor[i];
        if channels == 2 {
            mask[i] = mask[i].max(band_log_e[nb + i] - noise_floor[i]);
        }
    }
    let sig = mask;
    for i in 1..end {
        mask[i] = mask[i].max(mask[i - 1] - 2.0);
    }
    for i in (0..end.saturating_sub(1)).rev() {
        mask[i] = mask[i].max(mask[i + 1] - 3.0);
    }
    for i in 0..end {
        let smr = sig[i] - (max_depth - 12.0).max(0.0).max(mask[i]);
        let shift = (-((0.5 + smr).floor() as i32)).clamp(0, 5);
        spread_weight[i] = 32 >> shift;
    }

    if effective_bytes < (30 + 5 * lm) || lfe {
        for v in importance.iter_mut() {
            *v = 13;
        }
        return (max_depth, 0);
    }

    let mut follower: FixedVec<f32, CELT_NB_X_CH> = FixedVec::from_value(0.0f32, nb * channels);
    // One `last` for both channels (celt_encoder.c:1123).
    let mut last = 0usize;
    for c in 0..channels {
        let base = c * nb;
        let src = if let Some(b2) = band_log_e2 {
            b2
        } else {
            band_log_e
        };
        let mut band_log_e3: FixedVec<f32, CELT_NB_EBANDS> = FixedVec::from_value(0.0f32, end);
        for i in 0..end {
            let mut e = src[base + i];
            if lm == 0 && i < 8 {
                e = e.max(old_band_e[base + i]);
            }
            band_log_e3[i] = e;
        }

        follower[base] = band_log_e3[0];
        for i in 1..end {
            if band_log_e3[i] > band_log_e3[i - 1] + 0.5 {
                last = i;
            }
            follower[base + i] = (follower[base + i - 1] + 1.5).min(band_log_e3[i]);
        }
        for i in (0..last).rev() {
            follower[base + i] =
                follower[base + i].min((follower[base + i + 1] + 2.0).min(band_log_e3[i]));
        }

        let offset = 1.0f32;
        if end >= 5 {
            for i in 2..end - 2 {
                follower[base + i] =
                    follower[base + i].max(median5(&band_log_e3[i - 2..i + 3]) - offset);
            }
        }
        if end >= 3 {
            let l = median3(band_log_e3[0], band_log_e3[1], band_log_e3[2]) - offset;
            follower[base] = follower[base].max(l);
            follower[base + 1] = follower[base + 1].max(l);

            let r = median3(
                band_log_e3[end - 3],
                band_log_e3[end - 2],
                band_log_e3[end - 1],
            ) - offset;
            follower[base + end - 2] = follower[base + end - 2].max(r);
            follower[base + end - 1] = follower[base + end - 1].max(r);
        }
        // Clamp to noise floor (float C: GCONST etc are no-ops)
        for i in 0..end {
            follower[base + i] = follower[base + i].max(noise_floor[i]);
        }
    }
    if channels == 2 {
        for i in start..end {
            let l = follower[i];
            let r = follower[nb + i];
            let r2 = r.max(l - 4.0);
            let l2 = l.max(r - 4.0);
            follower[i] =
                ((band_log_e[i] - l2).max(0.0) + (band_log_e[nb + i] - r2).max(0.0)) * 0.5;
        }
    } else {
        for i in start..end {
            follower[i] = (band_log_e[i] - follower[i]).max(0.0);
        }
    }
    for i in start..end {
        // C 1182-1183: follower = max(follower, surround_dynalloc)
        follower[i] = follower[i].max(surround_dynalloc[i]);
    }
    // C 1184-1191: importance = floor(0.5+13*exp2(min(follower,4)))
    for i in start..end {
        let v = follower[i].min(4.0);
        importance[i] = (0.5 + 13.0 * celt_exp2(v)).floor() as i32;
    }
    if (!vbr || constrained_vbr) && !is_transient {
        for i in start..end {
            follower[i] *= 0.5;
        }
    }
    // Band weighting before capping: C does scale-then-min(4) (celt_encoder.c 1198-1204 + 1239).
    // Rust was min(4)-then-scale, which diverges for follower>2 in i<8 or >4 in i>=12.
    for i in start..end {
        if i < 8 {
            follower[i] *= 2.0;
        }
        if i >= 12 {
            follower[i] *= 0.5;
        }
    }
    // C 1206-1222: Compensate for Opus' under-allocation on tones
    if toneishness > 0.98 {
        // The division by M_PI, the .5 and the floor are double (C 1210).
        let freq_bin = (0.5 + f64::from(tone_freq * 120.0) / core::f64::consts::PI).floor() as i32;
        for i in start..end {
            let lo = mode.e_bands[i] as i32;
            let hi = mode.e_bands[i + 1] as i32;
            if freq_bin >= lo && freq_bin <= hi {
                follower[i] += 2.0;
            }
            if freq_bin >= lo - 1 && freq_bin <= hi + 1 {
                follower[i] += 1.0;
            }
            if freq_bin >= lo - 2 && freq_bin <= hi + 2 {
                follower[i] += 1.0;
            }
            if freq_bin >= lo - 3 && freq_bin <= hi + 3 {
                follower[i] += 0.5;
            }
        }
        if freq_bin >= mode.e_bands[end] as i32 {
            if end >= 1 {
                follower[end - 1] += 2.0;
            }
            if end >= 2 {
                follower[end - 2] += 1.0;
            }
        }
    }
    if analysis.valid {
        // C 1226-1230: leak_boost
        for i in start..end.min(19) {
            follower[i] += (1.0 / 64.0) * analysis.leak_boost[i] as f32;
        }
    }
    // At high rates the first band gets up to 1.5 more (celt_encoder.c:1232).
    if effective_bytes > 320 {
        follower[0] += 1.5f32.min(1e-3 * (effective_bytes - 320) as f32);
    }
    let mut tot_boost = 0i32;
    for i in start..end {
        let f = follower[i].min(4.0);

        let width = channels as i32 * (mode.e_bands[i + 1] - mode.e_bands[i]) as i32 * (1 << lm);
        let (boost, boost_bits) = if width < 6 {
            let b = f.floor().max(0.0) as i32;
            (b, (b * width) << BITRES)
        } else if width > 48 {
            let b = (f * 8.0).floor().max(0.0) as i32;
            (b, ((b * width) << BITRES) / 8)
        } else {
            let b = (f * width as f32 / 6.0).floor().max(0.0) as i32;
            (b, (b * 6) << BITRES)
        };

        // For CBR and non-transient CVBR frames, limit dynalloc to 2/3 of
        // the bits (celt_encoder.c:1253-1260); the last band gets the rest
        // of the cap in eighth bits.
        if (!vbr || (constrained_vbr && !is_transient))
            && (tot_boost + boost_bits) >> BITRES >> 3 > 2 * effective_bytes as i32 / 3
        {
            let cap_bits = (2 * effective_bytes as i32 / 3) << BITRES << 3;
            offsets[i] = cap_bits - tot_boost;
            tot_boost = cap_bits;
            break;
        }

        offsets[i] = boost;
        tot_boost += boost_bits;
    }
    (max_depth, tot_boost)
}

impl CeltEncoder {
    pub fn new(mode: &'static CeltMode, channels: usize) -> Self {
        Self::with_sampling_rate(mode, channels, 48000)
    }

    /// CELT encoder for `sampling_rate`, which need not be 48 kHz: the input
    /// is zero-stuffed by `resampling_factor(sampling_rate)` so the mode's
    /// 48 kHz geometry is used internally, exactly like libopus
    /// (`celt_encoder.c` `st->upsample`).
    pub fn with_sampling_rate(
        mode: &'static CeltMode,
        channels: usize,
        sampling_rate: i32,
    ) -> Self {
        // All internal buffers are sized for at most 2 channels; a larger
        // value would panic later inside `FixedVec::from_value` with an
        // unhelpful message (issue #27 deep scan).
        assert!(
            (1..=2).contains(&channels),
            "CeltEncoder::new: channels must be 1 or 2 (got {channels})"
        );
        let upsample = resampling_factor(sampling_rate);
        let overlap = mode.overlap;
        let channel_mem_size = 2048 + overlap;
        let syn_mem_size = channels * channel_mem_size;
        let nb_ebands = mode.nb_ebands;
        let nb_x_ch = nb_ebands * channels;
        let frame_x_ch = MAX_FRAME_SIZE * channels;
        let bufstride_x_ch = (MAX_FRAME_SIZE + overlap) * channels;
        // libopus initialises through OPUS_RESET_STATE
        // (celt_encoder.c:229).
        let mut enc = Self {
            mode,
            channels,
            arch: crate::Arch::select(),
            upsample,
            end_band: mode.nb_ebands,
            // celt_encoder.c:220.
            complexity: 5,
            syn_mem: FixedVec::from_value(0.0, syn_mem_size),
            old_band_e: FixedVec::from_value(0.0, nb_x_ch),
            preemph_mem: FixedVec::from_value(0.0, channels),
            tonal_average: 256,
            hf_average: 0,
            tapset_decision: 0,
            spread_decision: SPREAD_NORMAL,
            intensity: 0,
            last_coded_bands: 0,
            prefilter_mem: FixedVec::from_value(0.0, channels * COMBFILTER_MAXPERIOD),
            prefilter_period: 0,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            old_band_e2: FixedVec::from_value(0.0, nb_x_ch),
            old_band_e3: FixedVec::from_value(0.0, nb_x_ch),
            last_band_log_e: FixedVec::from_value(0.0, nb_x_ch),
            delayed_intra: 0.0,
            energy_error: [0.0; CELT_NB_X_CH],
            rng: 0,
            stream_channels: channels,
            lsb_depth: 24,
            overlap_max: 0.0,
            bitrate: OPUS_BITRATE_MAX,
            vbr: false,
            constrained_vbr: true,
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,

            w_in_buf: FixedVec::from_value(0.0, bufstride_x_ch),
            w_freq: FixedVec::from_value(0.0, frame_x_ch + 4),
            w_band_e: FixedVec::from_value(0.0, nb_x_ch),
            w_band_e2: FixedVec::from_value(0.0, nb_x_ch),

            w_x: FixedVec::from_value(0.0, frame_x_ch + STRIDE_ACCESS_PAD),
            w_band_log_e: FixedVec::from_value(0.0, nb_x_ch),
            w_band_log_e2: FixedVec::from_value(0.0, nb_x_ch),
            w_error: FixedVec::from_value(0.0, nb_x_ch),
            w_tf_res: FixedVec::from_value(0, nb_ebands),
            w_cap: FixedVec::from_value(0, nb_ebands),
            w_offsets: FixedVec::from_value(0, nb_ebands),
            w_pulses: FixedVec::from_value(0, nb_ebands),
            w_ebits: FixedVec::from_value(0, nb_x_ch),
            w_fine_priority: FixedVec::from_value(0, nb_x_ch),
            w_collapse_masks: FixedVec::from_value(0, nb_x_ch),

            w_prefilter_pre: FixedVec::from_value(
                0.0,
                channels * (COMBFILTER_MAXPERIOD + MAX_FRAME_SIZE),
            ),
            w_prefilter_pitch_buf: FixedVec::from_value(
                0.0,
                (COMBFILTER_MAXPERIOD + MAX_FRAME_SIZE) >> 1,
            ),
            w_prefilter_before: FixedVec::from_value(0.0, channels),
            w_prefilter_after: FixedVec::from_value(0.0, channels),
            w_transient_tmp: FixedVec::from_value(0.0, MAX_TRANSIENT_LEN),
            consec_transient: 0,

            analysis: AnalysisInfo::default(),
            stereo_saving: 0.0,
            spec_avg: 0.0,
            loss_rate: 0,
            silk_signal_type: 0,
            silk_offset: 0,
            disable_pf: false,
            force_intra: false,
            disable_inv: false,
            lfe: false,
            energy_mask: None,
            #[cfg(test)]
            decisions: FrameDecisions::default(),
        };
        enc.reset_state();
        enc
    }

    /// libopus `OPUS_RESET_STATE`: forget the signal history and the rate
    /// control, keeping the settings (complexity, bands, rate, VBR mode,
    /// loss rate, prediction).
    pub fn reset_state(&mut self) {
        self.syn_mem.fill(0.0);
        self.old_band_e.fill(0.0);
        self.preemph_mem.fill(0.0);
        self.tonal_average = 256;
        self.hf_average = 0;
        self.tapset_decision = 0;
        self.spread_decision = SPREAD_NORMAL;
        self.intensity = 0;
        self.last_coded_bands = 0;
        self.prefilter_mem.fill(0.0);
        self.prefilter_period = 0;
        self.prefilter_gain = 0.0;
        self.prefilter_tapset = 0;
        self.old_band_e2.fill(-28.0);
        self.old_band_e3.fill(-28.0);
        self.last_band_log_e.fill(0.0);
        self.delayed_intra = 1.0;
        self.energy_error = [0.0; CELT_NB_X_CH];
        self.rng = 0;
        self.overlap_max = 0.0;
        self.vbr_reservoir = 0;
        self.vbr_drift = 0;
        self.vbr_offset = 0;
        self.vbr_count = 0;
        self.consec_transient = 0;
        self.analysis = AnalysisInfo::default();
        self.stereo_saving = 0.0;
        self.spec_avg = 0.0;
        self.silk_signal_type = 0;
        self.silk_offset = 0;
        self.energy_mask = None;
    }

    /// libopus `CELT_SET_PREDICTION` (0, 1 or 2; other values are ignored,
    /// as libopus rejects them).
    pub fn set_prediction(&mut self, prediction: i32) {
        if (0..=2).contains(&prediction) {
            self.disable_pf = prediction <= 1;
            self.force_intra = prediction == 0;
        }
    }

    /// Set the highest coded band (`CELT_SET_END_BAND`). Must match the
    /// bandwidth the packet's TOC signals.
    /// libopus `CELT_SET_CHANNELS`: code `channels` (1 or 2, at most the
    /// input's) channels; other values are ignored.
    pub fn set_stream_channels(&mut self, channels: usize) {
        if (1..=2).contains(&channels) {
            self.stream_channels = channels.min(self.channels);
        }
    }

    /// libopus `CELT_SET_END_BAND` (1 to the band count; other values are
    /// ignored).
    pub fn set_end_band(&mut self, end: usize) {
        if (1..=self.mode.nb_ebands).contains(&end) {
            self.end_band = end;
        }
    }

    /// libopus `celt_encode_with_ec` with the coder's size as the byte
    /// budget. Errors with `BAD_ARG` for a budget below 2 bytes or a
    /// frame size other than 2.5, 5, 10 or 20 ms, leaving the state as is.
    pub fn encode(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        rc: &mut RangeCoder,
    ) -> Result<(), &'static str> {
        self.encode_impl(pcm, frame_size, rc, 0, None)
    }

    pub fn encode_with_start_band(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        rc: &mut RangeCoder,
        start_band: usize,
    ) -> Result<(), &'static str> {
        self.encode_impl(pcm, frame_size, rc, start_band, None)
    }

    pub fn encode_with_budget(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        rc: &mut RangeCoder,
        start_band: usize,
        total_bits: i32,
    ) -> Result<(), &'static str> {
        self.encode_impl(pcm, frame_size, rc, start_band, Some(total_bits))
    }

    /// As [`Self::encode_with_budget`]. `_is_vbr` has no effect: the frame is
    /// VBR when the encoder has VBR on and a bitrate (libopus `vbr_rate`).
    pub fn encode_with_budget_vbr(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        rc: &mut RangeCoder,
        start_band: usize,
        total_bits: i32,
        _is_vbr: bool,
    ) -> Result<(), &'static str> {
        self.encode_impl(pcm, frame_size, rc, start_band, Some(total_bits))
    }

    /// libopus `OPUS_SET_LSB_DEPTH` (8 to 24; other values are ignored).
    pub fn set_lsb_depth(&mut self, depth: i32) {
        if (8..=24).contains(&depth) {
            self.lsb_depth = depth;
        }
    }

    pub fn get_lsb_depth(&self) -> i32 {
        self.lsb_depth
    }
    /// libopus `OPUS_SET_BITRATE` on the CELT encoder: at most 750 kb/s per
    /// channel; rates of 500 b/s or less other than `OPUS_BITRATE_MAX` (-1)
    /// are ignored (celt_encoder.c:3001-3009).
    pub fn set_bitrate(&mut self, bitrate: i32) {
        if bitrate > 500 || bitrate == OPUS_BITRATE_MAX {
            self.bitrate = bitrate.min(750000 * self.channels as i32);
        }
    }
    pub fn set_vbr(&mut self, vbr: bool) {
        self.vbr = vbr;
    }
    /// libopus `OPUS_SET_PACKET_LOSS_PERC` on the CELT encoder (0 to 100;
    /// other values are ignored).
    pub fn set_loss_rate(&mut self, loss_rate: i32) {
        if (0..=100).contains(&loss_rate) {
            self.loss_rate = loss_rate;
        }
    }
    /// libopus `CELT_SET_SILK_INFO`: the hybrid SILK layer's signal type
    /// and quantization offset (Q10).
    pub fn set_silk_info(&mut self, signal_type: i32, offset: i32) {
        self.silk_signal_type = signal_type;
        self.silk_offset = offset;
    }
    pub fn set_constrained_vbr(&mut self, cvbr: bool) {
        self.constrained_vbr = cvbr;
    }
    /// libopus `CELT_SET_ANALYSIS`: the encoder's signal analysis of the
    /// frame, which steers the allocation, the trim, the VBR target and the
    /// pitch pre-filter where it is valid.
    pub fn set_analysis(&mut self, analysis: &AnalysisInfo) {
        self.analysis = *analysis;
    }

    fn encode_impl(
        &mut self,
        pcm: &[f32],
        api_frame_size: usize,
        rc: &mut RangeCoder,
        start_band: usize,
        explicit_total_bits: Option<i32>,
    ) -> Result<(), &'static str> {
        let mode = self.mode;
        // libopus CC (input channels) and C (coded channels): a mono stream
        // of stereo input codes the two channels' average.
        let cc = self.channels;
        let channels = self.stream_channels.min(cc);
        let upsample = self.upsample;
        // Internal (48 kHz) frame size; `pcm` is at the API rate.
        let frame_size = api_frame_size * upsample;
        let nb_ebands = mode.nb_ebands;
        let end = self.end_band.clamp(1, nb_ebands);
        let eff_end = end.min(mode.eff_ebands);
        let overlap = mode.overlap;

        // libopus's nbCompressedBytes is the coder's size or the budget;
        // the frame is 120 << lm samples at 48 kHz (celt_encoder.c:1832-1845).
        if explicit_total_bits.map_or(rc.storage as i32, |b| b / 8) < 2 {
            return Err(crate::error::BAD_ARG);
        }
        let Some(lm) = (0..=mode.max_lm).find(|&lm| mode.short_mdct_size << lm == frame_size)
        else {
            return Err(crate::error::BAD_ARG);
        };
        // The bands start_band..end are coded; an empty range is refused.
        if start_band >= end {
            return Err(crate::error::BAD_ARG);
        }

        dumpf!(
            "analysis",
            &[
                self.analysis.valid as i32 as f32,
                self.analysis.tonality,
                self.analysis.tonality_slope,
                self.analysis.noisiness,
                self.analysis.activity,
                self.analysis.music_prob,
                self.analysis.bandwidth as f32,
                self.analysis.activity_probability,
                self.analysis.max_pitch_ratio,
            ]
        );
        dumpf!(
            "pcm",
            &(0..api_frame_size * cc)
                .map(|i| pcm[(i % cc) * api_frame_size + i / cc])
                .collect::<Vec<f32>>()
        );
        // --- Silence detection (port of libopus celt_encoder.c: sample_max / overlap_max / lsb_depth) ---
        // libopus measures sample/overlap maxima on the API-rate `pcm`:
        // CC*(N-overlap)/upsample and CC*overlap/upsample interleaved
        // samples, the first (N-overlap)/upsample and the next
        // overlap/upsample samples of every channel.
        let head = (frame_size - overlap) / upsample;
        let tail = overlap / upsample;
        let channel_max = |r: core::ops::Range<usize>, init: f32| {
            (0..cc).fold(init, |m, c| {
                max_abs(&pcm[c * api_frame_size..][r.clone()], m)
            })
        };
        let sample_max = channel_max(0..head, self.overlap_max);
        let new_overlap_max = channel_max(head..head + tail, 0.0);
        self.overlap_max = new_overlap_max;
        let sample_max = sample_max.max(new_overlap_max);
        // Input beyond +-65536 clips the frame to +-2.0 (celt_encoder.c:2010-2016,
        // 592-597).
        let need_clip = sample_max > 65536.0;
        let syn_mem_size = 2048 + overlap;
        for c in 0..cc {
            let channel_offset = c * syn_mem_size;

            self.syn_mem.copy_within(
                channel_offset + frame_size..channel_offset + syn_mem_size,
                channel_offset,
            );

            let mut m = self.preemph_mem[c];
            let coef = mode.preemph[0];
            // Zero-stuff the API-rate input to 48 kHz (libopus celt_preemphasis
            // with `upsample`), then run the pre-emphasis filter over all N
            // internal samples.
            let syn = &mut self.syn_mem[channel_offset + syn_mem_size - frame_size..][..frame_size];
            let input = &pcm[c * api_frame_size..];
            let scale = |v: f32| {
                let x = v * 32768.0;
                if need_clip {
                    x.clamp(-65536.0, 65536.0)
                } else {
                    x
                }
            };
            if upsample == 1 {
                for (y, &v) in syn.iter_mut().zip(input) {
                    let x = scale(v);
                    *y = x - m;
                    m = x * coef;
                }
                self.preemph_mem[c] = m;
                continue;
            }
            for (chunk, &v) in syn.chunks_mut(upsample).zip(input) {
                for (k, y) in chunk.iter_mut().enumerate() {
                    let x = if k == 0 { scale(v) } else { 0.0 };
                    *y = x - m;
                    m = x * coef;
                }
            }
            self.preemph_mem[c] = m;
        }

        let buf_stride = frame_size + overlap;
        let in_buf = &mut self.w_in_buf[..buf_stride * cc];
        for c in 0..cc {
            let channel_offset = c * syn_mem_size;
            let in_buf_offset = c * buf_stride;

            let src_start = syn_mem_size - frame_size - overlap;
            in_buf[in_buf_offset..in_buf_offset + buf_stride].copy_from_slice(
                &self.syn_mem[channel_offset + src_start..channel_offset + syn_mem_size],
            );
        }

        // C 2017: tone and transient analysis see the *unfiltered* history in
        // the overlap head (the `prefilter_mem` tail). `in_buf`'s head is the
        // previous frame's prefiltered overlap, so analysing it as-is puts a
        // step between head and body wherever the comb filter was active,
        // which reads as a transient on steady pitched input (issue #38).
        let max_period = COMBFILTER_MAXPERIOD;
        for c in 0..cc {
            in_buf[c * buf_stride..c * buf_stride + overlap].copy_from_slice(
                &self.prefilter_mem[(c + 1) * max_period - overlap..(c + 1) * max_period],
            );
        }

        // libopus nbCompressedBytes and effectiveBytes (celt_encoder.c:1859-1923):
        // at most 1275 bytes, under CBR with a rate the rate's bytes. Under
        // VBR effectiveBytes is the CELT share in bytes, also in unconstrained
        // hybrid mode, where the range coder still holds the caller's full
        // capacity; under CBR the bytes left in the packet.
        let hybrid = start_band != 0;
        let tell_initial = rc.tell();
        let nb_filled_bytes_initial = ((tell_initial + 4) >> 3).max(0) as usize;
        let capacity = explicit_total_bits.unwrap_or_else(|| (rc.storage * 8) as i32) as usize / 8;
        let mut nb_compressed_bytes = capacity.min(1275);
        let mut effective_bytes = if self.vbr && self.bitrate != OPUS_BITRATE_MAX {
            (bitrate_to_bits(self.bitrate, mode.fs, frame_size as i32) / 8).max(1) as usize
        } else {
            if self.bitrate != OPUS_BITRATE_MAX {
                let mut tmp = self.bitrate * frame_size as i32;
                if tell_initial > 1 {
                    tmp += tell_initial * mode.fs;
                }
                let rate_bytes = (tmp + 4 * mode.fs) / (8 * mode.fs);
                nb_compressed_bytes = (nb_compressed_bytes as i32).min(rate_bytes).max(2) as usize;
            }
            nb_compressed_bytes.saturating_sub(nb_filled_bytes_initial)
        }
        .max(1);
        if nb_compressed_bytes < capacity {
            rc.shrink(nb_compressed_bytes as u32);
        }
        // Low-rate hybrid frames code weak transients with long blocks.
        let allow_weak_transients =
            hybrid && effective_bytes < 15 && self.silk_signal_type != TYPE_VOICED;

        // Tone detection (C 2022): use channel-contiguous in_buf with N+overlap, float path
        let (tone_freq, mut toneishness) = tone_detect(&*in_buf, cc, buf_stride, mode.fs);
        dumpf!("pre", in_buf);
        dumpf!("tone_freq", &[tone_freq]);
        dumpf!("toneishness", &[toneishness]);
        let mut tf_estimate = 0.0f32;
        let mut tf_chan = 0;
        let mut weak_transient = false;

        let mut is_transient = if self.complexity >= 1 && !self.lfe {
            transient_analysis(
                in_buf,
                buf_stride,
                cc,
                &mut tf_estimate,
                &mut tf_chan,
                allow_weak_transients,
                &mut weak_transient,
                tone_freq,
                toneishness,
                &mut self.w_transient_tmp,
            )
        } else {
            false
        };
        if cc == 2 && channels == 1 {
            tf_chan = 0;
        }
        dumpi!("isTransient", &[is_transient as i32]);
        dumpf!("tf_estimate", &[tf_estimate]);
        dumpi!("tf_chan", &[tf_chan as i32]);
        // Put the prefiltered overlap back for the prefilter and the MDCT
        // (C run_prefilter: `OPUS_COPY(in, st->in_mem, overlap)`).
        let in_mem_start = syn_mem_size - frame_size - overlap;
        for c in 0..cc {
            let src = c * syn_mem_size + in_mem_start;
            in_buf[c * buf_stride..c * buf_stride + overlap]
                .copy_from_slice(&self.syn_mem[src..src + overlap]);
        }
        // Clamp toneishness as in C 2034: toneishness = min(toneishness, 1 - tf_estimate)
        // For float, tf_estimate is log-domain; clamp to [0,1]
        toneishness = toneishness.min((1.0 - tf_estimate).clamp(0.0, 1.0));

        let mut total_bits = (nb_compressed_bytes * 8) as i32;
        let threshold = 1.0 / ((1 << self.lsb_depth) as f32);
        let mut silence = sample_max <= threshold;
        let tell_initial_frac = rc.tell_frac();
        let nb_compressed_initial = nb_compressed_bytes;
        if tell_initial == 1 {
            rc.encode_bit_logp(silence, 15);
        } else {
            silence = false;
        }
        if silence {
            if self.vbr && self.bitrate != OPUS_BITRATE_MAX {
                let target_bytes = nb_filled_bytes_initial + 2;
                if target_bytes < nb_compressed_bytes {
                    nb_compressed_bytes = target_bytes;
                    total_bits = (nb_compressed_bytes * 8) as i32;
                    rc.shrink(nb_compressed_bytes as u32);
                }
            }
            let cur_tell = rc.tell();
            let new_tell = (nb_compressed_bytes * 8) as i32;
            rc.nbits_total += new_tell - cur_tell;
        }
        // General VBR max bound (libopus 1936-1961) - constrained VBR
        if self.vbr && self.bitrate != OPUS_BITRATE_MAX {
            let vbr_rate = bitrate_to_bits(self.bitrate, mode.fs, frame_size as i32) << BITRES;
            let nb_available = nb_compressed_bytes.saturating_sub(nb_filled_bytes_initial);
            if self.constrained_vbr {
                let vbr_bound = vbr_rate;
                let max_allowed = core::cmp::min(
                    core::cmp::max(
                        if tell_initial == 1 { 2 } else { 0 },
                        (vbr_rate + vbr_bound - self.vbr_reservoir) >> (BITRES + 3),
                    ),
                    nb_available as i32,
                );
                if (max_allowed as usize) < nb_available {
                    nb_compressed_bytes = nb_filled_bytes_initial + max_allowed as usize;
                    total_bits = (nb_compressed_bytes * 8) as i32;
                    rc.shrink(nb_compressed_bytes as u32);
                }
            }
            // effectiveBytes would be vbr_rate>>(3+BITRES) for later lambda, but total_bits already reflects bound
        }

        // libopus nbAvailableBytes after the VBR bound, two in a silent VBR
        // frame (celt_encoder.c:1924-1993); effectiveBytes follows the
        // silent frame's budget too.
        let silent_vbr = silence && self.vbr && self.bitrate != OPUS_BITRATE_MAX;
        let nb_available = if silent_vbr {
            2
        } else {
            nb_compressed_bytes.saturating_sub(nb_filled_bytes_initial)
        };
        if silent_vbr {
            effective_bytes = nb_compressed_bytes;
        }
        // The pre-filter runs every frame; `enabled` as celt_encoder.c:2038.
        let tell = if silence { total_bits } else { rc.tell() };
        let enabled = ((self.lfe && nb_available > 3) || nb_available > 12 * channels)
            && start_band == 0
            && !silence
            && tell + 16 <= total_bits
            && !self.disable_pf;
        let prefilter_tapset = self.tapset_decision;
        let (pf_on, gain1, pitch_index) = run_prefilter(
            in_buf,
            &mut self.prefilter_mem,
            self.prefilter_period,
            self.prefilter_gain,
            self.prefilter_tapset,
            self.tapset_decision,
            mode.window,
            cc,
            frame_size,
            overlap,
            &mut self.w_prefilter_pre,
            &mut self.w_prefilter_pitch_buf,
            &mut self.w_prefilter_before,
            &mut self.w_prefilter_after,
            &self.analysis,
            self.loss_rate,
            tf_estimate,
            enabled,
            nb_available,
            tone_freq,
            toneishness,
            self.complexity,
            self.arch,
        );

        dumpi!("pf_on", &[pf_on as i32]);
        dumpi!("pitch_index", &[pitch_index as i32]);
        dumpf!("gain1", &[gain1]);
        dumpf!("inpf", in_buf);
        // run_prefilter raises the previous period to the shortest one
        // (celt_encoder.c:1545), which the pitch-change test then reads.
        self.prefilter_period = self.prefilter_period.max(COMBFILTER_MINPERIOD);
        // A change of pitch (celt_encoder.c:2042-2044), which the VBR target
        // pays for in tonal frames.
        let pitch_change = (gain1 > 0.4 || self.prefilter_gain > 0.4)
            && (!self.analysis.valid || f64::from(self.analysis.tonality) > 0.3)
            && (pitch_index as f64 > 1.26 * self.prefilter_period as f64
                || (pitch_index as f64) < 0.79 * self.prefilter_period as f64);

        // Save the prefiltered overlap for the next frame.
        // In libopus, st->in_mem stores the overlap separately and run_prefilter
        // copies it to/from in[]. Here we emulate that by updating syn_mem with
        // the last overlap samples of in_buf (which were prefiltered in place).
        let syn_mem_size = 2048 + overlap;
        for c in 0..cc {
            let channel_offset = c * syn_mem_size;
            let in_buf_offset = c * buf_stride;
            self.syn_mem[channel_offset + syn_mem_size - overlap..channel_offset + syn_mem_size]
                .copy_from_slice(&in_buf[in_buf_offset + frame_size..in_buf_offset + buf_stride]);
        }
        let freq = &mut self.w_freq[..frame_size * cc];
        let band_e = &mut self.w_band_e[..nb_ebands * channels];
        let band_log_e = &mut self.w_band_log_e[..nb_ebands * channels];
        let x_pad_end = (frame_size * channels + STRIDE_ACCESS_PAD).min(self.w_x.len());
        let x = &mut self.w_x[..x_pad_end];

        self.w_error[..nb_ebands * channels].fill(0.0);
        let error = &mut self.w_error[..nb_ebands * channels];

        if start_band == 0 && !silence && rc.tell() + 16 <= total_bits {
            rc.encode_bit_logp(pf_on, 1);
            if pf_on {
                let qg = (gain1 / 0.09375 - 1.0 + 0.5).floor() as i32;
                let qg = qg.clamp(0, 7);
                let pi = (pitch_index + 1) as u32;
                let octave = 32 - pi.leading_zeros();
                let octave = (octave as i32 - 5).max(0) as u32;
                rc.enc_uint(octave, 6);
                rc.enc_bits(pi - (16 << octave), 4 + octave);
                rc.enc_bits(qg as u32, 3);
                rc.encode_icdf(self.tapset_decision, &TAPSET_ICDF, 2);
            }
        }

        // The transient flag is coded only with room for it (not in 2.5 ms
        // frames); otherwise the decoder takes the frame as not transient,
        // and so does the encoder.
        let transient_flag_fits = lm > 0 && rc.tell() + 3 <= total_bits;
        let mut transient_got_disabled = false;
        if !transient_flag_fits {
            is_transient = false;
            transient_got_disabled = true;
        }
        let mut short_blocks = is_transient;
        let second_mdct = short_blocks && self.complexity >= 8;
        let mut have_band_log_e2 = second_mdct;
        if second_mdct {
            compute_mdcts(
                mode, false, in_buf, freq, channels, cc, frame_size, overlap, lm, upsample,
            );
            let band_e2 = &mut self.w_band_e2[..nb_ebands * channels];
            compute_band_energies(mode, freq, band_e2, end, channels, lm, self.arch);
            let band_log_e2 = &mut self.w_band_log_e2[..nb_ebands * channels];
            crate::bands::amp2log2(mode, eff_end, end, band_e2, band_log_e2, channels);
            for c in 0..channels {
                for i in 0..end {
                    band_log_e2[c * nb_ebands + i] += lm as f32 * 0.5;
                }
            }
        }
        compute_mdcts(
            mode,
            short_blocks,
            in_buf,
            freq,
            channels,
            cc,
            frame_size,
            overlap,
            lm,
            upsample,
        );
        compute_band_energies(mode, freq, band_e, end, channels, lm, self.arch);
        // An LFE stream keeps the energy of its lowest two bands
        // (celt_encoder.c:2099-2106).
        if self.lfe {
            for i in 2..end {
                band_e[i] = band_e[i].min(1e-4 * band_e[0]).max(1e-15);
            }
        }
        crate::bands::amp2log2(mode, eff_end, end, band_e, band_log_e, channels);

        // The masking between surround channels (celt_encoder.c:2108-2186).
        let mut surround_dynalloc = [0.0f32; CELT_NB_EBANDS];
        let (mut surround_trim, mut surround_masking) = (0.0f32, 0.0f32);
        if let (Some(mask), false, false) = (&self.energy_mask, hybrid, self.lfe) {
            let e_bands = mode.e_bands;
            let mask_end = 2.max(self.last_coded_bands as usize);
            let (mut mask_avg, mut diff, mut count) = (0.0f32, 0.0f32, 0i32);
            for c in 0..channels {
                for i in 0..mask_end {
                    let mut m = mask[nb_ebands * c + i].min(0.25).max(-2.0);
                    if m > 0.0 {
                        m *= 0.5;
                    }
                    let width = (e_bands[i + 1] - e_bands[i]) as i32;
                    mask_avg += m * width as f32;
                    count += width;
                    diff += m * (1 + 2 * i as i32 - mask_end as i32) as f32;
                }
            }
            mask_avg /= count as f32;
            mask_avg += 0.2;
            diff = diff * 6.0
                / (channels as i32 * (mask_end as i32 - 1) * (mask_end as i32 + 1) * mask_end as i32) as f32;
            diff *= 0.5;
            diff = diff.min(0.031).max(-0.031);
            let mut midband = 0;
            while (e_bands[midband + 1] as i32) < e_bands[mask_end] as i32 / 2 {
                midband += 1;
            }
            let mut count_dynalloc = 0;
            for i in 0..mask_end {
                let lin = mask_avg + diff * (i as i32 - midband as i32) as f32;
                let mut unmask = if channels == 2 {
                    mask[i].max(mask[nb_ebands + i])
                } else {
                    mask[i]
                };
                unmask = unmask.min(0.0);
                unmask -= lin;
                if unmask > 0.25 {
                    surround_dynalloc[i] = unmask - 0.25;
                    count_dynalloc += 1;
                }
            }
            if count_dynalloc >= 3 {
                mask_avg += 0.25;
                if mask_avg > 0.0 {
                    mask_avg = 0.0;
                    diff = 0.0;
                    surround_dynalloc[..mask_end].fill(0.0);
                } else {
                    for v in &mut surround_dynalloc[..mask_end] {
                        *v = (*v - 0.25).max(0.0);
                    }
                }
            }
            mask_avg += 0.2;
            surround_trim = 64.0 * diff;
            surround_masking = mask_avg;
        }

        // Temporal VBR (celt_encoder.c:2187-2203): how far this frame's
        // spectral level is above its running average; not for LFE.
        let temporal_vbr = if self.lfe {
            0.0
        } else {
            let mut follow = -10.0f32;
            let mut frame_avg = 0.0f32;
            let offset = if short_blocks { 0.5 * lm as f32 } else { 0.0 };
            for i in start_band..end {
                follow = (follow - 1.0).max(band_log_e[i] - offset);
                if channels == 2 {
                    follow = follow.max(band_log_e[i + nb_ebands] - offset);
                }
                frame_avg += follow;
            }
            frame_avg /= (end - start_band) as f32;
            let temporal_vbr = (frame_avg - self.spec_avg).clamp(-1.5, 3.0);
            self.spec_avg += 0.02 * temporal_vbr;
            temporal_vbr
        };

        // The last chance to catch a transient the time-domain analysis
        // missed: a sudden rise of the band energies (celt_encoder.c:2213-2235).
        if transient_flag_fits
            && !is_transient
            && self.complexity >= 5
            && !self.lfe
            && !hybrid
            && patch_transient_decision(
                band_log_e,
                &self.old_band_e,
                nb_ebands,
                start_band,
                end,
                channels,
            )
        {
            is_transient = true;
            short_blocks = true;
            // The long-block energies, compensated for the scaling of short
            // and long MDCTs, steer the dynalloc.
            let band_log_e2 = &mut self.w_band_log_e2[..nb_ebands * channels];
            band_log_e2.copy_from_slice(band_log_e);
            for c in 0..channels {
                for i in 0..end {
                    band_log_e2[c * nb_ebands + i] += lm as f32 * 0.5;
                }
            }
            have_band_log_e2 = true;
            compute_mdcts(
                mode, true, in_buf, freq, channels, cc, frame_size, overlap, lm, upsample,
            );
            compute_band_energies(mode, freq, band_e, end, channels, lm, self.arch);
            crate::bands::amp2log2(mode, eff_end, end, band_e, band_log_e, channels);
            tf_estimate = 0.2;
        }
        if transient_flag_fits {
            rc.encode_bit_logp(is_transient, 3);
        }
        normalise_bands(mode, freq, x, band_e, end, channels, (1 << lm) as usize);

        dumpf!("freq", freq);
        dumpf!("bandE", band_e);
        dumpf!("bandLogE", band_log_e);
        dumpf!("X", &x[..frame_size * channels]);

        self.w_tf_res[..nb_ebands].fill(0);
        let tf_res = &mut self.w_tf_res[..nb_ebands];
        let lambda = 80.max(20480 / effective_bytes + 2) as i32;

        // C runs dynalloc_analysis before tf_analysis: it produces `importance`
        // (used by the tf Viterbi) and `offsets` (encoded later, after spread).
        self.w_cap[..nb_ebands].fill(0);
        let cap = &mut self.w_cap[..nb_ebands];
        for (i, cap_i) in cap.iter_mut().enumerate() {
            let n = (mode.e_bands[i + 1] - mode.e_bands[i]) << lm;
            *cap_i = ((mode.cache.caps[nb_ebands * (2 * lm + channels - 1) + i] as i32 + 64)
                * channels as i32
                * n as i32)
                >> 2;
        }

        self.w_offsets[..nb_ebands].fill(0);
        let offsets = &mut self.w_offsets[..nb_ebands];

        let band_log_e2_opt = if have_band_log_e2 {
            Some(&self.w_band_log_e2[..nb_ebands * channels] as &[f32])
        } else {
            None
        };
        let mut importance = [13i32; MAX_NB_EBANDS];
        let mut spread_weight = [32i32; MAX_NB_EBANDS];
        // The deepest band above the noise floor and the boosts' total, for
        // the VBR target.
        let (max_depth, tot_boost) = dynalloc_analysis_simple(
            mode,
            band_log_e,
            band_log_e2_opt,
            &self.old_band_e,
            start_band,
            end,
            channels,
            lm,
            effective_bytes,
            is_transient,
            offsets,
            self.lsb_depth,
            self.vbr,
            self.constrained_vbr,
            &self.analysis,
            self.lfe,
            &surround_dynalloc,
            tone_freq,
            toneishness,
            &mut importance,
            &mut spread_weight,
        );

        dumpi!("offsets", &offsets[..end]);
        dumpi!("importance", &importance[..end]);
        dumpi!("spread_weight", &spread_weight[..end]);
        // TF resolution (celt_encoder.c:2242-2277): analysed only for CELT
        // frames with enough bytes; low-rate hybrid frames keep 5 ms
        // resolution, weak transients use the long window. Strongly tonal
        // frames skip the analysis and keep `is_transient` (C 2242's
        // `toneishness < QCONST32(.98f, 29)`; float build, post-clamp
        // toneishness from C 2034).
        let enable_tf_analysis = effective_bytes >= 15 * channels
            && !hybrid
            && self.complexity >= 2
            && !self.lfe
            && toneishness < 0.98;
        let tf_select = if enable_tf_analysis {
            tf_analysis(
                mode,
                end,
                is_transient,
                tf_res,
                lambda,
                x,
                frame_size,
                lm as i32,
                tf_estimate,
                tf_chan,
                &importance,
            )
        } else if hybrid && weak_transient {
            tf_res[..end].fill(1);
            0
        } else if hybrid && effective_bytes < 15 && self.silk_signal_type != TYPE_VOICED {
            tf_res[..end].fill(0);
            is_transient as i32
        } else {
            tf_res[..end].fill(is_transient as i32);
            0
        };
        dumpi!("tf_res", &tf_res[..end]);
        dumpi!("tf_select", &[tf_select]);
        #[cfg(test)]
        let tf_raw = {
            let mut raw = [0i32; MAX_NB_EBANDS];
            raw[..end].copy_from_slice(&tf_res[..end]);
            raw
        };
        // When the energy is stable, bias the coarse energy towards the
        // previous error (celt_encoder.c:2283-2294).
        for c in 0..channels {
            for i in start_band..end {
                let k = i + c * nb_ebands;
                if (band_log_e[k] - self.old_band_e[k]).abs() < 2.0 {
                    band_log_e[k] -= 0.25 * self.energy_error[k];
                }
            }
        }
        dumpf!("bandLogE_q", band_log_e);
        // Coarse energy after the dynalloc and tf analyses, which see the
        // previous frame's energies (celt_encoder.c:2248-2297).
        // C uses st->force_intra (CTL flag, 0 by default) instead of the old
        // intra_ener heuristic, so no intra forcing happens here.
        quant_coarse_energy_advanced(
            mode,
            start_band,
            end,
            end,
            band_log_e,
            &mut self.old_band_e,
            total_bits as u32,
            error,
            rc,
            channels,
            lm,
            nb_available,
            self.force_intra,
            &mut self.delayed_intra,
            self.complexity >= 4,
            self.loss_rate,
            self.lfe,
        );
        dumpf!("error", error);
        dumpf!("oldBandE", &self.old_band_e[..nb_ebands * channels]);
        tf_encode(
            start_band,
            end,
            is_transient,
            tf_res,
            lm as i32,
            tf_select,
            rc,
        );

        // 2.5 ms frames always use M/S (celt_encoder.c:2399-2401).
        let mut dual_stereo_val = if channels == 2 && lm != 0 {
            stereo_analysis(mode, x, lm as i32, frame_size) as i32
        } else {
            0
        };

        // libopus equiv_rate (celt_encoder.c:1925-1927): the packet's rate,
        // capped by the target bitrate, less the overhead of short frames.
        let overhead = (40 * channels as i32 + 20) * ((400 >> lm) - 50);
        let mut equiv_rate = ((nb_compressed_initial as i32 * 8 * 50) << (3 - lm)) - overhead;
        if self.bitrate != OPUS_BITRATE_MAX {
            equiv_rate = equiv_rate.min(self.bitrate - overhead);
        }
        if channels == 2 {
            self.intensity = hysteresis_decision(
                equiv_rate / 1000,
                &INTEN_THRESHOLDS,
                &INTEN_HYSTERESIS,
                self.intensity,
            );
            // libopus clamps to the coded bands: the band is coded relative
            // to the start band, 17 in a Hybrid frame.
            self.intensity = self.intensity.clamp(start_band as i32, end as i32);
        }

        // Without room for the spread symbol the decoder spreads normally,
        // and so does the encoder (celt_encoder.c:2346-2348).
        if self.lfe && rc.tell() + 4 <= total_bits {
            // celt_encoder.c:2304-2307.
            self.tapset_decision = 0;
            self.spread_decision = SPREAD_NORMAL;
            rc.encode_icdf(self.spread_decision, &SPREAD_ICDF, 5);
        } else if self.complexity == 0 && rc.tell() + 4 <= total_bits {
            self.spread_decision = SPREAD_NONE;
            rc.encode_icdf(self.spread_decision, &SPREAD_ICDF, 5);
        } else if rc.tell() + 4 <= total_bits {
            if hybrid {
                // celt_encoder.c:2308-2314: hybrid frames spread aggressively
                // unless transient.
                self.spread_decision = if is_transient {
                    SPREAD_NORMAL
                } else {
                    SPREAD_AGGRESSIVE
                };
            } else if is_transient || self.complexity < 3 || nb_available < 10 * channels {
                self.spread_decision = SPREAD_NORMAL;
            } else {
                // celt_encoder.c:2338-2340.
                let update_hf = pf_on && !is_transient;
                self.spread_decision = spreading_decision(
                    mode,
                    x,
                    &mut self.tonal_average,
                    self.spread_decision,
                    &mut self.hf_average,
                    &mut self.tapset_decision,
                    update_hf,
                    eff_end,
                    channels,
                    (1 << lm) as usize,
                    &spread_weight,
                );
            }
            rc.encode_icdf(self.spread_decision, &SPREAD_ICDF, 5);
        } else {
            self.spread_decision = SPREAD_NORMAL;
        }

        dumpi!("spread", &[self.spread_decision]);
        // For LFE everything interesting is in the first band
        // (celt_encoder.c:2351-2353).
        if self.lfe {
            offsets[0] = 8.min(effective_bytes as i32 / 3);
        }
        let mut dynalloc_logp = 6i32;
        let total_bits_bitres = total_bits << BITRES;
        let mut total_boost = 0i32;
        let mut tell_frac = rc.tell_frac();

        for i in start_band..end {
            let width =
                channels as i32 * (mode.e_bands[i + 1] - mode.e_bands[i]) as i32 * (1 << lm);
            let quanta = (width << BITRES).min((6 << BITRES).max(width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            let mut j = 0i32;

            while tell_frac + (dynalloc_loop_logp << BITRES) < total_bits_bitres - total_boost
                && boost < cap[i]
            {
                let flag = j < offsets[i];
                rc.encode_bit_logp(flag, dynalloc_loop_logp as u32);
                tell_frac = rc.tell_frac();
                if !flag {
                    break;
                }
                boost += quanta;
                total_boost += quanta;
                dynalloc_loop_logp = 1;
                j += 1;
            }

            if j > 0 {
                dynalloc_logp = 2.max(dynalloc_logp - 1);
            }
            offsets[i] = boost;
        }

        // C: the trim is only analysed and coded when it fits after the
        // boosts; otherwise both sides allocate with the default of 5
        // (celt_encoder.c:2397-2410).
        let mut alloc_trim = 5;
        if rc.tell_frac() + (6 << BITRES) <= total_bits_bitres - total_boost {
            // celt_encoder.c:2410-2413: no trim analysis without the low
            // bands or for LFE.
            if hybrid || self.lfe {
                self.stereo_saving = 0.0;
            } else {
                alloc_trim = alloc_trim_analysis(
                    mode,
                    x,
                    band_log_e,
                    end,
                    lm as i32,
                    channels,
                    frame_size,
                    &self.analysis,
                    &mut self.stereo_saving,
                    tf_estimate,
                    self.intensity,
                    surround_trim,
                    equiv_rate,
                    self.arch,
                );
            }
            rc.encode_icdf(alloc_trim, &TRIM_ICDF, 7);
        }
        dumpi!("alloc_trim", &[alloc_trim]);
        dumpi!("boosts", &offsets[..end]);
        dumpi!("intensity", &[self.intensity]);
        #[cfg(test)]
        {
            self.decisions = FrameDecisions {
                alloc_trim,
                tf_select,
                tf_res: tf_raw,
                is_transient,
                weak_transient,
            };
        }
        // Second VBR: compute target via simplified compute_vbr (libopus 2436-2534)
        // This makes packet size variable for VBR (beyond silence) while keeping reservoir in sync
        if self.vbr && self.bitrate != OPUS_BITRATE_MAX {
            let vbr_rate = bitrate_to_bits(self.bitrate, mode.fs, frame_size as i32) << BITRES;
            let lm_diff = mode.max_lm as i32 - lm as i32;
            let mut base_target = if !hybrid {
                vbr_rate - ((40 * channels as i32 + 20) << BITRES)
            } else {
                (0).max(vbr_rate - ((9 * channels as i32 + 4) << BITRES))
            };
            if self.constrained_vbr {
                base_target += self.vbr_offset >> lm_diff;
            }
            let cur_tell_frac = rc.tell_frac();
            // Use the exact initial tell_frac captured at entry (libopus tell0_frac),
            // not tell_initial<<BITRES which loses the fractional part.
            let min_allowed = {
                let a =
                    ((cur_tell_frac + total_boost + (1 << (BITRES + 3)) - 1) >> (BITRES + 3)) + 2;
                if hybrid {
                    let b =
                        (tell_initial_frac + (37 << BITRES) + total_boost + (1 << (BITRES + 3))
                            - 1)
                            >> (BITRES + 3);
                    a.max(b)
                } else {
                    a
                }
            };
            // nbCompressedBytes is current max (after first VBR bound), effectiveBytes ~ vbr_rate>>(3+BITRES) for lambda already
            dumpi!(
                "vbr_in",
                &[
                    base_target,
                    lm as i32,
                    equiv_rate,
                    self.last_coded_bands,
                    channels as i32,
                    self.intensity,
                    self.constrained_vbr as i32,
                    tot_boost,
                    pitch_change as i32
                ]
            );
            dumpf!(
                "vbr_inf",
                &[self.stereo_saving, tf_estimate, max_depth, temporal_vbr]
            );
            let mut target = if !hybrid {
                compute_vbr(
                    mode,
                    &self.analysis,
                    base_target,
                    lm as i32,
                    equiv_rate,
                    self.last_coded_bands,
                    channels as i32,
                    self.intensity,
                    self.constrained_vbr,
                    self.stereo_saving,
                    tot_boost,
                    tf_estimate,
                    i32::from(pitch_change),
                    max_depth,
                    self.lfe,
                    self.energy_mask.is_some(),
                    surround_masking,
                    temporal_vbr,
                )
            } else {
                let mut t = base_target;
                // Tonal frames (offset < 100) need more bits than noisy ones
                // (celt_encoder.c:2466-2467).
                if self.silk_offset < 100 {
                    t += (12 << BITRES) >> (3 - lm);
                }
                if self.silk_offset > 100 {
                    t -= (18 << BITRES) >> (3 - lm);
                }
                t += ((tf_estimate - 0.25) * 50.0 * (1 << BITRES) as f32) as i32;
                if tf_estimate > 0.7 {
                    t = t.max(50 << BITRES);
                }
                t
            };
            // libopus: "Don't attempt to use more than 510 kb/s, even for
            // frames smaller than 20 ms." It bounds unconstrained VBR.
            nb_compressed_bytes = nb_compressed_bytes.min(1275 >> (3 - lm));
            target += cur_tell_frac;
            // Signed, like libopus: a negative target gives min_allowed
            // (celt_encoder.c:2479-2481).
            let mut nb_available = ((target + (1 << (BITRES + 2))) >> (BITRES + 3))
                .max(min_allowed)
                .min(nb_compressed_bytes as i32) as usize;
            let mut delta = target - vbr_rate;
            target = (nb_available as i32) << (BITRES + 3);
            if silence {
                nb_available = 2;
                target = (2 * 8) << BITRES;
                delta = 0;
            }
            // Reservoir / drift update (libopus 2502-2529)
            // The float build's alpha: celt_rcp is 1/x and QCONST16 the plain
            // constant (celt_encoder.c:2501-2506).
            let alpha = if self.vbr_count < 970 {
                self.vbr_count += 1;
                1.0 / (self.vbr_count + 20) as f32
            } else {
                0.001
            };
            if self.constrained_vbr {
                self.vbr_reservoir += target - vbr_rate;
                // drift: MULT16_32_Q15(alpha, delta*(1<<lm_diff) - offset - drift) (libopus 2516)
                let delta_scaled = delta * (1 << lm_diff);
                let delta_minus = delta_scaled - self.vbr_offset - self.vbr_drift;
                self.vbr_drift += (alpha * delta_minus as f32) as i32;
                self.vbr_offset = -self.vbr_drift;
            }
            if self.constrained_vbr && self.vbr_reservoir < 0 {
                let adjust = (-self.vbr_reservoir) / (8 << BITRES);
                if !silence {
                    nb_available = (nb_available as i32 + adjust) as usize;
                }
                self.vbr_reservoir = 0;
            }
            nb_compressed_bytes = nb_compressed_bytes.min(nb_available);
            total_bits = (nb_compressed_bytes * 8) as i32;
            rc.shrink(nb_compressed_bytes as u32);
        }

        dumpi!("nbCompressedBytes", &[nb_compressed_bytes as i32]);
        self.w_pulses[..nb_ebands].fill(0);
        let pulses = &mut self.w_pulses[..nb_ebands];

        let stereo = channels > 1;
        let ebands_stereo = if stereo {
            nb_ebands * channels
        } else {
            nb_ebands
        };
        self.w_fine_priority[..ebands_stereo].fill(0);
        let fine_priority = &mut self.w_fine_priority[..ebands_stereo];
        self.w_ebits[..ebands_stereo].fill(0);
        let ebits = &mut self.w_ebits[..ebands_stereo];
        let mut balance = 0;

        // C: bits = packet_bits - tell - 1; then subtract anti_collapse_rsv
        // (celt_encoder.c:2614-2618) BEFORE the allocation.
        let mut alloc_bits = (total_bits << BITRES) - rc.tell_frac() - 1;
        let anti_collapse_rsv = if is_transient && lm >= 2 {
            if alloc_bits >= ((lm as i32 + 2) << BITRES) {
                1i32 << BITRES
            } else {
                0
            }
        } else {
            0
        };
        alloc_bits -= anti_collapse_rsv;
        // The highest band with signal (celt_encoder.c:2606-2625), by the
        // analysis where it is valid, but not below what the rate supports.
        let mut signal_bandwidth = end as i32 - 1;
        if self.analysis.valid {
            let c = channels as i32;
            let min_bandwidth = if equiv_rate < 32000 * c {
                13
            } else if equiv_rate < 48000 * c {
                16
            } else if equiv_rate < 60000 * c {
                18
            } else if equiv_rate < 80000 * c {
                19
            } else {
                20
            };
            signal_bandwidth = self.analysis.bandwidth.max(min_bandwidth);
        }
        if self.lfe {
            signal_bandwidth = 1;
        }

        let coded_bands = clt_compute_allocation(
            mode,
            start_band,
            end,
            offsets,
            cap,
            alloc_trim,
            &mut self.intensity,
            &mut dual_stereo_val,
            alloc_bits,
            &mut balance,
            pulses,
            ebits,
            fine_priority,
            channels as i32,
            lm as i32,
            rc,
            true,
            self.last_coded_bands,
            signal_bandwidth,
        );
        // libopus keeps a slowly moving count of the coded bands for the
        // allocator's hysteresis (celt_encoder.c:2629-2632).
        self.last_coded_bands = if self.last_coded_bands != 0 {
            (self.last_coded_bands + 1).min((self.last_coded_bands - 1).max(coded_bands))
        } else {
            coded_bands
        };

        dumpi!("pulses", &pulses[..end]);
        dumpi!("fine_quant", &ebits[..end]);
        dumpi!("fine_priority", &fine_priority[..end]);
        dumpi!("codedBands", &[coded_bands]);
        dumpi!("balance", &[balance]);
        quant_fine_energy(
            mode,
            start_band,
            end,
            &mut self.old_band_e,
            error,
            ebits,
            rc,
            channels,
        );
        self.energy_error = [0.0; CELT_NB_X_CH];

        self.w_collapse_masks[..nb_ebands * channels].fill(0);
        let collapse_masks = &mut self.w_collapse_masks[..nb_ebands * channels];
        let (x_split, y_split) = x.split_at_mut(frame_size);
        let y_opt = if channels == 2 { Some(y_split) } else { None };

        let mut dual_stereo = dual_stereo_val != 0;

        let theta_rdo = channels == 2 && !dual_stereo && self.complexity >= 8;
        let resynth = theta_rdo;

        quant_all_bands(
            true,
            mode,
            start_band,
            end,
            x_split,
            y_opt,
            collapse_masks,
            band_e,
            pulses,
            short_blocks,
            self.spread_decision,
            &mut dual_stereo,
            self.intensity as usize,
            tf_res,
            (total_bits << BITRES) - anti_collapse_rsv,
            &mut balance,
            rc,
            lm as i32,
            coded_bands,
            resynth,
            self.disable_inv,
            &mut self.rng,
            self.arch,
        );

        dumpf!("Xq", &x[..frame_size * channels]);
        if anti_collapse_rsv > 0 {
            let anti_collapse_on = if self.consec_transient < 2 {
                1u32
            } else {
                0u32
            };
            rc.enc_bits(anti_collapse_on, 1);
        }

        dumpf!("error_fine", error);
        dumpi!("bits_left", &[total_bits - rc.tell()]);
        quant_energy_finalise(
            mode,
            start_band,
            end,
            &mut self.old_band_e,
            error,
            ebits,
            fine_priority,
            total_bits - rc.tell(),
            rc,
            channels,
        );
        for c in 0..channels {
            for i in start_band..end {
                let k = i + c * nb_ebands;
                self.energy_error[k] = error[k].clamp(-0.5, 0.5);
            }
        }

        if silence {
            for v in self.old_band_e[..channels * nb_ebands].iter_mut() {
                *v = -28.0;
            }
        }

        // A mono stream's energies stand for both input channels
        // (celt_encoder.c:2778-2780).
        if cc == 2 && channels == 1 {
            self.old_band_e.copy_within(..nb_ebands, nb_ebands);
        }
        self.last_band_log_e.copy_from_slice(&self.old_band_e);

        if !is_transient {
            self.old_band_e3.copy_from_slice(&self.old_band_e2);
            self.old_band_e2.copy_from_slice(&self.old_band_e);
        } else {
            for i in 0..cc * nb_ebands {
                self.old_band_e2[i] = self.old_band_e2[i].min(self.old_band_e[i]);
            }
        }

        // Bands outside start..end start from zero energy when they come
        // back (celt_encoder.c:2790-2804).
        for c in 0..cc {
            for i in (0..start_band).chain(end..nb_ebands) {
                let k = c * nb_ebands + i;
                self.old_band_e[k] = 0.0;
                self.old_band_e2[k] = -28.0;
                self.old_band_e3[k] = -28.0;
            }
        }
        self.rng = rc.rng;

        // libopus keeps the pitch found even when the pre-filter is off (its
        // gain is then 0), and the tapset the frame started with.
        self.prefilter_period = pitch_index;
        self.prefilter_gain = gain1;
        self.prefilter_tapset = prefilter_tapset;

        if is_transient || transient_got_disabled {
            self.consec_transient += 1;
        } else {
            self.consec_transient = 0;
        }
        Ok(())
    }
}

/// Where the CELT decoder writes its output: planar, or interleaved (libopus),
/// optionally added onto what `pcm` holds (`accum`).
#[derive(Clone, Copy)]
pub(crate) enum PcmLayout {
    Planar,
    Interleaved { accum: bool },
}

pub struct CeltDecoder {
    mode: &'static CeltMode,
    channels: usize,
    /// libopus `CELTDecoder.arch`, chosen at init.
    arch: crate::Arch,
    /// Channels coded in the stream (libopus `CELT_SET_CHANNELS`). It drives
    /// the bitstream; `channels` drives synthesis and the output.
    stream_channels: usize,
    /// Decimation factor for non-48kHz API rates (1=48k, 2=24k, 3=16k, 4=12k, 6=8k).
    /// Mirrors libopus `CELTDecoder.downsample`.
    downsample: usize,
    decode_mem: FixedVec<f32, CELT_DECODE_MEM>,
    old_band_e: FixedVec<f32, CELT_NB_X_CH>,
    preemph_mem: FixedVec<f32, CELT_MAX_CHANNELS>,
    prefilter_period: usize,
    prefilter_period_old: usize,
    prefilter_gain: f32,
    prefilter_gain_old: f32,
    prefilter_tapset: i32,
    prefilter_tapset_old: i32,
    old_band_e2: FixedVec<f32, CELT_NB_X_CH>,
    old_band_e3: FixedVec<f32, CELT_NB_X_CH>,
    rng: u32,
    /// Background noise energy per band (libopus `backgroundLogE`).
    background_log_e: FixedVec<f32, CELT_NB_X_CH>,
    /// Lost time since the last decoded frame, in 2.5 ms units.
    loss_duration: u32,
    /// Concealed time since the last decoded frame, in 2.5 ms units.
    plc_duration: u32,
    /// Conceal with noise until two frames in a row decode (`skip_plc`).
    skip_plc: bool,
    /// Kind of the last frame (`plc::FRAME_*`).
    last_frame_type: u8,
    /// Pitch period the periodic concealment repeats.
    last_pitch_index: usize,
    /// Decay of the periodic concealment per channel, measured on the
    /// first lost frame (`plc_decay`).
    plc_decay: [f32; CELT_MAX_CHANNELS],
    /// Whether the next frame must fold a concealed overlap first.
    prefilter_and_fold: bool,
    /// Concealment LPC filter per channel.
    plc_lpc: [f32; plc::LPC_ORDER * CELT_MAX_CHANNELS],
    /// First band decoded and concealed (libopus `CELT_SET_START_BAND`).
    start_band: usize,
    /// End band decoded and concealed (libopus `CELT_SET_END_BAND`).
    end_band: usize,
    /// No phase inversion in intensity stereo (libopus `disable_inv`,
    /// `OPUS_SET_PHASE_INVERSION_DISABLED`); set for one output channel.
    pub(crate) disable_inv: bool,
    /// libopus `postfilter_period` (`OPUS_GET_PITCH`): the postfilter
    /// period of the last decoded frame, 0 without the postfilter, at least
    /// `COMBFILTER_MINPERIOD` after a noise concealment.
    pub(crate) postfilter_pitch: usize,
    /// The deep PLC fields of libopus's decoder.
    #[cfg(feature = "deep-plc")]
    pub(crate) deep_plc: plc::deep_plc::DeepPlcMem,

    w_tf_res: FixedVec<i32, CELT_NB_EBANDS>,
    w_cap: FixedVec<i32, CELT_NB_EBANDS>,
    w_offsets: FixedVec<i32, CELT_NB_EBANDS>,
    w_pulses: FixedVec<i32, CELT_NB_EBANDS>,
    w_ebits: FixedVec<i32, CELT_NB_X_CH>,
    w_fine_priority: FixedVec<i32, CELT_NB_X_CH>,
    w_x: FixedVec<f32, CELT_W_X_DEC>,
    w_collapse_masks: FixedVec<u32, CELT_NB_X_CH>,
    w_freq: FixedVec<f32, CELT_W_FREQ_DEC>,
    w_band_amp: FixedVec<f32, CELT_NB_X_CH>,
}

impl CeltDecoder {
    /// The decoder's RNG state, which libopus reports as the final range
    /// after concealment.
    pub(crate) fn rng(&self) -> u32 {
        self.rng
    }

    /// Sets the end band (libopus `CELT_SET_END_BAND`).
    pub(crate) fn set_end_band(&mut self, end: usize) {
        self.end_band = end;
    }

    /// Sets the start band (libopus `CELT_SET_START_BAND`).
    pub(crate) fn set_start_band(&mut self, start: usize) {
        self.start_band = start;
    }

    /// libopus `celt_decode_with_ec`: decodes `data` with the start and end
    /// bands set, reading through `rc` when given (a hybrid frame's shared
    /// range coder), or conceals the frame when `data` is `None` or at most
    /// one byte. Output is interleaved; `accum` adds it onto `pcm`. Returns
    /// the samples per channel, or `None` for a frame size CELT lacks.
    pub(crate) fn decode_with_ec(
        &mut self,
        data: Option<&[u8]>,
        rc: Option<&mut RangeCoder>,
        pcm: &mut [f32],
        frame_size: usize,
        accum: bool,
    ) -> Option<usize> {
        self.decode_with_ec_dred(data, rc, pcm, frame_size, accum, None)
    }

    /// libopus `celt_decode_with_ec_dred`: [`Self::decode_with_ec`] with the
    /// deep PLC state of the Opus decoder, which concealment uses and a
    /// decoded frame resets (`blend`).
    pub(crate) fn decode_with_ec_dred(
        &mut self,
        data: Option<&[u8]>,
        rc: Option<&mut RangeCoder>,
        pcm: &mut [f32],
        frame_size: usize,
        accum: bool,
        lpcnet: crate::silk::plc::DeepPlc<'_>,
    ) -> Option<usize> {
        let mode = self.mode;
        let n = frame_size * self.downsample;
        (0..=mode.max_lm).find(|&lm| mode.short_mdct_size << lm == n)?;
        let layout = PcmLayout::Interleaved { accum };
        let r = match data {
            Some(d) if d.len() > 1 => {
                if d.len() > 1275 {
                    return None;
                }
                let mut own;
                let rc = match rc {
                    Some(rc) => rc,
                    None => {
                        own = RangeCoder::new_decoder(d);
                        &mut own
                    }
                };
                // A bit of a hack in libopus too: opus_decode_native learns
                // that the PLC has ended.
                #[cfg(feature = "deep-plc")]
                if let Some(l) = lpcnet {
                    l.blend = 0;
                }
                #[cfg(not(feature = "deep-plc"))]
                let _ = lpcnet;
                let (start, end) = (self.start_band, self.end_band);
                self.decode_impl_from_rc(
                    rc,
                    (d.len() * 8) as i32,
                    frame_size,
                    pcm,
                    start,
                    end,
                    layout,
                )
            }
            _ => self.decode_lost_into(frame_size, pcm, self.start_band, layout, lpcnet),
        };
        (r == frame_size).then_some(r)
    }

    /// Create a CELT decoder. `sampling_rate` is the API sampling rate
    /// (8000–48000); the decoder always operates at 48 kHz internally and
    /// decimates by `resampling_factor(sampling_rate)` on output.
    pub fn new(mode: &'static CeltMode, channels: usize, sampling_rate: i32) -> Self {
        let mut decoder = Self::EMPTY;
        decoder.init(mode, channels, sampling_rate);
        decoder
    }

    /// A decoder with empty buffers for [`Self::init`] to set up. It is a
    /// constant so that it can be written straight into its final memory.
    pub(crate) const EMPTY: Self = Self {
        mode: crate::modes::default_mode(),
        channels: 0,
        arch: crate::Arch::Scalar,
        stream_channels: 0,
        downsample: 1,
        decode_mem: FixedVec::new(),
        old_band_e: FixedVec::new(),
        preemph_mem: FixedVec::new(),
        prefilter_period: 0,
        prefilter_period_old: 0,
        prefilter_gain: 0.0,
        prefilter_gain_old: 0.0,
        prefilter_tapset: 0,
        prefilter_tapset_old: 0,
        old_band_e2: FixedVec::new(),
        old_band_e3: FixedVec::new(),
        rng: 0,
        background_log_e: FixedVec::new(),
        loss_duration: 0,
        plc_duration: 0,
        skip_plc: true,
        last_frame_type: plc::FRAME_NONE,
        last_pitch_index: 0,
        plc_decay: [0.0; CELT_MAX_CHANNELS],
        prefilter_and_fold: false,
        plc_lpc: [0.0; plc::LPC_ORDER * CELT_MAX_CHANNELS],
        start_band: 0,
        end_band: 0,
        disable_inv: false,
        postfilter_pitch: 0,
        #[cfg(feature = "deep-plc")]
        deep_plc: plc::deep_plc::DeepPlcMem::EMPTY,
        w_tf_res: FixedVec::new(),
        w_cap: FixedVec::new(),
        w_offsets: FixedVec::new(),
        w_pulses: FixedVec::new(),
        w_ebits: FixedVec::new(),
        w_fine_priority: FixedVec::new(),
        w_x: FixedVec::new(),
        w_collapse_masks: FixedVec::new(),
        w_freq: FixedVec::new(),
        w_band_amp: FixedVec::new(),
    };

    /// Sets the decoder up as [`Self::new`] does, in place.
    pub(crate) fn init(&mut self, mode: &'static CeltMode, channels: usize, sampling_rate: i32) {
        // All internal buffers are sized for at most 2 channels
        // (issue #27 deep scan).
        assert!(
            (1..=2).contains(&channels),
            "CeltDecoder::new: channels must be 1 or 2 (got {channels})"
        );
        let overlap = mode.overlap;
        let nb_ebands = mode.nb_ebands;
        // Band and spectrum buffers are sized for a stereo stream whatever
        // the output (libopus keeps `2*nbEBands` energies): a mono decoder
        // may be handed one. Capacity is fixed, so this costs no memory.
        let nb_x_ch = nb_ebands * CELT_MAX_CHANNELS;
        let dec_frame_x_ch = DECODE_BUFFER_SIZE * CELT_MAX_CHANNELS;
        // Field by field: a whole `Self` value would need a large stack frame.
        self.mode = mode;
        self.arch = crate::Arch::select();
        self.channels = channels;
        self.stream_channels = channels;
        self.downsample = resampling_factor(sampling_rate);
        self.decode_mem
            .fill_to(channels * (DECODE_BUFFER_SIZE + overlap), 0.0);
        self.old_band_e.fill_to(nb_x_ch, 0.0);
        self.preemph_mem.fill_to(channels, 0.0);
        self.prefilter_period = COMBFILTER_MINPERIOD;
        self.prefilter_period_old = COMBFILTER_MINPERIOD;
        self.prefilter_gain = 0.0;
        self.prefilter_gain_old = 0.0;
        self.prefilter_tapset = 0;
        self.prefilter_tapset_old = 0;
        // oldLogE and oldLogE2 start at -28 dB, as libopus's reset (run at init) sets them.
        self.old_band_e2.fill_to(nb_x_ch, -28.0);
        self.old_band_e3.fill_to(nb_x_ch, -28.0);
        self.rng = 0;
        self.background_log_e.fill_to(nb_x_ch, 0.0);
        self.loss_duration = 0;
        self.plc_duration = 0;
        self.skip_plc = true;
        self.last_frame_type = plc::FRAME_NONE;
        self.last_pitch_index = 0;
        self.plc_decay = [0.0; CELT_MAX_CHANNELS];
        self.prefilter_and_fold = false;
        self.plc_lpc = [0.0; plc::LPC_ORDER * CELT_MAX_CHANNELS];
        self.start_band = 0;
        self.end_band = mode.eff_ebands;
        self.disable_inv = channels == 1;
        self.postfilter_pitch = 0;
        self.w_tf_res.fill_to(nb_ebands, 0);
        self.w_cap.fill_to(nb_ebands, 0);
        self.w_offsets.fill_to(nb_ebands, 0);
        self.w_pulses.fill_to(nb_ebands, 0);
        self.w_ebits.fill_to(nb_x_ch, 0);
        self.w_fine_priority.fill_to(nb_x_ch, 0);
        self.w_x.fill_to(dec_frame_x_ch + STRIDE_ACCESS_PAD, 0.0);
        self.w_collapse_masks.fill_to(nb_x_ch, 0);
        self.w_freq.fill_to(dec_frame_x_ch + 4, 0.0); // +4: NEON backward pre-rotation reads up to 3 elements past n2
        self.w_band_amp.fill_to(nb_x_ch, 0.0);
    }

    pub fn decode(&mut self, compressed: &[u8], frame_size: usize, pcm: &mut [f32]) -> usize {
        self.decode_impl(compressed, frame_size, pcm, 0, self.mode.nb_ebands)
    }

    /// Set the number of channels coded in the stream (libopus
    /// `CELT_SET_CHANNELS`). A mono stream plays on both channels of a stereo
    /// decoder; a stereo stream is downmixed by a mono one.
    pub fn set_stream_channels(&mut self, channels: usize) {
        assert!(
            (1..=CELT_MAX_CHANNELS).contains(&channels),
            "CeltDecoder::set_stream_channels: channels must be 1 or 2 (got {channels})"
        );
        self.stream_channels = channels;
    }

    /// Reset the decoder state (libopus `OPUS_RESET_STATE`). The stream
    /// channels and the start and end bands are settings and stay.
    pub fn reset_state(&mut self) {
        self.decode_mem.fill(0.0);
        self.old_band_e.fill(0.0);
        self.preemph_mem.fill(0.0);
        self.prefilter_period = COMBFILTER_MINPERIOD;
        self.prefilter_period_old = COMBFILTER_MINPERIOD;
        self.prefilter_gain = 0.0;
        self.prefilter_gain_old = 0.0;
        self.prefilter_tapset = 0;
        self.prefilter_tapset_old = 0;
        // -28 dB, as libopus's OPUS_RESET_STATE.
        self.old_band_e2.fill(-28.0);
        self.old_band_e3.fill(-28.0);
        self.rng = 0;
        self.background_log_e.fill(0.0);
        self.loss_duration = 0;
        self.plc_duration = 0;
        self.skip_plc = true;
        self.last_frame_type = plc::FRAME_NONE;
        self.last_pitch_index = 0;
        self.plc_decay = [0.0; CELT_MAX_CHANNELS];
        self.prefilter_and_fold = false;
        self.plc_lpc = [0.0; plc::LPC_ORDER * CELT_MAX_CHANNELS];
        self.postfilter_pitch = 0;
        #[cfg(feature = "deep-plc")]
        self.deep_plc.reset();
    }

    /// The postfilter period of the last decoded frame, 0 without the
    /// postfilter (libopus `OPUS_GET_PITCH`).
    pub fn postfilter_pitch(&self) -> usize {
        self.postfilter_pitch
    }

    pub fn decode_with_start_band(
        &mut self,
        compressed: &[u8],
        frame_size: usize,
        pcm: &mut [f32],
        start_band: usize,
    ) -> usize {
        self.decode_impl(compressed, frame_size, pcm, start_band, self.mode.nb_ebands)
    }

    pub fn decode_from_range_coder(
        &mut self,
        rc: &mut RangeCoder,
        total_bits: i32,
        frame_size: usize,
        pcm: &mut [f32],
        start_band: usize,
    ) -> usize {
        self.decode_impl_from_rc(
            rc,
            total_bits,
            frame_size,
            pcm,
            start_band,
            self.mode.nb_ebands,
            PcmLayout::Planar,
        )
    }

    pub fn decode_from_range_coder_with_band_range(
        &mut self,
        rc: &mut RangeCoder,
        total_bits: i32,
        frame_size: usize,
        pcm: &mut [f32],
        start_band: usize,
        end_band: usize,
    ) -> usize {
        self.decode_impl_from_rc(
            rc,
            total_bits,
            frame_size,
            pcm,
            start_band,
            end_band,
            PcmLayout::Planar,
        )
    }

    /// libopus `deemphasis` of the last `n` (48 kHz) samples of each
    /// channel's decode memory into `pcm`, keeping every `downsample`-th.
    pub(crate) fn deemphasis(&mut self, n: usize, pcm: &mut [f32], layout: PcmLayout) {
        // libopus saturates at SIG_SAT here only in the fixed-point build;
        // the float build's SATURATE is the identity (celt/arch.h:323).
        const VERY_SMALL: f32 = 1e-30;
        let cc = self.channels;
        let ds = self.downsample;
        let nd = n / ds;
        let coef0 = self.mode.preemph[0];
        let mem_len = DECODE_BUFFER_SIZE + self.mode.overlap;
        for c in 0..cc {
            let x = &self.decode_mem[c * mem_len + DECODE_BUFFER_SIZE - n..][..n];
            let mut m = self.preemph_mem[c];
            let (base, stride, accum) = match layout {
                PcmLayout::Planar => (c * nd, 1, false),
                PcmLayout::Interleaved { accum } => (c, cc, accum),
            };
            if ds == 1 && accum {
                // libopus adds VERY_SMALL last on this path.
                for (j, &xj) in x.iter().enumerate() {
                    let tmp = xj + m + VERY_SMALL;
                    m = coef0 * tmp;
                    pcm[base + j * stride] += (1.0 / 32768.0) * tmp;
                }
            } else if ds == 1 {
                for (j, &xj) in x.iter().enumerate() {
                    let tmp = xj + VERY_SMALL + m;
                    m = coef0 * tmp;
                    pcm[base + j * stride] = (1.0 / 32768.0) * tmp;
                }
            } else {
                // Every ds-th sample is an output.
                for (j, xs) in x.chunks(ds).enumerate() {
                    for (k, &xj) in xs.iter().enumerate() {
                        let tmp = xj + VERY_SMALL + m;
                        m = coef0 * tmp;
                        if k == 0 {
                            let y = &mut pcm[base + j * stride];
                            if accum {
                                *y += (1.0 / 32768.0) * tmp;
                            } else {
                                *y = (1.0 / 32768.0) * tmp;
                            }
                        }
                    }
                }
            }
            self.preemph_mem[c] = m;
        }
    }

    fn decode_impl(
        &mut self,
        compressed: &[u8],
        frame_size: usize,
        pcm: &mut [f32],
        start_band: usize,
        end_band: usize,
    ) -> usize {
        let total_bits = (compressed.len() * 8) as i32;
        let mut rc = RangeCoder::new_decoder(compressed);
        self.decode_impl_from_rc(
            &mut rc,
            total_bits,
            frame_size,
            pcm,
            start_band,
            end_band,
            PcmLayout::Planar,
        )
    }

    fn decode_impl_from_rc(
        &mut self,
        rc: &mut RangeCoder,
        total_bits: i32,
        frame_size: usize,
        pcm: &mut [f32],
        start_band: usize,
        end_band: usize,
        layout: PcmLayout,
    ) -> usize {
        let mode = self.mode;
        // `channels` are coded in the stream and drive the bitstream (libopus
        // `C`); `out_channels` are synthesised (`CC`).
        let channels = self.stream_channels;
        let out_channels = self.channels;
        let nb_ebands = mode.nb_ebands;
        let end_band = end_band.min(nb_ebands).max(start_band);
        let overlap = mode.overlap;

        // The API frame_size is in output samples. Internally CELT always
        // decodes at 48 kHz, so upscale by the downsample factor (libopus
        // celt_decoder.c:1196: `frame_size *= st->downsample`).
        let api_frame_size = frame_size;
        let frame_size = frame_size * self.downsample;

        // The decode geometry requires frame_size == short_mdct_size << lm.
        // A non-matching size makes the lm search below fall back to lm=0,
        // which silently decodes garbage for large frames and indexes out of
        // bounds for tiny ones — reject early (issue #27 deep scan).
        let mut lm_matched = false;
        for cand in 0..=mode.max_lm {
            if mode.short_mdct_size << cand == frame_size {
                lm_matched = true;
                break;
            }
        }
        if !lm_matched || frame_size < mode.short_mdct_size {
            return 0;
        }

        // Reject an undersized output buffer before spending the whole decode
        // (it used to panic only at the final sample copy, issue #27 deep
        // scan).
        if pcm.len() < api_frame_size * out_channels {
            return 0;
        }

        let mut lm = 0;
        while (mode.short_mdct_size << lm) != frame_size {
            lm += 1;
            if lm > mode.max_lm {
                break;
            }
        }
        if (mode.short_mdct_size << lm) != frame_size {
            lm = 0;
        }

        // Pitch-based concealment only after two good frames in a row.
        if self.loss_duration == 0 {
            self.skip_plc = false;
        }

        // A mono stream predicts from the louder channel of a stereo frame
        // before it (celt_decoder.c).
        if channels == 1 {
            let (first, second) = self.old_band_e.split_at_mut(nb_ebands);
            for (e, &e2) in first.iter_mut().zip(&second[..nb_ebands]) {
                *e = e.max(e2);
            }
        }

        let mut tell = rc.tell();
        let mut silence = false;
        if tell >= total_bits {
            silence = true;
        } else if tell == 1 {
            silence = rc.decode_bit_logp(15);
        }
        if silence {
            // Pretend we've read all remaining bits (libopus celt_decoder.c:1324-1328)
            // so that every subsequent `tell+...<=total_bits` guard correctly
            // sees no budget left and decoding proceeds with defaults.
            tell = total_bits;
            rc.nbits_total += tell - rc.tell();
        }

        let mut pf_on = false;
        let mut pitch_index = COMBFILTER_MINPERIOD;
        let mut gain1 = 0.0f32;
        let mut prefilter_tapset = 0;

        if start_band == 0 && !silence && rc.tell() + 16 <= total_bits {
            pf_on = rc.decode_bit_logp(1);
            if pf_on {
                let octave = rc.dec_uint(6);
                pitch_index = ((16 << octave) + rc.dec_bits(4 + octave)) as usize - 1;
                let qg = rc.dec_bits(3);
                if rc.tell() + 2 <= total_bits {
                    prefilter_tapset = rc.decode_icdf(&TAPSET_ICDF, 2) as usize;
                }
                gain1 = 0.09375 * (qg as f32 + 1.0);
            }
        }

        let mut is_transient = false;
        if lm > 0 && rc.tell() + 3 <= total_bits {
            is_transient = rc.decode_bit_logp(3);
        }
        let short_blocks = is_transient;

        let intra_ener = if rc.tell() + 3 <= total_bits {
            rc.decode_bit_logp(3)
        } else {
            false
        };

        if !intra_ener && self.loss_duration != 0 {
            self.make_energy_prediction_safe(start_band, end_band, lm);
        }
        // A concealed overlap is folded before the MDCT adds onto it.
        let fold = self.prefilter_and_fold.then(|| self.postfilter());
        unquant_coarse_energy(
            mode,
            start_band,
            end_band,
            &mut self.old_band_e,
            intra_ener,
            rc,
            channels,
            lm,
        );
        self.w_tf_res[..nb_ebands].fill(0);
        let tf_res = &mut self.w_tf_res[..nb_ebands];
        tf_decode(start_band, end_band, is_transient, tf_res, lm as i32, rc);

        let spread_decision = if rc.tell() + 4 <= total_bits {
            rc.decode_icdf(&SPREAD_ICDF, 5)
        } else {
            SPREAD_NORMAL
        };

        self.w_cap[..nb_ebands].fill(0);
        let cap = &mut self.w_cap[..nb_ebands];
        for (i, cap_i) in cap.iter_mut().enumerate() {
            let n = (mode.e_bands[i + 1] - mode.e_bands[i]) << lm;
            *cap_i = ((mode.cache.caps[nb_ebands * (2 * lm + channels - 1) + i] as i32 + 64)
                * channels as i32
                * n as i32)
                >> 2;
        }

        self.w_offsets[..nb_ebands].fill(0);
        let offsets = &mut self.w_offsets[..nb_ebands];
        let mut dynalloc_logp = 6i32;
        let mut total_bits_bitres = total_bits << BITRES;
        let mut tell_frac = rc.tell_frac();
        for i in start_band..end_band {
            let width =
                channels as i32 * (mode.e_bands[i + 1] - mode.e_bands[i]) as i32 * (1 << lm);
            let quanta = (width << BITRES).min((6i32 << BITRES).max(width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            while tell_frac + (dynalloc_loop_logp << BITRES) < total_bits_bitres && boost < cap[i] {
                let flag = rc.decode_bit_logp(dynalloc_loop_logp as u32);
                tell_frac = rc.tell_frac();
                if !flag {
                    break;
                }
                boost += quanta;
                total_bits_bitres -= quanta;
                dynalloc_loop_logp = 1;
            }
            offsets[i] = boost;
            if boost > 0 {
                dynalloc_logp = dynalloc_logp.max(2) - 1;
                dynalloc_logp = dynalloc_logp.max(2);
            }
        }

        let alloc_trim = if rc.tell_frac() + (6 << BITRES) <= total_bits_bitres {
            rc.decode_icdf(&TRIM_ICDF, 7)
        } else {
            5
        };
        let anti_collapse_rsv = if is_transient && lm >= 2 {
            let remaining = (total_bits << BITRES) - rc.tell_frac() - 1;
            if remaining >= ((lm as i32 + 2) << BITRES) {
                1i32 << BITRES
            } else {
                0
            }
        } else {
            0
        };

        let mut intensity = 0;
        let mut dual_stereo_val = if channels == 2 { 1 } else { 0 };
        let mut balance = 0;
        self.w_pulses[..nb_ebands].fill(0);
        let pulses = &mut self.w_pulses[..nb_ebands];

        let ebands_stereo = if channels > 1 {
            nb_ebands * channels
        } else {
            nb_ebands
        };
        self.w_fine_priority[..ebands_stereo].fill(0);
        let fine_priority = &mut self.w_fine_priority[..ebands_stereo];
        self.w_ebits[..ebands_stereo].fill(0);
        let ebits = &mut self.w_ebits[..ebands_stereo];

        let alloc_bits = (total_bits << BITRES) - rc.tell_frac() - 1 - anti_collapse_rsv;
        let coded_bands = clt_compute_allocation(
            mode,
            start_band,
            end_band,
            offsets,
            cap,
            alloc_trim,
            &mut intensity,
            &mut dual_stereo_val,
            alloc_bits,
            &mut balance,
            pulses,
            ebits,
            fine_priority,
            channels as i32,
            lm as i32,
            rc,
            false,
            0,
            end_band as i32 - 1,
        );

        unquant_fine_energy(
            mode,
            start_band,
            end_band,
            &mut self.old_band_e,
            ebits,
            rc,
            channels,
        );

        // Hard bound: `out_syn_idx = DECODE_BUFFER_SIZE - frame_size` below must
        // stay non-negative. libopus rejects frame sizes beyond DECODE_BUFFER_SIZE the same way
        // (celt_decoder.c). The old `+ overlap` bound let 2049..=2168 through and
        // caused an underflow/panic (issue #27 deep-scan).
        if frame_size > DECODE_BUFFER_SIZE {
            return 0;
        }

        self.w_x[..frame_size * channels].fill(0.0);

        let x_pad_end = (frame_size * channels + STRIDE_ACCESS_PAD).min(self.w_x.len());
        let x = &mut self.w_x[..x_pad_end];
        self.w_collapse_masks[..nb_ebands * channels].fill(0);
        let collapse_masks = &mut self.w_collapse_masks[..nb_ebands * channels];

        let (x_split, y_split) = x.split_at_mut(frame_size);
        let y_opt = if channels == 2 { Some(y_split) } else { None };

        let mut dual_stereo = dual_stereo_val != 0;
        self.w_band_amp[..nb_ebands * channels].fill(0.0);
        let band_amp = &mut self.w_band_amp[..nb_ebands * channels];
        log2amp(mode, nb_ebands, band_amp, &self.old_band_e, channels);
        quant_all_bands(
            false,
            mode,
            start_band,
            end_band,
            x_split,
            y_opt,
            collapse_masks,
            band_amp,
            pulses,
            short_blocks,
            spread_decision,
            &mut dual_stereo,
            intensity as usize,
            tf_res,
            (total_bits << BITRES) - anti_collapse_rsv,
            &mut balance,
            rc,
            lm as i32,
            coded_bands,
            true,
            self.disable_inv,
            &mut self.rng,
            self.arch,
        );
        // Trace X values for comparison with C decoder
        let mut anti_collapse_on = false;
        if anti_collapse_rsv > 0 {
            anti_collapse_on = rc.dec_bits(1) != 0;
        }
        unquant_energy_finalise(
            mode,
            start_band,
            end_band,
            &mut self.old_band_e,
            ebits,
            fine_priority,
            total_bits - rc.tell(),
            rc,
            channels,
        );
        if anti_collapse_on {
            self.rng = crate::bands::anti_collapse(
                mode,
                x,
                collapse_masks,
                lm as i32,
                channels,
                frame_size,
                start_band,
                nb_ebands,
                &self.old_band_e,
                &self.old_band_e2,
                &self.old_band_e3,
                pulses,
                self.rng,
                self.arch,
            );
        }
        if silence {
            // libopus celt_decoder.c:1530-1534 — silence frames carry no energy;
            // force the long-term predictor to a noise floor so the next packet
            // decodes cleanly. Without this the decoder's `oldBandE` retains
            // the previous frame's energies and the next decode diverges, which
            // is the reported "一時的に破壊" (temporarily destroyed) symptom.
            for v in self.old_band_e[..channels * nb_ebands].iter_mut() {
                *v = -28.0;
            }
        }
        // For silence the MDCT synthesis must still run so that overlap,
        // pre-emphasis and prefilter states advance, but with zeroed
        // frequency bins (libopus denormalise_bands(silence=1) sets
        // bound=0/start=end=0). Filling `w_freq` with zeros and skipping
        // denormalisation achieves the same effect without passing a
        // dedicated `silence` flag through `denormalise_bands`.
        let freq_len = frame_size * channels.max(out_channels);
        self.w_freq[..freq_len].fill(0.0);
        let freq = &mut self.w_freq[..freq_len];
        if !silence {
            // Recompute band_amp after unquant_energy_finalise, which adjusts old_band_e.
            // (Mirrors the encoder's resynth path: log2amp is called after quant_energy_finalise.)
            log2amp(mode, nb_ebands, band_amp, &self.old_band_e, channels);
            denormalise_bands(
                mode,
                x,
                freq,
                band_amp,
                start_band,
                end_band,
                channels,
                (1 << lm) as usize,
            );
            // Anti-aliasing: zero MDCT bins above the output Nyquist when
            // downsampling (libopus denormalise_bands `if(downsample!=1)
            // bound=IMIN(bound,N/downsample)`).
            if self.downsample > 1 {
                let bound = frame_size / self.downsample;
                for c in 0..channels {
                    for i in bound..frame_size {
                        freq[c * frame_size + i] = 0.0;
                    }
                }
            }
        }
        // Always trace freq and band_amp for comparison

        // Copy a mono stream to both outputs, or downmix a stereo stream to
        // one (libopus celt_synthesis).
        if channels == 1 && out_channels == 2 {
            freq.copy_within(..frame_size, frame_size);
        } else if channels == 2 && out_channels == 1 {
            let (left, right) = freq.split_at_mut(frame_size);
            for (l, &r) in left.iter_mut().zip(&right[..frame_size]) {
                *l = 0.5 * *l + 0.5 * r;
            }
        }

        let (shift, b) = if short_blocks {
            (mode.max_lm, 1 << lm)
        } else {
            (mode.max_lm - lm, 1)
        };
        let n = frame_size / b;

        for c in 0..out_channels {
            let channel_mem_offset = c * (DECODE_BUFFER_SIZE + overlap);

            let mem_size = DECODE_BUFFER_SIZE + overlap;
            self.decode_mem.copy_within(
                channel_mem_offset + frame_size..channel_mem_offset + mem_size,
                channel_mem_offset,
            );
            if let Some(pf) = &fold {
                plc::prefilter_and_fold(
                    &mut self.decode_mem[channel_mem_offset..channel_mem_offset + mem_size],
                    frame_size,
                    pf,
                    mode.window,
                    overlap,
                    self.arch,
                );
            }

            let out_syn_idx = DECODE_BUFFER_SIZE - frame_size;

            for i in 0..b {
                let block_freq_idx = c * frame_size + i;
                // Stride between short-block MDCT outputs is short_mdct_size (not n).
                // In libopus: out_syn[c] + NB*b, where NB = mode->shortMdctSize.
                // For non-transient b=1, i*n == 0 either way.
                let block_stride = if short_blocks {
                    mode.short_mdct_size
                } else {
                    n
                };
                let block_out_idx = channel_mem_offset + out_syn_idx + i * block_stride;
                let available_len = self.decode_mem.len() - block_out_idx;
                if available_len < n + overlap {
                    panic!(
                        "MDCT backward buffer too small: need {}, have {} (out_syn_idx={}, n={}, overlap={})",
                        n + overlap,
                        available_len,
                        out_syn_idx,
                        n,
                        overlap
                    );
                }
                self.mode.mdct.backward(
                    &freq[block_freq_idx..],
                    &mut self.decode_mem[block_out_idx..],
                    mode.window,
                    overlap,
                    shift,
                    b,
                );
            }

            // celt_synthesis saturates at SIG_SAT, which the float build's
            // SATURATE leaves out (celt/arch.h:323).

            // The postfilter runs in place on the synthesis; the earlier
            // output in decode_mem is its history (celt_decoder.c:1537-1548).
            let out = &mut self.decode_mem[channel_mem_offset..channel_mem_offset + mem_size];
            let short_n = mode.short_mdct_size;
            comb_filter_inplace(
                out,
                out_syn_idx,
                self.prefilter_period_old,
                self.prefilter_period,
                short_n,
                self.prefilter_gain_old,
                self.prefilter_gain,
                self.prefilter_tapset_old,
                self.prefilter_tapset,
                mode.window,
                overlap,
                self.arch,
            );
            if lm != 0 {
                comb_filter_inplace(
                    out,
                    out_syn_idx + short_n,
                    self.prefilter_period,
                    pitch_index,
                    frame_size - short_n,
                    self.prefilter_gain,
                    gain1,
                    self.prefilter_tapset,
                    prefilter_tapset as i32,
                    mode.window,
                    overlap,
                    self.arch,
                );
            }
        }
        self.deemphasis(frame_size, pcm, layout);

        self.prefilter_period_old = self.prefilter_period;
        self.prefilter_gain_old = self.prefilter_gain;
        self.prefilter_tapset_old = self.prefilter_tapset;

        self.postfilter_pitch = if pf_on { pitch_index } else { 0 };
        if pf_on {
            self.prefilter_period = pitch_index;
            self.prefilter_gain = gain1;
            self.prefilter_tapset = prefilter_tapset as i32;
        } else {
            self.prefilter_period = COMBFILTER_MINPERIOD;
            self.prefilter_gain = 0.0;
            self.prefilter_tapset = 0;
        }

        if lm > 0 {
            self.prefilter_period_old = self.prefilter_period;
            self.prefilter_gain_old = self.prefilter_gain;
            self.prefilter_tapset_old = self.prefilter_tapset;
        }

        if channels == 1 {
            self.old_band_e.copy_within(..nb_ebands, nb_ebands);
        }
        // Both halves, as libopus updates `oldLogE` over `2*nbEBands`.
        if !is_transient {
            self.old_band_e3.copy_from_slice(&self.old_band_e2);
            self.old_band_e2.copy_from_slice(&self.old_band_e);
        } else {
            for (e2, &e) in self.old_band_e2.iter_mut().zip(self.old_band_e.iter()) {
                *e2 = e2.min(e);
            }
        }
        // The background may rise 2.4 dB/s; after a gap (DTX) the update
        // packet carries the weight of all the missing ones.
        let max_background_increase =
            ((self.loss_duration as usize + (1 << lm)).min(160) as f32) * 0.001;
        for (bg, &e) in self.background_log_e.iter_mut().zip(self.old_band_e.iter()) {
            *bg = (*bg + max_background_increase).min(e);
        }
        // In case start or end change (libopus celt_decoder.c), the bands this frame didn't code
        // forget their energy and history. Both halves, as libopus clears them: a mono stream
        // keeps the second, and its next frame predicts from the larger of the two.
        for c in 0..CELT_MAX_CHANNELS {
            for i in (0..start_band).chain(end_band..nb_ebands) {
                self.old_band_e[c * nb_ebands + i] = 0.0;
                self.old_band_e2[c * nb_ebands + i] = -28.0;
                self.old_band_e3[c * nb_ebands + i] = -28.0;
            }
        }

        self.rng = rc.rng;
        self.loss_duration = 0;
        self.plc_duration = 0;
        self.last_frame_type = plc::FRAME_NORMAL;
        self.prefilter_and_fold = false;
        self.end_band = end_band;

        api_frame_size
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::{modes, range_coder::RangeCoder};

    /// tone_lpc's singularity test is den < .001f*(r00*r11)
    /// (celt_encoder.c:1347); this input lies between the two products.
    #[test]
    fn tone_lpc_tests_singularity_on_the_product() {
        let x = [
            1065354220u32,
            1065258483,
            1064974314,
            1064508070,
            1063861141,
            1063042093,
        ]
        .map(f32::from_bits);
        assert!(tone_lpc(
            &x,
            6,
            1,
            tone_correlations(&x, 1)[0],
            &mut [0.0; 2]
        ));
    }

    /// tone_detect's frequency is the double acos rounded once to float
    /// (celt_encoder.c:1394); the expected values are glibc's acos. glibc's
    /// acosf differs at the first two inputs, libm's acosf at -1.
    #[test]
    fn tone_freq_rounds_the_double_acos() {
        for (x, delay, want) in [
            (847807907u32, 1, 1070141402u32),
            (964846114, 2, 1061750708),
            (0xbf80_0000, 1, 1078530011),
            (0xbf80_0000, 2, 1070141403),
        ] {
            let lpc0 = 2.0 * f32::from_bits(x);
            assert_eq!(tone_freq(lpc0, delay).to_bits(), want, "{x} {delay}");
        }
    }

    /// The complex-root test sums in double (celt_encoder.c:1383); in
    /// float these pairs sum to exactly 0.
    #[test]
    fn tone_roots_test_sums_in_double() {
        for (l0, l1) in [(1066480345u32, 3198465355u32), (1070819174, 3207500703)] {
            assert!(tone_has_complex_roots([
                f32::from_bits(l0),
                f32::from_bits(l1)
            ]));
        }
        assert!(!tone_has_complex_roots([1.0, -0.25]));
    }

    /// tone_lpc limits lpc[0] to +-1.999999 only where num0/den reaches
    /// +-2 (celt_encoder.c:1354-1356); this input gives -1.9999999.
    #[test]
    fn tone_lpc_keeps_quotients_below_two() {
        let x = [
            1012005056u32,
            1023960200,
            3156529856,
            3188496314,
            1052018348,
        ]
        .map(f32::from_bits);
        let mut lpc = [0.0; 2];
        assert!(!tone_lpc(&x, 5, 1, tone_correlations(&x, 1)[0], &mut lpc));
        assert!(lpc[0] < -1.999999, "{lpc:?}");
    }

    /// dynalloc's tone bin is floor(.5 + tone_freq*120/M_PI) in double
    /// (celt_encoder.c:1210): 0.01308996882 rad/sample is bin 0.
    #[test]
    fn dynalloc_tone_bin_rounds_in_double() {
        let mode = modes::default_mode();
        let band_log_e: Vec<f32> = (0..21).map(|i| 4.0 - 0.3 * i as f32).collect();
        let old = [0.0f32; 21];
        let run = |tone_freq: f32| {
            let (mut offsets, mut importance, mut spread_weight) = ([0; 21], [0; 21], [0; 21]);
            let r = dynalloc_analysis_simple(
                mode,
                &band_log_e,
                None,
                &old,
                0,
                21,
                1,
                3,
                200,
                false,
                &mut offsets,
                24,
                true,
                false,
                &AnalysisInfo::default(),
                false,
                &[0.0; 21],
                tone_freq,
                0.99,
                &mut importance,
                &mut spread_weight,
            );
            (r, offsets, importance)
        };
        assert_eq!(run(0.01308996882), run(0.0));
    }

    /// A pure tone's comb-filter period is floor(.5 + 2*M_PI*multiple/freq)
    /// in double (celt_encoder.c:1452): 0.3807991147 rad/sample gives 16.
    #[test]
    fn tone_period_rounds_in_double() {
        let mode = modes::default_mode();
        let (n, overlap) = (960, mode.overlap);
        let mut in_buf: Vec<f32> = (0..n + overlap)
            .map(|i| (i as f32 * 0.3808).sin() * 1000.0)
            .collect();
        let mut mem = vec![0.0f32; COMBFILTER_MAXPERIOD];
        let mut pre = vec![0.0f32; COMBFILTER_MAXPERIOD + n];
        let mut pitch_buf = vec![0.0f32; (COMBFILTER_MAXPERIOD + n) >> 1];
        let (mut before, mut after) = ([0.0f32; 1], [0.0f32; 1]);
        let (_, _, pitch_index) = run_prefilter(
            &mut in_buf,
            &mut mem,
            0,
            0.0,
            0,
            0,
            mode.window,
            1,
            n,
            overlap,
            &mut pre,
            &mut pitch_buf,
            &mut before,
            &mut after,
            &AnalysisInfo::default(),
            0,
            0.0,
            true,
            100,
            0.3807991147041321,
            1.0,
            10,
            crate::Arch::Scalar,
        );
        assert_eq!(pitch_index, 16);
    }

    /// tf_estimate = celt_sqrt(MAX32(0, 0.0069f*min(163, tf_max) - 0.139))
    /// with the subtraction and root in double (celt_encoder.c:461). The
    /// first frame of a 527.1533 Hz tone at 48 kHz gives libopus's
    /// 0.8355611562728882.
    #[test]
    fn tf_estimate_subtracts_in_double() {
        let mode = modes::default_mode();
        let (n, overlap) = (960, mode.overlap);
        let mut input = vec![0.0f32; n + overlap];
        let mut m = 0.0f32;
        for i in 0..n {
            let x = (i as f64 * 527.1533f32 as f64 / 48000.0 * core::f64::consts::TAU).sin() as f32
                * 0.5
                * 32768.0;
            input[overlap + i] = x - m;
            m = x * mode.preemph[0];
        }
        let (mut tf_estimate, mut tf_chan, mut weak) = (0.0, 0, false);
        let mut tmp = vec![0.0; MAX_TRANSIENT_LEN];
        let (tone_freq, toneishness) = tone_detect(&input, 1, n + overlap, 48000);
        transient_analysis(
            &input,
            n + overlap,
            1,
            &mut tf_estimate,
            &mut tf_chan,
            false,
            &mut weak,
            tone_freq,
            toneishness,
            &mut tmp,
        );
        assert_eq!(tf_estimate, 0.8355611562728882);
    }

    /// A silent frame shrinks to two bytes when the encoder has VBR on and
    /// a bitrate (celt_encoder.c:1988-1994), through every entry point.
    #[test]
    fn silent_frames_shrink_under_vbr_only() {
        let silence = [0.0f32; 960];
        for (vbr, bitrate, is_vbr, bytes) in [
            (true, 64000, false, 2),
            (false, 64000, true, 100),
            (true, OPUS_BITRATE_MAX, true, 100),
        ] {
            let mut enc = CeltEncoder::new(modes::default_mode(), 1);
            enc.set_vbr(vbr);
            enc.set_bitrate(bitrate);
            let mut rc = RangeCoder::new_encoder(100);
            enc.encode_with_budget_vbr(&silence, 960, &mut rc, 0, 800, is_vbr)
                .unwrap();
            assert_eq!(
                rc.storage, bytes,
                "vbr {vbr} bitrate {bitrate} is_vbr {is_vbr}"
            );
        }
    }

    /// patch_transient_decision divides by the signed band count
    /// C*(end-1-IMAX(2,start)) (celt_encoder.c:504), which is negative for
    /// end bands 1 and 2: the mean stays -0 and the frame is no transient.
    #[test]
    fn end_bands_below_three_patch_no_transient() {
        let new_e = [10.0f32; 42];
        let old_e = [-10.0f32; 42];
        for end in [1, 2, 3] {
            for c in [1, 2] {
                assert!(!patch_transient_decision(&new_e, &old_e, 21, 0, end, c));
            }
        }
        assert!(patch_transient_decision(&new_e, &old_e, 21, 0, 4, 2));
    }

    /// A start band at or above the end band codes no band; those frames
    /// are refused (they panicked in the allocation before), and every
    /// start below the end band codes.
    #[test]
    fn start_bands_below_the_end_band_code() {
        let pcm: Vec<f32> = (0..1920)
            .map(|i| ((i * 7919) % 101) as f32 / 101.0 - 0.5)
            .collect();
        for end in [1, 2, 3, 13, 17, 21] {
            for start in [0, 1, 2, 5, 16, 17, 20, 21, 30] {
                let mut enc = CeltEncoder::new(modes::default_mode(), 2);
                enc.complexity = 10;
                enc.set_end_band(end);
                for _ in 0..3 {
                    let mut rc = RangeCoder::new_encoder(200);
                    let r = enc.encode_with_start_band(&pcm, 960, &mut rc, start);
                    assert_eq!(r.is_ok(), start < end, "start {start} end {end}");
                }
            }
        }
    }

    /// libopus's CELT ctls reject out-of-range values and keep the setting
    /// (celt_encoder.c:2956-3024); the bitrate is capped at 750 kb/s per
    /// channel.
    #[test]
    fn out_of_range_settings_are_ignored() {
        let mut enc = CeltEncoder::new(modes::default_mode(), 1);
        enc.set_lsb_depth(16);
        enc.set_lsb_depth(30);
        enc.set_loss_rate(10);
        enc.set_loss_rate(150);
        enc.set_bitrate(64000);
        enc.set_bitrate(400);
        enc.set_end_band(17);
        enc.set_end_band(0);
        enc.set_stream_channels(3);
        enc.set_prediction(1);
        enc.set_prediction(3);
        assert_eq!(
            (
                enc.lsb_depth,
                enc.loss_rate,
                enc.bitrate,
                enc.end_band,
                enc.stream_channels
            ),
            (16, 10, 64000, 17, 1)
        );
        assert_eq!((enc.disable_pf, enc.force_intra), (true, false));
        enc.set_bitrate(2_000_000);
        assert_eq!(enc.bitrate, 750000);
    }

    #[test]
    fn frames_without_a_transient_flag_are_coded_as_not_transient() {
        // A 2.5 ms frame codes no transient flag, so decoders take it as not
        // transient. libopus codes it with long blocks and counts it as a
        // transient frame (celt_encoder.c:1728-1735, 2805-2808): every frame
        // of a 2.5 ms stream counts, whatever its transient analysis says.
        let mut enc = CeltEncoder::new(modes::default_mode(), 1);
        enc.set_end_band(21);
        enc.complexity = 10;
        let code = |enc: &mut CeltEncoder, pcm: &[f32]| {
            let mut rc = RangeCoder::new_encoder(20);
            enc.encode_with_budget_vbr(pcm, pcm.len(), &mut rc, 0, 160, false)
                .unwrap();
            rc.done();
        };
        let tone: Vec<f32> = (0..120 * 8).map(|i| 0.1 * (i as f32 * 0.3).sin()).collect();
        for f in 0..7 {
            code(&mut enc, &tone[f * 120..(f + 1) * 120]);
        }
        let mut clicks = tone[840..].to_vec();
        for (i, x) in clicks[60..90].iter_mut().enumerate() {
            *x = if i % 2 == 0 { 0.9 } else { -0.9 };
        }
        code(&mut enc, &clicks);
        assert_eq!(enc.consec_transient, 8);
    }

    /// celt_encode_with_ec returns OPUS_BAD_ARG for a budget below 2 bytes
    /// or a frame size other than 120 << LM (celt_encoder.c:1832-1845),
    /// before it changes the state: the next frame codes like a new
    /// encoder's first. 48-sample frames (8 kHz PCM fed at 48 kHz) panicked
    /// in the MDCT before.
    #[test]
    fn bad_budgets_and_frame_sizes_are_refused() {
        let pcm: Vec<f32> = (0..960).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let first = |enc: &mut CeltEncoder| {
            let mut rc = RangeCoder::new_encoder(100);
            enc.encode(&pcm, 960, &mut rc).unwrap();
            rc.done();
            (rc.buf[..100].to_vec(), rc.rng)
        };
        let mut enc = CeltEncoder::new(modes::default_mode(), 1);
        let want = first(&mut enc);
        let mut enc = CeltEncoder::new(modes::default_mode(), 1);
        for (size, frame, bits) in [
            (1, 960, None),
            (100, 960, Some(15)),
            (100, 960, Some(-8)),
            (100, 48, None),
            (100, 100, Some(800)),
        ] {
            let mut rc = RangeCoder::new_encoder(size);
            let r = match bits {
                None => enc.encode(&pcm[..frame], frame, &mut rc),
                Some(b) => enc.encode_with_budget(&pcm[..frame], frame, &mut rc, 0, b),
            };
            assert_eq!(
                r,
                Err(crate::error::BAD_ARG),
                "{size} bytes, {frame} samples, {bits:?} bits"
            );
            assert_eq!(rc.tell(), 1);
        }
        assert_eq!(first(&mut enc), want);
    }

    // Issue #54: a frame whose comb filter is cancelled must still fade the
    // previous frame's filter out over the overlap, because the decoder's
    // post-filter always does. Pitched history into a silent frame: any
    // filtering adds energy, so the filter is cancelled after an "on" frame.
    #[test]
    fn cancelled_prefilter_still_fades_out_previous_filter() {
        let mode = modes::default_mode();
        let (n, overlap, max_period) = (960, mode.overlap, COMBFILTER_MAXPERIOD);
        let mut mem: Vec<f32> = (0..max_period)
            .map(|i| (i as f32 * 0.3).sin() * 1000.0)
            .collect();
        let mut in_buf = vec![0.0f32; n + overlap];
        let mut pre = vec![0.0f32; max_period + n];
        let mut pitch_buf = vec![0.0f32; (max_period + n) >> 1];
        let (mut before, mut after) = ([0.0f32], [0.0f32]);
        let (pf_on, gain, _) = run_prefilter(
            &mut in_buf,
            &mut mem,
            100,
            0.5,
            0,
            0,
            mode.window,
            1,
            n,
            overlap,
            &mut pre,
            &mut pitch_buf,
            &mut before,
            &mut after,
            &AnalysisInfo::default(),
            0,
            0.0,
            true,
            100,
            -1.0,
            0.0,
            9,
            crate::Arch::Scalar,
        );
        assert!(!pf_on && gain == 0.0, "expected a cancelled filter");
        let body = &in_buf[overlap..];
        assert!(
            body[..overlap].iter().any(|&v| v.abs() > 1.0),
            "previous filter was not faded out over the overlap"
        );
        assert!(
            body[overlap..].iter().all(|&v| v == 0.0),
            "frame past the overlap must be the unfiltered input"
        );
    }
}

#[cfg(all(test, feature = "std"))]
mod mode_switch_tests {
    use super::*;
    use crate::{modes, range_coder::RangeCoder};

    /// A 200 Hz harmonic series with a little noise, at 48 kHz from sample
    /// `start`.
    fn voiced(n: usize, start: usize, seed: &mut u32) -> Vec<f32> {
        (start..start + n)
            .map(|i| {
                *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let t = i as f32 / 48000.0;
                let harmonics: f32 = (1..8)
                    .map(|k| (2.0 * core::f32::consts::PI * 200.0 * k as f32 * t).sin() / k as f32)
                    .sum();
                0.2 * harmonics + ((*seed >> 9) as f32 / (1 << 23) as f32 - 0.5) * 0.01
            })
            .collect()
    }

    fn encoder() -> CeltEncoder {
        let mut enc = CeltEncoder::new(modes::default_mode(), 1);
        enc.set_end_band(21);
        enc.complexity = 10;
        enc
    }

    /// A CBR frame of `bytes` bytes.
    fn encode(enc: &mut CeltEncoder, pcm: &[f32], bytes: usize) -> Vec<u8> {
        let mut rc = RangeCoder::new_encoder(bytes as u32);
        enc.encode_with_budget_vbr(pcm, pcm.len(), &mut rc, 0, (bytes * 8) as i32, false)
            .unwrap();
        rc.done();
        rc.buf[..bytes].to_vec()
    }

    /// The post-filter and intra-energy flags of a 20 ms frame.
    fn flags(frame: &[u8]) -> (bool, bool) {
        let mut rc = RangeCoder::new_decoder(frame);
        assert!(!rc.decode_bit_logp(15), "silence");
        let postfilter = rc.decode_bit_logp(1);
        if postfilter {
            let octave = rc.dec_uint(6);
            rc.dec_bits(4 + octave);
            rc.dec_bits(3);
            rc.decode_icdf(&TAPSET_ICDF, 2);
        }
        rc.decode_bit_logp(3); // transient
        (postfilter, rc.decode_bit_logp(3))
    }

    #[test]
    fn prediction_off_drops_the_prefilter_and_codes_energy_intra() {
        // libopus CELT_SET_PREDICTION(0), which the first frame after a mode
        // switch uses: no pitch pre-filter and intra-coded energy.
        let (mut a, mut b) = (encoder(), encoder());
        let (mut sa, mut sb) = (1, 1);
        for f in 0..6 {
            encode(&mut a, &voiced(960, f * 960, &mut sa), 160);
            encode(&mut b, &voiced(960, f * 960, &mut sb), 160);
        }
        b.set_prediction(0);
        let with = flags(&encode(&mut a, &voiced(960, 6 * 960, &mut sa), 160));
        let without = flags(&encode(&mut b, &voiced(960, 6 * 960, &mut sb), 160));
        assert_eq!(with, (true, false));
        assert_eq!(without, (false, true));
    }

    #[test]
    fn reset_state_restarts_like_a_new_encoder() {
        // libopus OPUS_RESET_STATE, which a mode switch's redundancy frames
        // and prefill rely on.
        let mut used = encoder();
        let mut seed = 3;
        for f in 0..5 {
            encode(&mut used, &voiced(960, f * 960, &mut seed), 120);
        }
        used.reset_state();
        let frame = voiced(960, 0, &mut 9);
        assert_eq!(
            encode(&mut used, &frame, 120),
            encode(&mut encoder(), &frame, 120)
        );
    }
}

#[cfg(all(test, feature = "std"))]
mod hybrid_decision_tests {
    use super::*;
    use crate::{modes, range_coder::RangeCoder};

    /// Deterministic noise, channel-major like the CELT input.
    fn noise(n: usize, seed: &mut u32, amplitude: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((*seed >> 9) as f32 / (1 << 23) as f32 - 0.5) * amplitude
            })
            .collect()
    }

    fn vbr_encoder(channels: usize, bitrate: i32) -> CeltEncoder {
        let mut enc = CeltEncoder::new(modes::default_mode(), channels);
        enc.set_end_band(21);
        enc.set_bitrate(bitrate);
        enc.set_vbr(true);
        enc.set_constrained_vbr(false);
        enc.complexity = 9;
        enc
    }

    fn encode(enc: &mut CeltEncoder, pcm: &[f32], start_band: usize, bytes: usize) -> i32 {
        let mut rc = RangeCoder::new_encoder(bytes as u32);
        let vbr = enc.vbr;
        enc.encode_with_budget_vbr(pcm, 960, &mut rc, start_band, (bytes * 8) as i32, vbr)
            .unwrap();
        rc.tell()
    }

    #[test]
    fn hybrid_frames_fix_the_trim_and_spread_aggressively() {
        let mut enc = vbr_encoder(1, 8000);
        enc.set_silk_info(1, 240);
        let mut seed = 1;
        let mut steady = 0;
        for _ in 0..6 {
            let pcm = noise(960, &mut seed, 0.2);
            encode(&mut enc, &pcm, 17, 1499);
            assert_eq!(enc.decisions.alloc_trim, 5);
            let transient = enc.decisions.is_transient;
            steady += usize::from(!transient);
            let spread = if transient {
                SPREAD_NORMAL
            } else {
                SPREAD_AGGRESSIVE
            };
            assert_eq!(enc.spread_decision, spread);
        }
        assert!(steady >= 3);
    }

    /// A quiet frame with a loud click in its second half.
    fn click(seed: &mut u32) -> Vec<f32> {
        let mut pcm = noise(960, seed, 0.001);
        for (i, x) in pcm[700..760].iter_mut().enumerate() {
            *x += if i % 2 == 0 { 0.8 } else { -0.8 };
        }
        pcm
    }

    #[test]
    fn low_rate_hybrid_transients_keep_long_resolution() {
        // Unvoiced SILK: a weak transient codes TF on the long window, a
        // strong one keeps 5 ms resolution (celt_encoder.c:2261-2276).
        let mut enc = vbr_encoder(1, 2000);
        enc.set_silk_info(1, 240);
        let mut seed = 7;
        encode(&mut enc, &noise(960, &mut seed, 0.001), 17, 1499);
        encode(&mut enc, &click(&mut seed), 17, 1499);
        let d = enc.decisions;
        assert!(d.is_transient || d.weak_transient);
        if d.weak_transient {
            assert!(!d.is_transient);
            assert_eq!((d.tf_select, &d.tf_res[17..21]), (0, &[1; 4][..]));
        } else {
            assert_eq!((d.tf_select, &d.tf_res[17..21]), (1, &[0; 4][..]));
        }
        // Voiced SILK: no weak transients, TF follows the transient flag.
        let mut enc = vbr_encoder(1, 2000);
        enc.set_silk_info(2, 32);
        let mut seed = 7;
        encode(&mut enc, &noise(960, &mut seed, 0.001), 17, 1499);
        encode(&mut enc, &click(&mut seed), 17, 1499);
        let d = enc.decisions;
        assert!(d.is_transient && !d.weak_transient);
        assert_eq!((d.tf_select, &d.tf_res[17..21]), (0, &[1; 4][..]));
    }

    #[test]
    fn celt_frames_without_tf_analysis_follow_the_transient_flag() {
        // Too few bytes for TF analysis: tf_res is the transient flag.
        let mut enc = CeltEncoder::new(modes::default_mode(), 1);
        enc.set_end_band(21);
        enc.complexity = 9;
        let mut seed = 3;
        encode(&mut enc, &noise(960, &mut seed, 0.001), 0, 12);
        encode(&mut enc, &click(&mut seed), 0, 12);
        let d = enc.decisions;
        assert!(d.is_transient);
        assert_eq!(d.tf_select, 0);
        assert!(d.tf_res[..21].iter().all(|&tf| tf == 1));
    }

    #[test]
    fn weak_transients_are_found_only_when_allowed() {
        let mode = modes::default_mode();
        let len = 960 + mode.overlap;
        let mut tmp = vec![0.0f32; MAX_TRANSIENT_LEN];
        let mut found = false;
        for step in [1.5f32, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0] {
            let mut seed = 11;
            let mut x = noise(len, &mut seed, 1000.0);
            for v in &mut x[len / 2..] {
                *v *= step;
            }
            let analyse = |allow: bool, tmp: &mut [f32]| {
                let (mut tf_estimate, mut tf_chan, mut weak) = (0.0, 0, false);
                let transient = transient_analysis(
                    &x,
                    len,
                    1,
                    &mut tf_estimate,
                    &mut tf_chan,
                    allow,
                    &mut weak,
                    0.0,
                    0.0,
                    tmp,
                );
                (transient, weak)
            };
            let (transient, weak) = analyse(false, &mut tmp);
            assert!(!weak);
            let (allowed_transient, allowed_weak) = analyse(true, &mut tmp);
            assert!(!(allowed_transient && allowed_weak));
            found |= transient && allowed_weak;
        }
        assert!(found, "no step gave a weak transient");
    }

    #[test]
    fn stereo_intensity_follows_the_celt_rate() {
        let mut seed = 5;
        // Hybrid at a tiny CELT share: intensity stereo from the first band.
        let mut enc = vbr_encoder(2, 4000);
        enc.set_silk_info(1, 240);
        encode(&mut enc, &noise(1920, &mut seed, 0.2), 17, 1499);
        assert_eq!(enc.intensity, 17);
        // CELT at 24 kb/s: the equivalent rate is the target, not the
        // packet capacity, so intensity stereo starts at band 10.
        let mut enc = vbr_encoder(2, 24000);
        encode(&mut enc, &noise(1920, &mut seed, 0.2), 0, 1275);
        assert_eq!(enc.intensity, 10);
    }

    #[test]
    fn silk_offset_moves_the_hybrid_target() {
        let mut seed = 9;
        let pcm = noise(960, &mut seed, 0.2);
        let mut bits = [0; 2];
        for (bits, (signal_type, offset)) in bits.iter_mut().zip([(2, 32), (1, 240)]) {
            let mut enc = vbr_encoder(1, 6000);
            enc.set_silk_info(signal_type, offset);
            *bits = encode(&mut enc, &pcm, 17, 1499);
        }
        // +12 bits for tonal, -18 for noisy 20 ms frames.
        assert!(bits[0] > bits[1] + 16, "{bits:?}");
    }

    #[test]
    fn loss_rate_reaches_the_coarse_energy() {
        let mut out = [Vec::new(), Vec::new()];
        for (out, loss) in out.iter_mut().zip([0, 25]) {
            let mut enc = vbr_encoder(1, 32000);
            enc.set_loss_rate(loss);
            let mut seed = 13;
            for frame in 0..30 {
                let amplitude = if frame % 7 == 0 { 0.6 } else { 0.05 };
                out.push(encode(&mut enc, &noise(960, &mut seed, amplitude), 0, 1275));
            }
        }
        assert_ne!(out[0], out[1]);
    }

    #[test]
    fn coded_band_count_moves_by_one_band_per_frame() {
        let mut enc = CeltEncoder::new(modes::default_mode(), 1);
        enc.set_end_band(21);
        enc.complexity = 9;
        let mut seed = 17;
        let mut last: Option<i32> = None;
        for frame in 0..12 {
            let bytes = if frame % 2 == 0 { 1275 } else { 16 };
            encode(&mut enc, &noise(960, &mut seed, 0.3), 0, bytes);
            if let Some(last) = last {
                assert!((enc.last_coded_bands - last).abs() <= 1);
            }
            last = Some(enc.last_coded_bands);
        }
    }
}
