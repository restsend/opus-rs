use crate::bands::{
    SPREAD_NONE, SPREAD_NORMAL, compute_band_energies, denormalise_bands, haar1, log2amp,
    normalise_bands, quant_all_bands, spreading_decision,
};
use crate::modes::{CeltMode, SPREAD_ICDF, TAPSET_ICDF, TF_SELECT_TABLE, TRIM_ICDF};
use crate::quant_bands::{
    quant_coarse_energy_advanced, quant_energy_finalise, quant_fine_energy, unquant_coarse_energy,
    unquant_energy_finalise, unquant_fine_energy,
};
use crate::range_coder::RangeCoder;
use crate::rate::{BITRES, clt_compute_allocation};

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::fixedvec::FixedVec;
#[inline]
fn bitrate_to_bits(bitrate: i32, fs: i32, frame_size: i32) -> i32 {
    bitrate * 6 / (6 * fs / frame_size)
}
#[inline]
fn bits_to_bitrate(bits: i32, fs: i32, frame_size: i32) -> i32 {
    bits * (6 * fs / frame_size) / 6
}

const OPUS_BITRATE_MAX: i32 = -1;
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
    let mut coded_bands = if last_coded_bands != 0 { last_coded_bands } else { nb_ebands };
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
        let coded_stereo_dof = mode.e_bands[coded_stereo_bands as usize] as i32 * (1 << lm) - coded_stereo_bands;
        let max_frac = (0.8 * coded_stereo_dof as f32) / coded_bins as f32;
        let ss = stereo_saving.min(1.0);
        let adjust = ((ss - 0.1).max(0.0) * coded_stereo_dof as f32 * (1 << BITRES) as f32) as i32;
        let save = (max_frac * target as f32) as i32;
        target -= save.min(adjust);
    }
    target += tot_boost - (19 << lm);
    // tf boost: libopus SHL32(MULT16_32_Q15(tf-0.044, target),1) collapses to (tf-0.044)*target in float
    target += ((tf_estimate - 0.044) * target as f32) as i32;
    if analysis.valid && !lfe {
        let mut tonal = (analysis.tonality - 0.15).max(0.0) - 0.12;
        let mut tonal_target = target + ((coded_bins << BITRES) as f32 * 1.2 * tonal) as i32;
        if pitch_change != 0 {
            tonal_target += ((coded_bins << BITRES) as f32 * 0.8) as i32;
        }
        target = tonal_target;
    }
    if has_surround_mask && !lfe {
        let surround_target = target + (surround_masking * coded_bins as f32 * (1 << BITRES) as f32) as i32;
        target = (target / 4).max(surround_target);
    }
    // floor depth: only clamp when max_depth is meaningful (>0); stub call passes 0 to avoid spurious 0.665*base
    if max_depth > 0.0 {
        let bins = mode.e_bands[(nb_ebands - 2) as usize] as i32 * (1 << lm);
        let floor_depth = ((c * bins * (1 << BITRES) as i32) as f32 * max_depth) as i32;
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
const EMEANS_F: [f32; 25] = [
    6.4375, 6.25, 5.75, 5.3125, 5.0625, 4.8125, 4.5, 4.375, 4.875, 4.6875, 4.5625, 4.4375, 4.875, 4.625,
    4.3125, 4.5, 4.375, 4.625, 4.75, 4.4375, 3.75, 3.75, 3.75, 3.75, 3.75,
];
fn compute_max_depth(mode: &CeltMode, band_log_e: &[f32], nb_ebands: usize, c: usize, end: usize, lsb_depth: i32) -> f32 {
    let mut max_depth: f32 = -31.9;
    for ch in 0..c {
        for i in 0..end {
            let log_n = mode.log_n[i] as f32;
            let e_mean = if i < EMEANS_F.len() { EMEANS_F[i] } else { 0.0 };
            let noise_floor = 0.0625 * log_n + 0.5 + (9 - lsb_depth) as f32 - e_mean + 0.0062 * ((i + 5) * (i + 5)) as f32;
            let v = band_log_e[ch * nb_ebands + i] - noise_floor;
            if v > max_depth {
                max_depth = v;
            }
        }
    }
    max_depth
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

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn sum_abs_neon(x: &[f32], n: usize) -> f32 {
    let mut sum_vec = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 16 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        let x2 = vld1q_f32(x.as_ptr().add(i + 8));
        let x3 = vld1q_f32(x.as_ptr().add(i + 12));

        sum_vec = vfmaq_f32(sum_vec, vabsq_f32(x0), vdupq_n_f32(1.0));
        sum_vec = vfmaq_f32(sum_vec, vabsq_f32(x1), vdupq_n_f32(1.0));
        sum_vec = vfmaq_f32(sum_vec, vabsq_f32(x2), vdupq_n_f32(1.0));
        sum_vec = vfmaq_f32(sum_vec, vabsq_f32(x3), vdupq_n_f32(1.0));

        i += 16;
    }

    while i + 8 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        let x1 = vld1q_f32(x.as_ptr().add(i + 4));
        sum_vec = vfmaq_f32(sum_vec, vabsq_f32(x0), vdupq_n_f32(1.0));
        sum_vec = vfmaq_f32(sum_vec, vabsq_f32(x1), vdupq_n_f32(1.0));
        i += 8;
    }

    while i + 4 <= n {
        let x0 = vld1q_f32(x.as_ptr().add(i));
        sum_vec = vfmaq_f32(sum_vec, vabsq_f32(x0), vdupq_n_f32(1.0));
        i += 4;
    }

    let mut sum = vaddvq_f32(sum_vec);

    for j in i..n {
        sum += x[j].abs();
    }

    sum
}

#[inline(always)]
fn sum_abs(x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if crate::compat::x86_has_avx() {
            return sum_abs_avx(x, x.len());
        }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        sum_abs_neon(x, x.len())
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        x.iter().map(|&v| v.abs()).sum()
    }
}

const MAX_FRAME_SIZE: usize = 2880;

/// Largest CELT-internal synthesis buffer. The CELT decoder writes `n + overlap`
/// MDCT samples per channel where `n ≤ frame_size ≤ 960` (20 ms @ 48 kHz); 2048
/// is the libopus reference value (`DECODE_BUFFER_SIZE`) and gives ample margin.
/// (Reduced from a previous 3072 to shrink the heap-free footprint.)
const DECODE_BUFFER_SIZE: usize = 2048;

const INV_TABLE: [u8; 128] = [
    255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18, 17,
    16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8, 8,
    8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3,
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
    tmp2: &mut [f32],
) -> bool {
    let mut mask_metric = 0.0f32;
    let mut forward_decay = 0.0625f32;

    *weak_transient = false;
    if allow_weak_transients {
        forward_decay = 0.03125f32;
    }

    let len2 = len / 2;
    debug_assert!(len <= MAX_TRANSIENT_LEN);

    for c in 0..channels {
        let mut mem0 = 0.0f32;
        let mut mem1 = 0.0f32;

        for i in 0..len {
            let x = input[c * len + i];
            let y = mem0 + x;
            let mem00 = mem0;
            mem0 = mem0 - x + 0.5 * mem1;
            mem1 = x - mem00;
            tmp[i] = y;
        }

        tmp[..12].fill(0.0);

        let mut mean = 0.0f32;
        mem0 = 0.0f32;
        for i in 0..len2 {
            let x2 = (tmp[2 * i] * tmp[2 * i] + tmp[2 * i + 1] * tmp[2 * i + 1]) / 16.0;
            // C float build: PSHR32(x2, 12) is an identity, so mean += x2
            mean += x2;
            mem0 = x2 + (1.0 - forward_decay) * mem0;
            tmp2[i] = forward_decay * mem0;
        }

        mem0 = 0.0f32;
        let mut max_e = 0.0f32;
        for i in (0..len2).rev() {
            mem0 = tmp2[i] + 0.875 * mem0;
            tmp2[i] = 0.125 * mem0;
            if tmp2[i] > max_e {
                max_e = tmp2[i];
            }
        }

        mean = (mean * max_e * 0.5 * (len2 as f32)).sqrt();
        // C (float build): norm = len2 / (EPSILON + mean)  [EPSILON = 1e-15f]
        let norm = (len2 as f32) / (1e-15 + mean);

        // C computes the harmonic mean with INTEGER arithmetic (mask_metric and
        // unmask are opus_int32; the 64/6 scaling is an integer division).
        let mut unmask: i32 = 0;
        for i in (12..(len2 - 5)).step_by(4) {
            let id = (64.0 * norm * (tmp2[i] + 1e-15)).floor() as i32;
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

    // C (celt_encoder.c:460-461): tf_max = max(0, sqrt(27*mask_metric) - 42);
    // tf_estimate = sqrt(max(0, 0.0069*min(163, tf_max) - 0.139))  [float build]
    let tf_max = ((27.0 * mask_metric).sqrt() - 42.0).max(0.0);
    *tf_estimate = (0.0069 * tf_max.min(163.0) - 0.139).max(0.0).sqrt();

    is_transient
}

fn l1_metric(tmp: &[f32], n: usize, lm: i32, bias: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if n >= 16 && crate::compat::x86_has_avx() {
            return l1_metric_avx(tmp, n, lm, bias);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if n >= 16 {
            return unsafe { l1_metric_neon(tmp, n, lm, bias) };
        }
    }

    let mut l1 = 0.0f32;
    for &tv in tmp[..n].iter() {
        l1 += tv.abs();
    }
    l1 + (lm as f32) * bias * l1
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn sum_abs_avx(x: &[f32], n: usize) -> f32 {
    use core::arch::x86_64::*;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut i = 0usize;
    let sign_mask = _mm256_set1_ps(-0.0);

    while i + 16 <= n {
        let v0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let v1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));
        sum0 = _mm256_add_ps(sum0, _mm256_andnot_ps(sign_mask, v0));
        sum1 = _mm256_add_ps(sum1, _mm256_andnot_ps(sign_mask, v1));
        i += 16;
    }

    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        sum0 = _mm256_add_ps(sum0, _mm256_andnot_ps(sign_mask, v));
        i += 8;
    }

    let sum = _mm256_add_ps(sum0, sum1);
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let s4 = _mm_add_ps(lo, hi);
    let t1 = _mm_movehl_ps(s4, s4);
    let s2 = _mm_add_ps(s4, t1);
    let t2 = _mm_shuffle_ps(s2, s2, 0x55);
    let mut out = _mm_cvtss_f32(_mm_add_ss(s2, t2));

    for j in i..n {
        out += x[j].abs();
    }

    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn l1_metric_avx(tmp: &[f32], n: usize, lm: i32, bias: f32) -> f32 {
    let l1 = sum_abs_avx(tmp, n);
    l1 + (lm as f32) * bias * l1
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l1_metric_neon(tmp: &[f32], n: usize, lm: i32, bias: f32) -> f32 {
    unsafe {
        let mut sum4 = vdupq_n_f32(0.0);
        let mut i = 0;

        while i + 15 < n {
            let v0 = vld1q_f32(tmp.as_ptr().add(i));
            let v1 = vld1q_f32(tmp.as_ptr().add(i + 4));
            let v2 = vld1q_f32(tmp.as_ptr().add(i + 8));
            let v3 = vld1q_f32(tmp.as_ptr().add(i + 12));

            sum4 = vaddq_f32(sum4, vabsq_f32(v0));
            sum4 = vaddq_f32(sum4, vabsq_f32(v1));
            sum4 = vaddq_f32(sum4, vabsq_f32(v2));
            sum4 = vaddq_f32(sum4, vabsq_f32(v3));

            i += 16;
        }

        while i + 3 < n {
            let v = vld1q_f32(tmp.as_ptr().add(i));
            sum4 = vaddq_f32(sum4, vabsq_f32(v));
            i += 4;
        }

        let sum2 = vpaddq_f32(sum4, sum4);
        let sum1 = vpaddq_f32(sum2, sum2);
        let mut l1 = vgetq_lane_f32(sum1, 0);

        while i < n {
            l1 += tmp[i].abs();
            i += 1;
        }

        l1 + (lm as f32) * bias * l1
    }
}

const MAX_NB_EBANDS: usize = 21;

const MAX_TF_TMP: usize = 176;

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
    let mut tmp = [0.0f32; MAX_TF_TMP];
    let mut tmp_1 = [0.0f32; MAX_TF_TMP];

    let bias = 0.04 * (-0.25f32).max(0.5 - tf_estimate);

    for (i, metric_i) in metric[..len].iter_mut().enumerate() {
        let n = ((mode.e_bands[i + 1] - mode.e_bands[i]) as usize) << lm;
        let narrow = (mode.e_bands[i + 1] - mode.e_bands[i]) == 1;
        let offset = tf_chan * n0 + ((mode.e_bands[i] as usize) << lm);
        tmp[..n].copy_from_slice(&x[offset..offset + n]);

        let mut l1 = l1_metric(&tmp[..n], n, if is_transient { lm } else { 0 }, bias);
        let mut best_l1 = l1;
        let mut best_level = 0;

        if is_transient && !narrow {
            tmp_1[..n].copy_from_slice(&tmp[..n]);
            haar1(&mut tmp_1[..n], n >> lm, 1 << lm);
            l1 = l1_metric(&tmp_1[..n], n, lm + 1, bias);
            if l1 < best_l1 {
                best_l1 = l1;
                best_level = -1;
            }
        }

        for k in 0..(lm + if is_transient || narrow { 0 } else { 1 }) {
            let b = if is_transient { lm - k - 1 } else { k + 1 };

            haar1(&mut tmp[..n], n >> k, 1 << k);
            l1 = l1_metric(&tmp[..n], n, b, bias);

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
                        - 2 * TF_SELECT_TABLE[lm as usize]
                            [4 * (is_transient as usize) + 2 * sel]
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
        * (metric[0] - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel] as i32)
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
                * (metric[i] - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel] as i32)
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
        tf_res[i] = if tf_res[i + 1] == 1 { path1[i + 1] } else { path0[i + 1] };
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
    let mut sum_lr = 1e-9f32;
    let mut sum_ms = 1e-9f32;

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

    sum_ms *= core::f32::consts::FRAC_1_SQRT_2;
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

const PREFILTER_GAINS: [[f32; 3]; 3] = [
    [0.306_640_6, 0.217_041, 0.129_638_7],
    [0.463_867_2, 0.268_066_4, 0.0],
    [0.799_804_7, 0.100_097_7, 0.0],
];

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
) {
    #[cfg(target_arch = "aarch64")]
    {
        comb_filter_const_neon(y, x, y_idx, x_idx, t, n, g10, g11, g12);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        if crate::compat::x86_has_avx() {
            comb_filter_const_avx(y, x, y_idx, x_idx, t, n, g10, g11, g12);
            return;
        }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    unsafe {
        comb_filter_const_sse(y, x, y_idx, x_idx, t, n, g10, g11, g12);
        #[allow(clippy::needless_return)]
        return;
    }
    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "sse")
    )))]
    {
        comb_filter_const_scalar(y, x, y_idx, x_idx, t, n, g10, g11, g12);
    }
}

#[inline]
#[allow(dead_code)]
fn comb_filter_const_scalar(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    let mut x1;
    let mut x2;
    let mut x3;
    let mut x4;
    let mut x0;

    x4 = x[x_idx - t - 2];
    x3 = x[x_idx - t - 1];
    x2 = x[x_idx - t];
    x1 = x[x_idx - t + 1];

    for i in 0..n {
        x0 = x[x_idx + i - t + 2];
        y[y_idx + i] = x[x_idx + i] + g10 * x2 + g11 * (x1 + x3) + g12 * (x0 + x4);
        x4 = x3;
        x3 = x2;
        x2 = x1;
        x1 = x0;
    }
}

#[cfg(target_arch = "aarch64")]
fn comb_filter_const_neon(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    unsafe { comb_filter_const_neon_impl(y, x, y_idx, x_idx, t, n, g10, g11, g12) }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn comb_filter_const_neon_impl(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    use core::arch::aarch64::*;

    let g10v = vdupq_n_f32(g10);
    let g11v = vdupq_n_f32(g11);
    let g12v = vdupq_n_f32(g12);

    let xbase = x.as_ptr().add(x_idx);
    let ybase = y.as_mut_ptr().add(y_idx);

    let mut x0v = vld1q_f32(xbase.sub(t + 2));

    let mut i = 0;
    while i + 4 <= n {
        let x4v = vld1q_f32(xbase.add(i).sub(t - 2));

        let x2v = vextq_f32(x0v, x4v, 2);

        let x1v = vextq_f32(x0v, x4v, 1);

        let x3v = vextq_f32(x0v, x4v, 3);

        let xi = vld1q_f32(xbase.add(i));

        let mut yi = xi;
        yi = vfmaq_f32(yi, g10v, x2v);
        yi = vfmaq_f32(yi, g11v, vaddq_f32(x1v, x3v));
        yi = vfmaq_f32(yi, g12v, vaddq_f32(x4v, x0v));
        vst1q_f32(ybase.add(i), yi);

        x0v = x4v;
        i += 4;
    }

    let x0v_arr: [f32; 4] = core::mem::transmute(x0v);
    let mut sx4 = x0v_arr[0];
    let mut sx3 = x0v_arr[1];
    let mut sx2 = x0v_arr[2];
    let mut sx1 = x0v_arr[3];

    while i < n {
        let sx0 = x[x_idx + i - t + 2];
        y[y_idx + i] = x[x_idx + i] + g10 * sx2 + g11 * (sx1 + sx3) + g12 * (sx0 + sx4);
        sx4 = sx3;
        sx3 = sx2;
        sx2 = sx1;
        sx1 = sx0;
        i += 1;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
#[inline(always)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn comb_filter_const_sse(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    use core::arch::x86_64::*;

    let g10v = _mm_set1_ps(g10);
    let g11v = _mm_set1_ps(g11);
    let g12v = _mm_set1_ps(g12);

    let xbase = x.as_ptr().add(x_idx);
    let ybase = y.as_mut_ptr().add(y_idx);
    let mut x0v = _mm_loadu_ps(xbase.sub(t + 2));

    let mut i = 0;
    while i + 4 <= n {
        let x4v = _mm_loadu_ps(xbase.add(i).sub(t - 2));

        let x2v = _mm_shuffle_ps(x0v, x4v, 0x4e);

        let x1v = _mm_shuffle_ps(x0v, x2v, 0x99);

        let x3v = _mm_shuffle_ps(x2v, x4v, 0x99);

        let xi = _mm_loadu_ps(xbase.add(i));

        let mut yi = xi;
        yi = _mm_add_ps(yi, _mm_mul_ps(g10v, x2v));
        let yi2 = _mm_add_ps(
            _mm_mul_ps(g11v, _mm_add_ps(x3v, x1v)),
            _mm_mul_ps(g12v, _mm_add_ps(x4v, x0v)),
        );
        yi = _mm_add_ps(yi, yi2);
        _mm_storeu_ps(ybase.add(i), yi);

        x0v = x4v;
        i += 4;
    }

    let x0v_arr: [f32; 4] = core::mem::transmute(x0v);
    let mut sx4 = x0v_arr[0];
    let mut sx3 = x0v_arr[1];
    let mut sx2 = x0v_arr[2];
    let mut sx1 = x0v_arr[3];

    while i < n {
        let sx0 = x[x_idx + i - t + 2];
        y[y_idx + i] = x[x_idx + i] + g10 * sx2 + g11 * (sx1 + sx3) + g12 * (sx0 + sx4);
        sx4 = sx3;
        sx3 = sx2;
        sx2 = sx1;
        sx1 = sx0;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn comb_filter_const_avx(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    use core::arch::x86_64::*;

    let g10v = _mm256_set1_ps(g10);
    let g11v = _mm256_set1_ps(g11);
    let g12v = _mm256_set1_ps(g12);

    let xbase = x.as_ptr().add(x_idx);
    let ybase = y.as_mut_ptr().add(y_idx);

    let mut i = 0;

    while i + 16 <= n {
        let xi_a = _mm256_loadu_ps(xbase.add(i));
        let x0_a = _mm256_loadu_ps(xbase.add(i).sub(t + 2));
        let x4_a = _mm256_loadu_ps(xbase.add(i).sub(t - 2));

        let x2_a = _mm256_loadu_ps(xbase.add(i).sub(t));
        let x1x3_a = _mm256_add_ps(
            _mm256_loadu_ps(xbase.add(i).sub(t + 1)),
            _mm256_loadu_ps(xbase.add(i).sub(t - 1)),
        );
        let x0x4_a = _mm256_add_ps(x0_a, x4_a);

        let mut yi_a = xi_a;
        yi_a = _mm256_fmadd_ps(g10v, x2_a, yi_a);
        yi_a = _mm256_fmadd_ps(g11v, x1x3_a, yi_a);
        yi_a = _mm256_fmadd_ps(g12v, x0x4_a, yi_a);
        _mm256_storeu_ps(ybase.add(i), yi_a);

        let j = i + 8;
        let xi_b = _mm256_loadu_ps(xbase.add(j));
        let x0_b = _mm256_loadu_ps(xbase.add(j).sub(t + 2));
        let x4_b = _mm256_loadu_ps(xbase.add(j).sub(t - 2));
        let x2_b = _mm256_loadu_ps(xbase.add(j).sub(t));
        let x1x3_b = _mm256_add_ps(
            _mm256_loadu_ps(xbase.add(j).sub(t + 1)),
            _mm256_loadu_ps(xbase.add(j).sub(t - 1)),
        );
        let x0x4_b = _mm256_add_ps(x0_b, x4_b);

        let mut yi_b = xi_b;
        yi_b = _mm256_fmadd_ps(g10v, x2_b, yi_b);
        yi_b = _mm256_fmadd_ps(g11v, x1x3_b, yi_b);
        yi_b = _mm256_fmadd_ps(g12v, x0x4_b, yi_b);
        _mm256_storeu_ps(ybase.add(j), yi_b);

        i += 16;
    }

    while i + 8 <= n {
        let xi = _mm256_loadu_ps(xbase.add(i));
        let x0 = _mm256_loadu_ps(xbase.add(i).sub(t + 2));
        let x4 = _mm256_loadu_ps(xbase.add(i).sub(t - 2));
        let x2 = _mm256_loadu_ps(xbase.add(i).sub(t));
        let x1x3 = _mm256_add_ps(
            _mm256_loadu_ps(xbase.add(i).sub(t + 1)),
            _mm256_loadu_ps(xbase.add(i).sub(t - 1)),
        );
        let x0x4 = _mm256_add_ps(x0, x4);

        let mut yi = xi;
        yi = _mm256_fmadd_ps(g10v, x2, yi);
        yi = _mm256_fmadd_ps(g11v, x1x3, yi);
        yi = _mm256_fmadd_ps(g12v, x0x4, yi);
        _mm256_storeu_ps(ybase.add(i), yi);

        i += 8;
    }

    if i + 4 <= n {
        comb_filter_const_sse_fma(y, x, y_idx + i, x_idx + i, t, n - i, g10, g11, g12);
        return;
    }

    let mut sx4 = x[x_idx + i - t - 2];
    let mut sx3 = x[x_idx + i - t - 1];
    let mut sx2 = x[x_idx + i - t];
    let mut sx1 = x[x_idx + i - t + 1];
    while i < n {
        let sx0 = x[x_idx + i - t + 2];
        y[y_idx + i] = x[x_idx + i] + g10 * sx2 + g11 * (sx1 + sx3) + g12 * (sx0 + sx4);
        sx4 = sx3;
        sx3 = sx2;
        sx2 = sx1;
        sx1 = sx0;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn comb_filter_const_sse_fma(
    y: &mut [f32],
    x: &[f32],
    y_idx: usize,
    x_idx: usize,
    t: usize,
    n: usize,
    g10: f32,
    g11: f32,
    g12: f32,
) {
    use core::arch::x86_64::*;

    let g10v = _mm_set1_ps(g10);
    let g11v = _mm_set1_ps(g11);
    let g12v = _mm_set1_ps(g12);

    let xbase = x.as_ptr().add(x_idx);
    let ybase = y.as_mut_ptr().add(y_idx);
    let mut x0v = _mm_loadu_ps(xbase.sub(t + 2));

    let mut i = 0;
    while i + 4 <= n {
        let x4v = _mm_loadu_ps(xbase.add(i).sub(t - 2));
        let x2v = _mm_shuffle_ps(x0v, x4v, 0x4e);
        let x1v = _mm_shuffle_ps(x0v, x2v, 0x99);
        let x3v = _mm_shuffle_ps(x2v, x4v, 0x99);
        let xi = _mm_loadu_ps(xbase.add(i));

        let mut yi = xi;
        yi = _mm_fmadd_ps(g10v, x2v, yi);
        yi = _mm_fmadd_ps(g11v, _mm_add_ps(x1v, x3v), yi);
        yi = _mm_fmadd_ps(g12v, _mm_add_ps(x0v, x4v), yi);
        _mm_storeu_ps(ybase.add(i), yi);

        x0v = x4v;
        i += 4;
    }

    let x0v_arr: [f32; 4] = core::mem::transmute(x0v);
    let mut sx4 = x0v_arr[0];
    let mut sx3 = x0v_arr[1];
    let mut sx2 = x0v_arr[2];
    let mut sx1 = x0v_arr[3];
    while i < n {
        let sx0 = x[x_idx + i - t + 2];
        y[y_idx + i] = x[x_idx + i] + g10 * sx2 + g11 * (sx1 + sx3) + g12 * (sx0 + sx4);
        sx4 = sx3;
        sx3 = sx2;
        sx2 = sx1;
        sx1 = sx0;
        i += 1;
    }
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
        y[y_idx + i] = x[x_idx + i]
            + (1.0 - f)
                * (g00 * x[x_idx + i - t0]
                    + g01 * (x[x_idx + i - t0 + 1] + x[x_idx + i - t0 - 1])
                    + g02 * (x[x_idx + i - t0 + 2] + x[x_idx + i - t0 - 2]))
            + f * (g10 * x2 + g11 * (x1 + x3) + g12 * (x0 + x4));

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
            comb_filter_const(y, x, y_idx + i, x_idx + i, t1, n - i, g10, g11, g12);
        }
    }
}

/// In-place comb filter: buf[y_idx..y_idx+n] is both input and output.
/// Reference samples at buf[y_idx + i - T + offset] may already be filtered
/// if T < i, matching C libopus's in-place comb_filter(out, out, ...) behavior.
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
) {
    if g0 == 0.0 && g1 == 0.0 {
        // nothing to do; buf[y_idx..] already holds the input
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

    let mut inner_overlap = overlap;
    if g0 == g1 && t0 == t1 && tapset0 == tapset1 {
        inner_overlap = 0;
    }

    let mut i = 0;
    while i < inner_overlap && i < n {
        let idx = y_idx + i;
        let f = window[i] * window[i];
        let s = buf[idx]; // original input (not yet overwritten at idx)
        let r0 = buf[idx - t0];
        let r0p1 = buf[idx - t0 + 1];
        let r0m1 = buf[idx - t0 - 1];
        let r0p2 = buf[idx - t0 + 2];
        let r0m2 = buf[idx - t0 - 2];
        let r1 = buf[idx - t1];
        let r1p1 = buf[idx - t1 + 1];
        let r1m1 = buf[idx - t1 - 1];
        let r1p2 = buf[idx - t1 + 2];
        let r1m2 = buf[idx - t1 - 2];
        buf[idx] = s
            + (1.0 - f) * (g00 * r0 + g01 * (r0p1 + r0m1) + g02 * (r0p2 + r0m2))
            + f * (g10 * r1 + g11 * (r1p1 + r1m1) + g12 * (r1p2 + r1m2));
        i += 1;
    }

    // Constant region: only new filter (t1, g1)
    while i < n {
        let idx = y_idx + i;
        let s = buf[idx];
        let r1 = buf[idx - t1];
        let r1p1 = buf[idx - t1 + 1];
        let r1m1 = buf[idx - t1 - 1];
        let r1p2 = buf[idx - t1 + 2];
        let r1m2 = buf[idx - t1 - 2];
        buf[idx] = s + g10 * r1 + g11 * (r1p1 + r1m1) + g12 * (r1p2 + r1m2);
        i += 1;
    }
}

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
    nb_available_bytes: usize,
    tone_freq: f32,
    toneishness: f32,
    complexity: i32,
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

    let enabled = true;
    let mut pitch_index = COMBFILTER_MINPERIOD;
    let mut gain1 = 0.0f32;
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
            pitch_index = ((0.5 + 2.0 * core::f32::consts::PI * multiple / tone_freq) as usize)
                .min(COMBFILTER_MAXPERIOD - 2);
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
            crate::pitch::pitch_downsample(&pre_slices, pitch_buf, pitch_buf_len, channels, 2);
        }

        let search_max = max_period - 3 * min_period;
        let pitch_result = crate::pitch::pitch_search(
            &pitch_buf[max_period >> 1..],
            pitch_buf,
            frame_size,
            search_max,
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
            );
        }
    }

    let after = &mut after[..channels];
    for c in 0..channels {
        let start = c * buf_stride + overlap;
        after[c] = sum_abs(&in_buf[start..start + frame_size]);
    }

    let cancel_pitch = (0..channels).any(|c| after[c] > before[c]);

    if cancel_pitch {
        for c in 0..channels {
            in_buf[c * buf_stride + overlap..c * buf_stride + overlap + frame_size]
                .copy_from_slice(
                    &pre[c * pre_size + max_period..c * pre_size + max_period + frame_size],
                );
        }

        for c in 0..channels {
            if frame_size >= max_period {
                prefilter_mem[c * max_period..(c + 1) * max_period].copy_from_slice(
                    &pre[c * pre_size + frame_size..c * pre_size + frame_size + max_period],
                );
            } else {
                let shift = max_period - frame_size;
                prefilter_mem.copy_within(
                    c * max_period + frame_size..(c + 1) * max_period,
                    c * max_period,
                );
                prefilter_mem[c * max_period + shift..(c + 1) * max_period].copy_from_slice(
                    &pre[c * pre_size + max_period..c * pre_size + max_period + frame_size],
                );
            }
        }
        return (false, 0.0, pitch_index);
    }

    for c in 0..channels {
        if frame_size >= max_period {
            prefilter_mem[c * max_period..(c + 1) * max_period].copy_from_slice(
                &pre[c * pre_size + frame_size..c * pre_size + frame_size + max_period],
            );
        } else {
            let shift = max_period - frame_size;
            prefilter_mem.copy_within(
                c * max_period + frame_size..(c + 1) * max_period,
                c * max_period,
            );
            prefilter_mem[c * max_period + shift..(c + 1) * max_period].copy_from_slice(
                &pre[c * pre_size + max_period..c * pre_size + max_period + frame_size],
            );
        }
    }

    (pf_on, gain1, pitch_index)
}

/// Padding appended to `w_x` to absorb any SIMD over-shoot past the last band.
/// Sized to `MAX_PVQ_N` (352) — far larger than the widest SIMD access (16),
/// which is always loop-guarded (`while i + width <= n`). (Previously
/// `MAX_PVQ_N * 8`, which had no access pattern justifying it.)
const STRIDE_ACCESS_PAD: usize = crate::pvq::MAX_PVQ_N;

pub struct CeltEncoder {
    mode: &'static CeltMode,
    channels: usize,
    pub complexity: i32,
    syn_mem: FixedVec<f32, CELT_SYN_MEM>,
    enc_decode_mem: FixedVec<f32, CELT_SYN_MEM>,
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
    lsb_depth: i32,
    overlap_max: f32,
    bitrate: i32,
    vbr: bool,
    constrained_vbr: bool,
    vbr_reservoir: i32,
    vbr_drift: i32,
    vbr_offset: i32,
    vbr_count: i32,
    spec_avg: f32,
    stereo_saving: f32,

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
    w_band_amp_synth: FixedVec<f32, CELT_NB_X_CH>,
    consec_transient: i32,

    w_prefilter_pre: FixedVec<f32, CELT_PREFILTER_PRE>,
    w_prefilter_pitch_buf: FixedVec<f32, CELT_PREFILTER_PITCH>,
    w_prefilter_before: FixedVec<f32, CELT_MAX_CHANNELS>,
    w_prefilter_after: FixedVec<f32, CELT_MAX_CHANNELS>,

    w_transient_tmp: FixedVec<f32, MAX_TRANSIENT_LEN>,
    w_transient_tmp2: FixedVec<f32, { MAX_TRANSIENT_LEN / 2 }>,

    analysis: AnalysisInfo,
    loss_rate: i32,
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
    stereo_saving: &mut f32,
    tf_estimate: f32,
    intensity: i32,
    surround_trim: f32,
    equiv_rate: i32,
) -> i32 {
    let mut trim = 5.0f32;
    if equiv_rate < 64000 {
        trim = 4.0;
    } else if equiv_rate < 80000 {
        let frac = (equiv_rate - 64000) as f32 / 1024.0;
        trim = 4.0 + (1.0 / 16.0) * frac;
    }

    if channels == 2 {
        let mut sum = 0.0f32;
        for i in 0..8 {
            let offset = (mode.e_bands[i] as usize) << lm;
            let n = ((mode.e_bands[i + 1] - mode.e_bands[i]) as usize) << lm;
            let mut partial = 0.0f32;
            for j in 0..n {
                partial += x[offset + j] * x[n0 + offset + j];
            }
            sum += partial;
        }
        sum = (sum / 8.0).abs().min(1.0);
        let mut min_xc = sum;
        for i in 8..intensity as usize {
            let offset = (mode.e_bands[i] as usize) << lm;
            let n = ((mode.e_bands[i + 1] - mode.e_bands[i]) as usize) << lm;
            let mut partial = 0.0f32;
            for j in 0..n {
                partial += x[offset + j] * x[n0 + offset + j];
            }
            min_xc = min_xc.min(partial.abs());
        }
        min_xc = min_xc.min(1.0);

        let log_xc = (1.001 - sum * sum).log2();
        let log_xc2 = (log_xc * 0.5).max((1.001 - min_xc * min_xc).log2());

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

// Port of celt_encoder.c tone_lpc / tone_detect (float path, no FIXED_POINT shifts)
fn tone_lpc(x: &[f32], len: usize, delay: usize, lpc: &mut [f32; 2]) -> bool {
    // returns true on fail
    if len <= 2 * delay {
        return true;
    }
    let mut r00 = 0.0f32;
    let mut r01 = 0.0f32;
    let mut r11 = 0.0f32;
    let mut r02 = 0.0f32;
    let mut r12 = 0.0f32;
    let mut r22 = 0.0f32;
    for i in 0..len - 2 * delay {
        r00 += x[i] * x[i];
        r01 += x[i] * x[i + delay];
        r02 += x[i] * x[i + 2 * delay];
    }
    let mut edges = 0.0f32;
    for i in 0..delay {
        edges += x[len + i - 2 * delay] * x[len + i - 2 * delay] - x[i] * x[i];
    }
    r11 = r00 + edges;
    edges = 0.0;
    for i in 0..delay {
        edges += x[len + i - delay] * x[len + i - delay] - x[i + delay] * x[i + delay];
    }
    r22 = r11 + edges;
    edges = 0.0;
    for i in 0..delay {
        edges += x[len + i - 2 * delay] * x[len + i - delay] - x[i] * x[i + delay];
    }
    r12 = r01 + edges;
    // Reverse and sum to get backward contribution (float: simple sums)
    let (rr00, rr01, rr11, rr02, rr12, rr22) = (r00 + r22, r01 + r12, 2.0 * r11, 2.0 * r02, r12 + r01, r00 + r22);
    r00 = rr00;
    r01 = rr01;
    r11 = rr11;
    r02 = rr02;
    r12 = rr12;
    r22 = rr22;
    let den = r00 * r11 - r01 * r01;
    if den < 0.001 * r00 * r11 {
        return true;
    }
    let mut lpc1 = (r02 * r11 - r01 * r12) / den;
    let mut lpc0 = (r00 * r12 - r02 * r01) / den;
    // Clamp as in C (Q29 limits)
    lpc1 = lpc1.clamp(-1.0, 1.0);
    lpc0 = lpc0.clamp(-1.999999, 1.999999);
    // C does HALF check for lpc0, but float clamp suffices
    lpc[0] = lpc0;
    lpc[1] = lpc1;
    false
}

fn tone_detect(in_buf: &[f32], cc: usize, n: usize, fs: i32) -> (f32, f32) {
    // in_buf is at least N samples per channel, we use N = n (frame+overlap)
    // For CC==2, average channels; for float, no shift scaling.
    // Use FixedVec to stay no_std heap-free.
    let mut x: FixedVec<f32, 2048> = FixedVec::from_value(0.0, n);
    if cc == 2 {
        for i in 0..n {
            x[i] = 0.5 * (in_buf[i] + in_buf[i + n]);
        }
    } else {
        for i in 0..n {
            x[i] = in_buf[i];
        }
    }
    let mut delay = 1usize;
    let mut lpc = [0.0f32; 2];
    let mut fail = tone_lpc(&x[..n], n, delay, &mut lpc);
    while delay <= (fs as usize / 3000) && (fail || (lpc[0] > 1.0 && lpc[1] < 0.0)) {
        delay *= 2;
        fail = tone_lpc(&x[..n], n, delay, &mut lpc);
    }
    if !fail && lpc[0] * lpc[0] + 3.999999 * lpc[1] < 0.0 {
        let toneishness = -lpc[1];
        // acos needs libm in no_std
        #[cfg(feature = "std")]
        let freq = (0.5 * lpc[0]).acos() / delay as f32;
        #[cfg(not(feature = "std"))]
        let freq = crate::compat::Math::acos(0.5 * lpc[0]) / delay as f32;
        (freq, toneishness)
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
    cap: &[i32],
    lsb_depth: i32,
    vbr: bool,
    constrained_vbr: bool,
    analysis: &AnalysisInfo,
    surround_dynalloc: &[f32],
    tone_freq: f32,
    toneishness: f32,
    importance: &mut [i32],
) {
    offsets.fill(0);
    if effective_bytes < (30 + 5 * lm) {
        for v in importance.iter_mut() {
            *v = 13;
        }
        return;
    }

    let nb = mode.nb_ebands;
    let mut follower: FixedVec<f32, CELT_NB_X_CH> = FixedVec::from_value(0.0f32, nb * channels);
    for c in 0..channels {
        let base = c * nb;
        let src = if let Some(b2) = band_log_e2 { b2 } else { band_log_e };
        let mut band_log_e3: FixedVec<f32, CELT_NB_EBANDS> = FixedVec::from_value(0.0f32, end);
        for i in 0..end {
            let mut e = src[base + i];
            if lm == 0 && i < 8 {
                e = e.max(old_band_e[base + i]);
            }
            band_log_e3[i] = e;
        }

        let mut last = 0usize;
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
            let log_n = if i < mode.log_n.len() { mode.log_n[i] as f32 } else { 0.0 };
            let e_mean = if i < mode.e_means.len() { mode.e_means[i] } else { 0.0 };
            let noise_floor = 0.0625 * log_n + 0.5 + (9 - lsb_depth) as f32 - e_mean + 0.0062 * ((i + 5) * (i + 5)) as f32;
            follower[base + i] = follower[base + i].max(noise_floor);
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
        importance[i] = (0.5 + 13.0 * (2.0f32).powf(v)).floor() as i32;
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
        let freq_bin = (tone_freq * 120.0 / core::f32::consts::PI + 0.5).floor() as i32;
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
    // effBytes>320 boost (C 1232) — trivial, kept
    if effective_bytes > 320 {
        let add = (1.5f32).min(1e-3 * (effective_bytes as f32 - 320.0));
        if start < end {
            follower[start] += add;
        }
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

        // Cap only for CBR / non-transient constrained VBR (libopus 1254-1257)
        if (!vbr || (constrained_vbr && !is_transient)) {
            let cap_bits = ((2 * effective_bytes as i32) / 3) << (BITRES + 3);
            if tot_boost + boost_bits > cap_bits {
                offsets[i] = ((cap_bits - tot_boost) >> BITRES).max(0);
                break;
            }
        }

        let quanta = (width << BITRES).min((6 << BITRES).max(width));
        let mut boost_count = boost;
        let mut as_bits = boost_count * quanta;
        if as_bits > cap[i] {
            as_bits = cap[i];
            boost_count = as_bits / quanta;
        }

        offsets[i] = boost_count.max(0);
        tot_boost += boost_bits.max(0);
    }
}

impl CeltEncoder {
    pub fn new(mode: &'static CeltMode, channels: usize) -> Self {
        let overlap = mode.overlap;
        let channel_mem_size = 2048 + overlap;
        let syn_mem_size = channels * channel_mem_size;
        let nb_ebands = mode.nb_ebands;
        let nb_x_ch = nb_ebands * channels;
        let frame_x_ch = MAX_FRAME_SIZE * channels;
        let bufstride_x_ch = (MAX_FRAME_SIZE + overlap) * channels;
        Self {
            mode,
            channels,
            complexity: 9,
            syn_mem: FixedVec::from_value(0.0, syn_mem_size),
            enc_decode_mem: FixedVec::from_value(0.0, syn_mem_size),
            old_band_e: FixedVec::from_value(0.0, nb_x_ch),
            preemph_mem: FixedVec::from_value(0.0, channels),
            tonal_average: 256,
            hf_average: 0,
            tapset_decision: 0,
            spread_decision: SPREAD_NORMAL,
            intensity: 0,
            last_coded_bands: 0,
            prefilter_mem: FixedVec::from_value(0.0, channels * COMBFILTER_MAXPERIOD),
            prefilter_period: COMBFILTER_MINPERIOD,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            old_band_e2: FixedVec::from_value(0.0, nb_x_ch),
            old_band_e3: FixedVec::from_value(0.0, nb_x_ch),
            last_band_log_e: FixedVec::from_value(0.0, nb_x_ch),
            delayed_intra: 0.0,
            lsb_depth: 24,
            overlap_max: 0.0,
            bitrate: OPUS_BITRATE_MAX,
            vbr: false,
            constrained_vbr: true,
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,
            spec_avg: 0.0,
            stereo_saving: 0.0,

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
            w_band_amp_synth: FixedVec::from_value(0.0, nb_x_ch),

            w_prefilter_pre: FixedVec::from_value(0.0, channels * (COMBFILTER_MAXPERIOD + MAX_FRAME_SIZE)),
            w_prefilter_pitch_buf: FixedVec::from_value(0.0, (COMBFILTER_MAXPERIOD + MAX_FRAME_SIZE) >> 1),
            w_prefilter_before: FixedVec::from_value(0.0, channels),
            w_prefilter_after: FixedVec::from_value(0.0, channels),
            w_transient_tmp: FixedVec::from_value(0.0, MAX_TRANSIENT_LEN),
            w_transient_tmp2: FixedVec::from_value(0.0, MAX_TRANSIENT_LEN / 2),
            consec_transient: 0,

            analysis: AnalysisInfo::default(),
            loss_rate: 0,
        }
    }

    pub fn encode(&mut self, pcm: &[f32], frame_size: usize, rc: &mut RangeCoder) {
        self.encode_impl(pcm, frame_size, rc, 0, None, false)
    }

    pub fn encode_with_start_band(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        rc: &mut RangeCoder,
        start_band: usize,
    ) {
        self.encode_impl(pcm, frame_size, rc, start_band, None, false)
    }

    pub fn encode_with_budget(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        rc: &mut RangeCoder,
        start_band: usize,
        total_bits: i32,
    ) {
        self.encode_impl(pcm, frame_size, rc, start_band, Some(total_bits), false)
    }

    pub fn encode_with_budget_vbr(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        rc: &mut RangeCoder,
        start_band: usize,
        total_bits: i32,
        is_vbr: bool,
    ) {
        self.encode_impl(
            pcm,
            frame_size,
            rc,
            start_band,
            Some(total_bits),
            is_vbr,
        )
    }

    pub fn set_lsb_depth(&mut self, depth: i32) {
        self.lsb_depth = depth.clamp(8, 24);
    }

    pub fn get_lsb_depth(&self) -> i32 {
        self.lsb_depth
    }
    pub fn set_bitrate(&mut self, bitrate: i32) {
        self.bitrate = bitrate;
    }
    pub fn set_vbr(&mut self, vbr: bool) {
        self.vbr = vbr;
    }
    pub fn set_constrained_vbr(&mut self, cvbr: bool) {
        self.constrained_vbr = cvbr;
    }

    fn encode_impl(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        rc: &mut RangeCoder,
        start_band: usize,
        explicit_total_bits: Option<i32>,
        is_vbr: bool,
    ) {
        let mode = self.mode;
        let channels = self.channels;
        let nb_ebands = mode.nb_ebands;
        let overlap = mode.overlap;

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

        let syn_mem_size = 2048 + overlap;
        for c in 0..channels {
            let channel_offset = c * syn_mem_size;

            self.syn_mem.copy_within(
                channel_offset + frame_size..channel_offset + syn_mem_size,
                channel_offset,
            );

            let mut m = self.preemph_mem[c];
            let coef = mode.preemph[0];
            for i in 0..frame_size {
                let x = pcm[c * frame_size + i] * 32768.0;
                let val = x - m;
                self.syn_mem[channel_offset + syn_mem_size - frame_size + i] = val;
                m = x * coef;
            }
            self.preemph_mem[c] = m;
        }

        let buf_stride = frame_size + overlap;
        let in_buf = &mut self.w_in_buf[..buf_stride * channels];
        for c in 0..channels {
            let channel_offset = c * syn_mem_size;
            let in_buf_offset = c * buf_stride;

            let src_start = syn_mem_size - frame_size - overlap;
            in_buf[in_buf_offset..in_buf_offset + buf_stride].copy_from_slice(
                &self.syn_mem[channel_offset + src_start..channel_offset + syn_mem_size],
            );
        }

 // Tone detection (C 2022): use channel-contiguous in_buf with N+overlap, float path
 let mut tone_freq = -1.0f32;
 let mut toneishness = 0.0f32;
 {
 let (f, t) = tone_detect(&*in_buf, channels, buf_stride, mode.fs);
 tone_freq = f;
 toneishness = t;
 }
 let mut tf_estimate = 0.0f32;
 let mut tf_chan = 0;
 let mut weak_transient = false;

 let is_transient = if self.complexity >= 1 {
 transient_analysis(
 in_buf,
 buf_stride,
 channels,
 &mut tf_estimate,
 &mut tf_chan,
 false,
 &mut weak_transient,
 tone_freq,
 toneishness,
 &mut self.w_transient_tmp,
 &mut self.w_transient_tmp2,
 )
 } else {
 false
 };
 // Clamp toneishness as in C 2034: toneishness = min(toneishness, 1 - tf_estimate)
 // For float, tf_estimate is log-domain; clamp to [0,1]
 toneishness = toneishness.min((1.0 - tf_estimate).clamp(0.0, 1.0));

 // Check for pure tone: if tonality is very high, bypass pitch search
 // Keep original analysis.tonality check for prefilter gating, but dynalloc uses tone_detect result
 let analysis_toneishness = if self.analysis.valid {
 self.analysis.tonality
 } else {
 0.0
 };
 // analysis_toneishness kept for prefilter gating; tone_freq/toneishness from tone_detect used for dynalloc
 let pf_enabled =
 start_band == 0 && self.complexity >= 5 && analysis_toneishness < 0.99 && channels == 1;
        let (pf_on, gain1, pitch_index) = if pf_enabled {
            let tell = rc.tell();
            let nb_filled = ((tell + 4) >> 3).max(0) as usize;
            let total = explicit_total_bits.unwrap_or_else(|| (rc.buf.len() * 8) as i32);
            let nb_available = ((total / 8) as usize).saturating_sub(nb_filled);
            run_prefilter(
                in_buf,
                &mut self.prefilter_mem,
                self.prefilter_period,
                self.prefilter_gain,
                self.prefilter_tapset,
                self.tapset_decision,
                mode.window,
                channels,
                frame_size,
                overlap,
                &mut self.w_prefilter_pre,
                &mut self.w_prefilter_pitch_buf,
                &mut self.w_prefilter_before,
                &mut self.w_prefilter_after,
                &self.analysis,
                self.loss_rate,
                tf_estimate,
                nb_available,
                tone_freq,
                toneishness,
                self.complexity,
            )
        } else {
            (false, 0.0f32, COMBFILTER_MINPERIOD)
        };

        // Save the prefiltered overlap for the next frame.
        // In libopus, st->in_mem stores the overlap separately and run_prefilter
        // copies it to/from in[]. Here we emulate that by updating syn_mem with
        // the last overlap samples of in_buf (which were prefiltered in place).
        let syn_mem_size = 2048 + overlap;
        for c in 0..channels {
            let channel_offset = c * syn_mem_size;
            let in_buf_offset = c * buf_stride;
            self.syn_mem[channel_offset + syn_mem_size - overlap..channel_offset + syn_mem_size]
                .copy_from_slice(&in_buf[in_buf_offset + frame_size..in_buf_offset + buf_stride]);
        }
        let freq = &mut self.w_freq[..frame_size * channels];
        let band_e = &mut self.w_band_e[..nb_ebands * channels];
        let band_log_e = &mut self.w_band_log_e[..nb_ebands * channels];
        let x_pad_end = (frame_size * channels + STRIDE_ACCESS_PAD).min(self.w_x.len());
        let x = &mut self.w_x[..x_pad_end];

        let mut total_bits = explicit_total_bits.unwrap_or_else(|| (rc.buf.len() * 8) as i32);
        self.w_error[..nb_ebands * channels].fill(0.0);
        let error = &mut self.w_error[..nb_ebands * channels];

        // --- Silence detection (port of libopus celt_encoder.c: sample_max / overlap_max / lsb_depth) ---
        let mut sample_max = self.overlap_max;
        let n_nonoverlap = frame_size.saturating_sub(overlap);
        for c in 0..channels {
            let base = c * frame_size;
            for i in 0..n_nonoverlap {
                let v = pcm[base + i].abs();
                if v > sample_max {
                    sample_max = v;
                }
            }
        }
        let mut new_overlap_max = 0.0f32;
        for c in 0..channels {
            let base = c * frame_size;
            for i in n_nonoverlap..frame_size {
                let v = pcm[base + i].abs();
                if v > new_overlap_max {
                    new_overlap_max = v;
                }
            }
        }
        self.overlap_max = new_overlap_max;
        if new_overlap_max > sample_max {
            sample_max = new_overlap_max;
        }
        let threshold = 1.0 / ((1 << self.lsb_depth) as f32);
        let mut silence = sample_max <= threshold;
        if frame_size == 960 && channels == 2 {
        }
        let tell_initial = rc.tell();
        let tell_initial_frac = rc.tell_frac();
        let nb_filled_bytes_initial = ((tell_initial + 4) >> 3).max(0) as usize;
        let mut nb_compressed_bytes = (total_bits / 8) as usize;
        if tell_initial == 1 {
            rc.encode_bit_logp(silence, 15);
        } else {
            silence = false;
        }
        if silence {
            if is_vbr {
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
                    core::cmp::max(if tell_initial == 1 { 2 } else { 0 }, (vbr_rate + vbr_bound - self.vbr_reservoir) >> (BITRES + 3)),
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

        let mut short_blocks = false;
        if lm > 0 && rc.tell() + 3 <= total_bits {
            rc.encode_bit_logp(is_transient, 3);
            if is_transient {
                short_blocks = true;
            }
        }
        let second_mdct = short_blocks && self.complexity >= 8;
        if second_mdct {
            for c in 0..channels {
                let c_buf = c * buf_stride;
                let c_freq = c * frame_size;
                mode.mdct.forward(
                    &in_buf[c_buf..],
                    &mut freq[c_freq..],
                    mode.window,
                    overlap,
                    mode.max_lm - lm,
                    1,
                );
            }
            let band_e2 = &mut self.w_band_e2[..nb_ebands * channels];
            compute_band_energies(mode, freq, band_e2, nb_ebands, channels, lm);
            let band_log_e2 = &mut self.w_band_log_e2[..nb_ebands * channels];
            crate::bands::amp2log2(mode, 0, nb_ebands, band_e2, band_log_e2, channels);
            for c in 0..channels {
                for i in 0..nb_ebands {
                    band_log_e2[c * nb_ebands + i] += lm as f32 * 0.5;
                }
            }
        }
        if short_blocks {
            let b = 1 << lm;
            let n = frame_size / b;
            for c in 0..channels {
                let c_offset = c * buf_stride;
                for i in 0..b {
                    mode.mdct.forward(
                        &in_buf[c_offset + i * n..c_offset + buf_stride],
                        &mut freq[c * frame_size + i..],
                        mode.window,
                        overlap,
                        mode.max_lm,
                        b,
                    );
                }
            }

            compute_band_energies(mode, freq, band_e, nb_ebands, channels, lm);
            normalise_bands(
                mode,
                freq,
                x,
                band_e,
                nb_ebands,
                channels,
                (1 << lm) as usize,
            );
            crate::bands::amp2log2(mode, start_band, nb_ebands, band_e, band_log_e, channels);
        } else {
            // Long MDCT (non-transient)
            for c in 0..channels {
                let c_buf = c * buf_stride;
                let c_freq = c * frame_size;
                mode.mdct.forward(
                    &in_buf[c_buf..],
                    &mut freq[c_freq..],
                    mode.window,
                    overlap,
                    mode.max_lm - lm,
                    1,
                );
            }
            compute_band_energies(mode, freq, band_e, nb_ebands, channels, lm);
            normalise_bands(
                mode,
                freq,
                x,
                band_e,
                nb_ebands,
                channels,
                (1 << lm) as usize,
            );
            crate::bands::amp2log2(mode, start_band, nb_ebands, band_e, band_log_e, channels);
        }

        let intra_ener = if self.complexity >= 4 {
            false
        } else {
            self.old_band_e[..nb_ebands * channels]
                .iter()
                .all(|&e| e <= -27.0)
        };
        let _ = intra_ener; // C st->force_intra is used instead (CTL flag, 0 by default)
        quant_coarse_energy_advanced(
            mode,
            start_band,
            nb_ebands,
            nb_ebands,
            band_log_e,
            &mut self.old_band_e,
            total_bits as u32,
            error,
            rc,
            channels,
            lm,
            (total_bits / 8) as usize,
            false, // force_intra: C st->force_intra (CTL flag, 0 by default)
            &mut self.delayed_intra,
            self.complexity >= 4,
            0,
            false,
        );
        self.w_tf_res[..nb_ebands].fill(0);
        let tf_res = &mut self.w_tf_res[..nb_ebands];
        let effective_bytes = ((total_bits / 8) as usize).max(1);
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

        let band_log_e2_opt = if second_mdct {
            Some(&self.w_band_log_e2[..nb_ebands * channels] as &[f32])
        } else {
            None
        };
        let surround_dynalloc = [0.0f32; 21];
        let mut importance = [13i32; MAX_NB_EBANDS];
        dynalloc_analysis_simple(
            mode,
            band_log_e,
            band_log_e2_opt,
            &self.old_band_e,
            start_band,
            nb_ebands,
            channels,
            lm,
            effective_bytes,
            is_transient,
            offsets,
            cap,
            self.lsb_depth,
            self.vbr,
            self.constrained_vbr,
            &self.analysis,
            &surround_dynalloc,
            tone_freq,
            toneishness,
            &mut importance,
        );

        let tf_select = if self.complexity >= 2 && effective_bytes >= 15 * channels {
            tf_analysis(
                mode,
                nb_ebands,
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
        } else {
            0
        };
        tf_encode(
            start_band,
            nb_ebands,
            is_transient,
            tf_res,
            lm as i32,
            tf_select,
            rc,
        );

        let mut dual_stereo_val = if channels == 2 {
            stereo_analysis(mode, x, lm as i32, frame_size) as i32
        } else {
            0
        };

        let mut stereo_saving = 0.0f32;
        let equiv_rate = (total_bits * 48000) / frame_size as i32;
        if channels == 2 {
            self.intensity = hysteresis_decision(
                equiv_rate / 1000,
                &INTEN_THRESHOLDS,
                &INTEN_HYSTERESIS,
                self.intensity,
            );
            self.intensity = self.intensity.clamp(0, nb_ebands as i32);
        }

        if self.complexity == 0 {
            self.spread_decision = SPREAD_NONE;
            if rc.tell() + 4 <= total_bits {
                rc.encode_icdf(self.spread_decision, &SPREAD_ICDF, 5);
            }
        } else if rc.tell() + 4 <= total_bits {
            if is_transient || self.complexity < 3 || effective_bytes < 10 * channels {
                self.spread_decision = SPREAD_NORMAL;
            } else {
                let update_hf = lm == mode.max_lm;
                let spread_weights = [32i32; 21];
                self.spread_decision = spreading_decision(
                    mode,
                    x,
                    &mut self.tonal_average,
                    self.spread_decision,
                    &mut self.hf_average,
                    &mut self.tapset_decision,
                    update_hf,
                    nb_ebands,
                    channels,
                    (1 << lm) as usize,
                    &spread_weights,
                );
            }
            rc.encode_icdf(self.spread_decision, &SPREAD_ICDF, 5);
        } else {
            self.spread_decision = SPREAD_NORMAL;
        }

        let mut dynalloc_logp = 6i32;
        let total_bits_bitres = total_bits << BITRES;
        let mut total_boost = 0i32;
        let mut tell_frac = rc.tell_frac();

        for i in start_band..nb_ebands {
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

        let alloc_trim = alloc_trim_analysis(
            mode,
            x,
            band_log_e,
            nb_ebands,
            lm as i32,
            channels,
            frame_size,
            &mut stereo_saving,
            tf_estimate,
            self.intensity,
            0.0,
            equiv_rate,
        );
        if rc.tell_frac() + (6 << BITRES) <= total_bits_bitres - total_boost {
            rc.encode_icdf(alloc_trim, &TRIM_ICDF, 7);
        }
        // Second VBR: compute target via simplified compute_vbr (libopus 2436-2534)
        // This makes packet size variable for VBR (beyond silence) while keeping reservoir in sync
        if self.vbr && self.bitrate != OPUS_BITRATE_MAX {
            let vbr_rate = bitrate_to_bits(self.bitrate, mode.fs, frame_size as i32) << BITRES;
            let hybrid = start_band != 0;
            let lm_diff = mode.max_lm as i32 - lm as i32;
            let mut base_target = if !hybrid {
                vbr_rate - ((40 * channels as i32 + 20) << BITRES)
            } else {
                (0).max(vbr_rate - ((9 * channels as i32 + 4) << BITRES))
            };
            if self.constrained_vbr {
                base_target += self.vbr_offset >> lm_diff;
            }
            let tot_boost = total_boost;
            let tf_calib = 0; // simplified
            let cur_tell_frac = rc.tell_frac();
            // Use the exact initial tell_frac captured at entry (libopus tell0_frac),
            // not tell_initial<<BITRES which loses the fractional part.
            let min_allowed = {
                let a = ((cur_tell_frac + tot_boost + (1 << (BITRES + 3)) - 1) >> (BITRES + 3)) + 2;
                if hybrid {
                    let b = ((tell_initial_frac + (37 << BITRES) + tot_boost + (1 << (BITRES + 3)) - 1) >> (BITRES + 3));
                    a.max(b)
                } else {
                    a
                }
            };
            // nbCompressedBytes is current max (after first VBR bound), effectiveBytes ~ vbr_rate>>(3+BITRES) for lambda already
            let max_depth = compute_max_depth(mode, band_log_e, nb_ebands, channels, nb_ebands, self.lsb_depth);
            let mut target = if !hybrid {
                compute_vbr(
                    mode,
                    &self.analysis,
                    base_target,
                    lm as i32,
                    self.bitrate,
                    self.last_coded_bands,
                    channels as i32,
                    self.intensity,
                    self.constrained_vbr,
                    stereo_saving,
                    tot_boost,
                    tf_estimate,
                    0, // pitch_change stub
                    max_depth,
                    false,
                    false,
                    0.0,
                    0.0, // temporal_vbr stub
                )
            } else {
                let mut t = base_target;
                t += ((tf_estimate - 0.25) * 50.0 * (1 << BITRES) as f32) as i32;
                if tf_estimate > 0.7 {
                    t = t.max(50 << BITRES);
                }
                t
            };
            target += cur_tell_frac;
            let mut nb_available = ((target + (1 << (BITRES + 2))) >> (BITRES + 3)) as usize;
            nb_available = nb_available.max(min_allowed as usize);
            nb_available = nb_available.min(nb_compressed_bytes);
            let mut delta = target - vbr_rate;
            target = (nb_available as i32) << (BITRES + 3);
            if silence {
                nb_available = 2;
                target = 2 * 8 << BITRES;
                delta = 0;
            }
            // Reservoir / drift update (libopus 2502-2529)
            if self.vbr_count < 970 {
                self.vbr_count += 1;
            }
            let alpha = if self.vbr_count < 970 {
                // celt_rcp((vbr_count+20)<<16) in Q15 is 32768/(vbr_count+20)
                let v = self.vbr_count + 20;
                (32768 / v) as i32 // Q15
            } else {
                33 // QCONST16(0.001,15) ~33
            };
            if self.constrained_vbr {
                self.vbr_reservoir += target - vbr_rate;
                // drift: MULT16_32_Q15(alpha, delta*(1<<lm_diff) - offset - drift) (libopus 2516)
                let delta_scaled = delta * (1 << lm_diff);
                let delta_minus = delta_scaled - self.vbr_offset - self.vbr_drift;
                self.vbr_drift += ((alpha as i64 * delta_minus as i64) >> 15) as i32;
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

        let mut intensity = self.intensity;
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

        self.last_coded_bands = clt_compute_allocation(
            mode,
            start_band,
            nb_ebands,
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
            true,
            0,
            nb_ebands as i32 - 1,
        );

        quant_fine_energy(
            mode,
            start_band,
            nb_ebands,
            &mut self.old_band_e,
            error,
            ebits,
            rc,
            channels,
        );

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
            nb_ebands,
            x_split,
            y_opt,
            collapse_masks,
            band_e,
            pulses,
            short_blocks,
            self.spread_decision,
            &mut dual_stereo,
            intensity as usize,
            tf_res,
            (total_bits << BITRES) - anti_collapse_rsv,
            &mut balance,
            rc,
            lm as i32,
            self.last_coded_bands,
            resynth,
            false,
            &mut 0u32,
        );

        if anti_collapse_rsv > 0 {
            let anti_collapse_on = if self.consec_transient < 2 {
                1u32
            } else {
                0u32
            };
            rc.enc_bits(anti_collapse_on, 1);
        }

        quant_energy_finalise(
            mode,
            start_band,
            nb_ebands,
            &mut self.old_band_e,
            error,
            ebits,
            fine_priority,
            total_bits - rc.tell(),
            rc,
            channels,
        );

        if silence {
            for v in self.old_band_e[..channels * nb_ebands].iter_mut() {
                *v = -28.0;
            }
        }

        if resynth {
            let band_amp_synth = &mut self.w_band_amp_synth[..nb_ebands * channels];
            log2amp(mode, nb_ebands, band_amp_synth, &self.old_band_e, channels);
            // `w_freq` is no longer needed after analysis/quant; reuse it as the
            // synthesis scratch (avoids a separate `w_freq_synth` buffer).
            self.w_freq[..frame_size * channels].fill(0.0);
            let freq_synth = &mut self.w_freq[..frame_size * channels];
            denormalise_bands(
                mode,
                x,
                freq_synth,
                band_amp_synth,
                start_band,
                nb_ebands,
                channels,
                (1 << lm) as usize,
            );
            let (syn_shift, syn_b) = if is_transient {
                (mode.max_lm, 1 << lm)
            } else {
                (mode.max_lm - lm, 1)
            };
            let syn_n = frame_size / syn_b;
            let decode_buf_size = 2048;

            for c in 0..channels {
                let co = c * syn_mem_size;
                self.enc_decode_mem
                    .copy_within(co + frame_size..co + decode_buf_size + overlap, co);
            }

            for c in 0..channels {
                let co = c * syn_mem_size;
                let out_syn_idx = decode_buf_size - frame_size;
                for bi in 0..syn_b {
                    let syn_stride = if is_transient {
                        mode.short_mdct_size
                    } else {
                        syn_n
                    };
                    mode.mdct.backward(
                        &freq_synth[c * frame_size + bi..],
                        &mut self.enc_decode_mem[co + out_syn_idx + bi * syn_stride..],
                        mode.window,
                        overlap,
                        syn_shift,
                        syn_b,
                    );
                }
            }
        }

        self.last_band_log_e.copy_from_slice(&self.old_band_e);

        if !is_transient {
            self.old_band_e3.copy_from_slice(&self.old_band_e2);
            self.old_band_e2.copy_from_slice(&self.old_band_e);
        } else {
            for i in 0..channels * nb_ebands {
                self.old_band_e2[i] = self.old_band_e2[i].min(self.old_band_e[i]);
            }
        }

        rc.pad_to_bits(total_bits);

        if pf_on {
            self.prefilter_period = pitch_index;
            self.prefilter_gain = gain1;
            self.prefilter_tapset = self.tapset_decision;
        } else {
            self.prefilter_period = COMBFILTER_MINPERIOD;
            self.prefilter_gain = 0.0;
            self.prefilter_tapset = self.tapset_decision;
        }

        if is_transient {
            self.consec_transient += 1;
        } else {
            self.consec_transient = 0;
        }
    }
}

pub struct CeltDecoder {
    mode: &'static CeltMode,
    channels: usize,
    /// Decimation factor for non-48kHz API rates (1=48k, 2=24k, 3=16k, 4=12k, 6=8k).
    /// Mirrors libopus `CELTDecoder.downsample`.
    downsample: usize,
    decode_mem: FixedVec<f32, CELT_DECODE_MEM>,
    old_band_e: FixedVec<f32, CELT_NB_X_CH>,
    preemph_mem: FixedVec<f32, CELT_MAX_CHANNELS>,
    prefilter_mem: FixedVec<f32, CELT_PREFILTER_MEM>,
    prefilter_period: usize,
    prefilter_period_old: usize,
    prefilter_gain: f32,
    prefilter_gain_old: f32,
    prefilter_tapset: i32,
    prefilter_tapset_old: i32,
    old_band_e2: FixedVec<f32, CELT_NB_X_CH>,
    old_band_e3: FixedVec<f32, CELT_NB_X_CH>,
    rng: u32,

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
    w_pcm_frame: FixedVec<f32, DECODE_BUFFER_SIZE>,
    w_post: FixedVec<f32, { DECODE_BUFFER_SIZE + COMBFILTER_MAXPERIOD }>,
}

impl CeltDecoder {
    /// Create a CELT decoder. `sampling_rate` is the API sampling rate
    /// (8000–48000); the decoder always operates at 48 kHz internally and
    /// decimates by `resampling_factor(sampling_rate)` on output.
    pub fn new(mode: &'static CeltMode, channels: usize, sampling_rate: i32) -> Self {
        let overlap = mode.overlap;
        let nb_ebands = mode.nb_ebands;
        let nb_x_ch = nb_ebands * channels;
        let dec_frame_x_ch = DECODE_BUFFER_SIZE * channels;
        Self {
            mode,
            channels,
            downsample: resampling_factor(sampling_rate),
            decode_mem: FixedVec::from_value(0.0, channels * (DECODE_BUFFER_SIZE + overlap)),
            old_band_e: FixedVec::from_value(0.0, nb_x_ch),
            preemph_mem: FixedVec::from_value(0.0, channels),
            prefilter_mem: FixedVec::from_value(0.0, channels * COMBFILTER_MAXPERIOD),
            prefilter_period: COMBFILTER_MINPERIOD,
            prefilter_period_old: COMBFILTER_MINPERIOD,
            prefilter_gain: 0.0,
            prefilter_gain_old: 0.0,
            prefilter_tapset: 0,
            prefilter_tapset_old: 0,
            old_band_e2: FixedVec::from_value(0.0, nb_x_ch),
            old_band_e3: FixedVec::from_value(0.0, nb_x_ch),
            rng: 0,

            w_tf_res: FixedVec::from_value(0, nb_ebands),
            w_cap: FixedVec::from_value(0, nb_ebands),
            w_offsets: FixedVec::from_value(0, nb_ebands),
            w_pulses: FixedVec::from_value(0, nb_ebands),
            w_ebits: FixedVec::from_value(0, nb_x_ch),
            w_fine_priority: FixedVec::from_value(0, nb_x_ch),

            w_x: FixedVec::from_value(0.0, dec_frame_x_ch + STRIDE_ACCESS_PAD),
            w_collapse_masks: FixedVec::from_value(0, nb_x_ch),
            w_freq: FixedVec::from_value(0.0, dec_frame_x_ch + 4), // +4: NEON backward pre-rotation reads up to 3 elements past n2
            w_band_amp: FixedVec::from_value(0.0, nb_x_ch),
            w_pcm_frame: FixedVec::from_value(0.0, DECODE_BUFFER_SIZE),
            w_post: FixedVec::from_value(0.0, DECODE_BUFFER_SIZE + COMBFILTER_MAXPERIOD),
        }
    }

    pub fn decode(&mut self, compressed: &[u8], frame_size: usize, pcm: &mut [f32]) -> usize {
        self.decode_impl(compressed, frame_size, pcm, 0, self.mode.nb_ebands)
    }

    /// Reset all decoder state (equivalent to libopus `OPUS_RESET_STATE`).
    /// Used at SILK↔CELT mode transitions to avoid cross-mode artifacts.
    pub fn reset_state(&mut self) {
        self.decode_mem.fill(0.0);
        self.old_band_e.fill(0.0);
        self.preemph_mem.fill(0.0);
        self.prefilter_mem.fill(0.0);
        self.prefilter_period = COMBFILTER_MINPERIOD;
        self.prefilter_period_old = COMBFILTER_MINPERIOD;
        self.prefilter_gain = 0.0;
        self.prefilter_gain_old = 0.0;
        self.prefilter_tapset = 0;
        self.prefilter_tapset_old = 0;
        self.old_band_e2.fill(0.0);
        self.old_band_e3.fill(0.0);
        self.rng = 0;
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
        self.decode_impl_from_rc(rc, total_bits, frame_size, pcm, start_band, end_band)
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
        self.decode_impl_from_rc(&mut rc, total_bits, frame_size, pcm, start_band, end_band)
    }

    fn decode_impl_from_rc(
        &mut self,
        rc: &mut RangeCoder,
        total_bits: i32,
        frame_size: usize,
        pcm: &mut [f32],
        start_band: usize,
        end_band: usize,
    ) -> usize {
        let mode = self.mode;
        let channels = self.channels;
        let nb_ebands = mode.nb_ebands;
        let end_band = end_band.min(nb_ebands).max(start_band);
        let overlap = mode.overlap;

        // The API frame_size is in output samples. Internally CELT always
        // decodes at 48 kHz, so upscale by the downsample factor (libopus
        // celt_decoder.c:1196: `frame_size *= st->downsample`).
        let api_frame_size = frame_size;
        let frame_size = frame_size * self.downsample;

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
        if start_band != 0 {
            self.prefilter_gain = 0.0;
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

        if frame_size > DECODE_BUFFER_SIZE + overlap {
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
            false,
            &mut self.rng,
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
        self.w_freq[..frame_size * channels].fill(0.0);
        let freq = &mut self.w_freq[..frame_size * channels];
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

        let (shift, b) = if short_blocks {
            (mode.max_lm, 1 << lm)
        } else {
            (mode.max_lm - lm, 1)
        };
        let n = frame_size / b;

        for c in 0..channels {
            let channel_mem_offset = c * (DECODE_BUFFER_SIZE + overlap);

            let mem_size = DECODE_BUFFER_SIZE + overlap;
            self.decode_mem.copy_within(
                channel_mem_offset + frame_size..channel_mem_offset + mem_size,
                channel_mem_offset,
            );

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

            const SIG_SAT: f32 = 536870911.0;
            for i in 0..frame_size {
                let v = &mut self.decode_mem[channel_mem_offset + out_syn_idx + i];
                *v = v.clamp(-SIG_SAT, SIG_SAT);
            }

            self.w_pcm_frame[..frame_size].fill(0.0);
            let pcm_frame = &mut self.w_pcm_frame[..frame_size];

            pcm_frame.copy_from_slice(
                &self.decode_mem[channel_mem_offset + out_syn_idx
                    ..channel_mem_offset + out_syn_idx + frame_size],
            );
            if pf_on || self.prefilter_gain > 0.0 || self.prefilter_gain_old > 0.0 {
                // Set up w_post = [prefilter_mem | pcm_frame] for history access.
                // We apply combfilter in-place on w_post[COMBFILTER_MAXPERIOD..] so that
                // later samples can reference already-filtered earlier samples, matching C's
                // in-place comb_filter behavior.
                self.w_post[..COMBFILTER_MAXPERIOD].copy_from_slice(
                    &self.prefilter_mem[c * COMBFILTER_MAXPERIOD..(c + 1) * COMBFILTER_MAXPERIOD],
                );
                self.w_post[COMBFILTER_MAXPERIOD..COMBFILTER_MAXPERIOD + frame_size]
                    .copy_from_slice(pcm_frame);

                let short_n = mode.short_mdct_size;
                // Call 1: first short_n samples, transition old→current params
                // Apply in-place on w_post[COMBFILTER_MAXPERIOD..], output overwrites input
                comb_filter_inplace(
                    &mut self.w_post,
                    COMBFILTER_MAXPERIOD,
                    self.prefilter_period_old,
                    self.prefilter_period,
                    short_n,
                    self.prefilter_gain_old,
                    self.prefilter_gain,
                    self.prefilter_tapset_old,
                    self.prefilter_tapset,
                    mode.window,
                    overlap,
                );
                if lm != 0 {
                    // Call 2: remaining N-short_n samples, transition current→new params
                    comb_filter_inplace(
                        &mut self.w_post,
                        COMBFILTER_MAXPERIOD + short_n,
                        self.prefilter_period,
                        pitch_index,
                        frame_size - short_n,
                        self.prefilter_gain,
                        gain1,
                        self.prefilter_tapset,
                        prefilter_tapset as i32,
                        mode.window,
                        overlap,
                    );
                }

                pcm_frame.copy_from_slice(
                    &self.w_post[COMBFILTER_MAXPERIOD..COMBFILTER_MAXPERIOD + frame_size],
                );

                self.decode_mem[channel_mem_offset + out_syn_idx
                    ..channel_mem_offset + out_syn_idx + frame_size]
                    .copy_from_slice(pcm_frame);
            }
            let mut new_mem = [0.0f32; COMBFILTER_MAXPERIOD];
            if frame_size >= COMBFILTER_MAXPERIOD {
                new_mem.copy_from_slice(&pcm_frame[frame_size - COMBFILTER_MAXPERIOD..frame_size]);
            } else {
                new_mem[..COMBFILTER_MAXPERIOD - frame_size].copy_from_slice(
                    &self.prefilter_mem
                        [c * COMBFILTER_MAXPERIOD + frame_size..(c + 1) * COMBFILTER_MAXPERIOD],
                );
                new_mem[COMBFILTER_MAXPERIOD - frame_size..].copy_from_slice(pcm_frame);
            }
            self.prefilter_mem[c * COMBFILTER_MAXPERIOD..(c + 1) * COMBFILTER_MAXPERIOD]
                .copy_from_slice(&new_mem);

            let coef = mode.preemph[0];
            let mut m = self.preemph_mem[c];
            const VERY_SMALL: f32 = 1e-30f32;
            let ds = self.downsample;
            if ds == 1 {
                for i in 0..frame_size {
                    let x = pcm_frame[i];
                    let val = (x + VERY_SMALL + m).clamp(-SIG_SAT, SIG_SAT);
                    pcm[c * api_frame_size + i] = val * (1.0 / 32768.0);
                    m = val * coef;
                }
            } else {
                // Run deemphasis IIR over all internal samples, but only write
                // every downsample-th sample (libopus deemphasis() stride).
                for i in 0..frame_size {
                    let x = pcm_frame[i];
                    let val = (x + VERY_SMALL + m).clamp(-SIG_SAT, SIG_SAT);
                    if i % ds == 0 {
                        pcm[c * api_frame_size + i / ds] = val * (1.0 / 32768.0);
                    }
                    m = val * coef;
                }
            }
            self.preemph_mem[c] = m;
        }

        self.prefilter_period_old = self.prefilter_period;
        self.prefilter_gain_old = self.prefilter_gain;
        self.prefilter_tapset_old = self.prefilter_tapset;

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

        if !is_transient {
            self.old_band_e3.copy_from_slice(&self.old_band_e2);
            self.old_band_e2.copy_from_slice(&self.old_band_e);
        } else {
            let nb_ebands = mode.nb_ebands;
            for i in 0..channels * nb_ebands {
                self.old_band_e2[i] = self.old_band_e2[i].min(self.old_band_e[i]);
            }
        }

        self.rng = rc.rng;

        api_frame_size
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::{modes, range_coder::RangeCoder};

    // Regression test: directly drive CeltEncoder with an invalid frame_size=48,
    // bypassing the OpusEncoder::encode() validation layer.
    //
    // This reproduces the crash that was reported against opus-rs 0.1.19 when
    // G.729-decoded PCM (8 kHz) reached the 48 kHz Opus encoder without correct
    // resampling, producing a 48-sample frame instead of 480.
    //
    // Root cause: the lm-search in encode_impl finds no valid match for frame_size=48
    // (valid sizes are 120, 240, 480, 960) and silently falls back to lm=0.
    // With lm=0 and shift=max_lm=3: n=1920>>3=240, n2=120, overlap2=60.
    // The in_buf slice has only frame_size+overlap=168 elements, but forward()
    // requires input.len() >= n2+overlap2 = 180, so it panics immediately.
    // In opus-rs 0.1.19 this assertion was absent and the crash reached the MDCT
    // output write: "index out of bounds: the len is 48 but the index is 119".
    //
    // Either way: the call panics, confirming the crash path is real.
    // The fix in OpusEncoder::encode() returns Err before reaching CeltEncoder.
    #[test]
    #[should_panic]
    fn test_celt_frame_size_48_panics_confirms_crash_path() {
        let mode = modes::default_mode();
        let mut enc = CeltEncoder::new(mode, 1);
        // frame_size=48: lm-search fails, falls back to lm=0.
        // forward() will panic — either on the input-size assertion (0.1.21+) or
        // on the output write (0.1.19): "len is 48 but the index is 119".
        let pcm = vec![0.0f32; 48 + mode.overlap]; // supply ≥ frame_size samples
        let mut rc = RangeCoder::new_encoder(100);
        enc.encode_with_budget(&pcm, 48, &mut rc, 0, 800);
    }
}
