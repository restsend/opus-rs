use crate::bands::{
    SPREAD_NORMAL, compute_band_energies, denormalise_bands, haar1, log2amp, normalise_bands,
    quant_all_bands, spreading_decision,
};
use crate::modes::{CeltMode, SPREAD_ICDF, TAPSET_ICDF, TF_SELECT_TABLE, TRIM_ICDF};
use crate::quant_bands::{
    quant_coarse_energy, quant_energy_finalise, quant_fine_energy, unquant_coarse_energy,
    unquant_energy_finalise, unquant_fine_energy,
};
use crate::range_coder::RangeCoder;
use crate::rate::{BITRES, clt_compute_allocation};

const INV_TABLE: [u8; 128] = [
    255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18, 17,
    16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9, 8, 8, 8,
    8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
];

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
) -> bool {
    let mut mask_metric = 0.0f32;
    let mut forward_decay = 0.0625f32;

    *weak_transient = false;
    if allow_weak_transients {
        forward_decay = 0.03125f32;
    }

    let len2 = len / 2;
    let mut tmp = vec![0.0f32; len];

    for c in 0..channels {
        let mut mem0 = 0.0f32;
        let mut mem1 = 0.0f32;

        // High-pass filter
        for i in 0..len {
            let x = input[c * len + i];
            let y = mem0 + x;
            let mem00 = mem0;
            mem0 = mem0 - x + 0.5 * mem1;
            mem1 = x - mem00;
            tmp[i] = y;
        }

        // First 12 samples are bad
        for i in 0..12 {
            tmp[i] = 0.0;
        }

        let mut mean = 0.0f32;
        mem0 = 0.0f32;
        let mut tmp2 = vec![0.0f32; len2];
        for i in 0..len2 {
            let x2 = (tmp[2 * i] * tmp[2 * i] + tmp[2 * i + 1] * tmp[2 * i + 1]) / 16.0;
            mean += x2 / 4096.0;
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
        let norm = (len2 as f32) / (1e-10 + mean);

        let mut unmask = 0.0f32;
        for i in (12..(len2 - 5)).step_by(4) {
            let id = (64.0 * norm * (tmp2[i] + 1e-10)).floor() as i32;
            let id = id.clamp(0, 127) as usize;
            unmask += INV_TABLE[id] as f32;
        }

        unmask = 64.0 * unmask * 4.0 / (6.0 * (len2 as f32 - 17.0));
        if unmask > mask_metric {
            *tf_chan = c;
            mask_metric = unmask;
        }
    }

    let mut is_transient = mask_metric > 200.0;

    if toneishness > 0.98 && _tone_freq < 0.026 {
        is_transient = false;
        mask_metric = 0.0;
    }

    *tf_estimate = (mask_metric - 150.0).clamp(0.0, 1.0); // Rough estimate

    is_transient
}

fn l1_metric(tmp: &[f32], n: usize, lm: i32, bias: f32) -> f32 {
    let mut l1 = 0.0f32;
    for i in 0..n {
        l1 += tmp[i].abs();
    }
    // l1 = l1 * 16384.0; // Fixed-point artifact?
    l1 + (lm as f32) * bias * l1
}

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
) -> i32 {
    let mut metric = vec![0i32; len];
    let mut tmp = vec![0.0f32; ((mode.e_bands[len] - mode.e_bands[len - 1]) as usize) << lm];
    let mut tmp_1 = vec![0.0f32; ((mode.e_bands[len] - mode.e_bands[len - 1]) as usize) << lm];

    let bias = 0.04 * (-0.25f32).max(0.5 - tf_estimate);

    for i in 0..len {
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
            metric[i] = 2 * best_level;
        } else {
            metric[i] = -2 * best_level;
        }

        if narrow && (metric[i] == 0 || metric[i] == -2 * lm) {
            metric[i] -= 1;
        }
    }

    let mut tf_select = 0;
    let importance = vec![1.0f32; len]; // FIXME: use real importance
    let mut selcost = [0.0f32; 2];

    for sel in 0..2 {
        let mut cost0 = importance[0]
            * ((metric[0]
                - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel + 0]
                    as i32) as f32)
                .abs();
        let mut cost1 = importance[0]
            * ((metric[0]
                - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * sel + 1]
                    as i32) as f32)
                .abs()
            + (if is_transient { 0.0 } else { lambda as f32 });

        for i in 1..len {
            let curr0 = cost0.min(cost1 + lambda as f32);
            let curr1 = (cost0 + lambda as f32).min(cost1);
            cost0 = curr0
                + importance[i]
                    * ((metric[i]
                        - 2 * TF_SELECT_TABLE[lm as usize]
                            [4 * (is_transient as usize) + 2 * sel + 0]
                            as i32) as f32)
                        .abs();
            cost1 = curr1
                + importance[i]
                    * ((metric[i]
                        - 2 * TF_SELECT_TABLE[lm as usize]
                            [4 * (is_transient as usize) + 2 * sel + 1]
                            as i32) as f32)
                        .abs();
        }
        selcost[sel] = cost0.min(cost1);
    }

    if selcost[1] < selcost[0] {
        tf_select = 1;
    }

    let mut cost0 = importance[0]
        * ((metric[0]
            - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * tf_select + 0]
                as i32) as f32)
            .abs();
    let mut cost1 = importance[0]
        * ((metric[0]
            - 2 * TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 * tf_select + 1]
                as i32) as f32)
            .abs()
        + (if is_transient { 0.0 } else { lambda as f32 });

    tf_res[0] = if cost0 < cost1 { 0 } else { 1 };

    for i in 1..len {
        let curr0 = cost0.min(cost1 + lambda as f32);
        let curr1 = (cost0 + lambda as f32).min(cost1);
        cost0 = curr0
            + importance[i]
                * ((metric[i]
                    - 2 * TF_SELECT_TABLE[lm as usize]
                        [4 * (is_transient as usize) + 2 * tf_select + 0]
                        as i32) as f32)
                    .abs();
        cost1 = curr1
            + importance[i]
                * ((metric[i]
                    - 2 * TF_SELECT_TABLE[lm as usize]
                        [4 * (is_transient as usize) + 2 * tf_select + 1]
                        as i32) as f32)
                    .abs();
        tf_res[i] = if cost0 < cost1 { 0 } else { 1 };
    }

    tf_select as i32
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

    // Reserve space to code the tf_select decision (matching C)
    let tf_select_rsv = if lm > 0 && tell + logp + 1 <= budget {
        1
    } else {
        0
    };
    budget -= tf_select_rsv;

    for i in start..end {
        if tell + logp <= budget {
            rc.encode_bit_logp(tf_res[i] ^ curr != 0, logp as u32);
            tell = rc.tell();
            curr = tf_res[i];
            tf_changed |= curr;
        } else {
            tf_res[i] = curr;
        }
        logp = if is_transient { 4 } else { 5 };
    }

    if tf_select_rsv != 0
        && TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 0 + (tf_changed as usize)]
            != TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 + (tf_changed as usize)]
    {
        rc.encode_bit_logp(tf_select != 0, 1);
    } else {
        tf_select = 0;
    }

    // Apply TF_SELECT_TABLE to finalize tf_res (matching C)
    for i in start..end {
        tf_res[i] = TF_SELECT_TABLE[lm as usize]
            [4 * (is_transient as usize) + 2 * (tf_select as usize) + (tf_res[i] as usize)]
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

    let tf_select_rsv = if lm > 0 && tell + logp + 1 <= budget {
        1
    } else {
        0
    };
    let budget = budget - tf_select_rsv;

    for i in start..end {
        if tell + logp <= budget {
            curr ^= if rc.decode_bit_logp(logp as u32) {
                1
            } else {
                0
            };
            tell = rc.tell();
            tf_changed |= curr;
        }
        tf_res[i] = curr;
        logp = if is_transient { 4 } else { 5 };
    }

    let mut tf_select = 0;
    let _budget = budget + tf_select_rsv;
    if tf_select_rsv > 0
        && TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 0 + (tf_changed as usize)]
            != TF_SELECT_TABLE[lm as usize][4 * (is_transient as usize) + 2 + (tf_changed as usize)]
    {
        tf_select = if rc.decode_bit_logp(1) { 1 } else { 0 };
    }

    for i in start..end {
        tf_res[i] = TF_SELECT_TABLE[lm as usize]
            [4 * (is_transient as usize) + 2 * (tf_select as usize) + (tf_res[i] as usize)]
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

    sum_ms *= 0.707107f32;
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
    [0.306640625, 0.2170410156, 0.1296386719],
    [0.4638671875, 0.2680664062, 0.0],
    [0.7998046875, 0.1000976562, 0.0],
];

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
        if x_idx != y_idx || !std::ptr::eq(x.as_ptr(), y.as_ptr()) {
            for i in 0..n {
                y[y_idx + i] = x[x_idx + i];
            }
        }
        return;
    }

    let t0 = t0.max(COMBFILTER_MINPERIOD);
    let t1 = t1.max(COMBFILTER_MINPERIOD);

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
            for j in i..n {
                y[y_idx + j] = x[x_idx + j];
            }
        } else {
            comb_filter_const(y, x, y_idx + i, x_idx + i, t1, n - i, g10, g11, g12);
        }
    }
}

pub struct CeltEncoder {
    mode: &'static CeltMode,
    channels: usize,
    syn_mem: Vec<f32>,        // Size = channels * (2048 + overlap)
    enc_decode_mem: Vec<f32>, // Separate synthesis buffer for encoder resynth
    old_band_e: Vec<f32>,
    preemph_mem: Vec<f32>,
    tonal_average: i32,
    hf_average: i32,
    tapset_decision: i32,
    spread_decision: i32,
    intensity: i32,
    last_coded_bands: i32,
    prefilter_mem: Vec<f32>,
    prefilter_period: usize,
    prefilter_gain: f32,
    prefilter_tapset: i32,
    old_band_e2: Vec<f32>,
    old_band_e3: Vec<f32>,
    last_band_log_e: Vec<f32>,
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

impl CeltEncoder {
    pub fn new(mode: &'static CeltMode, channels: usize) -> Self {
        let overlap = mode.overlap;
        let channel_mem_size = 2048 + overlap;
        let syn_mem_size = channels * channel_mem_size;
        Self {
            mode,
            channels,
            syn_mem: vec![0.0; syn_mem_size],
            enc_decode_mem: vec![0.0; syn_mem_size],
            old_band_e: vec![-28.0; mode.nb_ebands * channels],
            preemph_mem: vec![0.0; channels],
            tonal_average: 256,
            hf_average: 0,
            tapset_decision: 0,
            spread_decision: SPREAD_NORMAL,
            intensity: 0,
            last_coded_bands: 0,
            prefilter_mem: vec![0.0; channels * COMBFILTER_MAXPERIOD],
            prefilter_period: COMBFILTER_MINPERIOD,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            old_band_e2: vec![-28.0; mode.nb_ebands * channels],
            old_band_e3: vec![-28.0; mode.nb_ebands * channels],
            last_band_log_e: vec![-28.0; mode.nb_ebands * channels],
        }
    }

    pub fn encode(&mut self, pcm: &[f32], frame_size: usize, rc: &mut RangeCoder) {
        self.encode_impl(pcm, frame_size, rc, 0)
    }

    /// Encode with start_band support for Hybrid mode.
    /// When start_band > 0, only bands from start_band..nb_ebands are quantized.
    pub fn encode_with_start_band(&mut self, pcm: &[f32], frame_size: usize, rc: &mut RangeCoder, start_band: usize) {
        self.encode_impl(pcm, frame_size, rc, start_band)
    }

    fn encode_impl(&mut self, pcm: &[f32], frame_size: usize, rc: &mut RangeCoder, start_band: usize) {
        let mut max_pcm = 0.0f32;
        for i in 0..frame_size {
            max_pcm = max_pcm.max(pcm[i].abs());
        }

        if frame_size == 960 {}

        let mode = self.mode;
        let channels = self.channels;
        let nb_ebands = mode.nb_ebands;

        if frame_size == 960 {}

        let mode = self.mode;
        let channels = self.channels;
        let nb_ebands = mode.nb_ebands;
        let overlap = mode.overlap;

        // Calculate LM
        let mut lm = 0;
        while (mode.short_mdct_size << lm) != frame_size {
            lm += 1;
            if lm > mode.max_lm {
                break;
            }
        }
        if (mode.short_mdct_size << lm) != frame_size {
            lm = 0; // Default or error
        }

        let syn_mem_size = 2048 + overlap;
        for c in 0..channels {
            let channel_offset = c * syn_mem_size;

            // Shift history
            for i in 0..syn_mem_size - frame_size {
                self.syn_mem[channel_offset + i] = self.syn_mem[channel_offset + i + frame_size];
            }

            // Pre-emphasize and put into syn_mem (at the end)
            let mut m = self.preemph_mem[c];
            let coef = mode.preemph[0];
            for i in 0..frame_size {
                let x = pcm[c * frame_size + i];
                let val = x - m;
                // We put it at the end of the 2048+overlap buffer
                self.syn_mem[channel_offset + syn_mem_size - frame_size + i] = val;
                m = x * coef;
            }
            self.preemph_mem[c] = m;
        }

        // Prepare input buffer for transient_analysis: overlap + frame_size per channel
        let buf_stride = frame_size + overlap;
        let mut in_buf = vec![0.0f32; buf_stride * channels];
        for c in 0..channels {
            let channel_offset = c * syn_mem_size;
            let in_buf_offset = c * buf_stride;
            // Copy: overlap history + current frame
            let src_start = syn_mem_size - frame_size - overlap;
            in_buf[in_buf_offset..in_buf_offset + buf_stride].copy_from_slice(
                &self.syn_mem[channel_offset + src_start..channel_offset + syn_mem_size],
            );
        }

        let mut tf_estimate = 0.0f32;
        let mut tf_chan = 0;
        let mut weak_transient = false;
        let is_transient = transient_analysis(
            &in_buf,
            buf_stride,
            channels,
            &mut tf_estimate,
            &mut tf_chan,
            false, // allow_weak_transients
            &mut weak_transient,
            0.0, // tone_freq (not used)
            0.0, // toneishness (not used)
        );

        let pf_on = false;
        let gain1 = 0.0f32;
        let pitch_index = 0usize;

        // MDCT
        let mut freq = vec![0.0f32; frame_size * channels];
        let (shift, b) = if is_transient {
            (mode.max_lm, 1 << lm)
        } else {
            (mode.max_lm - lm, 1)
        };
        let n = frame_size / b;

        for c in 0..channels {
            let channel_offset = c * syn_mem_size;
            // MDCT needs n + overlap samples
            // New samples are at [syn_mem_size - frame_size, syn_mem_size)
            // Historical samples are at [syn_mem_size - frame_size - overlap, syn_mem_size - frame_size)
            // So MDCT reads from [syn_mem_size - frame_size - overlap, syn_mem_size - frame_size + n)
            // For block i (where i*n are the sample positions within current frame):
            // Read from [syn_mem_size - frame_size - overlap + i*n, syn_mem_size - frame_size - overlap + i*n + n + overlap)
            let mdct_base = syn_mem_size - frame_size - overlap;

            if c == 0 && b == 1 && channels == 1 {
                let mut max_val = 0.0f32;
                let check_len = (frame_size + overlap).min(syn_mem_size - mdct_base);
                for j in 0..check_len {
                    max_val = max_val.max(self.syn_mem[channel_offset + mdct_base + j].abs());
                }
            }

            for i in 0..b {
                // Output is interleaved: out[b + c*N*B] where b is block index
                mode.mdct.forward(
                    &self.syn_mem[channel_offset + mdct_base + i * n..],
                    &mut freq[c * frame_size + i..],
                    mode.window,
                    overlap,
                    shift as usize,
                    b as usize, // stride
                );
            }
        }

        let mut band_e = vec![0.0f32; nb_ebands * channels];
        compute_band_energies(mode, &freq, &mut band_e, nb_ebands, channels, lm as usize);

        if frame_size == 960 {}

        if frame_size == 960 && channels == 1 {}

        let mut x = vec![0.0f32; frame_size * channels];
        normalise_bands(
            mode,
            &freq,
            &mut x,
            &band_e,
            nb_ebands,
            channels,
            (1 << lm) as usize,
        );

        // Normalized bands frequency coefficients
        if channels == 1 {
            let _ = freq[0]; // prevent compiler warning for unused freq
        }

        let mut band_log_e = vec![0.0f32; nb_ebands * channels];
        crate::bands::amp2log2(
            mode,
            nb_ebands,
            nb_ebands,
            &band_e,
            &mut band_log_e,
            channels,
        );

        // Use the full buffer size for total_bits calculation
        // This matches the C behavior when encoder is initialized with a large buffer
        let total_bits = (rc.buf.len() * 8) as i32;
        let mut error = vec![0.0f32; nb_ebands * channels];

        // 0. Silence flag (C order: first thing written)
        let tell = rc.tell();
        let silence = false; // We're encoding real audio
        if tell == 1 {
            rc.encode_bit_logp(silence, 15);
        }

        // 1. Pitch parameters
        if !silence && rc.tell() + 16 <= total_bits {
            rc.encode_bit_logp(pf_on, 1);
            if pf_on {
                let qg = (gain1 / 0.09375 - 1.0 + 0.5).floor() as i32;
                let qg = qg.clamp(0, 7);
                let pi = (pitch_index + 1) as u32;
                let octave = 31 - pi.leading_zeros();
                let octave = (octave as i32 - 5).max(0) as u32;
                rc.enc_uint(octave, 6);
                rc.enc_bits(pi - (16 << octave), 4 + octave);
                rc.enc_bits(qg as u32, 3);
                rc.encode_icdf(self.prefilter_tapset, &TAPSET_ICDF, 2);
            }
        }

        // 2. Transient bit
        let mut short_blocks = false;
        if lm > 0 && rc.tell() + 3 <= total_bits {
            rc.encode_bit_logp(is_transient, 3);
            if is_transient {
                short_blocks = true;
            }
        }

        if short_blocks {
            // Re-compute MDCT with short blocks
            let b = 1 << lm;
            let n = frame_size / b;
            for c in 0..channels {
                let c_offset = c * buf_stride;
                for i in 0..b {
                    mode.mdct.forward(
                        &in_buf[c_offset + overlap / 2 + i * n..c_offset + buf_stride],
                        &mut freq[c * frame_size + i..],
                        mode.window,
                        overlap,
                        mode.max_lm as usize,
                        b as usize,
                    );
                }
            }
            // And re-normalise
            compute_band_energies(mode, &freq, &mut band_e, nb_ebands, channels, lm as usize);
            normalise_bands(
                mode,
                &freq,
                &mut x,
                &band_e,
                nb_ebands,
                channels,
                (1 << lm) as usize,
            );
        }

        // 3. Coarse energy
        let intra_ener = false; // For now assuming no forced intra except transient
        quant_coarse_energy(
            mode,
            start_band,
            nb_ebands,
            &mut band_log_e,
            &mut self.old_band_e,
            (total_bits << 3) as u32,
            &mut error,
            rc,
            channels,
            lm as usize,
            is_transient || intra_ener,
        );

        // 4. TF Analysis
        let mut tf_res = vec![0i32; nb_ebands];
        let effective_bytes = (total_bits / 8) as usize;
        let lambda = 80.max(20480 / effective_bytes + 2) as i32;
        let tf_select = tf_analysis(
            mode,
            nb_ebands,
            is_transient, // Use the detected transient flag
            &mut tf_res,
            lambda,
            &x,
            frame_size,
            lm as i32,
            tf_estimate,
            tf_chan,
        );
        tf_encode(
            start_band,
            nb_ebands,
            is_transient,
            &mut tf_res,
            lm as i32,
            tf_select,
            rc,
        );

        // 5. Spread decision (must come after TF, before dynalloc, matching C order)
        let mut dual_stereo_val = if channels == 2 {
            stereo_analysis(mode, &x, lm as i32, frame_size) as i32
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

        // Spread decision - encode BEFORE dynalloc and trim (C order)
        if rc.tell() + 4 <= total_bits {
            let update_hf = lm == mode.max_lm;
            let spread_weights = vec![32i32; nb_ebands];
            self.spread_decision = spreading_decision(
                mode,
                &x,
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
            rc.encode_icdf(self.spread_decision, &SPREAD_ICDF, 5);
        } else {
            self.spread_decision = SPREAD_NORMAL;
        }

        // 6. Dynalloc encoding (per-band boost flags)
        let mut cap = vec![0i32; nb_ebands];
        for i in 0..nb_ebands {
            cap[i] = (mode.cache.caps[nb_ebands * (2 * lm + channels - 1) + i] as i32 + 64)
                * channels as i32
                * 2;
        }

        let mut offsets = vec![0i32; nb_ebands];
        let dynalloc_logp = 6i32;
        let total_bits_bitres = total_bits << BITRES;
        let total_boost = 0i32;
        {
            for i in 0..nb_ebands {
                let dynalloc_loop_logp = dynalloc_logp;
                let boost = 0i32;
                let tell_frac = rc.tell() << BITRES;
                // Encoder: offsets are all 0, so we always write flag=0 and break
                if tell_frac + (dynalloc_loop_logp << BITRES) < total_bits_bitres - total_boost
                    && boost < cap[i]
                {
                    rc.encode_bit_logp(false, dynalloc_loop_logp as u32);
                }
                offsets[i] = boost;
            }
        }

        // 7. Alloc trim (after spread and dynalloc, matching C order)
        let alloc_trim = alloc_trim_analysis(
            mode,
            &x,
            &band_log_e,
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
        if (rc.tell() << BITRES) + (6 << BITRES) <= total_bits_bitres - total_boost {
            rc.encode_icdf(alloc_trim, &TRIM_ICDF, 7);
        }

        // 8. Compute allocation
        let mut intensity = self.intensity;
        let mut pulses = vec![0i32; nb_ebands];
        // For stereo, ebits and fine_priority need to be channels * nb_ebands for quant_fine_energy
        let stereo = channels > 1;
        let ebands_stereo = if stereo { nb_ebands * channels } else { nb_ebands };
        let mut fine_priority = vec![0i32; ebands_stereo];
        let mut ebits = vec![0i32; ebands_stereo];
        let mut balance = 0;

        self.last_coded_bands = clt_compute_allocation(
            mode,
            start_band,
            nb_ebands,
            &offsets,
            &cap,
            alloc_trim,
            &mut intensity,
            &mut dual_stereo_val,
            total_bits << 3,
            &mut balance,
            &mut pulses,
            &mut ebits,
            &mut fine_priority,
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
            &mut error,
            &ebits,
            rc,
            channels,
        );

        // Spread decision already encoded above (after TF)
        let mut collapse_masks = vec![0u32; nb_ebands * channels];
        let (x_split, y_split) = x.split_at_mut(frame_size);
        let y_opt = if channels == 2 { Some(y_split) } else { None };

        let mut dual_stereo = dual_stereo_val != 0;
        quant_all_bands(
            true,
            mode,
            start_band,
            nb_ebands,
            x_split,
            y_opt,
            &mut collapse_masks,
            &band_e,
            &pulses,
            short_blocks,
            self.spread_decision,
            &mut dual_stereo,
            intensity as usize,
            &tf_res,
            total_bits << 3,
            &mut balance,
            rc,
            lm as i32,
            self.last_coded_bands,
            true,
        );

        quant_energy_finalise(
            mode,
            start_band,
            nb_ebands,
            &mut self.old_band_e,
            &mut error,
            &ebits,
            &fine_priority,
            (total_bits - rc.tell() as i32) << 3,
            rc,
            channels,
        );

        // Encoder synthesis: reconstruct the signal after PVQ to update overlap state.
        // This ensures the encoder and decoder have the same overlap for subsequent frames.
        {
            let mut band_amp_synth = vec![0.0f32; nb_ebands * channels];
            log2amp(
                mode,
                nb_ebands,
                &mut band_amp_synth,
                &self.old_band_e,
                channels,
            );
            let mut freq_synth = vec![0.0f32; frame_size * channels];
            denormalise_bands(
                mode,
                &x,
                &mut freq_synth,
                &band_amp_synth,
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

            // Shift the encoder's decode_mem
            for c in 0..channels {
                let co = c * syn_mem_size;
                for i in 0..decode_buf_size - frame_size + overlap {
                    self.enc_decode_mem[co + i] = self.enc_decode_mem[co + i + frame_size];
                }
            }

            // MDCT backward into enc_decode_mem
            for c in 0..channels {
                let co = c * syn_mem_size;
                let out_syn_idx = decode_buf_size - frame_size;
                for bi in 0..syn_b {
                    mode.mdct.backward(
                        &freq_synth[c * frame_size + bi..],
                        &mut self.enc_decode_mem[co + out_syn_idx + bi * syn_n..],
                        mode.window,
                        overlap,
                        syn_shift as usize,
                        syn_b as usize,
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

        while rc.tell() < total_bits {
            rc.enc_bits(0, 1);
        }

        if pf_on {
            self.prefilter_period = pitch_index;
            self.prefilter_gain = gain1;
        } else {
            self.prefilter_period = COMBFILTER_MINPERIOD;
            self.prefilter_gain = 0.0;
        }

        // self.old_band_e.copy_from_slice(&band_log_e); // REMOVED: keep quantized history

        let syn_mem_size = 2048 + overlap;
        // History and prefilter_mem updates
        for c in 0..channels {
            let channel_offset = c * syn_mem_size;
            let n = frame_size;
            let max_period = COMBFILTER_MAXPERIOD;
            if n >= max_period {
                self.prefilter_mem[c * max_period..(c + 1) * max_period].copy_from_slice(
                    &self.syn_mem
                        [channel_offset + syn_mem_size - max_period..channel_offset + syn_mem_size],
                );
            } else {
                let mut new_mem = vec![0.0f32; max_period];
                new_mem[..max_period - n]
                    .copy_from_slice(&self.prefilter_mem[c * max_period + n..(c + 1) * max_period]);
                new_mem[max_period - n..].copy_from_slice(
                    &self.syn_mem[channel_offset + syn_mem_size - n..channel_offset + syn_mem_size],
                );
                self.prefilter_mem[c * max_period..(c + 1) * max_period].copy_from_slice(&new_mem);
            }
        }
    }
}

pub struct CeltDecoder {
    mode: &'static CeltMode,
    channels: usize,
    decode_mem: Vec<f32>, // Size = channels * (2048 + overlap)
    old_band_e: Vec<f32>,
    preemph_mem: Vec<f32>,
    prefilter_mem: Vec<f32>,
    prefilter_period: usize,
    prefilter_gain: f32,
    prefilter_tapset: i32,
    old_band_e2: Vec<f32>,
    old_band_e3: Vec<f32>,
    rng: u32,
}

impl CeltDecoder {
    pub fn new(mode: &'static CeltMode, channels: usize) -> Self {
        let overlap = mode.overlap;
        let decode_buffer_size = 2048;
        Self {
            mode,
            channels,
            decode_mem: vec![0.0; channels * (decode_buffer_size + overlap)],
            old_band_e: vec![-28.0; mode.nb_ebands * channels],
            preemph_mem: vec![0.0; channels],
            prefilter_mem: vec![0.0; channels * COMBFILTER_MAXPERIOD],
            prefilter_period: COMBFILTER_MINPERIOD,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            old_band_e2: vec![-28.0; mode.nb_ebands * channels],
            old_band_e3: vec![-28.0; mode.nb_ebands * channels],
            rng: 0,
        }
    }

    pub fn decode(&mut self, compressed: &[u8], frame_size: usize, pcm: &mut [f32]) -> usize {
        self.decode_impl(compressed, frame_size, pcm, 0)
    }

    /// Decode with start_band support for Hybrid mode.
    /// `start_band > 0` means this CELT frame only contains data for bands
    /// from start_band..nb_ebands; the lower bands come from SILK.
    pub fn decode_with_start_band(&mut self, compressed: &[u8], frame_size: usize, pcm: &mut [f32], start_band: usize) -> usize {
        self.decode_impl(compressed, frame_size, pcm, start_band)
    }

    fn decode_impl(&mut self, compressed: &[u8], frame_size: usize, pcm: &mut [f32], start_band: usize) -> usize {
        let mode = self.mode;
        let channels = self.channels;
        let nb_ebands = mode.nb_ebands;
        let overlap = mode.overlap;

        // Calculate LM
        let mut lm = 0;
        while (mode.short_mdct_size << lm) != frame_size {
            lm += 1;
            if lm > mode.max_lm {
                break;
            }
        }
        if (mode.short_mdct_size << lm) != frame_size {
            lm = 0; // Default or error
        }

        let mut rc = RangeCoder::new_decoder(compressed.to_vec());
        let total_bits = (compressed.len() * 8) as i32;

        // Silence check (C order: first thing after init)
        let tell = rc.tell();
        let mut silence = false;
        if tell >= total_bits {
            silence = true;
        } else if tell == 1 {
            silence = rc.decode_bit_logp(15);
        }
        if silence {
            // Pretend we've read all the remaining bits
            // (skip to end of packet)
        }

        let mut pf_on = false;
        let mut pitch_index = COMBFILTER_MINPERIOD;
        let mut gain1 = 0.0f32;
        let mut prefilter_tapset = 0;
        if !silence && rc.tell() + 16 <= total_bits {
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
        let intra_ener = false;

        unquant_coarse_energy(
            mode,
            start_band,
            nb_ebands,
            &mut self.old_band_e,
            (total_bits << 3) as u32,
            &mut rc,
            channels,
            lm as usize,
            is_transient || intra_ener,
        );

        let mut tf_res = vec![0i32; nb_ebands];
        tf_decode(start_band, nb_ebands, is_transient, &mut tf_res, lm as i32, &mut rc);

        // Spread decision (C order: after TF, before dynalloc)
        let spread_decision = if rc.tell() + 4 <= total_bits {
            rc.decode_icdf(&SPREAD_ICDF, 5) as i32
        } else {
            SPREAD_NORMAL
        };

        // Cap init
        let mut cap = vec![0i32; nb_ebands];
        for i in 0..nb_ebands {
            cap[i] = (mode.cache.caps[nb_ebands * (2 * lm + channels - 1) + i] as i32 + 64)
                * channels as i32
                * 2;
        }

        // Dynalloc decoding (C order: after spread, before trim)
        let mut offsets = vec![0i32; nb_ebands];
        let mut dynalloc_logp = 6i32;
        let mut total_bits_bitres = total_bits << BITRES;
        let mut tell_frac = rc.tell() << BITRES;
        for i in 0..nb_ebands {
            let width =
                channels as i32 * (mode.e_bands[i + 1] - mode.e_bands[i]) as i32 * (1 << lm);
            let quanta = (width << BITRES).min((6i32 << BITRES).max(width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            while tell_frac + (dynalloc_loop_logp << BITRES) < total_bits_bitres && boost < cap[i] {
                let flag = rc.decode_bit_logp(dynalloc_loop_logp as u32);
                tell_frac = rc.tell() << BITRES;
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

        // Alloc trim (C order: after dynalloc)
        let alloc_trim = if (rc.tell() << BITRES) + (6 << BITRES) <= total_bits_bitres {
            rc.decode_icdf(&TRIM_ICDF, 7)
        } else {
            5 // Default trim
        };
        let anti_collapse_rsv = if is_transient && lm >= 2 {
            let remaining = (total_bits << BITRES) - (rc.tell() << BITRES) - 1;
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
        let mut pulses = vec![0i32; nb_ebands];
        // For stereo, ebits and fine_priority need to be channels * nb_ebands for unquant_fine_energy
        let ebands_stereo = if channels > 1 { nb_ebands * channels } else { nb_ebands };
        let mut fine_priority = vec![0i32; ebands_stereo];
        let mut ebits = vec![0i32; ebands_stereo];

        let coded_bands = clt_compute_allocation(
            mode,
            start_band,
            nb_ebands,
            &offsets,
            &cap,
            alloc_trim,
            &mut intensity,
            &mut dual_stereo_val,
            total_bits << 3,
            &mut balance,
            &mut pulses,
            &mut ebits,
            &mut fine_priority,
            channels as i32,
            lm as i32,
            &mut rc,
            false,
            0,
            nb_ebands as i32 - 1,
        );

        unquant_fine_energy(
            mode,
            start_band,
            nb_ebands,
            &mut self.old_band_e,
            &ebits,
            &mut rc,
            channels,
        );

        let mut x = vec![0.0f32; frame_size * channels];
        let mut collapse_masks = vec![0u32; nb_ebands * channels];

        // Shift decode buffer BEFORE PVQ decode (matching C)
        let decode_buffer_size = 2048;
        for c in 0..channels {
            let channel_mem_offset = c * (decode_buffer_size + overlap);
            for i in 0..decode_buffer_size - frame_size + overlap {
                self.decode_mem[channel_mem_offset + i] =
                    self.decode_mem[channel_mem_offset + i + frame_size];
            }
        }

        let (x_split, y_split) = x.split_at_mut(frame_size);
        let y_opt = if channels == 2 { Some(y_split) } else { None };

        let mut dual_stereo = dual_stereo_val != 0;
        let mut band_amp = vec![0.0f32; nb_ebands * channels];
        log2amp(mode, nb_ebands, &mut band_amp, &self.old_band_e, channels);

        quant_all_bands(
            false,
            mode,
            start_band,
            nb_ebands,
            x_split,
            y_opt,
            &mut collapse_masks,
            &band_amp,
            &pulses,
            short_blocks,
            spread_decision,
            &mut dual_stereo,
            intensity as usize,
            &tf_res,
            total_bits << 3,
            &mut balance,
            &mut rc,
            lm as i32,
            coded_bands,
            true,
        );

        // Anti-collapse
        let mut anti_collapse_on = false;
        if anti_collapse_rsv > 0 {
            anti_collapse_on = rc.dec_bits(1) != 0; // raw bit, not logp
        }

        unquant_energy_finalise(
            mode,
            start_band,
            nb_ebands,
            &mut self.old_band_e,
            &ebits,
            &fine_priority,
            (total_bits - rc.tell() as i32) << 3,
            &mut rc,
            channels,
        );

        if anti_collapse_on {
            self.rng = crate::bands::anti_collapse(
                mode,
                &mut x,
                &collapse_masks,
                lm as i32,
                channels,
                frame_size,
                0,
                nb_ebands,
                &self.old_band_e,
                &self.old_band_e2,
                &self.old_band_e3,
                &pulses,
                self.rng,
            );
        }

        let mut freq = vec![0.0f32; frame_size * channels];
        denormalise_bands(
            mode,
            &x,
            &mut freq,
            &self.old_band_e,
            start_band,
            nb_ebands,
            channels,
            (1 << lm) as usize,
        );

        let (shift, b) = if short_blocks {
            (mode.max_lm, 1 << lm)
        } else {
            (mode.max_lm - lm, 1)
        };
        let n = frame_size / b;

        for c in 0..channels {
            let channel_mem_offset = c * (decode_buffer_size + overlap);
            // C: out_syn[c] = decode_mem[c] + decode_buffer_size - N
            // where N is the full frame size (not the block size n)
            let out_syn_idx = decode_buffer_size - frame_size;

            for i in 0..b {
                let block_freq_idx = c * frame_size + i; // Interleaved start
                let block_out_idx = channel_mem_offset + out_syn_idx + i * n;
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
                    shift as usize,
                    b as usize,
                );
            }

            let mut pcm_frame = vec![0.0f32; frame_size];
            // After MDCT backward, samples are at decode_mem[out_syn_idx..]
            // because MDCT backward OLA'd the tail of the previous frame.
            for i in 0..frame_size {
                pcm_frame[i] = self.decode_mem[channel_mem_offset + out_syn_idx + i];
            }

            // Post-filter (pitch filter)
            if pf_on || self.prefilter_gain > 0.0 {
                let mut filtered = vec![0.0f32; frame_size];
                let mut post = vec![0.0f32; frame_size + COMBFILTER_MAXPERIOD];
                post[..COMBFILTER_MAXPERIOD].copy_from_slice(
                    &self.prefilter_mem[c * COMBFILTER_MAXPERIOD..(c + 1) * COMBFILTER_MAXPERIOD],
                );
                post[COMBFILTER_MAXPERIOD..].copy_from_slice(&pcm_frame);

                comb_filter(
                    &mut filtered,
                    &post,
                    0,
                    COMBFILTER_MAXPERIOD,
                    self.prefilter_period,
                    pitch_index,
                    frame_size,
                    self.prefilter_gain,
                    gain1,
                    self.prefilter_tapset,
                    prefilter_tapset as i32,
                    mode.window,
                    overlap,
                );
                pcm_frame.copy_from_slice(&filtered);
                // In C, post-filtered samples are written back to decode_mem
                for i in 0..frame_size {
                    self.decode_mem[channel_mem_offset + out_syn_idx + i] = pcm_frame[i];
                }
            }

            // Update prefilter_mem with the samples AFTER post-filtering
            let mut new_mem = vec![0.0f32; COMBFILTER_MAXPERIOD];
            if frame_size >= COMBFILTER_MAXPERIOD {
                new_mem.copy_from_slice(&pcm_frame[frame_size - COMBFILTER_MAXPERIOD..frame_size]);
            } else {
                new_mem[..COMBFILTER_MAXPERIOD - frame_size].copy_from_slice(
                    &self.prefilter_mem
                        [c * COMBFILTER_MAXPERIOD + frame_size..(c + 1) * COMBFILTER_MAXPERIOD],
                );
                new_mem[COMBFILTER_MAXPERIOD - frame_size..].copy_from_slice(&pcm_frame);
            }
            self.prefilter_mem[c * COMBFILTER_MAXPERIOD..(c + 1) * COMBFILTER_MAXPERIOD]
                .copy_from_slice(&new_mem);

            // De-preemphasis
            let coef = mode.preemph[0];
            let mut m = self.preemph_mem[c];
            for i in 0..frame_size {
                let x = pcm_frame[i];
                let val = x + m;
                pcm[c * frame_size + i] = val;
                m = val * coef;
            }
            self.preemph_mem[c] = m;
        }

        if pf_on {
            self.prefilter_period = pitch_index;
            self.prefilter_gain = gain1;
            self.prefilter_tapset = prefilter_tapset as i32;
        } else {
            self.prefilter_period = COMBFILTER_MINPERIOD;
            self.prefilter_gain = 0.0;
            self.prefilter_tapset = 0;
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

        frame_size
    }
}
