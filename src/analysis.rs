//! Signal analysis of the encoder input (libopus `src/analysis.c`, float
//! build): tonality, spectral leakage and bandwidth from a 480-point FFT of
//! 20 ms windows at 24 kHz, and a small network's music and activity
//! probabilities. libopus runs it in the float build for complexity 7 and up
//! at 16 kHz and above; its results steer the mode, the bandwidth, DTX and
//! several CELT decisions.

use crate::celt::AnalysisInfo;
#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::kiss_fft::{KissCpx, KissFftState, opus_fft};
use crate::mlp::{MAX_NEURONS, compute_probabilities};

const NB_FRAMES: usize = 8;
const NB_TBANDS: usize = 18;
/// 30 ms at 24 kHz.
const ANALYSIS_BUF_SIZE: usize = 720;
/// Past that many frames the count no longer matters.
const ANALYSIS_COUNT_MAX: i32 = 10000;
const DETECT_SIZE: usize = 100;
const LEAK_BANDS: usize = 19;
// The leakage boosts cover the analysis bands and the band above 12 kHz.
const _: () = assert!(NB_TBANDS + 1 == LEAK_BANDS);
const TRANSITION_PENALTY: f32 = 10.0;
const NB_TONAL_SKIP_BANDS: usize = 9;
const LEAKAGE_OFFSET: f32 = 2.5;
const LEAKAGE_SLOPE: f32 = 2.0;

#[rustfmt::skip]
const DCT_TABLE: [f32; 128] = [
    0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000,
    0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000, 0.250000,
    0.351851, 0.338330, 0.311806, 0.273300, 0.224292, 0.166664, 0.102631, 0.034654,
    -0.034654, -0.102631, -0.166664, -0.224292, -0.273300, -0.311806, -0.338330, -0.351851,
    0.346760, 0.293969, 0.196424, 0.068975, -0.068975, -0.196424, -0.293969, -0.346760,
    -0.346760, -0.293969, -0.196424, -0.068975, 0.068975, 0.196424, 0.293969, 0.346760,
    0.338330, 0.224292, 0.034654, -0.166664, -0.311806, -0.351851, -0.273300, -0.102631,
    0.102631, 0.273300, 0.351851, 0.311806, 0.166664, -0.034654, -0.224292, -0.338330,
    0.326641, 0.135299, -0.135299, -0.326641, -0.326641, -0.135299, 0.135299, 0.326641,
    0.326641, 0.135299, -0.135299, -0.326641, -0.326641, -0.135299, 0.135299, 0.326641,
    0.311806, 0.034654, -0.273300, -0.338330, -0.102631, 0.224292, 0.351851, 0.166664,
    -0.166664, -0.351851, -0.224292, 0.102631, 0.338330, 0.273300, -0.034654, -0.311806,
    0.293969, -0.068975, -0.346760, -0.196424, 0.196424, 0.346760, 0.068975, -0.293969,
    -0.293969, 0.068975, 0.346760, 0.196424, -0.196424, -0.346760, -0.068975, 0.293969,
    0.273300, -0.166664, -0.338330, 0.034654, 0.351851, 0.102631, -0.311806, -0.224292,
    0.224292, 0.311806, -0.102631, -0.351851, -0.034654, 0.338330, 0.166664, -0.273300,
];

#[rustfmt::skip]
const ANALYSIS_WINDOW: [f32; 240] = [
    0.000043, 0.000171, 0.000385, 0.000685, 0.001071, 0.001541, 0.002098, 0.002739,
    0.003466, 0.004278, 0.005174, 0.006156, 0.007222, 0.008373, 0.009607, 0.010926,
    0.012329, 0.013815, 0.015385, 0.017037, 0.018772, 0.020590, 0.022490, 0.024472,
    0.026535, 0.028679, 0.030904, 0.033210, 0.035595, 0.038060, 0.040604, 0.043227,
    0.045928, 0.048707, 0.051564, 0.054497, 0.057506, 0.060591, 0.063752, 0.066987,
    0.070297, 0.073680, 0.077136, 0.080665, 0.084265, 0.087937, 0.091679, 0.095492,
    0.099373, 0.103323, 0.107342, 0.111427, 0.115579, 0.119797, 0.124080, 0.128428,
    0.132839, 0.137313, 0.141849, 0.146447, 0.151105, 0.155823, 0.160600, 0.165435,
    0.170327, 0.175276, 0.180280, 0.185340, 0.190453, 0.195619, 0.200838, 0.206107,
    0.211427, 0.216797, 0.222215, 0.227680, 0.233193, 0.238751, 0.244353, 0.250000,
    0.255689, 0.261421, 0.267193, 0.273005, 0.278856, 0.284744, 0.290670, 0.296632,
    0.302628, 0.308658, 0.314721, 0.320816, 0.326941, 0.333097, 0.339280, 0.345492,
    0.351729, 0.357992, 0.364280, 0.370590, 0.376923, 0.383277, 0.389651, 0.396044,
    0.402455, 0.408882, 0.415325, 0.421783, 0.428254, 0.434737, 0.441231, 0.447736,
    0.454249, 0.460770, 0.467298, 0.473832, 0.480370, 0.486912, 0.493455, 0.500000,
    0.506545, 0.513088, 0.519630, 0.526168, 0.532702, 0.539230, 0.545751, 0.552264,
    0.558769, 0.565263, 0.571746, 0.578217, 0.584675, 0.591118, 0.597545, 0.603956,
    0.610349, 0.616723, 0.623077, 0.629410, 0.635720, 0.642008, 0.648271, 0.654508,
    0.660720, 0.666903, 0.673059, 0.679184, 0.685279, 0.691342, 0.697372, 0.703368,
    0.709330, 0.715256, 0.721144, 0.726995, 0.732807, 0.738579, 0.744311, 0.750000,
    0.755647, 0.761249, 0.766807, 0.772320, 0.777785, 0.783203, 0.788573, 0.793893,
    0.799162, 0.804381, 0.809547, 0.814660, 0.819720, 0.824724, 0.829673, 0.834565,
    0.839400, 0.844177, 0.848895, 0.853553, 0.858151, 0.862687, 0.867161, 0.871572,
    0.875920, 0.880203, 0.884421, 0.888573, 0.892658, 0.896677, 0.900627, 0.904508,
    0.908321, 0.912063, 0.915735, 0.919335, 0.922864, 0.926320, 0.929703, 0.933013,
    0.936248, 0.939409, 0.942494, 0.945503, 0.948436, 0.951293, 0.954072, 0.956773,
    0.959396, 0.961940, 0.964405, 0.966790, 0.969096, 0.971321, 0.973465, 0.975528,
    0.977510, 0.979410, 0.981228, 0.982963, 0.984615, 0.986185, 0.987671, 0.989074,
    0.990393, 0.991627, 0.992778, 0.993844, 0.994826, 0.995722, 0.996534, 0.997261,
    0.997902, 0.998459, 0.998929, 0.999315, 0.999615, 0.999829, 0.999957, 1.000000,
];

const TBANDS: [usize; NB_TBANDS + 1] = [
    4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160, 192, 240,
];

const STD_FEATURE_BIAS: [f32; 9] = [
    5.684947, 3.475288, 1.770634, 1.599784, 3.773215, 2.163313, 1.260756, 1.116868, 1.918795,
];

/// The encoder input as the analysis reads it (libopus `downmix_float` and
/// `downmix_int`): interleaved samples of `channels` channels.
#[derive(Clone, Copy)]
pub enum AnalysisPcm<'a> {
    F32(&'a [f32]),
    I16(&'a [i16]),
}

impl AnalysisPcm<'_> {
    /// libopus `downmix_float` / `downmix_int` with `c1 = 0`, `c2 = -2`: the
    /// sum of all channels of `y.len()` samples from `offset`, in the 16-bit
    /// scale; float input is capped at +6 dBFS.
    fn downmix(&self, y: &mut [f32], offset: usize, channels: usize) {
        let n = y.len();
        match *self {
            AnalysisPcm::F32(x) => {
                for (j, v) in y.iter_mut().enumerate() {
                    *v = x[(j + offset) * channels] * 32768.0;
                }
                for c in 1..channels {
                    for j in 0..n {
                        y[j] += x[(j + offset) * channels + c] * 32768.0;
                    }
                }
                for v in y.iter_mut() {
                    *v = v.clamp(-65536.0, 65536.0);
                    if v.is_nan() {
                        *v = 0.0;
                    }
                }
            }
            AnalysisPcm::I16(x) => {
                for (j, v) in y.iter_mut().enumerate() {
                    *v = f32::from(x[(j + offset) * channels]);
                }
                for c in 1..channels {
                    for j in 0..n {
                        y[j] += f32::from(x[(j + offset) * channels + c]);
                    }
                }
            }
        }
    }
}

/// libopus `fast_atan2f`.
#[inline]
fn fast_atan2f(y: f32, x: f32) -> f32 {
    const CA: f32 = 0.431_579_74;
    const CB: f32 = 0.678484; // libopus 0.67848403f
    const CC: f32 = 0.085_955_42;
    const CE: f32 = core::f64::consts::PI as f32 / 2.0;
    let x2 = x * x;
    let y2 = y * y;
    // For very small values, the answer does not matter.
    if x2 + y2 < 1e-18 {
        return 0.0;
    }
    if x2 < y2 {
        let den = (y2 + CB * x2) * (y2 + CC * x2);
        -x * y * (y2 + CA * x2) / den + if y < 0.0 { -CE } else { CE }
    } else {
        let den = (x2 + CB * y2) * (x2 + CC * y2);
        x * y * (x2 + CA * y2) / den + if y < 0.0 { -CE } else { CE }
            - if x * y < 0.0 { -CE } else { CE }
    }
}

/// libopus `float2int`: round to the nearest integer, ties to even.
#[inline]
fn float2int(x: f32) -> i32 {
    let f = x.floor();
    let d = x - f;
    let r = if d > 0.5 || (d == 0.5 && (f as i64) % 2 != 0) {
        f + 1.0
    } else {
        f
    };
    r as i32
}

/// libopus `MAX32` / `MIN32` (C ternaries).
#[inline]
fn max32(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}
#[inline]
fn min32(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// `(float)log(x)` of libopus: natural log in double precision.
#[inline]
fn logd(x: f32) -> f32 {
    f64::from(x).ln() as f32
}
/// `(float)sqrt(x)` of libopus, in double precision.
#[inline]
fn sqrtd(x: f64) -> f32 {
    x.sqrt() as f32
}

/// The analysis of libopus's float build (`silk_resampler_down2_hp`): halve
/// the rate with an all-pass pair and return the energy of the upper half
/// band, which the 24 kHz spectrum lacks.
fn resampler_down2_hp(s: &mut [f32; 3], out: &mut [f32], input: &[f32]) -> f32 {
    let len2 = input.len() / 2;
    let mut hp_ener = 0.0f32;
    for k in 0..len2 {
        // All-pass section for the even input sample.
        let in32 = input[2 * k];
        let y = in32 - s[0];
        let x = 0.607_437_1 * y;
        let mut out32 = s[0] + x;
        s[0] = in32 + x;
        let mut out32_hp = out32;
        // All-pass section for the odd input sample, added to the output of
        // the previous section.
        let in32 = input[2 * k + 1];
        let y = in32 - s[1];
        let x = 0.150_63 * y;
        out32 += s[1];
        out32 += x;
        s[1] = in32 + x;

        let y = -in32 - s[2];
        let x = 0.150_63 * y;
        out32_hp += s[2];
        out32_hp += x;
        s[2] = -in32 + x;

        hp_ener += out32_hp * out32_hp;
        out[k] = 0.5 * out32;
    }
    hp_ener
}

/// The analysis state (libopus `TonalityAnalysisState`).
pub struct TonalityAnalysisState {
    fs: i32,
    angle: [f32; 240],
    d_angle: [f32; 240],
    d2_angle: [f32; 240],
    inmem: [f32; ANALYSIS_BUF_SIZE],
    /// Number of usable samples in the buffer.
    mem_fill: usize,
    prev_band_tonality: [f32; NB_TBANDS],
    prev_tonality: f32,
    prev_bandwidth: i32,
    e: [[f32; NB_TBANDS]; NB_FRAMES],
    log_e: [[f32; NB_TBANDS]; NB_FRAMES],
    low_e: [f32; NB_TBANDS],
    high_e: [f32; NB_TBANDS],
    mean_e: [f32; NB_TBANDS + 1],
    mem: [f32; 32],
    cmean: [f32; 8],
    std: [f32; 9],
    e_tracker: f32,
    low_e_count: f32,
    e_count: usize,
    count: i32,
    analysis_offset: i32,
    write_pos: usize,
    read_pos: usize,
    read_subframe: i32,
    hp_ener_accum: f32,
    initialized: bool,
    rnn_state: [f32; MAX_NEURONS],
    downmix_state: [f32; 3],
    info: [AnalysisInfo; DETECT_SIZE],
}

impl TonalityAnalysisState {
    /// libopus `tonality_analysis_init`.
    pub fn new(fs: i32) -> Self {
        Self {
            fs,
            angle: [0.0; 240],
            d_angle: [0.0; 240],
            d2_angle: [0.0; 240],
            inmem: [0.0; ANALYSIS_BUF_SIZE],
            mem_fill: 0,
            prev_band_tonality: [0.0; NB_TBANDS],
            prev_tonality: 0.0,
            prev_bandwidth: 0,
            e: [[0.0; NB_TBANDS]; NB_FRAMES],
            log_e: [[0.0; NB_TBANDS]; NB_FRAMES],
            low_e: [0.0; NB_TBANDS],
            high_e: [0.0; NB_TBANDS],
            mean_e: [0.0; NB_TBANDS + 1],
            mem: [0.0; 32],
            cmean: [0.0; 8],
            std: [0.0; 9],
            e_tracker: 0.0,
            low_e_count: 0.0,
            e_count: 0,
            count: 0,
            analysis_offset: 0,
            write_pos: 0,
            read_pos: 0,
            read_subframe: 0,
            hp_ener_accum: 0.0,
            initialized: false,
            rnn_state: [0.0; MAX_NEURONS],
            downmix_state: [0.0; 3],
            info: [AnalysisInfo::zeroed(); DETECT_SIZE],
        }
    }

    /// libopus `tonality_analysis_reset`: forget the signal, keep the rate.
    pub(crate) fn reset(&mut self) {
        *self = Self::new(self.fs);
    }

    /// Whether the analysis ran since it was created or reset (libopus
    /// `initialized`).
    pub(crate) fn initialized(&self) -> bool {
        self.initialized
    }

    /// libopus `downmix_and_resample`: `subframe` samples at 24 kHz from
    /// `offset` (also at 24 kHz) of the input, downmixed to mono and brought
    /// to 24 kHz into `y`. Returns the energy above 12 kHz of 48 kHz input.
    fn downmix_and_resample(
        &mut self,
        x: AnalysisPcm<'_>,
        at: usize,
        subframe: usize,
        offset: usize,
        channels: usize,
    ) -> f32 {
        if subframe == 0 {
            return 0.0;
        }
        let (subframe, offset) = match self.fs {
            48000 => (subframe * 2, offset * 2),
            16000 => (subframe * 2 / 3, offset * 2 / 3),
            _ => (subframe, offset),
        };
        // At most 30 ms at 48 kHz.
        let mut tmp = [0.0f32; 2 * ANALYSIS_BUF_SIZE];
        let tmp = &mut tmp[..subframe];
        x.downmix(tmp, offset, channels);
        if channels == 2 {
            for v in tmp.iter_mut() {
                *v *= 0.5;
            }
        }
        let mut ret = 0.0f32;
        match self.fs {
            48000 => {
                ret = resampler_down2_hp(
                    &mut self.downmix_state,
                    &mut self.inmem[at..at + subframe / 2],
                    tmp,
                );
            }
            16000 => {
                // libopus: "Don't do this at home! This resampler is horrible
                // and it's only (barely) usable for the purpose of the
                // analysis".
                let mut tmp3x = [0.0f32; 3 * ANALYSIS_BUF_SIZE];
                let tmp3x = &mut tmp3x[..3 * subframe];
                for (j, &v) in tmp.iter().enumerate() {
                    tmp3x[3 * j] = v;
                    tmp3x[3 * j + 1] = v;
                    tmp3x[3 * j + 2] = v;
                }
                let n = 3 * subframe / 2;
                resampler_down2_hp(&mut self.downmix_state, &mut self.inmem[at..at + n], tmp3x);
            }
            _ => self.inmem[at..at + subframe].copy_from_slice(tmp),
        }
        ret * (1.0 / 32768.0 / 32768.0)
    }

    /// libopus `tonality_analysis`: add `len` samples (at the API rate) from
    /// `offset` of the frame and, once 20 ms at 24 kHz are buffered past the
    /// window's overlap, analyse them into the next `info` slot.
    fn tonality_analysis(
        &mut self,
        kfft: &KissFftState,
        x: AnalysisPcm<'_>,
        len: usize,
        offset: usize,
        channels: usize,
        lsb_depth: i32,
    ) {
        const N: usize = 480;
        const N2: usize = 240;
        let pi4 = (core::f64::consts::PI
            * core::f64::consts::PI
            * core::f64::consts::PI
            * core::f64::consts::PI) as f32;

        if !self.initialized {
            self.mem_fill = 240;
            self.initialized = true;
        }
        let alpha = 1.0 / (10.min(1 + self.count)) as f32;
        let alpha_e = 1.0 / (25.min(1 + self.count)) as f32;
        // Noise-floor decay for the bandwidth detection: -2.2 dB/second.
        let mut alpha_e2 = 1.0 / (100.min(1 + self.count)) as f32;
        if self.count <= 1 {
            alpha_e2 = 1.0;
        }

        // `len` and `offset` at 24 kHz from here on.
        let (len, offset) = match self.fs {
            48000 => (len / 2, offset / 2),
            16000 => (3 * len / 2, 3 * offset / 2),
            _ => (len, offset),
        };

        let fill = self.mem_fill;
        let take = len.min(ANALYSIS_BUF_SIZE - fill);
        self.hp_ener_accum += self.downmix_and_resample(x, fill, take, offset, channels);
        if self.mem_fill + len < ANALYSIS_BUF_SIZE {
            self.mem_fill += len;
            // Not enough to update the analysis.
            return;
        }
        let hp_ener = self.hp_ener_accum;
        let info_pos = self.write_pos;
        self.write_pos += 1;
        if self.write_pos >= DETECT_SIZE {
            self.write_pos -= DETECT_SIZE;
        }

        let sample_max = self.inmem.iter().fold(0.0f32, |m, &v| max32(m, v.abs()));
        let is_silence = sample_max <= 1.0 / (1u32 << lsb_depth) as f32;

        let mut fft_in = [KissCpx::default(); N];
        let mut out = [KissCpx::default(); N];
        let mut tonality = [0.0f32; 240];
        let mut noisiness = [0.0f32; 240];
        for i in 0..N2 {
            let w = ANALYSIS_WINDOW[i];
            fft_in[i].r = w * self.inmem[i];
            fft_in[i].i = w * self.inmem[N2 + i];
            fft_in[N - i - 1].r = w * self.inmem[N - i - 1];
            fft_in[N - i - 1].i = w * self.inmem[N + N2 - i - 1];
        }
        self.inmem
            .copy_within(ANALYSIS_BUF_SIZE - 240..ANALYSIS_BUF_SIZE, 0);
        let remaining = len - (ANALYSIS_BUF_SIZE - self.mem_fill);
        self.hp_ener_accum = self.downmix_and_resample(
            x,
            240,
            remaining,
            offset + ANALYSIS_BUF_SIZE - self.mem_fill,
            channels,
        );
        self.mem_fill = 240 + remaining;
        if is_silence {
            // On silence, copy the previous analysis.
            let prev_pos = (self.write_pos + DETECT_SIZE - 2) % DETECT_SIZE;
            self.info[info_pos] = self.info[prev_pos];
            return;
        }
        opus_fft(kfft, &fft_in, &mut out);
        // Any NaN on the input makes the whole output NaN.
        if out[0].r.is_nan() {
            self.info[info_pos].valid = false;
            return;
        }

        let mut tonality2 = [0.0f32; 240];
        for i in 1..N2 {
            let x1r = out[i].r + out[N - i].r;
            let x1i = out[i].i - out[N - i].i;
            let x2r = out[i].i + out[N - i].i;
            let x2i = out[N - i].r - out[i].r;

            let angle = (0.5 / core::f64::consts::PI) as f32 * fast_atan2f(x1i, x1r);
            let d_angle = angle - self.angle[i];
            let d2_angle = d_angle - self.d_angle[i];

            let angle2 = (0.5 / core::f64::consts::PI) as f32 * fast_atan2f(x2i, x2r);
            let d_angle2 = angle2 - angle;
            let d2_angle2 = d_angle2 - d_angle;

            let mut mod1 = d2_angle - float2int(d2_angle) as f32;
            noisiness[i] = mod1.abs();
            mod1 *= mod1;
            mod1 *= mod1;

            let mut mod2 = d2_angle2 - float2int(d2_angle2) as f32;
            noisiness[i] += mod2.abs();
            mod2 *= mod2;
            mod2 *= mod2;

            let avg_mod = 0.25 * (self.d2_angle[i] + mod1 + 2.0 * mod2);
            // An extra delay of two frames in the detection.
            tonality[i] = 1.0 / (1.0 + 40.0 * 16.0 * pi4 * avg_mod) - 0.015;
            // No delay, but less reliable.
            tonality2[i] = 1.0 / (1.0 + 40.0 * 16.0 * pi4 * mod2) - 0.015;

            self.angle[i] = angle2;
            self.d_angle[i] = d_angle2;
            self.d2_angle[i] = mod2;
        }
        for i in 2..N2 - 1 {
            let tt = min32(tonality2[i], max32(tonality2[i - 1], tonality2[i + 1]));
            tonality[i] = 0.9 * max32(tonality[i], tt - 0.1);
        }
        let mut frame_tonality = 0.0f32;
        let mut max_frame_tonality = 0.0f32;
        self.info[info_pos].activity = 0.0;
        let mut frame_noisiness = 0.0f32;
        let mut frame_stationarity = 0.0f32;
        if self.count == 0 {
            for b in 0..NB_TBANDS {
                self.low_e[b] = 1e10;
                self.high_e[b] = -1e10;
            }
        }
        let mut relative_e = 0.0f32;
        let mut frame_loudness = 0.0f32;
        let mut band_log2 = [0.0f32; NB_TBANDS + 1];
        let mut log_e = [0.0f32; NB_TBANDS];
        let mut band_tonality = [0.0f32; NB_TBANDS];
        let mut slope = 0.0f32;
        let bin_e = |i: usize| {
            out[i].r * out[i].r
                + out[N - i].r * out[N - i].r
                + out[i].i * out[i].i
                + out[N - i].i * out[N - i].i
        };
        let scale_ener = |e: f32| (1.0 / 32768.0 / 32768.0) * e;
        // The energy of the very first band is special because of DC.
        {
            let x1r = 2.0 * out[0].r;
            let x2r = 2.0 * out[0].i;
            let mut e = x1r * x1r + x2r * x2r;
            for i in 1..4 {
                e += bin_e(i);
            }
            let e = scale_ener(e);
            band_log2[0] = 0.5 * core::f32::consts::LOG2_E * logd(e + 1e-10);
        }
        for b in 0..NB_TBANDS {
            let mut e = 0.0f32;
            let mut t_e = 0.0f32;
            let mut n_e = 0.0f32;
            for i in TBANDS[b]..TBANDS[b + 1] {
                let be = scale_ener(bin_e(i));
                e += be;
                t_e += be * max32(0.0, tonality[i]);
                n_e += be * 2.0 * (0.5 - noisiness[i]);
            }
            // Extreme band energies could cause NaNs later.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(e < 1e9) || e.is_nan() {
                self.info[info_pos].valid = false;
                return;
            }

            self.e[self.e_count][b] = e;
            frame_noisiness += n_e / (1e-15 + e);

            frame_loudness += sqrtd(f64::from(e + 1e-10));
            log_e[b] = logd(e + 1e-10);
            band_log2[b + 1] = 0.5 * core::f32::consts::LOG2_E * logd(e + 1e-10);
            self.log_e[self.e_count][b] = log_e[b];
            if self.count == 0 {
                self.high_e[b] = log_e[b];
                self.low_e[b] = log_e[b];
            }
            if f64::from(self.high_e[b]) > f64::from(self.low_e[b]) + 7.5 {
                if self.high_e[b] - log_e[b] > log_e[b] - self.low_e[b] {
                    self.high_e[b] -= 0.01;
                } else {
                    self.low_e[b] += 0.01;
                }
            }
            if log_e[b] > self.high_e[b] {
                self.high_e[b] = log_e[b];
                self.low_e[b] = max32(self.high_e[b] - 15.0, self.low_e[b]);
            } else if log_e[b] < self.low_e[b] {
                self.low_e[b] = log_e[b];
                self.high_e[b] = min32(self.low_e[b] + 15.0, self.high_e[b]);
            }
            relative_e += (log_e[b] - self.low_e[b]) / (1e-5 + (self.high_e[b] - self.low_e[b]));

            let mut l1 = 0.0f32;
            let mut l2 = 0.0f32;
            for i in 0..NB_FRAMES {
                l1 += sqrtd(f64::from(self.e[i][b]));
                l2 += self.e[i][b];
            }

            let mut stationarity =
                min32(0.99, l1 / sqrtd(1e-15 + f64::from(NB_FRAMES as f32 * l2)));
            stationarity *= stationarity;
            stationarity *= stationarity;
            frame_stationarity += stationarity;
            band_tonality[b] = max32(t_e / (1e-15 + e), stationarity * self.prev_band_tonality[b]);
            frame_tonality += band_tonality[b];
            if b >= NB_TBANDS - NB_TONAL_SKIP_BANDS {
                frame_tonality -= band_tonality[b + NB_TONAL_SKIP_BANDS - NB_TBANDS];
            }
            max_frame_tonality = max32(
                max_frame_tonality,
                (1.0 + 0.03 * (b as i32 - NB_TBANDS as i32) as f32) * frame_tonality,
            );
            slope += band_tonality[b] * (b as i32 - 8) as f32;
            self.prev_band_tonality[b] = band_tonality[b];
        }

        let mut leakage_from = [0.0f32; NB_TBANDS + 1];
        let mut leakage_to = [0.0f32; NB_TBANDS + 1];
        leakage_from[0] = band_log2[0];
        leakage_to[0] = band_log2[0] - LEAKAGE_OFFSET;
        for b in 1..NB_TBANDS + 1 {
            let leak_slope = LEAKAGE_SLOPE * (TBANDS[b] - TBANDS[b - 1]) as f32 / 4.0;
            leakage_from[b] = min32(leakage_from[b - 1] + leak_slope, band_log2[b]);
            leakage_to[b] = max32(
                leakage_to[b - 1] - leak_slope,
                band_log2[b] - LEAKAGE_OFFSET,
            );
        }
        for b in (0..=NB_TBANDS - 2).rev() {
            let leak_slope = LEAKAGE_SLOPE * (TBANDS[b + 1] - TBANDS[b]) as f32 / 4.0;
            leakage_from[b] = min32(leakage_from[b + 1] + leak_slope, leakage_from[b]);
            leakage_to[b] = max32(leakage_to[b + 1] - leak_slope, leakage_to[b]);
        }
        for b in 0..NB_TBANDS + 1 {
            // The boost needed to overcome the analysis leakage from louder
            // neighbouring bands into a weaker band, and for a loud band whose
            // quantization noise leaks into its weaker neighbours.
            let boost = max32(0.0, leakage_to[b] - band_log2[b])
                + max32(0.0, band_log2[b] - (leakage_from[b] + LEAKAGE_OFFSET));
            self.info[info_pos].leak_boost[b] =
                255.min((0.5 + 64.0 * f64::from(boost)).floor() as i32) as u8;
        }

        let mut spec_variability = 0.0f32;
        for i in 0..NB_FRAMES {
            let mut mindist = 1e15f32;
            for j in 0..NB_FRAMES {
                let mut dist = 0.0f32;
                for k in 0..NB_TBANDS {
                    let tmp = self.log_e[i][k] - self.log_e[j][k];
                    dist += tmp * tmp;
                }
                if j != i {
                    mindist = min32(mindist, dist);
                }
            }
            spec_variability += mindist;
        }
        spec_variability = sqrtd(f64::from(
            spec_variability / NB_FRAMES as f32 / NB_TBANDS as f32,
        ));
        let mut bandwidth_mask = 0.0f32;
        let mut bandwidth = 0i32;
        let mut max_e = 0.0f32;
        let mut noise_floor = 5.7e-4 / (1i32 << 0.max(lsb_depth - 8)) as f32;
        noise_floor *= noise_floor;
        let mut below_max_pitch = 0.0f32;
        let mut above_max_pitch = 0.0f32;
        let mut is_masked = [false; NB_TBANDS + 1];
        for b in 0..NB_TBANDS {
            let band_start = TBANDS[b];
            let band_end = TBANDS[b + 1];
            let mut e = 0.0f32;
            for i in band_start..band_end {
                e += bin_e(i);
            }
            let e = scale_ener(e);
            max_e = max32(max_e, e);
            if band_start < 64 {
                below_max_pitch += e;
            } else {
                above_max_pitch += e;
            }
            self.mean_e[b] = max32((1.0 - alpha_e2) * self.mean_e[b], e);
            let em = max32(e, self.mean_e[b]);
            // A band is active if it is within 90 dB of the peak band and
            // above the PCM noise floor (b + 1: tbands misses CELT band 0).
            let width = (band_end - band_start) as f32;
            if e * 1e9 > max_e && (em > 3.0 * noise_floor * width || e > noise_floor * width) {
                bandwidth = b as i32 + 1;
            }
            // Whether the band is masked (13 dB/Bark spreading follower).
            let mask_ratio = if self.prev_bandwidth > b as i32 {
                0.01
            } else {
                0.05
            };
            is_masked[b] = e < mask_ratio * bandwidth_mask;
            bandwidth_mask = max32(0.05 * bandwidth_mask, e);
        }
        // The last two bands have no spectrum, only the energy above 12 kHz,
        // whose high-pass leaks some energy from below: a higher threshold.
        if self.fs == 48000 {
            let e = hp_ener * (1.0 / (60.0 * 60.0));
            let noise_ratio = if self.prev_bandwidth == 20 {
                10.0
            } else {
                30.0
            };
            above_max_pitch += e;
            let b = NB_TBANDS;
            self.mean_e[b] = max32((1.0 - alpha_e2) * self.mean_e[b], e);
            let em = max32(e, self.mean_e[b]);
            if em > 3.0 * noise_ratio * noise_floor * 160.0 || e > noise_ratio * noise_floor * 160.0
            {
                bandwidth = 20;
            }
            let mask_ratio = if self.prev_bandwidth == 20 {
                0.01
            } else {
                0.05
            };
            is_masked[b] = e < mask_ratio * bandwidth_mask;
        }
        self.info[info_pos].max_pitch_ratio = if above_max_pitch > below_max_pitch {
            below_max_pitch / above_max_pitch
        } else {
            1.0
        };
        // Resampling aliasing can leave a little energy in the first band cut,
        // so a masked last band does not count.
        if bandwidth == 20 && is_masked[NB_TBANDS] {
            bandwidth -= 2;
        } else if bandwidth > 0
            && bandwidth <= NB_TBANDS as i32
            && is_masked[bandwidth as usize - 1]
        {
            bandwidth -= 1;
        }
        if self.count <= 2 {
            bandwidth = 20;
        }
        frame_loudness = 20.0 * f64::from(frame_loudness).log10() as f32;
        self.e_tracker = max32(self.e_tracker - 0.003, frame_loudness);
        self.low_e_count *= 1.0 - alpha_e;
        if frame_loudness < self.e_tracker - 30.0 {
            self.low_e_count += alpha_e;
        }

        let mut bfcc = [0.0f32; 8];
        let mut mid_e = [0.0f32; 8];
        for i in 0..8 {
            let mut sum = 0.0f32;
            for b in 0..16 {
                sum += DCT_TABLE[i * 16 + b] * log_e[b];
            }
            bfcc[i] = sum;
        }
        for i in 0..8 {
            let mut sum = 0.0f32;
            for b in 0..16 {
                sum += DCT_TABLE[i * 16 + b] * 0.5 * (self.high_e[b] + self.low_e[b]);
            }
            mid_e[i] = sum;
        }

        frame_stationarity /= NB_TBANDS as f32;
        relative_e /= NB_TBANDS as f32;
        if self.count < 10 {
            relative_e = 0.5;
        }
        frame_noisiness /= NB_TBANDS as f32;
        let activity = frame_noisiness + (1.0 - frame_noisiness) * relative_e;
        self.info[info_pos].activity = activity;
        frame_tonality = max_frame_tonality / (NB_TBANDS - NB_TONAL_SKIP_BANDS) as f32;
        frame_tonality = max32(frame_tonality, self.prev_tonality * 0.8);
        self.prev_tonality = frame_tonality;

        slope /= 8.0 * 8.0;
        self.info[info_pos].tonality_slope = slope;

        self.e_count = (self.e_count + 1) % NB_FRAMES;
        self.count = (self.count + 1).min(ANALYSIS_COUNT_MAX);
        self.info[info_pos].tonality = frame_tonality;

        let mut features = [0.0f32; 25];
        let mem = &mut self.mem;
        for i in 0..4 {
            features[i] = -0.12299 * (bfcc[i] + mem[i + 24])
                + 0.49195 * (mem[i] + mem[i + 16])
                + 0.69693 * mem[i + 8]
                - 1.4349 * self.cmean[i];
        }
        for i in 0..4 {
            self.cmean[i] = (1.0 - alpha) * self.cmean[i] + alpha * bfcc[i];
        }
        for i in 0..4 {
            features[4 + i] = 0.63246 * (bfcc[i] - mem[i + 24]) + 0.31623 * (mem[i] - mem[i + 16]);
        }
        for i in 0..3 {
            features[8 + i] = 0.53452 * (bfcc[i] + mem[i + 24])
                - 0.26726 * (mem[i] + mem[i + 16])
                - 0.53452 * mem[i + 8];
        }
        if self.count > 5 {
            for i in 0..9 {
                self.std[i] = (1.0 - alpha) * self.std[i] + alpha * features[i] * features[i];
            }
        }
        for i in 0..4 {
            features[i] = bfcc[i] - mid_e[i];
        }
        for i in 0..8 {
            mem[i + 24] = mem[i + 16];
            mem[i + 16] = mem[i + 8];
            mem[i + 8] = mem[i];
            mem[i] = bfcc[i];
        }
        for i in 0..9 {
            features[11 + i] = sqrtd(f64::from(self.std[i])) - STD_FEATURE_BIAS[i];
        }
        features[18] = spec_variability - 0.78;
        features[20] = frame_tonality - 0.154723;
        features[21] = activity - 0.724643;
        features[22] = frame_stationarity - 0.743717;
        features[23] = slope + 0.069216;
        features[24] = self.low_e_count - 0.067930;

        let (music_prob, activity_probability) =
            compute_probabilities(&features, &mut self.rnn_state);
        let info = &mut self.info[info_pos];
        // Probability of speech or music rather than noise.
        info.activity_probability = activity_probability;
        info.music_prob = music_prob;
        info.bandwidth = bandwidth;
        self.prev_bandwidth = bandwidth;
        info.noisiness = frame_noisiness;
        info.valid = true;
    }

    /// libopus `tonality_get_info`: the analysis of the frame of `len`
    /// samples being coded, with the tonality, bandwidth and music
    /// probabilities of the neighbouring analysis frames folded in.
    pub(crate) fn get_info(&mut self, len: usize) -> AnalysisInfo {
        let det = DETECT_SIZE as isize;
        let write_pos = self.write_pos as isize;
        let mut pos = self.read_pos as isize;
        let mut curr_lookahead = write_pos - self.read_pos as isize;
        if curr_lookahead < 0 {
            curr_lookahead += det;
        }

        self.read_subframe += (len / (self.fs as usize / 400)) as i32;
        while self.read_subframe >= 8 {
            self.read_subframe -= 8;
            self.read_pos += 1;
        }
        if self.read_pos >= DETECT_SIZE {
            self.read_pos -= DETECT_SIZE;
        }

        // On long frames, look at the second analysis window rather than the
        // first.
        if len > self.fs as usize / 50 && pos != write_pos {
            pos += 1;
            if pos == det {
                pos = 0;
            }
        }
        if pos == write_pos {
            pos -= 1;
        }
        if pos < 0 {
            pos = det - 1;
        }
        let pos0 = pos;
        let mut info_out = self.info[pos as usize];
        if !info_out.valid {
            return info_out;
        }
        let mut tonality_max = info_out.tonality;
        let mut tonality_avg = info_out.tonality;
        let mut tonality_count = 1;
        // Pick the largest bandwidth among the neighbouring frames, to be safe.
        let mut bandwidth_span = 6;
        // Look ahead for a tone, as the tone detector lags.
        for _ in 0..3 {
            pos += 1;
            if pos == det {
                pos = 0;
            }
            if pos == write_pos {
                break;
            }
            let p = &self.info[pos as usize];
            tonality_max = max32(tonality_max, p.tonality);
            tonality_avg += p.tonality;
            tonality_count += 1;
            info_out.bandwidth = info_out.bandwidth.max(p.bandwidth);
            bandwidth_span -= 1;
        }
        pos = pos0;
        // Look back for a wider bandwidth than the current frame's.
        for _ in 0..bandwidth_span {
            pos -= 1;
            if pos < 0 {
                pos = det - 1;
            }
            if pos == write_pos {
                break;
            }
            info_out.bandwidth = info_out.bandwidth.max(self.info[pos as usize].bandwidth);
        }
        info_out.tonality = max32(tonality_avg / tonality_count as f32, tonality_max - 0.2);

        let mut mpos = pos0;
        let mut vpos = pos0;
        // With enough look-ahead, compensate for the ~5-frame delay of the
        // music probability and the ~1-frame delay of the activity.
        if curr_lookahead > 15 {
            mpos += 5;
            if mpos >= det {
                mpos -= det;
            }
            vpos += 1;
            if vpos >= det {
                vpos -= det;
            }
        }

        // The thresholds for which switching now is optimal against the
        // "badness" of switching later (see libopus): music_prob_min to switch
        // to music when coding speech, music_prob_max the other way.
        let mut prob_min = 1.0f32;
        let mut prob_max = 0.0f32;
        let vad_prob = self.info[vpos as usize].activity_probability;
        let mut prob_count = max32(0.1, vad_prob);
        let mut prob_avg = max32(0.1, vad_prob) * self.info[mpos as usize].music_prob;
        loop {
            mpos += 1;
            if mpos == det {
                mpos = 0;
            }
            if mpos == write_pos {
                break;
            }
            vpos += 1;
            if vpos == det {
                vpos = 0;
            }
            if vpos == write_pos {
                break;
            }
            let pos_vad = self.info[vpos as usize].activity_probability;
            prob_min = min32(
                (prob_avg - TRANSITION_PENALTY * (vad_prob - pos_vad)) / prob_count,
                prob_min,
            );
            prob_max = max32(
                (prob_avg + TRANSITION_PENALTY * (vad_prob - pos_vad)) / prob_count,
                prob_max,
            );
            prob_count += max32(0.1, pos_vad);
            prob_avg += max32(0.1, pos_vad) * self.info[mpos as usize].music_prob;
        }
        info_out.music_prob = prob_avg / prob_count;
        prob_min = min32(prob_avg / prob_count, prob_min);
        prob_max = max32(prob_avg / prob_count, prob_max);
        prob_min = max32(prob_min, 0.0);
        prob_max = min32(prob_max, 1.0);

        // Without enough look-ahead, make the best decision possible.
        if curr_lookahead < 10 {
            let mut pmin = prob_min;
            let mut pmax = prob_max;
            pos = pos0;
            // The min and max of the past.
            for _ in 0..(self.count - 1).min(15) {
                pos -= 1;
                if pos < 0 {
                    pos = det - 1;
                }
                pmin = min32(pmin, self.info[pos as usize].music_prob);
                pmax = max32(pmax, self.info[pos as usize].music_prob);
            }
            // Bias against switching on active audio.
            pmin = max32(0.0, pmin - 0.1 * vad_prob);
            pmax = min32(1.0, pmax + 0.1 * vad_prob);
            prob_min += (1.0 - 0.1 * curr_lookahead as f32) * (pmin - prob_min);
            prob_max += (1.0 - 0.1 * curr_lookahead as f32) * (pmax - prob_max);
        }
        info_out.music_prob_min = prob_min;
        info_out.music_prob_max = prob_max;
        info_out
    }

    /// libopus `run_analysis`: analyse the `frame_size` samples per channel of
    /// interleaved `pcm`, the frame about to be coded, and return the
    /// analysis for it.
    pub fn run(
        &mut self,
        kfft: &KissFftState,
        pcm: AnalysisPcm<'_>,
        frame_size: usize,
        channels: usize,
        lsb_depth: i32,
    ) -> AnalysisInfo {
        let fs = self.fs as usize;
        let mut analysis_frame_size = frame_size - (frame_size & 1);
        // Avoid wrapping around the analysis buffer.
        analysis_frame_size = analysis_frame_size.min((DETECT_SIZE - 5) * fs / 50);
        let mut pcm_len = analysis_frame_size as i32 - self.analysis_offset;
        let mut offset = self.analysis_offset as usize;
        while pcm_len > 0 {
            let len = (fs / 50).min(pcm_len as usize);
            self.tonality_analysis(kfft, pcm, len, offset, channels, lsb_depth);
            offset += fs / 50;
            pcm_len -= (fs / 50) as i32;
        }
        self.analysis_offset = analysis_frame_size as i32 - frame_size as i32;
        self.get_info(frame_size)
    }

    /// The read position, which a multiframe packet rewinds to read the
    /// analysis one frame at a time (libopus `read_pos`, `read_subframe`).
    pub(crate) fn read_position(&self) -> (usize, i32) {
        (self.read_pos, self.read_subframe)
    }

    pub(crate) fn set_read_position(&mut self, (pos, subframe): (usize, i32)) {
        self.read_pos = pos;
        self.read_subframe = subframe;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float2int_rounds_ties_to_even() {
        for (x, r) in [
            (0.5f32, 0),
            (1.5, 2),
            (2.5, 2),
            (-0.5, 0),
            (-1.5, -2),
            (0.49, 0),
            (-0.51, -1),
        ] {
            assert_eq!(float2int(x), r, "{x}");
        }
    }

    #[test]
    fn fast_atan2f_follows_atan2() {
        for (y, x) in [
            (1.0f32, 1.0f32),
            (-1.0, 2.0),
            (3.0, -0.5),
            (-0.2, -4.0),
            (0.0, 1.0),
        ] {
            assert!((fast_atan2f(y, x) - y.atan2(x)).abs() < 1e-4, "{y} {x}");
        }
    }
}
