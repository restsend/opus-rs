//! CELT packet loss concealment, ported from libopus `celt_decode_lost`
//! (`celt/celt_decoder.c`, float build).
//!
//! The first lost frames after two good ones repeat the last pitch period
//! through an LPC filter, fading with the decay the signal already had
//! ("periodic" concealment). Longer losses, a decoder just reset, and hybrid
//! frames (which conceal only the bands above SILK) fill the bands with noise
//! at the last energies, decaying towards the background ("noise"
//! concealment). The next decoded frame folds the concealed overlap back so it
//! blends with its MDCT.

use super::{
    COMBFILTER_MAXPERIOD, COMBFILTER_MINPERIOD, CeltDecoder, DECODE_BUFFER_SIZE, comb_filter,
    comb_filter_inplace,
};
use crate::bands::{celt_lcg_rand, denormalise_bands, log2amp, renormalise_vector};
use crate::celt_lpc::{autocorr, lpc as celt_lpc};
use crate::pitch::{pitch_downsample, pitch_search};

#[cfg(not(feature = "std"))]
use crate::compat::Math;

/// No frame yet, or a reset.
pub(super) const FRAME_NONE: u8 = 0;
/// Last frame decoded normally.
pub(super) const FRAME_NORMAL: u8 = 1;
/// Last frame concealed with noise.
const FRAME_PLC_NOISE: u8 = 2;
/// Last frame concealed by pitch repetition.
const FRAME_PLC_PERIODIC: u8 = 3;

const PLC_PITCH_LAG_MAX: usize = 720;
const PLC_PITCH_LAG_MIN: usize = 100;
const MAX_PERIOD: usize = 1024;
/// Order of the concealment LPC filter (`CELT_LPC_ORDER`).
pub(super) const LPC_ORDER: usize = 24;

impl CeltDecoder {
    /// Conceal one lost frame of `frame_size` samples at the API rate into
    /// planar `pcm` (`celt_decode_lost` plus deemphasis), from band
    /// `start_band` (17 in hybrid mode) to the last decoded frame's end band.
    ///
    /// Returns `frame_size`, or 0 for a frame size CELT does not have.
    pub fn decode_lost(&mut self, frame_size: usize, pcm: &mut [f32], start_band: usize) -> usize {
        let mode = self.mode;
        let n = frame_size * self.downsample;
        let Some(lm) = (0..=mode.max_lm).find(|&lm| mode.short_mdct_size << lm == n) else {
            return 0;
        };
        let cc = self.channels;
        if pcm.len() < frame_size * cc {
            return 0;
        }

        let periodic = self.plc_duration < 40 && start_band == 0 && !self.skip_plc;
        if periodic {
            self.conceal_periodic(n);
        } else {
            self.conceal_noise(n, lm, start_band, self.end_band);
        }
        self.loss_duration = (self.loss_duration + (1 << lm)).min(10_000);
        self.plc_duration = (self.plc_duration + (1 << lm)).min(10_000);
        self.last_frame_type = if periodic {
            FRAME_PLC_PERIODIC
        } else {
            FRAME_PLC_NOISE
        };

        let mem_len = DECODE_BUFFER_SIZE + mode.overlap;
        let out_syn = DECODE_BUFFER_SIZE - n;
        const SIG_SAT: f32 = 536870911.0;
        const VERY_SMALL: f32 = 1e-30;
        for c in 0..cc {
            let base = c * mem_len;
            // The postfilter memory continues from the concealed signal.
            self.prefilter_mem[c * COMBFILTER_MAXPERIOD..(c + 1) * COMBFILTER_MAXPERIOD]
                .copy_from_slice(
                    &self.decode_mem[base + DECODE_BUFFER_SIZE - COMBFILTER_MAXPERIOD
                        ..base + DECODE_BUFFER_SIZE],
                );
            let coef = mode.preemph[0];
            let mut m = self.preemph_mem[c];
            for i in 0..n {
                let x = self.decode_mem[base + out_syn + i];
                let val = (x + VERY_SMALL + m).clamp(-SIG_SAT, SIG_SAT);
                if i % self.downsample == 0 {
                    pcm[c * frame_size + i / self.downsample] = val * (1.0 / 32768.0);
                }
                m = val * coef;
            }
            self.preemph_mem[c] = m;
        }
        frame_size
    }

    /// Noise concealment: random spectra at decaying band energies.
    fn conceal_noise(&mut self, n: usize, lm: usize, start: usize, end: usize) {
        let mode = self.mode;
        let nb = mode.nb_ebands;
        let overlap = mode.overlap;
        let cc = self.channels;
        let mem_len = DECODE_BUFFER_SIZE + overlap;
        let end = end.min(nb).max(start);
        let eff_end = end.min(mode.eff_ebands).max(start);
        for c in 0..cc {
            let base = c * mem_len;
            self.decode_mem.copy_within(base + n..base + mem_len, base);
        }
        if self.prefilter_and_fold {
            for c in 0..cc {
                self.fold_overlap(c, n);
            }
        }

        let decay = if self.loss_duration == 0 { 1.5 } else { 0.5 };
        for c in 0..cc {
            for i in start..end {
                let e = &mut self.old_band_e[c * nb + i];
                *e = self.background_log_e[c * nb + i].max(*e - decay);
            }
        }

        let mut seed = self.rng;
        let x = &mut self.w_x[..n * cc];
        x.fill(0.0);
        for c in 0..cc {
            for i in start..eff_end {
                let b0 = n * c + ((mode.e_bands[i] as usize) << lm);
                let blen = ((mode.e_bands[i + 1] - mode.e_bands[i]) as usize) << lm;
                for v in &mut x[b0..b0 + blen] {
                    seed = celt_lcg_rand(seed);
                    *v = ((seed as i32) >> 20) as f32;
                }
                renormalise_vector(&mut x[b0..b0 + blen], blen, 1.0);
            }
        }
        self.rng = seed;

        let band_amp = &mut self.w_band_amp[..nb * cc];
        log2amp(mode, nb, band_amp, &self.old_band_e, cc);
        let freq = &mut self.w_freq[..n * cc];
        freq.fill(0.0);
        denormalise_bands(mode, x, freq, band_amp, start, eff_end, cc, 1 << lm);
        if self.downsample > 1 {
            let bound = n / self.downsample;
            for c in 0..cc {
                freq[c * n + bound..(c + 1) * n].fill(0.0);
            }
        }
        let out_syn = DECODE_BUFFER_SIZE - n;
        for c in 0..cc {
            let out = c * mem_len + out_syn;
            mode.mdct.backward(
                &self.w_freq[c * n..],
                &mut self.decode_mem[out..],
                mode.window,
                overlap,
                mode.max_lm - lm,
                1,
            );
        }

        // Run the postfilter with the last parameters.
        self.prefilter_period = self.prefilter_period.max(COMBFILTER_MINPERIOD);
        self.prefilter_period_old = self.prefilter_period_old.max(COMBFILTER_MINPERIOD);
        let short = mode.short_mdct_size;
        for c in 0..cc {
            let mem = &mut self.decode_mem[c * mem_len..(c + 1) * mem_len];
            comb_filter_inplace(
                mem,
                out_syn,
                self.prefilter_period_old,
                self.prefilter_period,
                short,
                self.prefilter_gain_old,
                self.prefilter_gain,
                self.prefilter_tapset_old,
                self.prefilter_tapset,
                mode.window,
                overlap,
            );
            if lm != 0 {
                comb_filter_inplace(
                    mem,
                    out_syn + short,
                    self.prefilter_period,
                    self.prefilter_period,
                    n - short,
                    self.prefilter_gain,
                    self.prefilter_gain,
                    self.prefilter_tapset,
                    self.prefilter_tapset,
                    mode.window,
                    overlap,
                );
            }
        }
        self.prefilter_period_old = self.prefilter_period;
        self.prefilter_gain_old = self.prefilter_gain;
        self.prefilter_tapset_old = self.prefilter_tapset;
        self.prefilter_and_fold = false;
        // Do not repeat the pitch again until two packets arrive in a row.
        self.skip_plc = true;
    }

    /// Periodic concealment: repeat the last pitch period through the LPC
    /// filter of the signal before the loss.
    fn conceal_periodic(&mut self, n: usize) {
        let mode = self.mode;
        let overlap = mode.overlap;
        let window = mode.window;
        let cc = self.channels;
        let mem_len = DECODE_BUFFER_SIZE + overlap;
        let first = self.last_frame_type != FRAME_PLC_PERIODIC;
        let (pitch_index, fade) = if first {
            let pitch = self.plc_pitch_search();
            self.last_pitch_index = pitch;
            (pitch, 1.0f32)
        } else {
            (self.last_pitch_index, 0.8)
        };
        // Excitation of two pitch periods, to see whether the signal decays.
        let exc_length = (2 * pitch_index).min(MAX_PERIOD);

        for c in 0..cc {
            let buf = &mut self.decode_mem[c * mem_len..(c + 1) * mem_len];
            let lpc = &mut self.plc_lpc[c * LPC_ORDER..(c + 1) * LPC_ORDER];
            // exc[i] here is libopus's exc[i - CELT_LPC_ORDER].
            let mut exc = [0.0f32; MAX_PERIOD + LPC_ORDER];
            exc.copy_from_slice(
                &buf[DECODE_BUFFER_SIZE - MAX_PERIOD - LPC_ORDER..DECODE_BUFFER_SIZE],
            );

            if first {
                // LPC of the last MAX_PERIOD samples, to work in the excitation domain.
                let mut ac = [0.0f32; LPC_ORDER + 1];
                autocorr(
                    &exc[LPC_ORDER..],
                    &mut ac,
                    Some(window),
                    overlap,
                    LPC_ORDER,
                    MAX_PERIOD,
                );
                // A noise floor of -40 dB, and lag windowing for a stable recursion.
                ac[0] *= 1.0001;
                for (i, a) in ac.iter_mut().enumerate().skip(1) {
                    *a -= *a * (0.008 * 0.008) * i as f32 * i as f32;
                }
                lpc.fill(0.0);
                celt_lpc(lpc, &ac, LPC_ORDER);
            }

            // Whiten the last exc_length samples (celt_fir with history).
            let start = LPC_ORDER + MAX_PERIOD - exc_length;
            let mut fir = [0.0f32; MAX_PERIOD];
            for (i, out) in fir.iter_mut().enumerate().take(exc_length) {
                let mut sum = exc[start + i];
                for k in (0..LPC_ORDER).rev() {
                    sum += lpc[k] * exc[start + i - k - 1];
                }
                *out = sum;
            }
            exc[start..start + exc_length].copy_from_slice(&fir[..exc_length]);

            // How fast the waveform decays, so concealment adds no energy.
            // (The float build's SHR32 is a no-op: no halving of E1.)
            let decay = {
                let (mut e1, mut e2) = (1.0f32, 1.0f32);
                let decay_length = exc_length >> 1;
                for i in 0..decay_length {
                    let a = exc[LPC_ORDER + MAX_PERIOD - decay_length + i];
                    e1 += a * a;
                    let b = exc[LPC_ORDER + MAX_PERIOD - 2 * decay_length + i];
                    e2 += b * b;
                }
                (e1.min(e2) / e2).sqrt()
            };

            // Make room for the new frame; the old overlap is not used.
            buf.copy_within(n..DECODE_BUFFER_SIZE, 0);

            // Extrapolate one period at a time, each one decayed further.
            let extrapolation_offset = MAX_PERIOD - pitch_index;
            let extrapolation_len = n + overlap;
            let mut attenuation = fade * decay;
            let mut s1 = 0.0f32;
            let mut j = 0usize;
            for i in 0..extrapolation_len {
                if j >= pitch_index {
                    j -= pitch_index;
                    attenuation *= decay;
                }
                buf[DECODE_BUFFER_SIZE - n + i] =
                    attenuation * exc[LPC_ORDER + extrapolation_offset + j];
                // Energy of the decoded signal whose excitation is copied.
                let tmp = buf[DECODE_BUFFER_SIZE - MAX_PERIOD - n + extrapolation_offset + j];
                s1 += tmp * tmp;
                j += 1;
            }

            // Back to the signal domain, continuing from the last decoded samples.
            let mut lpc_mem = [0.0f32; LPC_ORDER];
            for (i, m) in lpc_mem.iter_mut().enumerate() {
                *m = buf[DECODE_BUFFER_SIZE - n - 1 - i];
            }
            let out = &mut buf[DECODE_BUFFER_SIZE - n..DECODE_BUFFER_SIZE - n + extrapolation_len];
            for v in out.iter_mut() {
                let mut sum = *v;
                for (d, m) in lpc.iter().zip(lpc_mem.iter()) {
                    sum -= d * m;
                }
                lpc_mem.copy_within(..LPC_ORDER - 1, 1);
                lpc_mem[0] = sum;
                *v = sum;
            }

            // Attenuate a synthesis louder than what it copies; drop an
            // explosion (written to also catch a NaN, as libopus does).
            let s2: f32 = out.iter().map(|v| v * v).sum();
            if s1.partial_cmp(&(0.2 * s2)) != Some(core::cmp::Ordering::Greater) {
                out.fill(0.0);
            } else if s1 < s2 {
                let ratio = ((s1 + 1.0) / (s2 + 1.0)).sqrt();
                for (i, v) in out.iter_mut().enumerate() {
                    let g = if i < overlap {
                        1.0 - window[i] * (1.0 - ratio)
                    } else {
                        ratio
                    };
                    *v *= g;
                }
            }
        }
        self.prefilter_and_fold = true;
    }

    /// Pitch of the decoded history (`celt_plc_pitch_search`).
    fn plc_pitch_search(&self) -> usize {
        let mem_len = DECODE_BUFFER_SIZE + self.mode.overlap;
        let mut lp = [0.0f32; DECODE_BUFFER_SIZE >> 1];
        let channels: [&[f32]; 2] = [
            &self.decode_mem[..mem_len],
            &self.decode_mem[mem_len.min(self.decode_mem.len())..],
        ];
        pitch_downsample(
            &channels[..self.channels],
            &mut lp,
            DECODE_BUFFER_SIZE >> 1,
            self.channels,
            2,
        );
        whiten_lp(&mut lp);
        let pitch = pitch_search(
            &lp[PLC_PITCH_LAG_MAX >> 1..],
            &lp,
            DECODE_BUFFER_SIZE - PLC_PITCH_LAG_MAX,
            PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN,
        );
        PLC_PITCH_LAG_MAX - pitch
    }

    /// The postfilter parameters, old and current.
    pub(super) fn postfilter(&self) -> Postfilter {
        Postfilter {
            period_old: self.prefilter_period_old,
            period: self.prefilter_period,
            gain_old: self.prefilter_gain_old,
            gain: self.prefilter_gain,
            tapset_old: self.prefilter_tapset_old,
            tapset: self.prefilter_tapset,
        }
    }

    /// Fold the concealed overlap of channel `c`, whose decode memory is
    /// already shifted by `n` (`prefilter_and_fold`).
    fn fold_overlap(&mut self, c: usize, n: usize) {
        let mem_len = DECODE_BUFFER_SIZE + self.mode.overlap;
        let pf = self.postfilter();
        prefilter_and_fold(
            &mut self.decode_mem[c * mem_len..(c + 1) * mem_len],
            n,
            &pf,
            self.mode.window,
            self.mode.overlap,
        );
    }

    /// Keep the energy prediction of the first frame after a loss from
    /// jumping up: continue a falling trend, or take the lowest recent energy
    /// (the `loss_duration` branch of `celt_decode_with_ec`).
    pub(super) fn make_energy_prediction_safe(&mut self, start: usize, end: usize, lm: usize) {
        let nb = self.mode.nb_ebands;
        let missing = (self.loss_duration >> lm).min(10) as f32;
        // Shorter frames fluctuate more: play it safe.
        let safety = match lm {
            0 => 1.5,
            1 => 0.5,
            _ => 0.0,
        };
        for c in 0..2 {
            for i in start..end {
                let k = c * nb + i;
                let (e0, e1, e2) = (self.old_band_e[k], self.old_band_e2[k], self.old_band_e3[k]);
                self.old_band_e[k] = if e0 < e1.max(e2) {
                    let slope = (e1 - e0).max(0.5 * (e2 - e0)).min(2.0);
                    (e0 - ((1.0 + missing) * slope).max(0.0)).max(-20.0)
                } else {
                    e0.min(e1).min(e2)
                } - safety;
            }
        }
    }
}

/// Postfilter parameters, old and current (`postfilter_*` in libopus).
#[derive(Clone, Copy)]
pub(super) struct Postfilter {
    period_old: usize,
    period: usize,
    gain_old: f32,
    gain: f32,
    tapset_old: i32,
    tapset: i32,
}

/// Undo the postfilter on a concealed overlap and simulate TDAC on it, so it
/// blends with the next MDCT (`prefilter_and_fold`). `mem` is one channel's
/// decode memory, already shifted by `n`.
pub(super) fn prefilter_and_fold(
    mem: &mut [f32],
    n: usize,
    pf: &Postfilter,
    window: &[f32],
    overlap: usize,
) {
    let x0 = DECODE_BUFFER_SIZE - n;
    let mut etmp = [0.0f32; 128];
    comb_filter(
        &mut etmp,
        mem,
        0,
        x0,
        pf.period_old,
        pf.period,
        overlap,
        -pf.gain_old,
        -pf.gain,
        pf.tapset_old,
        pf.tapset,
        window,
        overlap,
    );
    for i in 0..overlap / 2 {
        mem[x0 + i] = window[i] * etmp[overlap - 1 - i] + window[overlap - i - 1] * etmp[i];
    }
}

/// The whitening `pitch_downsample` applies after decimating: a 4th-order
/// LPC with an added zero (float build).
fn whiten_lp(x_lp: &mut [f32]) {
    let len = x_lp.len();
    let mut ac = [0.0f32; 5];
    autocorr(x_lp, &mut ac, None, 0, 4, len);
    ac[0] *= 1.0001;
    for (i, a) in ac.iter_mut().enumerate().skip(1) {
        *a -= *a * (0.008 * i as f32) * (0.008 * i as f32);
    }
    let mut lpc = [0.0f32; 4];
    celt_lpc(&mut lpc, &ac, 4);
    let mut tmp = 1.0f32;
    for l in lpc.iter_mut() {
        tmp *= 0.9;
        *l *= tmp;
    }
    let c1 = 0.8f32;
    let lpc2 = [
        lpc[0] + 0.8,
        lpc[1] + c1 * lpc[0],
        lpc[2] + c1 * lpc[1],
        lpc[3] + c1 * lpc[2],
        c1 * lpc[3],
    ];
    let mut mem = [0.0f32; 5];
    for v in x_lp.iter_mut() {
        let sum = *v
            + lpc2[0] * mem[0]
            + lpc2[1] * mem[1]
            + lpc2[2] * mem[2]
            + lpc2[3] * mem[3]
            + lpc2[4] * mem[4];
        mem.copy_within(..4, 1);
        mem[0] = *v;
        *v = sum;
    }
}
