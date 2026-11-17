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
    COMBFILTER_MINPERIOD, CeltDecoder, DECODE_BUFFER_SIZE, PcmLayout, comb_filter,
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
/// Last frame concealed by deep PLC.
#[cfg(feature = "deep-plc")]
const FRAME_PLC_NEURAL: u8 = 4;
/// Last frame rebuilt from DRED.
#[cfg(feature = "dred")]
const FRAME_DRED: u8 = 5;

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
        self.decode_lost_into(frame_size, pcm, start_band, PcmLayout::Planar, None)
    }

    /// [`Self::decode_lost`] with the output written as `layout` says and
    /// the deep PLC state of the Opus decoder (libopus passes `lpcnet` to
    /// `celt_decode_lost`).
    pub(crate) fn decode_lost_into(
        &mut self,
        frame_size: usize,
        pcm: &mut [f32],
        start_band: usize,
        layout: PcmLayout,
        lpcnet: crate::silk::plc::DeepPlc<'_>,
    ) -> usize {
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
        #[allow(unused_mut)]
        let mut frame_type = if periodic {
            FRAME_PLC_PERIODIC
        } else {
            FRAME_PLC_NOISE
        };
        #[cfg(feature = "deep-plc")]
        if start_band == 0
            && let Some(l) = lpcnet.as_deref()
        {
            if self.deep_plc.complexity >= 5 && self.plc_duration < 80 && !self.skip_plc {
                frame_type = FRAME_PLC_NEURAL;
            }
            #[cfg(feature = "dred")]
            if l.fec_fill_pos > l.fec_read_pos {
                frame_type = FRAME_DRED;
            }
            #[cfg(not(feature = "dred"))]
            let _ = l;
        }
        if frame_type == FRAME_PLC_NOISE {
            self.conceal_noise(n, lm, start_band, self.end_band);
        } else {
            self.conceal_periodic(n, frame_type, lpcnet);
        }
        self.loss_duration = (self.loss_duration + (1 << lm)).min(10_000);
        self.plc_duration = (self.plc_duration + (1 << lm)).min(10_000);
        #[cfg(feature = "dred")]
        if frame_type == FRAME_DRED {
            self.plc_duration = 0;
            self.skip_plc = false;
        }
        self.last_frame_type = frame_type;

        self.deemphasis(n, pcm, layout);
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

        // 3 dB per 20 ms, and on the first noise frame the decay the
        // periodic or neural concealment before it skipped.
        let mut decay = (1 << lm) as f32 * 0.0625;
        if self.last_frame_type != FRAME_PLC_NOISE {
            decay += 1f32.max(self.loss_duration.min(160) as f32 * 0.0625);
        }
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
                renormalise_vector(&mut x[b0..b0 + blen], blen, 1.0, self.arch);
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
        self.postfilter_pitch = self.postfilter_pitch.max(COMBFILTER_MINPERIOD);
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
    fn conceal_periodic(
        &mut self,
        n: usize,
        frame_type: u8,
        lpcnet: crate::silk::plc::DeepPlc<'_>,
    ) {
        let mode = self.mode;
        let overlap = mode.overlap;
        let window = mode.window;
        let cc = self.channels;
        let mem_len = DECODE_BUFFER_SIZE + overlap;
        let curr_neural = is_neural(frame_type);
        let last_neural = is_neural(self.last_frame_type);
        let first = self.last_frame_type != FRAME_PLC_PERIODIC && !(last_neural && curr_neural);
        let (pitch_index, fade) = if first {
            let pitch = self.plc_pitch_search();
            self.last_pitch_index = pitch;
            (pitch, 1.0f32)
        } else {
            // Fade once a whole pitch period is concealed.
            let concealed = self.plc_duration as usize * mode.short_mdct_size;
            let fade = if concealed >= self.last_pitch_index {
                0.8
            } else {
                1.0
            };
            (self.last_pitch_index, fade)
        };
        #[cfg(feature = "deep-plc")]
        let mut lpcnet = lpcnet;
        #[cfg(feature = "deep-plc")]
        if curr_neural
            && !last_neural
            && let Some(l) = lpcnet.as_deref_mut()
        {
            deep_plc::update_plc_state(
                l,
                &self.decode_mem,
                mem_len,
                &mut self.deep_plc.preemphasis_mem,
                cc,
            );
        }
        #[cfg(not(feature = "deep-plc"))]
        let _ = lpcnet;
        // Excitation of two pitch periods, to see whether the signal
        // decays, and at least MAX_PERIOD to look for pitch multiples.
        let exc_length = MAX_PERIOD.max(2 * pitch_index);
        // sig[i] here is libopus's sig[i - CELT_LPC_ORDER].
        let mut sig = [0.0f32; 2 * PLC_PITCH_LAG_MAX + LPC_ORDER];
        let sig = &mut sig[..exc_length + LPC_ORDER];
        let mut exc = [0.0f32; 2 * PLC_PITCH_LAG_MAX];
        let exc = &mut exc[..exc_length];
        // After a decoded frame, the extrapolation completes its folded
        // right overlap.
        let unfold = self.last_frame_type == FRAME_NORMAL;
        let pf = self.postfilter();

        for c in 0..cc {
            let mut pitch_index = pitch_index;
            let buf = &mut self.decode_mem[c * mem_len..(c + 1) * mem_len];
            let lpc = &mut self.plc_lpc[c * LPC_ORDER..(c + 1) * LPC_ORDER];
            sig.copy_from_slice(
                &buf[DECODE_BUFFER_SIZE - exc_length - LPC_ORDER..DECODE_BUFFER_SIZE],
            );

            if first {
                // LPC of the last MAX_PERIOD samples, to work in the excitation domain.
                let mut ac = [0.0f32; LPC_ORDER + 1];
                autocorr(
                    &sig[LPC_ORDER + exc_length - MAX_PERIOD..],
                    &mut ac,
                    Some(window),
                    overlap,
                    LPC_ORDER,
                    MAX_PERIOD,
                    self.arch,
                );
                // A noise floor of -40 dB, and lag windowing for a stable recursion.
                ac[0] *= 1.0001;
                for (i, a) in ac.iter_mut().enumerate().skip(1) {
                    *a -= *a * (0.008 * 0.008) * i as f32 * i as f32;
                }
                lpc.fill(0.0);
                celt_lpc(lpc, &ac, LPC_ORDER);
            }

            // The excitation: sig whitened (celt_fir with history).
            for (i, out) in exc.iter_mut().enumerate() {
                let mut sum = sig[LPC_ORDER + i];
                for k in (0..LPC_ORDER).rev() {
                    sum += lpc[k] * sig[LPC_ORDER + i - k - 1];
                }
                *out = sum;
            }
            // 1 plus the energy of the pitch period of exc at `from`.
            let energy = |from: usize| {
                exc[from..from + pitch_index]
                    .iter()
                    .fold(1.0f32, |e, &x| e + x * x)
            };

            // How fast the waveform decays, so concealment adds no energy,
            // measured on the first lost frame and kept for the next ones.
            // (The float build's SHR32 is a no-op: no halving of E1.)
            let decay = if first {
                let e1 = energy(exc_length - pitch_index);
                let e2 = energy(exc_length - 2 * pitch_index);
                self.plc_decay[c] = (e1.min(e2) / e2).sqrt();
                self.plc_decay[c]
            } else {
                self.plc_decay[c]
            };

            // Repeat the largest multiple of the pitch period whose periods
            // all stay within 6 dB of the last one, so a short period does
            // not repeat many times.
            let e0 = energy(exc_length - pitch_index);
            let k_max = exc_length / pitch_index;
            let k = (2..=k_max)
                .find(|&k| {
                    let ek = energy(exc_length - k * pitch_index);
                    ek < 0.25 * e0 || e0 < 0.25 * ek
                })
                .unwrap_or(k_max + 1);
            pitch_index *= k - 1;

            // Make room for the new frame, keeping the folded overlap.
            let mut old_overlap = [0.0f32; 64];
            if unfold {
                old_overlap[..overlap / 2]
                    .copy_from_slice(&buf[DECODE_BUFFER_SIZE..DECODE_BUFFER_SIZE + overlap / 2]);
            }
            buf.copy_within(n..DECODE_BUFFER_SIZE, 0);

            // Extrapolate one period at a time, each one decayed further.
            let extrapolation_offset = exc_length - pitch_index;
            let extrapolation_len = n + overlap;
            let mut attenuation = fade * decay;
            let mut s1 = 0.0f32;
            let mut j = 0usize;
            for i in 0..extrapolation_len {
                if j >= pitch_index {
                    j -= pitch_index;
                    attenuation *= decay;
                }
                buf[DECODE_BUFFER_SIZE - n + i] = attenuation * exc[extrapolation_offset + j];
                // Expected energy of the attenuated signal whose excitation
                // is copied.
                let tmp = attenuation * sig[LPC_ORDER + extrapolation_offset + j];
                s1 += tmp * tmp;
                j += 1;
            }

            // Back to the signal domain, continuing from the last decoded samples.
            let mut lpc_mem = [0.0f32; LPC_ORDER];
            for (i, m) in lpc_mem.iter_mut().enumerate() {
                *m = buf[DECODE_BUFFER_SIZE - n - 1 - i];
            }
            let out = &mut buf[DECODE_BUFFER_SIZE - n..DECODE_BUFFER_SIZE - n + extrapolation_len];
            celt_iir(out, lpc, &lpc_mem);

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

            if unfold {
                // Unfold the old overlap with the TDAC complement of the
                // extrapolation, both before the postfilter.
                let x0 = DECODE_BUFFER_SIZE - n;
                let mut etmp = [0.0f32; 128];
                comb_filter(
                    &mut etmp,
                    buf,
                    0,
                    x0,
                    pf.period_old,
                    pf.period_old,
                    overlap,
                    -pf.gain_old,
                    -pf.gain_old,
                    pf.tapset_old,
                    pf.tapset_old,
                    &[],
                    0,
                );
                for i in 0..overlap / 2 {
                    let (w1, w2) = (window[i], window[overlap - 1 - i]);
                    let x2 = old_overlap[i];
                    let x1 = w2 * etmp[overlap - 1 - i] - w1 * etmp[i];
                    buf[x0 + i] = w2 * x2 - w1 * x1;
                    buf[x0 + overlap - 1 - i] = w1 * x2 + w2 * x1;
                }
                comb_filter_inplace(
                    buf,
                    x0,
                    pf.period_old,
                    pf.period_old,
                    overlap,
                    pf.gain_old,
                    pf.gain_old,
                    pf.tapset_old,
                    pf.tapset_old,
                    &[],
                    0,
                );
            }
        }
        #[cfg(feature = "deep-plc")]
        if curr_neural && let Some(l) = lpcnet {
            deep_plc::synthesize(self, l, n, last_neural);
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
        // `pitch_downsample` ends with the whitening FIR (libopus
        // pitch.c:192-229), so no extra whitening happens before
        // `pitch_search` (`celt_plc_pitch_search` has none).
        pitch_downsample(
            &channels[..self.channels],
            &mut lp,
            DECODE_BUFFER_SIZE >> 1,
            self.channels,
            2,
            self.arch,
        );
        let pitch = pitch_search(
            &lp[PLC_PITCH_LAG_MAX >> 1..],
            &lp,
            DECODE_BUFFER_SIZE - PLC_PITCH_LAG_MAX,
            PLC_PITCH_LAG_MAX - PLC_PITCH_LAG_MIN,
            self.arch,
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
        // libopus passes no window and no overlap: every sample is
        // filtered with the current parameters (celt_decoder.c:606-609).
        &[],
        0,
    );
    for i in 0..overlap / 2 {
        mem[x0 + i] = window[i] * etmp[overlap - 1 - i] + window[overlap - i - 1] * etmp[i];
    }
}

/// libopus `celt_iir` (celt/celt_lpc.c:221-276), in place: `x` filtered by
/// 1/A(z) with `den` and the past outputs `mem` (newest first), in libopus's
/// summation order. Outputs come four at a time: each sums the history
/// known when the group starts, oldest sample first, then adds the group's
/// own earlier outputs, newest first.
fn celt_iir(x: &mut [f32], den: &[f32], mem: &[f32; LPC_ORDER]) {
    const ORD: usize = LPC_ORDER;
    let n = x.len();
    let mut rden = [0.0f32; ORD];
    for i in 0..ORD {
        rden[i] = den[ORD - i - 1];
    }
    // y[i] is the output i samples before x[0] counted from ORD down; zero
    // where not computed yet.
    let mut y = [0.0f32; ORD + MAX_PERIOD + 2 * 120];
    for i in 0..ORD {
        y[i] = mem[ORD - i - 1];
    }
    let mut i = 0;
    while i + 3 < n {
        let mut sum = [x[i], x[i + 1], x[i + 2], x[i + 3]];
        for j in 0..ORD {
            for (k, s) in sum.iter_mut().enumerate() {
                *s -= rden[j] * y[i + j + k];
            }
        }
        y[i + ORD] = sum[0];
        sum[1] -= y[i + ORD] * den[0];
        y[i + ORD + 1] = sum[1];
        sum[2] -= y[i + ORD + 1] * den[0];
        sum[2] -= y[i + ORD] * den[1];
        y[i + ORD + 2] = sum[2];
        sum[3] -= y[i + ORD + 2] * den[0];
        sum[3] -= y[i + ORD + 1] * den[1];
        sum[3] -= y[i + ORD] * den[2];
        y[i + ORD + 3] = sum[3];
        x[i..i + 4].copy_from_slice(&sum);
        i += 4;
    }
    while i < n {
        let mut sum = x[i];
        for j in 0..ORD {
            sum -= rden[j] * y[i + j];
        }
        y[i + ORD] = sum;
        x[i] = sum;
        i += 1;
    }
}

/// Whether a frame type is a deep PLC or DRED frame.
fn is_neural(frame_type: u8) -> bool {
    #[cfg(feature = "deep-plc")]
    if frame_type == FRAME_PLC_NEURAL {
        return true;
    }
    #[cfg(feature = "dred")]
    if frame_type == FRAME_DRED {
        return true;
    }
    let _ = frame_type;
    false
}

/// The deep PLC part of libopus `celt_decode_lost` (ENABLE_DEEP_PLC): the
/// concealed frame comes from the LPCNet PLC at 16 kHz, resampled to 48 kHz.
#[cfg(feature = "deep-plc")]
pub(super) mod deep_plc {
    use super::{CeltDecoder, DECODE_BUFFER_SIZE};
    #[cfg(not(feature = "std"))]
    use crate::compat::Math;
    use crate::dnn::freq::{FRAME_SIZE, PREEMPHASIS};
    use crate::dnn::lpcnet_plc::LpcnetPlcState;
    use crate::dnn::vec::{max32, min32};

    const PLC_UPDATE_FRAMES: usize = 4;
    const PLC_UPDATE_SAMPLES: usize = PLC_UPDATE_FRAMES * FRAME_SIZE;
    const SINC_ORDER: usize = 48;
    /// `sinc_filter` of celt_decoder.c.
    const SINC_FILTER: [f32; SINC_ORDER + 1] = [
        4.2931e-05,
        -0.000190293,
        -0.000816132,
        -0.000637162,
        0.00141662,
        0.00354764,
        0.00184368,
        -0.00428274,
        -0.00856105,
        -0.0034003,
        0.00930201,
        0.0159616,
        0.00489785,
        -0.0169649,
        -0.0259484,
        -0.00596856,
        0.0286551,
        0.0405872,
        0.00649994,
        -0.0509284,
        -0.0716655,
        -0.00665212,
        0.134336,
        0.278927,
        0.339995,
        0.278927,
        0.134336,
        -0.00665212,
        -0.0716655,
        -0.0509284,
        0.00649994,
        0.0405872,
        0.0286551,
        -0.00596856,
        -0.0259484,
        -0.0169649,
        0.00489785,
        0.0159616,
        0.00930201,
        -0.0034003,
        -0.00856105,
        -0.00428274,
        0.00184368,
        0.00354764,
        0.00141662,
        -0.000637162,
        -0.000816132,
        -0.000190293,
        4.2931e-05,
    ];

    /// The deep PLC fields of libopus's CELTDecoder (`complexity`,
    /// `plc_pcm`, `plc_fill`, `plc_preemphasis_mem`).
    pub(crate) struct DeepPlcMem {
        /// The decoder complexity (`OPUS_SET_COMPLEXITY`), kept by a reset.
        pub(crate) complexity: i32,
        plc_pcm: [i16; PLC_UPDATE_SAMPLES],
        plc_fill: usize,
        pub(super) preemphasis_mem: f32,
    }

    impl DeepPlcMem {
        pub(crate) const EMPTY: Self = Self {
            complexity: 0,
            plc_pcm: [0; PLC_UPDATE_SAMPLES],
            plc_fill: 0,
            preemphasis_mem: 0.0,
        };

        /// `OPUS_RESET_STATE` (these fields follow `DECODER_RESET_START`).
        pub(crate) fn reset(&mut self) {
            *self = Self {
                complexity: self.complexity,
                ..Self::EMPTY
            };
        }
    }

    /// `float2int` (lrintf): round to nearest, ties to even.
    fn float2int(x: f32) -> i32 {
        let f = x.floor();
        let d = x - f;
        (if d > 0.5 || (d == 0.5 && (f as i64) % 2 != 0) {
            f + 1.0
        } else {
            f
        }) as i32
    }

    /// `update_plc_state`: feeds the last 40 ms of the decoded signal,
    /// downsampled to 16 kHz, to the deep PLC without consuming DRED
    /// features.
    pub(super) fn update_plc_state(
        lpcnet: &mut LpcnetPlcState<'static>,
        decode_mem: &[f32],
        mem_len: usize,
        preemphasis_mem: &mut f32,
        cc: usize,
    ) {
        let mut buf48k = [0.0f32; DECODE_BUFFER_SIZE];
        let mut buf16k = [0i16; PLC_UPDATE_SAMPLES];
        if cc == 1 {
            buf48k.copy_from_slice(&decode_mem[..DECODE_BUFFER_SIZE]);
        } else {
            for (i, b) in buf48k.iter_mut().enumerate() {
                *b = (0.5 * (decode_mem[i] + decode_mem[mem_len + i]) as f64) as f32;
            }
        }
        for i in 1..DECODE_BUFFER_SIZE {
            buf48k[i] += PREEMPHASIS * buf48k[i - 1];
        }
        *preemphasis_mem = buf48k[DECODE_BUFFER_SIZE - 1];
        let offset = DECODE_BUFFER_SIZE - SINC_ORDER - 1 - 3 * (PLC_UPDATE_SAMPLES - 1);
        for (i, o) in buf16k.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (j, &f) in SINC_FILTER.iter().enumerate() {
                sum += buf48k[3 * i + j + offset] * f;
            }
            *o = float2int(min32(32767.0, max32(-32767.0, sum))) as i16;
        }
        let (read_pos, skip) = (lpcnet.fec_read_pos, lpcnet.fec_skip);
        for k in 0..PLC_UPDATE_FRAMES {
            lpcnet.update(&buf16k[FRAME_SIZE * k..]);
        }
        lpcnet.fec_read_pos = read_pos;
        lpcnet.fec_skip = skip;
    }

    /// Replaces the periodic concealment of the frame (and its overlap)
    /// with the deep PLC output resampled to 48 kHz, cross-fading the first
    /// overlap when the previous frame was not neural.
    pub(super) fn synthesize(
        st: &mut CeltDecoder,
        lpcnet: &mut LpcnetPlcState<'static>,
        n: usize,
        last_neural: bool,
    ) {
        let overlap = st.mode.overlap;
        let window = st.mode.window;
        let cc = st.channels;
        let mem_len = DECODE_BUFFER_SIZE + overlap;
        let start = DECODE_BUFFER_SIZE - n;
        let mut buf_copy = [0.0f32; 2 * 120];
        for c in 0..cc {
            buf_copy[c * overlap..(c + 1) * overlap].copy_from_slice(
                &st.decode_mem[c * mem_len + start..c * mem_len + start + overlap],
            );
        }
        let samples_needed16k = (n + SINC_ORDER + overlap) / 3;
        let m = &mut st.deep_plc;
        if !last_neural {
            m.plc_fill = 0;
        }
        while m.plc_fill < samples_needed16k {
            lpcnet.conceal(&mut m.plc_pcm[m.plc_fill..m.plc_fill + FRAME_SIZE]);
            m.plc_fill += FRAME_SIZE;
        }
        let buf = &mut st.decode_mem[..mem_len];
        for i in 0..(n + overlap) / 3 {
            let mut sum = 0.0f32;
            for j in 0..17 {
                sum += (3 * m.plc_pcm[i + j] as i32) as f32 * SINC_FILTER[3 * j];
            }
            buf[start + 3 * i] = sum;
            let mut sum = 0.0f32;
            for j in 0..16 {
                sum += (3 * m.plc_pcm[i + j + 1] as i32) as f32 * SINC_FILTER[3 * j + 2];
            }
            buf[start + 3 * i + 1] = sum;
            let mut sum = 0.0f32;
            for j in 0..16 {
                sum += (3 * m.plc_pcm[i + j + 1] as i32) as f32 * SINC_FILTER[3 * j + 1];
            }
            buf[start + 3 * i + 2] = sum;
        }
        m.plc_pcm.copy_within(n / 3..m.plc_fill, 0);
        m.plc_fill -= n / 3;
        for i in 0..n {
            let tmp = buf[start + i];
            buf[start + i] -= PREEMPHASIS * m.preemphasis_mem;
            m.preemphasis_mem = tmp;
        }
        let mut overlap_mem = m.preemphasis_mem;
        for i in 0..overlap {
            let tmp = buf[DECODE_BUFFER_SIZE + i];
            buf[DECODE_BUFFER_SIZE + i] -= PREEMPHASIS * overlap_mem;
            overlap_mem = tmp;
        }
        // Deep PLC is mono.
        if cc == 2 {
            st.decode_mem.copy_within(..mem_len, mem_len);
        }
        if !last_neural {
            for c in 0..cc {
                for i in 0..overlap {
                    let d = &mut st.decode_mem[c * mem_len + start + i];
                    *d = (1.0 - window[i]) * buf_copy[c * overlap + i] + window[i] * *d;
                }
            }
        }
    }
}
