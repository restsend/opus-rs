use crate::kiss_fft::{KissCpx, KissFftState, opus_fft_impl};
use core::f32::consts::PI;

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::compat::with_scratch;
use crate::fixedvec::FixedVec;

const MAX_N2: usize = 960;
/// Maximum number of MDCT levels (`max_lm`). The default 48 kHz / 960 mode uses 3;
/// 4 leaves headroom for hybrid modes.
const MDCT_MAX_LM: usize = 4;
/// Bounded by the geometric sum of `n2` across all levels: `n/2 + n/4 + ... <= n`.
pub(crate) const MDCT_TRIG_MAX: usize = 2 * MAX_N2;

pub struct MdctLookup {
    pub n: usize,
    pub max_lm: usize,
    kfft: FixedVec<Option<KissFftState>, { MDCT_MAX_LM + 1 }>,
    trig: FixedVec<f32, MDCT_TRIG_MAX>,
}

impl MdctLookup {
    pub fn new(n: usize, max_lm: usize) -> Self {
        debug_assert!(max_lm <= MDCT_MAX_LM);
        let mut kfft: FixedVec<Option<KissFftState>, { MDCT_MAX_LM + 1 }> = FixedVec::new();
        let mut trig: FixedVec<f32, MDCT_TRIG_MAX> = FixedVec::new();
        let mut curr_n = n;

        for shift in 0..=max_lm {
            let n4 = curr_n / 4;

            if shift == 0 {
                kfft.push(KissFftState::new(n4));
            } else if let Some(base) = kfft.first().unwrap().as_ref() {
                kfft.push(KissFftState::new_sub(base, n4));
            } else {
                kfft.push(None);
            }

            let n2 = curr_n / 2;
            for i in 0..n2 {
                let angle = 2.0 * PI * (i as f32 + 0.125) / curr_n as f32;
                trig.push(angle.cos());
            }

            curr_n >>= 1;
        }

        Self {
            n,
            max_lm,
            kfft,
            trig,
        }
    }

    /// The MDCT of the static 48 kHz / 960-sample mode: `new(1920, 3)` from
    /// precomputed tables.
    pub(crate) const STATIC_1920: Self = {
        use crate::static_modes::*;
        let tw = FFT_TWIDDLES_480;
        Self {
            n: 1920,
            max_lm: 3,
            kfft: FixedVec::from_array(
                [
                    Some(KissFftState::from_tables(
                        480,
                        FFT_SHIFT_480,
                        FFT_FACTORS_480,
                        FFT_BITREV_480,
                        tw,
                    )),
                    Some(KissFftState::from_tables(
                        240,
                        FFT_SHIFT_240,
                        FFT_FACTORS_240,
                        FFT_BITREV_240,
                        tw,
                    )),
                    Some(KissFftState::from_tables(
                        120,
                        FFT_SHIFT_120,
                        FFT_FACTORS_120,
                        FFT_BITREV_120,
                        tw,
                    )),
                    Some(KissFftState::from_tables(
                        60,
                        FFT_SHIFT_60,
                        FFT_FACTORS_60,
                        FFT_BITREV_60,
                        tw,
                    )),
                    None,
                ],
                4,
            ),
            trig: FixedVec::from_array(MDCT_TRIG_1920, 1800),
        }
    };

    /// The FFT of the MDCT at `shift` (libopus `mdct.kfft[shift]`), which the
    /// encoder's signal analysis reuses.
    pub fn fft(&self, shift: usize) -> &KissFftState {
        self.kfft[shift].as_ref().expect("MDCT FFT plan")
    }

    fn get_trig(&self, shift: usize) -> (&[f32], usize) {
        let mut offset = 0;
        let mut curr_n = self.n;
        for _ in 0..shift {
            offset += curr_n / 2;
            curr_n >>= 1;
        }
        (&self.trig[offset..offset + curr_n / 2], curr_n / 4)
    }

    pub fn get_trig_debug(&self, shift: usize) -> &[f32] {
        let (trig, _) = self.get_trig(shift);
        trig
    }

    #[inline]
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let st = self.kfft[shift]
            .as_ref()
            .expect("FFT state not initialized");
        let n = self.n >> shift;
        let n2 = n / 2;
        let n4 = n / 4;
        let scale = st.scale();

        let (trig, _) = self.get_trig(shift);
        let overlap2 = overlap / 2;

        // The pre-rotation windowed reads span n2 + overlap samples (libopus
        // reads input[xp1] with xp1 up to n2-1+overlap2). The old bound
        // n2+overlap2 was too lax (issue #27 deep scan).
        assert!(
            input.len() >= n2 + overlap,
            "MDCT forward: input too small (need {}, have {})",
            n2 + overlap,
            input.len()
        );
        assert!(
            window.len() >= overlap,
            "MDCT forward: window too small (need {}, have {})",
            overlap,
            window.len()
        );
        assert!(
            output.len() >= n2,
            "MDCT forward: output buffer too small (need {}, have {})",
            n2,
            output.len()
        );

        with_scratch(n4, |f2: &mut [KissCpx]| {
            let (t0, t1) = trig.split_at(n4);
            let bitrev = &st.bitrev[..n4];
            // Window, shuffle and pre-rotate: the i-th folded pair (re, im)
            // goes straight to its bit-reversed FFT input.
            let mut rotate = |i: usize, re: f32, im: f32| {
                let yr = re * t0[i] - im * t1[i];
                let yi = im * t0[i] + re * t1[i];
                f2[bitrev[i] as usize] = KissCpx::new(yr * scale, yi * scale);
            };

            let mut xp1 = overlap2;
            let mut xp2 = n2 - 1 + overlap2;
            let mut wp1 = overlap2;
            let mut wp2 = overlap2.saturating_sub(1);

            let limit = overlap.div_ceil(4);
            let mid = n4.saturating_sub(limit);

            let loop1_iters = limit.min(n4);
            for i in 0..loop1_iters {
                let w1 = window[wp1];
                let w2 = window[wp2];
                rotate(
                    i,
                    input[xp1 + n2] * w2 + input[xp2] * w1,
                    input[xp1] * w1 - input[xp2 - n2] * w2,
                );
                xp1 += 2;
                xp2 -= 2;
                wp1 += 2;
                wp2 = wp2.saturating_sub(2);
            }

            for i in limit..mid {
                rotate(i, input[xp2], input[xp1]);
                xp1 += 2;
                xp2 -= 2;
            }

            // Loop 3 runs from `mid` to `n4` (C: for(;i<N4;i++)). When the
            // frame is short enough that loop 2 is empty (mid == limit, i.e.
            // n4 == 2*limit, e.g. 2.5 ms sub-frames at 48 kHz with shift=3),
            // this still needs n4 - mid pairs.
            let start = loop1_iters + mid.saturating_sub(limit);
            let mut wp1 = 0usize;
            let mut wp2 = overlap.saturating_sub(1);
            for i in start..start + n4.saturating_sub(mid) {
                let w1 = window[wp1];
                let w2 = window[wp2];
                rotate(
                    i,
                    -input[xp1 - n2] * w1 + input[xp2] * w2,
                    input[xp1] * w2 + input[xp2 + n2] * w1,
                );
                xp1 += 2;
                xp2 -= 2;
                wp1 += 2;
                wp2 = wp2.wrapping_sub(2);
            }

            opus_fft_impl(st, f2);

            for (i, fp) in f2.iter().enumerate() {
                let yr = fp.i * t1[i] - fp.r * t0[i];
                let yi = fp.r * t1[i] + fp.i * t0[i];
                output[i * 2 * stride] = yr;
                output[stride * (n2 - 1 - 2 * i)] = yi;
            }
        });
    }

    #[inline]
    pub fn backward(
        &self,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let st = self.kfft[shift]
            .as_ref()
            .expect("FFT state not initialized");
        let n = self.n >> shift;
        let n2 = n / 2;
        let n4 = n / 4;
        let overlap2 = overlap / 2;

        // Validate all buffer contracts up front: the pre-rotation reads
        // input[2*i*stride] and input[stride*(n2-1-2*i)], and the TDAC stage
        // reads window[..overlap]; failing after the FFT has run would waste
        // the whole transform (issue #27 deep scan).
        assert!(
            input.len() > stride * (n2 - 1),
            "MDCT backward: input too small (need {}, have {})",
            stride * (n2 - 1) + 1,
            input.len()
        );
        assert!(
            window.len() >= overlap,
            "MDCT backward: window too small (need {}, have {})",
            overlap,
            window.len()
        );
        assert!(
            output.len() >= overlap2 + n2,
            "MDCT backward: output too small (need {}, have {})",
            overlap2 + n2,
            output.len()
        );

        let (trig, _) = self.get_trig(shift);
        let (t0, t1) = trig.split_at(n4);

        with_scratch(n4, |f2: &mut [KissCpx]| {
            for (i, &rev) in st.bitrev[..n4].iter().enumerate() {
                let x1 = input[2 * i * stride];
                let x2 = input[stride * (n2 - 1 - 2 * i)];
                let yr = x2 * t0[i] + x1 * t1[i];
                let yi = x1 * t0[i] - x2 * t1[i];
                f2[rev as usize] = KissCpx::new(yi, yr);
            }

            opus_fft_impl(st, f2);

            for i in 0..((n4 + 1) >> 1) {
                let im0 = f2[i].r;
                let re0 = f2[i].i;
                let yr0 = re0 * t0[i] + im0 * t1[i];
                let yi0 = re0 * t1[i] - im0 * t0[i];

                let j = n4 - 1 - i;
                let im1 = f2[j].r;
                let re1 = f2[j].i;
                let yr1 = re1 * t0[j] + im1 * t1[j];
                let yi1 = re1 * t1[j] - im1 * t0[j];

                output[overlap2 + 2 * i] = yr0;
                output[overlap2 + n2 - 1 - 2 * i] = yi0;
                output[overlap2 + n2 - 2 - 2 * i] = yr1;
                output[overlap2 + 2 * i + 1] = yi1;
            }
        });

        // Mirror on both sides for TDAC.
        let (head, tail) = output[..overlap].split_at_mut(overlap2);
        let (w_head, w_tail) = window[..overlap].split_at(overlap2);
        for (((x2, x1), &wp1), &wp2) in head
            .iter_mut()
            .zip(tail.iter_mut().rev())
            .zip(w_head)
            .zip(w_tail.iter().rev())
        {
            let (a, b) = (*x2, *x1);
            *x2 = a * wp2 - b * wp1;
            *x1 = a * wp1 + b * wp2;
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod mdct_tests {
    use super::MdctLookup;

    #[test]
    // Miri perturbs the results of cos, as libm implementations may differ.
    #[cfg_attr(miri, ignore)]
    fn static_mode_matches_computed() {
        let computed = MdctLookup::new(1920, 3);
        let table = &crate::modes::default_mode().mdct;
        assert_eq!((table.n, table.max_lm), (computed.n, computed.max_lm));
        let bits = |t: &[f32]| t.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&table.trig), bits(&computed.trig));
        assert_eq!(table.kfft.len(), computed.kfft.len());
        for (a, b) in table.kfft.iter().zip(computed.kfft.iter()) {
            assert_eq!(
                a.as_ref().unwrap().fingerprint(),
                b.as_ref().unwrap().fingerprint()
            );
        }
    }

    #[test]
    fn test_mdct_backward_transient_no_blowup() {
        let mode = crate::modes::default_mode();
        let shift = 3;
        let n = mode.mdct.n >> shift; // 120
        let overlap = mode.overlap; // 120
        let stride = 8;

        let frame_size = 960usize;
        let mut freq = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            freq[i] = ((i as f32) * 0.01).sin() * 10.0;
        }

        let out_len = n + overlap; // 240
        let mut output0 = vec![0.0f32; out_len];
        let mut output1 = vec![0.0f32; out_len];

        mode.mdct.backward(
            &freq[0..],
            &mut output0,
            mode.window,
            overlap,
            shift,
            stride,
        );
        mode.mdct.backward(
            &freq[1..],
            &mut output1,
            mode.window,
            overlap,
            shift,
            stride,
        );

        let max0 = output0.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max1 = output1.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        assert!(max0.abs() < 500.0, "sub0 blowup: {}", max0);
        assert!(max1.abs() < 500.0, "sub1 blowup: {}", max1);
    }

    #[test]
    fn test_mdct_backward_stride1_neon_matches_scalar() {
        let mode = crate::modes::default_mode();
        let shift = 0; // non-transient full-size MDCT
        let n = mode.mdct.n >> shift; // 1920
        let n2 = n / 2; // 960
        let n4 = n / 4; // 480
        let overlap = mode.overlap; // 120
        let overlap2 = overlap / 2; // 60
        let stride = 1;

        let freq_len = n2;
        let mut freq = vec![0.0f32; freq_len + 4];
        for i in 0..freq_len {
            freq[i] = ((i as f32) * 0.01).sin() * 4577.0;
        }

        let out_len = overlap2 + n2; // 60 + 960 = 1020
        let mut output_hw = vec![0.0f32; out_len + 100];
        mode.mdct.backward(
            &freq[..],
            &mut output_hw,
            mode.window,
            overlap,
            shift,
            stride,
        );

        let st = mode.mdct.kfft[shift].as_ref().unwrap();
        let (trig, _) = mode.mdct.get_trig(shift);

        use crate::kiss_fft::KissCpx;
        let mut f2 = vec![KissCpx::new(0.0, 0.0); n4];
        for i in 0..n4 {
            let rev = st.bitrev[i] as usize;
            let x1 = freq[2 * i * stride];
            let x2 = freq[stride * (n2 - 1 - 2 * i)];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            f2[rev] = KissCpx::new(yi, yr);
        }
        crate::kiss_fft::opus_fft_impl(st, &mut f2);

        let mut output_scalar = vec![0.0f32; out_len + 100];
        for i in 0..((n4 + 1) >> 1) {
            let im0 = f2[i].r;
            let re0 = f2[i].i;
            let t0_0 = trig[i];
            let t1_0 = trig[n4 + i];
            let yr0 = re0 * t0_0 + im0 * t1_0;
            let yi0 = re0 * t1_0 - im0 * t0_0;
            let j = n4 - 1 - i;
            let im1 = f2[j].r;
            let re1 = f2[j].i;
            let t0_1 = trig[j];
            let t1_1 = trig[n4 + j];
            let yr1 = re1 * t0_1 + im1 * t1_1;
            let yi1 = re1 * t1_1 - im1 * t0_1;
            output_scalar[overlap2 + 2 * i] = yr0;
            output_scalar[overlap2 + n2 - 1 - 2 * i] = yi0;
            output_scalar[overlap2 + n2 - 2 - 2 * i] = yr1;
            output_scalar[overlap2 + 2 * i + 1] = yi1;
        }
        // TDAC
        for i in 0..overlap2 {
            let x1 = output_scalar[overlap - 1 - i];
            let x2 = output_scalar[i];
            let wp1 = mode.window[i];
            let wp2 = mode.window[overlap - 1 - i];
            output_scalar[i] = x2 * wp2 - x1 * wp1;
            output_scalar[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
        }

        let max_diff = output_hw[..out_len]
            .iter()
            .zip(output_scalar[..out_len].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.5,
            "stride=1 NEON vs scalar mismatch: max_diff={}",
            max_diff
        );
    }

    #[test]
    fn test_mdct_backward_neon_matches_scalar() {
        let mode = crate::modes::default_mode();
        let shift = 3;
        let n = mode.mdct.n >> shift; // 240
        let n2 = n / 2; // 120
        let n4 = n / 4; // 60
        let overlap = mode.overlap; // 120
        let overlap2 = overlap / 2; // 60
        let stride = 8;

        // Build a realistic freq vector (sine wave @ 440Hz)
        let frame_size = 960usize;
        let mut freq = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            freq[i] = ((i as f32) * 0.01).sin() * 200.0;
        }

        let out_len = n + overlap; // 360
        let mut output_hw = vec![0.0f32; out_len];
        mode.mdct.backward(
            &freq[0..],
            &mut output_hw,
            mode.window,
            overlap,
            shift,
            stride,
        );

        // Scalar reference
        let st = mode.mdct.kfft[shift].as_ref().unwrap();
        let (trig, _) = mode.mdct.get_trig(shift);

        use crate::kiss_fft::KissCpx;
        let mut f2 = vec![KissCpx::new(0.0, 0.0); n4];
        for i in 0..n4 {
            let rev = st.bitrev[i] as usize;
            let x1 = freq[2 * i * stride];
            let x2 = freq[stride * (n2 - 1 - 2 * i)];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            f2[rev] = KissCpx::new(yi, yr);
        }
        crate::kiss_fft::opus_fft_impl(st, &mut f2);

        let mut output_scalar = vec![0.0f32; out_len];
        for i in 0..((n4 + 1) >> 1) {
            let im0 = f2[i].r;
            let re0 = f2[i].i;
            let t0_0 = trig[i];
            let t1_0 = trig[n4 + i];
            let yr0 = re0 * t0_0 + im0 * t1_0;
            let yi0 = re0 * t1_0 - im0 * t0_0;
            let j = n4 - 1 - i;
            let im1 = f2[j].r;
            let re1 = f2[j].i;
            let t0_1 = trig[j];
            let t1_1 = trig[n4 + j];
            let yr1 = re1 * t0_1 + im1 * t1_1;
            let yi1 = re1 * t1_1 - im1 * t0_1;
            output_scalar[overlap2 + 2 * i] = yr0;
            output_scalar[overlap2 + n2 - 1 - 2 * i] = yi0;
            output_scalar[overlap2 + n2 - 2 - 2 * i] = yr1;
            output_scalar[overlap2 + 2 * i + 1] = yi1;
        }
        // TDAC
        for i in 0..overlap2 {
            let x1 = output_scalar[overlap - 1 - i];
            let x2 = output_scalar[i];
            let wp1 = mode.window[i];
            let wp2 = mode.window[overlap - 1 - i];
            output_scalar[i] = x2 * wp2 - x1 * wp1;
            output_scalar[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
        }

        for i in 0..out_len {
            let diff = (output_hw[i] - output_scalar[i]).abs();
            if diff > 1e-3 {
                eprintln!(
                    "Mismatch at output[{}]: hw={} scalar={} diff={}",
                    i, output_hw[i], output_scalar[i], diff
                );
            }
        }
        let max_diff = output_hw
            .iter()
            .zip(output_scalar.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.1,
            "NEON/HW vs scalar mismatch: max_diff={}",
            max_diff
        );
    }
}
