use crate::kiss_fft::{KissCpx, KissFftState, opus_fft_impl};
use std::f32::consts::PI;

const MAX_N2: usize = 960;
const MAX_N4: usize = 480;

pub struct MdctLookup {
    pub n: usize,
    pub max_lm: usize,
    kfft: Vec<Option<KissFftState>>,
    trig: Vec<f32>,
}

impl MdctLookup {
    pub fn new(n: usize, max_lm: usize) -> Self {
        let mut kfft = Vec::new();
        let mut trig = Vec::new();
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
        let st = self.kfft[shift].as_ref().expect("FFT state not initialized");
        let n = self.n >> shift;
        let n2 = n / 2;
        let n4 = n / 4;
        let scale = st.scale();

        let (trig, _) = self.get_trig(shift);
        let overlap2 = overlap / 2;

        let mut f_buf = [0.0f32; MAX_N2];
        let mut f2_buf = [KissCpx::new(0.0, 0.0); MAX_N4];
        let f = &mut f_buf[..n2];
        let f2 = &mut f2_buf[..n4];

        // Assert caller invariants so LLVM can prove all loop accesses in-bounds
        // and eliminate per-element conditional checks, enabling auto-vectorization.
        assert!(input.len() >= n + overlap);
        assert!(window.len() >= overlap);

        {
            let mut yp = 0usize;
            let mut xp1 = overlap2;
            let mut xp2 = n2 - 1 + overlap2;
            let mut wp1 = overlap2;
            // wp2 can underflow on the final post-loop decrement (value never read after),
            // so saturating_sub is used only here; all other pointers stay non-negative.
            let mut wp2 = overlap2.saturating_sub(1);

            let limit = (overlap + 3) / 4;
            let mid = if n4 > limit { n4 - limit } else { 0 };

            // Loop 1: windowed fold (first overlap region).
            // All indices proved valid when input.len()>=n+overlap, window.len()>=overlap.
            let loop1_iters = limit.min(n4);
            for _ in 0..loop1_iters {
                let w1 = window[wp1];
                let w2 = window[wp2];

                f[yp] = input[xp1 + n2] * w2 + input[xp2] * w1;
                yp += 1;

                f[yp] = input[xp1] * w1 - input[xp2 - n2] * w2;
                yp += 1;

                xp1 += 2;
                xp2 -= 2;
                wp1 += 2;
                wp2 = wp2.saturating_sub(2);
            }

            // Loop 2: no window (middle region, straight interleaved copy).
            for _ in limit..mid {
                f[yp] = input[xp2];
                yp += 1;

                f[yp] = input[xp1];
                yp += 1;
                xp1 += 2;
                xp2 -= 2;
            }

            // Loop 3: windowed fold (second overlap region).
            // At loop3 start, xp1 = n2 exactly (identity: overlap2 + 2*mid = n2).
            let loop3_iters = if mid > limit { n4 - mid } else { 0 };
            let mut wp1_l3 = 0usize;
            let mut wp2_l3 = overlap.saturating_sub(1);
            for _ in 0..loop3_iters {
                let w1 = window[wp1_l3];
                let w2 = window[wp2_l3];

                f[yp] = -input[xp1 - n2] * w1 + input[xp2] * w2;
                yp += 1;

                f[yp] = input[xp1] * w2 + input[xp2 + n2] * w1;
                yp += 1;

                xp1 += 2;
                xp2 -= 2;
                wp1_l3 += 2;
                wp2_l3 -= 2;
            }
        }

        // Pre-rotation with bitrev indexing
        for i in 0..n4 {
            let re = f[2 * i];
            let im = f[2 * i + 1];
            let t0 = trig[i];
            let t1 = trig[n4 + i];

            let yr = re * t0 - im * t1;
            let yi = im * t0 + re * t1;

            f2[st.bitrev[i] as usize] = KissCpx::new(yr * scale, yi * scale);
        }

        opus_fft_impl(st, f2);

        // Post-rotation
        for i in 0..n4 {
            let fp = &f2[i];
            let t0 = trig[i];
            let t1 = trig[n4 + i];

            let yr = fp.i * t1 - fp.r * t0;
            let yi = fp.r * t1 + fp.i * t0;

            output[i * 2 * stride] = yr;
            output[stride * (n2 - 1 - 2 * i)] = yi;
        }
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
        let st = self.kfft[shift].as_ref().expect("FFT state not initialized");
        let n = self.n >> shift;
        let n2 = n / 2;
        let n4 = n / 4;
        let overlap2 = overlap / 2;

        let (trig, _) = self.get_trig(shift);

        let mut f2_buf = [KissCpx::new(0.0, 0.0); MAX_N4];
        let f2 = &mut f2_buf[..n4];

        for i in 0..n4 {
            let rev = st.bitrev[i] as usize;
            let x1 = input[2 * i * stride];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            let t0 = trig[i];
            let t1 = trig[n4 + i];

            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;

            f2[rev] = KissCpx::new(yi, yr);
        }

        opus_fft_impl(st, f2);

        for i in 0..n4 {
            output[overlap2 + 2 * i] = f2[i].r;
            output[overlap2 + 2 * i + 1] = f2[i].i;
        }

        for i in 0..((n4 + 1) >> 1) {
            let re0 = output[overlap2 + 2 * i + 1];
            let im0 = output[overlap2 + 2 * i];
            let t0_0 = trig[i];
            let t1_0 = trig[n4 + i];

            let yr0 = re0 * t0_0 + im0 * t1_0;
            let yi0 = re0 * t1_0 - im0 * t0_0;

            let re1 = output[overlap2 + n2 - 1 - 2 * i];
            let im1 = output[overlap2 + n2 - 2 - 2 * i];
            let t0_1 = trig[n4 - i - 1];
            let t1_1 = trig[n2 - i - 1];

            let yr1 = re1 * t0_1 + im1 * t1_1;
            let yi1 = re1 * t1_1 - im1 * t0_1;

            output[overlap2 + 2 * i] = yr0;
            output[overlap2 + n2 - 1 - 2 * i] = yi0;
            output[overlap2 + n2 - 2 - 2 * i] = yr1;
            output[overlap2 + 2 * i + 1] = yi1;
        }

        // TDAC - mirror on both sides
        for i in 0..overlap2 {
            let x1 = output[overlap - 1 - i];
            let x2 = output[i];
            let wp1 = window[i];
            let wp2 = window[overlap - 1 - i];

            output[i] = x2 * wp2 - x1 * wp1;
            output[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
        }
    }
}
