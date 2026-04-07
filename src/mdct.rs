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
        let st = self.kfft[shift]
            .as_ref()
            .expect("FFT state not initialized");
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
        // Max input index accessed = n/2 + overlap/2 - 1 (see loop analysis), so we need
        // at least n/2 + overlap/2 elements. n + overlap is the theoretical over-estimate
        // but the actual accesses stay within n/2 + overlap/2 due to the fold structure.
        assert!(input.len() >= n2 + overlap2);
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
        let st = self.kfft[shift]
            .as_ref()
            .expect("FFT state not initialized");
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

        // Pre-rotate: Write to temp buffer starting at overlap2 (like C's out+(overlap>>1))
        let mut temp = vec![0.0f32; n + overlap];
        for i in 0..n4 {
            temp[overlap2 + 2 * i] = f2[i].r;
            temp[overlap2 + 2 * i + 1] = f2[i].i;
        }

        // Post-rotate from both ends
        // C reads re=yp0[1], im=yp0[0] (swapped because using FFT instead of IFFT)
        for i in 0..((n4 + 1) >> 1) {
            let im0 = temp[overlap2 + 2 * i];
            let re0 = temp[overlap2 + 2 * i + 1];
            let t0_0 = trig[i];
            let t1_0 = trig[n4 + i];

            let yr0 = re0 * t0_0 + im0 * t1_0;
            let yi0 = re0 * t1_0 - im0 * t0_0;

            let im1 = temp[overlap2 + n2 - 2 - 2 * i];
            let re1 = temp[overlap2 + n2 - 1 - 2 * i];
            let t0_1 = trig[n4 - i - 1];
            let t1_1 = trig[n2 - i - 1];

            let yr1 = re1 * t0_1 + im1 * t1_1;
            let yi1 = re1 * t1_1 - im1 * t0_1;

            temp[overlap2 + 2 * i] = yr0;
            temp[overlap2 + n2 - 1 - 2 * i] = yi0;
            temp[overlap2 + n2 - 2 - 2 * i] = yr1;
            temp[overlap2 + 2 * i + 1] = yi1;
        }

        // TDAC: Copy to output with windowing
        // C code's TDAC reads from:
        //   yp1 = out[0..overlap/2) - previous frame's overlap data (preserved by caller)
        //   xp1 = out[overlap-1..overlap/2) - current frame's IMDCT output
        // The caller must preserve overlap samples between frames for TDAC to work correctly.

        // Copy post-rotated data to output[overlap/2..overlap/2+n2]
        // This is where the IMDCT output goes (matching C's post-rotation output location)
        for i in 0..n2 {
            output[overlap2 + i] = temp[overlap2 + i];
        }

        // Apply TDAC to overlap region
        // C code: xp1 = out+overlap-1, yp1 = out
        // x1 = *xp1 (reads from out[overlap-1] down to out[overlap/2])
        // x2 = *yp1 (reads from out[0] up to out[overlap/2-1])
        // The key insight: x2 comes from the START of the output buffer,
        // which contains the previous frame's overlap data (preserved by caller)
        for i in 0..overlap2 {
            // x1: current frame's IMDCT output at the end of overlap region
            let x1 = output[overlap - 1 - i];
            // x2: previous frame's overlap data at the start of buffer (or zeros for first frame)
            let x2 = output[i];
            let wp1 = window[i];
            let wp2 = window[overlap - 1 - i];

            output[i] = x2 * wp2 - x1 * wp1;
            output[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
        }
    }
}
