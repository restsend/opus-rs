use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::Arc;

pub struct MdctLookup {
    pub n: usize,
    #[allow(dead_code)]
    max_lm: usize,
    ffts: Vec<Arc<dyn Fft<f32>>>,
    trig: Vec<f32>,
}

impl MdctLookup {
    pub fn new(n: usize, max_lm: usize) -> Self {
        let mut ffts = Vec::new();
        let mut planner = FftPlanner::new();
        let mut trig = Vec::new();
        let mut curr_n = n;

        for _ in 0..=max_lm {
            let n2 = curr_n / 2;
            let n4 = curr_n / 4;
            ffts.push(planner.plan_fft_forward(n4));
            // Opus CELT trig table:
            // trig[i] = cos(2*PI*(i+0.125)/N) for i in 0..N/2
            for i in 0..n2 {
                let angle = 2.0 * std::f64::consts::PI * (i as f64 + 0.125) / curr_n as f64;
                trig.push(angle.cos() as f32);
            }
            curr_n >>= 1;
        }

        Self {
            n,
            max_lm,
            ffts,
            trig,
        }
    }

    fn get_trig(&self, shift: usize) -> &[f32] {
        let mut offset = 0;
        let mut curr_n = self.n;
        for _ in 0..shift {
            offset += curr_n / 2;
            curr_n >>= 1;
        }
        &self.trig[offset..]
    }

    pub fn get_trig_debug(&self, shift: usize) -> &[f32] {
        self.get_trig(shift)
    }

    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let n = self.n >> shift;
        let n2 = n / 2;
        let n4 = n / 4;
        let fft = &self.ffts[shift];
        let trig = self.get_trig(shift);

        let mut f = vec![0.0f32; n2];
        let overlap2 = overlap / 2;

        // 1. Fold/Window (matching clt_mdct_forward_c)
        {
            let mut yp = 0;
            let mut xp1 = overlap2;
            let mut xp2 = n2 - 1 + overlap2;
            let mut wp1 = overlap2;
            let mut wp2 = overlap2 - 1;

            let limit = (overlap + 3) / 4;
            for _ in 0..limit {
                // *yp++ = S_MUL(xp1[N2], *wp2) + S_MUL(*xp2, *wp1);
                f[yp] = input[xp1 + n2] * window[wp2] + input[xp2] * window[wp1];
                yp += 1;
                // *yp++ = S_MUL(*xp1, *wp1)    - S_MUL(xp2[-N2], *wp2);
                f[yp] = input[xp1] * window[wp1] - input[xp2 - n2] * window[wp2];
                yp += 1;
                xp1 += 2;
                xp2 = xp2.wrapping_sub(2);
                wp1 += 2;
                wp2 = wp2.wrapping_sub(2);
            }

            let mut wp1_loop2 = 0;
            let mut wp2_loop2 = overlap - 1;

            for _ in limit..(n4 - limit) {
                // *yp++ = *xp2;
                f[yp] = input[xp2];
                yp += 1;
                // *yp++ = *xp1;
                f[yp] = input[xp1];
                yp += 1;
                xp1 += 2;
                xp2 = xp2.wrapping_sub(2);
            }

            for _ in (n4 - limit)..n4 {
                // *yp++ =  -S_MUL(xp1[-N2], *wp1) + S_MUL(*xp2, *wp2);
                f[yp] = -input[xp1 - n2] * window[wp1_loop2] + input[xp2] * window[wp2_loop2];
                yp += 1;
                // *yp++ = S_MUL(*xp1, *wp2)     + S_MUL(xp2[N2], *wp1);
                f[yp] = input[xp1] * window[wp2_loop2] + input[xp2 + n2] * window[wp1_loop2];
                yp += 1;
                xp1 += 2;
                xp2 = xp2.wrapping_sub(2);
                wp1_loop2 += 2;
                wp2_loop2 = wp2_loop2.wrapping_sub(2);
            }
        }

        // 2. Pre-rotation
        let mut f2 = vec![Complex::new(0.0, 0.0); n4];
        for i in 0..n4 {
            let re = f[2 * i];
            let im = f[2 * i + 1];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = re * t0 - im * t1;
            let yi = im * t0 + re * t1;
            f2[i] = Complex::new(yr, yi);
        }

        // 3. FFT
        fft.process(&mut f2);

        // 4. Post-rotation & Scaling
        let n4_scale = 1.0 / (n4 as f32);
        for i in 0..n4 {
            let fp = &f2[i]; // Use sequential access, fft.process already handled bitrev
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = (fp.im * t1 - fp.re * t0) * n4_scale;
            let yi = (fp.re * t1 + fp.im * t0) * n4_scale;
            output[i * 2 * stride] = yr;
            output[stride * (n2 - 1 - 2 * i)] = yi;
        }
    }

    pub fn backward(
        &self,
        input: &[f32],
        output: &mut [f32],
        window: &[f32],
        overlap: usize,
        shift: usize,
        stride: usize,
    ) {
        let n = self.n >> shift;
        let n2 = n / 2;
        let n4 = n / 4;
        let overlap2 = overlap / 2;
        let fft = &self.ffts[shift];
        let trig = self.get_trig(shift);

        let mut f2 = vec![Complex::new(0.0, 0.0); n4];

        // 1. Pre-rotation
        for i in 0..n4 {
            let x1 = input[2 * i * stride];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            f2[i] = Complex::new(yi, yr);
        }

        // 2. FFT
        fft.process(&mut f2);

        // 3. Post-rotation
        for i in 0..(n4 + 1) >> 1 {
            let re0 = f2[i].im;
            let im0 = f2[i].re;
            let t0_0 = trig[i];
            let t1_0 = trig[n4 + i];

            let yr0 = re0 * t0_0 + im0 * t1_0;
            let yi0 = re0 * t1_0 - im0 * t0_0;

            let re1 = f2[n4 - 1 - i].im;
            let im1 = f2[n4 - 1 - i].re;
            let t0_1 = trig[n4 - i - 1];
            let t1_1 = trig[n2 - i - 1];

            let yr1 = re1 * t0_1 + im1 * t1_1;
            let yi1 = re1 * t1_1 - im1 * t0_1;

            output[overlap2 + 2 * i] = yr0;
            output[overlap2 + n2 - 1 - 2 * i] = yi0;
            output[overlap2 + n2 - 2 - 2 * i] = yr1;
            output[overlap2 + 2 * i + 1] = yi1;
        }

        // 4. TDAC overlap-add with mirroring
        // output[0..overlap2] is the old tail
        // output[overlap2..overlap] is the start of the current frame (just written)
        // C code: xp1 = out+overlap-1, yp1 = out
        for i in 0..overlap2 {
            let x1 = output[overlap - 1 - i]; // New frame sample
            let x2 = output[i]; // Old tail sample
            let wp1 = window[i];
            let wp2 = window[overlap - 1 - i];

            // Re-check signs based on C implementation:
            // *yp1++ = SUB32_ovflw(S_MUL(x2, *wp2), S_MUL(x1, *wp1));
            // *xp1-- = ADD32_ovflw(S_MUL(x2, *wp1), S_MUL(x1, *wp2));
            // yp1 is out[i], xp1 is out[overlap-1-i]
            // x1 is *xp1 (end), x2 is *yp1 (start)

            output[i] = x2 * wp2 - x1 * wp1;
            output[overlap - 1 - i] = x2 * wp1 + x1 * wp2;
        }
    }
}
