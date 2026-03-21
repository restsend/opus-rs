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
        if input.len() < n + overlap {
            panic!("MDCT forward: input too short: input.len()={} but need n={} + overlap={} = {}", input.len(), n, overlap, n + overlap);
        }
        if window.len() < overlap {
            panic!("MDCT forward: window too short: window.len()={} but need overlap={}", window.len(), overlap);
        }
        if output.len() < n2 * stride {
            panic!("MDCT forward: output too short: output.len()={} but need n2*stride={}", output.len(), n2 * stride);
        }

        if self.get_trig(shift).len() < n2 {
            panic!("MDCT forward: trig table too short for shift={}", shift);
        }
        let fft = &self.ffts[shift];
        let trig = self.get_trig(shift);

        let mut f = vec![0.0f32; n2];
        let overlap2 = overlap / 2;

        {
            let limit = overlap.div_ceil(4);

            let mid = n4.saturating_sub(limit);

            let mut yp  = 0usize;
            let mut xp1 = overlap2;
            let mut xp2 = n2 - 1 + overlap2;
            let mut wp1 = overlap2;
            let mut wp2 = overlap2.wrapping_sub(1);

            let loop1_iters = limit.min(n4);
            for _ in 0..loop1_iters {

                let w1 = if wp1 < window.len() { window[wp1] } else { 0.0 };
                let w2 = if wp2 < window.len() { window[wp2] } else { 0.0 };

                let in1 = if xp1 + n2 < input.len() { input[xp1 + n2] } else { 0.0 };
                let in2 = if xp2 < input.len() { input[xp2] } else { 0.0 };
                f[yp] = in1 * w2 + in2 * w1;
                yp += 1;

                let in3 = if xp1 < input.len() { input[xp1] } else { 0.0 };
                let in4 = if xp2 >= n2 && xp2 - n2 < input.len() { input[xp2 - n2] } else { 0.0 };
                f[yp] = in3 * w1 - in4 * w2;
                yp += 1;

                xp1 += 2;
                xp2 = xp2.saturating_sub(2);
                wp1 += 2;
                wp2 = wp2.saturating_sub(2);
            }

            for _ in limit..mid {

                let in1 = if xp2 < input.len() { input[xp2] } else { 0.0 };
                f[yp] = in1;
                yp += 1;

                let in2 = if xp1 < input.len() { input[xp1] } else { 0.0 };
                f[yp] = in2;
                yp += 1;
                xp1 += 2;
                xp2 = xp2.saturating_sub(2);
            }

            let loop3_iters = if mid > limit { n4 - mid } else { 0 };
            let mut wp1_l3 = 0usize;
            let mut wp2_l3 = overlap.saturating_sub(1);
            for _ in 0..loop3_iters {

                let w1 = if wp1_l3 < window.len() { window[wp1_l3] } else { 0.0 };
                let w2 = if wp2_l3 < window.len() { window[wp2_l3] } else { 0.0 };

                let in1 = if xp1 >= n2 && xp1 - n2 < input.len() { input[xp1 - n2] } else { 0.0 };
                let in2 = if xp2 < input.len() { input[xp2] } else { 0.0 };
                f[yp] = -in1 * w1 + in2 * w2;
                yp += 1;

                let in3 = if xp1 < input.len() { input[xp1] } else { 0.0 };
                let in4 = if xp2 + n2 < input.len() { input[xp2 + n2] } else { 0.0 };
                f[yp] = in3 * w2 + in4 * w1;
                yp += 1;

                xp1 += 2;
                xp2 = xp2.saturating_sub(2);
                wp1_l3 += 2;
                wp2_l3 = wp2_l3.saturating_sub(2);
            }
        }

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

        fft.process(&mut f2);

        let n4_scale = 1.0 / (n4 as f32);
        for i in 0..n4 {
            let fp = &f2[i];
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

        for i in 0..n4 {
            let x1 = input[2 * i * stride];
            let x2 = input[stride * (n2 - 1 - 2 * i)];
            let t0 = trig[i];
            let t1 = trig[n4 + i];
            let yr = x2 * t0 + x1 * t1;
            let yi = x1 * t0 - x2 * t1;
            f2[i] = Complex::new(yi, yr);
        }

        fft.process(&mut f2);

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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_window(overlap: usize) -> Vec<f32> {
        (0..overlap)
            .map(|i| {
                let x = std::f32::consts::PI * (i as f32 + 0.5) / overlap as f32;
                x.sin()
            })
            .collect()
    }

    #[allow(dead_code)]
    fn check_perfect_reconstruction(shift: usize) {
        let n_base = 1920usize;
        let max_lm = 4usize;

        let overlap_base = 120usize;
        let overlap = overlap_base >> shift;
        let mdct = MdctLookup::new(n_base, max_lm);
        let window = make_window(overlap);

        let n = n_base >> shift;
        let n2 = n / 2;
        let overlap2 = overlap / 2;

        let signal: Vec<f32> = (0..(2 * n + overlap))
            .map(|i| {
                let t = i as f32 / n as f32;
                (2.0 * std::f32::consts::PI * 3.7 * t).sin() * 0.5
                    + (2.0 * std::f32::consts::PI * 7.3 * t).cos() * 0.3
            })
            .collect();

        let mut spec0 = vec![0.0f32; n2];
        mdct.forward(&signal[0..n + overlap], &mut spec0, &window, overlap, shift, 1);

        let frame1_start = n - overlap2;
        let mut spec1 = vec![0.0f32; n2];
        mdct.forward(&signal[frame1_start..frame1_start + n + overlap], &mut spec1, &window, overlap, shift, 1);

        let mut out0 = vec![0.0f32; n + overlap];
        mdct.backward(&spec0, &mut out0, &window, overlap, shift, 1);

        let mut out1 = vec![0.0f32; n + overlap];
        out1[..overlap].copy_from_slice(&out0[n..n + overlap]);
        mdct.backward(&spec1, &mut out1, &window, overlap, shift, 1);

        let mut max_err: f32 = 0.0;
        for i in 0..n2 {
            let expected = signal[n + i];
            let got = out1[overlap + i];
            let err = (got - expected).abs();
            if err > max_err { max_err = err; }
        }

        assert!(
            max_err < 1e-3,
            "shift={shift}: forward→backward max error = {max_err:.2e} (expected < 1e-3)"
        );
    }

}
