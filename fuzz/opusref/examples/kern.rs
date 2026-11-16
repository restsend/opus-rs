//! Scratch kernel comparison.
use opus_rs::kiss_fft::{KissCpx, opus_fft_impl};
use opusref::*;

fn main() {
    let mode = opus_rs::modes::default_mode();
    let mut seed = 7u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * 2000.0
    };
    for shift in 0..4 {
        let n4 = 480 >> shift;
        let x: Vec<f32> = (0..2 * n4).map(|_| rnd()).collect();
        let mut a: Vec<KissCpx> = (0..n4).map(|i| KissCpx::new(x[2 * i], x[2 * i + 1])).collect();
        opus_fft_impl(mode.mdct.fft(shift), &mut a);
        let mut b = x.clone();
        unsafe { ref_fft_impl(b.as_mut_ptr(), shift as i32) };
        let d = (0..n4).filter(|&i| a[i].r != b[2 * i] || a[i].i != b[2 * i + 1]).count();
        println!("fft shift {shift}: {d}/{n4} differ");
        let n = 1920 >> shift;
        let mut inp: Vec<f32> = (0..n / 2 + 120).map(|_| rnd()).collect();
        let mut o1 = vec![0f32; n / 2];
        let mut o2 = vec![0f32; n / 2];
        mode.mdct.forward(&inp, &mut o1, mode.window, 120, shift, 1);
        unsafe { ref_mdct_forward(inp.as_mut_ptr(), o2.as_mut_ptr(), shift as i32, 1) };
        let d = (0..n / 2).filter(|&i| o1[i] != o2[i]).count();
        println!("mdct fwd shift {shift}: {d}/{} differ", n / 2);
        let mut fin: Vec<f32> = (0..n / 2).map(|_| rnd()).collect();
        let mut b1: Vec<f32> = (0..n / 2 + 120).map(|_| rnd()).collect();
        let mut b2 = b1.clone();
        mode.mdct.backward(&fin, &mut b1, mode.window, 120, shift, 1);
        unsafe { ref_mdct_backward(fin.as_mut_ptr(), b2.as_mut_ptr(), shift as i32, 1) };
        let d = (0..b1.len()).filter(|&i| b1[i] != b2[i]).count();
        println!("mdct bwd shift {shift}: {d}/{} differ", b1.len());
    }
}
