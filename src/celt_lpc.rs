use crate::fixedvec::FixedVec;
use crate::pitch::{pitch_xcorr, xcorr_kernel};

/// libopus `_celt_lpc` (celt_lpc.c, float build).
pub fn lpc(lpc: &mut [f32], ac: &[f32], p: usize) {
    // The coefficients start at zero, so those after an early stop stay
    // zero; a NaN ac[0] fails the test and leaves them all zero.
    lpc[..p].fill(0.0);
    let mut error = ac[0];
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(error > 1e-10) {
        return;
    }

    for i in 0..p {
        let mut rr = 0.0f32;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;

        lpc[i] = r;
        for j in 0..i.div_ceil(2) {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }

        error = error - r * r * error;

        if error <= 0.001 * ac[0] {
            break;
        }
    }
}

pub fn autocorr(
    x: &[f32],
    ac: &mut [f32],
    window: Option<&[f32]>,
    overlap: usize,
    lag: usize,
    n: usize,
    arch: crate::Arch,
) {
    let xx_vec: FixedVec<f32, 2880>;
    let xx: &[f32] = if let Some(win) = window {
        if x.len() < n {
            return;
        }
        xx_vec = {
            let mut v: FixedVec<f32, 2880> = FixedVec::from_slice(&x[0..n]);
            for i in 0..overlap {
                v[i] *= win[i];
                v[n - 1 - i] *= win[i];
            }
            v
        };
        &xx_vec
    } else {
        &x[0..n]
    };

    let fast_n = n - lag;

    pitch_xcorr(xx, xx, ac, fast_n, lag + 1, arch);

    for k in 0..=lag {
        let mut d = 0.0f32;
        for i in (k + fast_n)..n {
            d += xx[i] * xx[i - k];
        }
        ac[k] += d;
    }
}

/// The largest order of [`celt_fir`].
const MAX_FIR_ORDER: usize = 24;

/// libopus `celt_fir_c` at `arch`: `y[i] = x[i] + sum_j num[ord-1-j] *
/// x[i+j-ord]` with the history in `x[..ord]` and the input in
/// `x[ord..ord + n]`. The outputs in whole groups of four are
/// `xcorr_kernel` at `arch` (libopus's SSE kernel at the x86 levels), the
/// rest sums in order.
pub fn celt_fir(x: &[f32], num: &[f32], y: &mut [f32], n: usize, ord: usize, arch: crate::Arch) {
    assert!(ord <= MAX_FIR_ORDER);
    let mut rnum = [0.0f32; MAX_FIR_ORDER];
    for (r, &v) in rnum[..ord].iter_mut().zip(num[..ord].iter().rev()) {
        *r = v;
    }
    let rnum = &rnum[..ord];
    let groups = n & !3;
    for i in (0..groups).step_by(4) {
        let sum: [f32; 4] = x[ord + i..ord + i + 4].try_into().unwrap();
        let out = match arch {
            crate::Arch::Scalar => xcorr_kernel::<4, false>(rnum, &x[i..], sum),
            _ => xcorr_kernel::<4, true>(rnum, &x[i..], sum),
        };
        y[i..i + 4].copy_from_slice(&out);
    }
    for i in groups..n {
        let mut sum = x[ord + i];
        for (j, &r) in rnum.iter().enumerate() {
            sum += r * x[i + j];
        }
        y[i] = sum;
    }
}
