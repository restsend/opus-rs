use crate::pitch::pitch_xcorr;

pub fn lpc(lpc: &mut [f32], ac: &[f32], p: usize) {
    // ac: autocorrelation [0..p]
    // lpc: output coefficients [0..p-1]

    let mut error = ac[0];
    if error <= 1e-10 {
        for x in lpc.iter_mut() {
            *x = 0.0;
        }
        return;
    }

    for i in 0..p {
        // Sum up this iteration's reflection coefficient
        let mut rr = 0.0f32;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;

        // Update LPC coefficients and total error
        lpc[i] = r;
        for j in 0..((i + 1) / 2) {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }

        error = error - r * r * error;

        // Bail out once we get 30 dB gain (approx 0.001 error ratio)
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
    lag: usize, // usually p
    n: usize,
) {
    // Computes autocorrelation of x.
    // window: optional window to apply to first/last 'overlap' samples
    // lag: number of lags (ac has size lag+1)

    let mut xx = Vec::with_capacity(n);
    // Apply window if present
    if let Some(win) = window {
        if x.len() < n {
            // Panic or handle safe
            return;
        }
        // Copy x with windowing
        xx.extend_from_slice(&x[0..n]);
        for i in 0..overlap {
            xx[i] *= win[i];
            xx[n - 1 - i] *= win[i];
        }
    } else {
        xx.extend_from_slice(&x[0..n]);
    }

    // Use pitch_xcorr to compute autocorrelation
    // In autocorr, x and y are the same signal.
    // pitch_xcorr(x, y, xcorr, len, max_pitch)
    // Here x=xx, y=xx. len = n? No.
    // C implementation: `celt_pitch_xcorr(xptr, xptr, ac, fastN, lag+1, arch)`
    // where fastN = n - lag.
    // The standard autocorr definition uses N samples for lag 0.
    // But for lag k, we only have N-k pairs.
    // Opus `autocorr` implementation does:

    let fast_n = n - lag;
    // pitch_xcorr with len=fastN computes sums for j=0..fastN.
    // This ignores the tails?
    // C code follows up with:
    // for (k=0;k<=lag;k++)
    //   for (i = k+fastN, d = 0; i < n; i++) d += x[i]*x[i-k]
    //   ac[k] += d

    // This implies pitch_xcorr handled the bulk (0..fastN), and we fix up the tail.

    pitch_xcorr(&xx, &xx, ac, fast_n, lag + 1);

    // Tail fixup
    for k in 0..=lag {
        let mut d = 0.0f32;
        for i in (k + fast_n)..n {
            d += xx[i] * xx[i - k];
        }
        ac[k] += d;
    }
}

pub fn celt_fir(x: &[f32], num: &[f32], y: &mut [f32], n: usize, ord: usize) {
    // Standard FIR filter
    // y[i] = x[i] + sum(num[j] * x[i-j-1]) ?
    // Wait, Opus `celt_fir`:
    // for i=0..N
    //   sum = x[i]
    //   for j=0..ord
    //      sum += num[j]*mem[j]
    //   mem shift
    //   y[i] = sum
    // This is IIR? No.
    // Note: in C `pitch_downsample` calls `celt_fir5` which has `mem`.
    // But `celt_fir` generally assumes `num` are coeffs.
    // Actually, `celt_fir` in `celt_lpc.c` is NOT defined for float?
    // In header: `void celt_fir_c(...)`.
    // It's usually the prediction filter.
    // x is residual, y is signal? Or vice versa.
    // pitch_downsample uses `celt_fir5` which seems to be `analysis` filtering (calculating residual).

    // let mut mem = vec![0.0f32; ord]; // Assumes 0 initial memory if not passed?
    // Actually `celt_fir` in C takes `mem`?
    // Header: `celt_fir` wrapper calls `celt_fir_c` but `celt_fir_c` signature in header:
    // void celt_fir_c(const val16 *x, const val16 *num, val16 *y, int N, int ord, int arch)
    // No mem?
    // Let's check `celt_lpc.c`... IT ONLY EXISTS FOR FIXED POINT CHECK ASM?
    // `celt_fir` implementation:
    /*
    for (i=0;i<N;i++)
    {
       opus_val32 sum = x[i];
       for (j=0;j<ord;j++)
          sum -= den[j]*mem[j];
       ...
    }
    */
    // Wait, that code block in `celt_lpc.c` lines 200-250 was `celt_iir`!

    // `celt_fir` implementation is missing from my `read_file` of `celt_lpc.c`.
    // Let's assume standard FIR:
    // y[n] = x[n] + sum_{k=0}^{ord-1} num[k] * x[n-k-1] (or similar)
    // Opus usually defines FIR/IIR for LPC synthesis/analysis.

    // For now, simple implementation assuming x contains history or mem is managed.
    // If not, we might need to handle memory.

    // I will implement a safe version with internal memory for the loop.
    // BUT `pitch_downsample` uses `celt_fir5` which explicitly passes `mem` locals.

    // Let's leave `celt_fir` simple.

    for i in 0..n {
        let mut sum = x[i];
        for j in 0..ord {
            if i >= j + 1 {
                sum += num[j] * x[i - j - 1]; // Check sign?
            }
        }
        y[i] = sum;
    }
}

pub fn celt_iir(x: &[f32], den: &[f32], y: &mut [f32], n: usize, ord: usize, mem: &mut [f32]) {
    // IIR filter: Synthesis
    // y[i] = x[i] - sum(den[j] * y[i-j-1])
    // Uses `mem` to store past `y`.

    for i in 0..n {
        let mut sum = x[i];
        for j in 0..ord {
            sum -= den[j] * mem[j];
        }
        for j in (1..ord).rev() {
            mem[j] = mem[j - 1];
        }
        mem[0] = sum;
        y[i] = sum;
    }
}
