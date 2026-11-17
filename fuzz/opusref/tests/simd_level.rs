//! opus-rs at libopus's x86 levels against the reference built as x86-64
//! configure builds it (feature `simd`), kernel by kernel.
#![cfg(feature = "simd")]
use opus_rs::Arch;
use opusref::*;

/// The levels this CPU has, with libopus's arch numbers.
fn levels() -> Vec<(Arch, i32)> {
    let cpu = Arch::native();
    [(Arch::Sse2, 2), (Arch::Sse4_1, 3), (Arch::Avx2, 4)]
        .into_iter()
        .filter(|&(a, _)| a <= cpu)
        .collect()
}

fn noise(n: usize, seed: &mut u32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((*seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * scale
        })
        .collect()
}

#[test]
fn inner_products() {
    let mut seed = 1;
    for (arch, c) in levels() {
        for n in 0..70 {
            let x = noise(n, &mut seed, 2.0);
            let y = noise(n, &mut seed, 3e3);
            let z = noise(n, &mut seed, 1e-2);
            let a = opus_rs::pitch::inner_prod(&x, &y, n, arch);
            let b = unsafe { ref_celt_inner_prod(x.as_ptr(), y.as_ptr(), n as i32, c) };
            assert_eq!(a.to_bits(), b.to_bits(), "inner_prod n={n} {arch:?}");
            let (a1, a2) = opus_rs::pitch::dual_inner_prod(&x, &y, &z, n, arch);
            let (mut b1, mut b2) = (0.0, 0.0);
            unsafe {
                ref_dual_inner_prod(
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    n as i32,
                    &mut b1,
                    &mut b2,
                    c,
                )
            };
            assert_eq!(
                (a1.to_bits(), a2.to_bits()),
                (b1.to_bits(), b2.to_bits()),
                "dual n={n}"
            );
        }
    }
}

#[test]
fn remove_doubling_and_band_kernels() {
    let mut seed = 2;
    let mode = opus_rs::modes::default_mode();
    for (arch, c) in levels() {
        for k in 0..200 {
            let x = noise(1004, &mut seed, [1e-3, 1.0, 3e3][k % 3]);
            let (prev_period, prev_gain) = (15 + k % 300, (k % 5) as f32 * 0.2);
            let t0 = 20 + (k * 37) % 990;
            let mut ta = t0;
            let ga = opus_rs::pitch::remove_doubling(
                &x,
                1024,
                15,
                960,
                &mut ta,
                prev_period,
                prev_gain,
                arch,
            );
            let (mut tb, mut y) = (t0 as i32, x.clone());
            let gb = unsafe {
                remove_doubling(
                    y.as_mut_ptr(),
                    1024,
                    15,
                    960,
                    &mut tb,
                    prev_period as i32,
                    prev_gain,
                    c,
                )
            };
            assert_eq!(
                (ta, ga.to_bits()),
                (tb as usize, gb.to_bits()),
                "remove_doubling k={k}"
            );

            let lm = k % 4;
            let x = noise(960 << lm >> 3, &mut seed, 1e3);
            let mut a = vec![0.0f32; 21];
            let mut b = vec![0.0f32; 21];
            opus_rs::bands::compute_band_energies(mode, &x, &mut a, 21, 1, lm, arch);
            unsafe {
                compute_band_energies(
                    ref_celt_mode48(),
                    x.as_ptr(),
                    b.as_mut_ptr(),
                    21,
                    1,
                    lm as i32,
                    c,
                )
            };
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );

            let n = 1 + k % 64;
            let (x, y) = (noise(n, &mut seed, 1.0), noise(n, &mut seed, 1.0));
            let a = opus_rs::bands::stereo_itheta(&x, &y, false, n, arch);
            let b = unsafe { stereo_itheta(x.as_ptr(), y.as_ptr(), 0, n as i32, c) };
            assert_eq!(a, b, "stereo_itheta n={n}");
            let (mut a, mut b) = (x.clone(), x);
            opus_rs::bands::renormalise_vector(&mut a, n, 0.7, arch);
            unsafe { renormalise_vector(b.as_mut_ptr(), n as i32, 0.7, c) };
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}
