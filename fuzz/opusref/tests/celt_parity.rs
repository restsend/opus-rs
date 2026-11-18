//! CELT-only (RestrictedLowDelay) byte identity with the scalar libopus
//! reference, one test per fixed divergence, and the DSP kernels the
//! encoder shares with it.
use opus_rs::{Application, OpusEncoder};
use opusref::*;

/// The signals of examples/parity.rs.
fn signal(sig: &str, f: usize, frame: usize, ch: usize, seed: &mut u32) -> Vec<f32> {
    (0..frame * ch)
        .map(|i| {
            let t = (f * frame + i / ch) as f32 / 48000.0;
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let noise = (*seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
            let voice = (0..20)
                .map(|h| ((h + 1) as f32 * 140.0 * t * 6.2832).sin() / (h + 1) as f32)
                .sum::<f32>()
                * 0.2;
            match sig {
                "voice" => voice,
                "noise" => noise * 0.5,
                "quiet" => noise * 3e-6,
                "sine" => (440.0 * t * 6.2832).sin() * 0.5,
                // Noise on the first channel, voice on the second.
                "split" if i % ch == 0 => noise * 0.5,
                "split" => voice,
                // Noise growing by 9 dB per 20 ms, up to -6 dBFS.
                "grow" => noise * (0.0005 * 2f32.powf(t * 75.0)).min(0.5),
                _ => unreachable!(),
            }
        })
        .collect()
}

/// Encodes `frames` frames at 48 kHz, complexity 10, with both encoders;
/// rc 0 VBR, 1 constrained VBR, 2 CBR. Returns per frame whether the
/// packets are identical.
fn identical(ch: usize, ms: f32, kbps: i32, rc: i32, sig: &str, frames: usize) -> Vec<bool> {
    let frame = (48.0 * ms) as usize;
    let mut e = OpusEncoder::new(48000, ch, Application::RestrictedLowDelay).unwrap();
    e.bitrate_bps = kbps * 1000;
    e.complexity = 10;
    e.use_cbr = rc == 2;
    e.set_vbr_constraint(rc == 1);
    let mut err = 0;
    let r = unsafe { opus_encoder_create(48000, ch as i32, 2051, &mut err) };
    unsafe {
        opus_encoder_ctl(r, SET_BITRATE, kbps * 1000);
        opus_encoder_ctl(r, SET_COMPLEXITY, 10);
        opus_encoder_ctl(r, SET_VBR, (rc != 2) as i32);
        opus_encoder_ctl(r, SET_VBR_CONSTRAINT, (rc == 1) as i32);
    }
    let mut seed = 1;
    let out = (0..frames)
        .map(|f| {
            let pcm = signal(sig, f, frame, ch, &mut seed);
            let (mut a, mut b) = ([0u8; 1500], [0u8; 1500]);
            let n = e.encode(&pcm, frame, &mut a).unwrap();
            let m =
                unsafe { opus_encode_float(r, pcm.as_ptr(), frame as i32, b.as_mut_ptr(), 1500) };
            a[..n] == b[..m as usize]
        })
        .collect();
    unsafe { opus_encoder_destroy(r) };
    out
}

/// RestrictedLowDelay DC-rejects its input like the other applications
/// (opus_encoder.c:2008).
#[test]
fn rld_input_is_dc_rejected() {
    assert_eq!(identical(1, 20.0, 24, 2, "sine", 2), [true, true]);
}

/// The static mode's MDCT twiddles, window and pre-emphasis are libopus's
/// static_modes_float.h literals. libopus interleaves each size's
/// twiddles as (-sin, cos) pairs; opus-rs keeps the N/4 cosines followed
/// by the N/4 negated sines.
#[test]
fn static_mode_tables_are_libopus_literals() {
    let mode = opus_rs::modes::default_mode();
    let trig = unsafe { std::slice::from_raw_parts(ref_mdct_trig(), 1800) };
    let mut at = 0;
    for shift in 0..4 {
        let t = mode.mdct.get_trig_debug(shift);
        let (cos, msin) = t.split_at(t.len() / 2);
        let pairs = &trig[at..at + t.len()];
        let ref_msin: Vec<f32> = pairs.iter().step_by(2).copied().collect();
        let ref_cos: Vec<f32> = pairs.iter().skip(1).step_by(2).copied().collect();
        assert_eq!(cos, ref_cos, "cos, shift {shift}");
        assert_eq!(msin, ref_msin, "-sin, shift {shift}");
        at += t.len();
    }
    assert_eq!(mode.window, unsafe {
        std::slice::from_raw_parts(ref_window(), 120)
    });
    assert_eq!(mode.preemph, unsafe { *ref_preemph().cast::<[f32; 4]>() });
}

/// The forward MDCT's third folding loop runs from where the first two
/// stop up to N/4 (mdct.c:183-200), also when an overlap above N/2
/// leaves the second loop empty.
#[test]
fn mdct_forward_folds_overlaps_above_half_the_frame() {
    let mode = opus_rs::modes::default_mode();
    let mut seed = 3u32;
    let mut rnd = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5
    };
    for (shift, overlap) in [(3, 120), (3, 136), (3, 200), (2, 240), (2, 300)] {
        let n2 = 960 >> shift;
        let window: Vec<f32> = (0..overlap).map(|_| rnd()).collect();
        let input: Vec<f32> = (0..n2 + overlap).map(|_| rnd() * 1000.0).collect();
        let mut a = vec![0.0f32; n2];
        mode.mdct
            .forward(&input, &mut a, &window, overlap, shift, 1);
        let mut b = vec![0.0f32; n2];
        let mut x = input.clone();
        unsafe {
            ref_mdct_forward_window(
                x.as_mut_ptr(),
                b.as_mut_ptr(),
                window.as_ptr(),
                overlap as i32,
                shift as i32,
                1,
            )
        };
        assert_eq!(a, b, "shift {shift} overlap {overlap}");
    }
}

/// The static mode's FFTs round like libopus's kiss_fft butterflies.
#[test]
fn fft_matches_libopus() {
    use opus_rs::kiss_fft::{KissCpx, opus_fft_impl};
    let mode = opus_rs::modes::default_mode();
    let mut seed = 7u32;
    for shift in 0..4 {
        let n = 480 >> shift;
        let x: Vec<f32> = (0..2 * n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * 2000.0
            })
            .collect();
        let mut a: Vec<KissCpx> = x.chunks(2).map(|c| KissCpx::new(c[0], c[1])).collect();
        opus_fft_impl(mode.mdct.fft(shift), &mut a);
        let mut b = x.clone();
        unsafe { ref_fft_impl(b.as_mut_ptr(), shift as i32) };
        let a: Vec<f32> = a.iter().flat_map(|c| [c.r, c.i]).collect();
        assert_eq!(a, b, "shift {shift}");
    }
}

/// The raw bits at the end of a VBR frame with free space stay at the end
/// (ec_enc_done zeroes the gap).
#[test]
fn free_space_stays_between_range_and_raw_bits() {
    assert_eq!(identical(1, 20.0, 64, 0, "voice", 2), [true, true]);
}

/// transient_analysis's harmonic mean reads libopus's inv_table.
#[test]
fn transient_analysis_uses_libopus_inv_table() {
    assert_eq!(identical(1, 20.0, 32, 1, "noise", 1), [true]);
}

/// amp2Log2 takes celt_log2 in double precision (mathops.h:346).
#[test]
fn band_log_energy_matches_libopus() {
    let mode = opus_rs::modes::default_mode();
    let mut seed = 3u32;
    for _ in 0..5000 {
        let e: Vec<f32> = (0..21)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                f32::from_bits(0x3000_0000 + (seed >> 5))
            })
            .collect();
        let mut a = [0f32; 21];
        let mut b = [0f32; 21];
        opus_rs::bands::amp2log2(mode, 21, 21, &e, &mut a, 1);
        unsafe { amp2Log2(ref_celt_mode48(), 21, 21, e.as_ptr(), b.as_mut_ptr(), 1) };
        assert_eq!(a, b);
    }
}

/// compute_band_energies adds 1e-27 after the sum of squares
/// (bands.c:160), which shows on very quiet bands.
#[test]
fn band_energies_add_the_floor_after_the_sum() {
    let mode = opus_rs::modes::default_mode();
    let mut seed = 9u32;
    for k in 0..200 {
        let scale = 10f32.powi(-(k % 20) - 5);
        let x: Vec<f32> = (0..960)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * scale
            })
            .collect();
        let mut a = [0f32; 21];
        let mut b = [0f32; 21];
        opus_rs::bands::compute_band_energies(mode, &x, &mut a, 21, 1, 3, opus_rs::Arch::select());
        unsafe {
            compute_band_energies(
                ref_celt_mode48(),
                x.as_ptr(),
                b.as_mut_ptr(),
                21,
                1,
                3,
                ref_arch(),
            )
        };
        assert_eq!(a, b, "scale {scale}");
    }
}

/// 2.5 ms frames' dynalloc reads the previous frame's energies, before
/// the coarse energy updates them (celt_encoder.c:2248).
#[test]
fn dynalloc_reads_the_previous_frame_energy() {
    assert_eq!(identical(1, 2.5, 192, 1, "noise", 2), [true, true]);
}

/// Stable bands bias the coarse energy by a quarter of the last frame's
/// fine-energy error (celt_encoder.c:2283-2294).
#[test]
fn coarse_energy_follows_the_previous_error() {
    assert_eq!(identical(1, 20.0, 32, 1, "noise", 5), [true; 5]);
}

/// compute_theta takes the Q14 part of stereo_itheta's Q30 angle.
#[test]
fn split_angle_is_truncated_from_q30() {
    assert_eq!(identical(1, 20.0, 64, 0, "voice", 5), [true; 5]);
}

/// Silent halves give a zero angle: the energies start at 0 (vq.c:729), so
/// celt_atan2p_norm sees them below its 1e-18 floor.
#[test]
fn silent_split_has_zero_angle() {
    let z = [0f32; 4];
    let c = unsafe { stereo_itheta(z.as_ptr(), z.as_ptr(), 0, 4, ref_arch()) };
    assert_eq!(
        opus_rs::bands::stereo_itheta(&z, &z, false, 4, opus_rs::Arch::select()),
        c
    );
}

/// How often stereo_itheta (Q30) of random vectors differs from libopus's,
/// with the second vector scaled by `y_scale`.
fn itheta_mismatches(y_scale: f32, stereo: &[bool]) -> usize {
    let mut seed = 11u32;
    let mut bad = 0;
    for k in 0..20000 {
        let n = 1 + k % 16;
        let mut v = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5
        };
        let x: Vec<f32> = (0..n).map(|_| v()).collect();
        let y: Vec<f32> = (0..n).map(|_| v() * y_scale).collect();
        for &stereo in stereo {
            let a = opus_rs::bands::stereo_itheta(&x, &y, stereo, n, opus_rs::Arch::select());
            let b = unsafe {
                stereo_itheta(x.as_ptr(), y.as_ptr(), stereo as i32, n as i32, ref_arch())
            };
            bad += (a != b) as usize;
        }
    }
    bad
}
/// celt_atan_norm evaluates 2/pi * (x + x*x^2*(A03 + ...)) (mathops.h:158);
/// small angles, where the higher terms vanish, show the operation order.
#[test]
fn small_split_angles_match_libopus() {
    assert_eq!(itheta_mismatches(0.001, &[false]), 0);
}

/// celt_atan_norm uses libopus's coefficients (mathops.h:149-156).
#[test]
fn split_angles_match_libopus() {
    assert_eq!(itheta_mismatches(1.0, &[false, true]), 0);
}

/// The intra/inter choice weighs only the coarse-energy changes the budget
/// forces, after the decay bound (quant_bands.c:212, 246).
#[test]
fn intra_decision_counts_only_budget_changes() {
    assert_eq!(identical(1, 20.0, 24, 2, "voice", 3), [true; 3]);
}

/// The VBR target's depth limit takes dynalloc's maxDepth, from the band
/// energies before the coarse-energy bias (celt_encoder.c:1068-1078).
#[test]
fn vbr_target_uses_dynalloc_max_depth() {
    assert_eq!(identical(1, 20.0, 128, 0, "quiet", 4), [true; 4]);
}

/// The constrained-VBR drift follows the float build's alpha, 1/(count+20)
/// and a truncating conversion (celt_encoder.c:2501-2516).
#[test]
fn vbr_drift_uses_the_float_alpha() {
    assert_eq!(identical(1, 10.0, 128, 1, "noise", 8), [true; 8]);
}

/// Stereo frames run the pitch pre-filter, with libopus's stereo cancel
/// test (celt_encoder.c:2038, 1555-1563).
#[test]
fn stereo_frames_use_the_pre_filter() {
    assert_eq!(identical(2, 10.0, 32, 0, "voice", 10), [true; 10]);
}

/// Stereo bands below the intensity band code the angle rounded down or
/// up, whichever correlates better with the input (bands.c:1808-1895).
#[test]
fn stereo_angle_uses_rate_distortion_choice() {
    assert_eq!(identical(2, 10.0, 32, 2, "voice", 12), [true; 12]);
}

/// Below 32 kb/s equivalent, the stereo CELT input fades towards mono
/// (opus_encoder.c:2320-2348).
#[test]
fn low_rate_stereo_fades_towards_mono() {
    assert_eq!(identical(2, 10.0, 24, 0, "noise", 10), [true; 10]);
}

/// Resynthesised pulses scale by (1/sqrt(Ryy))*gain (vq.c:150-166).
#[test]
fn pulses_resynthesise_with_reciprocal_sqrt() {
    assert_eq!(identical(2, 10.0, 96, 2, "sine", 3), [true; 3]);
}

/// How often renormalise_vector of random vectors with entries of size
/// `scale` differs from libopus's.
fn renormalise_mismatches(scale: f32) -> usize {
    let mut seed = 5u32;
    let mut bad = 0;
    for k in 0..20000 {
        let n = 1 + k % 24;
        let x: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5) * scale
            })
            .collect();
        let gain = 0.25 + (k % 7) as f32 * 0.125;
        let mut a = x.clone();
        let mut b = x.clone();
        opus_rs::bands::renormalise_vector(&mut a, n, gain, opus_rs::Arch::select());
        unsafe { renormalise_vector(b.as_mut_ptr(), n as i32, gain, ref_arch()) };
        bad += (a != b) as usize;
    }
    bad
}
/// renormalise_vector scales by (1/sqrt(E))*gain (vq.c:704-715).
#[test]
fn renormalise_uses_reciprocal_sqrt() {
    assert_eq!(renormalise_mismatches(1.0), 0);
}

/// renormalise_vector adds EPSILON after the sum of squares (vq.c:704).
#[test]
fn renormalise_adds_epsilon_after_the_sum() {
    assert_eq!(renormalise_mismatches(1e-7), 0);
}

/// The CBR/CVBR dynalloc cap triggers on whole bytes and leaves the last
/// boosted band the rest of the cap in eighth bits (celt_encoder.c:1253-1260).
#[test]
fn dynalloc_cap_follows_libopus() {
    assert_eq!(identical(2, 20.0, 32, 1, "sine", 4), [true; 4]);
}

/// The encoder's resynthesis seeds its folding noise with the previous
/// frame's final range (celt_encoder.c:2688, 2825).
#[test]
fn resynthesis_noise_seeds_from_the_last_range() {
    assert_eq!(identical(2, 20.0, 24, 2, "noise", 12), [true; 12]);
}

/// A frame whose band energies rise by more than 1 (6 dB) on average over
/// the previous frame's is coded as transient (celt_encoder.c:473-508,
/// 2213-2235).
#[test]
fn rising_band_energy_patches_the_transient_flag() {
    assert_eq!(identical(1, 20.0, 32, 0, "grow", 8), [true; 8]);
}

/// Below about 17-19 kb/s equivalent, stereo CELT frames code a mono
/// stream of the channels' average (opus_encoder.c:1427-1455,
/// celt_encoder.c:539-543).
#[test]
fn low_rate_stereo_codes_a_mono_stream() {
    assert_eq!(identical(2, 10.0, 24, 2, "voice", 10), [true; 10]);
}

/// 2.5 ms stereo frames always use M/S stereo (celt_encoder.c:2399-2401).
#[test]
fn short_stereo_frames_use_mid_side() {
    assert_eq!(identical(2, 2.5, 96, 0, "split", 12), [true; 12]);
}
