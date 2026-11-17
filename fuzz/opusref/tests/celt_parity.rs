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
                "sine" => (440.0 * t * 6.2832).sin() * 0.5,
                _ => unreachable!(),
            }
        })
        .collect()
}

/// Encodes `frames` frames at 48 kHz, complexity 10, with both encoders;
/// rc 0 VBR, 1 constrained VBR, 2 CBR. Returns per frame whether the
/// packets are identical.
fn identical(ch: usize, ms: usize, kbps: i32, rc: i32, sig: &str, frames: usize) -> Vec<bool> {
    let frame = 48 * ms;
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
            let m = unsafe { opus_encode_float(r, pcm.as_ptr(), frame as i32, b.as_mut_ptr(), 1500) };
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
    assert_eq!(identical(1, 20, 24, 2, "sine", 2), [true, true]);
}

/// The static mode's MDCT twiddles and window are libopus's
/// static_modes_float.h literals.
#[test]
fn static_mode_tables_are_libopus_literals() {
    let mode = opus_rs::modes::default_mode();
    let trig = unsafe { std::slice::from_raw_parts(ref_mdct_trig(), 1800) };
    let mut at = 0;
    for shift in 0..4 {
        let t = mode.mdct.get_trig_debug(shift);
        assert_eq!(t, &trig[at..at + t.len()], "shift {shift}");
        at += t.len();
    }
    assert_eq!(mode.window, unsafe { std::slice::from_raw_parts(ref_window(), 120) });
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
    assert_eq!(identical(1, 20, 64, 0, "voice", 2), [true, true]);
}

/// transient_analysis's harmonic mean reads libopus's inv_table.
#[test]
fn transient_analysis_uses_libopus_inv_table() {
    assert_eq!(identical(1, 20, 32, 1, "noise", 1), [true]);
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
        opus_rs::bands::amp2log2(mode, 0, 21, &e, &mut a, 1);
        unsafe { amp2Log2(ref_celt_mode48(), 21, 21, e.as_ptr(), b.as_mut_ptr(), 1) };
        assert_eq!(a, b);
    }
}
