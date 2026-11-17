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
