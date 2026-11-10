//! Unconstrained CELT VBR spends what libopus spends.
//!
//! libopus's VBR target (`compute_vbr`) adds a pitch change, temporal VBR,
//! the smoothed stereo saving and the signal's depth to the analysis
//! terms, at the equivalent rate. The tests code the fixture and a
//! synthetic voiced signal in CELT-only mode (restricted low delay,
//! complexity 9, unconstrained VBR) with opus-rs and with libopus 1.6.1,
//! whose CELT encoder opus-rs follows, and compare the mean rates.

#[path = "common/vbr.rs"]
mod vbr;

use opus_rs::{Application, OpusEncoder};
use opusic_sys as ffi;
use std::f32::consts::PI;

/// The mean rate (kb/s) of libopus 1.6.1 and of opus-rs coding `input`.
fn rates(sr: usize, ms: usize, bitrate: i32, input: &[f32]) -> (f64, f64) {
    let frame = sr * ms / 1000;
    let frames = input.len() / frame;
    let mut rs = OpusEncoder::new(sr as i32, 1, Application::RestrictedLowDelay).unwrap();
    rs.bitrate_bps = bitrate;
    rs.complexity = 9;
    rs.set_vbr_constraint(false);
    let mut error = 0;
    let c = unsafe {
        ffi::opus_encoder_create(
            sr as i32,
            1,
            ffi::OPUS_APPLICATION_RESTRICTED_LOWDELAY,
            &mut error,
        )
    };
    assert!(!c.is_null() && error == 0);
    unsafe {
        assert_eq!(
            ffi::opus_encoder_ctl(c, ffi::OPUS_SET_BITRATE_REQUEST, bitrate),
            0
        );
        assert_eq!(
            ffi::opus_encoder_ctl(c, ffi::OPUS_SET_COMPLEXITY_REQUEST, 9),
            0
        );
        assert_eq!(
            ffi::opus_encoder_ctl(c, ffi::OPUS_SET_VBR_CONSTRAINT_REQUEST, 0),
            0
        );
    }
    let (mut rs_bytes, mut c_bytes) = (0usize, 0usize);
    let mut packet = [0u8; 1500];
    for f in 0..frames {
        let pcm = &input[f * frame..(f + 1) * frame];
        rs_bytes += rs.encode(pcm, frame, &mut packet).unwrap();
        let n = unsafe {
            ffi::opus_encode_float(c, pcm.as_ptr(), frame as i32, packet.as_mut_ptr(), 1500)
        };
        assert!(n > 0);
        c_bytes += n as usize;
    }
    unsafe { ffi::opus_encoder_destroy(c) };
    let secs = (frames * frame) as f64 / sr as f64;
    let kbps = |bytes: usize| bytes as f64 * 8.0 / secs / 1000.0;
    (kbps(c_bytes), kbps(rs_bytes))
}

/// A voiced-speech-like signal (as in `issue_27_libopus_oracle.rs`): a
/// 140 Hz harmonic series with a slow pitch drift and a syllable-rate
/// envelope.
fn voiced(sr: usize, n: usize) -> Vec<f32> {
    let mut phase = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let f0 = 140.0 + 15.0 * (2.0 * PI * 0.7 * t).sin();
            phase += 2.0 * PI * f0 / sr as f32;
            let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin();
            let l: f32 = (1..=8)
                .map(|h| 0.25 / h as f32 * (h as f32 * phase).sin())
                .sum();
            l * env * 0.5
        })
        .collect()
}

fn check(label: &str, sr: usize, ms: usize, bitrate: i32, input: &[f32]) {
    let (libopus, opus_rs) = rates(sr, ms, bitrate, input);
    let diff = 100.0 * (opus_rs - libopus) / libopus;
    println!("{label}: libopus {libopus:.2} kb/s, opus-rs {opus_rs:.2} ({diff:+.2}%)");
    assert!(
        diff.abs() <= 1.0,
        "{label}: opus-rs {opus_rs:.2} kb/s, libopus {libopus:.2} kb/s"
    );
}

#[test]
fn speech_vbr_rate_follows_libopus() {
    let speech = vbr::speech(48000, 1, 48000 * 18);
    check("speech, 48 kHz, 20 ms, 32 kb/s", 48000, 20, 32000, &speech);
    check("speech, 48 kHz, 10 ms, 64 kb/s", 48000, 10, 64000, &speech);
}

#[test]
fn voiced_vbr_rate_follows_libopus() {
    check(
        "voiced, 24 kHz, 20 ms, 48 kb/s",
        24000,
        20,
        48000,
        &voiced(24000, 24000 * 4),
    );
}
