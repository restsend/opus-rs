//! The mode follows libopus where its signal analysis estimates speech.
//!
//! From complexity 7, at 16 kHz and above, libopus takes its estimate of
//! speech for the choice between SILK (or Hybrid) and CELT-only from the
//! analysis's music probability (the one for staying in the previous mode),
//! not from the application. The tests code with opus-rs and libopus at
//! complexity 9, at the API rate's bandwidth in both, and compare the mode
//! of every frame.

#[path = "common/vbr.rs"]
mod vbr;

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Encoder as CEnc,
};
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

/// A voiced-speech-like signal (as in `issue_27_libopus_oracle.rs`), which
/// the analysis takes for music after a third of a second.
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

/// The mode of every packet: S(ILK-only), H(ybrid) or C(ELT-only), for
/// opus-rs and for libopus.
fn modes(input: &[f32], sr: usize, voip: bool, bitrate: i32) -> (String, String) {
    let frame = sr / 50;
    let mut rs = OpusEncoder::new(
        sr as i32,
        1,
        if voip {
            Application::Voip
        } else {
            Application::Audio
        },
    )
    .unwrap();
    rs.bitrate_bps = bitrate;
    rs.complexity = 9;
    let mut c = CEnc::new(
        sr as u32,
        CCh::Mono,
        if voip { CApp::Voip } else { CApp::Audio },
    )
    .unwrap();
    c.set_bitrate(CBitrate::Bits(bitrate)).unwrap();
    c.set_complexity(9).unwrap();
    c.set_bandwidth(match sr {
        16000 => CBw::Wideband,
        24000 => CBw::Superwideband,
        _ => CBw::Fullband,
    })
    .unwrap();
    let letter = |p: &[u8]| match p[0] >> 3 {
        0..=11 => 'S',
        12..=15 => 'H',
        _ => 'C',
    };
    let (mut ours, mut theirs) = (String::new(), String::new());
    let mut packet = [0u8; 1500];
    for x in input.chunks_exact(frame) {
        let n = rs.encode(x, frame, &mut packet).unwrap();
        ours.push(letter(&packet[..n]));
        let n = c.encode_float(x, &mut packet).unwrap();
        theirs.push(letter(&packet[..n]));
    }
    (ours, theirs)
}

fn check(label: &str, input: &[f32], sr: usize, voip: bool, bitrate: i32) {
    let (ours, theirs) = modes(input, sr, voip, bitrate);
    // The analysis changes libopus's mode here: not one mode throughout.
    let first = theirs.chars().next().unwrap();
    assert!(
        theirs.chars().any(|m| m != first),
        "{label}: libopus {theirs}"
    );
    assert_eq!(
        ours, theirs,
        "{label}: opus-rs (first line) against libopus"
    );
}

#[test]
fn voip_codes_a_harmonic_signal_like_libopus() {
    // libopus starts in SILK and goes to CELT-only once the analysis finds
    // music; the application's estimate alone keeps SILK.
    check(
        "voiced, 16 kHz VoIP, 32 kb/s",
        &voiced(16000, 16000 * 4),
        16000,
        true,
        32000,
    );
    check(
        "voiced, 48 kHz VoIP, 32 kb/s",
        &voiced(48000, 48000 * 4),
        48000,
        true,
        32000,
    );
}

#[test]
fn modes_follow_libopus_on_the_speech_fixture() {
    check(
        "speech, 48 kHz VoIP, 32 kb/s",
        &vbr::speech(48000, 1, 48000 * 18),
        48000,
        true,
        32000,
    );
    check(
        "speech, 24 kHz Audio, 20 kb/s",
        &vbr::speech(24000, 1, 24000 * 18),
        24000,
        false,
        20000,
    );
}
