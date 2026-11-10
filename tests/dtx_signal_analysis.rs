//! DTX follows libopus where its signal analysis runs.
//!
//! From complexity 7, at 16 kHz and above, libopus decides DTX itself in
//! every mode: a frame is inactive when the analysis's activity
//! probability is below 0.1 and the frame is more than 25 dB below the
//! loudest active one, and SILK's voice activity detection is told so. The
//! tests code speech, a quiet noise floor, speech and digital silence with
//! opus-rs and libopus at complexity 9 and compare, frame by frame, which
//! packets are the ToC alone.

#[path = "common/vbr.rs"]
mod vbr;

use opus::{Application as CApp, Bitrate as CBitrate, Channels as CCh, Encoder as CEnc};
use opus_rs::{Application, OpusEncoder};

/// Speech 2 s, a noise floor 3 s, speech 2 s, digital silence 3 s.
fn input(rate: usize) -> Vec<f32> {
    let speech = vbr::speech(rate, 1, 4 * rate);
    let mut out = speech[..2 * rate].to_vec();
    let mut seed = 3u32;
    out.extend((0..3 * rate).map(|_| {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        0.0005 * ((seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0)
    }));
    out.extend_from_slice(&speech[2 * rate..]);
    out.extend(std::iter::repeat_n(0.0, 3 * rate));
    out
}

/// Whether each packet is the ToC alone, for opus-rs and for libopus.
fn toc_only(rate: usize, voip: bool, ms: usize) -> (Vec<bool>, Vec<bool>) {
    let pcm = input(rate);
    let frame = rate * ms / 1000;
    let mut rs = OpusEncoder::new(
        rate as i32,
        1,
        if voip {
            Application::Voip
        } else {
            Application::Audio
        },
    )
    .unwrap();
    rs.bitrate_bps = 32000;
    rs.complexity = 9;
    rs.set_dtx(true);
    let mut c = CEnc::new(
        rate as u32,
        CCh::Mono,
        if voip { CApp::Voip } else { CApp::Audio },
    )
    .unwrap();
    c.set_bitrate(CBitrate::Bits(32000)).unwrap();
    c.set_complexity(9).unwrap();
    c.set_dtx(true).unwrap();
    let (mut rs_dtx, mut c_dtx) = (Vec::new(), Vec::new());
    let mut packet = [0u8; 1500];
    for x in pcm.chunks_exact(frame) {
        rs_dtx.push(rs.encode(x, frame, &mut packet).unwrap() == 1);
        c_dtx.push(c.encode_float(x, &mut packet).unwrap() == 1);
    }
    (rs_dtx, c_dtx)
}

fn check(rate: usize, voip: bool, ms: usize) {
    let (rs, c) = toc_only(rate, voip, ms);
    let label = format!("{rate} Hz {} {ms} ms", if voip { "VoIP" } else { "Audio" });
    let frames_per_s = 1000 / ms;
    // libopus enters DTX on the noise floor, and in digital silence.
    let floor = 2 * frames_per_s..5 * frames_per_s;
    let count = |v: &[bool], r: core::ops::Range<usize>| v[r].iter().filter(|&&d| d).count();
    assert!(
        count(&c, floor.clone()) > floor.len() / 2,
        "{label}: libopus {} ToC-only frames on the noise floor",
        count(&c, floor.clone())
    );
    let differ: Vec<usize> = (0..rs.len()).filter(|&i| rs[i] != c[i]).collect();
    println!(
        "{label}: {} of {} frames agree; noise floor ToC-only opus-rs {}, libopus {}",
        rs.len() - differ.len(),
        rs.len(),
        count(&rs, floor.clone()),
        count(&c, floor.clone())
    );
    assert!(
        differ.len() * 100 <= rs.len(),
        "{label}: {} of {} frames differ from libopus (first {:?}); noise floor opus-rs {}, libopus {}",
        differ.len(),
        rs.len(),
        &differ[..differ.len().min(8)],
        count(&rs, floor.clone()),
        count(&c, floor)
    );
}

#[test]
fn audio_dtx_follows_libopus_on_a_noise_floor() {
    check(48000, false, 20);
    check(48000, false, 10);
    check(16000, false, 20);
}

#[test]
fn voip_dtx_follows_libopus_where_the_analysis_runs() {
    check(48000, true, 20);
    check(24000, true, 20);
}
