//! Concealment must not write files or print, whatever the environment says.
//!
//! The CELT concealment's pitch search kept a debugging trace: with
//! `OPUS_TRACE_PLC` set, every periodic concealment wrote its downsampled
//! history to `rs_lp_<n>.f32` in that directory and printed the lag to stderr.
//! The test sets the variable (it is the only test in this file, so no other
//! test runs with it), conceals CELT frames, and checks that no file appears
//! and that the audio is that of the same decode without the variable.

use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;
use std::fs;

const RATE: usize = 48_000;
const FRAME: usize = 960;
/// Single losses after good packets (periodic concealment) and a double one.
const LOST: [usize; 4] = [10, 20, 21, 30];

/// CELT packets of a voiced signal (harmonics of a gliding 150 Hz pitch).
fn packets() -> Vec<Vec<u8>> {
    let mut enc = OpusEncoder::new(RATE as i32, 1, Application::RestrictedLowDelay).unwrap();
    enc.bitrate_bps = 64_000;
    let mut phase = 0.0f32;
    let pcm: Vec<f32> = (0..40 * FRAME)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            phase += 2.0 * PI * (150.0 + 20.0 * (2.0 * PI * 0.9 * t).sin()) / RATE as f32;
            (1..=12)
                .map(|h| 0.2 / h as f32 * (h as f32 * phase).sin())
                .sum()
        })
        .collect();
    let mut out = vec![0u8; 1500];
    pcm.as_chunks::<FRAME>()
        .0
        .iter()
        .map(|frame| {
            let n = enc.encode(frame, FRAME, &mut out).unwrap();
            out[..n].to_vec()
        })
        .collect()
}

/// opus-rs's decode of `packets` with `LOST` concealed.
fn decode(packets: &[Vec<u8>]) -> Vec<f32> {
    let mut dec = OpusDecoder::new(RATE as i32, 1).unwrap();
    let mut buf = vec![0f32; FRAME];
    let mut pcm = Vec::new();
    for (k, packet) in packets.iter().enumerate() {
        let data: &[u8] = if LOST.contains(&k) { &[] } else { packet };
        let n = dec.decode(data, FRAME, &mut buf).unwrap();
        pcm.extend_from_slice(&buf[..n]);
    }
    pcm
}

#[test]
fn concealment_ignores_opus_trace_plc() {
    let packets = packets();
    assert!(
        packets.iter().all(|p| p[0] & 0x80 != 0),
        "the stream is CELT-only"
    );
    let plain = decode(&packets);

    let dir = std::env::temp_dir().join(format!("opus-rs-plc-trace-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    // SAFETY: this is the only test in this binary, and nothing else in the
    // process reads or writes the environment while it runs.
    unsafe { std::env::set_var("OPUS_TRACE_PLC", &dir) };
    let traced = decode(&packets);
    // SAFETY: as above.
    unsafe { std::env::remove_var("OPUS_TRACE_PLC") };

    let files: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    for f in &files {
        let _ = fs::remove_file(f);
    }
    let _ = fs::remove_dir(&dir);
    assert!(
        files.is_empty(),
        "concealment wrote {} file(s) into OPUS_TRACE_PLC, e.g. {:?}",
        files.len(),
        files.first()
    );
    assert!(
        traced == plain,
        "the decode with OPUS_TRACE_PLC set differs from the one without"
    );
}
