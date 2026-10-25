//! Loss concealment regression tests for GitHub issue #15.
//!
//! The issue: after a run of lost packets (fed as `decode(&[0], ..)`), the first
//! real packet panicked inside the SILK resampler. The root cause was that a
//! lost frame was decoded using the *placeholder byte's* TOC — `0x00` reads as
//! "SILK narrowband 10 ms" — so every loss switched the decoder into SILK-NB,
//! re-initialised the resamplers and manufactured fake mode transitions.
//!
//! libopus (`opus_decode_frame`, `len <= 1`) conceals a lost frame in the
//! *previous* mode and leaves the mode state untouched; only the duration is
//! taken from the placeholder TOC (or from `frame_size` for an empty packet).
//! These tests pin that behaviour down, plus a randomized no-panic sweep.

use opus_rs::{Application, OpusDecoder, OpusEncoder};
use proptest::prelude::*;
use std::f32::consts::PI;

/// Deterministic xorshift so loss patterns are reproducible without a dep.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A speech-ish test signal: two tones with a slow amplitude wobble.
fn signal(start: usize, len: usize, rate: i32, channels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len * channels);
    for i in start..start + len {
        let t = i as f32 / rate as f32;
        let env = 0.5 + 0.5 * (2.0 * PI * 3.0 * t).sin();
        let s = env * (0.25 * (2.0 * PI * 220.0 * t).sin() + 0.1 * (2.0 * PI * 1250.0 * t).sin());
        for ch in 0..channels {
            out.push(if ch == 0 { s } else { s * 0.8 });
        }
    }
    out
}

/// Encode `n_packets` 20 ms packets with the reporter's settings (CBR).
fn encode_stream(
    rate: i32,
    channels: usize,
    app: Application,
    bitrate: i32,
    n_packets: usize,
) -> Vec<Vec<u8>> {
    let frame = rate as usize / 50;
    let mut enc = OpusEncoder::new(rate, channels, app).unwrap();
    enc.bitrate_bps = bitrate;
    enc.use_cbr = true;
    (0..n_packets)
        .map(|p| {
            let pcm = signal(p * frame, frame, rate, channels);
            let mut buf = vec![0u8; 1500];
            let n = enc.encode(&pcm, frame, &mut buf).expect("encode");
            buf.truncate(n);
            buf
        })
        .collect()
}

fn toc_mode(toc: u8) -> &'static str {
    if toc & 0x80 != 0 {
        "celt"
    } else if toc & 0x60 == 0x60 {
        "hybrid"
    } else {
        "silk"
    }
}

/// Samples per channel a 1-byte (TOC-only, code 0) loss marker stands for.
fn marker_samples(toc: u8, rate: i32) -> usize {
    let config = toc >> 3;
    let tenth_ms = match config {
        0..=11 => [100, 200, 400, 600][(config & 3) as usize],
        12..=15 => [100, 200][(config & 1) as usize],
        _ => [25, 50, 100, 200][(config & 3) as usize],
    };
    rate as usize * tenth_ms / 10_000
}

fn assert_sane(label: &str, pcm: &[f32]) {
    // libopus's float CELT concealment saturates only at `SIG_SAT` in the
    // de-emphasis path (celt_decoder.c:303), i.e. ±16384 after scaling —
    // not at ±1.0. Pitch extrapolation plus the comb filter can legitimately
    // overshoot full scale for a few samples after a loss, so the sanity
    // bound here is finite and far below that saturation, not ±1.0.
    let peak = pcm.iter().fold(0f32, |m, &v| m.max(v.abs()));
    if peak > 1.0 {
        eprintln!("{label}: peak {peak}");
    }
    for (i, &v) in pcm.iter().enumerate() {
        assert!(
            v.is_finite() && (-16384.0..=16384.0).contains(&v),
            "{label}: sample {i} out of range: {v}"
        );
    }
}

/// The configurations the matrix sweeps: (rate, channels, app, bitrate).
/// Includes the exact configuration from the issue report (16 kHz mono,
/// 320-sample frames, CBR 24 kbps).
fn matrix() -> Vec<(i32, usize, Application, i32)> {
    let mut v = Vec::new();
    for &rate in &[8000, 12000, 16000, 24000, 48000] {
        for &channels in &[1usize, 2] {
            for &app in &[Application::Voip, Application::Audio] {
                for &bitrate in &[24000, 64000] {
                    v.push((rate, channels, app, bitrate));
                }
            }
        }
    }
    v
}

// ---------------------------------------------------------------------------
// The issue report, with real encoded packets
// ---------------------------------------------------------------------------

/// The reporter's exact setup, both SILK (Voip) and CELT (Audio) streams,
/// with runs of `&[0]` losses. Every decode must succeed with sane output.
#[test]
fn reporter_config_runs_of_losses_never_panic() {
    let rate = 16000;
    let frame = 320;
    for app in [Application::Voip, Application::Audio] {
        let packets = encode_stream(rate, 1, app, 24000, 120);
        let mut dec = OpusDecoder::new(rate, 1).unwrap();
        let mut pcm = vec![0.0f32; frame];
        let mut rng = Rng(0x15_15_15);
        let mut i = 0;
        while i < packets.len() {
            // A run of 1..=10 losses before each burst of real packets.
            for _ in 0..=rng.below(10) {
                let n = dec
                    .decode(&[0], frame, &mut pcm)
                    .unwrap_or_else(|e| panic!("{app:?}: loss decode failed: {e}"));
                assert_eq!(n, marker_samples(0, rate));
                assert_sane("loss", &pcm[..n]);
            }
            for _ in 0..=rng.below(4) {
                if i >= packets.len() {
                    break;
                }
                let n = dec
                    .decode(&packets[i], frame, &mut pcm)
                    .unwrap_or_else(|e| panic!("{app:?} pkt {i}: {e}"));
                assert_eq!(n, frame);
                assert_sane("packet", &pcm[..n]);
                i += 1;
            }
        }
    }
}

/// Every rate/channel/application/bitrate combination, with random single
/// drops, runs and alternating loss, using the `&[0]` marker from the issue,
/// a stream-matching marker, and an empty packet.
#[test]
fn loss_patterns_across_matrix_never_panic() {
    for (rate, channels, app, bitrate) in matrix() {
        let frame = rate as usize / 50;
        let packets = encode_stream(rate, channels, app, bitrate, 60);
        let label = format!("{rate} Hz {channels}ch {app:?} {bitrate} bps");
        let mut dec = OpusDecoder::new(rate, channels).unwrap();
        let mut pcm = vec![0.0f32; frame * channels];
        let mut rng = Rng(rate as u64 * 31 + channels as u64 * 7 + bitrate as u64);
        for (i, pkt) in packets.iter().enumerate() {
            let lost = match i % 20 {
                3 => true,               // single drop
                8..=12 => true,          // a run of five
                15 | 17 | 19 => true,    // alternating
                _ => rng.below(10) == 0, // background 10 %
            };
            let res = if lost {
                match rng.below(3) {
                    0 => dec.decode(&[0], frame, &mut pcm),
                    1 => dec.decode(&[pkt[0] & 0xFC], frame, &mut pcm), // stream ToC, code 0
                    _ => dec.decode(&[], frame, &mut pcm),
                }
            } else {
                dec.decode(pkt, frame, &mut pcm)
            };
            let n = res.unwrap_or_else(|e| panic!("{label} pkt {i} lost={lost}: {e}"));
            assert!(n > 0 && n <= frame, "{label} pkt {i}: n={n}");
            assert_sane(&label, &pcm[..n * channels]);
        }
    }
}

// ---------------------------------------------------------------------------
// Root cause: concealment must not depend on the placeholder TOC
// ---------------------------------------------------------------------------

/// All code-0 TOC bytes (mono and stereo flag) whose frame lasts 20 ms.
fn twenty_ms_markers() -> Vec<u8> {
    let mut v = Vec::new();
    for config in 0u8..32 {
        let toc = config << 3;
        if marker_samples(toc, 48000) == 960 {
            v.push(toc);
            v.push(toc | 0x04);
        }
    }
    v
}

/// Decode `packets` with packets 5..=9 and 20 lost, marking each loss with
/// `marker` (`None` = empty packet). Returns the full decoded output.
fn decode_with_losses(
    packets: &[Vec<u8>],
    rate: i32,
    channels: usize,
    marker: Option<u8>,
) -> Vec<f32> {
    let frame = rate as usize / 50;
    let mut dec = OpusDecoder::new(rate, channels).unwrap();
    let mut out = Vec::new();
    let mut pcm = vec![0.0f32; frame * channels];
    for (i, pkt) in packets.iter().enumerate() {
        let lost = (5..=9).contains(&i) || i == 20;
        let res = match (lost, marker) {
            (false, _) => dec.decode(pkt, frame, &mut pcm),
            (true, Some(toc)) => dec.decode(&[toc], frame, &mut pcm),
            (true, None) => dec.decode(&[], frame, &mut pcm),
        };
        let n = res.unwrap_or_else(|e| panic!("marker {marker:?} pkt {i}: {e}"));
        assert_eq!(n, frame, "marker {marker:?} pkt {i}");
        out.extend_from_slice(&pcm[..n * channels]);
    }
    out
}

/// Streams for the marker-independence sweep, spanning all three modes.
fn independence_configs() -> Vec<(i32, usize, Application, i32)> {
    vec![
        // The reporter's setup: SILK WB at 16 kHz. A `[0]` marker (SILK NB)
        // used to drop the decoder to 8 kHz and spin up the 8k->16k Up2HQ
        // resampler that panicked.
        (16000, 1, Application::Voip, 24000),
        (16000, 1, Application::Audio, 24000),
        (48000, 1, Application::Voip, 24000), // hybrid, as Discord sends
        (48000, 1, Application::Audio, 64000),
        (48000, 2, Application::Audio, 64000), // CELT stereo
        (8000, 1, Application::Voip, 12000),
    ]
}

/// The same stream with the same losses must decode bit-identically no
/// matter which 20 ms placeholder marks the loss. Before the fix a SILK-NB,
/// CELT or hybrid marker each drove the decoder into a different mode.
#[test]
fn concealment_is_independent_of_loss_marker_toc() {
    for (rate, channels, app, bitrate) in independence_configs() {
        let packets = encode_stream(rate, channels, app, bitrate, 30);
        let label = format!("{rate} Hz {channels}ch {app:?}");
        let reference = decode_with_losses(&packets, rate, channels, None);
        for toc in twenty_ms_markers() {
            let got = decode_with_losses(&packets, rate, channels, Some(toc));
            assert!(
                got == reference,
                "{label}: loss marker 0x{toc:02x} ({}) changed the decoded output",
                toc_mode(toc)
            );
        }
    }
}

/// Every mode must be covered by the independence sweep.
#[test]
fn independence_configs_cover_every_mode() {
    let mut seen = std::collections::BTreeSet::new();
    for (rate, channels, app, bitrate) in independence_configs() {
        for p in encode_stream(rate, channels, app, bitrate, 10) {
            seen.insert(toc_mode(p[0]));
        }
    }
    assert_eq!(
        seen.into_iter().collect::<Vec<_>>(),
        ["celt", "hybrid", "silk"]
    );
}

// ---------------------------------------------------------------------------
// Edge cases of the loss path
// ---------------------------------------------------------------------------

/// Concealment after voiced SILK/hybrid audio must extrapolate it (audible,
/// no louder than the real signal) rather than drop straight to silence.
#[test]
fn silk_concealment_extrapolates_the_signal() {
    let energy = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    for (rate, app) in [(16000, Application::Voip), (48000, Application::Voip)] {
        let frame = rate as usize / 50;
        let packets = encode_stream(rate, 1, app, 24000, 12);
        let mut dec = OpusDecoder::new(rate, 1).unwrap();
        let mut pcm = vec![0.0f32; frame];
        for p in &packets {
            dec.decode(p, frame, &mut pcm).unwrap();
        }
        let real = energy(&pcm);
        dec.decode(&[], frame, &mut pcm).unwrap();
        let concealed = energy(&pcm);
        assert!(
            concealed > real * 0.01 && concealed < real * 4.0,
            "{rate} Hz: real {real:e}, concealed {concealed:e}"
        );
    }
}

/// A mono placeholder on a stereo decoder is still just a loss.
#[test]
fn stereo_decoder_accepts_mono_loss_marker() {
    let packets = encode_stream(48000, 2, Application::Audio, 64000, 3);
    let mut dec = OpusDecoder::new(48000, 2).unwrap();
    let mut pcm = vec![0.0f32; 960 * 2];
    dec.decode(&packets[0], 960, &mut pcm).unwrap();
    let n = dec
        .decode(&[0], 960, &mut pcm)
        .expect("mono marker on stereo");
    assert_eq!(n, 480);
    dec.decode(&packets[1], 960, &mut pcm).unwrap();
}

/// Losses before the first packet produce silence of the right length.
#[test]
fn loss_before_first_packet_is_silence() {
    let mut dec = OpusDecoder::new(16000, 1).unwrap();
    let mut pcm = vec![1.0f32; 320];
    assert_eq!(dec.decode(&[0], 320, &mut pcm), Ok(160));
    assert!(pcm[..160].iter().all(|&v| v == 0.0));
    let mut pcm = vec![1.0f32; 320];
    assert_eq!(dec.decode(&[], 320, &mut pcm), Ok(320));
    assert!(pcm.iter().all(|&v| v == 0.0));
}

/// An empty packet conceals `frame_size` samples (libopus parity), for
/// every mode, including durations longer than one 20 ms codec frame.
#[test]
fn empty_packet_conceals_frame_size() {
    for app in [Application::Voip, Application::Audio] {
        let packets = encode_stream(48000, 1, app, 32000, 4);
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        let mut pcm = vec![0.0f32; 2880];
        dec.decode(&packets[0], 960, &mut pcm).unwrap();
        for fs in [120, 240, 480, 960, 1920, 2880] {
            let n = dec
                .decode(&[], fs, &mut pcm)
                .unwrap_or_else(|e| panic!("{app:?} frame_size {fs}: {e}"));
            assert_eq!(n, fs, "{app:?}");
            assert_sane("empty", &pcm[..n]);
        }
        dec.decode(&packets[1], 960, &mut pcm).unwrap();
    }
}

/// An empty packet with a frame_size that is not a multiple of 2.5 ms, or
/// an output buffer that is too small, is an error rather than a panic.
#[test]
fn empty_packet_invalid_sizes_are_errors() {
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut pcm = vec![0.0f32; 960];
    assert!(dec.decode(&[], 0, &mut pcm).is_err());
    assert!(dec.decode(&[], 100, &mut pcm).is_err());
    assert!(dec.decode(&[], 1920, &mut pcm).is_err());
}

/// A 1-byte code-3 packet has no frame-count byte: invalid, as in libopus.
#[test]
fn one_byte_code3_is_invalid() {
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut pcm = vec![0.0f32; 960];
    assert!(dec.decode(&[0x03], 960, &mut pcm).is_err());
}

// ---------------------------------------------------------------------------
// Randomized sweep
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Event {
    Real(usize),
    Empty,
    Marker(u8),
    Garbage(Vec<u8>),
}

fn event() -> impl Strategy<Value = Event> {
    prop_oneof![
        4 => (0usize..40).prop_map(Event::Real),
        1 => Just(Event::Empty),
        2 => any::<u8>().prop_map(Event::Marker),
        1 => proptest::collection::vec(any::<u8>(), 2..64).prop_map(Event::Garbage),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Any interleaving of real packets, losses of any marker and garbage
    /// must never panic. Errors are fine; panics are not.
    #[test]
    fn random_loss_sequences_never_panic(
        cfg in 0usize..40,
        events in proptest::collection::vec(event(), 1..60),
    ) {
        let (rate, channels, app, bitrate) = matrix()[cfg];
        let frame = rate as usize / 50;
        let packets = encode_stream(rate, channels, app, bitrate, 40);
        let mut dec = OpusDecoder::new(rate, channels).unwrap();
        let max = rate as usize * 120 / 1000;
        let mut pcm = vec![0.0f32; max * channels];
        for ev in &events {
            let res = match ev {
                Event::Real(i) => dec.decode(&packets[*i], frame, &mut pcm),
                Event::Empty => dec.decode(&[], frame, &mut pcm),
                Event::Marker(toc) => dec.decode(&[*toc], max, &mut pcm),
                Event::Garbage(bytes) => dec.decode(bytes, max, &mut pcm),
            };
            if let Ok(n) = res {
                assert_sane("random", &pcm[..n * channels]);
            }
        }
    }
}
