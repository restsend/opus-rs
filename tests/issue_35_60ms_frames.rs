//! Issue #35: the encoder rejected 40/60 ms frames.
//!
//! `frame_rate_from_params` required `sampling_rate % frame_size == 0`, so
//! 60 ms (16.67 frames/s) was rejected before encoding. libopus accepts 2.5–60
//! ms: SILK-only carries a 60 ms frame directly (3 internal SILK frames), while
//! CELT and Hybrid cap a frame at 20 ms and repacketize 20 ms sub-frames into a
//! code 1/2/3 packet.
//!
//! Every case is checked against libopus: opus-rs's packet must decode in
//! libopus with the full duration and track the input, and libopus's own
//! 40/60 ms packet must decode in opus-rs.

use opus::{
    Application as CApp, Channels as CCh, Decoder as CDec, Encoder as CEnc,
};
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

const MAX_FRAME: usize = 5760; // 120 ms @ 48 kHz, per channel

fn c_channels(ch: usize) -> CCh {
    if ch == 2 {
        CCh::Stereo
    } else {
        CCh::Mono
    }
}

/// Voiced-like harmonic series; the right channel is a phase-shifted copy.
fn voiced(sr: usize, ch: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * ch);
    let mut phase = 0.0f32;
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let f0 = 140.0 + 15.0 * (2.0 * PI * 0.7 * t).sin();
        phase += 2.0 * PI * f0 / sr as f32;
        let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin();
        let mut l = 0.0;
        let mut r = 0.0;
        for h in 1..=8 {
            let a = 0.25 / h as f32;
            l += a * (h as f32 * phase).sin();
            r += a * (h as f32 * phase + 0.3 * h as f32).sin();
        }
        out.push(l * env * 0.5);
        if ch == 2 {
            out.push(r * env * 0.4);
        }
    }
    out
}

fn mid(pcm: &[f32], ch: usize) -> Vec<f32> {
    pcm.chunks_exact(ch)
        .map(|p| p.iter().sum::<f32>() / ch as f32)
        .collect()
}

/// Worst per-window SNR (dB) of `test` against `reference` over consecutive
/// `win`-sample windows after `skip` samples, at the best lag in ±`max_lag`.
#[allow(clippy::needless_range_loop)]
fn worst_window_snr(
    reference: &[f32],
    test: &[f32],
    win: usize,
    skip: usize,
    max_lag: isize,
) -> f64 {
    let n = reference.len().min(test.len());
    let mut worst = f64::INFINITY;
    let mut start = skip;
    while start + win + max_lag as usize <= n {
        let mut best = f64::NEG_INFINITY;
        for lag in -max_lag..=max_lag {
            let (mut sig, mut err) = (0f64, 0f64);
            for i in start..start + win {
                let j = (i as isize + lag) as usize;
                let r = reference[i] as f64;
                let t = test[j] as f64;
                sig += r * r;
                err += (r - t) * (r - t);
            }
            if sig > 0.0 {
                best = best.max(10.0 * (sig / err.max(1e-20)).log10());
            }
        }
        worst = worst.min(best);
        start += win;
    }
    worst
}

/// Number of samples per channel an Opus packet of `ms` at `rate` must decode to.
fn expected(rate: usize, ms: usize) -> usize {
    rate * ms / 1000
}

/// opus-rs encodes `ms` frames; libopus decodes them. Returns the worst 20 ms
/// window SNR against the input (mid channel).
fn rust_enc_c_dec(
    rate: usize,
    ch: usize,
    app: Application,
    capp: CApp,
    br: i32,
    ms: usize,
    cbr: bool,
) -> f64 {
    let frame = expected(rate, ms);
    let packets = 1000 / ms;
    let input = voiced(rate, ch, frame * packets);
    let mut enc = OpusEncoder::new(rate as i32, ch, app).unwrap();
    enc.bitrate_bps = br;
    enc.use_cbr = cbr;
    let mut dec = CDec::new(rate as u32, c_channels(ch)).unwrap();
    let (mut ref_pcm, mut out_pcm) = (Vec::new(), Vec::new());
    let mut pkt = vec![0u8; 8000];
    let mut buf = vec![0f32; MAX_FRAME * ch];
    for p in 0..packets {
        let n = enc
            .encode(&input[p * frame * ch..(p + 1) * frame * ch], frame, &mut pkt)
            .unwrap_or_else(|e| panic!("{rate} {app:?} {ms}ms p{p}: opus-rs encode failed: {e}"));
        let got = dec
            .decode_float(&pkt[..n], &mut buf, false)
            .unwrap_or_else(|e| panic!("{rate} {app:?} {ms}ms p{p}: libopus rejected packet: {e}"));
        assert_eq!(
            got,
            frame,
            "{rate} {app:?} {ms}ms p{p}: libopus decoded {got}, want {frame}"
        );
        ref_pcm.extend_from_slice(&input[p * frame * ch..(p + 1) * frame * ch]);
        out_pcm.extend_from_slice(&buf[..got * ch]);
    }
    let _ = capp;
    worst_window_snr(&mid(&ref_pcm, ch), &mid(&out_pcm, ch), rate / 50, frame, (rate * 40 / 1000) as isize)
}

/// libopus encodes `ms` frames; opus-rs decodes them.
fn c_enc_rust_dec(
    rate: usize,
    ch: usize,
    capp: CApp,
    ms: usize,
) -> f64 {
    let frame = expected(rate, ms);
    let packets = 1000 / ms;
    let input = voiced(rate, ch, frame * packets);
    let mut enc = CEnc::new(rate as u32, c_channels(ch), capp).unwrap();
    let mut dec = OpusDecoder::new(rate as i32, ch).unwrap();
    let (mut ref_pcm, mut out_pcm) = (Vec::new(), Vec::new());
    let mut pkt = vec![0u8; 8000];
    let mut buf = vec![0f32; MAX_FRAME * ch];
    for p in 0..packets {
        let n = enc
            .encode_float(&input[p * frame * ch..(p + 1) * frame * ch], &mut pkt)
            .unwrap();
        let got = dec
            .decode(&pkt[..n], MAX_FRAME, &mut buf)
            .unwrap_or_else(|e| panic!("{rate} {ms}ms p{p}: opus-rs decode failed: {e}"));
        assert_eq!(
            got,
            frame,
            "{rate} {ms}ms p{p}: opus-rs decoded {got}, want {frame}"
        );
        ref_pcm.extend_from_slice(&input[p * frame * ch..(p + 1) * frame * ch]);
        out_pcm.extend_from_slice(&buf[..got * ch]);
    }
    worst_window_snr(&mid(&ref_pcm, ch), &mid(&out_pcm, ch), rate / 50, frame, (rate * 40 / 1000) as isize)
}

/// SILK is not waveform-matching; CELT is. Both clear this comfortably for
/// clean 40/60 ms packets; a dropped sub-frame or wrong duration does not.
const FLOOR_DB: f64 = 3.0;

fn check_all(
    rate: usize,
    ch: usize,
    app: Application,
    capp: CApp,
    br: i32,
    ms: usize,
    cbr_modes: &[bool],
) {
    for &cbr in cbr_modes {
        let snr = rust_enc_c_dec(rate, ch, app, capp, br, ms, cbr);
        assert!(
            snr > FLOOR_DB,
            "opus-rs->libopus {rate} {ch}ch {app:?} {br} cbr={cbr} {ms}ms: worst {snr:.1} dB"
        );
        let snr = c_enc_rust_dec(rate, ch, capp, ms);
        assert!(
            snr > FLOOR_DB,
            "libopus->opus-rs {rate} {ch}ch {capp:?} {ms}ms: worst {snr:.1} dB"
        );
    }
}

#[test]
fn silk_60ms_mono_and_stereo() {
    check_all(16000, 1, Application::Voip, CApp::Voip, 24000, 60, &[false, true]);
    check_all(16000, 2, Application::Voip, CApp::Voip, 32000, 60, &[false, true]);
    check_all(8000, 1, Application::Voip, CApp::Voip, 16000, 60, &[false, true]);
}

#[test]
fn silk_40ms_mono_and_stereo() {
    check_all(16000, 1, Application::Voip, CApp::Voip, 24000, 40, &[false, true]);
    check_all(16000, 2, Application::Voip, CApp::Voip, 32000, 40, &[false, true]);
}

#[test]
fn celt_60ms_mono_and_stereo() {
    check_all(48000, 1, Application::Audio, CApp::Audio, 64000, 60, &[false, true]);
    check_all(48000, 2, Application::Audio, CApp::Audio, 96000, 60, &[false, true]);
}

#[test]
fn celt_40ms_mono_and_stereo() {
    check_all(48000, 1, Application::Audio, CApp::Audio, 64000, 40, &[false, true]);
    check_all(48000, 2, Application::Audio, CApp::Audio, 96000, 40, &[false, true]);
}

#[test]
fn hybrid_60ms_mono_and_stereo() {
    // 48 kHz Voip at 32-48 kbps selects Hybrid (SILK + CELT above band 17).
    // VBR only: Hybrid *stereo* CBR is already broken at 20 ms on `main`
    // (opus-rs->libopus ~0 dB), a pre-existing bug independent of frame size
    // and out of scope for #35.
    check_all(48000, 1, Application::Voip, CApp::Voip, 32000, 60, &[false, true]);
    check_all(48000, 2, Application::Voip, CApp::Voip, 48000, 60, &[false]);
}

#[test]
fn issue_repro_exact_config() {
    // From the issue: OpusEncoder::new(16000, 2, Voip), 60 ms = 960 samples,
    // at a rate libopus codes in SILK (it codes the default 64 kb/s of this
    // input as CELT).
    let mut enc = OpusEncoder::new(16000, 2, Application::Voip).unwrap();
    enc.bitrate_bps = 32000;
    let pcm = vec![0.1f32; 960 * 2];
    let mut out = vec![0u8; 4000];
    let n = enc.encode(&pcm, 960, &mut out).expect("60 ms must encode");
    assert!(n > 1);
    // TOC config 11 = SILK wideband 60 ms.
    assert_eq!(out[0] >> 3, 11, "expected SILK WB 60 ms TOC, got {:#04x}", out[0]);

    let mut dec = OpusDecoder::new(16000, 2).unwrap();
    let mut decoded = vec![0f32; 960 * 2];
    let got = dec.decode(&out[..n], 960, &mut decoded).unwrap();
    assert_eq!(got, 960);
}
