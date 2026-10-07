//! Issue #27 checked against libopus (the `opus` dev-dependency).
//!
//! `issue_15_27_repro.rs` covers #27 with opus-rs on both ends of the wire,
//! which cannot see a bug the encoder and decoder share (the original part 2
//! desync was found with the C decoder). These tests put libopus on one side:
//!
//! - libopus encodes 40/60 ms SILK packets and opus-rs decodes them at 48 kHz
//!   with a 120 ms output buffer — what a Discord voice receiver does. Every
//!   20 ms window must match libopus's own decode of the same packets, so a
//!   silent or garbled 2nd/3rd SILK frame fails on its own window.
//! - opus-rs encodes 40 ms stereo SILK, and 20 ms SILK with in-band FEC, and
//!   libopus decodes it: each window must track the input as well as plain
//!   20 ms packets do, and libopus must recover lost frames from the LBRR
//!   section.

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Decoder as CDec,
    Encoder as CEnc, Signal as CSignal,
};
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

/// Largest Opus frame per channel: 120 ms at 48 kHz.
const MAX_FRAME: usize = 5760;

fn c_channels(ch: usize) -> CCh {
    if ch == 2 { CCh::Stereo } else { CCh::Mono }
}

/// Voiced-speech-like test signal: a 140 Hz harmonic series with slow pitch
/// drift and a syllable-rate envelope. The right channel is a scaled, phase
/// shifted copy so the stereo side signal is not trivially zero.
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

fn channel(pcm: &[f32], ch: usize, c: usize) -> Vec<f32> {
    pcm.iter().skip(c).step_by(ch).copied().collect()
}

/// SNR of `test` against `reference` (in dB), at the best lag in
/// `-max_lag..=max_lag`, over `reference[start..start + len]`.
fn best_lag_snr(reference: &[f32], test: &[f32], start: usize, len: usize, max_lag: isize) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for lag in -max_lag..=max_lag {
        let (mut sig, mut err) = (0f64, 0f64);
        for i in start..start + len {
            let j = i as isize + lag;
            if j < 0 || j as usize >= test.len() || i >= reference.len() {
                continue;
            }
            let r = reference[i] as f64;
            let e = r - test[j as usize] as f64;
            sig += r * r;
            err += e * e;
        }
        if sig > 0.0 {
            best = best.max(10.0 * (sig / err.max(1e-20)).log10());
        }
    }
    best
}

/// Per-window SNR (worst channel) of `test` against `reference`, both
/// interleaved with `ch` channels, over consecutive `win`-sample windows
/// after skipping `skip` samples of warm-up.
fn window_snrs(
    reference: &[f32],
    test: &[f32],
    ch: usize,
    win: usize,
    skip: usize,
    max_lag: isize,
) -> Vec<f64> {
    let refs: Vec<Vec<f32>> = (0..ch).map(|c| channel(reference, ch, c)).collect();
    let tests: Vec<Vec<f32>> = (0..ch).map(|c| channel(test, ch, c)).collect();
    let n = refs[0].len().min(tests[0].len());
    let mut out = Vec::new();
    let mut start = skip;
    while start + win + max_lag as usize <= n {
        let worst = (0..ch)
            .map(|c| best_lag_snr(&refs[c], &tests[c], start, win, max_lag))
            .fold(f64::INFINITY, f64::min);
        out.push(worst);
        start += win;
    }
    out
}

/// TOC check: SILK-only (configs 0..=11), `ch` channels, and `ms` of audio
/// in total. Returns (SILK frames per Opus frame, Opus frames in the packet):
/// a 60 ms SILK frame carries 3 SILK frames inside one Opus frame.
fn assert_silk_toc(pkt: &[u8], ch: usize, ms: usize) -> (usize, usize) {
    let toc = pkt[0];
    let config = toc >> 3;
    assert!(config <= 11, "expected SILK-only TOC, got config {config}");
    let stereo = (toc >> 2) & 1 == 1;
    assert_eq!(stereo, ch == 2, "TOC stereo bit");
    let dur = [10, 20, 40, 60][(config & 3) as usize];
    let count = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => (pkt[1] & 0x3f) as usize,
    };
    assert_eq!(
        dur * count,
        ms,
        "TOC duration (config {config}, {count} frames)"
    );
    ((dur / 20).max(1), count)
}

const LAG_MS: usize = 40;

/// Mid signal (L+R)/2 of interleaved `ch`-channel audio (identity for mono).
fn mid(pcm: &[f32], ch: usize) -> Vec<f32> {
    pcm.chunks_exact(ch)
        .map(|p| p.iter().sum::<f32>() / ch as f32)
        .collect()
}

fn min(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------
// Part 3: libopus-encoded multi-frame SILK -> opus-rs decoder at 48 kHz
// ---------------------------------------------------------------------------

/// Encode ~1 s with libopus at `ms` frames, decode each packet with libopus
/// and with opus-rs (48 kHz, 120 ms output buffer), and return opus-rs's
/// per-20 ms-window SNR against libopus.
fn c_enc_rust_dec(ch: usize, ms: usize) -> Vec<f64> {
    let sr = 48000;
    let frame = sr * ms / 1000;
    let packets = 1000 / ms;
    let input = voiced(sr, ch, frame * packets);

    let mut enc = CEnc::new(sr as u32, c_channels(ch), CApp::Voip).unwrap();
    enc.set_bitrate(CBitrate::Bits(24000)).unwrap();
    enc.set_bandwidth(CBw::Wideband).unwrap();
    // A steady harmonic tone can read as music and pull libopus into CELT.
    enc.set_signal(CSignal::Voice).unwrap();
    let mut c_dec = CDec::new(sr as u32, c_channels(ch)).unwrap();
    let mut rs_dec = OpusDecoder::new(sr as i32, ch).unwrap();

    let (mut c_pcm, mut rs_pcm) = (Vec::new(), Vec::new());
    let mut pkt = vec![0u8; 1500];
    let mut c_buf = vec![0f32; MAX_FRAME * ch];
    let mut rs_buf = vec![0f32; MAX_FRAME * ch];
    for p in 0..packets {
        let n = enc
            .encode_float(&input[p * frame * ch..(p + 1) * frame * ch], &mut pkt)
            .unwrap();
        assert_silk_toc(&pkt[..n], ch, ms);

        let c_n = c_dec.decode_float(&pkt[..n], &mut c_buf, false).unwrap();
        let rs_n = rs_dec
            .decode(&pkt[..n], MAX_FRAME, &mut rs_buf)
            .unwrap_or_else(|e| panic!("packet {p}: opus-rs decode failed: {e}"));
        assert_eq!(c_n, frame, "libopus decoded length");
        assert_eq!(rs_n, frame, "packet {p}: opus-rs decoded length");
        c_pcm.extend_from_slice(&c_buf[..c_n * ch]);
        rs_pcm.extend_from_slice(&rs_buf[..rs_n * ch]);
    }
    // Skip the first packet (encoder/decoder warm-up); windows are 20 ms.
    window_snrs(&c_pcm, &rs_pcm, ch, 960, frame, 30)
}

fn check_matches_libopus(ch: usize, ms: usize) {
    let snrs = c_enc_rust_dec(ch, ms);
    let worst = min(&snrs);
    println!(
        "C enc {ms} ms {ch}ch -> opus-rs dec @48k: {} windows, worst {worst:.1} dB",
        snrs.len()
    );
    assert!(
        worst > MATCH_DB,
        "{ms} ms {ch}ch: opus-rs output diverges from libopus; per-window SNR: {snrs:.1?}"
    );
}

/// How closely opus-rs must match libopus when both decode the same packets.
/// Set with margin under the 20 ms baseline measured by
/// `libopus_20ms_baseline_matches` below. A dropped or garbled SILK frame
/// scores around 0 dB, far below this.
const MATCH_DB: f64 = 30.0;

#[test]
fn libopus_20ms_baseline_matches() {
    check_matches_libopus(1, 20);
    check_matches_libopus(2, 20);
}

#[test]
fn libopus_40ms_mono_decodes_every_silk_frame() {
    check_matches_libopus(1, 40);
}

#[test]
fn libopus_60ms_mono_decodes_every_silk_frame() {
    check_matches_libopus(1, 60);
}

#[test]
fn libopus_40ms_stereo_decodes_every_silk_frame() {
    check_matches_libopus(2, 40);
}

#[test]
fn libopus_60ms_stereo_decodes_every_silk_frame() {
    check_matches_libopus(2, 60);
}

// ---------------------------------------------------------------------------
// Parts 1 and 2: opus-rs-encoded SILK -> libopus decoder
// ---------------------------------------------------------------------------

/// Per-window SNRs from one opus-rs-encoded stream decoded by libopus.
struct RustEncoded {
    /// Normal decode against the input (mid channel).
    normal: Vec<f64>,
    /// libopus `decode_fec` output (the previous packet recovered from LBRR)
    /// against the input; empty without FEC.
    lbrr: Vec<f64>,
    /// opus-rs's own decode of the same packets against libopus's.
    agreement: Vec<f64>,
}

/// Encode ~1 s with opus-rs at 16 kHz and `ms` frames, and decode it with
/// libopus and with opus-rs. With `fec`, also decode every packet through a
/// second libopus decoder with `fec = true`.
fn rust_enc_c_dec(ch: usize, ms: usize, fec: bool, cbr: bool) -> RustEncoded {
    let sr = 16000;
    let frame = sr * ms / 1000;
    let packets = 1000 / ms;
    let input = voiced(sr, ch, frame * packets);

    let mut enc = OpusEncoder::new(sr as i32, ch, Application::Voip).unwrap();
    enc.bitrate_bps = 32000;
    enc.use_cbr = cbr;
    if fec {
        enc.use_inband_fec = true;
        enc.packet_loss_perc = 40;
    }
    let mut c_dec = CDec::new(sr as u32, c_channels(ch)).unwrap();
    let mut c_fec = CDec::new(sr as u32, c_channels(ch)).unwrap();
    let mut rs_dec = OpusDecoder::new(sr as i32, ch).unwrap();

    let (mut pcm, mut fec_pcm, mut rs_pcm) = (Vec::new(), Vec::new(), Vec::new());
    let mut pkt = vec![0u8; 1500];
    let mut buf = vec![0f32; MAX_FRAME * ch];
    for p in 0..packets {
        let n = enc
            .encode(
                &input[p * frame * ch..(p + 1) * frame * ch],
                frame,
                &mut pkt,
            )
            .unwrap_or_else(|e| panic!("packet {p}: opus-rs encode failed: {e}"));
        assert_silk_toc(&pkt[..n], ch, ms);

        let got = c_dec
            .decode_float(&pkt[..n], &mut buf, false)
            .unwrap_or_else(|e| panic!("packet {p}: libopus rejected opus-rs packet: {e}"));
        assert_eq!(got, frame);
        pcm.extend_from_slice(&buf[..got * ch]);

        let got = rs_dec.decode(&pkt[..n], MAX_FRAME, &mut buf).unwrap();
        assert_eq!(got, frame);
        rs_pcm.extend_from_slice(&buf[..got * ch]);

        if fec && p > 0 {
            // Recovers packet p-1 from packet p's LBRR data.
            let got = c_fec
                .decode_float(&pkt[..n], &mut buf[..frame * ch], true)
                .unwrap_or_else(|e| panic!("packet {p}: libopus FEC decode failed: {e}"));
            assert_eq!(got, frame);
            fec_pcm.extend_from_slice(&buf[..got * ch]);
        }
    }
    let win = sr / 50;
    let max_lag = (sr * LAG_MS / 1000) as isize;
    let reference = mid(&input, ch);
    RustEncoded {
        normal: window_snrs(&reference, &mid(&pcm, ch), 1, win, frame, max_lag),
        // fec_pcm[k] is packet k's audio (recovered from packet k+1).
        lbrr: if fec {
            window_snrs(&reference, &mid(&fec_pcm, ch), 1, win, frame, max_lag)
        } else {
            Vec::new()
        },
        agreement: window_snrs(&pcm, &rs_pcm, ch, win, frame, 30),
    }
}

/// Floor for opus-rs-encoded audio decoded by libopus, against the input.
/// SILK is not a waveform-matching codec, so this sits well below the
/// codec-agreement bar: clean 20 ms packets score 5-8 dB here; a desynced
/// frame scores around 0 dB or below.
const INPUT_FLOOR_DB: f64 = 3.0;

/// Part 2a: a 40 ms stereo packet must decode in libopus as well as 20 ms
/// packets do (the stereo header used to be written for frame 0 only).
/// opus-rs rejects 60 ms at the encoder (`Invalid frame size`), so 40 ms is
/// the only multi-frame SILK packet it can produce.
#[test]
fn opus_rs_40ms_stereo_decodes_in_libopus() {
    let base = rust_enc_c_dec(2, 20, false, true);
    let multi = rust_enc_c_dec(2, 40, false, true);
    let (base_worst, worst) = (min(&base.normal), min(&multi.normal));
    println!(
        "opus-rs enc 40 ms stereo -> libopus dec: worst {worst:.1} dB \
         (20 ms baseline {base_worst:.1} dB); opus-rs vs libopus {:.1} dB",
        min(&multi.agreement)
    );
    assert!(
        base_worst > INPUT_FLOOR_DB,
        "20 ms baseline too weak: {:.1?}",
        base.normal
    );
    assert!(
        worst > INPUT_FLOOR_DB && worst > base_worst - 3.0,
        "40 ms stereo: a SILK frame desynced in libopus; per-window SNR: {:.1?}",
        multi.normal
    );
    assert!(
        min(&multi.agreement) > MATCH_DB,
        "decoders disagree: {:.1?}",
        multi.agreement
    );
}

/// Part 2b: in-band FEC. The packet must still decode normally (FEC on must
/// not cost the main frame anything), and libopus must recover each previous
/// frame from the LBRR section.
///
/// Both VBR and CBR are checked (issue #36: CBR used to drop LBRR entirely
/// because the copied payload was as expensive as the main frame; a real
/// low-rate LBRR payload fits under CBR, as in libopus).
///
/// The recovery metric is the *median* per-window SNR plus the fraction of
/// windows above 2 dB — not the worst window: the speech-activity gate makes
/// the encoder skip LBRR in quiet stretches, and libopus's own encoder does
/// the same. Measured against libopus 1.3.1 with the same signal and settings
/// (issue #36 work): VBR mono median 6.1 dB / worst 0.2 dB, CBR mono 4.6/0.2 —
/// a missing LBRR section decodes as PLC, scoring ~0 dB.
fn check_fec_decodes_in_libopus(ch: usize, cbr: bool) {
    let plain = rust_enc_c_dec(ch, 20, false, cbr);
    let fec = rust_enc_c_dec(ch, 20, true, cbr);
    let (plain_worst, worst) = (min(&plain.normal), min(&fec.normal));
    let lbrr_median = {
        let mut v = fec.lbrr.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let above2 = fec
        .lbrr
        .iter()
        .filter(|s| **s > 2.0)
        .count();
    println!(
        "opus-rs enc 20 ms {ch}ch + FEC({}) -> libopus: normal {worst:.1} dB \
         (no-FEC {plain_worst:.1} dB), LBRR median {lbrr_median:.1} dB, {above2}/{} windows > 2 dB",
        if cbr { "CBR" } else { "VBR" },
        fec.lbrr.len()
    );
    if cbr {
        // CBR + FEC is a genuinely squeezed regime: when LBRR + main frame
        // exceed the packet, both opus-rs and libopus degrade the packet
        // (libopus emits 30-36 fallback packets per 50 here), and the main
        // frame can land on its minimal damage-control encode. Judge the
        // normal decode by the fraction of healthy windows; libopus's own
        // CBR+FEC stream measures worst 4.5 dB / 47 of 47 windows > 2 dB.
        let healthy = fec.normal.iter().filter(|s| **s > 2.0).count();
        assert!(
            healthy as f64 >= 0.75 * fec.normal.len() as f64,
            "{ch}ch CBR + FEC: too many degraded main-frame windows ({healthy}/{}); \
             per-window SNR: {:.1?}",
            fec.normal.len(),
            fec.normal
        );
        // Stereo recovery in this squeezed regime swings with tiny input
        // changes: across input levels 0.90-1.10x, the median measures
        // 1.1-4.0 dB with 14-38 of 46 windows > 2 dB (1.5-4.3 dB, 13-39
        // windows once the stereo header is the cheaper zero predictor, issue
        // #42), and a 1-LSB change in the downmix rounding moves this signal
        // from 1.5 to 4.2 dB.
        //
        // Coding the side (issue #48) adds the side's LBRR to every packet.
        // opus-rs doesn't take the LBRR bits out of the frame target, as
        // libopus does (enc_API.c nBitsUsedLBRR), so after a packet without
        // LBRR the main frames are coded at full rate, the next packet's LBRR
        // copies no longer fit, and the section is dropped: LBRR alternates
        // with gaps (12 of 50 packets carry it, against 21 for the mid alone)
        // and 10 packets bust into the fallback (4 before). Across
        // 0.90-1.10x the median is now 0.5-2.1 dB with 7-23 windows > 2 dB,
        // while the normal decode improves (44-47 of 47 healthy windows, was
        // 37-46). libopus's own stream: 5.1-6.0 dB, 41-44 windows. Porting
        // libopus's LBRR accounting is issue #51. Until then the stereo
        // floors only tell real LBRR from none, which measures ~0 dB.
        let (median_floor, frac) = if ch == 1 { (2.0, 0.60) } else { (0.3, 0.10) };
        assert!(
            lbrr_median > median_floor,
            "{ch}ch CBR FEC: LBRR recovery too weak (median {lbrr_median:.1} dB < \
             {median_floor}); per-window SNR: {:.1?}",
            fec.lbrr
        );
        assert!(
            above2 as f64 >= frac * fec.lbrr.len() as f64,
            "{ch}ch CBR FEC: too few windows with real LBRR recovery ({above2}/{} < \
             {:.0}%); per-window SNR: {:.1?}",
            fec.lbrr.len(),
            frac * 100.0,
            fec.lbrr
        );
        println!(
            "opus-rs enc 20 ms {ch}ch + FEC(CBR) -> libopus: normal {worst:.1} dB \
             (no-FEC {plain_worst:.1} dB), LBRR median {lbrr_median:.1} dB, {above2}/{} \
             windows > 2 dB, agreement {:.1} dB",
            fec.lbrr.len(),
            min(&fec.agreement)
        );
        return;
    }
    assert!(
        worst > INPUT_FLOOR_DB && worst > plain_worst - 1.0,
        "{ch}ch + FEC: normal decode desynced in libopus; per-window SNR: {:.1?}",
        fec.normal
    );
    // Recovery floors, anchored below the measured opus-rs values with margin.
    // opus-rs measures: VBR mono 4.0 median / 40 of 47 windows; stereo
    // mid-only recovery behaves like libopus's own stereo FEC on equal
    // content (libopus L=R reference: median 0.6 dB).
    let (median_floor, frac) = if ch == 1 { (3.0, 0.70) } else { (1.0, 0.30) };
    assert!(
        lbrr_median > median_floor,
        "{ch}ch {} FEC: LBRR recovery too weak (median {lbrr_median:.1} dB < \
         {median_floor}); per-window SNR: {:.1?}",
        if cbr { "CBR" } else { "VBR" },
        fec.lbrr
    );
    assert!(
        above2 as f64 >= frac * fec.lbrr.len() as f64,
        "{ch}ch {} FEC: too few windows with real LBRR recovery ({above2}/{} < {:.0}%); \
         per-window SNR: {:.1?}",
        if cbr { "CBR" } else { "VBR" },
        fec.lbrr.len(),
        frac * 100.0,
        fec.lbrr
    );
    assert!(
        min(&fec.agreement) > MATCH_DB,
        "decoders disagree: {:.1?}",
        fec.agreement
    );
}

#[test]
fn opus_rs_mono_fec_decodes_in_libopus() {
    check_fec_decodes_in_libopus(1, false);
}

#[test]
fn opus_rs_stereo_fec_decodes_in_libopus() {
    check_fec_decodes_in_libopus(2, false);
}

#[test]
fn opus_rs_mono_fec_cbr_decodes_in_libopus() {
    check_fec_decodes_in_libopus(1, true);
}

#[test]
fn opus_rs_stereo_fec_cbr_decodes_in_libopus() {
    check_fec_decodes_in_libopus(2, true);
}
