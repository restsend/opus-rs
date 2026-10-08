//! Issue #48: stereo SILK and Hybrid code mid/side like libopus's
//! `silk_stereo_LR_to_MS` and a second SILK channel for the side, checked per
//! channel against libopus (the `opus` dev-dependency).
//!
//! Each case encodes one stereo signal with opus-rs and with libopus at the
//! same settings, decodes both streams with libopus, and compares what comes
//! out against the input, channel by channel. Assertions are relative to
//! libopus's own encoder. The opus-rs decoder must agree with libopus on
//! opus-rs's packets.
//!
//! Coding the mid alone gives both channels the same signal. The cases where
//! that loses most: content that isn't amplitude-panned (two independent
//! voices) and anti-phase content, whose mid is silent.

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Decoder as CDec,
    Encoder as CEnc, Signal as CSignal,
};
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

/// Largest Opus frame per channel: 120 ms at 48 kHz.
const MAX_FRAME: usize = 5760;
const CH: usize = 2;
const SECONDS: usize = 2;
/// Best-lag search range for the input comparisons.
const LAG_MS: usize = 40;

/// A harmonic series on `f0` Hz with slow pitch drift and a syllable-rate
/// envelope; `phase_step` offsets harmonic `h` by `h * phase_step`.
fn harmonic(sr: usize, n: usize, f0: f32, syllable_hz: f32, phase_step: f32) -> Vec<f32> {
    let mut phase = 0.0f32;
    (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let f = f0 + 0.1 * f0 * (2.0 * PI * 0.7 * t).sin();
            phase += 2.0 * PI * f / sr as f32;
            let env = 0.6 + 0.4 * (2.0 * PI * syllable_hz * t).sin();
            let s: f32 = (1..=8)
                .map(|h| 0.25 / h as f32 * (h as f32 * (phase + phase_step)).sin())
                .sum();
            s * env
        })
        .collect()
}

fn interleave(l: &[f32], r: &[f32]) -> Vec<f32> {
    l.iter().zip(r).flat_map(|(&l, &r)| [l, r]).collect()
}

/// The `voiced()` signal of `issue_42_stereo_channels.rs`: R is a scaled,
/// phase-shifted copy of L.
fn voiced(sr: usize, n: usize) -> Vec<f32> {
    let l = harmonic(sr, n, 140.0, 3.0, 0.0);
    let r = harmonic(sr, n, 140.0, 3.0, 0.3);
    let l: Vec<f32> = l.iter().map(|s| 0.5 * s).collect();
    let r: Vec<f32> = r.iter().map(|s| 0.4 * s).collect();
    interleave(&l, &r)
}

/// Amplitude-panned speech: L = 0.9·s, R = 0.3·s. The side is exactly 0.5·mid,
/// so the stereo predictor alone carries the image.
fn panned(sr: usize, n: usize) -> Vec<f32> {
    let s = harmonic(sr, n, 140.0, 3.0, 0.0);
    let (l, r): (Vec<f32>, Vec<f32>) = s.iter().map(|&s| (0.45 * s, 0.15 * s)).unzip();
    interleave(&l, &r)
}

/// Two unrelated voices: 140 Hz on the left, 205 Hz on the right, with
/// different syllable rates. The mid predicts neither channel well.
fn two_voices(sr: usize, n: usize) -> Vec<f32> {
    let l = harmonic(sr, n, 140.0, 3.0, 0.0);
    let r = harmonic(sr, n, 205.0, 2.3, 1.0);
    let l: Vec<f32> = l.iter().map(|s| 0.45 * s).collect();
    let r: Vec<f32> = r.iter().map(|s| 0.45 * s).collect();
    interleave(&l, &r)
}

/// R = -L: the mid is silent and the whole signal is in the side.
fn anti_phase(sr: usize, n: usize) -> Vec<f32> {
    let s = harmonic(sr, n, 140.0, 3.0, 0.0);
    let (l, r): (Vec<f32>, Vec<f32>) = s.iter().map(|&s| (0.45 * s, -0.45 * s)).unzip();
    interleave(&l, &r)
}

fn channel(pcm: &[f32], c: usize) -> Vec<f32> {
    pcm.iter().skip(c).step_by(CH).copied().collect()
}

fn mid(pcm: &[f32]) -> Vec<f32> {
    pcm.chunks_exact(CH).map(|p| (p[0] + p[1]) / 2.0).collect()
}

/// Sums over `reference[start..start + len]` against `test` shifted by `lag`:
/// (Σr², Σr·t, Σ(r−t)²).
fn sums(reference: &[f32], test: &[f32], start: usize, len: usize, lag: isize) -> (f64, f64, f64) {
    let (mut rr, mut rt, mut err) = (0f64, 0f64, 0f64);
    for i in start..start + len {
        let j = i as isize + lag;
        if j < 0 || j as usize >= test.len() || i >= reference.len() {
            continue;
        }
        let (r, t) = (reference[i] as f64, test[j as usize] as f64);
        rr += r * r;
        rt += r * t;
        err += (r - t) * (r - t);
    }
    (rr, rt, err)
}

fn snr_db(reference: &[f32], test: &[f32], start: usize, len: usize, lag: isize) -> f64 {
    let (rr, _, err) = sums(reference, test, start, len, lag);
    10.0 * (rr / err.max(1e-20)).log10()
}

/// The lag in `-max_lag..=max_lag` at which `test` best matches `reference`.
fn best_lag(reference: &[f32], test: &[f32], start: usize, len: usize, max_lag: isize) -> isize {
    (-max_lag..=max_lag)
        .max_by(|&a, &b| {
            snr_db(reference, test, start, len, a)
                .total_cmp(&snr_db(reference, test, start, len, b))
        })
        .unwrap()
}

/// What one decoded stream looks like against the input.
#[derive(Debug)]
struct Channels {
    /// Least-squares gain of each decoded channel against the input mid.
    gain: [f64; 2],
    /// SNR of each decoded channel against the same input channel (dB).
    snr: [f64; 2],
    /// Normalized correlation of the decoded L and R.
    corr: f64,
}

/// The measured span: skip 100 ms of encoder start-up, and stop short of the
/// lag search range.
fn span(sr: usize, n: usize) -> (usize, usize) {
    let start = sr / 10;
    (start, n - start - sr * LAG_MS / 1000)
}

/// Measures `decoded` against `input` (both interleaved stereo), at the lag
/// that best aligns the decoded signal with the input: on the mid, or, when
/// the input mid is silent, on the left channel.
fn measure(input: &[f32], decoded: &[f32], sr: usize) -> Channels {
    let (in_mid, out_mid) = (mid(input), mid(decoded));
    let max_lag = (sr * LAG_MS / 1000) as isize;
    let (start, len) = span(sr, in_mid.len().min(out_mid.len()));
    let mid_energy: f64 = in_mid[start..start + len]
        .iter()
        .map(|&x| (x * x) as f64)
        .sum();
    let lag = if mid_energy > 1e-3 {
        best_lag(&in_mid, &out_mid, start, len, max_lag)
    } else {
        best_lag(
            &channel(input, 0),
            &channel(decoded, 0),
            start,
            len,
            max_lag,
        )
    };
    let mut out = Channels {
        gain: [0.0; 2],
        snr: [0.0; 2],
        corr: lr_correlation(decoded, sr),
    };
    for c in 0..CH {
        let (in_c, out_c) = (channel(input, c), channel(decoded, c));
        let (rr, rt, _) = sums(&in_mid, &out_c, start, len, lag);
        out.gain[c] = rt / rr.max(1e-20);
        out.snr[c] = snr_db(&in_c, &out_c, start, len, lag);
    }
    out
}

/// Normalized correlation of the L and R of `pcm` (interleaved), over the
/// measured span.
fn lr_correlation(pcm: &[f32], sr: usize) -> f64 {
    let (l, r) = (channel(pcm, 0), channel(pcm, 1));
    let (start, len) = span(sr, l.len());
    let (ll, lr, _) = sums(&l, &r, start, len, 0);
    let (rr, _, _) = sums(&r, &r, start, len, 0);
    lr / (ll * rr).sqrt().max(1e-20)
}

/// The best SNR any decoder that outputs the mid in both channels could
/// reach on channel `c`: the input channel against its least-squares fit
/// from the input mid.
fn mid_only_bound_db(input: &[f32], c: usize, sr: usize) -> f64 {
    let (in_c, in_mid) = (channel(input, c), mid(input));
    let (start, len) = span(sr, in_c.len());
    let (mm, mc, _) = sums(&in_mid, &in_c, start, len, 0);
    let (cc, _, _) = sums(&in_c, &in_c, start, len, 0);
    10.0 * (cc / (cc - mc * mc / mm).max(1e-20)).log10()
}

/// Worst per-window SNR of channel `c` of `test` against `reference`, at the
/// best lag within ±30 samples of each 20 ms window.
fn worst_window_snr(reference: &[f32], test: &[f32], c: usize, sr: usize, skip: usize) -> f64 {
    let (r, t) = (channel(reference, c), channel(test, c));
    let (win, max_lag) = (sr / 50, 30isize);
    let n = r.len().min(t.len());
    let mut worst = f64::INFINITY;
    let mut start = skip;
    while start + win + max_lag as usize <= n {
        let lag = best_lag(&r, &t, start, win, max_lag);
        worst = worst.min(snr_db(&r, &t, start, win, lag));
        start += win;
    }
    worst
}

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    sr: usize,
    bitrate: i32,
    /// Alternate between `bitrate` and this every 10 packets.
    switch_to: Option<i32>,
    cbr: bool,
    frame_ms: usize,
    bandwidth: CBw,
    /// In-band FEC at this packet-loss percentage.
    fec_loss: Option<i32>,
    /// How far below libopus's per-channel SNR opus-rs may fall.
    snr_margin_db: f64,
    /// Whether to require opus-rs's decoder to agree with libopus's on
    /// opus-rs's packets.
    check_decoder: bool,
    signal: fn(usize, usize) -> Vec<f32>,
}

impl Case {
    const fn new(
        name: &'static str,
        sr: usize,
        bitrate: i32,
        signal: fn(usize, usize) -> Vec<f32>,
    ) -> Self {
        Self {
            name,
            sr,
            bitrate,
            switch_to: None,
            cbr: true,
            frame_ms: 20,
            bandwidth: match sr {
                16000 => CBw::Wideband,
                24000 => CBw::Superwideband,
                _ => CBw::Fullband,
            },
            fec_loss: None,
            snr_margin_db: 1.0,
            check_decoder: true,
            signal,
        }
    }
}

/// One encoder's stream, decoded by libopus.
struct Stream {
    decoded: Vec<f32>,
    /// With FEC: libopus's `decode_fec` of each packet, i.e. the previous
    /// packet recovered from its LBRR data.
    recovered: Vec<f32>,
    bytes: usize,
}

struct Measured {
    input: Vec<f32>,
    /// opus-rs's packets, decoded by libopus.
    rs: Stream,
    /// libopus's packets, decoded by libopus.
    c: Stream,
    /// Per channel, worst-window SNR of opus-rs's decode of opus-rs's packets
    /// against libopus's decode of the same packets.
    agreement: [f64; 2],
}

fn run(case: &Case) -> Measured {
    let sr = case.sr;
    let frame = sr * case.frame_ms / 1000;
    let packets = SECONDS * 1000 / case.frame_ms;
    let input = (case.signal)(sr, frame * packets);

    let mut rs_enc = OpusEncoder::new(sr as i32, CH, Application::Voip).unwrap();
    rs_enc.bitrate_bps = case.bitrate;
    rs_enc.use_cbr = case.cbr;

    // libopus at opus-rs's settings, pinned to voice and to a stereo stream
    // (below ~18 kb/s it would otherwise switch to mono).
    let mut c_enc = CEnc::new(sr as u32, CCh::Stereo, CApp::Voip).unwrap();
    c_enc.set_bitrate(CBitrate::Bits(case.bitrate)).unwrap();
    c_enc.set_vbr(!case.cbr).unwrap();
    c_enc.set_complexity(rs_enc.complexity).unwrap();
    c_enc.set_bandwidth(case.bandwidth).unwrap();
    c_enc.set_signal(CSignal::Voice).unwrap();
    c_enc.set_force_channels(Some(CCh::Stereo)).unwrap();
    if let Some(loss) = case.fec_loss {
        rs_enc.use_inband_fec = true;
        rs_enc.packet_loss_perc = loss;
        c_enc.set_inband_fec(true).unwrap();
        c_enc.set_packet_loss_perc(loss).unwrap();
    }

    let mut rs_dec = OpusDecoder::new(sr as i32, CH).unwrap();
    let mut rs_by_rs = Vec::new();
    let mut streams = [(); 2].map(|_| {
        (
            CDec::new(sr as u32, CCh::Stereo).unwrap(),
            CDec::new(sr as u32, CCh::Stereo).unwrap(),
            Stream {
                decoded: Vec::new(),
                recovered: Vec::new(),
                bytes: 0,
            },
        )
    });
    let (mut pkt, mut buf) = (vec![0u8; 1500], vec![0f32; MAX_FRAME * CH]);
    for p in 0..packets {
        let pcm = &input[p * frame * CH..(p + 1) * frame * CH];
        if let Some(other) = case.switch_to {
            let rate = if (p / 10) % 2 == 0 {
                case.bitrate
            } else {
                other
            };
            rs_enc.bitrate_bps = rate;
            c_enc.set_bitrate(CBitrate::Bits(rate)).unwrap();
        }
        for (k, (dec, fec_dec, stream)) in streams.iter_mut().enumerate() {
            let n = if k == 0 {
                rs_enc.encode(pcm, frame, &mut pkt).unwrap_or_else(|e| {
                    panic!("{}: packet {p}: opus-rs encode failed: {e}", case.name)
                })
            } else {
                c_enc.encode_float(pcm, &mut pkt).unwrap()
            };
            let pkt = &pkt[..n];
            assert_eq!(
                (pkt[0] >> 2) & 1,
                1,
                "{}: packet {p}: TOC stereo bit",
                case.name
            );
            stream.bytes += n;
            let got = dec.decode_float(pkt, &mut buf, false).unwrap_or_else(|e| {
                panic!(
                    "{}: packet {p}: libopus rejected the packet: {e}",
                    case.name
                )
            });
            assert_eq!(got, frame);
            stream.decoded.extend_from_slice(&buf[..got * CH]);
            if case.fec_loss.is_some() && p > 0 {
                let got = fec_dec
                    .decode_float(pkt, &mut buf[..frame * CH], true)
                    .unwrap();
                assert_eq!(got, frame);
                stream.recovered.extend_from_slice(&buf[..got * CH]);
            }
            if k == 0 {
                let got = rs_dec.decode(pkt, MAX_FRAME, &mut buf).unwrap();
                assert_eq!(got, frame);
                rs_by_rs.extend_from_slice(&buf[..got * CH]);
            }
        }
    }

    let [(_, _, rs), (_, _, c)] = streams;
    let agreement = [0, 1].map(|ch| worst_window_snr(&rs.decoded, &rs_by_rs, ch, sr, sr / 10));
    Measured {
        input,
        rs,
        c,
        agreement,
    }
}

/// Decoder agreement on opus-rs's packets (issue_27's `MATCH_DB`).
const MATCH_DB: f64 = 30.0;
/// How far each decoded channel's gain against the input mid may stray from
/// libopus's.
const GAIN_MARGIN: f64 = 0.1;

/// Measures both streams, prints them, and checks what every case shares:
/// per-channel gain and SNR close to libopus's, and decoder agreement.
fn check(case: &Case) -> (Measured, Channels, Channels) {
    let m = run(case);
    let (rs, c) = (
        measure(&m.input, &m.rs.decoded, case.sr),
        measure(&m.input, &m.c.decoded, case.sr),
    );
    let kbps = |bytes: usize| bytes as f64 * 8.0 / SECONDS as f64 / 1000.0;
    println!(
        "{:<30} opus-rs: L {:+.2}·mid  R {:+.2}·mid  SNR L {:+.1} R {:+.1} dB  L·R {:+.2}  {:.1} kb/s",
        case.name,
        rs.gain[0],
        rs.gain[1],
        rs.snr[0],
        rs.snr[1],
        rs.corr,
        kbps(m.rs.bytes)
    );
    println!(
        "{:<30} libopus: L {:+.2}·mid  R {:+.2}·mid  SNR L {:+.1} R {:+.1} dB  L·R {:+.2}  {:.1} kb/s",
        "",
        c.gain[0],
        c.gain[1],
        c.snr[0],
        c.snr[1],
        c.corr,
        kbps(m.c.bytes)
    );
    println!(
        "{:<30} opus-rs vs libopus decoder: L {:.1} R {:.1} dB",
        "", m.agreement[0], m.agreement[1]
    );
    for (ch, name) in ["L", "R"].iter().enumerate() {
        assert!(
            (rs.gain[ch] - c.gain[ch]).abs() <= GAIN_MARGIN,
            "{}: decoded {name} = {:+.2}·mid, libopus {:+.2}·mid",
            case.name,
            rs.gain[ch],
            c.gain[ch]
        );
        assert!(
            rs.snr[ch] >= c.snr[ch] - case.snr_margin_db,
            "{}: {name} SNR {:+.1} dB, libopus {:+.1} dB",
            case.name,
            rs.snr[ch],
            c.snr[ch]
        );
        assert!(
            !case.check_decoder || m.agreement[ch] > MATCH_DB,
            "{}: opus-rs and libopus decode opus-rs's {name} differently: {:.1} dB",
            case.name,
            m.agreement[ch]
        );
    }
    (m, rs, c)
}

/// Amplitude-panned speech at 16 kb/s: libopus codes it as panned mono, the
/// mid with a predictor and no side. The L/R balance must survive. Coding the
/// mid with a zero predictor decoded both channels as the mid (+0.83·mid,
/// R at -1.7 dB). CBR, because SILK-only VBR doesn't follow the bitrate yet
/// (#43).
#[test]
fn panned_speech_keeps_its_balance_at_16k() {
    let (_, rs, _) = check(&Case::new("panned SILK WB 16k CBR", 16000, 16000, panned));
    // The input is 1.5·mid and 0.5·mid; the predictor fades toward mono at
    // this rate, but L stays well above R.
    assert!(
        rs.gain[0] > 1.5 * rs.gain[1],
        "L {:+.2}·mid vs R {:+.2}·mid: the panning was lost",
        rs.gain[0],
        rs.gain[1]
    );
}

/// Two unrelated voices need the side: each channel must come out better
/// than any mid-only decoder could make it.
fn check_two_voices(case: Case) {
    let (m, rs, c) = check(&case);
    for (ch, name) in ["L", "R"].iter().enumerate() {
        let bound = mid_only_bound_db(&m.input, ch, case.sr);
        assert!(
            rs.snr[ch] > bound + 2.0,
            "{}: {name} SNR {:+.1} dB, but a mid-only decoder reaches {bound:+.1} dB \
             (libopus {:+.1})",
            case.name,
            rs.snr[ch],
            c.snr[ch]
        );
    }
    assert!(
        (rs.corr - c.corr).abs() < 0.15,
        "{}: decoded L/R correlation {:+.2}, libopus {:+.2}",
        case.name,
        rs.corr,
        c.corr
    );
}

#[test]
fn two_voices_keep_their_own_channels_silk_wb() {
    check_two_voices(Case::new(
        "two voices SILK WB 16k CBR",
        16000,
        32000,
        two_voices,
    ));
}

#[test]
fn two_voices_keep_their_own_channels_hybrid_fb() {
    check_two_voices(Case::new(
        "two voices Hybrid FB 48k CBR",
        48000,
        32000,
        two_voices,
    ));
}

/// R = -L has a silent mid: coding the mid alone decodes silence.
#[test]
fn anti_phase_survives() {
    let (_, rs, c) = check(&Case::new(
        "anti-phase SILK WB 16k CBR",
        16000,
        32000,
        anti_phase,
    ));
    assert!(
        rs.corr < -0.8,
        "decoded L/R correlation {:+.2} (libopus {:+.2}), expected about -1",
        rs.corr,
        c.corr
    );
}

/// 10, 40 and 60 ms SILK packets: 10 ms frames smooth the stereo width over
/// shorter steps, and 40/60 ms packets carry two or three frames, each with
/// its own stereo header and mid/side split.
#[test]
fn every_silk_frame_size_codes_the_side() {
    for frame_ms in [10, 40, 60] {
        for cbr in [true, false] {
            let name = format!(
                "voiced SILK WB {frame_ms} ms {}",
                if cbr { "CBR" } else { "VBR" }
            );
            check(&Case {
                frame_ms,
                cbr,
                // opus-rs's decoder doesn't match libopus on 10 ms SILK
                // frames, libopus's own included, mono or stereo; the
                // encoder is judged through libopus's decoder.
                check_decoder: frame_ms != 10,
                ..Case::new(Box::leak(name.into_boxed_str()), 16000, 32000, voiced)
            });
        }
    }
}

/// Stereo in-band FEC: each channel's LBRR travels in the next packet, after
/// the stereo header of the frame it repeats. With two unrelated voices, a
/// recovery from the mid alone comes out with identical channels (L·R +1.00,
/// as before #48); recovered from both channels they stay as far apart as in
/// the normal decode.
///
/// Both encoders narrow this image (normal decode L·R +0.57 for opus-rs,
/// +0.54 for libopus): the LBRR bits come out of SILK's target (#51), and the
/// width control sees the lower rate. The recovery measures +0.66, libopus's
/// +0.68. Hybrid VBR, because SILK-only VBR doesn't follow the bitrate yet
/// (#43).
#[test]
fn stereo_fec_recovers_both_channels() {
    let case = Case {
        fec_loss: Some(20),
        cbr: false,
        ..Case::new("two voices Hybrid FB VBR FEC", 48000, 32000, two_voices)
    };
    let (m, rs, c) = check(&case);
    // recovered[k] is packet k's audio, recovered from packet k + 1.
    let (rs_rec, c_rec) = (
        lr_correlation(&m.rs.recovered, case.sr),
        lr_correlation(&m.c.recovered, case.sr),
    );
    println!(
        "{:<30} FEC-recovered L·R: opus-rs {rs_rec:+.2} (normal {:+.2}), libopus {c_rec:+.2} \
         (normal {:+.2})",
        "", rs.corr, c.corr
    );
    assert!(
        rs_rec < 0.8 && (rs_rec - c_rec).abs() < 0.15 && (rs_rec - rs.corr).abs() < 0.3,
        "{}: FEC-recovered L·R {rs_rec:+.2} (libopus {c_rec:+.2}), normal decode {:+.2}: \
         the side's LBRR is missing",
        case.name,
        rs.corr
    );
}

/// Switching between Hybrid and CELT at 48 kHz restarts SILK each time: the
/// side encoder and the stereo state are reset and prefilled with both
/// channels.
///
/// The transitions themselves cost opus-rs about 1.3 dB per channel against
/// libopus here, coding the mid only or not (mid-only: L +4.3, R +4.2 dB),
/// and opus-rs's decoder doesn't match libopus across these transitions,
/// libopus's own stream included. So the margin is wider and the encoder is
/// judged through libopus's decoder.
#[test]
fn hybrid_celt_switching_restarts_the_side() {
    check(&Case {
        switch_to: Some(96000),
        snr_margin_db: 2.0,
        check_decoder: false,
        ..Case::new("voiced 48k 24<->96 kb/s CBR", 48000, 24000, voiced)
    });
}
