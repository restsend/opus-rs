//! Issue #42 checked against libopus (the `opus` dev-dependency), per channel.
//!
//! Every older stereo check compared only the mid (L+R)/2, and that hid two
//! encoder bugs:
//!
//! - The stereo predictor written for mid-only frames was not neutral. Index
//!   0 dequantizes to `pred_Q13 = (0, -13364)`, and the decoder applies it to
//!   mid-only frames as well, so the side came back as -1.63·mid: L ≈
//!   -0.63·mid (phase-inverted), R ≈ 2.63·mid. The mid stayed correct.
//! - Hybrid fed SILK the interleaved L,R,L,R… input as if it were mono, so at
//!   24/48 kHz even the mid was garbage.
//!
//! opus-rs codes stereo SILK as mid only, so the bar is correct mono in both
//! channels, close to what libopus's own encoder achieves with real side
//! coding. Each config encodes the same signal with opus-rs and with libopus
//! at the same settings, decodes both with libopus, and measures each decoded
//! channel against the input. The opus-rs decoder must also agree with
//! libopus on opus-rs's packets.

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

/// Voiced-speech-like test signal, as in `issue_27_libopus_oracle.rs`: a
/// 140 Hz harmonic series with slow pitch drift and a syllable-rate envelope.
/// The right channel is a scaled, phase-shifted copy, so the side signal is
/// not trivially zero.
fn voiced(sr: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * CH);
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
        out.push(r * env * 0.4);
    }
    out
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
    /// SNR of the decoded mid against the input mid (dB).
    mid_snr: f64,
}

/// Measures `decoded` against `input` (both interleaved stereo). The lag is
/// the one that best aligns the decoded mid with the input mid, as in the
/// issue's repro; every measure uses it.
fn measure(input: &[f32], decoded: &[f32], sr: usize) -> Channels {
    let (in_mid, out_mid) = (mid(input), mid(decoded));
    let max_lag = (sr * LAG_MS / 1000) as isize;
    // Skip 100 ms of encoder start-up; stop short of the lag search range.
    let start = sr / 10;
    let len = in_mid.len().min(out_mid.len()) - start - max_lag as usize;
    let lag = best_lag(&in_mid, &out_mid, start, len, max_lag);
    let mut out = Channels {
        gain: [0.0; 2],
        snr: [0.0; 2],
        mid_snr: snr_db(&in_mid, &out_mid, start, len, lag),
    };
    for c in 0..CH {
        let (in_c, out_c) = (channel(input, c), channel(decoded, c));
        let (rr, rt, _) = sums(&in_mid, &out_c, start, len, lag);
        out.gain[c] = rt / rr;
        out.snr[c] = snr_db(&in_c, &out_c, start, len, lag);
    }
    out
}

/// Worst per-20 ms-window SNR of channel `c` of `test` against `reference`,
/// at the best lag within ±`max_lag` of each window.
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

/// TOC check: `config` in `configs` and the stereo bit set.
fn assert_toc(who: &str, pkt: &[u8], configs: std::ops::RangeInclusive<u8>) {
    let config = pkt[0] >> 3;
    assert!(
        configs.contains(&config),
        "{who}: TOC config {config}, expected {configs:?}"
    );
    assert_eq!((pkt[0] >> 2) & 1, 1, "{who}: TOC stereo bit");
}

struct Config {
    name: &'static str,
    sr: usize,
    cbr: bool,
    bandwidth: CBw,
    /// TOC configs both encoders must produce.
    configs: std::ops::RangeInclusive<u8>,
}

struct Measured {
    /// opus-rs's packets, decoded by libopus.
    rs: Channels,
    /// libopus's packets, decoded by libopus.
    c: Channels,
    /// Per channel, worst-window SNR of opus-rs's decode of opus-rs's packets
    /// against libopus's decode of the same packets.
    agreement: [f64; 2],
}

fn run(cfg: &Config) -> Measured {
    let sr = cfg.sr;
    let frame = sr / 50;
    let packets = SECONDS * 50;
    let input = voiced(sr, frame * packets);

    let mut rs_enc = OpusEncoder::new(sr as i32, CH, Application::Voip).unwrap();
    rs_enc.bitrate_bps = 32000;
    rs_enc.use_cbr = cfg.cbr;

    // libopus at opus-rs's settings. A steady harmonic tone can read as music
    // and pull libopus into CELT, and below ~18 kb/s libopus would switch to
    // a mono stream, so pin the signal type and the channel count.
    let mut c_enc = CEnc::new(sr as u32, CCh::Stereo, CApp::Voip).unwrap();
    c_enc.set_bitrate(CBitrate::Bits(32000)).unwrap();
    c_enc.set_vbr(!cfg.cbr).unwrap();
    c_enc.set_complexity(rs_enc.complexity).unwrap();
    c_enc.set_bandwidth(cfg.bandwidth).unwrap();
    c_enc.set_signal(CSignal::Voice).unwrap();
    c_enc.set_force_channels(Some(CCh::Stereo)).unwrap();

    let mut c_dec_rs = CDec::new(sr as u32, CCh::Stereo).unwrap();
    let mut c_dec_c = CDec::new(sr as u32, CCh::Stereo).unwrap();
    let mut rs_dec = OpusDecoder::new(sr as i32, CH).unwrap();

    let (mut rs_by_c, mut c_by_c, mut rs_by_rs) = (Vec::new(), Vec::new(), Vec::new());
    let (mut pkt, mut buf) = (vec![0u8; 1500], vec![0f32; MAX_FRAME * CH]);
    for p in 0..packets {
        let pcm = &input[p * frame * CH..(p + 1) * frame * CH];

        let n = rs_enc
            .encode(pcm, frame, &mut pkt)
            .unwrap_or_else(|e| panic!("packet {p}: opus-rs encode failed: {e}"));
        let rs_pkt = &pkt[..n];
        assert_toc(&format!("opus-rs packet {p}"), rs_pkt, cfg.configs.clone());

        let got = c_dec_rs.decode_float(rs_pkt, &mut buf, false).unwrap();
        assert_eq!(got, frame);
        rs_by_c.extend_from_slice(&buf[..got * CH]);

        let got = rs_dec.decode(rs_pkt, MAX_FRAME, &mut buf).unwrap();
        assert_eq!(got, frame);
        rs_by_rs.extend_from_slice(&buf[..got * CH]);

        let n = c_enc.encode_float(pcm, &mut pkt).unwrap();
        assert_toc(
            &format!("libopus packet {p}"),
            &pkt[..n],
            cfg.configs.clone(),
        );
        let got = c_dec_c.decode_float(&pkt[..n], &mut buf, false).unwrap();
        assert_eq!(got, frame);
        c_by_c.extend_from_slice(&buf[..got * CH]);
    }

    let skip = sr / 10;
    Measured {
        rs: measure(&input, &rs_by_c, sr),
        c: measure(&input, &c_by_c, sr),
        agreement: [0, 1].map(|c| worst_window_snr(&rs_by_c, &rs_by_rs, c, sr, skip)),
    }
}

/// How far below libopus's per-channel SNR opus-rs may fall. Coding the mid
/// only costs the channel whose shape differs most from the mid; measured
/// about 1.3 dB on R here.
const CHANNEL_MARGIN_DB: f64 = 3.0;
/// How far below libopus's mid SNR opus-rs may fall.
const MID_MARGIN_DB: f64 = 2.0;
/// Decoder agreement on opus-rs's packets, clean (issue_27's `MATCH_DB`).
const MATCH_DB: f64 = 30.0;

fn check(cfg: Config) {
    let m = run(&cfg);
    println!(
        "{:<22} opus-rs: L {:+.2}·mid  R {:+.2}·mid  SNR L {:+.1} R {:+.1} mid {:+.1} dB",
        cfg.name, m.rs.gain[0], m.rs.gain[1], m.rs.snr[0], m.rs.snr[1], m.rs.mid_snr
    );
    println!(
        "{:<22} libopus: L {:+.2}·mid  R {:+.2}·mid  SNR L {:+.1} R {:+.1} mid {:+.1} dB",
        "", m.c.gain[0], m.c.gain[1], m.c.snr[0], m.c.snr[1], m.c.mid_snr
    );
    println!(
        "{:<22} opus-rs vs libopus decoder: L {:.1} R {:.1} dB",
        "", m.agreement[0], m.agreement[1]
    );

    for (c, name) in ["L", "R"].iter().enumerate() {
        assert!(
            (0.6..=1.2).contains(&m.rs.gain[c]),
            "{}: decoded {name} = {:+.2}·mid, expected the mid in both channels \
             (libopus: {:+.2}·mid)",
            cfg.name,
            m.rs.gain[c],
            m.c.gain[c]
        );
        assert!(
            m.rs.snr[c] >= m.c.snr[c] - CHANNEL_MARGIN_DB,
            "{}: {name} SNR {:+.1} dB, libopus {:+.1} dB",
            cfg.name,
            m.rs.snr[c],
            m.c.snr[c]
        );
    }
    assert!(
        m.rs.mid_snr >= m.c.mid_snr - MID_MARGIN_DB,
        "{}: mid SNR {:+.1} dB, libopus {:+.1} dB",
        cfg.name,
        m.rs.mid_snr,
        m.c.mid_snr
    );
    for (c, name) in ["L", "R"].iter().enumerate() {
        assert!(
            m.agreement[c] > MATCH_DB,
            "{}: opus-rs and libopus decode opus-rs's {name} differently: {:.1} dB",
            cfg.name,
            m.agreement[c]
        );
    }
}

#[test]
fn silk_wb_16k_vbr_decodes_mid_in_both_channels() {
    check(Config {
        name: "SILK WB 16 kHz VBR",
        sr: 16000,
        cbr: false,
        bandwidth: CBw::Wideband,
        configs: 8..=11,
    });
}

#[test]
fn silk_wb_16k_cbr_decodes_mid_in_both_channels() {
    check(Config {
        name: "SILK WB 16 kHz CBR",
        sr: 16000,
        cbr: true,
        bandwidth: CBw::Wideband,
        configs: 8..=11,
    });
}

#[test]
fn hybrid_swb_24k_vbr_decodes_mid_in_both_channels() {
    check(Config {
        name: "Hybrid SWB 24 kHz VBR",
        sr: 24000,
        cbr: false,
        bandwidth: CBw::Superwideband,
        configs: 12..=13,
    });
}

#[test]
fn hybrid_fb_48k_vbr_decodes_mid_in_both_channels() {
    check(Config {
        name: "Hybrid FB 48 kHz VBR",
        sr: 48000,
        cbr: false,
        bandwidth: CBw::Fullband,
        configs: 14..=15,
    });
}

#[test]
fn hybrid_fb_48k_cbr_decodes_mid_in_both_channels() {
    check(Config {
        name: "Hybrid FB 48 kHz CBR",
        sr: 48000,
        cbr: true,
        bandwidth: CBw::Fullband,
        configs: 14..=15,
    });
}
