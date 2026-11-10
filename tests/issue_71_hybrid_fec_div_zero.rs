//! Issue #71: with in-band FEC at 24/48 kHz and 20-28 kb/s, the very first
//! frame panicked with "attempt to divide by zero" in `interp_bits2pulses`
//! (src/rate.rs).
//!
//! Root cause: `decide_fec` (loss > 5%) walks the frame bandwidth down until
//! FEC fits the rate, and the hybrid → SILK-only mode fixup must re-run on
//! that post-FEC bandwidth. opus-rs only ran the fixup before `decide_fec`,
//! so the frame stayed Hybrid with a Wideband bandwidth: CELT started at
//! band 17 with end band 17 — zero coded bands — and `interp_bits2pulses`
//! divided by `e_bands[17] - e_bands[17] == 0`.
//!
//! The fix has two halves:
//! - the FEC decision consumes the equivalent rate recomputed with the
//!   decided mode (full SILK packet-loss term, none for CELT); the previous
//!   pre-decision half-loss variant flipped FEC decisions near the thresholds;
//! - the hybrid → SILK-only demotion runs *after* `decide_fec`, on the
//!   post-FEC bandwidth.
//!
//! Known remaining gap (out of scope): for FEC-on SILK-only frames the
//! reference SILK layer picks its internal sample rate from the bitrate while
//! opus-rs derives it from the (walked-down) frame bandwidth, so the TOC
//! *bandwidth* can differ within the same mode; stereo hybrid payload sizes
//! also differ for unrelated reasons. These tests therefore pin the issue
//! (no panic, decodable packets), mode parity against the reference encoder
//! everywhere, and full TOC parity where opus-rs is otherwise bit-faithful.
//!
//! The reference side is the `opus` dev-dependency.

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Encoder as CEnc,
    Signal as CSignal,
};
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::io::Read;

const LOSS_PERC: i32 = 10;
const RATES: [i32; 6] = [12000, 16000, 20000, 24000, 28000, 32000];
/// (API sample rate, samples per frame): 10 ms and 20 ms at the two
/// hybrid-capable rates, the band the issue panics in.
const CASES: [(usize, usize); 4] = [(24000, 240), (24000, 480), (48000, 480), (48000, 960)];
const PACKETS: usize = 6;

/// A voiced harmonic series (140 Hz, drifting) under a syllable-rate
/// envelope, the same signal shape as `issue_51_cbr_fec_budget.rs`.
fn voiced(sr: usize, n: usize, ch: usize) -> Vec<i16> {
    let mut out = Vec::with_capacity(n * ch);
    let mut phase = 0.0f32;
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let f0 = 140.0 + 15.0 * (2.0 * core::f32::consts::PI * 0.7 * t).sin();
        phase += 2.0 * core::f32::consts::PI * f0 / sr as f32;
        let env = 0.6 + 0.4 * (2.0 * core::f32::consts::PI * 3.0 * t).sin();
        let mut s = 0.0f32;
        for h in 1..=8 {
            s += 0.25 / h as f32 * (h as f32 * phase).sin();
        }
        let s = (s * env * 0.5 * 32768.0) as i16;
        for _ in 0..ch {
            out.push(s);
        }
    }
    out
}

/// Real speech (the repo's fixture), resampled to the API rate: CBR packet
/// sizes are content-independent, and VBR averages need speech-like input to
/// be meaningful.
fn speech(sr: usize, n: usize, ch: usize) -> Vec<i16> {
    let mut f = std::fs::File::open("fixtures/answer_16k.wav").expect("fixture wav");
    let mut header = [0u8; 44];
    f.read_exact(&mut header).expect("wav header");
    let mut data = Vec::new();
    f.read_to_end(&mut data).expect("wav data");
    let mono: Vec<i16> = data
        .chunks(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    let src_rate = u32::from_le_bytes([header[24], header[25], header[26], header[27]]) as i32;
    let mut s = if src_rate == sr as i32 {
        mono
    } else {
        let out_len = mono.len() as i64 * sr as i64 / src_rate as i64;
        let mut r = opus_rs::silk::resampler::SilkResampler::default();
        r.init(src_rate, sr as i32);
        let mut out = vec![0i16; out_len as usize];
        r.process(&mut out, &mono, mono.len() as i32);
        out
    };
    s.truncate(n * ch);
    if ch == 2 {
        s = s.iter().flat_map(|&x| [x, x]).collect();
    }
    s.resize(n * ch, 0);
    s
}

/// opus-rs packets for one matrix cell.
fn rs_packets(sr: usize, frame: usize, ch: usize, bitrate: i32, fec: bool) -> Vec<Vec<u8>> {
    let mut enc = OpusEncoder::new(sr as i32, ch, Application::Voip).unwrap();
    enc.bitrate_bps = bitrate;
    enc.use_inband_fec = fec;
    enc.packet_loss_perc = LOSS_PERC;
    // The reference encoder is pinned to a bandwidth (see `c_packets`).
    // Since set_bandwidth, the port no longer derives the bandwidth from
    // the API rate, so force the same one here.
    enc.set_bandwidth(if sr == 24000 {
        opus_rs::Bandwidth::Superwideband
    } else {
        opus_rs::Bandwidth::Fullband
    });
    let input = voiced(sr, frame * PACKETS, ch);
    let mut out = vec![0u8; 1500];
    (0..PACKETS)
        .map(|p| {
            let n = enc
                .encode_i16(
                    &input[p * frame * ch..(p + 1) * frame * ch],
                    frame,
                    &mut out,
                )
                .unwrap_or_else(|e| panic!("opus-rs encode failed at {sr} Hz/{bitrate} b/s: {e}"));
            assert!(
                n > 1,
                "opus-rs produced a TOC-only packet at {sr} Hz/{bitrate} b/s"
            );
            out[..n].to_vec()
        })
        .collect()
}

/// Reference packets for the same cell, pinned to opus-rs's fixed bandwidth
/// (the port derives it from the API rate instead of adapting it): SWB at
/// 24 kHz, FB at 48 kHz.
fn c_packets(sr: usize, frame: usize, ch: usize, bitrate: i32, fec: bool) -> Vec<Vec<u8>> {
    let mut enc = CEnc::new(
        sr as u32,
        if ch == 2 { CCh::Stereo } else { CCh::Mono },
        CApp::Voip,
    )
    .unwrap();
    enc.set_signal(CSignal::Voice).unwrap();
    enc.set_bandwidth(if sr == 24000 {
        CBw::Superwideband
    } else {
        CBw::Fullband
    })
    .unwrap();
    enc.set_complexity(9).unwrap();
    enc.set_vbr(true).unwrap();
    enc.set_bitrate(CBitrate::Bits(bitrate)).unwrap();
    if fec {
        enc.set_inband_fec(true).unwrap();
        enc.set_packet_loss_perc(LOSS_PERC).unwrap();
    }
    let input = voiced(sr, frame * PACKETS, ch);
    let mut out = vec![0u8; 1500];
    (0..PACKETS)
        .map(|p| {
            let n = enc
                .encode(&input[p * frame * ch..(p + 1) * frame * ch], &mut out)
                .unwrap();
            out[..n].to_vec()
        })
        .collect()
}

/// TOC config -> mode class: 0-11 SILK-only, 12-19 hybrid, 20-31 CELT-only.
fn toc_mode(toc: u8) -> &'static str {
    match toc >> 3 {
        0..=11 => "silk",
        12..=19 => "hybrid",
        _ => "celt",
    }
}

#[test]
fn issue71_matrix_encodes_and_matches_reference_mode() {
    for ch in [1usize, 2] {
        for &(sr, frame) in &CASES {
            for &bitrate in &RATES {
                for fec in [false, true] {
                    let rs = rs_packets(sr, frame, ch, bitrate, fec);
                    let c = c_packets(sr, frame, ch, bitrate, fec);
                    for p in 0..PACKETS {
                        assert_eq!(
                            toc_mode(rs[p][0]),
                            toc_mode(c[p][0]),
                            "mode mismatch (ch={ch}, {sr} Hz, {frame} samples, {bitrate} b/s, \
                             fec={fec}, packet {p}): opus-rs cfg {} vs reference cfg {}",
                            rs[p][0] >> 3,
                            c[p][0] >> 3,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn issue71_without_fec_toc_matches_reference() {
    // Without FEC no bandwidth walk happens, so opus-rs is TOC-faithful here
    // (mono; stereo hybrid payload sizes differ for unrelated reasons).
    for &(sr, frame) in &CASES {
        for &bitrate in &RATES {
            let rs = rs_packets(sr, frame, 1, bitrate, false);
            let c = c_packets(sr, frame, 1, bitrate, false);
            for p in 0..PACKETS {
                assert_eq!(
                    rs[p][0], c[p][0],
                    "TOC mismatch ({sr} Hz, {frame} samples, {bitrate} b/s, fec=off, packet {p})"
                );
            }
        }
    }
}

/// Total bytes opus-rs produces for real speech at one cell of the band.
fn rs_speech_bytes(sr: usize, frame: usize, bitrate: i32, vbr: bool, fec: bool) -> usize {
    let mut enc = OpusEncoder::new(sr as i32, 1, Application::Voip).unwrap();
    enc.bitrate_bps = bitrate;
    enc.use_cbr = !vbr;
    enc.use_inband_fec = fec;
    enc.packet_loss_perc = LOSS_PERC;
    let input = speech(sr, frame * 400, 1);
    let mut out = vec![0u8; 1500];
    let mut total = 0;
    for p in 0..400 {
        let n = enc
            .encode_i16(&input[p * frame..(p + 1) * frame], frame, &mut out)
            .unwrap_or_else(|e| panic!("opus-rs encode failed at {sr} Hz/{bitrate} b/s: {e}"));
        total += n;
    }
    total
}

/// Same for the reference encoder, pinned to opus-rs's fixed bandwidth state.
fn c_speech_bytes(sr: usize, frame: usize, bitrate: i32, vbr: bool, fec: bool) -> usize {
    let mut enc = CEnc::new(sr as u32, CCh::Mono, CApp::Voip).unwrap();
    enc.set_signal(CSignal::Voice).unwrap();
    enc.set_bandwidth(if sr == 24000 {
        CBw::Superwideband
    } else {
        CBw::Fullband
    })
    .unwrap();
    enc.set_complexity(9).unwrap();
    enc.set_vbr(vbr).unwrap();
    enc.set_bitrate(CBitrate::Bits(bitrate)).unwrap();
    if fec {
        enc.set_inband_fec(true).unwrap();
        enc.set_packet_loss_perc(LOSS_PERC).unwrap();
    }
    let input = speech(sr, frame * 400, 1);
    let mut out = vec![0u8; 1500];
    let mut total = 0;
    for p in 0..400 {
        let n = enc
            .encode(&input[p * frame..(p + 1) * frame], &mut out)
            .unwrap();
        total += n;
    }
    total
}

#[test]
fn issue71_cbr_bitrate_matches_reference_exactly() {
    // Under CBR both encoders must land on the nominal packet size for every
    // frame of the band — including with FEC on, whose LBRR section has to
    // fit inside the same budget (the counterpart of issue #51).
    for &(sr, frame) in &CASES {
        for &bitrate in &RATES {
            for fec in [false, true] {
                let rs = rs_speech_bytes(sr, frame, bitrate, false, fec);
                let c = c_speech_bytes(sr, frame, bitrate, false, fec);
                assert_eq!(
                    rs, c,
                    "CBR total bytes differ ({sr} Hz, {frame} samples, {bitrate} b/s, fec={fec})"
                );
            }
        }
    }
}

#[test]
fn issue71_vbr_rescued_cells_match_reference_bitrate() {
    // The cells the fix rescued (FEC engages after the bandwidth walk and the
    // frame demotes to SILK-only) must land at the reference's actual VBR
    // rate; before the fix they panicked on the first frame. Everything else
    // in VBR hybrid has a pre-existing overshoot (silk/CELT VBR target
    // accuracy) and is intentionally not asserted here.
    for &(sr, frame) in &CASES {
        for &(bitrate, ms) in &[(24000i32, 10), (28000, 10), (20000, 20), (24000, 20)] {
            if frame != sr * ms as usize / 1000 {
                continue;
            }
            let rs = rs_speech_bytes(sr, frame, bitrate, true, true) as f64;
            let c = c_speech_bytes(sr, frame, bitrate, true, true) as f64;
            let delta_pct = 100.0 * (rs - c) / c;
            assert!(
                delta_pct.abs() < 5.0,
                "VBR rate off for rescued cell ({sr} Hz, {frame} samples, {bitrate} b/s, \
                 fec=on): opus-rs {rs:.0} B vs reference {c:.0} B ({delta_pct:+.1}%)"
            );
        }
    }
}

#[test]
fn issue71_exact_repro_from_report() {
    // The reproducer from the issue, verbatim (silence, mono, 24 kHz, 20 ms).
    let mut enc = OpusEncoder::new(24_000, 1, Application::Voip).unwrap();
    enc.bitrate_bps = 24_000;
    enc.use_inband_fec = true;
    enc.packet_loss_perc = 10;
    let pcm = vec![0i16; 480];
    let mut packet = [0u8; 1500];
    let n = enc.encode_i16(&pcm, 480, &mut packet).unwrap();
    assert!(n > 1, "expected a real packet, got {n} bytes");
    // And it round-trips through the decoder.
    let mut dec = OpusDecoder::new(24_000, 1).unwrap();
    let mut pcm_out = vec![0f32; 480];
    let written = dec.decode(&packet[..n], 480, &mut pcm_out).unwrap();
    assert_eq!(written, 480);
}
