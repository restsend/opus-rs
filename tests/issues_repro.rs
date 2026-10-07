//! Reproduction tests for GitHub issues #5, #6, #10.
//!
//! Each test mirrors the minimal reproducer from the corresponding issue.

use opus_rs::OpusDecoder;

fn try_decode(label: &str, pkt: &[u8], frame_size: usize, channels: usize) {
    let mut dec = OpusDecoder::new(48000, channels).unwrap();
    let mut pcm = vec![0.0f32; frame_size * channels];
    match dec.decode(pkt, frame_size, &mut pcm) {
        Ok(n) => println!("{label}: Ok({n})"),
        Err(e) => println!("{label}: Err(\"{e}\")"),
    }
}

// ---------------------------------------------------------------------------
// Issue #10: Frame lengths decoded with a 15-bit continuation scheme instead
// of RFC 6716 §3.2.1 — any explicit length >= 128 is mis-parsed.
// ---------------------------------------------------------------------------
#[test]
fn issue_10_code2_first_frame_200_bytes() {
    // code 2, first frame 200 bytes. RFC 6716 §3.2.1 writes 200 as the
    // single byte 0xC8 — any value below 252 is a one-byte length.
    let mut a = vec![0xfa_u8, 0xC8];
    a.extend(std::iter::repeat_n(0xAA, 200)); // frame 1
    a.extend(std::iter::repeat_n(0xBB, 50)); // frame 2
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut pcm = vec![0.0f32; 1920];
    let res = dec.decode(&a, 1920, &mut pcm);
    assert!(
        res.is_ok(),
        "code2 first frame 200B should decode, got: {res:?}"
    );
}

#[test]
fn issue_10_code3_vbr_first_frame_300_bytes() {
    // code 3, VBR, 2 frames, first frame 300 bytes.
    // RFC writes 300 as [252, 12] -> 12 * 4 + 252 = 300.
    let mut b = vec![0xfb_u8, 0x82, 252, 12];
    b.extend(std::iter::repeat_n(0xAA, 300));
    b.extend(std::iter::repeat_n(0xBB, 50));
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut pcm = vec![0.0f32; 1920];
    let res = dec.decode(&b, 1920, &mut pcm);
    assert!(
        res.is_ok(),
        "code3 VBR first frame 300B should decode, got: {res:?}"
    );
}

#[test]
fn issue_10_control_code2_first_frame_100_bytes() {
    // control: same shape with a 100-byte first frame, below the 128 threshold,
    // where both schemes happen to agree.
    let mut c = vec![0xfa_u8, 100];
    c.extend(std::iter::repeat_n(0xAA, 100));
    c.extend(std::iter::repeat_n(0xBB, 50));
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut pcm = vec![0.0f32; 1920];
    let res = dec.decode(&c, 1920, &mut pcm);
    assert!(res.is_ok(), "control should decode, got: {res:?}");
}

// ---------------------------------------------------------------------------
// Issue #6: Valid code-3 CBR (VBR=0) multi-frame packets rejected — VBR bit
// ignored in code-3 parsing.
// ---------------------------------------------------------------------------
#[test]
fn issue_6_code3_cbr_vbr0_multi_frame() {
    // TOC 0xbb: CELT, 20 ms frames, code 3. Count byte 0x03: V=0 (CBR), M=3.
    // Remaining 6 bytes = 3 frames x 2 bytes, no length fields.
    let pkt: [u8; 8] = [0xbb, 0x03, 0xff, 0xfe, 0xff, 0xfe, 0xff, 0xfe];
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut pcm = vec![0.0f32; 2880]; // 3 x 20 ms @ 48 kHz
    match dec.decode(&pkt, 2880, &mut pcm) {
        Ok(n) => println!("decoded {n} samples (expected)"),
        Err(e) => panic!("Err: {e} (BUG: valid packet rejected)"),
    }
}

// ---------------------------------------------------------------------------
// Issue #5: Panic on valid SILK 40/60 ms frames: SILK workspace hardcoded
// for 20 ms (src/lib.rs:978).
// ---------------------------------------------------------------------------
// The exact triggering packet from the issue is a real libopus output with
// TOC 0x58 (SILK, wideband, 60 ms, code 0). We synthesize a minimal SILK
// 60 ms packet by encoding with the library's own encoder to verify the
// decode path no longer panics for the larger internal frame size. We use a
// code-0 SILK packet whose TOC requests 60 ms; the payload may be garbage
// but the panic happens in buffer slicing *before* SILK consumes payload,
// so we only check that we don't panic with a slice OOB.
#[test]
fn issue_5_silk_60ms_no_panic_wb_mono() {
    // TOC 0x58 = SILK WB 60ms code 0 (per RFC 6716 table).
    // internal_frame_size = 60ms * 16kHz = 960 samples (mono).
    // Before fix: w_pcm_i16 has only 640 slots -> slice[..960] panics.
    let pkt = vec![0x58_u8, 0x00];
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut pcm = vec![0.0f32; 2880]; // 60ms @ 48kHz
    // We expect a decode error (garbage payload) but NOT a panic.
    let _ = dec.decode(&pkt, 2880, &mut pcm);
}

#[test]
fn issue_5_silk_40ms_stereo_no_panic() {
    // TOC 0x3B = SILK WB 40ms stereo code 3... use a simpler code-0 stereo.
    // TOC bits: silk=0, stereo bit 0x04. WB 40ms config: (toc>>3)&0x3 == 2.
    // 0b0001_0100 = 0x14: silk, WB(2<<5=0x40)... let's just craft: WB=0x40, 40ms config=2 -> (2<<3)=0x10, stereo=0x04 => 0x74?
    // Actually SILK TOC: bits[7:6]=00 silk, bits[5:4]=bw, bits[3:2]=config, bit[1:0]=code.
    // For WB: bits[5:4] with wb mapping; here we rely on frame_duration_ms_from_toc for SILK using bits[4:3].
    // frame_duration uses (toc>>3)&0x3: 40ms -> 2. stereo bit = 0x04.
    // SILK bandwidth_from_toc: (toc>>5)&0x3, WB -> 2 => bits[6:5]=10 => toc has 0x40.
    // So toc = 0x40 | (2<<3) | 0x04 | code0 = 0x40 | 0x10 | 0x04 = 0x54.
    let toc: u8 = 0x54; // SILK WB 40ms stereo code 0
    let pkt = vec![toc, 0x00];
    let mut dec = OpusDecoder::new(48000, 2).unwrap();
    let mut pcm = vec![0.0f32; 2 * 1920]; // 40ms @ 48kHz stereo
    // Before fix: internal 40ms*16k*2ch = 1280 > 640 -> panic.
    let _ = dec.decode(&pkt, 1920, &mut pcm);
}

// ---------------------------------------------------------------------------
// Code 1 hardening: odd-length payload must be rejected (RFC 6716 §3.2.1).
// ---------------------------------------------------------------------------
#[test]
fn code1_odd_length_rejected() {
    // TOC 0xE9 = CELT 20ms mono code 1. 5 data bytes = odd -> invalid.
    let pkt = [0xE9_u8, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut pcm = vec![0.0f32; 1920];
    let res = dec.decode(&pkt, 1920, &mut pcm);
    assert!(
        res.is_err(),
        "odd-length code 1 should be rejected, got {res:?}"
    );
}

#[test]
fn code1_even_length_accepted() {
    // Same shape, but 6 data bytes (even) -> two equal 3-byte frames.
    let pkt = [0xE9_u8, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut pcm = vec![0.0f32; 1920];
    let res = dec.decode(&pkt, 1920, &mut pcm);
    assert!(res.is_ok(), "even-length code 1 should decode, got {res:?}");
}

// ---------------------------------------------------------------------------
// Issue #7 sub-item 1: frame_size mismatch must return Err, not panic.
// ---------------------------------------------------------------------------
#[test]
fn issue_7_frame_size_zero_no_panic() {
    let pkt = [0xF8_u8, 0x00, 0x00, 0x00, 0x00]; // CELT 20ms mono
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut pcm = vec![];
    let res = dec.decode(&pkt, 0, &mut pcm);
    assert!(res.is_err(), "frame_size=0 should Err, got {res:?}");
}

#[test]
fn issue_7_frame_size_too_small_no_panic() {
    // Correct frame_size for CELT 20ms @ 48kHz mono is 960; pass 120.
    let pkt = [0xF8_u8, 0x00, 0x00, 0x00, 0x00];
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut pcm = vec![0.0f32; 120];
    let res = dec.decode(&pkt, 120, &mut pcm);
    assert!(res.is_err(), "too-small frame_size should Err, got {res:?}");
}

#[test]
fn issue_7_decode_returns_actual_sample_count() {
    // CELT 20ms @ 48kHz mono → should return 960, not whatever frame_size was passed.
    let pkt = [0xF8_u8, 0x00, 0x00, 0x00, 0x00];
    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let mut pcm = vec![0.0f32; 1920]; // pass double the needed size
    let n = dec.decode(&pkt, 1920, &mut pcm).unwrap();
    assert_eq!(
        n, 960,
        "should return actual decoded samples (960), got {n}"
    );
    // tail should be zero-filled
    assert!(
        pcm[960..].iter().all(|&x| x == 0.0),
        "tail should be zero-filled"
    );
}

// ---------------------------------------------------------------------------
// Issue #7 sub-item 3: Stereo SILK must decode M/S, not replicate mono.
//
// libopus encodes these streams: opus-rs's own encoder codes stereo as the
// mid only (issue #42), so its packets cannot show whether the decoder
// applies the predictor and the side channel.
// ---------------------------------------------------------------------------

/// Encode 0.5 s of 16 kHz stereo SILK with libopus, with `l`/`r` giving each
/// channel's sample at time `t`, and decode it with opus-rs and with libopus.
/// Returns both interleaved outputs, opus-rs's first.
fn libopus_stereo_silk(l: impl Fn(f64) -> f32, r: impl Fn(f64) -> f32) -> (Vec<f32>, Vec<f32>) {
    use opus::{
        Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh,
        Decoder as CDec, Encoder as CEnc, Signal as CSignal,
    };
    let (sr, fs) = (16000, 320);
    let mut enc = CEnc::new(sr, CCh::Stereo, CApp::Voip).unwrap();
    enc.set_bitrate(CBitrate::Bits(32000)).unwrap();
    enc.set_bandwidth(CBw::Wideband).unwrap();
    enc.set_signal(CSignal::Voice).unwrap();
    enc.set_force_channels(Some(CCh::Stereo)).unwrap();
    let mut c_dec = CDec::new(sr, CCh::Stereo).unwrap();
    let mut rs_dec = OpusDecoder::new(sr as i32, 2).unwrap();

    let (mut rs_out, mut c_out) = (Vec::new(), Vec::new());
    let (mut pkt, mut buf) = (vec![0u8; 1500], vec![0f32; fs * 2]);
    for p in 0..25 {
        let pcm: Vec<f32> = (p * fs..(p + 1) * fs)
            .flat_map(|i| {
                let t = i as f64 / sr as f64;
                [l(t), r(t)]
            })
            .collect();
        let n = enc.encode_float(&pcm, &mut pkt).unwrap();
        assert!(pkt[0] >> 3 <= 11, "libopus did not code SILK-only");
        assert_eq!(rs_dec.decode(&pkt[..n], fs, &mut buf).unwrap(), fs);
        rs_out.extend_from_slice(&buf);
        assert_eq!(c_dec.decode_float(&pkt[..n], &mut buf, false).unwrap(), fs);
        c_out.extend_from_slice(&buf);
    }
    (rs_out, c_out)
}

/// SNR (dB) of channel `c` of `test` against the same channel of
/// `reference`, both interleaved stereo.
fn channel_snr(reference: &[f32], test: &[f32], c: usize) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for (r, t) in reference.iter().zip(test).skip(c).step_by(2) {
        sig += (*r as f64).powi(2);
        err += (*r as f64 - *t as f64).powi(2);
    }
    10.0 * (sig / err.max(1e-30)).log10()
}

/// opus-rs's SILK decoder is fixed-point like libopus's, so the two agree
/// to float rounding (~190 dB measured). A predictor dequantized one step
/// off (the old 6553 constant) or a stereo state wrongly reset shows as a
/// small but systematic error, far below this.
const SILK_MATCH_DB: f64 = 100.0;

fn assert_matches_libopus(rs: &[f32], c: &[f32]) {
    for (ch, name) in [(0, "L"), (1, "R")] {
        let snr = channel_snr(c, rs, ch);
        assert!(
            snr > SILK_MATCH_DB,
            "{name}: opus-rs vs libopus {snr:.1} dB"
        );
    }
}

#[test]
fn issue_7_stereo_silk_channels_differ() {
    // Left = 440 Hz tone, Right = silence. Different per channel.
    let tone = |t: f64| (440.0 * t * 2.0 * std::f64::consts::PI).sin() as f32 * 0.3;
    let (out, c_out) = libopus_stereo_silk(tone, |_| 0.0);

    let l_max = out.iter().step_by(2).fold(0f32, |m, x| m.max(x.abs()));
    assert!(
        l_max > 0.01,
        "Left channel should have energy, got max={l_max}"
    );
    // Right should differ substantially from Left (not a mono copy).
    let frames = out.len() / 2;
    let l_minus_r: f32 = out.chunks_exact(2).map(|p| (p[0] - p[1]).abs()).sum();
    assert!(
        l_minus_r / frames as f32 > 0.005,
        "L and R should differ (M/S decoding), got avg diff={}",
        l_minus_r / frames as f32
    );
    assert_matches_libopus(&out, &c_out);
}

/// A quadrature pair (same tone, 90° apart): equal power and spectrum but
/// uncorrelated, so libopus codes the side with both predictors quantized to
/// exactly zero. The decoder used to clear its stereo state on every such
/// frame (a heuristic keyed on a zero predictor), which libopus never does.
#[test]
fn issue_7_stereo_silk_zero_predictor_matches_libopus() {
    use std::f64::consts::PI;
    let (out, c_out) = libopus_stereo_silk(
        |t| (300.0 * t * 2.0 * PI).sin() as f32 * 0.3,
        |t| (300.0 * t * 2.0 * PI).cos() as f32 * 0.3,
    );
    assert_matches_libopus(&out, &c_out);
}

// helper used by the println-based debug tests
#[allow(dead_code)]
fn _dbg() {
    try_decode(
        "dbg",
        &[0xbb, 0x03, 0xff, 0xfe, 0xff, 0xfe, 0xff, 0xfe],
        2880,
        1,
    );
}

// ---------------------------------------------------------------------------
// SILK PLC: a lost frame (ToC-only packet) must produce non-silence
// concealment, not zeros or a panic.
// ---------------------------------------------------------------------------
#[test]
fn silk_plc_produces_concealment_audio() {
    use opus_rs::{Application, OpusDecoder, OpusEncoder};
    let sr = 16000;
    let fs = 320;
    let mut enc = OpusEncoder::new(sr, 1, Application::Voip).unwrap();
    enc.bitrate_bps = 24000;

    // Encode several frames of a sine wave.
    let mut packets = Vec::new();
    for i in 0..6 {
        let input: Vec<f32> = (0..fs)
            .map(|j| {
                let t = (i * fs + j) as f64 / sr as f64;
                (440.0 * t * 2.0 * std::f64::consts::PI).sin() as f32 * 0.3
            })
            .collect();
        let mut pkt = vec![0u8; 512];
        let n = enc.encode(&input, fs, &mut pkt).unwrap();
        packets.push(pkt[..n].to_vec());
    }

    let mut dec = OpusDecoder::new(sr, 1).unwrap();
    for pkt in &packets {
        let mut buf = vec![0.0f32; 640];
        dec.decode(pkt, fs, &mut buf).unwrap();
    }

    // Decode a lost frame (single ToC byte, SILK WB 20ms → PLC).
    let lost_pkt = vec![packets[0][0] & 0xFC];
    let mut buf = vec![0.0f32; 640];
    let n = dec.decode(&lost_pkt, fs, &mut buf).unwrap();
    let rms = (buf[..n].iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    assert!(
        rms > 0.01,
        "SILK PLC should produce concealment audio, got rms={rms}"
    );
}
