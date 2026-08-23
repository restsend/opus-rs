//! Reproduction tests for GitHub issues #5, #6, #10.
//!
//! Each test mirrors the minimal reproducer from the corresponding issue.

use opus_rs::{Application, OpusDecoder, OpusEncoder};

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
    a.extend(std::iter::repeat(0xAA).take(200)); // frame 1
    a.extend(std::iter::repeat(0xBB).take(50)); // frame 2
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
    b.extend(std::iter::repeat(0xAA).take(300));
    b.extend(std::iter::repeat(0xBB).take(50));
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
    c.extend(std::iter::repeat(0xAA).take(100));
    c.extend(std::iter::repeat(0xBB).take(50));
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
// ---------------------------------------------------------------------------
#[test]
fn issue_7_stereo_silk_channels_differ() {
    let sr = 16000;
    let ch = 2;
    let fs = 320; // 20ms at 16kHz

    let mut enc = OpusEncoder::new(sr, ch, Application::Voip).unwrap();
    enc.bitrate_bps = 32000;

    // Left = 440 Hz tone, Right = silence. Different per channel.
    let mut pcm = vec![0.0f32; fs * ch];
    for i in 0..fs {
        let t = i as f64 / sr as f64;
        let val = (440.0 * t * 2.0 * std::f64::consts::PI).sin() as f32 * 0.3;
        pcm[i * 2] = val; // L
        pcm[i * 2 + 1] = 0.0; // R (silence)
    }

    let mut packet = vec![0u8; 400];
    let n = enc.encode(&pcm, fs, &mut packet).unwrap();

    let mut dec = OpusDecoder::new(sr, ch).unwrap();
    let mut out = vec![0.0f32; fs * ch];
    let decoded = dec.decode(&packet[..n], fs, &mut out).unwrap();
    assert_eq!(decoded, fs);

    // Verify L and R are NOT identical (old code replicated mono → L==R).
    let mut l_max = 0.0f32;
    let mut r_max = 0.0f32;
    for i in 0..fs {
        l_max = l_max.max(out[i * 2].abs());
        r_max = r_max.max(out[i * 2 + 1].abs());
    }
    // Left should have significant energy.
    assert!(
        l_max > 0.01,
        "Left channel should have energy, got max={l_max}"
    );
    // Right should differ significantly from Left (not a mono copy).
    // With M/S stereo and different L/R, R will not be zero but should be
    // substantially different from L.
    let mut l_minus_r = 0.0f32;
    for i in 0..fs {
        l_minus_r += (out[i * 2] - out[i * 2 + 1]).abs();
    }
    assert!(
        l_minus_r / fs as f32 > 0.005,
        "L and R should differ (M/S decoding), got avg diff={}",
        l_minus_r / fs as f32
    );
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
