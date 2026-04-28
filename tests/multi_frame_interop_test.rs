use opus_rs::{Application, OpusDecoder, OpusEncoder};

fn make_sine(freq: f64, sample_rate: i32, frame_size: usize, channels: usize, seed: usize) -> Vec<f32> {
    (0..frame_size * channels)
        .map(|i| {
            let t = (seed * frame_size + i / channels) as f64 / sample_rate as f64;
            let sample = (freq * t * 2.0 * std::f64::consts::PI).sin() as f32;
            sample * 0.3
        })
        .collect()
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|&x| x * x).sum();
    (sum / samples.len() as f32).sqrt()
}

/// Encode one frame, return full packet bytes
fn encode_full(encoder: &mut OpusEncoder, pcm: &[f32], frame_size: usize, buf_size: usize) -> Vec<u8> {
    let mut packet = vec![0u8; buf_size];
    let n = encoder.encode(pcm, frame_size, &mut packet).unwrap();
    packet.truncate(n);
    packet
}

/// Build baseline: decode two single-frame packets sequentially through one decoder.
/// Returns RMS of each frame's output.
fn baseline_two(
    decoder: &mut OpusDecoder,
    pkt0: &[u8],
    pkt1: &[u8],
    frame_size: usize,
    channels: usize,
) -> (f32, f32) {
    let mut out0 = vec![0.0f32; frame_size * channels];
    decoder.decode(pkt0, frame_size, &mut out0).unwrap();
    let rms0 = rms(&out0);
    let mut out1 = vec![0.0f32; frame_size * channels];
    decoder.decode(pkt1, frame_size, &mut out1).unwrap();
    let rms1 = rms(&out1);
    (rms0, rms1)
}

#[test]
fn test_celt_multi_frame_code_0_and_1() {
    let sr = 48000;
    let ch = 1;
    let fs = 960;

    let mut enc = OpusEncoder::new(sr, ch, Application::RestrictedLowDelay).unwrap();
    enc.bitrate_bps = 64000;

    let pcm0 = make_sine(440.0, sr, fs, ch, 0);
    let pcm1 = make_sine(440.0, sr, fs, ch, 1);

    let pkt0 = encode_full(&mut enc, &pcm0, fs, 400);
    let pkt1 = encode_full(&mut enc, &pcm1, fs, 400);
    assert_eq!(pkt0[0] & 0x03, 0, "expected code 0");
    assert_eq!(pkt1[0] & 0x03, 0, "expected code 0");

    // Baseline: decode sequentially through one decoder
    let mut dec = OpusDecoder::new(sr, ch).unwrap();
    let (bl_rms0, bl_rms1) = baseline_two(&mut dec, &pkt0, &pkt1, fs, ch);
    assert!(bl_rms0 > 0.01);
    assert!(bl_rms1 > 0.01);

    // Code 1 packet from the two payloads
    let payload0 = &pkt0[1..];
    let payload1 = &pkt1[1..];
    let toc1 = (pkt0[0] & 0xFC) | 0x01;
    let mut c1 = vec![toc1];
    c1.extend_from_slice(payload0);
    c1.extend_from_slice(payload1);

    let mut dec_mf = OpusDecoder::new(sr, ch).unwrap();
    let mut out = vec![0.0f32; fs * 2];
    dec_mf.decode(&c1, fs * 2, &mut out).unwrap();
    let r0 = rms(&out[..fs]);
    let r1 = rms(&out[fs..]);
    assert!((r0 / bl_rms0 - 1.0).abs() < 0.02, "frame 0: {:.6} vs {:.6}", r0, bl_rms0);
    assert!((r1 / bl_rms1 - 1.0).abs() < 0.02, "frame 1: {:.6} vs {:.6}", r1, bl_rms1);
}

#[test]
fn test_celt_multi_frame_code_2() {
    let sr = 48000;
    let ch = 1;
    let fs = 960;

    let mut enc = OpusEncoder::new(sr, ch, Application::RestrictedLowDelay).unwrap();
    enc.bitrate_bps = 64000;

    let pcm0 = make_sine(440.0, sr, fs, ch, 0);
    let pcm1 = make_sine(880.0, sr, fs, ch, 1);
    let pkt0 = encode_full(&mut enc, &pcm0, fs, 400);
    let pkt1 = encode_full(&mut enc, &pcm1, fs, 400);

    let mut dec = OpusDecoder::new(sr, ch).unwrap();
    let (bl_rms0, bl_rms1) = baseline_two(&mut dec, &pkt0, &pkt1, fs, ch);

    // Code 2: two unequal frames
    let payload0 = &pkt0[1..];
    let payload1 = &pkt1[1..];
    let toc2 = (pkt0[0] & 0xFC) | 0x02;
    let first_len = payload0.len();
    let mut c2 = vec![toc2];
    if first_len < 128 {
        c2.push(first_len as u8);
    } else {
        c2.push(0x80 | (first_len >> 8) as u8);
        c2.push(first_len as u8);
    }
    c2.extend_from_slice(payload0);
    c2.extend_from_slice(payload1);

    let mut dec_mf = OpusDecoder::new(sr, ch).unwrap();
    let mut out = vec![0.0f32; fs * 2];
    dec_mf.decode(&c2, fs * 2, &mut out).unwrap();
    let r0 = rms(&out[..fs]);
    let r1 = rms(&out[fs..]);
    assert!((r0 / bl_rms0 - 1.0).abs() < 0.02, "frame 0: {:.6} vs {:.6}", r0, bl_rms0);
    assert!((r1 / bl_rms1 - 1.0).abs() < 0.02, "frame 1: {:.6} vs {:.6}", r1, bl_rms1);
}

#[test]
fn test_celt_multi_frame_code_3_with_padding() {
    let sr = 48000;
    let ch = 1;
    let fs = 960;

    let mut enc = OpusEncoder::new(sr, ch, Application::RestrictedLowDelay).unwrap();
    enc.bitrate_bps = 64000;

    let pcm0 = make_sine(440.0, sr, fs, ch, 0);
    let pcm1 = make_sine(660.0, sr, fs, ch, 1);
    let pkt0 = encode_full(&mut enc, &pcm0, fs, 400);
    let pkt1 = encode_full(&mut enc, &pcm1, fs, 400);

    let mut dec = OpusDecoder::new(sr, ch).unwrap();
    let (bl_rms0, bl_rms1) = baseline_two(&mut dec, &pkt0, &pkt1, fs, ch);

    // Code 3 with padding: 2 equal-sized frames
    let payload0 = &pkt0[1..];
    let payload1 = &pkt1[1..];
    let toc3 = (pkt0[0] & 0xFC) | 0x03;
    let max_len = payload0.len().max(payload1.len());
    let mut p0 = payload0.to_vec();
    let mut p1 = payload1.to_vec();
    p0.resize(max_len, 0);
    p1.resize(max_len, 0);

    let mut c3 = vec![toc3];
    c3.push(0x42);
    c3.push(0x00);
    c3.extend_from_slice(&p0);
    c3.extend_from_slice(&p1);

    let mut dec_mf = OpusDecoder::new(sr, ch).unwrap();
    let mut out = vec![0.0f32; fs * 2];
    dec_mf.decode(&c3, fs * 2, &mut out).unwrap();
    let r0 = rms(&out[..fs]);
    let r1 = rms(&out[fs..]);
    assert!((r0 / bl_rms0 - 1.0).abs() < 0.05, "frame 0: {:.6} vs {:.6}", r0, bl_rms0);
    assert!((r1 / bl_rms1 - 1.0).abs() < 0.05, "frame 1: {:.6} vs {:.6}", r1, bl_rms1);
}

#[test]
fn test_celt_multi_frame_code_3_self_delimiting() {
    let sr = 48000;
    let ch = 1;
    let fs = 960;

    let mut enc = OpusEncoder::new(sr, ch, Application::RestrictedLowDelay).unwrap();
    enc.bitrate_bps = 64000;

    let pcm0 = make_sine(440.0, sr, fs, ch, 0);
    let pcm1 = make_sine(880.0, sr, fs, ch, 1);
    let pkt0 = encode_full(&mut enc, &pcm0, fs, 400);
    let pkt1 = encode_full(&mut enc, &pcm1, fs, 400);

    let mut dec = OpusDecoder::new(sr, ch).unwrap();
    let (bl_rms0, bl_rms1) = baseline_two(&mut dec, &pkt0, &pkt1, fs, ch);

    let payload0 = &pkt0[1..];
    let payload1 = &pkt1[1..];
    let toc3 = (pkt0[0] & 0xFC) | 0x03;
    let len0 = payload0.len();
    let mut c3 = vec![toc3];
    c3.push(0x02);
    if len0 < 128 {
        c3.push(len0 as u8);
    } else {
        c3.push(0x80 | (len0 >> 8) as u8);
        c3.push(len0 as u8);
    }
    c3.extend_from_slice(payload0);
    c3.extend_from_slice(payload1);

    let mut dec_mf = OpusDecoder::new(sr, ch).unwrap();
    let mut out = vec![0.0f32; fs * 2];
    dec_mf.decode(&c3, fs * 2, &mut out).unwrap();
    let r0 = rms(&out[..fs]);
    let r1 = rms(&out[fs..]);
    assert!((r0 / bl_rms0 - 1.0).abs() < 0.02, "frame 0: {:.6} vs {:.6}", r0, bl_rms0);
    assert!((r1 / bl_rms1 - 1.0).abs() < 0.02, "frame 1: {:.6} vs {:.6}", r1, bl_rms1);
}

#[test]
fn test_silk_multi_frame_code_3_padding_roundtrip() {
    let sr = 16000;
    let ch = 1;
    let fs = 320;
    let target_size = 100usize;

    let mut enc = OpusEncoder::new(sr, ch, Application::Voip).unwrap();
    enc.bitrate_bps = 24000;
    enc.use_cbr = true;

    let mut dec = OpusDecoder::new(sr, ch).unwrap();

    for frame_idx in 0..3 {
        let pcm = make_sine(440.0, sr, fs, ch, frame_idx);
        let mut packet = vec![0u8; target_size];
        let n = enc.encode(&pcm, fs, &mut packet).unwrap();
        packet.truncate(n);

        let code = packet[0] & 0x03;
        assert!(code == 0 || code == 3, "expected code 0 or 3, got {}", code);

        let mut out = vec![0.0f32; fs];
        let decoded = dec.decode(&packet, fs, &mut out).unwrap();
        assert_eq!(decoded, fs);
        assert!(rms(&out) > 0.01, "frame {} should have non-zero output", frame_idx);
    }
}

#[test]
fn test_hybrid_multi_frame_roundtrip() {
    let sr = 48000;
    let ch = 1;
    let fs = 960;

    let mut enc = OpusEncoder::new(sr, ch, Application::Audio).unwrap();
    enc.bitrate_bps = 64000;

    let pcm0 = make_sine(440.0, sr, fs, ch, 0);
    let pcm1 = make_sine(660.0, sr, fs, ch, 1);
    let pkt0 = encode_full(&mut enc, &pcm0, fs, 1000);
    let pkt1 = encode_full(&mut enc, &pcm1, fs, 1000);

    let mut dec = OpusDecoder::new(sr, ch).unwrap();
    let (bl_rms0, bl_rms1) = baseline_two(&mut dec, &pkt0, &pkt1, fs, ch);

    let payload0 = &pkt0[1..];
    let payload1 = &pkt1[1..];
    let toc1 = (pkt0[0] & 0xFC) | 0x01;
    let mut c1 = vec![toc1];
    c1.extend_from_slice(payload0);
    c1.extend_from_slice(payload1);

    let mut dec_mf = OpusDecoder::new(sr, ch).unwrap();
    let mut out = vec![0.0f32; fs * 2];
    dec_mf.decode(&c1, fs * 2, &mut out).unwrap();

    let r0 = rms(&out[..fs * ch]);
    let r1 = rms(&out[fs * ch..]);
    assert!((r0 / bl_rms0 - 1.0).abs() < 0.1, "frame 0: {:.6} vs {:.6}", r0, bl_rms0);
    assert!((r1 / bl_rms1 - 1.0).abs() < 0.1, "frame 1: {:.6} vs {:.6}", r1, bl_rms1);
}

#[test]
fn test_multi_frame_code_2_unequal_payloads() {
    let sr = 48000;
    let ch = 1;
    let fs = 960;

    let mut enc = OpusEncoder::new(sr, ch, Application::RestrictedLowDelay).unwrap();
    enc.bitrate_bps = 64000;

    let pcm_silence = vec![0.0f32; fs];
    let pcm_tone = make_sine(440.0, sr, fs, ch, 0);
    let pkt_sil = encode_full(&mut enc, &pcm_silence, fs, 400);
    // fresh encoder for tone to avoid state interference
    let mut enc2 = OpusEncoder::new(sr, ch, Application::RestrictedLowDelay).unwrap();
    enc2.bitrate_bps = 64000;
    let pkt_tone = encode_full(&mut enc2, &pcm_tone, fs, 400);

    // Baseline: both frames through same decoder
    let mut dec = OpusDecoder::new(sr, ch).unwrap();
    let (_, bl_tone) = baseline_two(&mut dec, &pkt_sil, &pkt_tone, fs, ch);

    // Code 2: silence first (smaller), tone second
    let payload_sil = &pkt_sil[1..];
    let payload_tone = &pkt_tone[1..];
    let toc2 = (pkt_sil[0] & 0xFC) | 0x02;
    let first_len = payload_sil.len();
    let mut c2 = vec![toc2];
    if first_len < 128 {
        c2.push(first_len as u8);
    } else {
        c2.push(0x80 | (first_len >> 8) as u8);
        c2.push(first_len as u8);
    }
    c2.extend_from_slice(payload_sil);
    c2.extend_from_slice(payload_tone);

    let mut dec_mf = OpusDecoder::new(sr, ch).unwrap();
    let mut out = vec![0.0f32; fs * 2];
    dec_mf.decode(&c2, fs * 2, &mut out).unwrap();

    let r_sil = rms(&out[..fs]);
    let r_tone = rms(&out[fs..]);
    assert!(r_sil < 0.001, "silence frame should be near zero, got {:.6}", r_sil);
    assert!((r_tone / bl_tone - 1.0).abs() < 0.02, "tone frame: {:.6} vs {:.6}", r_tone, bl_tone);
}

#[test]
fn test_multi_frame_all_codes_stereo() {
    let sr = 48000;
    let ch = 2;
    let fs = 960;

    let mut enc = OpusEncoder::new(sr, ch, Application::Audio).unwrap();
    enc.bitrate_bps = 96000;

    let pcm0 = make_sine(440.0, sr, fs, ch, 0);
    let pcm1 = make_sine(660.0, sr, fs, ch, 1);
    let pkt0 = encode_full(&mut enc, &pcm0, fs, 1000);
    let pkt1 = encode_full(&mut enc, &pcm1, fs, 1000);

    let mut dec = OpusDecoder::new(sr, ch).unwrap();
    let (bl_rms0, bl_rms1) = baseline_two(&mut dec, &pkt0, &pkt1, fs, ch);

    let payload0 = &pkt0[1..];
    let payload1 = &pkt1[1..];
    let base_toc = pkt0[0] & 0xFC;

    // Code 1
    let mut c1 = vec![base_toc | 0x01];
    c1.extend_from_slice(payload0);
    c1.extend_from_slice(payload1);
    let mut dec_mf = OpusDecoder::new(sr, ch).unwrap();
    let mut out = vec![0.0f32; fs * 2 * ch];
    dec_mf.decode(&c1, fs * 2, &mut out).unwrap();
    let r0 = rms(&out[..fs * ch]);
    let r1 = rms(&out[fs * ch..]);
    assert!((r0 / bl_rms0 - 1.0).abs() < 0.05, "C1 f0: {:.6} vs {:.6}", r0, bl_rms0);
    assert!((r1 / bl_rms1 - 1.0).abs() < 0.05, "C1 f1: {:.6} vs {:.6}", r1, bl_rms1);

    // Code 2
    let first_len = payload0.len();
    let mut c2 = vec![base_toc | 0x02];
    if first_len < 128 {
        c2.push(first_len as u8);
    } else {
        c2.push(0x80 | (first_len >> 8) as u8);
        c2.push(first_len as u8);
    }
    c2.extend_from_slice(payload0);
    c2.extend_from_slice(payload1);
    let mut dec_mf = OpusDecoder::new(sr, ch).unwrap();
    let mut out = vec![0.0f32; fs * 2 * ch];
    dec_mf.decode(&c2, fs * 2, &mut out).unwrap();
    assert!((rms(&out[..fs * ch]) / bl_rms0 - 1.0).abs() < 0.05, "C2 f0 mismatch");
    assert!((rms(&out[fs * ch..]) / bl_rms1 - 1.0).abs() < 0.05, "C2 f1 mismatch");

    // Code 3 with padding
    let max_len = payload0.len().max(payload1.len());
    let mut p0 = payload0.to_vec();
    let mut p1 = payload1.to_vec();
    p0.resize(max_len, 0);
    p1.resize(max_len, 0);
    let mut c3 = vec![base_toc | 0x03];
    c3.push(0x42);
    c3.push(0x00);
    c3.extend_from_slice(&p0);
    c3.extend_from_slice(&p1);
    let mut dec_mf = OpusDecoder::new(sr, ch).unwrap();
    let mut out = vec![0.0f32; fs * 2 * ch];
    dec_mf.decode(&c3, fs * 2, &mut out).unwrap();
    assert!((rms(&out[..fs * ch]) / bl_rms0 - 1.0).abs() < 0.05, "C3p f0 mismatch");
    assert!((rms(&out[fs * ch..]) / bl_rms1 - 1.0).abs() < 0.05, "C3p f1 mismatch");

    // Code 3 self-delimiting
    let len0 = payload0.len();
    let mut c3s = vec![base_toc | 0x03];
    c3s.push(0x02);
    if len0 < 128 {
        c3s.push(len0 as u8);
    } else {
        c3s.push(0x80 | (len0 >> 8) as u8);
        c3s.push(len0 as u8);
    }
    c3s.extend_from_slice(payload0);
    c3s.extend_from_slice(payload1);
    let mut dec_mf = OpusDecoder::new(sr, ch).unwrap();
    let mut out = vec![0.0f32; fs * 2 * ch];
    dec_mf.decode(&c3s, fs * 2, &mut out).unwrap();
    assert!((rms(&out[..fs * ch]) / bl_rms0 - 1.0).abs() < 0.05, "C3s f0 mismatch");
    assert!((rms(&out[fs * ch..]) / bl_rms1 - 1.0).abs() < 0.05, "C3s f1 mismatch");
}
