use opus_rs::{Application, OpusDecoder, OpusEncoder};

fn max_abs(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

#[test]
fn vbr_near_silence_192k_not_huge() {
    // PCM that previously generated fc 7f fd f8 and decoded to ~1.7e5 in libopus.
    // With the C-faithful delay ring + dc_reject the near-silence is not treated
    // as digital silence, so the VBR packet is large (fc 7f fd padding header),
    // but it must decode back to near-silence without a burst. (libopus 1.6 also
    // emits a large fc-prefixed VBR packet here; its target is ~766 bytes while
    // the Rust's simplified compute_vbr targets ~480 — a known VBR gap.)
    let frame_size = 960;
    let channels = 2;
    let mut pcm = vec![0.0f32; frame_size * channels];
    for i in 0..frame_size {
        let v = 1e-06 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin();
        for c in 0..channels {
            pcm[i * channels + c] = v;
        }
    }
    let mut enc = OpusEncoder::new(48000, channels, Application::Audio).unwrap();
    enc.bitrate_bps = 192000;
    enc.use_cbr = false;
    let mut out = vec![0u8; 1276];
    let n = enc.encode(&pcm, frame_size, &mut out).unwrap();
    assert!(n > 0 && n <= 1276);
    let mut dec = OpusDecoder::new(48000, channels).unwrap();
    let mut pcm_out = vec![0.0f32; frame_size * channels];
    let decoded = dec.decode(&out[..n], frame_size, &mut pcm_out).unwrap();
    assert!(pcm_out.iter().all(|x| x.is_finite()), "non-finite after decode");
    let ma = max_abs(&pcm_out);
    assert!(ma < 10.0, "max_abs {} too large for near-silence (expected <10)", ma);
    assert!(decoded > 0, "decode returned 0");
}

#[test]
fn high_bitrate_matrix_48k_stereo() {
    // 20ms 48k stereo tone at 0.3 amp 440Hz, across bitrates
    for &kbps in &[128, 160, 176, 192, 256] {
        let frame_size = 960;
        let channels = 2;
        let mut pcm = vec![0.0f32; frame_size * channels];
        for i in 0..frame_size {
            let v = 0.3 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin();
            for c in 0..channels {
                pcm[i * channels + c] = v + (c as f32 * 0.001);
            }
        }
        let mut enc = OpusEncoder::new(48000, channels, Application::Audio).unwrap();
        enc.bitrate_bps = kbps * 1000;
        enc.use_cbr = false;
        let mut out = vec![0u8; 1276];
        let n = enc.encode(&pcm, frame_size, &mut out).unwrap();
        let mut dec = OpusDecoder::new(48000, channels).unwrap();
        let mut pcm_out = vec![0.0f32; frame_size * channels];
        let res = dec.decode(&out[..n], frame_size, &mut pcm_out);
        assert!(res.is_ok(), "decode failed at {} kbps: {:?}", kbps, res);
        assert!(pcm_out.iter().all(|x| x.is_finite()), "non-finite at {} kbps", kbps);
        let ma = max_abs(&pcm_out);
        assert!(ma < 10.0, "max_abs {} too large at {} kbps", ma, kbps);
        assert!(ma < 1e5, "huge amplitude at {} kbps", kbps);
    }
}

#[test]
fn per_packet_fresh_decoder_192k() {
    // Same PCM as high bitrate, but decode each packet with a fresh decoder
    let frame_size = 960;
    let channels = 2;
    let mut pcm = vec![0.0f32; frame_size * channels];
    for i in 0..frame_size {
        let v = 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin();
        for c in 0..channels {
            pcm[i * channels + c] = v;
        }
    }
    let mut enc = OpusEncoder::new(48000, channels, Application::Audio).unwrap();
    enc.bitrate_bps = 192000;
    enc.use_cbr = false;
    let mut out = vec![0u8; 1276];
    let n = enc.encode(&pcm, frame_size, &mut out).unwrap();
    let pkt = out[..n].to_vec();
    // Decode with fresh decoder per packet (single packet)
    let mut dec = OpusDecoder::new(48000, channels).unwrap();
    let mut pcm_out = vec![0.0f32; frame_size * channels];
    let res = dec.decode(&pkt, frame_size, &mut pcm_out).unwrap();
    assert!(res > 0);
    assert!(pcm_out.iter().all(|x| x.is_finite()));
    assert!(max_abs(&pcm_out) < 10.0);
}
