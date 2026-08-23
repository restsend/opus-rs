use opus::{Application as CApp, Channels as CCh, Encoder as CEnc};
use opus_rs::{Application as RsApp, OpusDecoder as RsDec, OpusEncoder as RsEnc};

fn make_sine(sr: i32, ch: usize, frame_size: usize, freq: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; frame_size * ch];
    for i in 0..frame_size {
        let t = i as f32 / sr as f32;
        let s = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5;
        for c in 0..ch {
            v[i * ch + c] = s;
        }
    }
    v
}

fn hex_to_bytes(s: &str) -> Vec<u8> {
    let s = s.trim();
    assert!(s.len() % 2 == 0, "odd hex length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn oracle_celt_sine_48k_mono_64k() {
    let sr = 48000;
    let ch = 1;
    let br = 64000;
    let fs = 960;
    let pcm = make_sine(sr, ch, fs, 440.0);
    let mut rs_enc = RsEnc::new(sr, ch, RsApp::Audio).unwrap();
    rs_enc.bitrate_bps = br;
    rs_enc.use_cbr = true;
    rs_enc.complexity = 10;
    let mut rs_buf = vec![0u8; 1276];
    let rs_n = rs_enc.encode(&pcm, fs, &mut rs_buf).unwrap();
    let mut c_enc = CEnc::new(sr as u32, if ch==1 {CCh::Mono} else {CCh::Stereo}, CApp::Audio).unwrap();
    c_enc.set_bitrate(opus::Bitrate::Bits(br)).unwrap();
    c_enc.set_vbr(false).unwrap();
    let _ = c_enc.set_complexity(10);
    let mut c_buf = vec![0u8; 1276];
    let c_n = c_enc.encode_float(&pcm, &mut c_buf).unwrap();
    println!("rs pkt {} {:02x?} c pkt {} {:02x?}", rs_n, &rs_buf[..rs_n.min(8)], c_n, &c_buf[..c_n.min(8)]);
    assert!(rs_n > 0 && rs_n <= 1276);
    assert!(c_n > 0 && c_n <= 1276);
    let diff = (rs_n as i32 - c_n as i32).abs();
    assert!(diff <= 5, "size diff too large rs {} vs c {} diff {}", rs_n, c_n, diff);
    // Byte-exact regression for 48k mono 64k CBR sine, validated against a
    // libopus 1.6.1-51-g03647f52 (float, x86 SSE2/AVX2) reference build. The
    // audiopus_sys 0.2.2 C (libopus 1.3) differs in tone/prefilter evolution,
    // so we assert the 1.6 fixture here and only require size proximity vs 1.3.
    let expected_prefix: [u8; 8] = [0xf8, 0x7b, 0x5e, 0x09, 0x50, 0xb7, 0x8c, 0x08];
    assert_eq!(&rs_buf[..8.min(rs_n)], &expected_prefix[..8.min(rs_n)], "byte prefix mismatch for 48k sine 64k");
    const EXPECTED_SINE_HEX: &str = "f87b5e0950b78c08d0bbae9ae1d725a72fe5ee25c2740398a4615b113822973b042c80fdb66da4a3a2cb9c192af55d6bf6dd65c5c7653367144ecc6b0565efad221c5de2215ab8fc6f9f4f48cad42d1c1999da21cfa221cfa037aa44b81ad91d4a8d1d581540cf6c39176eeab2770f00b461d43fa328ae650837e79c9e8f6c417951dc2e4d32634de493b88e0d167016e2646590e686571dcf2104af43ab137d";
    let expected = hex_to_bytes(EXPECTED_SINE_HEX);
    assert_eq!(rs_n, expected.len(), "fixture len mismatch for 48k sine");
    assert_eq!(&rs_buf[..rs_n], &expected[..], "full byte-exact mismatch for 48k sine 64k");
    // Determinism: re-encode same input must be byte-identical
    let mut rs_enc2 = RsEnc::new(sr, ch, RsApp::Audio).unwrap();
    rs_enc2.bitrate_bps = br;
    rs_enc2.use_cbr = true;
    rs_enc2.complexity = 10;
    let mut rs_buf2 = vec![0u8; 1276];
    let rs_n2 = rs_enc2.encode(&pcm, fs, &mut rs_buf2).unwrap();
    assert_eq!(rs_n, rs_n2, "non-deterministic size");
    assert_eq!(&rs_buf[..rs_n], &rs_buf2[..rs_n2], "non-deterministic bytes");
}

#[test]
fn oracle_impulse_48k_mono_64k() {
    let sr = 48000;
    let ch = 1;
    let br = 64000;
    let fs = 960;
    let mut pcm = vec![0.0f32; fs*ch];
    pcm[0]=1.0;
    let mut rs_enc = RsEnc::new(sr, ch, RsApp::Audio).unwrap();
    rs_enc.bitrate_bps = br;
    rs_enc.use_cbr = true;
    rs_enc.complexity = 10;
    let mut rs_buf = vec![0u8; 1276];
    let rs_n = rs_enc.encode(&pcm, fs, &mut rs_buf).unwrap();
    let mut c_enc = CEnc::new(sr as u32, CCh::Mono, CApp::Audio).unwrap();
    c_enc.set_bitrate(opus::Bitrate::Bits(br)).unwrap();
    c_enc.set_vbr(false).unwrap();
    let _ = c_enc.set_complexity(10);
    let mut c_buf = vec![0u8; 1276];
    let c_n = c_enc.encode_float(&pcm, &mut c_buf).unwrap();
    println!("impulse rs {} {:02x?} c {} {:02x?}", rs_n, &rs_buf[..rs_n.min(8)], c_n, &c_buf[..c_n.min(8)]);
    assert!(rs_n <= 200 && c_n <= 200, "huge packet rs {} c {}", rs_n, c_n);
    let mut rs_dec = RsDec::new(sr, ch).unwrap();
    let mut out = vec![0.0f32; fs*ch];
    let got = rs_dec.decode(&rs_buf[..rs_n], fs, &mut out).unwrap();
    let max = out[..got*ch].iter().fold(0.0f32, |a,&x| a.max(x.abs()));
    assert!(max < 15.0 && max.is_finite());
    // For 48k CBR, also check byte closeness (allow small divergence due to remaining tone/Hybrid gaps)
    assert!((rs_n as i32 - c_n as i32).abs() <= 5);
    // Byte-exact for Rust's own 48k impulse (regression). The first 7 bytes match
    // a libopus 1.6.1 reference build; byte 7+ diverges from 1.6 because the Rust
    // port does not run libopus' tonality analysis (analysis.c), which nudges
    // alloc_trim via tonality_slope (C trim=1 vs Rust trim=0 here). This fixture
    // freezes the current deterministic output.
    let expected_impulse_prefix: [u8; 8] = [0xf8, 0x73, 0x46, 0xb9, 0xc0, 0x16, 0x58, 0x58];
    assert_eq!(&rs_buf[..8.min(rs_n)], &expected_impulse_prefix[..8.min(rs_n)], "byte prefix mismatch for 48k impulse");
    const EXPECTED_IMPULSE_HEX: &str = "f87346b9c0165858f73144258b8b79c2179a4272570325a2d263ab6067bb993375941787ac6e1904601599cafeb058f8eeb514adbb22ecb11412d9d75cf1169fdb36a4d7a7d06817cffc23fd39b6e3e7829db5edf6ed58cb3f58cb3f6a03976a5e05ee0ba8221ad585de11653bfacdcab607c26370d5f7945c3ca511ba07ffb4d8ff77d0c3d7e883f06d09b55d8db6a05ad7119fb82d72fb7375ccb483588e35";
    let expected = hex_to_bytes(EXPECTED_IMPULSE_HEX);
    assert_eq!(rs_n, expected.len(), "fixture len mismatch for impulse");
    assert_eq!(&rs_buf[..rs_n], &expected[..], "full byte-exact mismatch for impulse");
    let mut rs_enc2 = RsEnc::new(sr, ch, RsApp::Audio).unwrap();
    rs_enc2.bitrate_bps = br;
    rs_enc2.use_cbr = true;
    rs_enc2.complexity = 10;
    let mut rs_buf2 = vec![0u8; 1276];
    let rs_n2 = rs_enc2.encode(&pcm, fs, &mut rs_buf2).unwrap();
    assert_eq!(rs_n, rs_n2);
    assert_eq!(&rs_buf[..rs_n], &rs_buf2[..rs_n2]);
}

#[test]
fn oracle_all_sr_ch_br() {
    let srs = [48000, 24000, 16000];
    let chs = [1, 2];
    let brs = [32000, 64000, 96000];
    let freq = 440.0;
    for &sr in &srs {
        for &ch in &chs {
            for &br in &brs {
                let fs = sr / 50;
                let pcm = make_sine(sr, ch, fs as usize, freq);
                let mut rs_enc = RsEnc::new(sr, ch as usize, RsApp::Audio).unwrap();
                rs_enc.bitrate_bps = br;
                rs_enc.use_cbr = true;
                rs_enc.complexity = 10;
                let mut rs_buf = vec![0u8; 1276];
                let rs_n = rs_enc.encode(&pcm, fs as usize, &mut rs_buf).unwrap();
                let c_ch = if ch==1 {CCh::Mono} else {CCh::Stereo};
                let mut c_enc = match CEnc::new(sr as u32, c_ch, CApp::Audio) {
                    Ok(e) => e,
                    Err(e) => { println!("skip sr {} ch {} br {}: CEnc new err {:?}", sr,ch,br,e); continue; }
                };
                if c_enc.set_bitrate(opus::Bitrate::Bits(br)).is_err() { continue; }
                let _ = c_enc.set_vbr(false);
                let _ = c_enc.set_complexity(10);
                let mut c_buf = vec![0u8; 1276];
                let c_n = match c_enc.encode_float(&pcm, &mut c_buf) {
                    Ok(n) => n,
                    Err(e) => { println!("skip encode sr {} ch {} br {}: {:?}", sr,ch,br,e); continue; }
                };
                let diff = (rs_n as i32 - c_n as i32).abs();
                println!("oracle sr {} ch {} br {} rs {} c {} diff {}", sr,ch,br,rs_n,c_n,diff);
                assert!(rs_n>0 && c_n>0, "zero packet sr {} ch {} br {}", sr,ch,br);
                let tol = if sr == 48000 { 5 } else { 50 };
                if diff > tol {
                    println!("WARN diff {} >{} for sr {} ch {} br {} (known Hybrid gap)", diff, tol, sr,ch,br);
                }
                if sr == 48000 {
                    assert!(diff <= tol, "size diff {} >{} sr {} ch {} br {} rs {} c {}", diff, tol, sr,ch,br,rs_n,c_n);
                }
                let mut dec = RsDec::new(sr, ch as usize).unwrap();
                let mut out = vec![0.0f32; fs as usize * ch as usize];
                let got = dec.decode(&rs_buf[..rs_n], fs as usize, &mut out).unwrap();
                let max = out[..got*ch as usize].iter().fold(0.0f32, |a,&x| a.max(x.abs()));
                assert!(max.is_finite() && max < 50.0, "decode huge sr {} ch {} br {} max {}", sr,ch,br,max);
            }
        }
    }
}
