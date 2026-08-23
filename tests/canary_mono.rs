use opus_rs::{Application as RsApp, OpusDecoder as RsDec, OpusEncoder as RsEnc};

#[test]
fn mono_impulse_48k_64k_finite_small_packet() {
    let sr = 48000;
    let ch = 1;
    let br = 64000;
    let fs = 960; // 20ms @48k
    let mut pcm = vec![0.0f32; fs * ch];
    pcm[0] = 1.0;
    let mut enc = RsEnc::new(sr, ch, RsApp::Audio).unwrap();
    enc.bitrate_bps = br;
    let mut buf = vec![0u8; 1276];
    let n = enc.encode(&pcm, fs, &mut buf).unwrap();
    assert!(n > 0 && n <= 200, "packet huge {} bytes: {:02x?}", n, &buf[..n.min(8)]);
    let mut dec = RsDec::new(sr, ch).unwrap();
    let mut out = vec![0.0f32; fs * ch];
    let got = dec.decode(&buf[..n], fs, &mut out).unwrap();
    let max = out[..got * ch].iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    assert!(max.is_finite() && max < 15.0, "decode huge/finite max {} pkt {:?}", max, &buf[..n.min(8)]);
}
