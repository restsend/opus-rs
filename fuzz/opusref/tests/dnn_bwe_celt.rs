//! CELT-only frames after SILK wideband frames with the bandwidth extension
//! active: libopus main resets the extended mode to OSCE_MODE_CELT_ONLY and
//! decodes them (1.6.1 kept OSCE_MODE_SILK_BBWE, skipped the CELT decode and
//! wrote zeros). opus-rs must write the same into an output buffer that
//! holds earlier samples.
#![cfg(feature = "dnn")]
use opusref::dnn::speechlike;
use opusref::*;

const OPUS_SET_OSCE_BWE: i32 = 4054;

fn encode(app: i32, bitrate: i32, bandwidth: i32, frames: usize) -> Vec<Vec<u8>> {
    let sig = speechlike(16000 * frames / 50, 7);
    let pcm: Vec<f32> = (0..960 * frames).map(|i| sig[i / 3] / 32768.0).collect();
    let mut out = Vec::new();
    unsafe {
        let mut err = 0;
        let enc = opus_encoder_create(48000, 1, app, &mut err);
        assert_eq!(err, 0);
        opus_encoder_ctl(enc, SET_BITRATE, bitrate);
        opus_encoder_ctl(enc, SET_MAX_BANDWIDTH, bandwidth);
        for f in pcm.chunks_exact(960) {
            let mut pkt = [0u8; 1500];
            let len = opus_encode_float(enc, f.as_ptr(), 960, pkt.as_mut_ptr(), 1500);
            assert!(len > 0);
            out.push(pkt[..len as usize].to_vec());
        }
        opus_encoder_destroy(enc);
    }
    out
}

#[test]
fn celt_frames_after_bbwe_are_decoded() {
    let silk = encode(2048, 20000, 1103, 5);
    let celt = encode(2051, 48000, 1105, 5);
    assert!(silk.iter().all(|p| p[0] >> 3 == 9), "SILK WB 20 ms");
    assert!(celt.iter().all(|p| p[0] >> 3 == 31), "CELT FB 20 ms");
    let mut rs = opus_rs::OpusDecoder::new(48000, 1).unwrap();
    rs.set_complexity(4).unwrap();
    rs.set_osce_bwe(true);
    unsafe {
        let mut err = 0;
        let dec = opus_decoder_create(48000, 1, &mut err);
        opus_decoder_ctl(dec, SET_COMPLEXITY, 4);
        opus_decoder_ctl(dec, OPUS_SET_OSCE_BWE, 1);
        for (k, p) in silk.iter().chain(&celt).enumerate() {
            let mut a = vec![0.25f32; 960];
            let mut b = vec![0.25f32; 960];
            let na = opus_decode_float(dec, p.as_ptr(), p.len() as i32, a.as_mut_ptr(), 960, 0);
            let nb = rs.decode(p, 960, &mut b).unwrap();
            assert_eq!(na as usize, nb, "frame {k}");
            let d = a
                .iter()
                .zip(&b)
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            assert!(d <= 1e-6, "frame {k} differs by {d}");
        }
        opus_decoder_destroy(dec);
    }
}
