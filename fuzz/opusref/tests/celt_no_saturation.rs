//! The float build of libopus does not saturate CELT's synthesis or
//! deemphasis at SIG_SAT (its SATURATE is the identity), so an extreme
//! packet can decode beyond ±16384. opus-rs must give the same samples.
use opus_rs::OpusDecoder;
use opusref::*;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

#[test]
fn extreme_celt_frame_is_not_saturated() {
    // A random CELT FB 20 ms packet whose libopus output peaks at 27553.6.
    let p = unhex("f89ea8a5fcb9f6d22e62ce3a060868ae625a7e06ed2a38412a54a702beea93");
    let mut rs = OpusDecoder::new(48000, 1).unwrap();
    let mut err = 0;
    let c = unsafe { opus_decoder_create(48000, 1, &mut err) };
    let (mut a, mut b) = (vec![0f32; 960], vec![0f32; 960]);
    rs.decode(&p, 960, &mut a).unwrap();
    unsafe { opus_decode_float(c, p.as_ptr(), p.len() as i32, b.as_mut_ptr(), 960, 0) };
    unsafe { opus_decoder_destroy(c) };
    let peak = b.iter().fold(1f32, |m, v| m.max(v.abs()));
    let d = a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(d <= 4e-6 * peak, "max difference {d} (peak {peak})");
}
