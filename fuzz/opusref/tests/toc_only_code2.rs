//! A 1-byte packet with ToC code 2: opus_packet_parse_impl needs a size
//! byte for the first frame and rejects the packet. opus-rs must reject it
//! too, where it used to conceal two frames.
use opus_rs::OpusDecoder;
use opusref::*;

#[test]
fn a_toc_only_code_2_packet_is_invalid() {
    let p = [0x0au8]; // SILK NB 20 ms, code 2
    let mut rs = OpusDecoder::new(16000, 1).unwrap();
    let mut err = 0;
    let c = unsafe { opus_decoder_create(16000, 1, &mut err) };
    let mut out = vec![0f32; 1920];
    let r = rs.decode(&p, 1920, &mut out);
    let n = unsafe { opus_decode_float(c, p.as_ptr(), 1, out.as_mut_ptr(), 1920, 0) };
    unsafe { opus_decoder_destroy(c) };
    assert!(n < 0, "libopus rejects it");
    assert!(r.is_err(), "opus-rs rejects it");
}
