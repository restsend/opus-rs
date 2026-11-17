//! Packets opus_packet_parse_impl rejects: a frame of more than 1275 bytes
//! (its size is not coded) and a code-3 packet longer than 120 ms. opus-rs
//! must reject the same packets.
use opus_rs::OpusDecoder;
use opusref::*;

/// (opus-rs accepted, libopus result) for one packet on fresh decoders.
fn both(p: &[u8]) -> (bool, i32) {
    let mut rs = OpusDecoder::new(48000, 1).unwrap();
    let mut err = 0;
    let c = unsafe { opus_decoder_create(48000, 1, &mut err) };
    let mut out = vec![0f32; 5760 * 2];
    let r = rs.decode(p, 5760 * 2, &mut out).is_ok();
    let n =
        unsafe { opus_decode_float(c, p.as_ptr(), p.len() as i32, out.as_mut_ptr(), 5760 * 2, 0) };
    unsafe { opus_decoder_destroy(c) };
    (r, n)
}

#[test]
fn a_frame_over_1275_bytes_is_invalid() {
    // CELT FB 20 ms, code 0, 1300 bytes of payload.
    let mut p = vec![0xf8u8];
    p.extend((0..1300).map(|i| (i * 37) as u8));
    let (rs, c) = both(&p);
    assert!(c < 0, "libopus rejects it");
    assert!(!rs, "opus-rs rejects it");
}

#[test]
fn a_code_3_packet_over_120_ms_is_invalid() {
    // CELT FB 20 ms, code 3, 7 CBR frames of 10 bytes: 140 ms.
    let mut p = vec![0xfbu8, 7];
    p.extend((0..70).map(|i| (i * 53 + 1) as u8));
    let (rs, c) = both(&p);
    assert!(c < 0, "libopus rejects it");
    assert!(!rs, "opus-rs rejects it");
}
