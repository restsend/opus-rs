//! A uniform symbol decoded above its range must be clamped, as libopus does.
//!
//! `ec_dec_uint` reads values of more than 8 bits as a range-coded top part
//! and raw low bits (RFC 6716 section 4.1.5). The two together can exceed
//! ft - 1 in a packet an encoder did not produce; libopus's decoder then
//! sets its error flag and returns ft - 1 (entdec.c). opus-rs returned the
//! larger value: a CELT split angle with 256 steps decoded as 257, past a
//! full split (16448 instead of 16384), and the rest of the frame decoded
//! differently from libopus. Differential decoding of mutated test vector
//! packets showed 294 such frames, all from this.

use opus::{Channels, Decoder as CDec};
use opus_rs::OpusDecoder;
use opus_rs::range_coder::RangeCoder;

#[test]
fn dec_uint_clamps_values_past_its_range() {
    let mut clamped = 0;
    for first in 0..=255u8 {
        for last in [0x00u8, 0x80, 0xff] {
            let data = [first, 0xff, 0xff, last];
            let mut rc = RangeCoder::new_decoder(&data);
            let v = rc.dec_uint(257);
            assert!(v <= 256, "{data:02x?}: {v}");
            if rc.error != 0 {
                assert_eq!(v, 256, "{data:02x?}");
                clamped += 1;
            }
        }
    }
    assert!(clamped > 0, "no input reached past the range");
}

/// A CELT packet (super-wideband, 10 ms) from a test vector with one bit
/// flipped, whose split angles reach past their range.
const PACKET: [u8; 112] = [
    0xd4, 0x82, 0x70, 0x26, 0x6e, 0xca, 0xec, 0x61, 0x51, 0xaa, 0x4f, 0x38, 0x8d, 0x86, 0x86, 0xf9,
    0xa1, 0x45, 0x5b, 0xc9, 0x81, 0x33, 0xe5, 0xc4, 0xb8, 0x57, 0x69, 0xaa, 0xdc, 0x6c, 0xac, 0x1b,
    0x67, 0x8d, 0x44, 0x3b, 0x5e, 0xe1, 0x64, 0x5b, 0xb0, 0xfc, 0xa3, 0xbf, 0xf9, 0xfc, 0x31, 0x0b,
    0x67, 0xae, 0x49, 0xac, 0x14, 0x3c, 0x46, 0xf1, 0xa1, 0x79, 0x20, 0xb6, 0x1e, 0xce, 0x4e, 0x8d,
    0x89, 0x8c, 0xb2, 0x8c, 0x8c, 0x9d, 0xf4, 0x14, 0xd8, 0x11, 0xd8, 0x55, 0x51, 0x6e, 0xc8, 0x70,
    0x1a, 0xbc, 0xb9, 0xfd, 0xab, 0x7e, 0x24, 0xde, 0x6f, 0xb2, 0xfd, 0x9d, 0xb1, 0xb7, 0x36, 0x54,
    0x96, 0x3f, 0xf9, 0x62, 0x14, 0xea, 0x81, 0xfd, 0x15, 0x18, 0x39, 0x7c, 0x05, 0xc6, 0x7a, 0x52,
];

#[test]
fn celt_frame_with_an_out_of_range_angle_decodes_like_libopus() {
    // Stereo decoders: mono decoders of a stereo stream treat phase
    // inversion differently in libopus by design (RFC 8251 section 10).
    let mut c_dec = CDec::new(48000, Channels::Stereo).unwrap();
    let mut rs_dec = OpusDecoder::new(48000, 2).unwrap();
    let (mut a, mut b) = (vec![0f32; 5760 * 2], vec![0f32; 5760 * 2]);
    let n = c_dec.decode_float(&PACKET, &mut a, false).unwrap();
    assert_eq!(rs_dec.decode(&PACKET, 5760, &mut b), Ok(n));
    let (mut s, mut e) = (0f64, 0f64);
    for (x, y) in a[..2 * n].iter().zip(&b[..2 * n]) {
        s += (*x as f64).powi(2);
        e += (*x as f64 - *y as f64).powi(2);
    }
    let snr = 10.0 * (s / e.max(1e-30)).log10();
    assert!(snr > 60.0, "{snr:.1} dB from libopus");
}
