//! ec_dec_uint (celt/entdec.c:233-236) decodes a value past its range as
//! the largest one and flags an error; a corrupt CELT packet reaches that in
//! its PVQ indices. opus-rs must decode the same shapes.
use opus_rs::OpusDecoder;
use opusref::*;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn out_of_range_uint_decodes_as_the_largest() {
    // A random CELT packet (super-wideband 5 ms ToC) with such an index.
    let p = unhex(
        "e8d8b5e1d172cf1a4442773c41abe4823ad99c5329330fb6036350a607189505bcdab834e2abef9ed3445da6ed1e58cf19a75e1986452ed3250c7a7382571f2a068ce542c2ce2787456fef3825485e5139627828b98899a8a8b43ca15175d608ba",
    );
    let mut rs = OpusDecoder::new(48000, 1).unwrap();
    let mut err = 0;
    let c = unsafe { opus_decoder_create(48000, 1, &mut err) };
    let (mut a, mut b) = (vec![0f32; 5760], vec![0f32; 5760]);
    let n = rs.decode(&p, 5760, &mut a).unwrap();
    let m = unsafe { opus_decode_float(c, p.as_ptr(), p.len() as i32, b.as_mut_ptr(), 5760, 0) };
    unsafe { opus_decoder_destroy(c) };
    assert_eq!(n as i32, m);
    let d = a[..n]
        .iter()
        .zip(&b[..n])
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
    assert!(d <= 4e-6, "max difference {d}");
}
