//! The Laplace coder of the DRED latents: `ec_laplace_encode_p0` and
//! `ec_laplace_decode_p0` (libopus `celt/laplace.c`) with the 16-bit icdf
//! coder they use (`ec_enc_icdf16`, `ec_dec_icdf16`).

use crate::range_coder::RangeCoder;

/// `ec_enc_icdf16`: codes symbol `s` of the inverse cumulative table
/// `icdf` (total `1 << ftb`).
pub fn ec_enc_icdf16(enc: &mut RangeCoder, s: usize, icdf: &[u16], ftb: u32) {
    let r = enc.rng >> ftb;
    if s > 0 {
        enc.val = enc
            .val
            .wrapping_add(enc.rng.wrapping_sub(r.wrapping_mul(icdf[s - 1] as u32)));
        enc.rng = r.wrapping_mul((icdf[s - 1] as u32).wrapping_sub(icdf[s] as u32));
    } else {
        enc.rng = enc.rng.wrapping_sub(r.wrapping_mul(icdf[s] as u32));
    }
    // A zero-width interval at the bottom: only renormalizes.
    enc.encode(0, 1, 1);
}

/// `ec_dec_icdf16`.
pub fn ec_dec_icdf16(dec: &mut RangeCoder, icdf: &[u16], ftb: u32) -> usize {
    let mut s = dec.rng;
    let d = dec.val;
    let r = s >> ftb;
    let mut ret = 0;
    let mut t;
    loop {
        t = s;
        s = r.wrapping_mul(icdf[ret] as u32);
        if d >= s {
            break;
        }
        ret += 1;
    }
    dec.val = d.wrapping_sub(s);
    dec.rng = t.wrapping_sub(s);
    // Leaves `val` and `rng` as they are and renormalizes.
    dec.update(0, 1, 1);
    ret
}

/// The sign table of `ec_laplace_*_p0`: zero with probability `p0/32768`,
/// then positive and negative.
fn sign_icdf(p0: u16) -> [u16; 3] {
    let s0 = 32768u16.wrapping_sub(p0);
    [s0, s0 / 2, 0]
}

/// The magnitude table of `ec_laplace_*_p0`.
fn magnitude_icdf(decay: u16) -> [u16; 8] {
    let mut icdf = [0u16; 8];
    icdf[0] = (decay as i32).max(7) as u16;
    for i in 1..7 {
        let v = (icdf[i - 1] as i32 * decay as i32) >> 15;
        icdf[i] = v.max(7 - i as i32) as u16;
    }
    icdf
}

/// `ec_laplace_encode_p0`: codes `value` with the zero probability `p0`
/// and the decay `decay` (both Q15).
pub fn ec_laplace_encode_p0(enc: &mut RangeCoder, value: i32, p0: u16, decay: u16) {
    let s = if value == 0 {
        0
    } else if value > 0 {
        1
    } else {
        2
    };
    ec_enc_icdf16(enc, s, &sign_icdf(p0), 15);
    let mut value = value.wrapping_abs();
    if value != 0 {
        let icdf = magnitude_icdf(decay);
        value = value.wrapping_sub(1);
        loop {
            ec_enc_icdf16(enc, value.min(7) as usize, &icdf, 15);
            value -= 7;
            if value < 0 {
                break;
            }
        }
    }
}

/// `ec_laplace_decode_p0`.
pub fn ec_laplace_decode_p0(dec: &mut RangeCoder, p0: u16, decay: u16) -> i32 {
    let s: i32 = match ec_dec_icdf16(dec, &sign_icdf(p0), 15) {
        0 => return 0,
        1 => 1,
        _ => -1,
    };
    let icdf = magnitude_icdf(decay);
    let mut value = 1i32;
    loop {
        let v = ec_dec_icdf16(dec, &icdf, 15);
        value = value.wrapping_add(v as i32);
        if v != 7 {
            break;
        }
    }
    s.wrapping_mul(value)
}
