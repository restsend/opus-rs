#![cfg(feature = "dnn")]
//! DRED of opus-rs (quantizer tables and schedule, Laplace coder, RDOVAE
//! encoder and decoder, `DREDEnc` and `OpusDRED`) against libopus: every
//! float must have the same bits and every payload the same bytes.
use opus_rs::dnn::dred::{coding, laplace, stats};
use opus_rs::range_coder::RangeCoder;
use opusref::dnn::Rng;
use opusref::dnn_dred as c;

#[test]
fn stats_tables_match() {
    unsafe {
        assert_eq!(stats::DRED_LATENT_QUANT_SCALES_Q8, c::dred_latent_quant_scales_q8);
        assert_eq!(stats::DRED_LATENT_DEAD_ZONE_Q8, c::dred_latent_dead_zone_q8);
        assert_eq!(stats::DRED_LATENT_R_Q8, c::dred_latent_r_q8);
        assert_eq!(stats::DRED_LATENT_P0_Q8, c::dred_latent_p0_q8);
        assert_eq!(stats::DRED_STATE_QUANT_SCALES_Q8, c::dred_state_quant_scales_q8);
        assert_eq!(stats::DRED_STATE_DEAD_ZONE_Q8, c::dred_state_dead_zone_q8);
        assert_eq!(stats::DRED_STATE_R_Q8, c::dred_state_r_q8);
        assert_eq!(stats::DRED_STATE_P0_Q8, c::dred_state_p0_q8);
    }
}

#[test]
fn compute_quantizer_matches() {
    for q0 in 0..16 {
        for dq in 0..8 {
            for qmax in q0..16 {
                for i in 0..60 {
                    let a = unsafe { c::ref_compute_quantizer(q0, dq, qmax, i) };
                    assert_eq!(a, coding::compute_quantizer(q0, dq, qmax, i), "{q0} {dq} {qmax} {i}");
                }
            }
        }
    }
}

/// Random Laplace-coded symbols: mostly small values, some up to ±40, with
/// `p0` and `decay` as DRED makes them from its Q8 tables, or any Q15
/// value. `p0` stays below 32767: at 32767 the negative sign has zero
/// probability and coding it leaves a zero range, which hangs both coders.
fn laplace_case(r: &mut Rng, n: usize) -> (Vec<i32>, Vec<u16>, Vec<u16>) {
    let mut vals = Vec::new();
    let mut p0 = Vec::new();
    let mut decay = Vec::new();
    for _ in 0..n {
        let v = match r.below(4) {
            0 => 0,
            1 => r.below(5) as i32 - 2,
            2 => r.below(17) as i32 - 8,
            _ => r.below(81) as i32 - 40,
        };
        vals.push(v);
        if r.below(2) == 0 {
            p0.push(((1 + r.below(254)) << 7) as u16);
            decay.push(((1 + r.below(254)) << 7) as u16);
        } else {
            p0.push(1 + r.below(32766) as u16);
            decay.push(r.below(32768) as u16);
        }
    }
    (vals, p0, decay)
}

#[test]
fn laplace_p0_matches() {
    let mut r = Rng(11);
    for case in 0..400 {
        let n = 1 + r.below(200) as usize;
        let (vals, p0, decay) = laplace_case(&mut r, n);
        let cap = [8usize, 40, 200, 1000][case % 4];
        let cres = c::laplace_p0_encode(&vals, &p0, &decay, cap);
        let mut enc = RangeCoder::new_encoder(cap as u32);
        for i in 0..n {
            laplace::ec_laplace_encode_p0(&mut enc, vals[i], p0[i], decay[i]);
        }
        let tell = enc.tell();
        enc.done();
        let Some((cbuf, ctell)) = cres else {
            assert_ne!(enc.error, 0, "case {case}: C failed, Rust did not");
            continue;
        };
        assert_eq!(enc.error, 0, "case {case}");
        assert_eq!(ctell, tell, "case {case}");
        assert_eq!(cbuf, enc.buf[..cap].to_vec(), "case {case}");

        // Decode the payload, and also a corrupted copy of it.
        let mut bytes = cbuf.clone();
        for pass in 0..2 {
            let (cv, ct) = c::laplace_p0_decode(&p0, &decay, &bytes);
            let mut dec = RangeCoder::new_decoder(&bytes);
            let rv: Vec<i32> = (0..n).map(|i| laplace::ec_laplace_decode_p0(&mut dec, p0[i], decay[i])).collect();
            assert_eq!(cv, rv, "case {case} pass {pass}");
            assert_eq!(ct, dec.tell(), "case {case} pass {pass}");
            if pass == 0 {
                assert_eq!(cv, vals, "case {case}: round trip");
            }
            for b in bytes.iter_mut() {
                *b = r.next_u32() as u8;
            }
        }
    }
}

