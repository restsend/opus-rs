use crate::modes::CeltMode;
use crate::range_coder::{BITRES, RangeCoder};

pub const PRED_COEF: [f32; 4] = [
    29440.0 / 32768.0,
    26112.0 / 32768.0,
    21248.0 / 32768.0,
    16384.0 / 32768.0,
];
pub const BETA_COEF: [f32; 4] = [
    30147.0 / 32768.0,
    22282.0 / 32768.0,
    12124.0 / 32768.0,
    6554.0 / 32768.0,
];
pub const BETA_INTRA: f32 = 4915.0 / 32768.0;

pub const E_PROB_MODEL: [[[u8; 42]; 2]; 4] = [
    [
        [
            72, 127, 65, 129, 66, 128, 65, 128, 64, 128, 62, 128, 64, 128, 64, 128, 92, 78, 92, 79,
            92, 78, 90, 79, 116, 41, 115, 40, 114, 40, 132, 26, 132, 26, 145, 17, 161, 12, 176, 10,
            177, 11,
        ],
        [
            24, 179, 48, 138, 54, 135, 54, 132, 53, 134, 56, 133, 55, 132, 55, 132, 61, 114, 70,
            96, 74, 88, 75, 88, 87, 74, 89, 66, 91, 67, 100, 59, 108, 50, 120, 40, 122, 37, 97, 43,
            78, 50,
        ],
    ],
    [
        [
            83, 78, 84, 81, 88, 75, 86, 74, 87, 71, 90, 73, 93, 74, 93, 74, 109, 40, 114, 36, 117,
            34, 117, 34, 143, 17, 145, 18, 146, 19, 162, 12, 165, 10, 178, 7, 189, 6, 190, 8, 177,
            9,
        ],
        [
            23, 178, 54, 115, 63, 102, 66, 98, 69, 99, 74, 89, 71, 91, 73, 91, 78, 89, 86, 80, 92,
            66, 93, 64, 102, 59, 103, 60, 104, 60, 117, 52, 123, 44, 138, 35, 133, 31, 97, 38, 77,
            45,
        ],
    ],
    [
        [
            61, 90, 93, 60, 105, 42, 107, 41, 110, 45, 116, 38, 113, 38, 112, 38, 124, 26, 132, 27,
            136, 19, 140, 20, 155, 14, 159, 16, 158, 18, 170, 13, 177, 10, 187, 8, 192, 6, 175, 9,
            159, 10,
        ],
        [
            21, 178, 59, 110, 71, 86, 75, 85, 84, 83, 91, 66, 88, 73, 87, 72, 92, 75, 98, 72, 105,
            58, 107, 54, 115, 52, 114, 55, 112, 56, 129, 51, 132, 40, 150, 33, 140, 29, 98, 35, 77,
            42,
        ],
    ],
    [
        [
            42, 121, 96, 66, 108, 43, 111, 40, 117, 44, 123, 32, 120, 36, 119, 33, 127, 33, 134,
            34, 139, 21, 147, 23, 152, 20, 158, 25, 154, 26, 166, 21, 173, 16, 184, 13, 184, 10,
            150, 13, 139, 15,
        ],
        [
            22, 178, 63, 114, 74, 82, 84, 83, 92, 82, 103, 62, 96, 72, 96, 67, 101, 73, 107, 72,
            113, 55, 118, 52, 125, 52, 118, 52, 117, 55, 135, 49, 137, 39, 157, 32, 145, 29, 97,
            33, 77, 40,
        ],
    ],
];

pub const SMALL_ENERGY_ICDF: [u8; 3] = [2, 1, 0];

fn loss_distortion(
    e_bands: &[f32],
    old_e_bands: &[f32],
    start: usize,
    end: usize,
    len: usize,
    channels: usize,
) -> f32 {
    let mut dist = 0.0f32;
    for c in 0..channels {
        let off = c * len;
        for i in start..end.min(len) {
            let d = e_bands[off + i] - old_e_bands[off + i];
            dist += d * d;
        }
    }
    dist.min(200.0)
}

#[allow(clippy::too_many_arguments)]
fn quant_coarse_energy_impl(
    m: &CeltMode,
    start: usize,
    end: usize,
    e_bands: &[f32],
    old_e_bands: &mut [f32],
    budget: u32,
    tell_start: i32,
    prob_model: &[u8; 42],
    error: &mut [f32],
    enc: &mut RangeCoder,
    channels: usize,
    lm: usize,
    intra: bool,
    max_decay: f32,
    lfe: bool,
) -> i32 {
    let coef = if intra { 0.0 } else { PRED_COEF[lm] };
    let beta = if intra { BETA_INTRA } else { BETA_COEF[lm] };
    let mut prev = [0.0f32; 2];
    let mut badness = 0i32;

    if tell_start + 3 <= budget as i32 {
        enc.encode_bit_logp(intra, 3);
    }

    for i in start..end {
        for c in 0..channels {
            let x = e_bands[c * m.nb_ebands + i];
            let old_e_val = old_e_bands[c * m.nb_ebands + i];
            let old_e = old_e_val.max(-9.0);
            let f = x - coef * old_e - prev[c];

            let mut qi = ((f + 0.5).floor() as i32).clamp(-32767, 32767);
            let qi0 = qi;

            let decay_bound = old_e_val.max(-28.0) - max_decay;
            if qi < 0 && x < decay_bound {
                qi = qi.saturating_add(((decay_bound - x) as i32).max(0));
                if qi > 0 {
                    qi = 0;
                }
            }

            let tell = enc.tell();
            let bits_left = budget as i32 - tell - 3 * channels as i32 * (end - i) as i32;
            if i != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = qi.min(1);
                }
                if bits_left < 16 {
                    qi = qi.max(-1);
                }
            }
            if lfe && i >= 2 {
                qi = qi.min(0);
            }

            if tell + 15 <= budget as i32 {
                let prob_idx = 2 * i.min(20);
                let fs = (prob_model[prob_idx] as u32) << 7;
                let decay = (prob_model[prob_idx + 1] as i32) << 6;
                enc.laplace_encode(&mut qi, fs, decay);
            } else if tell + 2 <= budget as i32 {
                qi = qi.clamp(-1, 1);
                enc.encode_icdf(
                    (2 * qi) ^ (if qi < 0 { -1 } else { 0 }),
                    &SMALL_ENERGY_ICDF,
                    2,
                );
            } else if tell < budget as i32 {
                qi = qi.min(0);
                enc.encode_bit_logp(qi != 0, 1);
            } else {
                qi = -1;
            }

            badness = badness.saturating_add(qi0.saturating_sub(qi).saturating_abs());

            let q = qi as f32;
            error[c * m.nb_ebands + i] = f - q;
            let tmp = coef * old_e + prev[c] + q;
            old_e_bands[c * m.nb_ebands + i] = tmp;
            prev[c] = prev[c] + q - beta * q;
        }
    }

    if lfe { 0 } else { badness }
}

#[allow(clippy::too_many_arguments)]
pub fn quant_coarse_energy_advanced(
    m: &CeltMode,
    start: usize,
    end: usize,
    eff_end: usize,
    e_bands: &[f32],
    old_e_bands: &mut [f32],
    budget: u32,
    error: &mut [f32],
    enc: &mut RangeCoder,
    channels: usize,
    lm: usize,
    nb_available_bytes: usize,
    force_intra: bool,
    delayed_intra: &mut f32,
    mut two_pass: bool,
    loss_rate: i32,
    lfe: bool,
) {
    let mut intra = force_intra
        || (!two_pass
            && *delayed_intra > 2.0 * channels as f32 * (end.saturating_sub(start)) as f32
            && nb_available_bytes > (end.saturating_sub(start)) * channels);

    let intra_bias = ((budget as f32) * (*delayed_intra) * (loss_rate as f32)
        / ((channels as f32) * 512.0)) as i32;
    let new_distortion =
        loss_distortion(e_bands, old_e_bands, start, eff_end, m.nb_ebands, channels);

    let tell = enc.tell();
    if tell + 3 > budget as i32 {
        two_pass = false;
        intra = false;
    }

    let mut max_decay = if end - start > 10 {
        16.0f32.min(0.125 * nb_available_bytes as f32)
    } else {
        16.0f32
    };
    if lfe {
        max_decay = 3.0;
    }

    let enc_start_state = enc.clone();
    let mut old_e_bands_intra = old_e_bands.to_vec();
    let mut error_intra = error.to_vec();
    let mut badness1 = 0i32;
    let mut tell_intra = 0i32;
    let intra_prob = &E_PROB_MODEL[lm][1];

    if two_pass || intra {
        badness1 = quant_coarse_energy_impl(
            m,
            start,
            end,
            e_bands,
            &mut old_e_bands_intra,
            budget,
            tell,
            intra_prob,
            &mut error_intra,
            enc,
            channels,
            lm,
            true,
            max_decay,
            lfe,
        );
        tell_intra = crate::tell_frac_inline!(enc);
    }

    if !intra {
        let enc_intra_state = enc.clone();

        *enc = enc_start_state.clone();
        let inter_prob = &E_PROB_MODEL[lm][0];
        let badness2 = quant_coarse_energy_impl(
            m,
            start,
            end,
            e_bands,
            old_e_bands,
            budget,
            tell,
            inter_prob,
            error,
            enc,
            channels,
            lm,
            false,
            max_decay,
            lfe,
        );

        if two_pass
            && (badness1 < badness2
                || (badness1 == badness2
                    && crate::tell_frac_inline!(enc) + intra_bias > tell_intra))
        {
            *enc = enc_intra_state;
            old_e_bands.copy_from_slice(&old_e_bands_intra);
            error.copy_from_slice(&error_intra);
            intra = true;
        }
    } else {
        old_e_bands.copy_from_slice(&old_e_bands_intra);
        error.copy_from_slice(&error_intra);
    }

    if intra {
        *delayed_intra = new_distortion;
    } else {
        let pred2 = PRED_COEF[lm] * PRED_COEF[lm];
        *delayed_intra = pred2 * *delayed_intra + new_distortion;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn quant_coarse_energy(
    m: &CeltMode,
    start: usize,
    end: usize,
    e_bands: &[f32],
    old_e_bands: &mut [f32],
    budget: u32,
    error: &mut [f32],
    enc: &mut RangeCoder,
    channels: usize,
    lm: usize,
    force_intra: bool,
    nb_available_bytes: usize,
) {
    let mut delayed_intra = 0.0f32;
    quant_coarse_energy_advanced(
        m,
        start,
        end,
        end,
        e_bands,
        old_e_bands,
        budget,
        error,
        enc,
        channels,
        lm,
        nb_available_bytes,
        force_intra,
        &mut delayed_intra,
        false,
        0,
        false,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn unquant_coarse_energy(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    intra: bool,
    dec: &mut RangeCoder,
    channels: usize,
    lm: usize,
) {
    let prob_model = &E_PROB_MODEL[lm][if intra { 1 } else { 0 }];
    let coef = if intra { 0.0 } else { PRED_COEF[lm] };
    let beta = if intra { BETA_INTRA } else { BETA_COEF[lm] };
    debug_assert!(channels <= 2);
    let mut prev = [0.0f32; 2];
    let budget = (dec.storage * 8) as i32;

    for i in start..end {
        for c in 0..channels {
            let qi;
            let tell = dec.tell();
            if budget - tell >= 15 {
                let prob_idx = 2 * i.min(20);
                let fs = (prob_model[prob_idx] as u32) << 7;
                let decay = (prob_model[prob_idx + 1] as i32) << 6;
                qi = dec.laplace_decode(fs, decay);
            } else if budget - tell >= 2 {
                let s = dec.decode_icdf(&SMALL_ENERGY_ICDF, 2);
                qi = (s >> 1) ^ -(s & 1);
            } else if budget - tell >= 1 {
                qi = if dec.decode_bit_logp(1) { -1 } else { 0 };
            } else {
                qi = -1;
            }

            // Clamp in-place, matching C: oldEBands[i] = MAXG(-GCONST(9.f), oldEBands[i])
            old_e_bands[c * m.nb_ebands + i] = old_e_bands[c * m.nb_ebands + i].max(-9.0);
            let old_e = old_e_bands[c * m.nb_ebands + i];

            let q = qi as f32;
            let tmp = coef * old_e + prev[c] + q;
            old_e_bands[c * m.nb_ebands + i] = tmp;
            prev[c] = prev[c] + q - beta * q;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn quant_fine_energy(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    error: &mut [f32],
    fine_quant: &[i32],
    enc: &mut RangeCoder,
    channels: usize,
) {
    for i in start..end {
        for c in 0..channels {
            let bits = fine_quant[i];
            if bits <= 0 {
                continue;
            }
            let mut q = ((error[c * m.nb_ebands + i] + 0.5) * (1 << bits) as f32).floor() as i32;
            q = q.max(0).min((1 << bits) - 1);
            enc.enc_bits(q as u32, bits as u32);
            let offset = (q as f32 + 0.5) / (1 << bits) as f32 - 0.5;
            old_e_bands[c * m.nb_ebands + i] += offset;
            error[c * m.nb_ebands + i] -= offset;
        }
    }
}

pub fn unquant_fine_energy(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    fine_quant: &[i32],
    dec: &mut RangeCoder,
    channels: usize,
) {
    for i in start..end {
        for c in 0..channels {
            let bits = fine_quant[i];
            if bits <= 0 {
                continue;
            }
            let q = dec.dec_bits(bits as u32);
            let offset = (q as f32 + 0.5) / (1 << bits) as f32 - 0.5;
            old_e_bands[c * m.nb_ebands + i] += offset;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn quant_energy_finalise(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    error: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    bits_left: i32,
    enc: &mut RangeCoder,
    channels: usize,
) {
    let mut bits_left = bits_left;
    for priority in 0..2 {
        let mut i = start;
        while i < end && bits_left >= channels as i32 {
            if fine_quant[i] >= 8 || fine_priority[i] != priority {
                i += 1;
                continue;
            }
            let mut c = 0;
            while c < channels {
                let q2 = if error[i + c * m.nb_ebands] < 0.0 {
                    0
                } else {
                    1
                };
                enc.enc_bits(q2 as u32, 1);
                let offset =
                    (q2 as f32 - 0.5) * (1i32 << (14 - fine_quant[i] - 1)) as f32 * (1.0 / 16384.0);
                old_e_bands[i + c * m.nb_ebands] += offset;
                error[i + c * m.nb_ebands] -= offset;
                bits_left -= 1;
                c += 1;
            }
            i += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn unquant_energy_finalise(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    bits_left: i32,
    dec: &mut RangeCoder,
    channels: usize,
) {
    let mut bits_left = bits_left;
    for priority in 0..2 {
        let mut i = start;
        while i < end && bits_left >= channels as i32 {
            if fine_quant[i] >= 8 || fine_priority[i] != priority {
                i += 1;
                continue;
            }
            let mut c = 0;
            while c < channels {
                let q2 = dec.dec_bits(1);
                let offset =
                    (q2 as f32 - 0.5) * (1i32 << (14 - fine_quant[i] - 1)) as f32 * (1.0 / 16384.0);
                old_e_bands[i + c * m.nb_ebands] += offset;
                bits_left -= 1;
                c += 1;
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::range_coder::RangeCoder;

    #[test]
    fn test_coarse_fine_energy() {
        let mode = crate::modes::default_mode();
        let mut e_bands = vec![0.0; mode.nb_ebands];
        for (i, v) in e_bands.iter_mut().enumerate() {
            *v = 5.0 + (i as f32 * 0.5).sin() * 2.0;
        }

        let mut old_e_bands = vec![0.0; mode.nb_ebands];
        let mut error = vec![0.0; mode.nb_ebands];
        let mut enc = RangeCoder::new_encoder(1000);

        quant_coarse_energy(
            mode,
            0,
            mode.nb_ebands,
            &e_bands,
            &mut old_e_bands,
            10000,
            &mut error,
            &mut enc,
            1,
            3,
            false,
            80,
        );

        let mut fine_quant = vec![0; mode.nb_ebands];
        for (i, v) in fine_quant.iter_mut().enumerate() {
            *v = (i % 3) as i32;
        }

        quant_fine_energy(
            mode,
            0,
            mode.nb_ebands,
            &mut old_e_bands,
            &mut error,
            &fine_quant,
            &mut enc,
            1,
        );

        let mut fine_priority = vec![0i32; mode.nb_ebands];
        for (i, v) in fine_priority.iter_mut().enumerate() {
            *v = (i % 2) as i32;
        }

        quant_energy_finalise(
            mode,
            0,
            mode.nb_ebands,
            &mut old_e_bands,
            &mut error,
            &fine_quant,
            &fine_priority,
            10,
            &mut enc,
            1,
        );

        enc.done();
        let _compressed = &enc.buf;

        let mut dec = RangeCoder::new_decoder(&enc.buf);

        let mut decoded_old_e_bands = vec![0.0; mode.nb_ebands];
        let intra = dec.decode_bit_logp(3);
        unquant_coarse_energy(
            mode,
            0,
            mode.nb_ebands,
            &mut decoded_old_e_bands,
            intra,
            &mut dec,
            1,
            3,
        );

        unquant_fine_energy(
            mode,
            0,
            mode.nb_ebands,
            &mut decoded_old_e_bands,
            &fine_quant,
            &mut dec,
            1,
        );

        unquant_energy_finalise(
            mode,
            0,
            mode.nb_ebands,
            &mut decoded_old_e_bands,
            &fine_quant,
            &fine_priority,
            10,
            &mut dec,
            1,
        );

        for i in 0..mode.nb_ebands {
            if (decoded_old_e_bands[i] - old_e_bands[i]).abs() >= 1e-5 {
                println!(
                    "Mismatch at band {}: enc={} dec={} diff={}",
                    i,
                    old_e_bands[i],
                    decoded_old_e_bands[i],
                    (decoded_old_e_bands[i] - old_e_bands[i]).abs()
                );
            }
            assert!((decoded_old_e_bands[i] - old_e_bands[i]).abs() < 1e-5);
        }
    }

    /// Regression test: extreme/corrupt energy values must not cause an
    /// "attempt to add with overflow" panic in `quant_coarse_energy_impl`.
    /// Previously `(qi0 - qi).abs()` overflowed i32 when the float->int cast
    /// of `qi` saturated near i32::MAX/MIN.
    #[test]
    fn test_coarse_energy_extreme_no_overflow() {
        let mode = crate::modes::default_mode();
        let n = mode.nb_ebands;

        for &extreme in &[f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 1.0e30, -1.0e30] {
            let e_bands = vec![extreme; n];
            let mut old_e_bands = vec![0.0; n];
            let mut error = vec![0.0; n];
            let mut enc = RangeCoder::new_encoder(1000);

            // tiny budget forces the `else { qi = -1 }` path, which combined
            // with a saturated qi0 triggered the overflow at the badness line.
            quant_coarse_energy(
                mode,
                0,
                n,
                &e_bands,
                &mut old_e_bands,
                0,
                &mut error,
                &mut enc,
                1,
                3,
                false,
                80,
            );

            // large budget exercises the laplace-encode path with extreme qi.
            let mut old_e_bands2 = vec![0.0; n];
            let mut error2 = vec![0.0; n];
            let mut enc2 = RangeCoder::new_encoder(1000);
            quant_coarse_energy(
                mode,
                0,
                n,
                &e_bands,
                &mut old_e_bands2,
                10000,
                &mut error2,
                &mut enc2,
                1,
                3,
                false,
                80,
            );
        }
    }
}
