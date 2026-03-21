use crate::modes::CeltMode;
use crate::range_coder::RangeCoder;

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
    intra: bool,
) {
    let prob_model = &E_PROB_MODEL[lm][if intra { 1 } else { 0 }];
    let coef = if intra { 0.0 } else { PRED_COEF[lm] };
    let beta = if intra { BETA_INTRA } else { BETA_COEF[lm] };
    let mut prev = vec![0.0f32; channels];

    let mut max_decay = 16.0f32;
    if lm == 0 {
        max_decay = 8.0;
    }
    if lm >= 2 {
        max_decay = 32.0;
    }

    enc.encode_bit_logp(intra, 3);

    for i in start..end {
        for c in 0..channels {
            let x = e_bands[c * m.nb_ebands + i];
            let old_e_val = old_e_bands[c * m.nb_ebands + i];
            let old_e = old_e_val.max(-9.0);
            let f = x - coef * old_e - prev[c];

            let mut qi = (f + 0.5).floor() as i32;

            let decay_bound = old_e_val.max(-28.0) - max_decay;
            if qi < 0 && x < decay_bound {
                qi += (decay_bound - x).floor() as i32;
                if qi > 0 {
                    qi = 0;
                }
            }

            let tell = enc.tell() << 3;
            let bits_left = budget as i32 - tell - 3 * channels as i32 * (end - i) as i32;
            if i != start && bits_left < 30 {
                if bits_left < 24 {
                    qi = qi.min(1);
                }
                if bits_left < 16 {
                    qi = qi.max(-1);
                }
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

            let q = qi as f32;
            error[c * m.nb_ebands + i] = f - q;
            let tmp = coef * old_e + prev[c] + q;
            old_e_bands[c * m.nb_ebands + i] = tmp;
            prev[c] = prev[c] + q - beta * q;

            if i < 3 {
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn unquant_coarse_energy(
    m: &CeltMode,
    start: usize,
    end: usize,
    old_e_bands: &mut [f32],
    budget: u32,
    dec: &mut RangeCoder,
    channels: usize,
    lm: usize,
    mut intra: bool,
) {
    let tell = dec.tell() << 3;
    if tell + 3 <= budget as i32 {
        intra = dec.decode_bit_logp(3);
    }
    let prob_model = &E_PROB_MODEL[lm][if intra { 1 } else { 0 }];
    let coef = if intra { 0.0 } else { PRED_COEF[lm] };
    let beta = if intra { BETA_INTRA } else { BETA_COEF[lm] };
    let mut prev = vec![0.0f32; channels];

    for i in start..end {
        for c in 0..channels {
            let old_e = old_e_bands[c * m.nb_ebands + i].max(-9.0);

            let qi;
            let tell = dec.tell() << 3;
            if tell + 15 <= budget as i32 {
                let prob_idx = 2 * i.min(20);
                let fs = (prob_model[prob_idx] as u32) << 7;
                let decay = (prob_model[prob_idx + 1] as i32) << 6;
                qi = dec.laplace_decode(fs, decay);
            } else if tell + 2 <= budget as i32 {
                let s = dec.decode_icdf(&SMALL_ENERGY_ICDF, 2);
                qi = (s >> 1) ^ -(s & 1);
            } else if tell < budget as i32 {
                qi = if dec.decode_bit_logp(1) { -1 } else { 0 };
            } else {
                qi = -1;
            }

            let q = qi as f32;
            let tmp = coef * old_e + prev[c] + q;
            old_e_bands[c * m.nb_ebands + i] = tmp;
            prev[c] = prev[c] + q - beta * q;

            if i < 3 {
            }
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
            let bits = fine_quant[c * m.nb_ebands + i];
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
            let bits = fine_quant[c * m.nb_ebands + i];
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
        for i in start..end {
            for c in 0..channels {
                if bits_left >= 8
                    && fine_priority[c * m.nb_ebands + i] == priority
                    && fine_quant[c * m.nb_ebands + i] < 7
                {
                    let q = if error[c * m.nb_ebands + i] >= 0.0 {
                        1
                    } else {
                        0
                    };
                    enc.enc_bits(q as u32, 1);
                    let offset = if q == 1 { 0.25 } else { -0.25 };
                    old_e_bands[c * m.nb_ebands + i] += offset;
                    error[c * m.nb_ebands + i] -= offset;
                    bits_left -= 8;
                }
            }
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
        for i in start..end {
            for c in 0..channels {
                if bits_left >= 8
                    && fine_priority[c * m.nb_ebands + i] == priority
                    && fine_quant[c * m.nb_ebands + i] < 7
                {
                    let q = dec.dec_bits(1);
                    let offset = if q == 1 { 0.25 } else { -0.25 };
                    old_e_bands[c * m.nb_ebands + i] += offset;
                    bits_left -= 8;
                }
            }
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

        let mut dec = RangeCoder::new_decoder(enc.buf.clone());

        let mut decoded_old_e_bands = vec![0.0; mode.nb_ebands];
        unquant_coarse_energy(
            mode,
            0,
            mode.nb_ebands,
            &mut decoded_old_e_bands,
            10000,
            &mut dec,
            1,
            3,
            false,
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
}
