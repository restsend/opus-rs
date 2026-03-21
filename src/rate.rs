use crate::modes::CeltMode;
use crate::range_coder::RangeCoder;
use std::cmp::{max, min};

pub const BITRES: i32 = 3;
pub const FINE_OFFSET: i32 = 21;
pub const QTHETA_OFFSET: i32 = 4;
pub const QTHETA_OFFSET_TWOPHASE: i32 = 16;
pub const MAX_FINE_BITS: i32 = 8;

pub const LOG2_FRAC_TABLE: [u8; 24] = [
    0, 8, 13, 16, 19, 21, 23, 24, 26, 27, 28, 29, 30, 31, 32, 32, 33, 34, 34, 35, 36, 36, 37, 37,
];

pub fn get_pulses(i: i32) -> i32 {
    if i < 8 {
        i
    } else {
        let shift = (i >> 3) - 1;
        if shift >= 31 {
            return 0x7FFFFFFF;
        }
        (8 + (i & 7)) << shift
    }
}

pub fn bits2pulses(m: &CeltMode, band: usize, mut lm: i32, bits: i32) -> i32 {
    lm += 1;
    let cache_index = m.cache.index[lm as usize * m.nb_ebands + band];
    if cache_index < 0 {
        return 0;
    }
    let cache = &m.cache.bits[cache_index as usize..];

    let mut lo = 0;
    let hi_limit = cache[0];
    let mut hi = hi_limit as usize;
    let bits_minus_one = bits - 1;

    for _ in 0..6 {
        let mid = (lo + hi + 1) >> 1;
        if (cache[mid] as i32) >= bits_minus_one {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    if bits_minus_one - (if lo == 0 { -1 } else { cache[lo] as i32 })
        <= (cache[hi] as i32) - bits_minus_one
    {
        lo as i32
    } else {
        hi as i32
    }
}

pub fn pulses2bits(m: &CeltMode, band: usize, mut lm: i32, pulses: i32) -> i32 {
    if pulses == 0 {
        return 0;
    }
    lm += 1;
    let cache_index = m.cache.index[lm as usize * m.nb_ebands + band];
    if cache_index < 0 {
        return 0;
    }
    let cache = &m.cache.bits[cache_index as usize..];
    (cache[pulses as usize] as i32) + 1
}

#[allow(clippy::too_many_arguments)]
pub fn clt_compute_allocation(
    m: &CeltMode,
    start: usize,
    end: usize,
    offsets: &[i32],
    cap: &[i32],
    alloc_trim: i32,
    intensity: &mut i32,
    dual_stereo: &mut i32,
    mut total: i32,
    balance_out: &mut i32,
    pulses: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    rc: &mut RangeCoder,
    encode: bool,
    prev: i32,
    signal_bandwidth: i32,
) -> i32 {
    let tell = rc.tell() << BITRES;
    total -= tell;
    total = max(total, 0);
    let nb_ebands = m.nb_ebands;
    let mut skip_start = start;

    let skip_rsv = if total >= (1 << BITRES) {
        1 << BITRES
    } else {
        0
    };
    total -= skip_rsv;

    let mut intensity_rsv = 0;
    let mut dual_stereo_rsv = 0;
    if c == 2 {
        intensity_rsv = LOG2_FRAC_TABLE[end - start] as i32;
        if intensity_rsv > total {
            intensity_rsv = 0;
        } else {
            total -= intensity_rsv;
            dual_stereo_rsv = if total >= (1 << BITRES) {
                1 << BITRES
            } else {
                0
            };
            total -= dual_stereo_rsv;
        }
    }

    let mut thresh = vec![0; nb_ebands];
    let mut trim_offset = vec![0; nb_ebands];

    for j in start..end {
        thresh[j] = max(
            c << BITRES,
            ((3 * (m.e_bands[j + 1] - m.e_bands[j]) as i32) << (lm + BITRES)) >> 4,
        );
        trim_offset[j] = (c
            * (m.e_bands[j + 1] - m.e_bands[j]) as i32
            * (alloc_trim - 5 - lm)
            * (end - j - 1) as i32
            * (1 << (lm + BITRES)))
            >> 6;
        if (m.e_bands[j + 1] - m.e_bands[j]) << lm == 1 {
            trim_offset[j] -= c << BITRES;
        }
    }

    let mut lo = 1;
    let mut hi = m.nb_alloc_vectors as i32 - 1;
    while lo <= hi {
        let mut done = false;
        let mut psum = 0;
        let mid = (lo + hi) >> 1;
        for j in (start..end).rev() {
            let n = (m.e_bands[j + 1] - m.e_bands[j]) as i32;
            let mut bitsj =
                (c * n * m.alloc_vectors[mid as usize * m.alloc_stride + j] as i32) << lm >> 2;
            if bitsj > 0 {
                bitsj = max(0, bitsj + trim_offset[j]);
            }
            bitsj += offsets[j];
            if bitsj >= thresh[j] || done {
                done = true;
                psum += min(bitsj, cap[j]);
            } else {
                if bitsj >= (c << BITRES) {
                    psum += c << BITRES;
                }
            }
        }
        if psum > total {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    let hi_final = lo as usize;
    let lo_final = (lo - 1) as usize;

    let mut bits1 = vec![0; nb_ebands];
    let mut bits2 = vec![0; nb_ebands];

    for j in start..end {
        let n = (m.e_bands[j + 1] - m.e_bands[j]) as i32;
        let mut bits1j = (c * n * m.alloc_vectors[lo_final * m.alloc_stride + j] as i32) << lm >> 2;
        let mut bits2j = if hi_final >= m.nb_alloc_vectors {
            cap[j]
        } else {
            (c * n * m.alloc_vectors[hi_final * m.alloc_stride + j] as i32) << lm >> 2
        };

        if bits1j > 0 {
            bits1j = max(0, bits1j + trim_offset[j]);
        }
        if bits2j > 0 {
            bits2j = max(0, bits2j + trim_offset[j]);
        }
        if lo_final > 0 {
            bits1j += offsets[j];
        }
        bits2j += offsets[j];
        if offsets[j] > 0 {
            skip_start = j;
        }
        bits2j = max(0, bits2j - bits1j);
        bits1[j] = bits1j;
        bits2[j] = bits2j;
    }

    interp_bits2pulses(
        m,
        start,
        end,
        skip_start,
        &bits1,
        &bits2,
        &thresh,
        cap,
        total,
        balance_out,
        skip_rsv,
        intensity,
        intensity_rsv,
        dual_stereo,
        dual_stereo_rsv,
        pulses,
        ebits,
        fine_priority,
        c,
        lm,
        rc,
        encode,
        prev,
        signal_bandwidth,
    )
}

#[allow(clippy::too_many_arguments)]
fn interp_bits2pulses(
    m: &CeltMode,
    start: usize,
    end: usize,
    skip_start: usize,
    bits1: &[i32],
    bits2: &[i32],
    thresh: &[i32],
    cap: &[i32],
    total: i32,
    balance_out: &mut i32,
    skip_rsv: i32,
    intensity: &mut i32,
    intensity_rsv: i32,
    dual_stereo: &mut i32,
    dual_stereo_rsv: i32,
    pulses: &mut [i32],
    ebits: &mut [i32],
    fine_priority: &mut [i32],
    c: i32,
    lm: i32,
    rc: &mut RangeCoder,
    encode: bool,
    prev: i32,
    signal_bandwidth: i32,
) -> i32 {
    let mut psum: i32;
    let mut lo = 0;
    let mut hi = 1 << 6;
    let alloc_floor = c << BITRES;
    let stereo = if c > 1 { 1 } else { 0 };
    let log_m = lm << BITRES;

    let mut bits = vec![0; m.nb_ebands];

    for _ in 0..6 {
        let mid = (lo + hi) >> 1;
        psum = 0;
        let mut done = false;
        for j in (start..end).rev() {
            let tmp = bits1[j] + ((mid * bits2[j]) >> 6);
            if tmp >= thresh[j] || done {
                done = true;
                psum += min(tmp, cap[j]);
            } else {
                if tmp >= alloc_floor {
                    psum += alloc_floor;
                }
            }
        }
        if psum > total {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    psum = 0;
    let mut done = false;
    for j in (start..end).rev() {
        let mut tmp = bits1[j] + ((lo * bits2[j]) >> 6);
        if tmp < thresh[j] && !done {
            if tmp >= alloc_floor {
                tmp = alloc_floor;
            } else {
                tmp = 0;
            }
        } else {
            done = true;
        }
        tmp = min(tmp, cap[j]);
        bits[j] = tmp;
        psum += tmp;
    }

    let mut coded_bands = end;
    let mut total_with_rsv = total;
    loop {
        if coded_bands <= start {
            break;
        }
        let j = coded_bands - 1;
        if j <= skip_start {
            total_with_rsv += skip_rsv;
            break;
        }

        let left = total_with_rsv - psum;
        let nb_samples = (m.e_bands[coded_bands] - m.e_bands[start]) as i32;
        let percoeff = left / nb_samples;
        let left_rem = left - nb_samples * percoeff;
        let rem = max(left_rem - (m.e_bands[j] - m.e_bands[start]) as i32, 0);
        let band_width = (m.e_bands[coded_bands] - m.e_bands[j]) as i32;
        let mut band_bits = bits[j] + percoeff * band_width + rem;

        if band_bits >= max(thresh[j], alloc_floor + (1 << BITRES)) {
            if encode {
                let depth_threshold = if coded_bands > 17 {
                    if (j as i32) < prev { 7 } else { 9 }
                } else {
                    0
                };
                if coded_bands <= start + 2
                    || (band_bits > ((depth_threshold * band_width) << lm << BITRES) >> 4
                        && (j as i32) <= signal_bandwidth)
                {
                    rc.encode_bit_logp(true, 1);
                    break;
                }
                rc.encode_bit_logp(false, 1);
            } else {
                if rc.decode_bit_logp(1) {
                    break;
                }
            }
            psum += 1 << BITRES;
            band_bits -= 1 << BITRES;
        }
        psum -= bits[j] + intensity_rsv;
        let mut new_intensity_rsv = intensity_rsv;
        if intensity_rsv > 0 {
            new_intensity_rsv = LOG2_FRAC_TABLE[j - start] as i32;
        }
        psum += new_intensity_rsv;
        if band_bits >= alloc_floor {
            psum += alloc_floor;
            bits[j] = alloc_floor;
        } else {
            bits[j] = 0;
        }
        coded_bands -= 1;
    }

    let mut intensity_rsv_final = intensity_rsv;
    if intensity_rsv_final > 0 {
        if encode {
            *intensity = min(*intensity, coded_bands as i32);
            rc.enc_uint(
                (*intensity - start as i32) as u32,
                (coded_bands + 1 - start) as u32,
            );
        } else {
            *intensity = start as i32 + rc.dec_uint((coded_bands + 1 - start) as u32) as i32;
        }
        intensity_rsv_final = LOG2_FRAC_TABLE[*intensity as usize - start] as i32;
    } else {
        *intensity = 0;
    }
    total_with_rsv -= intensity_rsv - intensity_rsv_final;

    let mut dual_stereo_rsv_final = dual_stereo_rsv;
    if *intensity <= start as i32 {
        total_with_rsv += dual_stereo_rsv_final;
        dual_stereo_rsv_final = 0;
    }
    if dual_stereo_rsv_final > 0 {
        if encode {
            rc.encode_bit_logp(*dual_stereo != 0, 1);
        } else {
            *dual_stereo = if rc.decode_bit_logp(1) { 1 } else { 0 };
        }
    } else {
        *dual_stereo = 0;
    }

    let mut left = total_with_rsv - psum;
    let nb_samples = (m.e_bands[coded_bands] - m.e_bands[start]) as i32;
    let percoeff = left / nb_samples;
    left -= nb_samples * percoeff;
    for (j, bits_j) in bits[start..coded_bands].iter_mut().enumerate().map(|(i, v)| (i + start, v)) {
        *bits_j += percoeff * (m.e_bands[j + 1] - m.e_bands[j]) as i32;
    }
    for (j, bits_j) in bits[start..coded_bands].iter_mut().enumerate().map(|(i, v)| (i + start, v)) {
        let tmp = min(left, (m.e_bands[j + 1] - m.e_bands[j]) as i32);
        *bits_j += tmp;
        left -= tmp;
    }

    let mut balance = 0;
    for j in start..coded_bands {
        let n0 = (m.e_bands[j + 1] - m.e_bands[j]) as i32;
        let n = n0 << lm;
        let bit = bits[j] + balance;

        if n > 1 {
            let excess = max(bit - cap[j], 0);
            bits[j] = bit - excess;

            let den = c * n
                + (if c == 2 && n > 2 && *dual_stereo == 0 && (j as i32) < *intensity {
                    1
                } else {
                    0
                });
            let nc_log_n = den * (m.log_n[j] as i32 + log_m);
            let mut offset = (nc_log_n >> 1) - den * FINE_OFFSET;

            if n == 2 {
                offset += den << BITRES >> 2;
            }

            if bits[j] + offset < (den * 2) << BITRES {
                offset += nc_log_n >> 2;
            } else if bits[j] + offset < (den * 3) << BITRES {
                offset += nc_log_n >> 3;
            }

            ebits[j] = max(0, bits[j] + offset + (den << (BITRES - 1)));
            ebits[j] = (ebits[j] / den) >> BITRES;

            if c * ebits[j] > (bits[j] >> BITRES) {
                ebits[j] = bits[j] >> stereo >> BITRES;
            }
            ebits[j] = min(ebits[j], MAX_FINE_BITS);
            fine_priority[j] = if ebits[j] * (den << BITRES) >= bits[j] + offset {
                1
            } else {
                0
            };
            bits[j] -= (c * ebits[j]) << BITRES;
            balance = excess;
        } else {
            let excess = max(0, bit - (c << BITRES));
            bits[j] = bit - excess;
            ebits[j] = 0;
            fine_priority[j] = 1;
            balance = excess;
        }

        if balance > 0 {
            let extra_fine = min(balance >> (stereo + BITRES), MAX_FINE_BITS - ebits[j]);
            ebits[j] += extra_fine;
            let extra_bits = (extra_fine * c) << BITRES;
            fine_priority[j] = if extra_bits >= balance { 1 } else { 0 };
            balance -= extra_bits;
        }
        pulses[j] = bits[j];
    }
    *balance_out = balance;

    for j in coded_bands..end {
        ebits[j] = bits[j] >> stereo >> BITRES;
        bits[j] = 0;
        fine_priority[j] = if ebits[j] < 1 { 1 } else { 0 };
        pulses[j] = 0;
    }

    coded_bands as i32
}
