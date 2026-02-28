use crate::modes::CeltMode;
use crate::pvq::*;
use crate::range_coder::RangeCoder;
use crate::rate::{BITRES, bits2pulses, get_pulses, pulses2bits};

const MIN_STEREO_ENERGY: f32 = 1e-10;

pub struct BandCtx<'a> {
    pub encode: bool,
    pub m: &'a CeltMode,
    pub i: usize,
    pub band_e: &'a [f32],
    pub rc: &'a mut RangeCoder,
    pub spread: i32,
    pub remaining_bits: i32,
    pub resynth: bool,
    pub tf_change: i32,
    pub intensity: usize,
    pub theta_round: i32,
    pub avoid_split_noise: bool,
    pub arch: i32,
    pub disable_inv: bool,
}

pub fn bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    let ec_ilog = |x: u32| -> i32 {
        if x == 0 {
            0
        } else {
            32 - x.leading_zeros() as i32
        }
    };
    let lc = ec_ilog(icos.max(0) as u32);
    let ls = ec_ilog(isin.max(0) as u32);
    let icos_shifted = if lc > 0 {
        icos.max(0) << (15 - lc).max(0)
    } else {
        0
    };
    let isin_shifted = if ls > 0 {
        isin.max(0) << (15 - ls).max(0)
    } else {
        0
    };
    let fract_mul = |a: i32, b: i32| -> i32 { (a * b + 16384) >> 15 };
    (ls - lc) * (1 << 11) + fract_mul(isin_shifted, fract_mul(isin_shifted, -2597) + 7932)
        - fract_mul(icos_shifted, fract_mul(icos_shifted, -2597) + 7932)
}

fn celt_sudiv(n: i32, d: i32) -> i32 {
    if n < 0 {
        -((-n + (d >> 1)) / d)
    } else {
        (n + (d >> 1)) / d
    }
}

pub const SPREAD_NONE: i32 = 0;
pub const SPREAD_LIGHT: i32 = 1;
pub const SPREAD_NORMAL: i32 = 2;
pub const SPREAD_AGGRESSIVE: i32 = 3;

pub fn spreading_decision(
    m: &CeltMode,
    x_buf: &[f32],
    average: &mut i32,
    last_decision: i32,
    hf_average: &mut i32,
    tapset_decision: &mut i32,
    update_hf: bool,
    end: usize,
    channels: usize,
    m_val: usize,
    spread_weight: &[i32],
) -> i32 {
    let mut sum = 0;
    let mut nb_bands = 0;
    let n0 = m_val * m.short_mdct_size;
    let mut hf_sum = 0;

    if m_val * (m.e_bands[end] as usize - m.e_bands[end - 1] as usize) <= 8 {
        return SPREAD_NONE;
    }

    for c in 0..channels {
        for i in 0..end {
            let n = m_val * (m.e_bands[i + 1] as usize - m.e_bands[i] as usize);
            if n <= 8 {
                continue;
            }

            let mut tcount = [0; 3];
            let offset = m_val * m.e_bands[i] as usize + c * n0;
            let x = &x_buf[offset..offset + n];

            for j in 0..n {
                let x2n = x[j] * x[j] * (n as f32);
                if x2n < 0.25 {
                    tcount[0] += 1;
                }
                if x2n < 0.0625 {
                    tcount[1] += 1;
                }
                if x2n < 0.015625 {
                    tcount[2] += 1;
                }
            }

            if i > m.nb_ebands - 4 {
                hf_sum += 32 * (tcount[1] + tcount[0]) / (n as i32);
            }

            let tmp = (if 2 * tcount[2] >= (n as i32) { 1 } else { 0 })
                + (if 2 * tcount[1] >= (n as i32) { 1 } else { 0 })
                + (if 2 * tcount[0] >= (n as i32) { 1 } else { 0 });
            sum += tmp * spread_weight[i];
            nb_bands += spread_weight[i];
        }
    }

    if update_hf {
        if hf_sum > 0 {
            hf_sum /= (channels as i32) * (4 - m.nb_ebands as i32 + end as i32);
        }
        *hf_average = (*hf_average + hf_sum) >> 1;
        hf_sum = *hf_average;

        if *tapset_decision == 2 {
            hf_sum += 4;
        } else if *tapset_decision == 0 {
            hf_sum -= 4;
        }

        if hf_sum > 22 {
            *tapset_decision = 2;
        } else if hf_sum > 18 {
            *tapset_decision = 1;
        } else {
            *tapset_decision = 0;
        }
    }

    if nb_bands == 0 {
        return SPREAD_NORMAL;
    }

    let mut sum_scaled = (sum << 8) / nb_bands;
    sum_scaled = (sum_scaled + *average) >> 1;
    *average = sum_scaled;

    let sum_final = (3 * sum_scaled + (((3 - last_decision) << 7) + 64) + 2) >> 2;

    if sum_final < 80 {
        SPREAD_AGGRESSIVE
    } else if sum_final < 256 {
        SPREAD_NORMAL
    } else if sum_final < 384 {
        SPREAD_LIGHT
    } else {
        SPREAD_NONE
    }
}

pub fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    let n = n0 >> 1;
    for i in 0..stride {
        for j in 0..n {
            let tmp1 = 0.70710678 * x[stride * 2 * j + i];
            let tmp2 = 0.70710678 * x[stride * (2 * j + 1) + i];
            x[stride * 2 * j + i] = tmp1 + tmp2;
            x[stride * (2 * j + 1) + i] = tmp1 - tmp2;
        }
    }
}

pub fn compute_qn(n: usize, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    static EXP2_TABLE8: [i16; 8] = [16384, 17866, 19483, 21247, 23170, 25267, 27554, 30048];
    let mut n2 = (2 * n as i32) - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let mut qb = celt_sudiv(b + n2 * offset, n2);
    qb = qb.min(b - pulse_cap - (4 << BITRES));
    qb = qb.min(8 << BITRES);
    if qb < (1i32 << BITRES >> 1) {
        1
    } else {
        let val = EXP2_TABLE8[(qb & 0x7) as usize] as i32;
        let shift = 14 - (qb >> BITRES);
        let raw = if shift >= 0 && shift < 32 {
            val >> shift
        } else {
            0
        };
        let qn = (raw + 1) >> 1 << 1;
        qn.min(256)
    }
}

pub fn stereo_itheta(x: &[f32], y: &[f32], stereo: bool, n: usize) -> i32 {
    let mut emid = 1e-15f32;
    let mut eside = 1e-15f32;
    if stereo {
        for i in 0..n {
            let m = x[i] + y[i];
            let s = x[i] - y[i];
            emid += m * m;
            eside += s * s;
        }
    } else {
        for i in 0..n {
            emid += x[i] * x[i];
            eside += y[i] * y[i];
        }
    }
    let theta = (eside.sqrt()).atan2(emid.sqrt());
    (0.5 + 16384.0 * theta / (std::f32::consts::PI / 2.0)) as i32
}

pub struct SplitCtx {
    pub inv: bool,
    pub imid: i32,
    pub iside: i32,
    pub delta: i32,
    pub itheta: i32,
    pub qalloc: i32,
}

pub fn compute_theta(
    ctx: &mut BandCtx,
    sctx: &mut SplitCtx,
    x: &[f32],
    y: &[f32],
    n: usize,
    b: &mut i32,
    b_blocks: i32,
    _b0: i32,
    lm: i32,
    stereo: bool,
    fill: &mut u32,
) {
    let pulse_cap = ctx.m.log_n[ctx.i] as i32 + (lm << BITRES);
    let offset = (pulse_cap >> 1) - if stereo && n == 2 { 16 } else { 4 };
    let mut qn = compute_qn(n, *b, offset, pulse_cap, stereo);

    if stereo && ctx.i >= ctx.intensity {
        qn = 1;
    }

    let mut itheta = 0;
    if ctx.encode {
        itheta = stereo_itheta(x, y, stereo, n);
    }

    let tell_start = ctx.rc.tell() << 3;

    if qn != 1 {
        if ctx.encode {
            if !stereo || ctx.theta_round == 0 {
                itheta = (itheta * qn + 8192) >> 14;
                if !stereo && ctx.avoid_split_noise && itheta > 0 && itheta < qn {
                    let unquantized = (itheta * 16384) / qn;
                    let angle = (unquantized as f32) * (std::f32::consts::PI * 0.5 / 16384.0);
                    let imid = (32768.0 * angle.cos()) as i32;
                    let iside = (32768.0
                        * ((16384 - unquantized) as f32 * (std::f32::consts::PI * 0.5 / 16384.0))
                            .cos()) as i32;
                    let delta =
                        (((n as i32 - 1) << 7) * bitexact_log2tan(iside, imid) + 16384) >> 15;
                    if delta > *b {
                        itheta = qn;
                    } else if delta < -*b {
                        itheta = 0;
                    }
                }
            } else {
                let bias = if itheta > 8192 {
                    32767 / qn
                } else {
                    -32767 / qn
                };
                let down = (itheta * qn + bias) >> 14;
                let down = down.clamp(0, qn - 1);
                if ctx.theta_round < 0 {
                    itheta = down;
                } else {
                    itheta = down + 1;
                }
            }
        }

        if stereo && n > 2 {
            let p0 = 3;
            let x0 = qn / 2;
            let ft = p0 * (x0 + 1) + x0;
            if ctx.encode {
                let fl = if itheta <= x0 {
                    p0 * itheta
                } else {
                    (itheta - 1 - x0) + (x0 + 1) * p0
                };
                let fh = if itheta <= x0 {
                    p0 * (itheta + 1)
                } else {
                    (itheta - x0) + (x0 + 1) * p0
                };
                ctx.rc.encode(fl as u32, fh as u32, ft as u32);
            } else {
                let fs = ctx.rc.decode(ft as u32);
                if fs < (x0 + 1) as u32 * p0 as u32 {
                    itheta = fs as i32 / p0;
                } else {
                    itheta = (x0 + 1) as i32 + (fs as i32 - (x0 + 1) * p0);
                }
                let fl = if itheta <= x0 {
                    p0 * itheta
                } else {
                    (itheta - 1 - x0) + (x0 + 1) * p0
                };
                let fh = if itheta <= x0 {
                    p0 * (itheta + 1)
                } else {
                    (itheta - x0) + (x0 + 1) * p0
                };
                ctx.rc.update(fl as u32, fh as u32, ft as u32);
            }
        } else if b_blocks > 1 || stereo {
            if ctx.encode {
                ctx.rc.enc_uint(itheta as u32, (qn + 1) as u32);
            } else {
                let itheta_dec = ctx.rc.dec_uint((qn + 1) as u32) as i32;
                itheta = itheta_dec;
            }
        } else {
            let ft = ((qn >> 1) + 1) * ((qn >> 1) + 1);
            if ctx.encode {
                let fs = if itheta <= (qn >> 1) {
                    itheta + 1
                } else {
                    qn + 1 - itheta
                };
                let fl = if itheta <= (qn >> 1) {
                    itheta * (itheta + 1) >> 1
                } else {
                    ft - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1)
                };
                ctx.rc.encode(fl as u32, (fl + fs) as u32, ft as u32);
            } else {
                let fm = ctx.rc.decode(ft as u32) as i32;
                if fm < ((qn >> 1) * ((qn >> 1) + 1) >> 1) {
                    itheta = (((8 * fm + 1) as f32).sqrt() as i32 - 1) >> 1;
                    let fl = itheta * (itheta + 1) >> 1;
                    let fs = itheta + 1;
                    ctx.rc.update(fl as u32, (fl + fs) as u32, ft as u32);
                } else {
                    itheta = (2 * (qn + 1) - (((8 * (ft - fm - 1) + 1) as f32).sqrt() as i32)) >> 1;
                    let fs = qn + 1 - itheta;
                    let fl = ft - ((qn + 1 - itheta) * (qn + 2 - itheta) >> 1);
                    ctx.rc.update(fl as u32, (fl + fs) as u32, ft as u32);
                }
            }
        }
        itheta = (itheta as i64 * 16384 + qn as i64 / 2) as i32 / qn;
    } else {
        if stereo && ctx.i >= ctx.intensity {
            if ctx.encode {
                let mut emid = 1e-15f32;
                let mut eside = 1e-15f32;
                for i in 0..n {
                    let m = x[i] + y[i];
                    let s = x[i] - y[i];
                    emid += m * m;
                    eside += s * s;
                }
                let inv = eside > emid;
                ctx.rc.encode_bit_logp(inv, 1);
                itheta = 0;
                sctx.inv = inv;
            } else {
                sctx.inv = ctx.rc.decode_bit_logp(1);
                itheta = 0;
            }
        } else {
            itheta = 8192;
        }
    }

    sctx.itheta = itheta;
    sctx.qalloc = ((ctx.rc.tell() << 3) - tell_start) as i32;

    if itheta == 0 {
        sctx.imid = 32767;
        sctx.iside = 0;
        sctx.delta = -16384;
        *fill &= (1 << b_blocks) - 1;
    } else if itheta == 16384 {
        sctx.imid = 0;
        sctx.iside = 32767;
        sctx.delta = 16384;
        *fill &= !((1 << b_blocks) - 1);
    } else {
        let angle = (itheta as f32) * (std::f32::consts::PI * 0.5 / 16384.0);
        sctx.imid = (32768.0 * angle.cos()) as i32;
        sctx.iside = (32768.0
            * ((16384 - itheta) as f32 * (std::f32::consts::PI * 0.5 / 16384.0)).cos())
            as i32;
        sctx.delta =
            (((n as i32 - 1) << 7) * bitexact_log2tan(sctx.iside, sctx.imid) + 16384) >> 15;
    }
}

#[inline(never)]
pub fn quant_partition(
    ctx: &mut BandCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    gain: f32,
    fill: u32,
) -> u32 {
    if n > 1 && b >= (1 << 3) {
        let mut sctx = SplitCtx {
            inv: false,
            imid: 0,
            iside: 0,
            delta: 0,
            itheta: 0,
            qalloc: 0,
        };
        let mut b_mut = b;
        let mut fill_mut = fill;
        let (x_mid, x_side) = x.split_at_mut(n / 2);
        compute_theta(
            ctx,
            &mut sctx,
            x_mid,
            x_side,
            n / 2,
            &mut b_mut,
            (b_blocks + 1) >> 1,
            b_blocks,
            lm,
            false,
            &mut fill_mut,
        );
        // sctx.imid = sctx.itheta;

        ctx.remaining_bits -= sctx.qalloc;
        let mbits = (0).max((b_mut - sctx.delta) / 2).min(b_mut);
        let mut sbits = b_mut - mbits;
        let mut mbits = mbits;

        let mut rebalance = ctx.remaining_bits;
        let mut cm;
        if mbits >= sbits {
            cm = quant_partition(
                ctx,
                x_mid,
                n / 2,
                mbits,
                (b_blocks + 1) >> 1,
                lowband,
                lm,
                gain * (sctx.imid as f32 / 32768.0),
                fill_mut,
            );
            rebalance = mbits - (rebalance - ctx.remaining_bits);
            if rebalance > (3 << 3) && sctx.itheta != 0 {
                sbits += rebalance - (3 << 3);
            }
            cm |= quant_partition(
                ctx,
                x_side,
                n / 2,
                sbits,
                (b_blocks + 1) >> 1,
                None,
                lm,
                gain * (sctx.iside as f32 / 32768.0),
                fill_mut >> b_blocks,
            ) << (b_blocks >> 1);
        } else {
            cm = quant_partition(
                ctx,
                x_side,
                n / 2,
                sbits,
                (b_blocks + 1) >> 1,
                None,
                lm,
                gain * (sctx.iside as f32 / 32768.0),
                fill_mut >> b_blocks,
            ) << (b_blocks >> 1);
            rebalance = sbits - (rebalance - ctx.remaining_bits);
            if rebalance > (3 << 3) && sctx.itheta != 16384 {
                mbits += rebalance - (3 << 3);
            }
            cm |= quant_partition(
                ctx,
                x_mid,
                n / 2,
                mbits,
                (b_blocks + 1) >> 1,
                lowband,
                lm,
                gain * (sctx.imid as f32 / 32768.0),
                fill_mut,
            );
        }
        return cm;
    } else {
        let q = bits2pulses(ctx.m, ctx.i, lm, b);
        let curr_bits = pulses2bits(ctx.m, ctx.i, lm, q);
        ctx.remaining_bits -= curr_bits;

        if q != 0 {
            let k = get_pulses(q);
            if ctx.encode {
                alg_quant(
                    x,
                    n,
                    k,
                    ctx.spread,
                    b_blocks as usize,
                    ctx.rc,
                    gain,
                    ctx.resynth,
                )
            } else {
                alg_unquant(x, n, k, ctx.spread, b_blocks as usize, ctx.rc, gain)
            }
        } else {
            if ctx.resynth {
                let mut seed = ctx.rc.tell() as u32; // Just a dummy seed for now.
                if let Some(ref lb) = lowband {
                    for j in 0..n {
                        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                        x[j] = lb[j]
                            + if seed & 0x8000 != 0 {
                                1.0 / 256.0
                            } else {
                                -1.0 / 256.0
                            };
                    }
                } else {
                    for j in 0..n {
                        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                        x[j] = ((seed as i32 >> 20) as f32) / 16384.0;
                    }
                }
                renormalise_vector(x, n, gain);
            }
            if lowband.is_some() {
                fill
            } else {
                (1 << b_blocks) - 1
            }
        }
    }
}

pub fn deinterleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0.0f32; n];
    if hadamard {
        let offset = match stride {
            2 => 0,
            4 => 2,
            8 => 6,
            16 => 14,
            _ => 0,
        };
        let ordery = &ORDERY_TABLE[offset..offset + stride];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[ordery[i] as usize * n0 + j] = x[j * stride + i];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[i * n0 + j] = x[j * stride + i];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

pub fn interleave_hadamard(x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let mut tmp = vec![0.0f32; n];
    if hadamard {
        let offset = match stride {
            2 => 0,
            4 => 2,
            8 => 6,
            16 => 14,
            _ => 0,
        };
        let ordery = &ORDERY_TABLE[offset..offset + stride];
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[ordery[i] as usize * n0 + j];
            }
        }
    } else {
        for i in 0..stride {
            for j in 0..n0 {
                tmp[j * stride + i] = x[i * n0 + j];
            }
        }
    }
    x[..n].copy_from_slice(&tmp);
}

const ORDERY_TABLE: [i32; 30] = [
    1, 0, 3, 0, 2, 1, 7, 0, 4, 3, 6, 1, 5, 2, 15, 0, 8, 7, 12, 3, 11, 4, 14, 1, 9, 6, 13, 2, 10, 5,
];

fn quant_band_n1(
    ctx: &mut BandCtx,
    x: &mut [f32],
    y: Option<&mut [f32]>,
    lowband_out: Option<&mut [f32]>,
) -> u32 {
    let mut sign = 0;
    if ctx.remaining_bits >= 1 << BITRES {
        if ctx.encode {
            sign = if x[0] < 0.0 { 1 } else { 0 };
            ctx.rc.enc_bits(sign as u32, 1);
        } else {
            sign = ctx.rc.dec_bits(1) as i32;
        }
        ctx.remaining_bits -= 1 << BITRES;
    }
    if ctx.resynth {
        x[0] = if sign != 0 { -1.0 } else { 1.0 };
    }
    if let Some(y_val) = y {
        let mut y_sign = 0;
        if ctx.remaining_bits >= 1 << BITRES {
            if ctx.encode {
                y_sign = if y_val[0] < 0.0 { 1 } else { 0 };
                ctx.rc.enc_bits(y_sign as u32, 1);
            } else {
                y_sign = ctx.rc.dec_bits(1) as i32;
            }
            ctx.remaining_bits -= 1 << BITRES;
        }
        if ctx.resynth {
            y_val[0] = if y_sign != 0 { -1.0 } else { 1.0 };
        }
    }
    if let Some(l_out) = lowband_out {
        l_out[0] = x[0] / 16.0;
    }
    1
}

pub fn quant_band(
    ctx: &mut BandCtx,
    x: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    gain: f32,
    fill: u32,
) -> u32 {
    let n0 = n;
    let b0 = b_blocks;
    let long_blocks = b0 == 1;

    if n == 1 {
        return quant_band_n1(ctx, x, None, lowband_out);
    }

    let mut b_blocks = b_blocks;
    let mut n_b = n / b_blocks as usize;
    let mut time_divide = 0;
    let mut recombine = 0;
    let mut tf_change_local = ctx.tf_change;
    let mut fill = fill;

    if tf_change_local > 0 {
        recombine = tf_change_local;
    }

    // Make a mutable copy of lowband for transforms
    let mut lowband_buf: Option<Vec<f32>> = lowband.map(|lb| lb.to_vec());

    static BIT_INTERLEAVE_TABLE: [u8; 16] = [0, 1, 1, 1, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3];

    // Band recombining to increase frequency resolution
    for k in 0..recombine {
        if ctx.encode {
            haar1(x, n >> k, 1 << k);
        }
        if let Some(ref mut lb) = lowband_buf {
            haar1(lb, n >> k, 1 << k);
        }
        fill = (BIT_INTERLEAVE_TABLE[(fill & 0xF) as usize] as u32)
            | ((BIT_INTERLEAVE_TABLE[(fill >> 4) as usize] as u32) << 2);
    }
    b_blocks >>= recombine;
    n_b <<= recombine;

    // Increasing the time resolution
    while n_b & 1 == 0 && tf_change_local < 0 {
        if ctx.encode {
            haar1(x, n_b, b_blocks as usize);
        }
        if let Some(ref mut lb) = lowband_buf {
            haar1(lb, n_b, b_blocks as usize);
        }
        fill |= fill << b_blocks;
        b_blocks <<= 1;
        n_b >>= 1;
        time_divide += 1;
        tf_change_local += 1;
    }

    let b0_after = b_blocks;
    let n_b0 = n_b;

    // Reorganize samples in time order
    if b_blocks > 1 {
        if ctx.encode {
            deinterleave_hadamard(
                x,
                n_b >> recombine as usize,
                (b_blocks << recombine) as usize,
                long_blocks,
            );
        }
        if let Some(ref mut lb) = lowband_buf {
            deinterleave_hadamard(
                lb,
                n_b >> recombine as usize,
                (b_blocks << recombine) as usize,
                long_blocks,
            );
        }
    }

    let cm = quant_partition(
        ctx,
        x,
        n,
        b,
        b_blocks,
        lowband_buf.as_mut().map(|v| v.as_mut_slice()),
        lm,
        gain,
        fill,
    );

    // Undo transforms in resynth
    if ctx.resynth {
        let mut cm = cm;

        if b_blocks > 1 {
            interleave_hadamard(
                x,
                n_b >> recombine as usize,
                (b0_after << recombine) as usize,
                long_blocks,
            );
        }

        // Undo time-freq changes
        let mut n_b_undo = n_b0;
        let mut b_undo = b0_after;
        for _ in 0..time_divide {
            b_undo >>= 1;
            n_b_undo <<= 1;
            cm |= cm >> b_undo;
            haar1(x, n_b_undo, b_undo as usize);
        }

        static BIT_DEINTERLEAVE_TABLE: [u8; 16] = [
            0x00, 0x03, 0x0C, 0x0F, 0x30, 0x33, 0x3C, 0x3F, 0xC0, 0xC3, 0xCC, 0xCF, 0xF0, 0xF3,
            0xFC, 0xFF,
        ];
        for k in 0..recombine {
            cm = BIT_DEINTERLEAVE_TABLE[cm as usize & 0xF] as u32;
            haar1(x, n0 >> k, 1 << k);
        }
        let mut b_final = b0_after;
        b_final <<= recombine;

        // Scale output for later folding
        if let Some(lb_out) = lowband_out {
            // C: n = celt_sqrt(SHL32(EXTEND32(N0),22)) — in float this is sqrt(N0 * (1<<22))
            // But in float: EXTEND32 = identity, SHL32 = identity, so n = sqrt(N0)
            // Actually C uses celt_sqrt which is different... but in float mode celt_sqrt is just sqrtf
            let scale = (n0 as f32).sqrt();
            for j in 0..n0 {
                lb_out[j] = scale * x[j];
            }
        }
        cm &= (1u32 << b_final) - 1;
        return cm;
    }

    cm
}

pub fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, side: f32, n: usize) {
    for i in 0..n {
        let x_val = x[i] * mid;
        let y_val = y[i] * side;
        x[i] = x_val - y_val;
        y[i] = x_val + y_val;
    }
}

pub fn quant_band_stereo(
    ctx: &mut BandCtx,
    x: &mut [f32],
    y: &mut [f32],
    n: usize,
    b: i32,
    b_blocks: i32,
    lowband: Option<&mut [f32]>,
    lm: i32,
    lowband_out: Option<&mut [f32]>,
    _gain: f32,
    fill: u32,
) -> u32 {
    if n == 1 {
        return quant_band_n1(ctx, x, Some(y), lowband_out);
    }

    if ctx.encode {
        if ctx.band_e[ctx.i] < MIN_STEREO_ENERGY
            || ctx.band_e[ctx.m.nb_ebands + ctx.i] < MIN_STEREO_ENERGY
        {
            if ctx.band_e[ctx.i] > ctx.band_e[ctx.m.nb_ebands + ctx.i] {
                y.copy_from_slice(x);
            } else {
                x.copy_from_slice(y);
            }
        }
    }

    let mut sctx = SplitCtx {
        inv: false,
        imid: 0,
        iside: 0,
        delta: 0,
        itheta: 0,
        qalloc: 0,
    };
    let mut b_mut = b;
    let mut fill_mut = fill;
    compute_theta(
        ctx,
        &mut sctx,
        x,
        y,
        n,
        &mut b_mut,
        b_blocks,
        b_blocks,
        lm,
        true,
        &mut fill_mut,
    );

    let mid_gain = sctx.imid as f32 / 32768.0;
    let side_gain = sctx.iside as f32 / 32768.0;

    if n == 2 {
        let mut mbits = b_mut;
        let mut sbits = 0;
        if sctx.itheta != 0 && sctx.itheta != 16384 {
            sbits = 1 << BITRES;
        }
        mbits -= sbits;
        let c = sctx.itheta > 8192;
        ctx.remaining_bits -= sctx.qalloc + sbits;

        let mut sign = 0;
        if sbits != 0 {
            if ctx.encode {
                sign = if c {
                    if (y[0] * x[1] - y[1] * x[0]) < 0.0 {
                        1
                    } else {
                        0
                    }
                } else {
                    if (x[0] * y[1] - x[1] * y[0]) < 0.0 {
                        1
                    } else {
                        0
                    }
                };
                ctx.rc.enc_bits(sign as u32, 1);
            } else {
                sign = ctx.rc.dec_bits(1) as i32;
            }
        }
        let sign_val = (1 - 2 * sign) as f32;
        let cm = if c {
            let cm = quant_band(
                ctx,
                y,
                n,
                mbits,
                b_blocks,
                lowband,
                lm,
                lowband_out,
                1.0,
                fill,
            );
            x[0] = -sign_val * y[1];
            x[1] = sign_val * y[0];
            cm
        } else {
            let cm = quant_band(
                ctx,
                x,
                n,
                mbits,
                b_blocks,
                lowband,
                lm,
                lowband_out,
                1.0,
                fill,
            );
            y[0] = -sign_val * x[1];
            y[1] = sign_val * x[0];
            cm
        };

        if ctx.resynth {
            let x0 = x[0];
            let x1 = x[1];
            let y0 = y[0];
            let y1 = y[1];
            x[0] = mid_gain * x0 - side_gain * y0;
            x[1] = mid_gain * x1 - side_gain * y1;
            y[0] = mid_gain * x0 + side_gain * y0;
            y[1] = mid_gain * x1 + side_gain * y1;
        }
        return cm;
    }

    ctx.remaining_bits -= sctx.qalloc;
    let mut mbits = (0).max((b_mut - sctx.delta) / 2).min(b_mut);
    let mut sbits = b_mut - mbits;

    let mut rebalance = ctx.remaining_bits;
    let mut cm;

    if mbits >= sbits {
        cm = quant_band(
            ctx,
            x,
            n,
            mbits,
            b_blocks,
            lowband,
            lm,
            lowband_out,
            1.0,
            fill_mut,
        );
        rebalance = mbits - (rebalance - ctx.remaining_bits);
        if rebalance > (3 << 3) && sctx.itheta != 0 {
            sbits += rebalance - (3 << 3);
        }
        cm |= quant_band(
            ctx,
            y,
            n,
            sbits,
            b_blocks,
            None,
            lm,
            None,
            side_gain,
            fill_mut >> b_blocks,
        ) << (b_blocks >> 1);
    } else {
        cm = quant_band(
            ctx,
            y,
            n,
            sbits,
            b_blocks,
            None,
            lm,
            None,
            side_gain,
            fill_mut >> b_blocks,
        ) << (b_blocks >> 1);
        rebalance = sbits - (rebalance - ctx.remaining_bits);
        if rebalance > (3 << 3) && sctx.itheta != 16384 {
            mbits += rebalance - (3 << 3);
        }
        cm |= quant_band(
            ctx,
            x,
            n,
            mbits,
            b_blocks,
            lowband,
            lm,
            lowband_out,
            1.0,
            fill_mut,
        );
    }

    if ctx.resynth {
        stereo_merge(x, y, mid_gain, side_gain, n);
        if sctx.inv {
            for i in 0..n {
                y[i] = -y[i];
            }
        }
    }
    cm
}

pub fn quant_all_bands(
    encode: bool,
    m: &CeltMode,
    start: usize,
    end: usize,
    x: &mut [f32],
    mut y: Option<&mut [f32]>,
    collapse_masks: &mut [u32],
    band_e: &[f32],
    pulses: &[i32],
    short_blocks: bool,
    spread: i32,
    dual_stereo: &mut bool,
    intensity: usize,
    tf_res: &[i32],
    total_bits: i32,
    balance: &mut i32,
    rc: &mut RangeCoder,
    lm: i32,
    coded_bands: i32,
    resynth: bool,
) {
    let mut balance_val = *balance;
    let b_blocks = if short_blocks { 1 << lm } else { 1 };
    let c_channels = if y.is_some() { 2 } else { 1 };
    let m_val = 1usize << lm as usize;

    // Norm array for storing decoded band coefficients (for band folding)
    let norm_offset = m_val * (m.e_bands[start] as usize);
    let norm_size = m_val * (m.e_bands[m.nb_ebands - 1] as usize) - norm_offset;
    let mut norm = vec![0.0f32; norm_size];
    // TODO: norm2 for stereo second channel

    let mut lowband_offset: usize = 0;
    let mut update_lowband = true;
    let mut avoid_split_noise = b_blocks > 1;

    for i in start..end {
        let offset = m_val * (m.e_bands[i] as usize);
        let n = m_val * ((m.e_bands[i + 1] - m.e_bands[i]) as usize);
        let last = i == end - 1;

        // Use tell_frac for fractional bit precision (matches C's ec_tell_frac)
        let tell = rc.tell_frac();
        if i != start {
            balance_val -= tell;
        }
        let remaining_bits = total_bits - tell - 1;

        let mut b = 0i32;
        if i < coded_bands as usize {
            let curr_balance = celt_sudiv(balance_val, 3i32.min(coded_bands - i as i32));
            b = 0i32.max(16383i32.min((remaining_bits + 1).min(pulses[i] + curr_balance)));
        }

        // Norm position for this band in the norm array
        let norm_pos = m_val * (m.e_bands[i] as usize) - norm_offset;

        // Update lowband_offset for band folding
        let band_start = m_val * (m.e_bands[i] as usize);
        let bands_start = m_val * (m.e_bands[start] as usize);
        if resynth
            && (band_start as i32 - n as i32 >= bands_start as i32 || i == start + 1)
            && (update_lowband || lowband_offset == 0)
        {
            lowband_offset = i;
        }

        let tf_change = tf_res[i];

        // Compute effective_lowband and collapse masks from fold region
        let mut effective_lowband: i32 = -1;
        let mut x_cm: u32;
        let mut y_cm: u32;

        if n <= 64 && i < 5 {}

        if lowband_offset != 0 && (spread != SPREAD_AGGRESSIVE || b_blocks > 1 || tf_change < 0) {
            effective_lowband = 0i32.max(
                (m_val * m.e_bands[lowband_offset] as usize) as i32 - norm_offset as i32 - n as i32,
            );
            let el_abs = effective_lowband as usize + norm_offset;

            // Find fold region: bands overlapping [el_abs, el_abs+n)
            let mut fold_start = lowband_offset;
            loop {
                if fold_start == 0 {
                    break;
                }
                fold_start -= 1;
                if m_val * (m.e_bands[fold_start] as usize) <= el_abs {
                    break;
                }
            }
            let mut fold_end = lowband_offset;
            while fold_end < i && m_val * (m.e_bands[fold_end] as usize) < el_abs + n {
                fold_end += 1;
            }

            x_cm = 0;
            y_cm = 0;
            for fi in fold_start..fold_end {
                x_cm |= collapse_masks[fi * c_channels];
                y_cm |= collapse_masks[fi * c_channels + c_channels - 1];
            }
        } else {
            x_cm = (1u32 << b_blocks) - 1;
            y_cm = (1u32 << b_blocks) - 1;
        }

        let mut ctx = BandCtx {
            encode,
            m,
            i,
            band_e,
            rc,
            spread,
            remaining_bits,
            resynth,
            tf_change,
            intensity,
            theta_round: 0,
            avoid_split_noise,
            arch: 0,
            disable_inv: false,
        };

        if *dual_stereo && i == intensity {
            *dual_stereo = false;
        }

        // Prepare lowband scratch (clone from norm for folding)
        let mut lowband_scratch: Option<Vec<f32>> = if effective_lowband >= 0 {
            let lb_start = effective_lowband as usize;
            let lb_end = lb_start + n;
            if lb_end <= norm.len() {
                Some(norm[lb_start..lb_end].to_vec())
            } else {
                None
            }
        } else {
            None
        };

        let x_slice = &mut x[offset..offset + n];
        if *dual_stereo {
            let y_slice = &mut y.as_mut().unwrap()[offset..offset + n];
            let lb_x = lowband_scratch.as_mut().map(|v| v.as_mut_slice());
            let lb_out_x = if !last && norm_pos + n <= norm.len() {
                Some(&mut norm[norm_pos..norm_pos + n])
            } else {
                None
            };
            x_cm = quant_band(
                &mut ctx,
                x_slice,
                n,
                b / 2,
                b_blocks,
                lb_x,
                lm,
                lb_out_x,
                1.0,
                x_cm,
            );
            y_cm = quant_band(
                &mut ctx,
                y_slice,
                n,
                b / 2,
                b_blocks,
                None,
                lm,
                None,
                1.0,
                y_cm,
            );
        } else {
            if let Some(y_all) = y.as_mut() {
                let y_slice = &mut y_all[offset..offset + n];
                let lb = lowband_scratch.as_mut().map(|v| v.as_mut_slice());
                let lb_out = if !last && norm_pos + n <= norm.len() {
                    Some(&mut norm[norm_pos..norm_pos + n])
                } else {
                    None
                };
                x_cm = quant_band_stereo(
                    &mut ctx,
                    x_slice,
                    y_slice,
                    n,
                    b,
                    b_blocks,
                    lb,
                    lm,
                    lb_out,
                    1.0,
                    x_cm | y_cm,
                );
                y_cm = x_cm;
            } else {
                let lb = lowband_scratch.as_mut().map(|v| v.as_mut_slice());
                let lb_out = if !last && norm_pos + n <= norm.len() {
                    Some(&mut norm[norm_pos..norm_pos + n])
                } else {
                    None
                };
                x_cm = quant_band(&mut ctx, x_slice, n, b, b_blocks, lb, lm, lb_out, 1.0, x_cm);
                y_cm = x_cm;
            }
        }

        collapse_masks[i * c_channels] = x_cm;
        if c_channels == 2 {
            collapse_masks[i * c_channels + 1] = y_cm;
        }

        // CRITICAL: Update balance (was completely missing!)
        balance_val += pulses[i] + tell;

        // Track whether previous band had enough quality for folding
        update_lowband = b > ((n as i32) << BITRES);

        // After first band, no longer avoid split noise
        avoid_split_noise = false;
    }
    *balance = balance_val;
}

pub fn compute_band_energies(
    m: &CeltMode,
    x: &[f32],
    band_e: &mut [f32],
    end: usize,
    channels: usize,
    lm: usize,
) {
    let frame_size = m.short_mdct_size << lm;
    for c in 0..channels {
        for i in 0..end {
            let offset = (m.e_bands[i] as usize) << lm;
            let n = ((m.e_bands[i + 1] - m.e_bands[i]) as usize) << lm;
            let mut sum = 1e-15f32;
            let slice = &x[c * frame_size..];
            for j in 0..n {
                sum += slice[offset + j].powi(2);
            }
            band_e[c * m.nb_ebands + i] = sum.sqrt();
        }
    }
}

pub fn amp2log2(
    m: &CeltMode,
    eff_ebands: usize,
    end: usize,
    band_e: &[f32],
    band_log_e: &mut [f32],
    channels: usize,
) {
    for c in 0..channels {
        for i in 0..eff_ebands {
            let val = band_e[c * m.nb_ebands + i].max(1e-10);
            band_log_e[c * m.nb_ebands + i] = val.log2() - m.e_means[i];
        }
        for i in eff_ebands..end {
            band_log_e[c * m.nb_ebands + i] = -14.0;
        }
    }
}

pub fn log2amp(m: &CeltMode, end: usize, band_e: &mut [f32], band_log_e: &[f32], channels: usize) {
    for c in 0..channels {
        for i in 0..end {
            band_e[c * m.nb_ebands + i] =
                2.0f32.powf(band_log_e[c * m.nb_ebands + i] + m.e_means[i]);
        }
    }
}

pub fn normalise_bands(
    m: &CeltMode,
    freq: &[f32],
    x: &mut [f32],
    band_e: &[f32],
    end: usize,
    channels: usize,
    m_val: usize,
) {
    let lm = m_val.trailing_zeros() as usize;
    let frame_size = m.short_mdct_size << lm;
    for c in 0..channels {
        for i in 0..end {
            let offset = (m.e_bands[i] as usize) << lm;
            let n = ((m.e_bands[i + 1] - m.e_bands[i]) as usize) << lm;
            let norm = 1.0 / (1e-15 + band_e[c * m.nb_ebands + i]);
            for j in 0..n {
                x[c * frame_size + offset + j] = freq[c * frame_size + offset + j] * norm;
            }
        }
    }
}

pub fn denormalise_bands(
    m: &CeltMode,
    x: &[f32],
    freq: &mut [f32],
    band_e: &[f32],
    start: usize,
    end: usize,
    channels: usize,
    m_val: usize,
) {
    let lm = m_val.trailing_zeros() as usize;
    let frame_size = m.short_mdct_size << lm;

    for c in 0..channels {
        for i in start..end {
            let offset = (m.e_bands[i] as usize) << lm;
            let n = ((m.e_bands[i + 1] - m.e_bands[i]) as usize) << lm;
            let band_log = band_e[c * m.nb_ebands + i];
            let g = (2.0f32).powf(band_log + m.e_means[i]);
            for j in 0..n {
                freq[c * frame_size + offset + j] = x[c * frame_size + offset + j] * g;
            }
        }
    }
}

pub fn celt_lcg_rand(seed: u32) -> u32 {
    seed.wrapping_mul(1103515245).wrapping_add(12345)
}

pub fn renormalise_vector(x: &mut [f32], n: usize, gain: f32) {
    let mut e = 1e-15f32;
    for i in 0..n {
        e += x[i] * x[i];
    }
    let norm = gain / e.sqrt();
    for i in 0..n {
        x[i] *= norm;
    }
}

pub fn anti_collapse(
    m: &CeltMode,
    x_buf: &mut [f32],
    collapse_masks: &[u32],
    lm: i32,
    channels: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[f32],
    prev1_log_e: &[f32],
    prev2_log_e: &[f32],
    pulses: &[i32],
    mut seed: u32,
) -> u32 {
    for i in start..end {
        let n0 = (m.e_bands[i + 1] - m.e_bands[i]) as usize;
        let depth = if n0 > 0 {
            (1 + pulses[i]) / n0 as i32 >> lm
        } else {
            0
        };

        let thresh = 0.5 * (-(0.125 * depth as f32)).exp2();
        let sqrt_1 = 1.0 / ((n0 << lm) as f32).sqrt();

        for c in 0..channels {
            let p1 = prev1_log_e[c * m.nb_ebands + i];
            let p2 = prev2_log_e[c * m.nb_ebands + i];

            let (p1_adj, p2_adj) = if channels == 1 && prev1_log_e.len() >= 2 * m.nb_ebands {
                (
                    p1.max(prev1_log_e[m.nb_ebands + i]),
                    p2.max(prev2_log_e[m.nb_ebands + i]),
                )
            } else {
                (p1, p2)
            };

            let e_diff = log_e[c * m.nb_ebands + i] - p1_adj.min(p2_adj);
            let e_diff = e_diff.max(0.0);

            let mut r = 2.0 * (-e_diff).exp2();
            if lm == 3 {
                r *= 1.41421356f32;
            }
            r = r.min(thresh);
            r = r * sqrt_1;

            let x_offset = c * size + ((m.e_bands[i] as usize) << lm);
            let mut renormalize = false;
            for k in 0..(1 << lm) {
                if (collapse_masks[i * channels + c] & (1 << k)) == 0 {
                    for j in 0..n0 {
                        seed = celt_lcg_rand(seed);
                        x_buf[x_offset + (j << lm) + k] = if (seed & 0x8000) != 0 { r } else { -r };
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                renormalise_vector(&mut x_buf[x_offset..x_offset + (n0 << lm)], n0 << lm, 1.0);
            }
        }
    }
    seed
}
