#[inline(always)]
pub fn silk_rshift64(a: i64, shift: i32) -> i64 {
    a >> shift
}

#[inline(always)]
pub fn silk_lshift_sat32(a: i32, shift: i32) -> i32 {
    let result = (a as i64) << shift;
    if result > i32::MAX as i64 {
        i32::MAX
    } else if result < i32::MIN as i64 {
        i32::MIN
    } else {
        result as i32
    }
}

#[inline(always)]
pub fn silk_add_pos_sat32(a: i32, b: i32) -> i32 {
    let res = (a as i64) + (b as i64);
    if res > i32::MAX as i64 {
        i32::MAX
    } else if res < i32::MIN as i64 {
        i32::MIN
    } else {
        res as i32
    }
}

#[inline(always)]
pub fn silk_smlaww_64(a: i64, b: i64, c: i32) -> i64 {
    a.wrapping_add((b * (c as i64)) >> 16)
}

#[inline(always)]
pub fn silk_smulww(a: i32, b: i32) -> i32 {
    (((a as i64) * (b as i64)) >> 16) as i32
}

#[inline(always)]
pub fn silk_smlaww(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(silk_smulww(b, c))
}

#[inline(always)]
pub fn silk_rshift_round(a: i32, shift: i32) -> i32 {
    if shift <= 0 {
        return a << -shift;
    }
    if shift >= 31 {
        return 0;
    }
    (a >> shift) + ((a >> (shift - 1)) & 1)
}

#[inline(always)]
pub fn silk_rshift_round64(a: i64, shift: i32) -> i64 {
    if shift <= 0 {
        return a << -shift;
    }
    (a >> shift) + ((a >> (shift - 1)) & 1)
}

#[inline(always)]
pub fn silk_add_rshift(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b >> shift)
}

#[inline(always)]
pub fn silk_div32_16(a: i32, b: i32) -> i32 {
    a / b
}

#[inline(always)]
pub fn silk_div32_varq(a32: i32, b32: i32, qres: i32) -> i32 {
    debug_assert!(b32 != 0);
    debug_assert!(qres >= 0);

    let a_headrm = if a32 == 0 {
        31
    } else {
        (a32.wrapping_abs().leading_zeros() as i32) - 1
    };
    let a32_nrm = if a_headrm >= 0 {
        a32 << a_headrm
    } else {
        a32 >> (-a_headrm)
    };
    let b_headrm = if b32 == 0 {
        31
    } else {
        (b32.wrapping_abs().leading_zeros() as i32) - 1
    };
    let b32_nrm = if b_headrm >= 0 {
        b32 << b_headrm
    } else {
        b32 >> (-b_headrm)
    };

    let b32_inv = silk_div32_16(i32::MAX >> 2, b32_nrm >> 16);

    let mut result = silk_smulwb(a32_nrm, b32_inv);

    let a32_nrm2 =
        a32_nrm.wrapping_sub((silk_smmul(b32_nrm, result) as u32).wrapping_shl(3) as i32);

    result = silk_smlawb(result, a32_nrm2, b32_inv);

    let lshift = 29 + a_headrm - b_headrm - qres;
    if lshift < 0 {
        silk_lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

#[inline(always)]
pub fn silk_div32(a: i32, b: i32) -> i32 {
    a / b
}

#[inline(always)]
pub fn silk_limit(a: i32, limit_low: i32, limit_high: i32) -> i32 {
    silk_limit_32(a, limit_low, limit_high)
}

#[inline(always)]
pub fn silk_limit_int(a: i32, limit_low: i32, limit_high: i32) -> i32 {
    silk_limit_32(a, limit_low, limit_high)
}

#[inline(always)]
pub fn silk_smulbb(a: i32, b: i32) -> i32 {
    (a as i16 as i32) * (b as i16 as i32)
}

#[inline(always)]
pub fn silk_smulwb(a: i32, b: i32) -> i32 {
    (((a as i64) * (b as i16 as i64)) >> 16) as i32
}

#[inline(always)]
pub fn silk_smlawb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(silk_smulwb(b, c))
}

#[inline(always)]
pub fn silk_smulwt(a: i32, b: i32) -> i32 {
    ((a as i64 * (b as i64 >> 16)) >> 16) as i32
}

#[inline(always)]
pub fn silk_smlawt(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(silk_smulwt(b, c))
}

#[inline(always)]
pub fn silk_smlabb(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add((b as i16 as i32).wrapping_mul(c as i16 as i32))
}

#[inline(always)]
pub fn silk_mla(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(b.wrapping_mul(c))
}

#[inline(always)]
pub fn silk_mul(a: i32, b: i32) -> i32 {
    a.wrapping_mul(b)
}

#[inline(always)]
pub fn silk_rshift(a: i32, shift: i32) -> i32 {
    a >> shift
}

#[inline(always)]
pub fn silk_rshift32(a: i32, shift: i32) -> i32 {
    a >> shift
}

#[inline(always)]
pub fn silk_lshift(a: i32, shift: i32) -> i32 {
    a << shift
}

#[inline(always)]
pub fn silk_sub_rshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_sub(b >> shift)
}

#[inline(always)]
pub fn silk_smulll(a: i64, b: i64) -> i64 {
    a.wrapping_mul(b)
}

#[inline(always)]
pub fn silk_smull(a: i32, b: i32) -> i64 {
    (a as i64) * (b as i64)
}

#[inline(always)]
pub fn silk_smmul(a: i32, b: i32) -> i32 {
    (((a as i64) * (b as i64)) >> 32) as i32
}

#[inline(always)]
pub fn silk_clz32(a: i32) -> i32 {
    a.leading_zeros() as i32
}

#[inline(always)]
pub fn silk_clz64(a: i64) -> i32 {
    a.leading_zeros() as i32
}

#[inline(always)]
pub fn silk_inverse32_varq(b32: i32, qres: i32) -> i32 {
    if b32 == 0 {
        return i32::MAX;
    }

    let b_headrm = (b32.wrapping_abs().leading_zeros() as i32) - 1;
    let b32_nrm = if b_headrm >= 0 {
        b32 << b_headrm
    } else {
        b32 >> (-b_headrm)
    };

    let b32_inv = silk_div32_16(i32::MAX >> 2, b32_nrm >> 16);

    let result = b32_inv << 16;

    let err_q32 = ((1i32 << 29) - silk_smulwb(b32_nrm, b32_inv)) << 3;
    let result = silk_smlaww(result, err_q32, b32_inv);

    let lshift = 61 - b_headrm - qres;
    if lshift <= 0 {
        silk_lshift_sat32(result, -lshift)
    } else if lshift < 32 {
        result >> lshift
    } else {
        0
    }
}

#[inline(always)]
pub fn silk_clz_frac(input: i32, lz: &mut i32, frac_q7: &mut i32) {
    let lzeros = silk_clz32(input);
    *lz = lzeros;
    *frac_q7 = (input.rotate_right((24 - lzeros) as u32)) & 0x7f;
}

#[inline(always)]
pub fn silk_sqrt_approx(x: i32) -> i32 {
    let mut y: i32;
    let mut lz = 0;
    let mut frac_q7 = 0;

    if x <= 0 {
        return 0;
    }

    silk_clz_frac(x, &mut lz, &mut frac_q7);

    if (lz & 1) != 0 {
        y = 32768;
    } else {
        y = 46214;
    }

    y >>= lz >> 1;

    y = silk_smlawb(y, y, silk_smulbb(213, frac_q7));

    y
}

#[inline(always)]
pub fn silk_min_int(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

#[inline(always)]
pub fn silk_max_int(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

#[inline(always)]
pub fn silk_max_32(a: i32, b: i32) -> i32 {
    silk_max_int(a, b)
}

#[inline(always)]
pub fn silk_sat16(a: i32) -> i32 {
    if a > i16::MAX as i32 {
        i16::MAX as i32
    } else if a < i16::MIN as i32 {
        i16::MIN as i32
    } else {
        a
    }
}

#[inline(always)]
pub fn silk_add_sat16(a: i16, b: i16) -> i16 {
    let res = (a as i32) + (b as i32);
    if res > i16::MAX as i32 {
        i16::MAX
    } else if res < i16::MIN as i32 {
        i16::MIN
    } else {
        res as i16
    }
}

#[inline(always)]
pub fn silk_add_sat32(a: i32, b: i32) -> i32 {
    let res = (a as i64) + (b as i64);
    if res > i32::MAX as i64 {
        i32::MAX
    } else if res < i32::MIN as i64 {
        i32::MIN
    } else {
        res as i32
    }
}

#[inline(always)]
pub fn silk_rand(seed: i32) -> i32 {
    (907633515u32.wrapping_add((seed as u32).wrapping_mul(196314165u32))) as i32
}

#[inline(always)]
pub fn silk_add32_ovflw(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

#[inline(always)]
pub fn silk_sub32_ovflw(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

#[inline(always)]
pub fn silk_add32(a: i32, b: i32) -> i32 {
    a.wrapping_add(b)
}

#[inline(always)]
pub fn silk_sub32(a: i32, b: i32) -> i32 {
    a.wrapping_sub(b)
}

#[inline(always)]
pub fn silk_sub_sat32(a1: i32, a2: i32) -> i32 {
    let res = (a1 as i64) - (a2 as i64);
    if res > i32::MAX as i64 {
        i32::MAX
    } else if res < i32::MIN as i64 {
        i32::MIN
    } else {
        res as i32
    }
}

#[inline(always)]
pub fn silk_sub_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_sub(b << shift)
}

#[inline(always)]
pub fn silk_limit_32(a: i32, low: i32, high: i32) -> i32 {
    if a < low {
        low
    } else if a > high {
        high
    } else {
        a
    }
}

#[inline(always)]
pub fn silk_add_lshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b << shift)
}

#[inline(always)]
pub fn silk_min_32(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

#[inline(always)]
pub fn silk_add_rshift32(a: i32, b: i32, shift: i32) -> i32 {
    a.wrapping_add(b >> shift)
}

#[inline(always)]
pub fn silk_lin2log(gain_q16: i32) -> i32 {
    if gain_q16 <= 0 {
        return 0;
    }
    let mut lz = 0;
    let mut frac_q7 = 0;
    silk_clz_frac(gain_q16, &mut lz, &mut frac_q7);

    silk_smlawb(frac_q7, silk_mul(frac_q7, 128 - frac_q7), 179) + silk_lshift(31 - lz, 7)
}

#[inline(always)]
pub fn silk_log2lin(log_gain_q7: i32) -> i32 {
    if log_gain_q7 < 0 {
        return 0;
    }
    if log_gain_q7 >= 3967 {
        return i32::MAX;
    }
    let out = silk_lshift(1, log_gain_q7 >> 7);
    let frac_q7 = log_gain_q7 & 0x7F;

    let val = silk_smlawb(frac_q7, silk_smulbb(frac_q7, 128 - frac_q7), -174);
    if log_gain_q7 < 2048 {
        silk_add_rshift32(out, silk_mul(out, val), 7)
    } else {
        out + silk_mul(out >> 7, val)
    }
}
