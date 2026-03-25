pub const EC_SYM_BITS: u32 = 8;
pub const EC_CODE_BITS: u32 = 32;
pub const EC_SYM_MAX: u32 = (1 << EC_SYM_BITS) - 1;
pub const EC_CODE_SHIFT: u32 = EC_CODE_BITS - EC_SYM_BITS - 1;
pub const EC_CODE_TOP: u32 = 1 << (EC_CODE_BITS - 1);
pub const EC_CODE_BOT: u32 = EC_CODE_TOP >> EC_SYM_BITS;
pub const EC_CODE_EXTRA: u32 = (EC_CODE_BITS - 2) % EC_SYM_BITS + 1;
pub const BITRES: i32 = 3;

#[derive(Clone)]
pub struct RangeCoder {
    pub buf: Vec<u8>,
    pub storage: u32,
    pub end_offs: u32,
    pub end_window: u32,
    pub nend_bits: i32,
    pub nbits_total: i32,
    pub offs: u32,
    pub rng: u32,
    pub val: u64,
    pub ext: u32,
    pub rem: i32,
    pub error: i32,
}

impl RangeCoder {
    pub fn new_encoder(size: u32) -> Self {
        RangeCoder {
            buf: vec![0; size as usize],
            storage: size,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: 33,
            offs: 0,
            rng: 1 << 31,
            val: 0,
            ext: 0,
            rem: -1,
            error: 0,
        }
    }

    pub fn new_decoder(data: &[u8]) -> Self {
        let storage = data.len() as u32;
        let buf = data.to_vec();
        let mut rc = RangeCoder {
            buf,
            storage,
            end_offs: 0,
            end_window: 0,
            nend_bits: 0,
            nbits_total: (EC_CODE_BITS + 1
                - ((EC_CODE_BITS - EC_CODE_EXTRA) / EC_SYM_BITS) * EC_SYM_BITS)
                as i32,
            offs: 0,
            rng: 1 << EC_CODE_EXTRA,
            val: 0,
            ext: 0,
            rem: 0,
            error: 0,
        };

        rc.rem = rc.read_byte() as i32;
        rc.val = (rc.rng - 1 - (rc.rem as u32 >> (EC_SYM_BITS - EC_CODE_EXTRA))) as u64;

        rc.normalize_decoder();
        rc
    }

    fn normalize_decoder(&mut self) {
        while self.rng <= EC_CODE_BOT {
            self.nbits_total += EC_SYM_BITS as i32;
            self.rng <<= EC_SYM_BITS;
            if self.rng == 0 {
                debug_assert!(
                    false,
                    "normalize_decoder: rng=0 after shift, corrupt bitstream"
                );
                self.error = 1;
                self.rng = 1;
                return;
            }

            let sym = self.rem;
            self.rem = self.read_byte() as i32;

            let combined_sym = ((sym << EC_SYM_BITS) | self.rem) >> (EC_SYM_BITS - EC_CODE_EXTRA);
            self.val = ((self.val << EC_SYM_BITS) + (EC_SYM_MAX & !combined_sym as u32) as u64)
                & (EC_CODE_TOP as u64 - 1);
        }
    }

    fn read_byte(&mut self) -> u8 {
        if self.offs < self.storage {
            let b = self.buf[self.offs as usize];
            self.offs += 1;
            b
        } else {
            0
        }
    }

    pub fn enc_uint(&mut self, fl: u32, ft: u32) {
        if ft > (1 << 8) {
            let mut ft = ft - 1;
            let s = 32 - ft.leading_zeros() as i32 - 8;
            self.enc_bits(fl & ((1 << s) - 1), s as u32);
            let fl = fl >> s;
            ft >>= s;
            ft += 1;
            self.encode(fl, fl + 1, ft);
        } else if ft > 1 {
            self.encode(fl, fl + 1, ft);
        }
    }

    pub fn dec_uint(&mut self, ft: u32) -> u32 {
        if ft > (1 << 8) {
            let mut ft = ft - 1;
            let s = 32 - ft.leading_zeros() as i32 - 8;
            let r = self.dec_bits(s as u32);
            ft >>= s;
            ft += 1;
            let fs = self.decode(ft);
            self.update(fs, fs + 1, ft);
            (fs << s) | r
        } else if ft > 1 {
            let fs = self.decode(ft);
            self.update(fs, fs + 1, ft);
            fs
        } else {
            0
        }
    }

    pub fn enc_bits(&mut self, val: u32, bits: u32) {
        if bits == 0 {
            return;
        }
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        if (used as u32) + bits > EC_CODE_BITS {
            while used >= EC_SYM_BITS as i32 {
                self.write_byte_at_end((window & EC_SYM_MAX) as u8);
                window >>= EC_SYM_BITS;
                used -= EC_SYM_BITS as i32;
            }
        }
        window |= (val & ((1 << bits) - 1)) << used;
        used += bits as i32;
        self.end_window = window;
        self.nend_bits = used;
        self.nbits_total += bits as i32;
    }

    pub fn dec_bits(&mut self, bits: u32) -> u32 {
        if bits == 0 {
            return 0;
        }
        let mut window = self.end_window;
        let mut used = self.nend_bits;
        if used < bits as i32 {
            loop {
                let byte = if self.end_offs < self.storage {
                    self.end_offs += 1;
                    self.buf[(self.storage - self.end_offs) as usize]
                } else {
                    0
                };
                window |= (byte as u32) << used;
                used += 8;
                if used > 32 - 8 {
                    break;
                }
            }
        }
        let ret = window & ((1 << bits) - 1);
        self.end_window = window >> bits;
        self.nend_bits = used - bits as i32;
        self.nbits_total += bits as i32;
        ret
    }

    pub fn tell_frac(&self) -> i32 {
        static CORRECTION: [u32; 8] = [35733, 38967, 42495, 46340, 50535, 55109, 60097, 65535];
        let nbits = self.nbits_total << BITRES;
        let mut l = 32 - self.rng.leading_zeros() as i32;
        let r = self.rng >> (l - 16);
        let mut b = (r >> 12) - 8;
        if b < 7 && r > CORRECTION[b as usize] {
            b += 1;
        }
        l = (l << 3) + b as i32;
        nbits - l
    }

    pub fn tell(&self) -> i32 {
        (self.tell_frac() + 7) >> 3
    }

    fn write_byte(&mut self, value: u8) {
        if self.offs + self.end_offs < self.storage {
            self.buf[self.offs as usize] = value;
            self.offs += 1;
        } else {
            self.error = 1;
        }
    }

    fn carry_out(&mut self, c: i32) {
        if c != EC_SYM_MAX as i32 {
            let carry = c >> EC_SYM_BITS;
            if self.rem >= 0 {
                self.write_byte((self.rem + carry) as u8);
            }
            if self.ext > 0 {
                let sym = (EC_SYM_MAX as i32 + carry) & EC_SYM_MAX as i32;
                for _ in 0..self.ext {
                    self.write_byte(sym as u8);
                }
                self.ext = 0;
            }
            self.rem = c & EC_SYM_MAX as i32;
        } else {
            self.ext += 1;
        }
    }

    pub fn encode(&mut self, fl: u32, fh: u32, ft: u32) {
        if ft == 0 {
            return;
        }
        let r = self.rng / ft;
        if fl > 0 {
            self.val += (self.rng - r * (ft - fl)) as u64;
            self.rng = r * (fh - fl);
        } else {
            self.rng -= r * (ft - fh);
        }
        self.normalize_encoder();
    }

    fn normalize_encoder(&mut self) {
        if self.rng == 0 {
            self.error = 1;
            self.rng = 1;
            return;
        }
        while self.rng <= EC_CODE_BOT {
            self.carry_out((self.val >> EC_CODE_SHIFT) as i32);
            self.val = (self.val << EC_SYM_BITS) & (EC_CODE_TOP as u64 - 1);
            self.rng <<= EC_SYM_BITS;
            self.nbits_total = self.nbits_total.wrapping_add(EC_SYM_BITS as i32);
        }
    }

    pub fn encode_bit_logp(&mut self, val: bool, logp: u32) {
        let s = self.rng >> logp;
        let r = self.rng - s;
        if val {
            self.val += r as u64;
            self.rng = s;
        } else {
            self.rng = r;
        }
        self.normalize_encoder();
    }

    pub fn encode_icdf(&mut self, s: i32, icdf: &[u8], ftb: u32) {
        let r = self.rng >> ftb;
        if s > 0 {
            let val = icdf[(s - 1) as usize] as u32;
            self.val += (self.rng as u64) - (r as u64 * val as u64);
            // The last symbol uses an implicit lower bound of 0
            let lower = icdf.get(s as usize).copied().unwrap_or(0) as u32;
            let diff = val - lower;
            debug_assert!(
                diff > 0,
                "encode_icdf: zero-probability symbol s={s}, icdf={icdf:?}, ftb={ftb} \
                 (icdf[{prev}]={val} == icdf[{s}]={lower})",
                prev = s - 1,
            );
            self.rng = r * diff;
        } else {
            let val = icdf[s as usize] as u32;
            let full = 1u32 << ftb;
            debug_assert!(
                val < full,
                "encode_icdf: zero-probability symbol s=0, icdf={icdf:?}, ftb={ftb} \
                 (icdf[0]={val} == 2^ftb={full}, symbol has zero probability)"
            );
            self.rng -= r * val;
        }
        self.normalize_encoder();
    }

    pub fn decode_bit_logp(&mut self, logp: u32) -> bool {
        let s = self.rng >> logp;
        let ret = self.val < s as u64;
        if !ret {
            self.val -= s as u64;
            self.rng -= s;
        } else {
            self.rng = s;
        }
        self.normalize_decoder();
        ret
    }

    /// Decode a symbol using an inverse CDF table.
    /// Uses do-while pattern like C opus for better performance.
    pub fn decode_icdf(&mut self, icdf: &[u8], ftb: u32) -> i32 {
        let mut s = self.rng;
        let d = self.val as u32;
        let r = s >> ftb;
        let mut ret = 0;
        let mut t;

        // Do-while loop: at least one iteration is guaranteed
        // This matches C opus behavior and is faster for typical small icdf tables
        loop {
            t = s;
            s = r * (icdf[ret] as u32);
            ret += 1;
            if d >= s {
                break;
            }
        }

        self.val = (d - s) as u64;
        self.rng = t - s;
        self.normalize_decoder();
        (ret - 1) as i32
    }

    pub fn decode(&mut self, ft: u32) -> u32 {
        let r = self.rng / ft;
        self.ext = r;
        let s = (self.val / r as u64) as u32;
        ft - ft.min(s + 1)
    }

    pub fn update(&mut self, fl: u32, fh: u32, ft: u32) {
        let s = self.ext * (ft - fh);
        self.val -= s as u64;
        self.rng = if fl > 0 {
            self.ext * (fh - fl)
        } else {
            self.rng - s
        };

        self.normalize_decoder();
    }

    pub fn laplace_encode(&mut self, value: &mut i32, fs: u32, decay: i32) {
        let mut val = *value;
        let mut fl = 0;
        let mut fs_val = fs;

        if val != 0 {
            let s = if val < 0 { -1 } else { 0 };
            val = (val + s) ^ s;
            fl = fs_val;
            fs_val = self.laplace_get_freq1(fs_val, decay);

            let mut i = 1;
            while fs_val > 0 && i < val {
                fs_val *= 2;
                fl += fs_val + 2;
                fs_val = ((fs_val as i32 * decay) >> 15) as u32;
                i += 1;
            }

            if fs_val == 0 {
                let ndi_max = 32768 - fl + 1 - 1;
                let ndi_max = (ndi_max as i32 - s) >> 1;
                let di = (val - i).min(ndi_max - 1);
                fl += (2 * di + 1 + s) as u32;
                fs_val = 1u32.min(32768 - fl);
                *value = (i + di + s) ^ s;
            } else {
                fs_val += 1;
                fl += fs_val & (!s as u32);
            }
        }
        self.encode(fl, fl + fs_val, 1 << 15);
    }

    fn laplace_get_freq1(&self, fs0: u32, decay: i32) -> u32 {
        let ft = 32768 - (2 * 16) - fs0;
        ((ft as i32 * (16384 - decay)) >> 15) as u32
    }

    pub fn laplace_decode(&mut self, fs: u32, decay: i32) -> i32 {
        let fm = self.decode(1 << 15);
        let mut fl = 0;
        let mut fs_val = fs;
        let mut val = 0;

        if fm >= fs_val {
            val += 1;
            fl = fs_val;
            fs_val = self.laplace_get_freq1(fs_val, decay) + 1;

            while fs_val > 1 && fm >= fl + 2 * fs_val {
                fs_val *= 2;
                fl += fs_val;
                fs_val = (((fs_val as i32 - 2) * decay) >> 15) as u32 + 1;
                val += 1;
            }

            if fs_val <= 1 {
                let di = (fm - fl) >> 1;
                val += di as i32;
                fl += 2 * di;
            }

            if fm < fl + fs_val {
                val = -val;
            } else {
                fl += fs_val;
            }
        }

        self.update(fl, fl + fs_val.min(32768 - fl), 1 << 15);
        val
    }

    fn write_byte_at_end(&mut self, value: u8) {
        if self.offs + self.end_offs < self.storage {
            self.end_offs += 1;
            let idx = (self.storage - self.end_offs) as usize;
            self.buf[idx] = value;
        } else {
            self.error = 1;
        }
    }

    pub fn patch_initial_bits(&mut self, val: u32, nbits: u32) {
        let shift = EC_SYM_BITS - nbits;
        let mask = ((1u32 << nbits) - 1) << shift;
        if self.offs > 0 {
            self.buf[0] = ((self.buf[0] as u32 & !mask) | (val << shift)) as u8;
        } else if self.rem >= 0 {
            self.rem = ((self.rem as u32 & !mask) | (val << shift)) as i32;
        } else if self.rng <= (EC_CODE_TOP >> nbits) {
            let mask64 = (mask as u64) << EC_CODE_SHIFT;
            self.val = (self.val & !mask64) | ((val as u64) << (EC_CODE_SHIFT + shift));
        } else {
            self.error = -1;
        }
    }

    pub fn done(&mut self) {
        let ilog = 32 - self.rng.leading_zeros();
        let mut l = (EC_CODE_BITS - ilog) as i32;
        let mut msk = (EC_CODE_TOP as u64 - 1) >> l;
        let mut end = (self.val + msk) & !msk;

        if (end | msk) >= self.val + self.rng as u64 {
            l += 1;
            msk >>= 1;
            end = (self.val + msk) & !msk;
        }

        while l > 0 {
            self.carry_out((end >> EC_CODE_SHIFT) as i32);
            end = (end << EC_SYM_BITS) & (EC_CODE_TOP as u64 - 1);
            l -= EC_SYM_BITS as i32;
        }

        if self.rem >= 0 || self.ext > 0 {
            self.carry_out(0);
        }

        let mut window = self.end_window;
        let mut used = self.nend_bits;
        while used >= EC_SYM_BITS as i32 {
            self.write_byte_at_end((window & EC_SYM_MAX) as u8);
            window >>= EC_SYM_BITS;
            used -= EC_SYM_BITS as i32;
        }

        if self.error == 0 {
            for i in self.offs..(self.storage - self.end_offs) {
                self.buf[i as usize] = 0;
            }

            if used > 0 {
                if self.end_offs >= self.storage {
                    self.error = -1;
                } else {
                    let idx = (self.storage - self.end_offs - 1) as usize;
                    self.buf[idx] |= window as u8;

                    self.end_offs += 1;
                }
            }
        }
    }

    pub fn finish(&mut self) -> Vec<u8> {
        self.done();

        let mut result = Vec::with_capacity((self.offs + self.end_offs) as usize);
        result.extend_from_slice(&self.buf[0..self.offs as usize]);
        result.extend_from_slice(
            &self.buf[(self.storage - self.end_offs) as usize..self.storage as usize],
        );
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_laplace() {
        let mut enc = RangeCoder::new_encoder(100);
        let mut val = -3;
        let fs = 100 << 7;
        let decay = 120 << 6;
        enc.laplace_encode(&mut val, fs, decay);
        enc.done();

        assert_eq!(enc.offs, 1);
        assert_eq!(enc.buf[0], 224);

        let mut dec = RangeCoder::new_decoder(&enc.buf[..enc.offs as usize]);
        let decoded_val = dec.laplace_decode(fs, decay);
        assert_eq!(decoded_val, -3);
    }

    #[test]
    fn test_icdf_consistency() {
        let mut enc = RangeCoder::new_encoder(1024);
        let icdf = [2, 1, 0];
        enc.encode_icdf(0, &icdf, 2);
        enc.encode_icdf(1, &icdf, 2);
        enc.encode_icdf(2, &icdf, 2);
        enc.done();
        let data = enc.buf[..enc.offs as usize].to_vec();

        let mut dec = RangeCoder::new_decoder(&data);
        let s0 = dec.decode_icdf(&icdf, 2);
        let s1 = dec.decode_icdf(&icdf, 2);
        let s2 = dec.decode_icdf(&icdf, 2);

        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
    }

    /// encode_icdf 的最后一个符号（s == icdf.len() - 1）之前会 panic（index OOB），
    /// 修复后应当正确编解码为最后一个符号索引。
    #[test]
    fn test_icdf_last_symbol_no_oob() {
        // ftb=8 → 总频率 256
        // 3 个符号，每个频率约 85，均不为 0
        // icdf 语义：icdf[i] = (总频率 - 前 i+1 个符号的累积频率)
        // symbol 0: 256 - 86 = 170  → icdf[0] = 170
        // symbol 1: 170 - 85 = 85   → icdf[1] = 85
        // symbol 2: 85  - 85 = 0    → icdf[2] = 0  (最后必须为 0)
        let icdf: &[u8] = &[170, 85, 0];
        let ftb = 8u32;

        // 对每个符号做一次 encode → done → decode，验证无 panic 且往返正确
        for sym in 0..3i32 {
            let mut enc = RangeCoder::new_encoder(256);
            enc.encode_icdf(sym, icdf, ftb); // sym==2 之前会 OOB panic
            enc.done();
            let data = enc.buf[..enc.offs as usize].to_vec();

            let mut dec = RangeCoder::new_decoder(&data);
            let decoded = dec.decode_icdf(icdf, ftb);
            assert_eq!(decoded, sym, "往返失败: 编码 symbol={sym} 解码得 {decoded}");
        }
    }

    /// decode_icdf 在 icdf 末尾不为 0 时会死循环/OOB；
    /// 用标准（末尾为 0）的表验证解码器正常终止。
    #[test]
    fn test_icdf_decode_terminates() {
        // 使用真实 Opus 风格的 ICDF 表（末尾必须为 0）
        // ftb=8，总频率=256；四个等概率符号各占 64
        let icdf: &[u8] = &[192, 128, 64, 0];
        let ftb = 8u32;

        let symbols = [0i32, 1, 2, 3];
        let mut enc = RangeCoder::new_encoder(256);
        for &s in &symbols {
            enc.encode_icdf(s, icdf, ftb);
        }
        enc.done();
        let data = enc.buf[..enc.offs as usize].to_vec();

        let mut dec = RangeCoder::new_decoder(&data);
        for &expected in &symbols {
            let got = dec.decode_icdf(icdf, ftb);
            assert_eq!(got, expected, "解码器输出 {got}，期望 {expected}");
        }
    }

    #[test]
    fn test_bits_only() {
        let mut enc = RangeCoder::new_encoder(1024);

        enc.enc_bits(1, 1);
        enc.enc_bits(5, 3);
        enc.enc_bits(7, 3);
        enc.enc_bits(0, 2);

        let data = enc.finish();
        let mut dec = RangeCoder::new_decoder(&data);

        let b1 = dec.dec_bits(1);
        let b2 = dec.dec_bits(3);
        let b3 = dec.dec_bits(3);
        let b4 = dec.dec_bits(2);

        assert_eq!(b1, 1);
        assert_eq!(b2, 5);
        assert_eq!(b3, 7);
        assert_eq!(b4, 0);
    }

    #[test]
    fn test_interleaved_bits_entropy() {
        let mut enc = RangeCoder::new_encoder(1024);

        enc.enc_bits(1, 1);

        enc.encode(10, 20, 100);

        enc.enc_bits(5, 3);

        enc.encode(50, 60, 100);

        let data = enc.finish();

        let mut dec = RangeCoder::new_decoder(&data);

        let b1 = dec.dec_bits(1);
        let d1 = dec.decode(100);
        dec.update(10, 20, 100);
        let b2 = dec.dec_bits(3);
        let d2 = dec.decode(100);
        dec.update(50, 60, 100);

        assert_eq!(b1, 1);
        assert!((10..20).contains(&d1), "d1={} expected in [10, 20)", d1);
        assert_eq!(b2, 5);
        assert!((50..60).contains(&d2), "d2={} expected in [50, 60)", d2);
    }
}
