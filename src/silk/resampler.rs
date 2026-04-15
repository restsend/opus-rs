use crate::silk::macros::*;

pub const SILK_RESAMPLER_1_3_COEFS: [i16; 20] = [
    16102, -15162, -13, 0, 20, 26, 5, -31, -43, -4, 65, 90, 7, -157, -248, -44, 593, 1583, 2612,
    3271,
];
const RESAMPLER_DOWN_ORDER_FIR2: usize = 36;
const RESAMPLER_DOWN_BATCH_48K: usize = 480;

#[derive(Clone)]
pub struct SilkResamplerDown1_3 {
    pub ar2: [i32; 2],
    pub fir: [i32; RESAMPLER_DOWN_ORDER_FIR2],
}

impl Default for SilkResamplerDown1_3 {
    fn default() -> Self {
        Self {
            ar2: [0; 2],
            fir: [0; RESAMPLER_DOWN_ORDER_FIR2],
        }
    }
}

pub fn silk_resampler_down_1_3(
    state: &mut SilkResamplerDown1_3,
    output: &mut [i16],
    input: &[i16],
) {
    let total_in = input.len();
    const BUF_SIZE: usize = RESAMPLER_DOWN_BATCH_48K + RESAMPLER_DOWN_ORDER_FIR2;
    let a0 = SILK_RESAMPLER_1_3_COEFS[0];
    let a1 = SILK_RESAMPLER_1_3_COEFS[1];
    let fir_coefs = &SILK_RESAMPLER_1_3_COEFS[2..];

    let mut in_pos = 0usize;
    let mut out_pos = 0usize;

    while in_pos < total_in {
        let n = (total_in - in_pos).min(RESAMPLER_DOWN_BATCH_48K);

        let mut buf = [0i32; BUF_SIZE];
        buf[..RESAMPLER_DOWN_ORDER_FIR2].copy_from_slice(&state.fir);

        let buf_out = &mut buf[RESAMPLER_DOWN_ORDER_FIR2..RESAMPLER_DOWN_ORDER_FIR2 + n];
        for k in 0..n {
            let out32 = state.ar2[0].wrapping_add((input[in_pos + k] as i32) << 8);
            buf_out[k] = out32;
            let out_q10 = out32 << 2;
            state.ar2[0] = state.ar2[1].wrapping_add(silk_smulwb(out_q10, a0 as i32));
            state.ar2[1] = silk_smulwb(out_q10, a1 as i32);
        }

        let n_out = n / 3;
        for j in 0..n_out {
            let bp = j * 3;
            let mut res_q6 = 0i32;

            let mut i = 0;
            while i < 18 {
                let coef0 = fir_coefs[i] as i32;
                let sym0 = buf[bp + i].wrapping_add(buf[bp + 35 - i]);
                res_q6 = res_q6.wrapping_add(silk_smulwb(sym0, coef0));

                if i + 1 < 18 {
                    let coef1 = fir_coefs[i + 1] as i32;
                    let sym1 = buf[bp + i + 1].wrapping_add(buf[bp + 35 - i - 1]);
                    res_q6 = res_q6.wrapping_add(silk_smulwb(sym1, coef1));
                }
                i += 2;
            }

            if out_pos < output.len() {
                output[out_pos] = silk_sat16(silk_rshift_round(res_q6, 6)) as i16;
                out_pos += 1;
            }
        }

        state
            .fir
            .copy_from_slice(&buf[n..n + RESAMPLER_DOWN_ORDER_FIR2]);

        in_pos += n;
    }
}

#[derive(Clone, Default)]
pub struct SilkResamplerDown1_6 {
    pub stage1: SilkResamplerDown1_3,
    pub stage2: [i32; 2],
}

pub fn silk_resampler_down_1_6(
    state: &mut SilkResamplerDown1_6,
    output: &mut [i16],
    input: &[i16],
) {
    const BATCH_IN: usize = RESAMPLER_DOWN_BATCH_48K;
    const BATCH_MID: usize = BATCH_IN / 3;

    let mut in_pos = 0usize;
    let mut out_pos = 0usize;

    while in_pos < input.len() {
        let n_in = (input.len() - in_pos).min(BATCH_IN);
        let n_mid = n_in / 3;
        let n_out = n_mid / 2;

        let mut mid = [0i16; BATCH_MID];
        silk_resampler_down_1_3(
            &mut state.stage1,
            &mut mid[..n_mid],
            &input[in_pos..in_pos + n_in],
        );
        silk_resampler_down2(
            &mut state.stage2,
            &mut output[out_pos..out_pos + n_out],
            &mid[..n_mid],
            n_mid as i32,
        );

        in_pos += n_in;
        out_pos += n_out;
    }
}

pub const SILK_RESAMPLER_DOWN2_0: i32 = 9872;
pub const SILK_RESAMPLER_DOWN2_1: i32 = 39809 - 65536;

pub const SILK_RESAMPLER_2_3_COEFS_LQ: [i16; 6] = [-2797, -6507, 4697, 10739, 1567, 8276];

const SILK_RESAMPLER_UP2_HQ_0: [i16; 3] = [1746, 14986, (39083 - 65536) as i16];
const SILK_RESAMPLER_UP2_HQ_1: [i16; 3] = [6854, 25769, (55542 - 65536) as i16];

const SILK_RESAMPLER_FRAC_FIR_12: [[i16; 4]; 12] = [
    [189, -600, 617, 30567],
    [117, -159, -1070, 29704],
    [52, 221, -2392, 28276],
    [-4, 529, -3350, 26341],
    [-48, 758, -3956, 23973],
    [-80, 905, -4235, 21254],
    [-99, 972, -4222, 18278],
    [-107, 967, -3957, 15143],
    [-103, 896, -3487, 11950],
    [-91, 773, -2865, 8798],
    [-71, 611, -2143, 5784],
    [-46, 425, -1375, 2996],
];

const RESAMPLER_MAX_BATCH_SIZE_MS: i32 = 10;
const RESAMPLER_ORDER_FIR_12: usize = 8;

const DELAY_MATRIX_DEC: [[i8; 6]; 3] =
    [[4, 0, 2, 0, 0, 0], [0, 9, 4, 7, 4, 4], [0, 3, 12, 7, 7, 7]];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResamplerMode {
    Copy,
    Up2HQ,
    IirFir,
}

#[derive(Clone)]
pub struct SilkResampler {
    s_iir: [i32; 6],

    s_fir: [i16; RESAMPLER_ORDER_FIR_12],

    delay_buf: [i16; 48],

    input_delay: i32,

    fs_in_khz: i32,

    fs_out_khz: i32,

    batch_size: i32,

    inv_ratio_q16: i32,

    mode: ResamplerMode,
}

impl Default for SilkResampler {
    fn default() -> Self {
        Self {
            s_iir: [0; 6],
            s_fir: [0; RESAMPLER_ORDER_FIR_12],
            delay_buf: [0; 48],
            input_delay: 0,
            fs_in_khz: 0,
            fs_out_khz: 0,
            batch_size: 0,
            inv_ratio_q16: 0,
            mode: ResamplerMode::Copy,
        }
    }
}

fn rate_id(rate_hz: i32) -> usize {
    match rate_hz {
        8000 => 0,
        12000 => 1,
        16000 => 2,
        24000 => 3,
        48000 => 4,
        _ => 5,
    }
}

impl SilkResampler {
    pub fn init(&mut self, fs_hz_in: i32, fs_hz_out: i32) -> i32 {
        *self = Self::default();

        let in_id = rate_id(fs_hz_in);
        let out_id = rate_id(fs_hz_out);

        if in_id > 2 || out_id > 5 {
            return -1;
        }

        self.input_delay = DELAY_MATRIX_DEC[in_id][out_id] as i32;
        self.fs_in_khz = fs_hz_in / 1000;
        self.fs_out_khz = fs_hz_out / 1000;
        self.batch_size = self.fs_in_khz * RESAMPLER_MAX_BATCH_SIZE_MS;

        if fs_hz_out == fs_hz_in {
            self.mode = ResamplerMode::Copy;
        } else if fs_hz_out == fs_hz_in * 2 {
            self.mode = ResamplerMode::Up2HQ;
        } else {
            self.mode = ResamplerMode::IirFir;
        }

        let up2x = if self.mode == ResamplerMode::IirFir {
            1
        } else {
            0
        };
        self.inv_ratio_q16 = ((((fs_hz_in as i64) << (14 + up2x)) / fs_hz_out as i64) << 2) as i32;

        while silk_smulww(self.inv_ratio_q16, fs_hz_out) < (fs_hz_in << up2x) {
            self.inv_ratio_q16 += 1;
        }

        0
    }

    pub fn process(&mut self, out: &mut [i16], input: &[i16], in_len: i32) -> i32 {
        if in_len < self.fs_in_khz {
            return -1;
        }

        let n_samples = self.fs_in_khz - self.input_delay;

        self.delay_buf[self.input_delay as usize..self.fs_in_khz as usize]
            .copy_from_slice(&input[..n_samples as usize]);

        match self.mode {
            ResamplerMode::Copy => {
                out[..self.fs_in_khz as usize]
                    .copy_from_slice(&self.delay_buf[..self.fs_in_khz as usize]);
                let remaining = (in_len - self.fs_in_khz) as usize;
                out[self.fs_out_khz as usize..self.fs_out_khz as usize + remaining]
                    .copy_from_slice(&input[n_samples as usize..n_samples as usize + remaining]);
            }
            ResamplerMode::Up2HQ => {
                silk_resampler_private_up2_hq(
                    &mut self.s_iir,
                    &mut out[..],
                    &self.delay_buf[..self.fs_in_khz as usize],
                    self.fs_in_khz,
                );
                silk_resampler_private_up2_hq(
                    &mut self.s_iir,
                    &mut out[self.fs_out_khz as usize..],
                    &input[n_samples as usize..],
                    in_len - self.fs_in_khz,
                );
            }
            ResamplerMode::IirFir => {
                self.iir_fir_resample(
                    out,
                    &self.delay_buf.clone(),
                    self.fs_in_khz,
                    &input[n_samples as usize..],
                    in_len - self.fs_in_khz,
                );
            }
        }

        let delay = self.input_delay as usize;
        if delay > 0 {
            let src_start = (in_len as usize).saturating_sub(delay);
            self.delay_buf[..delay].copy_from_slice(&input[src_start..src_start + delay]);
        }

        0
    }

    fn iir_fir_resample(
        &mut self,
        out: &mut [i16],
        first_block: &[i16],
        first_len: i32,
        rest: &[i16],
        rest_len: i32,
    ) {
        const MAX_BATCH_IN: usize = 480;
        const MAX_BUF: usize = 2 * MAX_BATCH_IN + RESAMPLER_ORDER_FIR_12;

        let mut out_idx = 0usize;

        let mut in_idx = 0usize;
        let mut remaining = first_len;

        while remaining > 0 {
            let n_samples_in = remaining.min(self.batch_size) as usize;

            let buf_len = 2 * n_samples_in + RESAMPLER_ORDER_FIR_12;
            let mut buf_arr = [0i16; MAX_BUF];
            let buf = &mut buf_arr[..buf_len];

            buf[..RESAMPLER_ORDER_FIR_12].copy_from_slice(&self.s_fir);

            silk_resampler_private_up2_hq(
                &mut self.s_iir,
                &mut buf[RESAMPLER_ORDER_FIR_12..],
                &first_block[in_idx..in_idx + n_samples_in],
                n_samples_in as i32,
            );

            let max_index_q16 = (n_samples_in as i32) << 17;
            let index_increment_q16 = self.inv_ratio_q16;

            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                let table_index = silk_smulwb(index_q16 & 0xFFFF, 12) as usize;
                let buf_idx = (index_q16 >> 16) as usize;

                let mut res_q15 = silk_smulbb(
                    buf[buf_idx] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[table_index][0] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 1] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[table_index][1] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 2] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[table_index][2] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 3] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[table_index][3] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 4] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][3] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 5] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][2] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 6] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][1] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 7] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][0] as i32,
                );

                if out_idx < out.len() {
                    out[out_idx] = silk_sat16(silk_rshift_round(res_q15, 15)) as i16;
                    out_idx += 1;
                }
                index_q16 += index_increment_q16;
            }

            in_idx += n_samples_in;
            remaining -= n_samples_in as i32;

            self.s_fir
                .copy_from_slice(&buf[2 * n_samples_in..2 * n_samples_in + RESAMPLER_ORDER_FIR_12]);
        }

        let mut in_idx = 0usize;
        let mut remaining = rest_len;

        while remaining > 0 {
            let n_samples_in = remaining.min(self.batch_size) as usize;

            let buf_len = 2 * n_samples_in + RESAMPLER_ORDER_FIR_12;
            let mut buf_arr = [0i16; MAX_BUF];
            let buf = &mut buf_arr[..buf_len];

            buf[..RESAMPLER_ORDER_FIR_12].copy_from_slice(&self.s_fir);

            silk_resampler_private_up2_hq(
                &mut self.s_iir,
                &mut buf[RESAMPLER_ORDER_FIR_12..],
                &rest[in_idx..in_idx + n_samples_in],
                n_samples_in as i32,
            );

            let max_index_q16 = (n_samples_in as i32) << 17;
            let index_increment_q16 = self.inv_ratio_q16;

            let mut index_q16 = 0i32;
            while index_q16 < max_index_q16 {
                let table_index = silk_smulwb(index_q16 & 0xFFFF, 12) as usize;
                let buf_idx = (index_q16 >> 16) as usize;

                let mut res_q15 = silk_smulbb(
                    buf[buf_idx] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[table_index][0] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 1] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[table_index][1] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 2] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[table_index][2] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 3] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[table_index][3] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 4] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][3] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 5] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][2] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 6] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][1] as i32,
                );
                res_q15 = silk_smlabb(
                    res_q15,
                    buf[buf_idx + 7] as i32,
                    SILK_RESAMPLER_FRAC_FIR_12[11 - table_index][0] as i32,
                );

                if out_idx < out.len() {
                    out[out_idx] = silk_sat16(silk_rshift_round(res_q15, 15)) as i16;
                    out_idx += 1;
                }
                index_q16 += index_increment_q16;
            }

            in_idx += n_samples_in;
            remaining -= n_samples_in as i32;

            self.s_fir
                .copy_from_slice(&buf[2 * n_samples_in..2 * n_samples_in + RESAMPLER_ORDER_FIR_12]);
        }
    }
}

pub fn silk_resampler_private_up2_hq(s: &mut [i32], out: &mut [i16], input: &[i16], len: i32) {
    for k in 0..len as usize {
        let in32 = (input[k] as i32) << 10;

        let y = in32 - s[0];
        let x = silk_smulwb(y, SILK_RESAMPLER_UP2_HQ_0[0] as i32);
        let out32_1 = s[0] + x;
        s[0] = in32 + x;

        let y = out32_1 - s[1];
        let x = silk_smulwb(y, SILK_RESAMPLER_UP2_HQ_0[1] as i32);
        let out32_2 = s[1] + x;
        s[1] = out32_1 + x;

        let y = out32_2 - s[2];
        let x = silk_smlawb(y, y, SILK_RESAMPLER_UP2_HQ_0[2] as i32);
        let out32_1 = s[2] + x;
        s[2] = out32_2 + x;

        out[2 * k] = silk_sat16(silk_rshift_round(out32_1, 10)) as i16;

        let y = in32 - s[3];
        let x = silk_smulwb(y, SILK_RESAMPLER_UP2_HQ_1[0] as i32);
        let out32_1 = s[3] + x;
        s[3] = in32 + x;

        let y = out32_1 - s[4];
        let x = silk_smulwb(y, SILK_RESAMPLER_UP2_HQ_1[1] as i32);
        let out32_2 = s[4] + x;
        s[4] = out32_1 + x;

        let y = out32_2 - s[5];
        let x = silk_smlawb(y, y, SILK_RESAMPLER_UP2_HQ_1[2] as i32);
        let out32_1 = s[5] + x;
        s[5] = out32_2 + x;

        out[2 * k + 1] = silk_sat16(silk_rshift_round(out32_1, 10)) as i16;
    }
}

pub fn silk_resampler_down2(s: &mut [i32], out: &mut [i16], input: &[i16], in_len: i32) {
    let len2 = in_len >> 1;
    let mut in32: i32;
    let mut out32: i32;
    let mut y: i32;
    let mut x: i32;

    for k in 0..len2 as usize {
        in32 = (input[2 * k] as i32) << 10;

        y = in32.wrapping_sub(s[0]);
        x = silk_smlawb(y, y, SILK_RESAMPLER_DOWN2_1);
        out32 = s[0].wrapping_add(x);
        s[0] = in32.wrapping_add(x);

        in32 = (input[2 * k + 1] as i32) << 10;

        y = in32.wrapping_sub(s[1]);
        x = silk_smulwb(y, SILK_RESAMPLER_DOWN2_0);
        out32 = out32.wrapping_add(s[1]);
        out32 = out32.wrapping_add(x);
        s[1] = in32.wrapping_add(x);

        out[k] = silk_sat16(silk_rshift_round(out32, 11)) as i16;
    }
}

pub fn silk_resampler_private_ar2(
    s: &mut [i32],
    out_q8: &mut [i32],
    input: &[i16],
    a_q14: &[i16],
    len: i32,
) {
    let mut out32: i32;
    for k in 0..len as usize {
        out32 = s[0].wrapping_add((input[k] as i32) << 8);
        s[0] = s[1].wrapping_add(silk_smlawb(out32, out32, a_q14[0] as i32));
        s[1] = silk_smlawb(0, out32, a_q14[1] as i32);
        out_q8[k] = out32;
    }
}

const RESAMPLER_MAX_BATCH_SIZE_IN: i32 = 480;
const ORDER_FIR: usize = 4;

pub fn silk_resampler_down2_3(s: &mut [i32], out: &mut [i16], input: &[i16], in_len: i32) {
    let mut n_samples_in: i32;
    let mut counter: i32;
    let mut res_q6: i32;
    let mut buf = [0i32; (RESAMPLER_MAX_BATCH_SIZE_IN as usize) + ORDER_FIR];
    let mut in_idx = 0;
    let mut out_idx = 0;
    let mut remaining_len = in_len;

    buf[0..ORDER_FIR].copy_from_slice(&s[0..ORDER_FIR]);

    while remaining_len > 0 {
        n_samples_in = remaining_len.min(RESAMPLER_MAX_BATCH_SIZE_IN);

        silk_resampler_private_ar2(
            &mut s[ORDER_FIR..ORDER_FIR + 2],
            &mut buf[ORDER_FIR..ORDER_FIR + n_samples_in as usize],
            &input[in_idx..in_idx + n_samples_in as usize],
            &SILK_RESAMPLER_2_3_COEFS_LQ,
            n_samples_in,
        );

        let mut buf_ptr = 0;
        counter = n_samples_in;
        while counter > 2 {
            res_q6 = silk_smulwb(buf[buf_ptr], SILK_RESAMPLER_2_3_COEFS_LQ[2] as i32);
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_ptr + 1],
                SILK_RESAMPLER_2_3_COEFS_LQ[3] as i32,
            );
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_ptr + 2],
                SILK_RESAMPLER_2_3_COEFS_LQ[5] as i32,
            );
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_ptr + 3],
                SILK_RESAMPLER_2_3_COEFS_LQ[4] as i32,
            );

            out[out_idx] = silk_sat16(silk_rshift_round(res_q6, 9)) as i16;
            out_idx += 1;

            res_q6 = silk_smulwb(buf[buf_ptr + 1], SILK_RESAMPLER_2_3_COEFS_LQ[4] as i32);
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_ptr + 2],
                SILK_RESAMPLER_2_3_COEFS_LQ[5] as i32,
            );
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_ptr + 3],
                SILK_RESAMPLER_2_3_COEFS_LQ[3] as i32,
            );
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_ptr + 4],
                SILK_RESAMPLER_2_3_COEFS_LQ[2] as i32,
            );

            out[out_idx] = silk_sat16(silk_rshift_round(res_q6, 8)) as i16;
            out_idx += 1;

            buf_ptr += 3;
            counter -= 3;
        }

        in_idx += n_samples_in as usize;
        remaining_len -= n_samples_in;

        if remaining_len > 0 {
            for i in 0..ORDER_FIR {
                buf[i] = buf[n_samples_in as usize + i];
            }
        } else {
            s[0..ORDER_FIR]
                .copy_from_slice(&buf[n_samples_in as usize..n_samples_in as usize + ORDER_FIR]);
            break;
        }
    }
}
