#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

pub mod bands;
pub mod celt;
pub mod celt_lpc;
pub mod hp_cutoff;
pub mod kiss_fft;
pub mod mdct;
pub mod modes;
pub mod pitch;
pub mod pvq;
pub mod quant_bands;
pub mod range_coder;
pub mod rate;
pub mod silk;

pub use silk::{SilkResampler, SilkResamplerDown1_3, SilkResamplerDown1_6};

pub use celt::{CeltDecoder, CeltEncoder};
use hp_cutoff::hp_cutoff;
use range_coder::RangeCoder;
use silk::control_codec::silk_control_encoder;
use silk::enc_api::silk_encode;
use silk::init_encoder::silk_init_encoder;
use silk::lin2log::silk_lin2log;
use silk::log2lin::silk_log2lin;
use silk::macros::*;
use silk::resampler::{silk_resampler_down2, silk_resampler_down2_3};
use silk::structs::SilkEncoderState;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Application {
    Voip = 2048,
    Audio = 2049,
    RestrictedLowDelay = 2051,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bandwidth {
    Auto = -1000,
    Narrowband = 1101,
    Mediumband = 1102,
    Wideband = 1103,
    Superwideband = 1104,
    Fullband = 1105,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpusMode {
    SilkOnly,
    Hybrid,
    CeltOnly,
}

pub struct OpusEncoder {
    celt_enc: CeltEncoder,
    silk_enc: Box<SilkEncoderState>,
    application: Application,
    sampling_rate: i32,
    channels: usize,
    bandwidth: Bandwidth,
    pub bitrate_bps: i32,
    pub complexity: i32,
    pub use_cbr: bool,

    pub use_inband_fec: bool,

    pub packet_loss_perc: i32,
    silk_initialized: bool,
    mode: OpusMode,
    prev_enc_mode: Option<OpusMode>,

    variable_hp_smth2_q15: i32,
    hp_mem: Vec<i32>,

    buf_filtered: Vec<i16>,
    buf_silk_input: Vec<i16>,
    buf_stereo_mid: Vec<i16>,
    buf_stereo_side: Vec<i16>,
    buf_celt_input: Vec<f32>,
    down2_state_first: [i32; 2],
    down2_state_second: [i32; 2],
    down2_3_state: [i32; 6],
    down_1_3_state: silk::resampler::SilkResamplerDown1_3,

    rc: RangeCoder,
}

fn compute_equiv_rate(
    bitrate: i32,
    channels: usize,
    frame_rate: i32,
    vbr: bool,
    complexity: i32,
    loss: i32,
) -> i32 {
    let mut equiv = bitrate;
    if frame_rate > 50 {
        equiv -= (40 * channels as i32 + 20) * (frame_rate - 50);
    }
    if !vbr {
        equiv -= equiv / 12;
    }
    equiv = equiv * (90 + complexity) / 100;
    if loss > 0 {
        equiv -= equiv * loss / (12 * loss + 20);
    }
    equiv
}

fn compute_mode_threshold(
    application: Application,
    channels: usize,
    prev_was_celt: bool,
    has_prev_mode: bool,
    voice_est: i32,
) -> i32 {
    let mode_voice = if channels == 1 { 64000 } else { 44000 };
    let mode_music = 10000;

    let diff = mode_voice - mode_music;
    let offset = (voice_est * voice_est * diff) >> 14;
    let mut threshold = mode_music + offset;

    if application == Application::Voip {
        threshold += 8000;
    }

    if has_prev_mode {
        if prev_was_celt {
            threshold -= 4000;
        } else {
            threshold += 4000;
        }
    }

    if application == Application::RestrictedLowDelay {
        threshold = 0;
    }

    threshold
}

fn compute_silk_rate_for_hybrid(rate_bps: i32, frame20ms: bool) -> i32 {
    const RATE_TABLE: &[(i32, i32, i32)] = &[
        (0, 0, 0),
        (12000, 10000, 10000),
        (16000, 13500, 13500),
        (20000, 16000, 16000),
        (24000, 18000, 18000),
        (32000, 22000, 22000),
        (64000, 38000, 38000),
    ];
    let n = RATE_TABLE.len();
    let mut i = 1;
    while i < n && RATE_TABLE[i].0 <= rate_bps {
        i += 1;
    }
    if i == n {
        let (x_last, r10_last, r20_last) = RATE_TABLE[n - 1];
        let base = if frame20ms { r20_last } else { r10_last };
        base + (rate_bps - x_last) / 2
    } else {
        let (x0, lo10, lo20) = RATE_TABLE[i - 1];
        let (x1, hi10, hi20) = RATE_TABLE[i];
        let (lo, hi) = if frame20ms {
            (lo20, hi20)
        } else {
            (lo10, hi10)
        };
        (lo * (x1 - rate_bps) + hi * (rate_bps - x0)) / (x1 - x0)
    }
}

#[cfg(test)]
mod silk_rate_tests {
    use super::compute_silk_rate_for_hybrid;

    #[test]
    fn test_reference_table_exact_entries() {
        assert_eq!(compute_silk_rate_for_hybrid(12000, true), 10000);
        assert_eq!(compute_silk_rate_for_hybrid(16000, true), 13500);
        assert_eq!(compute_silk_rate_for_hybrid(20000, true), 16000);
        assert_eq!(compute_silk_rate_for_hybrid(24000, true), 18000);
        assert_eq!(compute_silk_rate_for_hybrid(32000, true), 22000);
        assert_eq!(compute_silk_rate_for_hybrid(64000, true), 38000);
    }

    #[test]
    fn test_32kbps_gives_22kbps_silk() {
        assert_eq!(compute_silk_rate_for_hybrid(32000, true), 22000);
    }

    #[test]
    fn test_interpolation_between_table_entries() {
        let r = compute_silk_rate_for_hybrid(18000, true);
        assert_eq!(r, 14750);
    }

    #[test]
    fn test_above_table_max_gives_half_extra() {
        let r = compute_silk_rate_for_hybrid(72000, true);
        assert_eq!(r, 38000 + (72000 - 64000) / 2);
    }
}

impl OpusEncoder {
    pub fn new(
        sampling_rate: i32,
        channels: usize,
        application: Application,
    ) -> Result<Self, &'static str> {
        if ![8000, 12000, 16000, 24000, 48000].contains(&sampling_rate) {
            return Err("Invalid sampling rate");
        }
        if ![1, 2].contains(&channels) {
            return Err("Invalid number of channels");
        }

        let mode = modes::default_mode();
        let celt_enc = CeltEncoder::new(mode, channels);

        let mut silk_enc = Box::new(SilkEncoderState::default());
        if silk_init_encoder(&mut silk_enc, 0) != 0 {
            return Err("SILK encoder initialization failed");
        }

        let (opus_mode, bw) = match application {
            Application::Voip => {
                let bw = match sampling_rate {
                    8000 => Bandwidth::Narrowband,
                    12000 => Bandwidth::Mediumband,
                    16000 => Bandwidth::Wideband,
                    24000 => Bandwidth::Superwideband,
                    48000 => Bandwidth::Fullband,
                    _ => Bandwidth::Narrowband,
                };

                let mode = if sampling_rate > 16000 {
                    OpusMode::Hybrid
                } else {
                    OpusMode::SilkOnly
                };
                (mode, bw)
            }
            Application::RestrictedLowDelay => {
                let bw = match sampling_rate {
                    8000 => Bandwidth::Narrowband,
                    12000 => Bandwidth::Mediumband,
                    16000 => Bandwidth::Wideband,
                    24000 => Bandwidth::Superwideband,
                    _ => Bandwidth::Fullband,
                };
                (OpusMode::CeltOnly, bw)
            }
            Application::Audio => {
                if sampling_rate <= 16000 {
                    let bw = match sampling_rate {
                        8000 => Bandwidth::Narrowband,
                        12000 => Bandwidth::Mediumband,
                        _ => Bandwidth::Wideband,
                    };
                    (OpusMode::SilkOnly, bw)
                } else {
                    let bw = match sampling_rate {
                        24000 => Bandwidth::Superwideband,
                        _ => Bandwidth::Fullband,
                    };
                    (OpusMode::Hybrid, bw)
                }
            }
        };

        use silk::lin2log::silk_lin2log;
        let variable_hp_smth2_q15 = silk_lin2log(60) << 8;

        Ok(Self {
            celt_enc,
            silk_enc,
            application,
            sampling_rate,
            channels,
            bandwidth: bw,
            bitrate_bps: 64000,
            complexity: 9,
            use_cbr: false,
            use_inband_fec: false,
            packet_loss_perc: 0,
            silk_initialized: false,
            prev_enc_mode: None,
            mode: opus_mode,
            variable_hp_smth2_q15,
            hp_mem: vec![0; channels * 2],

            buf_filtered: Vec::new(),
            buf_silk_input: Vec::new(),
            buf_stereo_mid: Vec::new(),
            buf_stereo_side: Vec::new(),
            buf_celt_input: Vec::new(),
            down2_state_first: [0; 2],
            down2_state_second: [0; 2],
            down2_3_state: [0; 6],
            down_1_3_state: silk::resampler::SilkResamplerDown1_3::default(),
            rc: RangeCoder::new_encoder(1),
        })
    }

    pub fn enable_hybrid_mode(&mut self) -> Result<(), &'static str> {
        if self.sampling_rate != 24000 && self.sampling_rate != 48000 {
            return Err("Hybrid mode requires 24kHz or 48kHz sampling rate");
        }
        let bw = if self.sampling_rate == 48000 {
            Bandwidth::Fullband
        } else {
            Bandwidth::Superwideband
        };
        self.mode = OpusMode::Hybrid;
        self.bandwidth = bw;
        self.silk_initialized = false;
        Ok(())
    }

    pub fn encode(
        &mut self,
        input: &[f32],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        if output.len() < 2 {
            return Err("Output buffer too small");
        }

        let frame_rate = frame_rate_from_params(self.sampling_rate, frame_size)
            .ok_or("Invalid frame size for sampling rate")?;

        // Mode selection: match C's opus_encode_native() behavior.
        // C reference auto-selects between SILK_ONLY and CELT_ONLY; Hybrid is
        // produced afterwards by bandwidth overrides (SILK-only + FB/SWB → Hybrid).
        let mut mode = if self.application == Application::RestrictedLowDelay {
            OpusMode::CeltOnly
        } else {
            let equiv = compute_equiv_rate(
                self.bitrate_bps,
                self.channels,
                frame_rate,
                !self.use_cbr,
                self.complexity,
                self.packet_loss_perc,
            );
            let prev_was_celt = self.prev_enc_mode == Some(OpusMode::CeltOnly);
            let has_prev_mode = self.prev_enc_mode.is_some();
            let voice_est = match self.application {
                Application::Voip => 115,
                Application::Audio => 48,
                Application::RestrictedLowDelay => 0,
            };
            let threshold = compute_mode_threshold(
                self.application,
                self.channels,
                prev_was_celt,
                has_prev_mode,
                voice_est,
            );
            if equiv >= threshold && self.sampling_rate >= 24000 {
                OpusMode::CeltOnly
            } else {
                OpusMode::SilkOnly
            }
        };

        let curr_bw = self.bandwidth;
        if mode == OpusMode::SilkOnly
            && (curr_bw == Bandwidth::Superwideband || curr_bw == Bandwidth::Fullband)
        {
            mode = OpusMode::Hybrid;
        }
        if mode == OpusMode::Hybrid
            && (curr_bw == Bandwidth::Narrowband
                || curr_bw == Bandwidth::Mediumband
                || curr_bw == Bandwidth::Wideband)
        {
            mode = OpusMode::SilkOnly;
        }

        if mode == OpusMode::CeltOnly {
            match frame_rate {
                400 | 200 | 100 | 50 => {}
                _ => return Err("Unsupported frame size for CELT-only mode"),
            }
        }

        if mode == OpusMode::Hybrid {
            match frame_rate {
                100 | 50 => {}
                _ => return Err("Unsupported frame size for Hybrid mode"),
            }
        }

        if mode == OpusMode::SilkOnly {
            match frame_rate {
                400 | 200 | 100 | 50 | 25 => {}
                _ => return Err("Unsupported frame size for SILK-only mode"),
            }
        }

        let toc = gen_toc(mode, frame_rate, self.bandwidth, self.channels);
        output[0] = toc;

        let target_bits =
            (self.bitrate_bps as i64 * frame_size as i64 / self.sampling_rate as i64) as i32;
        let cbr_bytes = ((target_bits + 4) / 8) as usize;
        let max_data_bytes = output.len();

        let n_bytes = cbr_bytes.min(max_data_bytes).max(1);

        let init_rc_size = n_bytes - 1;
        self.rc.reset_for_encode(init_rc_size as u32);

        if mode == OpusMode::SilkOnly || mode == OpusMode::Hybrid {
            let silk_fs_khz = if mode == OpusMode::Hybrid {
                16
            } else {
                self.sampling_rate.min(16000) / 1000
            };

            let frame_ms = (frame_size as i32 * 1000) / self.sampling_rate;
            if !self.silk_initialized || self.silk_enc.s_cmn.fs_khz != silk_fs_khz {
                let silk_init_bitrate = (((n_bytes - 1) * 8) as i64 * self.sampling_rate as i64
                    / frame_size as i64) as i32;
                silk_control_encoder(
                    &mut self.silk_enc,
                    silk_fs_khz,
                    frame_ms,
                    silk_init_bitrate,
                    self.complexity,
                );
                self.silk_enc.s_cmn.use_cbr = if self.use_cbr { 1 } else { 0 };

                self.silk_enc.s_cmn.n_channels = self.channels as i32;
                self.silk_initialized = true;
                self.down2_state_first = [0; 2];
                self.down2_state_second = [0; 2];
                self.down2_3_state = [0; 6];
                self.down_1_3_state = silk::resampler::SilkResamplerDown1_3::default();
            }

            self.silk_enc.s_cmn.use_in_band_fec = if self.use_inband_fec { 1 } else { 0 };
            self.silk_enc.s_cmn.packet_loss_perc = self.packet_loss_perc.clamp(0, 100);

            self.silk_enc.s_cmn.lbrr_enabled = if self.use_inband_fec { 1 } else { 0 };

            if self.silk_enc.s_cmn.lbrr_gain_increases == 0 {
                self.silk_enc.s_cmn.lbrr_gain_increases = 2;
            }

            let hp_freq_smth1 = if mode == OpusMode::CeltOnly {
                silk_lin2log(60) << 8
            } else {
                self.silk_enc.s_cmn.variable_hp_smth1_q15
            };

            const VARIABLE_HP_SMTH_COEF2_Q16: i32 = 984;
            self.variable_hp_smth2_q15 = silk_smlawb(
                self.variable_hp_smth2_q15,
                hp_freq_smth1 - self.variable_hp_smth2_q15,
                VARIABLE_HP_SMTH_COEF2_Q16,
            );

            let cutoff_hz = silk_log2lin(silk_rshift(self.variable_hp_smth2_q15, 8));

            let required_size = frame_size * self.channels;
            self.buf_filtered.resize(required_size, 0);
            if self.application == Application::Voip {
                hp_cutoff(
                    input,
                    cutoff_hz,
                    &mut self.buf_filtered,
                    &mut self.hp_mem,
                    frame_size,
                    self.channels,
                    self.sampling_rate,
                );
            } else {
                for (i, &x) in input.iter().enumerate() {
                    self.buf_filtered[i] = (x * 32768.0).clamp(-32768.0, 32767.0) as i16;
                }
            }

            let input_i16 = &self.buf_filtered;

            let silk_input: &[i16] = if mode == OpusMode::SilkOnly && self.sampling_rate > 16000 {
                if self.sampling_rate == 48000 {
                    let stage1_size = frame_size / 2;
                    let mut stage1_buf = [0i16; 480];
                    silk_resampler_down2(
                        &mut self.down2_state_first,
                        &mut stage1_buf[..stage1_size],
                        input_i16,
                        frame_size as i32,
                    );
                    let silk_frame_size = stage1_size * 2 / 3;
                    self.buf_silk_input.resize(silk_frame_size, 0);
                    silk_resampler_down2_3(
                        &mut self.down2_3_state,
                        &mut self.buf_silk_input,
                        &stage1_buf[..stage1_size],
                        stage1_size as i32,
                    );
                    &self.buf_silk_input
                } else if self.sampling_rate == 24000 {
                    let silk_frame_size = frame_size * 2 / 3;
                    self.buf_silk_input.resize(silk_frame_size, 0);
                    silk_resampler_down2_3(
                        &mut self.down2_3_state,
                        &mut self.buf_silk_input,
                        input_i16,
                        frame_size as i32,
                    );
                    &self.buf_silk_input
                } else {
                    input_i16
                }
            } else if mode == OpusMode::SilkOnly && self.channels == 2 {
                let frame_length = input_i16.len() / 2;
                self.buf_stereo_mid.resize(frame_length, 0);
                self.buf_stereo_side.resize(frame_length, 0);
                for i in 0..frame_length {
                    let l = input_i16[2 * i] as i32;
                    let r = input_i16[2 * i + 1] as i32;
                    self.buf_stereo_mid[i] = ((l + r) / 2) as i16;
                    self.buf_stereo_side[i] = (l - r) as i16;
                }

                self.silk_enc.stereo.side.resize(frame_length, 0);
                self.silk_enc
                    .stereo
                    .side
                    .copy_from_slice(&self.buf_stereo_side[..frame_length]);
                &self.buf_stereo_mid
            } else if mode == OpusMode::Hybrid && self.sampling_rate > 16000 {
                if self.sampling_rate == 48000 {
                    let silk_frame_size = frame_size / 3;
                    self.buf_silk_input.resize(silk_frame_size, 0);
                    silk::resampler::silk_resampler_down_1_3(
                        &mut self.down_1_3_state,
                        &mut self.buf_silk_input,
                        input_i16,
                    );
                } else {
                    let silk_frame_size = frame_size * 2 / 3;
                    self.buf_silk_input.resize(silk_frame_size, 0);
                    silk_resampler_down2_3(
                        &mut self.down2_3_state,
                        &mut self.buf_silk_input,
                        input_i16,
                        frame_size as i32,
                    );
                }
                &self.buf_silk_input
            } else {
                input_i16
            };

            let mut pn_bytes = 0;

            let silk_rate_for_calc = if mode == OpusMode::Hybrid {
                16000
            } else {
                self.sampling_rate
            };
            let silk_frame_len = silk_input.len();

            let silk_bitrate = if mode == OpusMode::Hybrid {
                let frame_duration_ms = frame_size as i32 * 1000 / self.sampling_rate;
                let frame20ms = frame_duration_ms >= 20;
                compute_silk_rate_for_hybrid(self.bitrate_bps, frame20ms)
            } else {
                (8i64 * (n_bytes - 1) as i64 * silk_rate_for_calc as i64 / silk_frame_len as i64)
                    as i32
            };
            let silk_max_bits = if mode == OpusMode::Hybrid {
                let total_max_bits = ((n_bytes - 1) * 8) as i32;
                if self.use_cbr {
                    let silk_bits = (silk_bitrate as i64 * silk_frame_len as i64
                        / silk_rate_for_calc as i64) as i32;
                    let other_bits = 0i32.max(total_max_bits - silk_bits);
                    0i32.max(total_max_bits - other_bits * 3 / 4)
                } else {
                    let frame_duration_ms = frame_size as i32 * 1000 / self.sampling_rate;
                    let frame20ms = frame_duration_ms >= 20;
                    let max_bit_rate = compute_silk_rate_for_hybrid(
                        total_max_bits * self.sampling_rate / frame_size as i32,
                        frame20ms,
                    );
                    max_bit_rate * frame_size as i32 / self.sampling_rate
                }
            } else {
                ((n_bytes - 1) * 8) as i32
            };
            let silk_use_cbr = if mode == OpusMode::Hybrid && self.use_cbr {
                0
            } else if self.use_cbr {
                1
            } else {
                0
            };
            let ret = silk_encode(
                &mut self.silk_enc,
                silk_input,
                silk_input.len(),
                &mut self.rc,
                &mut pn_bytes,
                silk_bitrate,
                silk_max_bits,
                silk_use_cbr,
                1,
            );
            if ret != 0 {
                return Err("SILK encoding failed");
            }
        }

        if mode == OpusMode::Hybrid {
            self.rc.encode_bit_logp(false, 12); // redundancy = 0
        }

        if mode == OpusMode::Hybrid {
            let nb_compr_bytes = (n_bytes - 1) as u32;
            self.rc.shrink(nb_compr_bytes);
        }

        let silk_ret_bytes = if mode == OpusMode::SilkOnly {
            ((self.rc.tell() + 7) >> 3) as usize
        } else {
            0
        };

        if mode == OpusMode::CeltOnly || mode == OpusMode::Hybrid {
            self.celt_enc.complexity = self.complexity;
            let start_band = if mode == OpusMode::Hybrid { 17 } else { 0 };
            let total_packet_bits = ((n_bytes - 1) * 8) as i32;

            let celt_input: &[f32] = if self.channels == 1 {
                input
            } else {
                let n = frame_size * self.channels;
                self.buf_celt_input.resize(n, 0.0);
                for i in 0..frame_size {
                    for ch in 0..self.channels {
                        self.buf_celt_input[ch * frame_size + i] = input[i * self.channels + ch];
                    }
                }
                &self.buf_celt_input
            };

            if self.rc.tell() <= total_packet_bits {
                self.celt_enc.encode_with_budget(
                    celt_input,
                    frame_size,
                    &mut self.rc,
                    start_band,
                    total_packet_bits,
                );
            }
        }

        self.rc.done();

        if mode == OpusMode::SilkOnly {
            let mut ret = silk_ret_bytes.min(self.rc.storage as usize);
            while ret > 2 && self.rc.buf[ret - 1] == 0 {
                ret -= 1;
            }

            let target_total = if self.use_cbr {
                n_bytes.min(output.len())
            } else {
                (ret + 1).min(output.len())
            };

            let silk_len = ret;

            if !self.use_cbr || silk_len + 1 >= target_total {
                // VBR or payload fills the target: simple code 0 packet
                output[0] = toc;
                let copy_len = silk_len.min(target_total - 1);
                output[1..1 + copy_len].copy_from_slice(&self.rc.buf[..copy_len]);
                return Ok((copy_len + 1).min(output.len()));
            }

            output[0] = toc | 0x03;

            if silk_len + 2 >= target_total {
                output[1] = 0x01;
                let copy_len = (target_total - 2).min(silk_len);
                output[2..2 + copy_len].copy_from_slice(&self.rc.buf[..copy_len]);
                self.prev_enc_mode = Some(mode);
                return Ok(target_total.min(output.len()));
            }

            let pad_amount = target_total - silk_len - 2;
            output[1] = 0x41;

            let nb_255s = (pad_amount - 1) / 255;
            let mut ptr = 2;
            for _ in 0..nb_255s {
                output[ptr] = 255;
                ptr += 1;
            }
            output[ptr] = (pad_amount - 255 * nb_255s - 1) as u8;
            ptr += 1;

            output[ptr..ptr + silk_len].copy_from_slice(&self.rc.buf[..silk_len]);
            ptr += silk_len;

            let fill_end = target_total.min(output.len());
            for byte in output[ptr..fill_end].iter_mut() {
                *byte = 0;
            }

            self.prev_enc_mode = Some(mode);
            return Ok(target_total.min(output.len()));
        }

        let payload_len = n_bytes - 1;
        output[1..1 + payload_len].copy_from_slice(&self.rc.buf[..payload_len]);
        self.prev_enc_mode = Some(mode);
        Ok(n_bytes)
    }
}

pub struct OpusDecoder {
    celt_dec: CeltDecoder,
    silk_dec: silk::dec_api::SilkDecoder,
    sampling_rate: i32,
    channels: usize,

    prev_mode: Option<OpusMode>,
    frame_size: usize,

    bandwidth: Bandwidth,

    stream_channels: usize,

    silk_resampler: silk::resampler::SilkResampler,

    prev_internal_rate: i32,

    pub hybrid_skip_celt: bool,

    w_pcm_i16: Vec<i16>,
    w_silk_out: Vec<f32>,
    w_pcm_resampled: Vec<i16>,
    w_celt_planar: Vec<f32>,
    w_celt_out: Vec<f32>,
}

impl OpusDecoder {
    pub fn new(sampling_rate: i32, channels: usize) -> Result<Self, &'static str> {
        if ![8000, 12000, 16000, 24000, 48000].contains(&sampling_rate) {
            return Err("Invalid sampling rate");
        }
        if ![1, 2].contains(&channels) {
            return Err("Invalid number of channels");
        }

        let mode = modes::default_mode();
        let celt_dec = CeltDecoder::new(mode, channels);

        let mut silk_dec = silk::dec_api::SilkDecoder::new();
        silk_dec.init(sampling_rate.min(16000), channels as i32);
        silk_dec.channel_state[0].fs_api_hz = sampling_rate;

        Ok(Self {
            celt_dec,
            silk_dec,
            sampling_rate,
            channels,
            prev_mode: None,
            frame_size: 0,
            bandwidth: Bandwidth::Auto,
            stream_channels: channels,
            silk_resampler: silk::resampler::SilkResampler::default(),
            prev_internal_rate: 0,
            hybrid_skip_celt: false,

            w_pcm_i16: vec![0i16; 640],

            w_silk_out: vec![0.0f32; 5760 * channels],
            w_pcm_resampled: vec![0i16; 5760 * channels],
            w_celt_planar: vec![0.0f32; 5760 * channels],
            w_celt_out: vec![0.0f32; 5760 * channels],
        })
    }

    pub fn decode(
        &mut self,
        input: &[u8],
        frame_size: usize,
        output: &mut [f32],
    ) -> Result<usize, &'static str> {
        if input.is_empty() {
            return Err("Input packet empty");
        }

        let toc = input[0];
        let mode = mode_from_toc(toc);
        let packet_channels = channels_from_toc(toc);
        let bandwidth = bandwidth_from_toc(toc);
        let frame_duration_ms = frame_duration_ms_from_toc(toc);

        if packet_channels != self.channels {
            return Err("Channel count mismatch between packet and decoder");
        }

        let code = toc & 0x03;
        let frame_count: usize;
        let frame_payloads: Vec<&[u8]>;

        match code {
            0 => {
                frame_count = 1;
                frame_payloads = vec![&input[1..]];
            }
            1 => {
                frame_count = 2;
                let half = (input.len() - 1) / 2;
                if half == 0 {
                    return Err("Code 1: empty frame");
                }
                frame_payloads = vec![&input[1..1 + half], &input[1 + half..]];
            }
            2 => {
                frame_count = 2;
                let data = &input[1..];
                if data.is_empty() {
                    return Err("Code 2 packet has no data");
                }
                let (first_len, header_size) = if data[0] & 0x80 != 0 {
                    if data.len() < 2 {
                        return Err("Code 2 packet too short for 2-byte length");
                    }
                    (((data[0] & 0x7F) as usize) << 8 | data[1] as usize, 2)
                } else {
                    (data[0] as usize, 1)
                };
                if header_size + first_len > data.len() {
                    return Err("Code 2: first frame size exceeds packet");
                }
                frame_payloads = vec![
                    &data[header_size..header_size + first_len],
                    &data[header_size + first_len..],
                ];
            }
            3 => {
                if input.len() < 2 {
                    return Err("Code 3 packet too short");
                }
                let count_byte = input[1];
                let n_frames = (count_byte & 0x3F) as usize;
                if n_frames < 1 || n_frames > 48 {
                    return Err("Code 3: invalid frame count");
                }
                frame_count = n_frames;
                let padding_flag = (count_byte & 0x40) != 0;

                if padding_flag {
                    let mut ptr = 2usize;
                    let mut pad_len = 0usize;
                    loop {
                        if ptr >= input.len() {
                            return Err("Padding overflow");
                        }
                        let p = input[ptr] as usize;
                        ptr += 1;
                        if p == 255 {
                            pad_len += 254;
                        } else {
                            pad_len += p;
                            break;
                        }
                    }

                    let end = input.len().saturating_sub(pad_len);
                    if ptr > end {
                        return Err("Padding exceeds packet");
                    }
                    let compressed = &input[ptr..end];
                    let frame_len = compressed.len() / frame_count;
                    if frame_len == 0 {
                        return Err("Code 3 with padding: empty frame");
                    }
                    frame_payloads = compressed.chunks(frame_len).collect();
                    if frame_payloads.len() != frame_count {
                        return Err("Code 3: frame count mismatch");
                    }
                } else {
                    // Self-delimiting format:
                    // - Single frame: no length prefix, remaining data is the frame
                    // - Multi-frame: lengths for all frames except the last
                    let mut payload_ptr = 2usize;
                    if frame_count == 1 {
                        frame_payloads = vec![&input[payload_ptr..]];
                    } else {
                        let mut payloads = Vec::with_capacity(frame_count);
                        for _f in 0..frame_count - 1 {
                            if payload_ptr >= input.len() {
                                return Err("Code 3: unexpected end in self-delimiting header");
                            }
                            let (frame_len, header_bytes) = if input[payload_ptr] & 0x80 != 0 {
                                if payload_ptr + 2 > input.len() {
                                    return Err("Code 3: short frame length");
                                }
                                (
                                    ((input[payload_ptr] & 0x7F) as usize) << 8
                                        | input[payload_ptr + 1] as usize,
                                    2,
                                )
                            } else {
                                (input[payload_ptr] as usize, 1)
                            };
                            payload_ptr += header_bytes;
                            if payload_ptr + frame_len > input.len() {
                                return Err("Code 3: frame length exceeds packet");
                            }
                            payloads.push(&input[payload_ptr..payload_ptr + frame_len]);
                            payload_ptr += frame_len;
                        }
                        // Last frame: no length prefix, remaining data
                        if payload_ptr > input.len() {
                            return Err("Code 3: no data for last frame");
                        }
                        payloads.push(&input[payload_ptr..]);
                        frame_payloads = payloads;
                    }
                }
            }
            _ => unreachable!(),
        }

        self.frame_size = frame_size;
        self.bandwidth = bandwidth;
        self.stream_channels = packet_channels;

        let sub_frame_size = frame_size / frame_count;
        let sub_output_len = sub_frame_size * self.channels;

        match mode {
            OpusMode::SilkOnly => {
                let internal_sample_rate = match bandwidth {
                    Bandwidth::Narrowband => 8000,
                    Bandwidth::Mediumband => 12000,
                    Bandwidth::Wideband => 16000,
                    _ => 16000,
                };
                let internal_frame_size =
                    (frame_duration_ms * internal_sample_rate / 1000) as usize;

                if self.sampling_rate != internal_sample_rate
                    && internal_sample_rate != self.prev_internal_rate
                {
                    self.silk_resampler
                        .init(internal_sample_rate, self.sampling_rate);
                    self.prev_internal_rate = internal_sample_rate;
                }

                for (fi, payload) in frame_payloads.iter().enumerate() {
                    let mut rc = RangeCoder::new_decoder(payload);
                    let pcm_i16_len = internal_frame_size * self.channels;
                    debug_assert!(pcm_i16_len <= self.w_pcm_i16.len());

                    let ret = {
                        let (silk_dec, pcm_i16) = (&mut self.silk_dec, &mut self.w_pcm_i16);
                        silk_dec.decode(
                            &mut rc,
                            &mut pcm_i16[..pcm_i16_len],
                            silk::decode_frame::FLAG_DECODE_NORMAL,
                            true,
                            frame_duration_ms,
                            internal_sample_rate,
                        )
                    };

                    if ret < 0 {
                        return Err("SILK decoding failed");
                    }

                    let decoded_samples = ret as usize;
                    let out_start = fi * sub_output_len;

                    // SILK only decodes channel 0 (mono). For multi-channel output,
                    // replicate the mono samples to every channel.
                    if self.sampling_rate == internal_sample_rate {
                        let frames = decoded_samples.min(sub_frame_size);
                        for i in 0..frames {
                            let v = self.w_pcm_i16[i] as f32 / 32768.0;
                            for ch in 0..self.channels {
                                let idx = out_start + i * self.channels + ch;
                                if idx < output.len() {
                                    output[idx] = v;
                                }
                            }
                        }
                    } else {
                        let ratio = self.sampling_rate as f64 / internal_sample_rate as f64;
                        let out_len =
                            ((decoded_samples as f64 * ratio) as usize).min(sub_frame_size);
                        debug_assert!(out_len <= self.w_pcm_resampled.len());
                        {
                            let (silk_res, pcm_i16, pcm_out) = (
                                &mut self.silk_resampler,
                                &self.w_pcm_i16,
                                &mut self.w_pcm_resampled,
                            );
                            silk_res.process(
                                &mut pcm_out[..out_len],
                                &pcm_i16[..decoded_samples],
                                decoded_samples as i32,
                            );
                        }
                        let frames = out_len.min(sub_frame_size);
                        for i in 0..frames {
                            let v = self.w_pcm_resampled[i] as f32 / 32768.0;
                            for ch in 0..self.channels {
                                let idx = out_start + i * self.channels + ch;
                                if idx < output.len() {
                                    output[idx] = v;
                                }
                            }
                        }
                    }
                }
                self.prev_mode = Some(OpusMode::SilkOnly);
                Ok(frame_size)
            }

            OpusMode::CeltOnly => {
                let celt_end_band = self.celt_end_band_from_toc(toc);

                for (fi, payload) in frame_payloads.iter().enumerate() {
                    let mut rc = RangeCoder::new_decoder(payload);
                    let total_bits = (payload.len() * 8) as i32;
                    let needed = sub_frame_size * self.channels;
                    let out_start = fi * needed;
                    let out_end = (out_start + needed).min(output.len());

                    if output.len() < out_end {
                        return Err("Output buffer too small");
                    }

                    if self.channels == 1 {
                        self.celt_dec.decode_from_range_coder_with_band_range(
                            &mut rc,
                            total_bits,
                            sub_frame_size,
                            &mut output[out_start..out_end],
                            0,
                            celt_end_band,
                        );
                        for sample in &mut output[out_start..out_end] {
                            *sample = sample.clamp(-1.0, 1.0);
                        }
                    } else {
                        self.celt_dec.decode_from_range_coder_with_band_range(
                            &mut rc,
                            total_bits,
                            sub_frame_size,
                            &mut self.w_celt_planar[..needed],
                            0,
                            celt_end_band,
                        );
                        for i in 0..sub_frame_size {
                            for ch in 0..self.channels {
                                let idx = out_start + i * self.channels + ch;
                                output[idx] =
                                    self.w_celt_planar[ch * sub_frame_size + i].clamp(-1.0, 1.0);
                            }
                        }
                    }
                }
                self.prev_mode = Some(OpusMode::CeltOnly);
                Ok(frame_size)
            }

            OpusMode::Hybrid => {
                let internal_sample_rate = 16000;
                let internal_frame_size =
                    (frame_duration_ms * internal_sample_rate / 1000) as usize;
                let celt_end_band = self.celt_end_band_from_toc(toc);

                if self.sampling_rate != internal_sample_rate
                    && internal_sample_rate != self.prev_internal_rate
                {
                    self.silk_resampler
                        .init(internal_sample_rate, self.sampling_rate);
                    self.prev_internal_rate = internal_sample_rate;
                }

                for (fi, payload) in frame_payloads.iter().enumerate() {
                    let mut rc = RangeCoder::new_decoder(payload);
                    let pcm_silk_i16_len = internal_frame_size * self.channels;
                    debug_assert!(pcm_silk_i16_len <= self.w_pcm_i16.len());

                    let ret = {
                        let (silk_dec, pcm_i16) = (&mut self.silk_dec, &mut self.w_pcm_i16);
                        silk_dec.decode(
                            &mut rc,
                            &mut pcm_i16[..pcm_silk_i16_len],
                            silk::decode_frame::FLAG_DECODE_NORMAL,
                            true,
                            frame_duration_ms,
                            internal_sample_rate,
                        )
                    };

                    if ret < 0 {
                        return Err("SILK decoding failed");
                    }

                    let silk_out_len = sub_frame_size * self.channels;
                    self.w_silk_out[..silk_out_len].fill(0.0);
                    if ret > 0 {
                        let decoded_samples = ret as usize;
                        // SILK only decodes channel 0 (mono). For multi-channel output,
                        // replicate the mono samples to every channel in w_silk_out.
                        if self.sampling_rate == internal_sample_rate {
                            let frames = decoded_samples.min(sub_frame_size);
                            for i in 0..frames {
                                let v = self.w_pcm_i16[i] as f32 / 32768.0;
                                for ch in 0..self.channels {
                                    let idx = i * self.channels + ch;
                                    if idx < silk_out_len {
                                        self.w_silk_out[idx] = v;
                                    }
                                }
                            }
                        } else {
                            let ratio = self.sampling_rate as f64 / internal_sample_rate as f64;
                            let out_len =
                                ((decoded_samples as f64 * ratio) as usize).min(sub_frame_size);
                            debug_assert!(out_len <= self.w_pcm_resampled.len());
                            {
                                let (silk_res, pcm_i16, pcm_resampled) = (
                                    &mut self.silk_resampler,
                                    &self.w_pcm_i16,
                                    &mut self.w_pcm_resampled,
                                );
                                silk_res.process(
                                    &mut pcm_resampled[..out_len],
                                    &pcm_i16[..decoded_samples],
                                    decoded_samples as i32,
                                );
                            }
                            let frames = out_len.min(sub_frame_size);
                            for i in 0..frames {
                                let v = self.w_pcm_resampled[i] as f32 / 32768.0;
                                for ch in 0..self.channels {
                                    let idx = i * self.channels + ch;
                                    if idx < silk_out_len {
                                        self.w_silk_out[idx] = v;
                                    }
                                }
                            }
                        }
                    }

                    let total_bits = (payload.len() * 8) as i32;
                    let redundancy = rc.decode_bit_logp(12);
                    let skip_celt = if redundancy {
                        let _ = rc.decode_bit_logp(1);
                        true
                    } else {
                        false
                    };

                    if skip_celt {
                        self.w_celt_out[..silk_out_len].fill(0.0);
                    } else {
                        let (celt_dec, celt_planar) = (&mut self.celt_dec, &mut self.w_celt_planar);
                        celt_dec.decode_from_range_coder_with_band_range(
                            &mut rc,
                            total_bits,
                            sub_frame_size,
                            &mut celt_planar[..silk_out_len],
                            17,
                            celt_end_band,
                        );

                        if self.channels == 1 {
                            self.w_celt_out[..silk_out_len]
                                .copy_from_slice(&self.w_celt_planar[..silk_out_len]);
                        } else {
                            for i in 0..sub_frame_size {
                                for ch in 0..self.channels {
                                    self.w_celt_out[i * self.channels + ch] =
                                        self.w_celt_planar[ch * sub_frame_size + i];
                                }
                            }
                        }
                    }

                    let out_start = fi * silk_out_len;
                    let total = silk_out_len.min(output.len() - out_start);
                    for j in 0..total {
                        output[out_start + j] =
                            (self.w_silk_out[j] + self.w_celt_out[j]).clamp(-1.0, 1.0);
                    }
                }
                self.prev_mode = Some(OpusMode::Hybrid);
                Ok(frame_size)
            }
        }
    }
}

impl OpusDecoder {
    #[inline(always)]
    fn celt_end_band_from_toc(&self, toc: u8) -> usize {
        let mode = modes::default_mode();
        let top = mode.eff_ebands;
        if mode_from_toc(toc) == OpusMode::CeltOnly && toc >= 0x80 {
            const FROM_OPUS_TABLE: [u8; 16] = [
                0x80, 0x88, 0x90, 0x98, 0x40, 0x48, 0x50, 0x58, 0x20, 0x28, 0x30, 0x38, 0x00, 0x08,
                0x10, 0x18,
            ];
            let idx = ((toc >> 3) - 16) as usize;
            let data0 = FROM_OPUS_TABLE[idx] | (toc & 0x7);
            let trim = (data0 >> 5) as usize;
            return top.saturating_sub(2 * trim).max(1);
        }
        top
    }
}

fn frame_rate_from_params(sampling_rate: i32, frame_size: usize) -> Option<i32> {
    let frame_size = frame_size as i32;
    if frame_size == 0 || sampling_rate % frame_size != 0 {
        return None;
    }
    Some(sampling_rate / frame_size)
}

fn gen_toc(mode: OpusMode, frame_rate: i32, bandwidth: Bandwidth, channels: usize) -> u8 {
    let mut rate = frame_rate;
    let mut period = 0;
    while rate < 400 {
        rate <<= 1;
        period += 1;
    }

    let mut toc = match mode {
        OpusMode::SilkOnly => {
            let bw = (bandwidth as i32 - Bandwidth::Narrowband as i32) << 5;
            let per = (period - 2) << 3;
            (bw | per) as u8
        }
        OpusMode::CeltOnly => {
            let mut tmp = bandwidth as i32 - Bandwidth::Mediumband as i32;
            if tmp < 0 {
                tmp = 0;
            }
            let per = period << 3;
            (0x80 | (tmp << 5) | per) as u8
        }
        OpusMode::Hybrid => {
            let base_config = if bandwidth == Bandwidth::Superwideband {
                12
            } else {
                14
            };
            let period_offset = if frame_rate >= 100 { 0 } else { 1 };
            ((base_config + period_offset) << 3) as u8
        }
    };

    if channels == 2 {
        toc |= 0x04;
    }
    toc
}

fn mode_from_toc(toc: u8) -> OpusMode {
    if toc & 0x80 != 0 {
        OpusMode::CeltOnly
    } else if toc & 0x60 == 0x60 {
        OpusMode::Hybrid
    } else {
        OpusMode::SilkOnly
    }
}

fn bandwidth_from_toc(toc: u8) -> Bandwidth {
    let mode = mode_from_toc(toc);
    match mode {
        OpusMode::SilkOnly => {
            let bw_bits = (toc >> 5) & 0x03;
            match bw_bits {
                0 => Bandwidth::Narrowband,
                1 => Bandwidth::Mediumband,
                2 => Bandwidth::Wideband,
                _ => Bandwidth::Wideband,
            }
        }
        OpusMode::Hybrid => {
            let bw_bit = (toc >> 4) & 0x01;
            if bw_bit == 0 {
                Bandwidth::Superwideband
            } else {
                Bandwidth::Fullband
            }
        }
        OpusMode::CeltOnly => {
            let bw_bits = (toc >> 5) & 0x03;
            match bw_bits {
                0 => Bandwidth::Mediumband,
                1 => Bandwidth::Wideband,
                2 => Bandwidth::Superwideband,
                3 => Bandwidth::Fullband,
                _ => Bandwidth::Fullband,
            }
        }
    }
}

fn frame_duration_ms_from_toc(toc: u8) -> i32 {
    let mode = mode_from_toc(toc);
    match mode {
        OpusMode::SilkOnly => {
            let config = (toc >> 3) & 0x03;
            match config {
                0 => 10,
                1 => 20,
                2 => 40,
                3 => 60,
                _ => 20,
            }
        }
        OpusMode::Hybrid => {
            let config = (toc >> 3) & 0x01;
            if config == 0 { 10 } else { 20 }
        }
        OpusMode::CeltOnly => {
            let config = (toc >> 3) & 0x03;
            match config {
                0 => 2,
                1 => 5,
                2 => 10,
                3 => 20,
                _ => 20,
            }
        }
    }
}

fn channels_from_toc(toc: u8) -> usize {
    if toc & 0x04 != 0 { 2 } else { 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_size_from_toc(toc: u8, sampling_rate: i32) -> Option<usize> {
        let mode = mode_from_toc(toc);
        match mode {
            OpusMode::CeltOnly => {
                let period = ((toc >> 3) & 0x03) as i32;
                let frame_rate = 400 >> period;
                if frame_rate == 0 || sampling_rate % frame_rate != 0 {
                    return None;
                }
                Some((sampling_rate / frame_rate) as usize)
            }
            OpusMode::SilkOnly => {
                let duration_ms = frame_duration_ms_from_toc(toc);
                Some((sampling_rate as i64 * duration_ms as i64 / 1000) as usize)
            }
            OpusMode::Hybrid => {
                let duration_ms = frame_duration_ms_from_toc(toc);
                Some((sampling_rate as i64 * duration_ms as i64 / 1000) as usize)
            }
        }
    }

    #[test]
    fn gen_toc_matches_celt_reference_values() {
        let sampling_rate = 48_000;
        let cases = [
            (120usize, 0xE0u8),
            (240usize, 0xE8u8),
            (480usize, 0xF0u8),
            (960usize, 0xF8u8),
        ];

        for (frame_size, expected_toc) in cases {
            let frame_rate = frame_rate_from_params(sampling_rate, frame_size).unwrap();
            let toc = gen_toc(OpusMode::CeltOnly, frame_rate, Bandwidth::Fullband, 1);
            assert_eq!(
                toc, expected_toc,
                "frame_size {} expected TOC {:02X} got {:02X}",
                frame_size, expected_toc, toc
            );
            let decoded_size = frame_size_from_toc(toc, sampling_rate).unwrap();
            assert_eq!(decoded_size, frame_size);
        }

        let stereo_toc = gen_toc(
            OpusMode::CeltOnly,
            frame_rate_from_params(sampling_rate, 960).unwrap(),
            Bandwidth::Fullband,
            2,
        );
        assert_eq!(channels_from_toc(stereo_toc), 2);
    }

    #[test]
    fn test_celt_decoder_large_frame_sizes() {
        let sampling_rate = 48000;
        let channels = 1;

        let mut decoder = OpusDecoder::new(sampling_rate, channels).unwrap();

        let frame_sizes = [120, 240, 480, 960];

        for frame_size in frame_sizes {
            let toc = gen_toc(
                OpusMode::CeltOnly,
                frame_rate_from_params(sampling_rate, frame_size).unwrap(),
                Bandwidth::Fullband,
                channels,
            );
            let packet = [toc, 0, 0, 0, 0];

            let mut output = vec![0.0f32; frame_size * channels];

            let _ = decoder.decode(&packet, frame_size, &mut output);
        }

        let channels = 2;
        let mut decoder = OpusDecoder::new(sampling_rate, channels).unwrap();

        for frame_size in frame_sizes {
            let toc = gen_toc(
                OpusMode::CeltOnly,
                frame_rate_from_params(sampling_rate, frame_size).unwrap(),
                Bandwidth::Fullband,
                channels,
            );
            let packet = [toc, 0, 0, 0, 0];

            let mut output = vec![0.0f32; frame_size * channels];
            let _ = decoder.decode(&packet, frame_size, &mut output);
        }
    }

    #[test]
    fn test_celt_decoder_edge_case_frame_sizes() {
        let sampling_rate = 48000;
        let channels = 1;
        let mut decoder = OpusDecoder::new(sampling_rate, channels).unwrap();

        let edge_sizes = [2048, 2167, 2168, 2169, 2880, 3072];

        for frame_size in edge_sizes {
            let mut output = vec![0.0f32; frame_size * channels];

            let _ = decoder.decode(&[0x80, 0, 0, 0], frame_size, &mut output);
        }
    }

    // Regression test for: "index out of bounds: the len is 48 but the index is 119"
    // Root cause: frame_size=48 at 48kHz gives frame_rate=1000, which is not a valid
    // Hybrid-mode frame rate but was not validated.  CELT's lm-search then silently
    // fell back to lm=0, computed n2=120, and wrote output[119] into a 48-element
    // slice.  Triggered via G.729-decoded PCM (8kHz) passed to a 48kHz Opus encoder
    // without proper resampling, so the encoder received 48 samples instead of 480.
    #[test]
    fn test_invalid_small_frame_size_returns_error_not_panic() {
        let mut enc = OpusEncoder::new(48000, 2, Application::Voip).unwrap();
        enc.bitrate_bps = 64000;
        enc.complexity = 5;
        enc.use_cbr = true;

        // 48 samples at 48kHz = 1ms → frame_rate=1000, invalid for Hybrid mode.
        let input = vec![0.0f32; 48 * 2]; // stereo interleaved
        let mut output = vec![0u8; 256];

        let result = enc.encode(&input, 48, &mut output);
        assert!(
            result.is_err(),
            "encode with invalid frame_size=48 should return Err, not panic"
        );
    }

    // Also verify that the Audio application path (always Hybrid at 48 kHz) rejects
    // the same bad frame size.
    #[test]
    fn test_invalid_small_frame_size_audio_application_returns_error() {
        let mut enc = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
        let input = vec![0.0f32; 48];
        let mut output = vec![0u8; 256];

        let result = enc.encode(&input, 48, &mut output);
        assert!(
            result.is_err(),
            "Audio/48kHz encoder with frame_size=48 should return Err"
        );
    }
}
