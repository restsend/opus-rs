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
    down2_state_first: [i32; 2],
    down2_state_second: [i32; 2],
    down2_3_state: [i32; 6],
    down_1_3_state: silk::resampler::SilkResamplerDown1_3,

    rc: RangeCoder,
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
        // For Hybrid mode, decide whether to stay in Hybrid or switch to CELT-only
        // based on the equivalent bitrate vs a threshold derived from voice estimation.
        // NOTE: CELT-only is currently disabled because the RS CELT encoder has
        // bitstream incompatibilities with the C reference decoder (~3dB SNR).
        // Once the CELT encoder is fixed, uncomment the mode selection below.
        let mode = self.mode;
        /* TODO: enable when CELT encoder is fixed
        let mode = if self.mode == OpusMode::Hybrid {
            let equiv = compute_equiv_rate(
                self.bitrate_bps,
                self.channels,
                frame_rate,
                self.use_cbr,
            );
            let prev_was_celt = self.prev_enc_mode == Some(OpusMode::CeltOnly);
            let threshold =
                compute_mode_threshold(self.application, self.channels, prev_was_celt);
            if equiv >= threshold {
                OpusMode::CeltOnly
            } else {
                OpusMode::Hybrid
            }
        } else {
            self.mode
        };
        */

        if mode == OpusMode::CeltOnly {
            match frame_rate {
                400 | 200 | 100 | 50 => {}
                _ => return Err("Unsupported frame size for CELT-only mode"),
            }
        }

        let toc = gen_toc(mode, frame_rate, self.bandwidth, self.channels);
        output[0] = toc;

        let target_bits =
            (self.bitrate_bps as i64 * frame_size as i64 / self.sampling_rate as i64) as i32;
        let cbr_bytes = ((target_bits + 4) / 8) as usize;
        let max_data_bytes = output.len();

        let n_bytes = cbr_bytes.min(max_data_bytes).max(1);

        // Initialize range coder with CBR-capped buffer size, matching C's
        // ec_enc_init(&enc, data, max_data_bytes-1) where max_data_bytes is CBR-capped.
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
                // Match C: for VBR Hybrid, maxBits starts at (max_data_bytes-1)*8,
                // then constrained via compute_silk_rate_for_hybrid.
                // For CBR Hybrid, allow SILK to steal up to 25% of remaining bits.
                let total_max_bits = ((n_bytes - 1) * 8) as i32;
                if self.use_cbr {
                    let silk_bits = (silk_bitrate as i64 * silk_frame_len as i64
                        / silk_rate_for_calc as i64) as i32;
                    let other_bits = 0i32.max(total_max_bits - silk_bits);
                    0i32.max(total_max_bits - other_bits * 3 / 4)
                } else {
                    // Constrained VBR: compute max SILK bits from total budget
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
            // For Hybrid CBR, force SILK to use VBR (matching C behavior)
            let silk_use_cbr = if mode == OpusMode::Hybrid && self.use_cbr {
                0
            } else {
                if self.use_cbr { 1 } else { 0 }
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

        // In Hybrid mode, the C encoder always writes a redundancy flag after SILK encoding.
        // The C decoder reads this flag. Without it, the decoder's range coder state diverges,
        // corrupting CELT decoding. We always write redundancy=0 (no redundancy frame).
        // C: if (st->mode == MODE_HYBRID) ec_enc_bit_logp(&enc, redundancy, 12);
        if mode == OpusMode::Hybrid {
            if self.rc.tell() + 37 <= ((n_bytes - 1) * 8) as i32 {
                self.rc.encode_bit_logp(false, 12); // redundancy = 0
            }
        }

        // For hybrid mode, shrink the range coder buffer to the CELT target size
        // after SILK encoding, like C's ec_enc_shrink(&enc, nb_compr_bytes).
        // With init_rc_size == n_bytes-1, shrink is a no-op but keeps the logic consistent.
        if mode == OpusMode::Hybrid {
            let nb_compr_bytes = (n_bytes - 1) as u32;
            self.rc.shrink(nb_compr_bytes);
        }

        // For SILK-only, capture byte count BEFORE done(), matching C's:
        //   ret = (ec_tell(&enc)+7)>>3; ec_enc_done(&enc); nb_compr_bytes = ret;
        let silk_ret_bytes = if mode == OpusMode::SilkOnly {
            ((self.rc.tell() + 7) >> 3) as usize
        } else {
            0
        };

        if mode == OpusMode::CeltOnly || mode == OpusMode::Hybrid {
            self.celt_enc.complexity = self.complexity;
            let start_band = if mode == OpusMode::Hybrid { 17 } else { 0 };
            let total_packet_bits = ((n_bytes - 1) * 8) as i32;
            // Only encode CELT if we haven't already busted the budget
            if self.rc.tell() <= total_packet_bits {
                self.celt_enc.encode_with_budget(
                    input,
                    frame_size,
                    &mut self.rc,
                    start_band,
                    total_packet_bits,
                );
            }
        }

        self.rc.done();

        if mode == OpusMode::SilkOnly {
            // Use the byte count captured before done(), strip trailing zeros
            // from the contiguous buffer (like C does).
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

            // CBR: pad with code 3 framing
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
        let payload_data;

        match code {
            0 => {
                payload_data = &input[1..];
            }
            3 => {
                if input.len() < 2 {
                    return Err("Code 3 packet too short");
                }
                let count_byte = input[1];
                let _frame_count = (count_byte & 0x3F) as usize;
                let padding_flag = (count_byte & 0x40) != 0;

                let mut ptr = 2usize;
                if padding_flag {
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
                    payload_data = &input[ptr..end];
                } else {
                    payload_data = &input[ptr..];
                }
            }
            _ => {
                payload_data = &input[1..];
            }
        }

        self.frame_size = frame_size;
        self.bandwidth = bandwidth;
        self.stream_channels = packet_channels;

        match mode {
            OpusMode::SilkOnly => {
                let internal_sample_rate = match bandwidth {
                    Bandwidth::Narrowband => 8000,
                    Bandwidth::Mediumband => 12000,
                    Bandwidth::Wideband => 16000,
                    _ => 16000,
                };

                let mut rc = RangeCoder::new_decoder(payload_data);
                let internal_frame_size =
                    (frame_duration_ms * internal_sample_rate / 1000) as usize;
                let pcm_i16_len = internal_frame_size * self.channels;
                debug_assert!(pcm_i16_len <= self.w_pcm_i16.len());

                let payload_size_ms = frame_duration_ms;

                let ret = {
                    let (silk_dec, pcm_i16) = (&mut self.silk_dec, &mut self.w_pcm_i16);
                    silk_dec.decode(
                        &mut rc,
                        &mut pcm_i16[..pcm_i16_len],
                        silk::decode_frame::FLAG_DECODE_NORMAL,
                        true,
                        payload_size_ms,
                        internal_sample_rate,
                    )
                };

                if ret < 0 {
                    return Err("SILK decoding failed");
                }

                let decoded_samples = ret as usize;

                if self.sampling_rate == internal_sample_rate {
                    let n = decoded_samples.min(frame_size).min(output.len());
                    for i in 0..n {
                        output[i] = self.w_pcm_i16[i] as f32 / 32768.0;
                    }
                    self.prev_mode = Some(OpusMode::SilkOnly);
                    Ok(n)
                } else {
                    if internal_sample_rate != self.prev_internal_rate {
                        self.silk_resampler
                            .init(internal_sample_rate, self.sampling_rate);
                        self.prev_internal_rate = internal_sample_rate;
                    }

                    let ratio = self.sampling_rate as f64 / internal_sample_rate as f64;
                    let out_len = ((decoded_samples as f64 * ratio) as usize).min(frame_size);
                    debug_assert!(out_len <= self.w_pcm_resampled.len());
                    let ret2 = {
                        let (silk_res, pcm_i16, pcm_out) = (
                            &mut self.silk_resampler,
                            &self.w_pcm_i16,
                            &mut self.w_pcm_resampled,
                        );
                        silk_res.process(
                            &mut pcm_out[..out_len],
                            &pcm_i16[..decoded_samples],
                            decoded_samples as i32,
                        )
                    };
                    let _ = ret2;

                    let n = out_len.min(output.len());
                    for i in 0..n {
                        output[i] = self.w_pcm_resampled[i] as f32 / 32768.0;
                    }
                    self.prev_mode = Some(OpusMode::SilkOnly);
                    Ok(n)
                }
            }

            OpusMode::CeltOnly => {
                self.celt_dec.decode(payload_data, frame_size, output);
                self.prev_mode = Some(OpusMode::CeltOnly);
                Ok(frame_size)
            }

            OpusMode::Hybrid => {
                let internal_sample_rate = match bandwidth {
                    Bandwidth::Superwideband => 16000,
                    Bandwidth::Fullband => 16000,
                    _ => 16000,
                };

                let mut rc = RangeCoder::new_decoder(payload_data);
                let internal_frame_size =
                    (frame_duration_ms * internal_sample_rate / 1000) as usize;
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

                let silk_out_len = frame_size * self.channels;
                debug_assert!(silk_out_len <= self.w_silk_out.len());
                self.w_silk_out[..silk_out_len].fill(0.0);
                if ret > 0 {
                    let decoded_samples = ret as usize;
                    if self.sampling_rate == internal_sample_rate {
                        let n = decoded_samples.min(frame_size);
                        for i in 0..n {
                            self.w_silk_out[i] = self.w_pcm_i16[i] as f32 / 32768.0;
                        }
                    } else {
                        if internal_sample_rate != self.prev_internal_rate {
                            self.silk_resampler
                                .init(internal_sample_rate, self.sampling_rate);
                            self.prev_internal_rate = internal_sample_rate;
                        }
                        let ratio = self.sampling_rate as f64 / internal_sample_rate as f64;
                        let out_len = ((decoded_samples as f64 * ratio) as usize).min(frame_size);
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
                        for i in 0..out_len.min(frame_size) {
                            self.w_silk_out[i] = self.w_pcm_resampled[i] as f32 / 32768.0;
                        }
                    }
                }

                let celt_out_len = frame_size * self.channels;
                debug_assert!(celt_out_len <= self.w_celt_out.len());
                if self.hybrid_skip_celt {
                    self.w_celt_out[..celt_out_len].fill(0.0);
                } else {
                    let total_bits = (payload_data.len() * 8) as i32;
                    let (celt_dec, celt_out) = (&mut self.celt_dec, &mut self.w_celt_out);
                    celt_dec.decode_from_range_coder(
                        &mut rc,
                        total_bits,
                        frame_size,
                        &mut celt_out[..celt_out_len],
                        17,
                    );
                }

                let n = frame_size.min(output.len());
                for i in 0..n {
                    output[i] = (self.w_silk_out[i] + self.w_celt_out[i]).clamp(-1.0, 1.0);
                }

                self.prev_mode = Some(OpusMode::Hybrid);
                Ok(n)
            }
        }
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
}
