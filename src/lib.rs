#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

mod compat;
mod fixedvec;

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
use hp_cutoff::{dc_reject_float, hp_cutoff, hp_cutoff_float};
use range_coder::RangeCoder;
use silk::control_codec::silk_control_encoder;
use silk::enc_api::silk_encode;
use silk::init_encoder::silk_init_encoder;
use silk::lin2log::silk_lin2log;
use silk::log2lin::silk_log2lin;
use silk::macros::*;
use silk::resampler::{silk_resampler_down2, silk_resampler_down2_3};
use silk::structs::SilkEncoderState;
use crate::fixedvec::FixedVec;

// --- Heap-free buffer capacity constants (worst case: 2 channels). ---
const OPUS_MAX_CHANNELS: usize = 2;
/// Largest API frame in samples/channel (120 ms @ 48 kHz = 5760). Used by the
/// encoder's per-frame input buffers (sized `frame_size * channels`).
const OPUS_MAX_FRAME: usize = 5760;
/// Largest *single-frame* samples/channel (60 ms @ 48 kHz = 2880). The decoder's
/// staging buffers hold one sub-frame at a time, so they're sized to this — not
/// the full packet. (Halves the decoder footprint vs. a naive 5760/channel.)
const OPUS_MAX_SUBFRAME: usize = 2880;
/// Decoder per-sub-frame staging cap: `OPUS_MAX_SUBFRAME * max_channels`.
const OPUS_SUBFRAME_SCRATCH: usize = OPUS_MAX_SUBFRAME * OPUS_MAX_CHANNELS;
/// High-pass filter state memory (`channels * 2`).
const OPUS_HP_MEM: usize = OPUS_MAX_CHANNELS * 2;
/// Decoder `w_pcm_i16` cap (`960 * max_channels`).
const OPUS_PCM_I16: usize = 960 * OPUS_MAX_CHANNELS;
/// Decoder `prev_pcm_tail` cap (`240 * max_channels`).
const OPUS_PCM_TAIL: usize = 240 * OPUS_MAX_CHANNELS;
/// Max number of frames encoded in one Opus packet (RFC 6716 caps at 48 for
/// 2.5 ms codes in a 120 ms packet).
const OPUS_MAX_PACKET_FRAMES: usize = 48;
/// RFC 6716 §3.1: a single Opus packet carries at most 1276 bytes of data.
const OPUS_MAX_PACKET_BYTES: usize = 1276;
/// C `OpusEncoder.delay_buffer[MAX_ENCODER_BUFFER*2]` (480 samples * 2 ch).
/// Holds the delay-compensation ring feeding the CELT encoder.
const OPUS_DELAY_BUF: usize = 480 * OPUS_MAX_CHANNELS;
/// CELT/hybrid input staging cap: `(delay_compensation + frame)*channels`.
/// Worst case 60 ms stereo @ 48 kHz: (192 + 2880) * 2 = 6144.
const OPUS_PCM_BUF: usize = (192 + OPUS_MAX_SUBFRAME) * OPUS_MAX_CHANNELS;

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
    #[cfg(not(feature = "heap"))]
    celt_enc: CeltEncoder,
    #[cfg(feature = "heap")]
    celt_enc: Box<CeltEncoder>,
    #[cfg(not(feature = "heap"))]
    silk_enc: SilkEncoderState,
    #[cfg(feature = "heap")]
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
    hp_mem: FixedVec<i32, OPUS_HP_MEM>,

    #[cfg(not(feature = "heap"))]
    buf_filtered: FixedVec<i16, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_filtered: Box<FixedVec<i16, OPUS_MAX_FRAME>>,
    #[cfg(not(feature = "heap"))]
    buf_silk_input: FixedVec<i16, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_silk_input: Box<FixedVec<i16, OPUS_MAX_FRAME>>,
    #[cfg(not(feature = "heap"))]
    buf_stereo_mid: FixedVec<i16, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_stereo_mid: Box<FixedVec<i16, OPUS_MAX_FRAME>>,
    #[cfg(not(feature = "heap"))]
    buf_stereo_side: FixedVec<i16, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_stereo_side: Box<FixedVec<i16, OPUS_MAX_FRAME>>,
    #[cfg(not(feature = "heap"))]
    buf_celt_input: FixedVec<f32, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_celt_input: Box<FixedVec<f32, OPUS_MAX_FRAME>>,
    /// C `delay_buffer[MAX_ENCODER_BUFFER*2]`: delay-compensation ring feeding
    /// the CELT encoder (see `encoder_buffer` / `delay_compensation`).
    #[cfg(not(feature = "heap"))]
    delay_buffer: FixedVec<f32, OPUS_DELAY_BUF>,
    #[cfg(feature = "heap")]
    delay_buffer: Box<FixedVec<f32, OPUS_DELAY_BUF>>,
    /// Interleaved `pcm_buf` staging: `delay prefix + filtered frame` exactly
    /// like C `opus_encode_frame_native()` builds it before CELT encoding.
    #[cfg(not(feature = "heap"))]
    buf_celt_pcm: FixedVec<f32, OPUS_PCM_BUF>,
    #[cfg(feature = "heap")]
    buf_celt_pcm: Box<FixedVec<f32, OPUS_PCM_BUF>>,
    /// C `encoder_buffer` (= Fs/100): samples of ring history kept between
    /// frames (0 for restricted-latency applications).
    encoder_buffer: usize,
    /// C `delay_compensation` (= Fs/250): 4 ms lookahead prefix prepended to
    /// each CELT frame (0 for restricted-latency applications).
    delay_compensation: usize,
    /// Float filter state for `dc_reject_float` / `hp_cutoff_float`
    /// (C `st->hp_mem[4]`, opus_val32 in the float build).
    hp_mem_float: [f32; 4],
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

#[cfg(all(test, feature = "std"))]
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

/// Uniformly borrow a codec-state field that is a `Box<T>` under the `heap`
/// feature and a plain `T` otherwise, so call sites don't need `#[cfg]` on
/// every `&self.field` / `&mut self.field`.
///
/// Under `heap` the field is a `Box<...>`, so `T: Deref(DerefMut)` resolves to
/// the box itself and the `Target` is the inner type; without `heap` the field
/// is the inner type directly (`?Sized` lets `&mut [i16]`-style coercion choose
/// the slice target).
#[cfg(feature = "heap")]
#[inline(always)]
fn state_ref<T: core::ops::Deref>(b: &T) -> &T::Target {
    b
}
#[cfg(not(feature = "heap"))]
#[inline(always)]
fn state_ref<T: ?Sized>(b: &T) -> &T {
    b
}

#[cfg(feature = "heap")]
#[inline(always)]
fn state_mut<T: core::ops::DerefMut>(b: &mut T) -> &mut T::Target {
    b
}
#[cfg(not(feature = "heap"))]
#[inline(always)]
fn state_mut<T: ?Sized>(b: &mut T) -> &mut T {
    b
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
        #[cfg(feature = "heap")]
        let celt_enc = Box::new(CeltEncoder::new(mode, channels));
        #[cfg(not(feature = "heap"))]
        let celt_enc = CeltEncoder::new(mode, channels);

        #[cfg(feature = "heap")]
        let mut silk_enc = Box::new(SilkEncoderState::default());
        #[cfg(not(feature = "heap"))]
        let mut silk_enc = SilkEncoderState::default();
        if silk_init_encoder(state_mut(&mut silk_enc), 0) != 0 {
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
            hp_mem: FixedVec::from_value(0, channels * 2),

            #[cfg(not(feature = "heap"))]
            buf_filtered: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_filtered: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_silk_input: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_silk_input: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_stereo_mid: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_stereo_mid: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_stereo_side: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_stereo_side: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_celt_input: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_celt_input: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            delay_buffer: FixedVec::from_value(0.0, OPUS_DELAY_BUF),
            #[cfg(feature = "heap")]
            delay_buffer: Box::new(FixedVec::from_value(0.0, OPUS_DELAY_BUF)),
            #[cfg(not(feature = "heap"))]
            buf_celt_pcm: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_celt_pcm: Box::new(FixedVec::new()),
            encoder_buffer: if matches!(application, Application::RestrictedLowDelay) {
                0
            } else {
                (sampling_rate / 100) as usize
            },
            delay_compensation: if matches!(application, Application::RestrictedLowDelay) {
                0
            } else {
                (sampling_rate / 250) as usize
            },
            hp_mem_float: [0.0; 4],
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

        // Cap at the Opus per-packet maximum (RFC 6716); the range coder's buffer
        // is heap-free and sized to this constant.
        let mut n_bytes = cbr_bytes
            .min(max_data_bytes)
            .max(1)
            .min(OPUS_MAX_PACKET_BYTES);
        let init_rc_size = n_bytes - 1;
        self.rc.reset_for_encode(init_rc_size as u32);

        // C opus_encode_frame_native: high-pass cutoff state is updated for ALL
        // modes (celt_encoder.c:1969-1977); the filtered signal is what feeds
        // both SILK (hybrid) and CELT. Compute it here so CELT-only frames also
        // keep `variable_hp_smth2_q15` in sync with libopus.
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
                    state_mut(&mut self.silk_enc),
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

            let required_size = frame_size * self.channels;
            self.buf_filtered.resize(required_size, 0);
            if self.application == Application::Voip {
                hp_cutoff(
                    input,
                    cutoff_hz,
                    state_mut(&mut self.buf_filtered),
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

            let input_i16 = state_ref(&self.buf_filtered);

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
                        state_mut(&mut self.buf_silk_input),
                        &stage1_buf[..stage1_size],
                        stage1_size as i32,
                    );
                    state_ref(&self.buf_silk_input)
                } else if self.sampling_rate == 24000 {
                    let silk_frame_size = frame_size * 2 / 3;
                    self.buf_silk_input.resize(silk_frame_size, 0);
                    silk_resampler_down2_3(
                        &mut self.down2_3_state,
                        state_mut(&mut self.buf_silk_input),
                        input_i16,
                        frame_size as i32,
                    );
                    state_ref(&self.buf_silk_input)
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
                state_ref(&self.buf_stereo_mid)
            } else if mode == OpusMode::Hybrid && self.sampling_rate > 16000 {
                if self.sampling_rate == 48000 {
                    let silk_frame_size = frame_size / 3;
                    self.buf_silk_input.resize(silk_frame_size, 0);
                    silk::resampler::silk_resampler_down_1_3(
                        &mut self.down_1_3_state,
                        state_mut(&mut self.buf_silk_input),
                        input_i16,
                    );
                } else {
                    let silk_frame_size = frame_size * 2 / 3;
                    self.buf_silk_input.resize(silk_frame_size, 0);
                    silk_resampler_down2_3(
                        &mut self.down2_3_state,
                        state_mut(&mut self.buf_silk_input),
                        input_i16,
                        frame_size as i32,
                    );
                }
                state_ref(&self.buf_silk_input)
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
                state_mut(&mut self.silk_enc),
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

        // libopus parity: adjust nbCompressedBytes for CELT/Hybrid based on tell (celt_encoder.c:1913-1921)
        // tmp = bitrate*frame_size + tell*Fs; nbCompressed = (tmp+4*Fs)/(8*Fs)
        // This is the CBR branch (vbr==0) in celt_encoder.c; for VBR the encoder
        // does *not* do this adjustment - it uses the vbr_bound logic instead.
        // Doing it for VBR would double-shrink and corrupt the budget.
        if mode != OpusMode::SilkOnly && self.use_cbr {
            let tell = self.rc.tell();
            if tell > 1 {
                let tmp = self.bitrate_bps as i64 * frame_size as i64 + tell as i64 * self.sampling_rate as i64;
                let adjusted = ((tmp + 4 * self.sampling_rate as i64) / (8 * self.sampling_rate as i64)) as usize;
                let new_n = adjusted.min(max_data_bytes).max(1).min(OPUS_MAX_PACKET_BYTES);
                if new_n < n_bytes {
                    n_bytes = new_n;
                    // shrink range coder to new size (keep SILK bytes, trim tail)
                    let new_payload = n_bytes - 1;
                    self.rc.shrink(new_payload as u32);
                }
            }
        }
        if mode == OpusMode::CeltOnly || mode == OpusMode::Hybrid {
            self.celt_enc.complexity = self.complexity;
            let start_band = if mode == OpusMode::Hybrid { 17 } else { 0 };
            let total_packet_bits = ((n_bytes - 1) * 8) as i32;
            // Propagate bitrate/VBR to CeltEncoder for accurate VBR handling (libopus parity)
            let celt_bitrate = if mode == OpusMode::Hybrid {
                let frame_ms = frame_size as i32 * 1000 / self.sampling_rate;
                let frame20ms = frame_ms >= 20;
                let silk_rate = compute_silk_rate_for_hybrid(self.bitrate_bps, frame20ms);
                (self.bitrate_bps - silk_rate).max(8000)
            } else {
                self.bitrate_bps
            };
            self.celt_enc.set_bitrate(celt_bitrate);
            self.celt_enc.set_vbr(!self.use_cbr);
            // libopus: Hybrid VBR is unconstrained (can steal from SILK), CELT-only constrained
            self.celt_enc.set_constrained_vbr(mode == OpusMode::CeltOnly);

            // Build the CELT input exactly like C `opus_encode_frame_native`:
            //   pcm_buf = [delay-compensation prefix from ring]
            //             [dc_reject / hp_cutoff'd current frame]
            // CELT then reads pcm_buf[0..frame_size*channels] (opus_encoder.c:
            // 1966-2010, 2493). This delay + filter step is what libopus feeds
            // its CELT encoder; passing the raw input diverges from 1.6.
            let celt_input: &[f32] = if self.delay_compensation > 0 {
                let delay = self.delay_compensation;
                let ebuf = self.encoder_buffer;
                let ch = self.channels;
                let total = (delay + frame_size) * ch;
                self.buf_celt_pcm.resize(total, 0.0);
                // 1. delay prefix from the ring (C: OPUS_COPY at 1967)
                let prefix = delay * ch;
                let src_start = (ebuf - delay) * ch;
                self.buf_celt_pcm[..prefix]
                    .copy_from_slice(&self.delay_buffer[src_start..src_start + prefix]);
                // 2. filter current frame into pcm_buf[delay..] (C: 2002-2010)
                let out = &mut self.buf_celt_pcm[prefix..];
                if self.application == Application::Voip {
                    hp_cutoff_float(
                        input,
                        cutoff_hz,
                        out,
                        &mut self.hp_mem_float,
                        frame_size,
                        ch,
                        self.sampling_rate,
                    );
                } else {
                    dc_reject_float(
                        input,
                        3,
                        out,
                        &mut self.hp_mem_float,
                        frame_size,
                        ch,
                        self.sampling_rate,
                    );
                }
                // 3. float NaN guard (C: 2016-2028)
                let mut sum = 0.0f32;
                for &v in &self.buf_celt_pcm[prefix..] {
                    sum += v * v;
                }
                if !(sum < 1e9) || sum.is_nan() {
                    self.buf_celt_pcm[prefix..].fill(0.0);
                    self.hp_mem_float = [0.0; 4];
                }
                // 4. delay ring update (C: 2300-2312)
                let keep = ebuf as i64 - (frame_size as i64 + delay as i64);
                if keep > 0 {
                    let keep = keep as usize;
                    self.delay_buffer
                        .copy_within(ch * frame_size..ch * (frame_size + keep), 0);
                    let dst = ch * keep;
                    let n = (frame_size + delay) * ch;
                    self.delay_buffer[dst..dst + n].copy_from_slice(&self.buf_celt_pcm[..n]);
                } else {
                    let n = ebuf * ch;
                    let src = (frame_size + delay - ebuf) * ch;
                    self.delay_buffer[..n]
                        .copy_from_slice(&self.buf_celt_pcm[src..src + n]);
                }
                // 5. deinterleave pcm_buf[0..frame_size*ch] (interleaved) to
                //    channel-major for the Rust CELT encoder.
                let n = frame_size * ch;
                self.buf_celt_input.resize(n, 0.0);
                for i in 0..frame_size {
                    for c in 0..ch {
                        self.buf_celt_input[c * frame_size + i] = self.buf_celt_pcm[i * ch + c];
                    }
                }
                state_ref(&self.buf_celt_input)
            } else if self.channels == 1 {
                input
            } else {
                let n = frame_size * self.channels;
                self.buf_celt_input.resize(n, 0.0);
                for i in 0..frame_size {
                    for ch in 0..self.channels {
                        self.buf_celt_input[ch * frame_size + i] = input[i * self.channels + ch];
                    }
                }
                state_ref(&self.buf_celt_input)
            };

            if self.rc.tell() <= total_packet_bits {
                let is_vbr = !self.use_cbr;
                self.celt_enc.encode_with_budget_vbr(
                    celt_input,
                    frame_size,
                    &mut self.rc,
                    start_band,
                    total_packet_bits,
                    is_vbr,
                );
            }
        }

        self.rc.done();

        if self.rc.error != 0 {
            return Err("Range coder buffer overflow: encoded data exceeds packet budget");
        }

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

        // For CELT/Hybrid, respect possible VBR shrink performed by CeltEncoder (e.g. silence)
        let payload_len = (self.rc.storage as usize).min(output.len() - 1);
        output[1..1 + payload_len].copy_from_slice(&self.rc.buf[..payload_len]);
        self.prev_enc_mode = Some(mode);
        Ok(payload_len + 1)
    }
}

pub struct OpusDecoder {
    #[cfg(not(feature = "heap"))]
    celt_dec: CeltDecoder,
    #[cfg(feature = "heap")]
    celt_dec: Box<CeltDecoder>,
    #[cfg(not(feature = "heap"))]
    silk_dec: silk::dec_api::SilkDecoder,
    #[cfg(feature = "heap")]
    silk_dec: Box<silk::dec_api::SilkDecoder>,
    sampling_rate: i32,
    channels: usize,

    prev_mode: Option<OpusMode>,

    /// Whether the previous frame had redundancy (mode transition marker).
    prev_redundancy: bool,
    frame_size: usize,

    bandwidth: Bandwidth,

    stream_channels: usize,

    silk_resampler: silk::resampler::SilkResampler,

    /// Second resampler instance for stereo channel 1.
    silk_resampler_2: silk::resampler::SilkResampler,

    prev_internal_rate: i32,

    pub hybrid_skip_celt: bool,

    #[cfg(not(feature = "heap"))]
    w_pcm_i16: FixedVec<i16, OPUS_PCM_I16>,
    #[cfg(feature = "heap")]
    w_pcm_i16: Box<FixedVec<i16, OPUS_PCM_I16>>,
    #[cfg(not(feature = "heap"))]
    w_silk_out: FixedVec<f32, OPUS_SUBFRAME_SCRATCH>,
    #[cfg(feature = "heap")]
    w_silk_out: Box<FixedVec<f32, OPUS_SUBFRAME_SCRATCH>>,
    #[cfg(not(feature = "heap"))]
    w_pcm_resampled: FixedVec<i16, OPUS_SUBFRAME_SCRATCH>,
    #[cfg(feature = "heap")]
    w_pcm_resampled: Box<FixedVec<i16, OPUS_SUBFRAME_SCRATCH>>,
    #[cfg(not(feature = "heap"))]
    w_celt_planar: FixedVec<f32, OPUS_SUBFRAME_SCRATCH>,
    #[cfg(feature = "heap")]
    w_celt_planar: Box<FixedVec<f32, OPUS_SUBFRAME_SCRATCH>>,
    #[cfg(not(feature = "heap"))]
    w_celt_out: FixedVec<f32, OPUS_SUBFRAME_SCRATCH>,
    #[cfg(feature = "heap")]
    w_celt_out: Box<FixedVec<f32, OPUS_SUBFRAME_SCRATCH>>,

    /// Tail of the previous frame's output, used for smooth_fade at mode
    /// transitions (libopus pcm_transition + smooth_fade).
    #[cfg(not(feature = "heap"))]
    prev_pcm_tail: FixedVec<f32, OPUS_PCM_TAIL>,
    #[cfg(feature = "heap")]
    prev_pcm_tail: Box<FixedVec<f32, OPUS_PCM_TAIL>>,
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
        #[cfg(feature = "heap")]
        let celt_dec = Box::new(CeltDecoder::new(mode, channels, sampling_rate));
        #[cfg(not(feature = "heap"))]
        let celt_dec = CeltDecoder::new(mode, channels, sampling_rate);

        #[cfg(feature = "heap")]
        let mut silk_dec = Box::new(silk::dec_api::SilkDecoder::new());
        #[cfg(not(feature = "heap"))]
        let mut silk_dec = silk::dec_api::SilkDecoder::new();
        silk_dec.init(sampling_rate.min(16000), channels as i32);
        silk_dec.channel_state[0].fs_api_hz = sampling_rate;

        Ok(Self {
            celt_dec,
            silk_dec,
            sampling_rate,
            channels,
            prev_mode: None,
            prev_redundancy: false,
            frame_size: 0,
            bandwidth: Bandwidth::Auto,
            stream_channels: channels,
            silk_resampler: silk::resampler::SilkResampler::default(),
            silk_resampler_2: silk::resampler::SilkResampler::default(),
            prev_internal_rate: 0,
            hybrid_skip_celt: false,

            #[cfg(not(feature = "heap"))]
            w_pcm_i16: FixedVec::from_value(0i16, 960 * channels),
            #[cfg(feature = "heap")]
            w_pcm_i16: Box::new(FixedVec::from_value(0i16, 960 * channels)),

            #[cfg(not(feature = "heap"))]
            w_silk_out: FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels),
            #[cfg(feature = "heap")]
            w_silk_out: Box::new(FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels)),
            #[cfg(not(feature = "heap"))]
            w_pcm_resampled: FixedVec::from_value(0i16, OPUS_MAX_SUBFRAME * channels),
            #[cfg(feature = "heap")]
            w_pcm_resampled: Box::new(FixedVec::from_value(0i16, OPUS_MAX_SUBFRAME * channels)),
            #[cfg(not(feature = "heap"))]
            w_celt_planar: FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels),
            #[cfg(feature = "heap")]
            w_celt_planar: Box::new(FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels)),
            #[cfg(not(feature = "heap"))]
            w_celt_out: FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels),
            #[cfg(feature = "heap")]
            w_celt_out: Box::new(FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels)),

            #[cfg(not(feature = "heap"))]
            prev_pcm_tail: FixedVec::from_value(0.0f32, 240 * channels),
            #[cfg(feature = "heap")]
            prev_pcm_tail: Box::new(FixedVec::from_value(0.0f32, 240 * channels)),
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

        // A packet of 0 or 1 bytes (ToC only) is a lost/DTX frame. libopus
        // triggers PLC in this case (opus_decoder.c:315-321). We decode the
        // frame using the previous mode's concealment.
        let lost_frame = input.len() <= 1;

        if packet_channels != self.channels {
            return Err("Channel count mismatch between packet and decoder");
        }

        let code = toc & 0x03;
        let frame_count: usize;
        let frame_payloads: FixedVec<&[u8], OPUS_MAX_PACKET_FRAMES>;

        match code {
            0 => {
                frame_count = 1;
                frame_payloads = FixedVec::from_slice(&[&input[1..]]);
            }
            1 => {
                frame_count = 2;
                let data_len = input.len() - 1;
                // RFC 6716 §3.2.1: code 1 carries two equal-size (CBR) frames,
                // so the payload length must be even. libopus rejects odd lengths.
                if data_len % 2 != 0 {
                    return Err("Code 1: payload length must be even");
                }
                let half = data_len / 2;
                if half == 0 {
                    return Err("Code 1: empty frame");
                }
                frame_payloads = FixedVec::from_slice(&[&input[1..1 + half], &input[1 + half..]]);
            }
            2 => {
                frame_count = 2;
                let data = &input[1..];
                if data.is_empty() {
                    return Err("Code 2 packet has no data");
                }
                let (first_len, header_size) = parse_frame_size(data)?;
                if header_size + first_len > data.len() {
                    return Err("Code 2: first frame size exceeds packet");
                }
                frame_payloads = FixedVec::from_slice(&[
                    &data[header_size..header_size + first_len],
                    &data[header_size + first_len..],
                ]);
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
                // Bit 6 = padding flag, bit 7 = VBR flag (RFC 6716 §3.2.1).
                let padding_flag = (count_byte & 0x40) != 0;
                let vbr = (count_byte & 0x80) != 0;

                // Parse the optional padding length bytes that follow the count
                // byte. The padding *content* (pad_len bytes) lives at the end of
                // the packet and is not part of any frame.
                let mut ptr = 2usize;
                let mut pad_len = 0usize;
                if padding_flag {
                    loop {
                        if ptr >= input.len() {
                            return Err("Code 3: padding overflow");
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
                }
                if ptr + pad_len > input.len() {
                    return Err("Code 3: padding exceeds packet");
                }
                let payload_end = input.len() - pad_len;
                let payload = &input[ptr..payload_end];

                let mut payloads: FixedVec<&[u8], OPUS_MAX_PACKET_FRAMES> = FixedVec::new();
                if frame_count == 1 {
                    // Single frame: the entire payload region is the frame, both
                    // for VBR and CBR (no length prefix is present).
                    payloads.push(payload);
                } else if vbr {
                    // VBR (V=1): per-frame lengths for all frames except the last,
                    // which takes the remaining bytes (RFC 6716 §3.2.1).
                    let mut cursor = 0usize;
                    for i in 0..frame_count {
                        if i + 1 < frame_count {
                            if cursor >= payload.len() {
                                return Err("Code 3: unexpected end in VBR header");
                            }
                            let (frame_len, header_bytes) =
                                parse_frame_size(&payload[cursor..])?;
                            cursor += header_bytes;
                            if cursor + frame_len > payload.len() {
                                return Err("Code 3: frame length exceeds packet");
                            }
                            payloads.push(&payload[cursor..cursor + frame_len]);
                            cursor += frame_len;
                        } else {
                            // Last frame: remaining bytes, no length prefix.
                            if cursor > payload.len() {
                                return Err("Code 3: no data for last frame");
                            }
                            payloads.push(&payload[cursor..]);
                        }
                    }
                } else {
                    // CBR (V=0): remaining bytes are split equally into M frames
                    // (RFC 6716 §3.2.1: "the remaining bytes are split into M
                    // equal chunks").
                    if payload.len() % frame_count != 0 {
                        return Err("Code 3 CBR: payload not divisible by frame count");
                    }
                    let frame_len = payload.len() / frame_count;
                    for i in 0..frame_count {
                        payloads.push(&payload[i * frame_len..(i + 1) * frame_len]);
                    }
                }
                frame_payloads = payloads;
            }
            _ => unreachable!(),
        }

        self.frame_size = frame_size;
        self.bandwidth = bandwidth;
        self.stream_channels = packet_channels;

        // Derive the actual per-frame sample count from the TOC, not from the
        // caller's frame_size. This prevents panics in bands.rs/celt.rs when
        // the caller passes a mismatched frame_size (issue #7 sub-item 1):
        // the internal decoders always get the correct geometry.
        let toc_frame_size = frame_samples_from_toc(toc, self.sampling_rate)
            .ok_or("Invalid TOC for sampling rate")?;
        let decoded_total = toc_frame_size * frame_count;
        if frame_size < decoded_total {
            return Err("frame_size too small for packet");
        }
        if output.len() < decoded_total * self.channels {
            return Err("Output buffer too small for packet");
        }
        // Zero-fill any extra space the caller provided beyond what the packet
        // actually produces, so stale data is never left in the buffer.
        if output.len() > decoded_total * self.channels {
            for v in &mut output[decoded_total * self.channels..] {
                *v = 0.0;
            }
        }
        let sub_frame_size = toc_frame_size;
        let sub_output_len = sub_frame_size * self.channels;

        // Detect mode transition and reset CELT decoder state to prevent
        // cross-mode artifacts (libopus opus_decoder.c:602-604).
        // This is the primary fix for issue #8/#9 alignment divergence:
        // stale CELT MDCT/prefilter state at SILK↔CELT boundaries causes
        // discontinuities that accumulate across transitions.
        let mode_transition = match self.prev_mode {
            Some(prev) if prev != mode && !self.prev_redundancy => true,
            _ => false,
        };
        if mode_transition {
            self.celt_dec.reset_state();
        }

        // Generate SILK PLC audio for the mode-transition bridge. libopus
        // synthesizes 5ms (F5) of pitch-extrapolated audio in the OLD mode
        // (opus_decoder.c:387-391) and crossfades it with the new frame. We
        // reuse the F5-sized prev_pcm_tail buffer for this bridge.
        let f5_bridge = self.sampling_rate as usize / 200; // F5 = Fs/200
        if mode_transition
            && f5_bridge > 0
            && matches!(
                self.prev_mode,
                Some(OpusMode::SilkOnly) | Some(OpusMode::Hybrid)
            )
            && self.prev_internal_rate > 0
        {
            let internal_rate = self.prev_internal_rate;
            let plc_internal_len = (10 * internal_rate / 1000) as usize;
            let mut plc_rc = RangeCoder::new_decoder(&[]);
            let mut plc_i16: FixedVec<i16, OPUS_PCM_I16> =
                FixedVec::from_value(0i16, plc_internal_len * self.channels);
            let n = self.silk_dec.decode(
                &mut plc_rc,
                &mut plc_i16,
                silk::decode_frame::FLAG_PACKET_LOST,
                true,
                10,
                internal_rate,
            );
            if n > 0 {
                let bridge_ch = f5_bridge * self.channels;
                let bridge_len = bridge_ch.min(self.prev_pcm_tail.len());
                if internal_rate == self.sampling_rate {
                    // No resampling: copy PLC samples directly (ch0 planar).
                    let n_us = n as usize;
                    for ch in 0..self.channels {
                        let src_base = ch * n_us;
                        for i in 0..(bridge_len / self.channels).min(n_us) {
                            let dst = i * self.channels + ch;
                            if dst < bridge_len {
                                self.prev_pcm_tail[dst] = plc_i16[src_base + i] as f32 / 32768.0;
                            }
                        }
                    }
                } else if self.silk_resampler.is_initialized() {
                    // Resample channel 0 to the API rate for the bridge.
                    let ratio = self.sampling_rate as f64 / internal_rate as f64;
                    let out_len = ((n as f64 * ratio) as usize).min(f5_bridge);
                    let n_us = n as usize;
                    let mut resampled: FixedVec<i16, OPUS_MAX_FRAME> = FixedVec::from_value(0i16, out_len);
                    self.silk_resampler.process(
                        &mut resampled,
                        &plc_i16[..n_us],
                        n,
                    );
                    for i in 0..out_len {
                        if i < bridge_len / self.channels {
                            for ch in 0..self.channels {
                                self.prev_pcm_tail[i * self.channels + ch] =
                                    resampled[i] as f32 / 32768.0;
                            }
                        }
                    }
                }
            }
        }

        // Track whether this packet uses Hybrid redundancy.
        let mut has_redundancy = false;

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
                    self.silk_resampler_2
                        .init(internal_sample_rate, self.sampling_rate);
                }
                // Always track the SILK internal rate so the mode-transition
                // PLC bridge can be generated (even when no resampling is
                // needed, e.g. 16kHz decoder + SILK WB).
                self.prev_internal_rate = internal_sample_rate;

                for (fi, payload) in frame_payloads.iter().enumerate() {
                    let mut rc = RangeCoder::new_decoder(payload);
                    let pcm_i16_len = internal_frame_size * self.channels;
                    debug_assert!(pcm_i16_len <= self.w_pcm_i16.len());

                    let ret = {
                        let (silk_dec, pcm_i16) =
                            (state_mut(&mut self.silk_dec), state_mut(&mut self.w_pcm_i16));
                        let lost_flag = if lost_frame {
                            silk::decode_frame::FLAG_PACKET_LOST
                        } else {
                            silk::decode_frame::FLAG_DECODE_NORMAL
                        };
                        silk_dec.decode(
                            &mut rc,
                            &mut pcm_i16[..pcm_i16_len],
                            lost_flag,
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

                    // SILK decoder outputs planar: ch0 at [0..fl], ch1 at [fl..2*fl].
                    if self.sampling_rate == internal_sample_rate {
                        let frames = decoded_samples.min(sub_frame_size);
                        for i in 0..frames {
                            for ch in 0..self.channels {
                                let src = if ch == 0 { i } else { internal_frame_size + i };
                                let v = self.w_pcm_i16[src] as f32 / 32768.0;
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
                        debug_assert!(out_len * self.channels <= self.w_pcm_resampled.len());
                        // Resample channel 0.
                        {
                            let (res, inp, out) = (
                                &mut self.silk_resampler,
                                state_ref(&self.w_pcm_i16),
                                state_mut(&mut self.w_pcm_resampled),
                            );
                            res.process(
                                &mut out[..out_len],
                                &inp[..decoded_samples],
                                decoded_samples as i32,
                            );
                        }
                        // Resample channel 1 (stereo only).
                        if self.channels == 2 {
                            let (res, inp, out) = (
                                &mut self.silk_resampler_2,
                                state_ref(&self.w_pcm_i16),
                                state_mut(&mut self.w_pcm_resampled),
                            );
                            res.process(
                                &mut out[out_len..2 * out_len],
                                &inp[internal_frame_size..internal_frame_size + decoded_samples],
                                decoded_samples as i32,
                            );
                        }
                        let frames = out_len.min(sub_frame_size);
                        for i in 0..frames {
                            for ch in 0..self.channels {
                                let v = self.w_pcm_resampled[ch * out_len + i] as f32 / 32768.0;
                                let idx = out_start + i * self.channels + ch;
                                if idx < output.len() {
                                    output[idx] = v;
                                }
                            }
                        }
                    }
                }
                decoded_total
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
                decoded_total
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
                    self.silk_resampler_2
                        .init(internal_sample_rate, self.sampling_rate);
                }
                self.prev_internal_rate = internal_sample_rate;

                for (fi, payload) in frame_payloads.iter().enumerate() {
                    let mut rc = RangeCoder::new_decoder(payload);
                    let pcm_silk_i16_len = internal_frame_size * self.channels;
                    debug_assert!(pcm_silk_i16_len <= self.w_pcm_i16.len());

                    let ret = {
                        let (silk_dec, pcm_i16) =
                            (state_mut(&mut self.silk_dec), state_mut(&mut self.w_pcm_i16));
                        let lost_flag = if lost_frame {
                            silk::decode_frame::FLAG_PACKET_LOST
                        } else {
                            silk::decode_frame::FLAG_DECODE_NORMAL
                        };
                        silk_dec.decode(
                            &mut rc,
                            &mut pcm_i16[..pcm_silk_i16_len],
                            lost_flag,
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
                        // SILK decoder outputs planar: ch0 at [0..fl], ch1 at [fl..2*fl].
                        if self.sampling_rate == internal_sample_rate {
                            let frames = decoded_samples.min(sub_frame_size);
                            for i in 0..frames {
                                for ch in 0..self.channels {
                                    let src = if ch == 0 { i } else { internal_frame_size + i };
                                    let v = self.w_pcm_i16[src] as f32 / 32768.0;
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
                            debug_assert!(out_len * self.channels <= self.w_pcm_resampled.len());
                            // Resample channel 0.
                            {
                                let (res, inp, out) = (
                                    &mut self.silk_resampler,
                                    state_ref(&self.w_pcm_i16),
                                    state_mut(&mut self.w_pcm_resampled),
                                );
                                res.process(
                                    &mut out[..out_len],
                                    &inp[..decoded_samples],
                                    decoded_samples as i32,
                                );
                            }
                            // Resample channel 1 (stereo only).
                            if self.channels == 2 {
                                let (res, inp, out) = (
                                    &mut self.silk_resampler_2,
                                    state_ref(&self.w_pcm_i16),
                                    state_mut(&mut self.w_pcm_resampled),
                                );
                                res.process(
                                    &mut out[out_len..2 * out_len],
                                    &inp[internal_frame_size..internal_frame_size + decoded_samples],
                                    decoded_samples as i32,
                                );
                            }
                            let frames = out_len.min(sub_frame_size);
                            for i in 0..frames {
                                for ch in 0..self.channels {
                                    let v = self.w_pcm_resampled[ch * out_len + i] as f32 / 32768.0;
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
                        let _celt_to_silk = rc.decode_bit_logp(1);
                        has_redundancy = true;
                        // When redundancy is present, the redundant CELT frame
                        // provides the transition audio. We skip the main CELT
                        // decode for this sub-frame (the SILK output stands alone)
                        // — a simplified version of libopus's behaviour where the
                        // redundant frame is decoded separately and crossfaded.
                        true
                    } else {
                        false
                    };

                    if skip_celt {
                        self.w_celt_out[..silk_out_len].fill(0.0);
                    } else {
                        let (celt_dec, celt_planar) =
                            (state_mut(&mut self.celt_dec), state_mut(&mut self.w_celt_planar));
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
                decoded_total
            }
        };

        // Apply PLC-style bridging at mode transitions (libopus
        // opus_decoder.c:660-679). The first F2_5 of the output is replaced
        // with the previous frame's tail (PLC bridge), and the next F2_5 is
        // crossfaded between the bridge and the new frame's CELT output.
        // F5 = Fs/200, F2_5 = Fs/400.
        let f2_5 = self.sampling_rate as usize / 400;
        let f5 = f2_5 * 2;
        if mode_transition && f5 > 0 && decoded_total >= f5 {
            let window = modes::default_mode().window;
            let inc = (48000 / self.sampling_rate) as usize;
            let f2_5_ch = f2_5 * self.channels;
            let f5_ch = f5 * self.channels;
            // First F2_5: pure bridging audio from previous frame's tail.
            output[..f2_5_ch].copy_from_slice(&self.prev_pcm_tail[..f2_5_ch]);
            // Next F2_5: crossfade bridge → new CELT output.
            let new_mid: FixedVec<f32, OPUS_PCM_TAIL> = FixedVec::from_slice(&output[f2_5_ch..f5_ch]);
            smooth_fade(
                &self.prev_pcm_tail[f2_5_ch..f5_ch],
                &new_mid,
                &mut output[f2_5_ch..f5_ch],
                f2_5,
                self.channels,
                window,
                inc,
            );
        }

        // Save the tail of this frame for the next transition (F5 samples).
        let tail_len = f5 * self.channels;
        let out_total = decoded_total * self.channels;
        if out_total >= tail_len && tail_len <= self.prev_pcm_tail.len() {
            self.prev_pcm_tail[..tail_len]
                .copy_from_slice(&output[out_total - tail_len..out_total]);
        }

        self.prev_mode = Some(mode);
        self.prev_redundancy = has_redundancy;
        Ok(decoded_total)
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

/// Compute the per-frame sample count implied by the TOC byte at a given
/// sampling rate. For CELT this uses the frame-rate derivation (which handles
/// the 2.5 ms case correctly, unlike integer millisecond arithmetic).
fn frame_samples_from_toc(toc: u8, sampling_rate: i32) -> Option<usize> {
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
        OpusMode::SilkOnly | OpusMode::Hybrid => {
            let duration_ms = frame_duration_ms_from_toc(toc);
            Some((sampling_rate as i64 * duration_ms as i64 / 1000) as usize)
        }
    }
}

fn channels_from_toc(toc: u8) -> usize {
    if toc & 0x04 != 0 { 2 } else { 1 }
}

/// Crossfade two signals using a squared-sine window (libopus smooth_fade).
/// `window` is the 120-sample CELT window at 48 kHz; `inc` = 48000/Fs strides it.
fn smooth_fade(
    in1: &[f32],
    in2: &[f32],
    out: &mut [f32],
    overlap: usize,
    channels: usize,
    window: &[f32],
    inc: usize,
) {
    for c in 0..channels {
        for i in 0..overlap {
            let wi = i * inc;
            if wi >= window.len() {
                break;
            }
            let w = window[wi] * window[wi];
            out[i * channels + c] = w * in2[i * channels + c] + (1.0 - w) * in1[i * channels + c];
        }
    }
}

/// Parse an Opus frame length per RFC 6716 §3.2.1, identical to libopus
/// `parse_size()`:
///   - `0`: no frame (DTX / lost packet)
///   - `1..=251`: length of the frame in bytes (one byte consumed)
///   - `252..=255`: a second byte is read; length = `second*4 + first`
///
/// Returns `(length, bytes_consumed)`.
fn parse_frame_size(data: &[u8]) -> Result<(usize, usize), &'static str> {
    let first = *data.first().ok_or("truncated frame length")? as usize;
    if first < 252 {
        Ok((first, 1))
    } else {
        let second = *data.get(1).ok_or("truncated frame length")? as usize;
        Ok((second * 4 + first, 2))
    }
}

#[cfg(all(test, feature = "std"))]
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
