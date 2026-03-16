pub mod bands;
pub mod celt;
pub mod celt_lpc;
pub mod hp_cutoff;
pub mod mdct;
pub mod modes;
pub mod pitch;
pub mod pvq;
pub mod quant_bands;
pub mod range_coder;
pub mod rate;
pub mod silk;

use celt::{CeltDecoder, CeltEncoder};
use hp_cutoff::hp_cutoff;
use range_coder::RangeCoder;
use silk::control_codec::silk_control_encoder;
use silk::enc_api::{silk_encode, silk_encode_prefill};
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
    /// Enable in-band FEC (LBRR). When set to true, the encoder will
    /// include redundant copies of previous frames (at lower quality) to
    /// allow packet-loss recovery on the decoder side.
    pub use_inband_fec: bool,
    /// Expected packet loss percentage (0-100). Used to determine whether
    /// LBRR encoding is worth the overhead. Non-zero value enables LBRR
    /// when `use_inband_fec` is also true.
    pub packet_loss_perc: i32,
    silk_initialized: bool,
    mode: OpusMode,
    // HP filter state
    variable_hp_smth2_q15: i32,
    hp_mem: Vec<i32>, // [4] for stereo, [2] for mono
    // Pre-allocated buffers for encoding (reuse across frames to avoid allocations)
    buf_filtered: Vec<i16>,
    buf_silk_input: Vec<i16>,
    buf_stereo_mid: Vec<i16>,
    buf_stereo_side: Vec<i16>,
    down2_state_first: [i32; 2],
    down2_state_second: [i32; 2],
    down2_3_state: [i32; 6],
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
        if silk_init_encoder(&mut *silk_enc, 0) != 0 {
            return Err("SILK encoder initialization failed");
        }

        let (opus_mode, bw) = match application {
            Application::Voip => (OpusMode::SilkOnly, Bandwidth::Narrowband),
            _ => (OpusMode::CeltOnly, Bandwidth::Fullband),
        };

        // Initialize HP filter: min cutoff = 60 Hz, in log scale Q15
        // VARIABLE_HP_MIN_CUTOFF_HZ = 60
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
            complexity: 0,
            use_cbr: false,
            use_inband_fec: false,
            packet_loss_perc: 0,
            silk_initialized: false,
            mode: opus_mode,
            variable_hp_smth2_q15,
            hp_mem: vec![0; channels * 2],
            // Pre-allocated buffers (will be resized on first use)
            buf_filtered: Vec::new(),
            buf_silk_input: Vec::new(),
            buf_stereo_mid: Vec::new(),
            buf_stereo_side: Vec::new(),
            down2_state_first: [0; 2],
            down2_state_second: [0; 2],
            down2_3_state: [0; 6],
        })
    }

    /// Enable Hybrid mode (SILK + CELT).
    /// This is only valid for sampling rates of 24000 or 48000 Hz.
    /// In Hybrid mode, SILK handles the low-frequency portion and CELT handles the high-frequency.
    /// IMPORTANT: Hybrid mode requires the input to be at the encoder's sampling rate (24kHz or 48kHz).
    /// The SILK encoder will internally use 16kHz for its processing.
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
        self.silk_initialized = false; // Force re-initialization at SILK internal rate
        Ok(())
    }

    pub fn encode(
        &mut self,
        input: &[f32],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        if output.len() < 1 {
            return Err("Output buffer too small");
        }

        let frame_rate = frame_rate_from_params(self.sampling_rate, frame_size)
            .ok_or("Invalid frame size for sampling rate")?;

        // Mode decision can be dynamic here, for now using stored mode
        let mode = self.mode;
        if mode == OpusMode::CeltOnly {
            match frame_rate {
                400 | 200 | 100 | 50 => {}
                _ => return Err("Unsupported frame size for CELT-only mode"),
            }
        }

        let toc = gen_toc(mode, frame_rate, self.bandwidth, self.channels);
        output[0] = toc;

        // C: cbr_bytes = IMIN((bitrate_to_bits(bitrate, Fs, frame_size)+4)/8, max_data_bytes)
        // bitrate_to_bits = bitrate * 6 / (6 * Fs / frame_size) = bitrate * frame_size / Fs
        let target_bits =
            (self.bitrate_bps as i64 * frame_size as i64 / self.sampling_rate as i64) as i32;
        let cbr_bytes = ((target_bits + 4) / 8) as usize;
        let max_data_bytes = output.len();

        // In CBR mode, use cbr_bytes; in VBR mode, use full buffer
        let n_bytes = if self.use_cbr {
            cbr_bytes.min(max_data_bytes).max(1)
        } else {
            max_data_bytes
        };

        // C: ec_enc_init(&enc, data+1, orig_max_data_bytes-1)
        // Range coder buffer is payload only (excluding TOC byte)
        let mut rc = RangeCoder::new_encoder((max_data_bytes - 1) as u32);

        if mode == OpusMode::SilkOnly || mode == OpusMode::Hybrid {
            /* Initialize/configure SILK encoder if needed.
               For Hybrid mode, SILK operates at 16kHz internal rate regardless of
               the encoder's API sampling rate (24kHz or 48kHz). The input must be
               downsampled before passing to SILK.
            */
            let silk_fs_khz = if mode == OpusMode::Hybrid {
                16 // SILK always uses 16kHz in Hybrid (SWB) mode
            } else {
                self.sampling_rate / 1000
            };
            // SILK frame duration matches the API frame duration
            let frame_ms = (frame_size as i32 * 1000) / self.sampling_rate;
            if !self.silk_initialized || self.silk_enc.s_cmn.fs_khz != silk_fs_khz as i32 {
                let silk_init_bitrate = (((n_bytes - 1) * 8) as i64 * self.sampling_rate as i64
                    / frame_size as i64) as i32;
                silk_control_encoder(
                    &mut *self.silk_enc,
                    silk_fs_khz as i32,
                    frame_ms,
                    silk_init_bitrate,
                    self.complexity,
                );
                self.silk_enc.s_cmn.use_cbr = if self.use_cbr { 1 } else { 0 };
                // Set number of channels for stereo support
                self.silk_enc.s_cmn.n_channels = self.channels as i32;
                self.silk_initialized = true;
                self.down2_state_first = [0; 2];
                self.down2_state_second = [0; 2];
                self.down2_3_state = [0; 6];

                let encoder_buffer = self.sampling_rate as usize / 100; // Fs/100 = 80 samples @8kHz
                let prefill_zeros = vec![0i16; encoder_buffer];
                silk_encode_prefill(
                    &mut *self.silk_enc,
                    &prefill_zeros,
                    0, // activity = inactive (silence)
                );
            }
            // Always update FEC and packet loss settings (may change between calls)
            self.silk_enc.s_cmn.use_in_band_fec = if self.use_inband_fec { 1 } else { 0 };
            self.silk_enc.s_cmn.packet_loss_perc = self.packet_loss_perc.clamp(0, 100);
            // Enable LBRR in the state struct when FEC is requested
            self.silk_enc.s_cmn.lbrr_enabled = if self.use_inband_fec { 1 } else { 0 };
            // LBRR gain increase (C default: 2 steps ~= 3dB quality reduction for LBRR copy)
            if self.silk_enc.s_cmn.lbrr_gain_increases == 0 {
                self.silk_enc.s_cmn.lbrr_gain_increases = 2;
            }

            // Apply HP filter before SILK encoding (matching C opus_encode_native)
            // Update variable_HP_smth2_Q15 from SILK's smth1
            let hp_freq_smth1 = if mode == OpusMode::CeltOnly {
                silk_lin2log(60) << 8 // VARIABLE_HP_MIN_CUTOFF_HZ = 60
            } else {
                self.silk_enc.s_cmn.variable_hp_smth1_q15
            };

            // Second-order smoother: smth2 = smth2 + COEF2 * (smth1 - smth2)
            // VARIABLE_HP_SMTH_COEF2 = 0.0025 (Q16 = 164)
            const VARIABLE_HP_SMTH_COEF2_Q16: i32 = 164;
            self.variable_hp_smth2_q15 = silk_smlawb(
                self.variable_hp_smth2_q15,
                hp_freq_smth1 - self.variable_hp_smth2_q15,
                VARIABLE_HP_SMTH_COEF2_Q16,
            );

            // Convert from log scale to Hz
            let cutoff_hz = silk_log2lin(silk_rshift(self.variable_hp_smth2_q15, 8));

            // Apply HP filter to input (reuse pre-allocated buffer)
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
                // No HP filter for non-VOIP, just convert
                for (i, &x) in input.iter().enumerate() {
                    self.buf_filtered[i] = (x * 32768.0).clamp(-32768.0, 32767.0) as i16;
                }
            }

            let input_i16 = &self.buf_filtered;

            // Convert input to SILK format (reuse pre-allocated buffers)
            // For SILK stereo: convert L/R to Mid (L+R)/2 and Side (L-R) for encoding.
            // For Hybrid mode: downsample from API rate to SILK's 16kHz internal rate.
            let silk_input: &[i16] = if mode == OpusMode::SilkOnly && self.channels == 2 {
                // Convert interleaved stereo (L R L R ...) to Mid channel
                // Mid = (L + R) / 2, Side = L - R
                let frame_length = input_i16.len() / 2;
                self.buf_stereo_mid.resize(frame_length, 0);
                self.buf_stereo_side.resize(frame_length, 0);
                for i in 0..frame_length {
                    // L is at index 2*i, R is at index 2*i+1
                    let l = input_i16[2 * i] as i32;
                    let r = input_i16[2 * i + 1] as i32;
                    self.buf_stereo_mid[i] = ((l + r) / 2) as i16;
                    self.buf_stereo_side[i] = (l - r) as i16;
                }
                // Store side for later stereo encoding
                self.silk_enc.stereo.side = self.buf_stereo_side.clone();
                &self.buf_stereo_mid
            } else if mode == OpusMode::Hybrid && self.sampling_rate > 16000 {
                if self.sampling_rate == 48000 {
                    let stage1_size = frame_size / 2;
                    let mut stage1_buf = vec![0i16; stage1_size];
                    silk_resampler_down2(
                        &mut self.down2_state_first,
                        &mut stage1_buf,
                        input_i16,
                        frame_size as i32,
                    );
                    let silk_frame_size = stage1_size * 2 / 3;
                    self.buf_silk_input.resize(silk_frame_size, 0);
                    silk_resampler_down2_3(
                        &mut self.down2_3_state,
                        &mut self.buf_silk_input,
                        &stage1_buf,
                        stage1_size as i32,
                    );
                } else {
                    // 24kHz → 16kHz via silk_resampler_down2_3 (2/3 ratio)
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

            let silk_max_bits = if mode == OpusMode::Hybrid {
                ((n_bytes - 1) * 8 * 2 / 5) as i32 // ~40% for SILK in Hybrid
            } else {
                ((n_bytes - 1) * 8) as i32
            };
            let silk_rate_for_calc = if mode == OpusMode::Hybrid {
                16000
            } else {
                self.sampling_rate
            };
            let silk_frame_len = silk_input.len();
            let silk_bitrate = if mode == OpusMode::Hybrid {
                (silk_max_bits as i64 * silk_rate_for_calc as i64 / silk_frame_len as i64) as i32
            } else {
                // Match C: silk_mode.bitRate = 8 * cbr_bytes * (Fs / frame_size)
                (8i64 * n_bytes as i64 * silk_rate_for_calc as i64 / silk_frame_len as i64) as i32
            };
            let ret = silk_encode(
                &mut *self.silk_enc,
                silk_input,
                silk_input.len(),
                &mut rc,
                &mut pn_bytes,
                silk_bitrate,
                silk_max_bits,
                if self.use_cbr { 1 } else { 0 },
                1, // activity = 1 (assume active)
            );
            if ret != 0 {
                return Err("SILK encoding failed");
            }
        }

        if mode == OpusMode::CeltOnly || mode == OpusMode::Hybrid {
            let start_band = if mode == OpusMode::Hybrid { 17 } else { 0 };
            self.celt_enc
                .encode_with_start_band(input, frame_size, &mut rc, start_band);
        }

        rc.done();

        let silk_payload: Vec<u8> = if mode == OpusMode::SilkOnly {
            // Build the complete output buffer
            let mut combined = Vec::with_capacity(rc.storage as usize);
            combined.extend_from_slice(&rc.buf[0..rc.offs as usize]);
            combined.extend_from_slice(
                &rc.buf[(rc.storage - rc.end_offs) as usize..rc.storage as usize],
            );

            // Strip trailing zeros (C: while(ret>2&&data[ret]==0)ret--)
            while combined.len() > 2 && combined[combined.len() - 1] == 0 {
                combined.pop();
            }

            combined
        } else {
            Vec::new()
        };

        let total_bytes = if mode == OpusMode::SilkOnly {
            silk_payload.len()
        } else {
            n_bytes
        };

        let payload_bytes = total_bytes.min(output.len() - 1);
        let ret_with_toc = payload_bytes + 1; // +1 for TOC byte

        if mode == OpusMode::SilkOnly {
            let target_total = if self.use_cbr {
                n_bytes.min(output.len())
            } else {
                ret_with_toc
            };

            // Clamp frame data to the space available after TOC + count byte (2 bytes)
            let max_frame_bytes = target_total
                .saturating_sub(2)
                .min(output.len().saturating_sub(2));
            let frame_len = payload_bytes.min(max_frame_bytes);

            output[0] = toc | 0x03;
            output[1] = 0x01;

            if frame_len > 0 {
                output[2..2 + frame_len].copy_from_slice(&silk_payload[..frame_len]);
            }

            let out_len = output.len();
            let written = 2 + frame_len;
            for byte in output[written..target_total.min(out_len)].iter_mut() {
                *byte = 0;
            }

            return Ok(target_total.min(out_len));
        }

        if mode == OpusMode::SilkOnly {
            output[1..1 + payload_bytes].copy_from_slice(&silk_payload[..payload_bytes]);
        } else {
            output[1..1 + payload_bytes].copy_from_slice(&rc.buf[..payload_bytes]);
        }
        Ok(ret_with_toc)
    }
}

pub struct OpusDecoder {
    celt_dec: CeltDecoder,
    silk_dec: silk::dec_api::SilkDecoder,
    sampling_rate: i32,
    channels: usize,
    // State tracking for mode transitions
    prev_mode: Option<OpusMode>,
    frame_size: usize,
    /// Internal bandwidth from previous frame
    bandwidth: Bandwidth,
    /// Stream channels from previous frame
    stream_channels: usize,
    /// SILK decoder resampler (internal rate → API rate)
    silk_resampler: silk::resampler::SilkResampler,
    /// Previous internal sample rate (for resampler re-init)
    prev_internal_rate: i32,
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

        // Parse packet structure: handle Code 0, 1, 2, 3
        let code = toc & 0x03;
        let payload_data;

        match code {
            0 => {
                // Code 0: one frame
                payload_data = &input[1..];
            }
            3 => {
                // Code 3: arbitrary number of frames (CBR or VBR)
                if input.len() < 2 {
                    return Err("Code 3 packet too short");
                }
                let count_byte = input[1];
                let _frame_count = (count_byte & 0x3F) as usize;
                let padding_flag = (count_byte & 0x40) != 0;

                // Parse padding
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
                            pad_len += 254; // 255 means 254 data bytes + next count byte
                        } else {
                            pad_len += p;
                            break;
                        }
                    }
                    // Frame data is between ptr and (input.len() - pad_len)
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
                // Code 1 / 2: two frames
                payload_data = &input[1..];
            }
        }

        self.frame_size = frame_size;
        self.bandwidth = bandwidth;
        self.stream_channels = packet_channels;

        match mode {
            OpusMode::SilkOnly => {
                // Determine internal sample rate from bandwidth
                let internal_sample_rate = match bandwidth {
                    Bandwidth::Narrowband => 8000,
                    Bandwidth::Mediumband => 12000,
                    Bandwidth::Wideband => 16000,
                    _ => 16000,
                };

                // Decode SILK frame
                let mut rc = RangeCoder::new_decoder(payload_data.to_vec());
                let internal_frame_size =
                    (frame_duration_ms * internal_sample_rate / 1000) as usize;
                let mut pcm_i16 = vec![0i16; internal_frame_size * self.channels];

                let payload_size_ms = frame_duration_ms;

                let ret = self.silk_dec.decode(
                    &mut rc,
                    &mut pcm_i16,
                    silk::decode_frame::FLAG_DECODE_NORMAL,
                    true, // new_packet
                    payload_size_ms,
                    internal_sample_rate,
                );

                if ret < 0 {
                    return Err("SILK decoding failed");
                }

                let decoded_samples = ret as usize;

                // Sample rate conversion via SilkResampler
                if self.sampling_rate == internal_sample_rate {
                    // No resampling needed: direct i16 → f32 conversion
                    let n = decoded_samples.min(frame_size).min(output.len());
                    for i in 0..n {
                        output[i] = pcm_i16[i] as f32 / 32768.0;
                    }
                    self.prev_mode = Some(OpusMode::SilkOnly);
                    Ok(n)
                } else {
                    // (Re-)initialize resampler if internal rate changed
                    if internal_sample_rate != self.prev_internal_rate {
                        self.silk_resampler
                            .init(internal_sample_rate, self.sampling_rate);
                        self.prev_internal_rate = internal_sample_rate;
                    }

                    let ratio = self.sampling_rate as f64 / internal_sample_rate as f64;
                    let out_len = ((decoded_samples as f64 * ratio) as usize).min(frame_size);
                    let mut pcm_out = vec![0i16; out_len];
                    self.silk_resampler.process(
                        &mut pcm_out,
                        &pcm_i16[..decoded_samples],
                        decoded_samples as i32,
                    );

                    let n = out_len.min(output.len());
                    for i in 0..n {
                        output[i] = pcm_out[i] as f32 / 32768.0;
                    }
                    self.prev_mode = Some(OpusMode::SilkOnly);
                    Ok(n)
                }
            }

            OpusMode::CeltOnly => {
                // CELT decoding (existing path)
                self.celt_dec.decode(payload_data, frame_size, output);
                self.prev_mode = Some(OpusMode::CeltOnly);
                Ok(frame_size)
            }

            OpusMode::Hybrid => {
                let internal_sample_rate = match bandwidth {
                    Bandwidth::Superwideband => 16000, // 24kHz SWB: SILK at 16kHz
                    Bandwidth::Fullband => 16000,      // 48kHz FB:  SILK at 16kHz
                    _ => 16000,
                };

                // Step 1: Decode SILK low-frequency portion
                let mut rc = RangeCoder::new_decoder(payload_data.to_vec());
                let internal_frame_size =
                    (frame_duration_ms * internal_sample_rate / 1000) as usize;
                let mut pcm_silk_i16 = vec![0i16; internal_frame_size * self.channels];

                let ret = self.silk_dec.decode(
                    &mut rc,
                    &mut pcm_silk_i16,
                    silk::decode_frame::FLAG_DECODE_NORMAL,
                    true,
                    frame_duration_ms,
                    internal_sample_rate,
                );

                // Convert SILK PCM to f32 at output rate
                let mut silk_out = vec![0.0f32; frame_size * self.channels];
                if ret > 0 {
                    let decoded_samples = ret as usize;
                    if self.sampling_rate == internal_sample_rate {
                        let n = decoded_samples.min(frame_size);
                        for i in 0..n {
                            silk_out[i] = pcm_silk_i16[i] as f32 / 32768.0;
                        }
                    } else {
                        // Resample SILK output to API rate
                        if internal_sample_rate != self.prev_internal_rate {
                            self.silk_resampler
                                .init(internal_sample_rate, self.sampling_rate);
                            self.prev_internal_rate = internal_sample_rate;
                        }
                        let ratio = self.sampling_rate as f64 / internal_sample_rate as f64;
                        let out_len = ((decoded_samples as f64 * ratio) as usize).min(frame_size);
                        let mut pcm_resampled = vec![0i16; out_len];
                        self.silk_resampler.process(
                            &mut pcm_resampled,
                            &pcm_silk_i16[..decoded_samples],
                            decoded_samples as i32,
                        );
                        for i in 0..out_len.min(frame_size) {
                            silk_out[i] = pcm_resampled[i] as f32 / 32768.0;
                        }
                    }
                }

                // Step 2: Decode CELT high-frequency portion (start_band = 17)
                let mut celt_out = vec![0.0f32; frame_size * self.channels];
                // start_band = 17 matches the SWB/FB Hybrid crossover point
                self.celt_dec
                    .decode_with_start_band(payload_data, frame_size, &mut celt_out, 17);

                let n = frame_size.min(output.len());
                for i in 0..n {
                    output[i] = silk_out[i] + celt_out[i];
                    // Clamp to avoid clipping
                    output[i] = output[i].clamp(-1.0, 1.0);
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
            // Hybrid configs: 16-19 = SWB, 20-23 = FB
            // Period 0 = 10ms, period 1 = 20ms
            // (Hybrid only supports 10ms and 20ms)
            let base_config = if bandwidth == Bandwidth::Superwideband {
                16
            } else {
                20
            };
            let hybrid_period = if frame_rate >= 100 { 0 } else { 1 }; // 100fps=10ms, 50fps=20ms
            ((base_config + hybrid_period) << 3) as u8
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

/// Extract bandwidth from TOC byte
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

/// Extract frame duration in milliseconds from TOC byte
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
                0 => {
                    // 2.5 ms
                    // Return 2 for 2.5ms (caller will handle)
                    2 // Approximation; actual is 2.5ms
                }
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
}
