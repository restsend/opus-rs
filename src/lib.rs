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
use silk::enc_api::silk_encode;
use silk::init_encoder::silk_init_encoder;
use silk::lin2log::silk_lin2log;
use silk::log2lin::silk_log2lin;
use silk::macros::*;
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
    silk_initialized: bool,
    mode: OpusMode,
    // HP filter state
    variable_hp_smth2_q15: i32,
    hp_mem: Vec<i32>, // [4] for stereo, [2] for mono
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
            silk_initialized: false,
            mode: opus_mode,
            variable_hp_smth2_q15,
            hp_mem: vec![0; channels * 2],
        })
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
            /* Initialize/configure SILK encoder if needed */
            let fs_khz = self.sampling_rate / 1000;
            let frame_ms = (frame_size as i32 * 1000) / self.sampling_rate;
            if !self.silk_initialized || self.silk_enc.s_cmn.fs_khz != fs_khz as i32 {
                let silk_init_bitrate = (((n_bytes - 1) * 8) as i64 * self.sampling_rate as i64
                    / frame_size as i64) as i32;
                silk_control_encoder(
                    &mut *self.silk_enc,
                    fs_khz as i32,
                    frame_ms,
                    silk_init_bitrate,
                    self.complexity,
                );
                self.silk_enc.s_cmn.use_cbr = if self.use_cbr { 1 } else { 0 };
                self.silk_initialized = true;
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

            // Apply HP filter to input
            let mut filtered_i16 = vec![0i16; frame_size * self.channels];
            if self.application == Application::Voip {
                hp_cutoff(
                    input,
                    cutoff_hz,
                    &mut filtered_i16,
                    &mut self.hp_mem,
                    frame_size,
                    self.channels,
                    self.sampling_rate,
                );
            } else {
                // No HP filter for non-VOIP, just convert
                for (i, &x) in input.iter().enumerate() {
                    filtered_i16[i] = (x * 32768.0).clamp(-32768.0, 32767.0) as i16;
                }
            }

            let input_i16 = filtered_i16;

            let mut pn_bytes = 0;

            /* Use the top-level silk_encode which handles:
            - VAD, LBRR preamble, SNR control, HP variable cutoff,
            - multi-frame packets, and VAD flag patching */
            // C: st->silk_mode.maxBits = (max_data_bytes-1)*8;
            // The -1 accounts for the TOC byte which is not part of the SILK payload
            let silk_max_bits = ((n_bytes - 1) * 8) as i32;
            let silk_bitrate =
                (silk_max_bits as i64 * self.sampling_rate as i64 / frame_size as i64) as i32;
            let ret = silk_encode(
                &mut *self.silk_enc,
                &input_i16,
                input_i16.len(),
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
            self.celt_enc.encode(input, frame_size, &mut rc);
        }

        rc.done();

        // For SILK-only mode: the actual size is determined by range coder usage.
        // C uses ec_tell() for size, then strips trailing zeros from the combined output
        // Range coder writes from both ends: [0..offs] and [storage-end_offs..storage]
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

        // Build output: TOC + payload
        // ret_bytes = total payload size (SILK or CELT data from rc)
        let payload_bytes = total_bytes.min(output.len() - 1);
        let ret_with_toc = payload_bytes + 1; // +1 for TOC byte

        // For SILK-only mode, always use Code 3 format (matching C behavior)
        // This provides self-delimiting frame boundaries
        if mode == OpusMode::SilkOnly {
            let target_total = if self.use_cbr {
                n_bytes.min(output.len())
            } else {
                ret_with_toc
            };
            let frame_len = payload_bytes;

            // Code 3 header takes 2 bytes (TOC + count)
            let available_for_frame_and_pad = if target_total > 2 {
                target_total - 2
            } else {
                0
            };
            let pad_amount = available_for_frame_and_pad.saturating_sub(frame_len);

            if pad_amount > 0 {
                // Code 3 with padding
                output[0] = toc | 0x03; // Code 3
                let count_byte = 1u8 | 0x40; // 1 frame, padding flag set
                output[1] = count_byte;

                // Encode padding length (RFC 6716 §3.2.1)
                let nb_255s = (pad_amount - 1) / 255;
                let mut ptr = 2usize;
                for _ in 0..nb_255s {
                    output[ptr] = 255;
                    ptr += 1;
                }
                output[ptr] = (pad_amount - 255 * nb_255s - 1) as u8;
                ptr += 1;

                // Copy frame data
                output[ptr..ptr + frame_len].copy_from_slice(&silk_payload[..frame_len]);
                ptr += frame_len;

                // Fill padding with zeros
                while ptr < target_total {
                    output[ptr] = 0;
                    ptr += 1;
                }

                return Ok(target_total);
            } else {
                // Code 3 without padding (matching C output format)
                output[0] = toc | 0x03; // Code 3
                output[1] = 0x01; // 1 frame, no padding flag
                output[2..2 + frame_len].copy_from_slice(&silk_payload[..frame_len]);
                return Ok(2 + frame_len);
            }
        }

        // No CBR padding needed — write as Code 0
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
        // Initialize SILK decoder with API sample rate
        // Internal SILK rate will be set from the TOC
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

                // Convert i16 to f32 and handle sample rate conversion
                // If internal rate matches API rate, direct copy
                // Otherwise, we need resampling (simplified: just output at internal rate)
                let output_samples = if self.sampling_rate == internal_sample_rate {
                    decoded_samples
                } else {
                    // Simple linear interpolation resampling
                    let ratio = self.sampling_rate as f64 / internal_sample_rate as f64;
                    let out_len = (decoded_samples as f64 * ratio) as usize;
                    out_len.min(frame_size)
                };

                if self.sampling_rate == internal_sample_rate {
                    // Direct conversion
                    for i in 0..output_samples.min(output.len()) {
                        output[i] = pcm_i16[i] as f32 / 32768.0;
                    }
                } else {
                    // Simple resampling (linear interpolation)
                    let ratio = internal_sample_rate as f64 / self.sampling_rate as f64;
                    for i in 0..output_samples.min(output.len()) {
                        let src_pos = i as f64 * ratio;
                        let src_idx = src_pos as usize;
                        let frac = src_pos - src_idx as f64;
                        if src_idx + 1 < decoded_samples {
                            let s0 = pcm_i16[src_idx] as f64 / 32768.0;
                            let s1 = pcm_i16[src_idx + 1] as f64 / 32768.0;
                            output[i] = (s0 + frac * (s1 - s0)) as f32;
                        } else if src_idx < decoded_samples {
                            output[i] = pcm_i16[src_idx] as f32 / 32768.0;
                        } else {
                            output[i] = 0.0;
                        }
                    }
                }

                self.prev_mode = Some(OpusMode::SilkOnly);
                Ok(output_samples.min(frame_size))
            }

            OpusMode::CeltOnly => {
                // CELT decoding (existing path)
                self.celt_dec.decode(payload_data, frame_size, output);
                self.prev_mode = Some(OpusMode::CeltOnly);
                Ok(frame_size)
            }

            OpusMode::Hybrid => {
                // Hybrid mode: SILK + CELT
                // For now, just decode CELT part
                // TODO: implement full hybrid decode
                return Err("Hybrid mode not yet supported");
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
            let bw = (bandwidth as i32 - Bandwidth::Superwideband as i32) << 4;
            let per = (period - 2) << 3;
            (0x60 | bw | per) as u8
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
