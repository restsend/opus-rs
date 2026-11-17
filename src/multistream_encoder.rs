//! libopus's multistream encoder (`src/opus_multistream_encoder.c`): one
//! [`OpusEncoder`] per elementary stream, each coding one or two input
//! channels picked by the channel mapping, with the bitrate shared between
//! the streams and the stream packets joined, self-delimited, into one
//! multistream packet.

use crate::encoder_ctl::frame_size_select;
use crate::error::{BAD_ARG, BUFFER_TOO_SMALL, INTERNAL_ERROR};
use crate::repacketizer::Repacketizer;
use crate::{Application, Bandwidth, FrameDuration, OpusEncoder, PcmInput, Signal, bits_to_bitrate, bitrate_to_bits};

/// libopus `MS_FRAME_TMP`: the most one stream codes into (six frames).
const MS_FRAME_TMP: usize = 6 * 1275 + 12;

/// The input samples of one call.
#[derive(Clone, Copy)]
enum MsPcm<'a> {
    F32(&'a [f32]),
    I16(&'a [i16]),
    I24(&'a [i32]),
}

/// libopus `OpusMSEncoder` created by `opus_multistream_encoder_create`.
pub struct MultistreamEncoder {
    encoders: Vec<OpusEncoder>,
    channels: usize,
    coupled: usize,
    mapping: Vec<u8>,
    /// libopus `bitrate_bps` of the whole packet.
    bitrate_bps: i32,
    application: Application,
    sampling_rate: i32,
    variable_duration: FrameDuration,
}

impl MultistreamEncoder {
    /// libopus `opus_multistream_encoder_create`: `channels` input
    /// channels coded in `streams` streams, the first `coupled` of them
    /// stereo; `mapping[i]` is the decoded channel input channel `i` goes
    /// to (coupled streams' left and right first, then mono streams; 255
    /// drops it). `error::BAD_ARG` for an invalid layout or settings.
    pub fn new(
        sampling_rate: i32,
        channels: usize,
        streams: usize,
        coupled: usize,
        mapping: &[u8],
        application: Application,
    ) -> Result<Self, &'static str> {
        if !(1..=255).contains(&channels)
            || coupled > streams
            || streams < 1
            || streams > 255 - coupled
            || streams + coupled > channels
            || mapping.len() < channels
        {
            return Err(BAD_ARG);
        }
        let mapping = mapping[..channels].to_vec();
        // libopus validate_layout and validate_encoder_layout.
        if mapping.iter().any(|&m| m as usize >= streams + coupled && m != 255) {
            return Err(BAD_ARG);
        }
        let mut encoders = Vec::with_capacity(streams);
        for s in 0..streams {
            let ch = if s < coupled { 2 } else { 1 };
            encoders.push(OpusEncoder::new(sampling_rate, ch, application).map_err(|_| BAD_ARG)?);
        }
        let enc = MultistreamEncoder {
            encoders,
            channels,
            coupled,
            mapping,
            bitrate_bps: OpusEncoder::BITRATE_AUTO,
            application,
            sampling_rate,
            variable_duration: FrameDuration::Argument,
        };
        if (0..streams).any(|s| enc.stream_channels(s).is_none()) {
            return Err(BAD_ARG);
        }
        Ok(enc)
    }

    /// The input channels of stream `s` (libopus `get_left_channel` and
    /// `get_right_channel`, or `get_mono_channel`).
    fn stream_channels(&self, s: usize) -> Option<(usize, Option<usize>)> {
        let find = |m: usize| self.mapping.iter().position(|&x| x as usize == m);
        if s < self.coupled {
            Some((find(2 * s)?, Some(find(2 * s + 1)?)))
        } else {
            Some((find(s + self.coupled)?, None))
        }
    }

    /// Encodes a frame of interleaved float input of `frame_size` samples
    /// per channel (libopus `opus_multistream_encode_float`) and returns the
    /// packet length.
    pub fn encode(&mut self, pcm: &[f32], frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        self.encode_native(MsPcm::F32(pcm), frame_size, out)
    }

    /// [`Self::encode`] for 16-bit input (libopus `opus_multistream_encode`).
    pub fn encode_i16(&mut self, pcm: &[i16], frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        self.encode_native(MsPcm::I16(pcm), frame_size, out)
    }

    /// [`Self::encode`] for 24-bit input in 32-bit integers (libopus
    /// `opus_multistream_encode24`).
    pub fn encode24(&mut self, pcm: &[i32], frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        self.encode_native(MsPcm::I24(pcm), frame_size, out)
    }

    /// libopus `surround_rate_allocation` without an LFE stream, at least
    /// 500 b/s each.
    fn rate_allocation(&self, frame_size: usize) -> Vec<i32> {
        let fs = self.sampling_rate;
        let streams = self.encoders.len() as i32;
        let nb_coupled = self.coupled as i32;
        let nb_uncoupled = streams - nb_coupled;
        let nb_normal = 2 * nb_coupled + nb_uncoupled;
        let channel_offset = 40 * 50.max(fs / frame_size as i32);
        let bitrate = match self.bitrate_bps {
            OpusEncoder::BITRATE_AUTO => nb_normal * (channel_offset + fs + 10000),
            OpusEncoder::BITRATE_MAX => nb_normal * 750_000,
            b => b,
        };
        let stream_offset = ((bitrate - channel_offset * nb_normal) / nb_normal / 2).clamp(0, 20000);
        let coupled_ratio = 512;
        let total = (nb_uncoupled << 8) + coupled_ratio * nb_coupled;
        let channel_rate = (256
            * (bitrate as i64 - (stream_offset * (nb_coupled + nb_uncoupled)) as i64 - (channel_offset * nb_normal) as i64)
            / total as i64) as i32;
        (0..streams)
            .map(|s| {
                let rate = if s < nb_coupled {
                    2 * channel_offset + 0.max(stream_offset + (channel_rate.wrapping_mul(coupled_ratio) >> 8))
                } else {
                    channel_offset + 0.max(stream_offset + channel_rate)
                };
                rate.max(500)
            })
            .collect()
    }

    /// libopus `opus_multistream_encode_native`.
    fn encode_native(&mut self, pcm: MsPcm<'_>, analysis_frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        let fs = self.sampling_rate;
        let vbr = self.encoders[0].vbr();
        let frame_size = frame_size_select(fs, analysis_frame_size, self.variable_duration).ok_or(BAD_ARG)?;
        let streams = self.encoders.len();
        let len = match pcm {
            MsPcm::F32(x) => x.len(),
            MsPcm::I16(x) => x.len(),
            MsPcm::I24(x) => x.len(),
        };
        if len < analysis_frame_size * self.channels {
            return Err(BAD_ARG);
        }
        let hundred_ms = fs / frame_size as i32 == 10;
        let smallest = streams * 2 - 1 + if hundred_ms { streams } else { 0 };
        if out.len() < smallest {
            return Err(BUFFER_TOO_SMALL);
        }
        let rates = self.rate_allocation(frame_size);
        let rate_sum: i32 = rates.iter().sum();
        let mut max_data_bytes = out.len() as i32;
        if !vbr {
            if self.bitrate_bps == OpusEncoder::BITRATE_AUTO {
                max_data_bytes = max_data_bytes.min((bitrate_to_bits(rate_sum, fs, frame_size) + 4) / 8);
            } else if self.bitrate_bps != OpusEncoder::BITRATE_MAX {
                let bytes = (bitrate_to_bits(self.bitrate_bps, fs, frame_size) + 4) / 8;
                max_data_bytes = max_data_bytes.min((smallest as i32).max(bytes));
            }
        }
        for (enc, &rate) in self.encoders.iter_mut().zip(&rates) {
            let _ = enc.set_bitrate(rate);
        }
        let n = analysis_frame_size;
        let mut f32_buf = vec![0.0f32; 2 * n];
        let mut i16_buf = vec![0i16; 2 * n];
        let mut i24_buf = vec![0i32; 2 * n];
        let mut tmp = vec![0u8; MS_FRAME_TMP];
        let mut tot_size = 0usize;
        for s in 0..streams {
            let (left, right) = self.stream_channels(s).ok_or(INTERNAL_ERROR)?;
            let chans: &[usize] = &match right {
                Some(r) => [left, r],
                None => [left, left],
            }[..1 + right.is_some() as usize];
            let c = chans.len();
            // The stream's channels of the call's samples, the analysis's
            // too (libopus copy_channel_in, and downmix with c1 and c2).
            for (k, &ch) in chans.iter().enumerate() {
                for i in 0..n {
                    let at = i * self.channels + ch;
                    match pcm {
                        MsPcm::F32(x) => f32_buf[i * c + k] = x[at],
                        MsPcm::I16(x) => i16_buf[i * c + k] = x[at],
                        MsPcm::I24(x) => {
                            i24_buf[i * c + k] = x[at];
                            f32_buf[i * c + k] = (1.0 / 32768.0 / 256.0) * x[at] as f32;
                        }
                    }
                }
            }
            let mut curr_max = max_data_bytes - tot_size as i32;
            curr_max -= 0.max(2 * (streams - s - 1) as i32 - 1);
            if hundred_ms {
                curr_max -= (streams - s - 1) as i32;
            }
            curr_max = curr_max.min(MS_FRAME_TMP as i32);
            let last = s == streams - 1;
            if !last {
                curr_max -= if curr_max > 253 { 2 } else { 1 };
            }
            let enc = &mut self.encoders[s];
            if !vbr && last {
                let _ = enc.set_bitrate(bits_to_bitrate(curr_max * 8, fs, frame_size));
            }
            if curr_max <= 0 {
                return Err(BAD_ARG);
            }
            let tmp = &mut tmp[..curr_max as usize];
            let len = match pcm {
                MsPcm::F32(_) => enc.encode_selected(PcmInput::F32(&f32_buf[..n * c]), n, frame_size, tmp, None),
                MsPcm::I16(_) => enc.encode_selected(PcmInput::I16(&i16_buf[..n * c]), n, frame_size, tmp, None),
                MsPcm::I24(_) => enc.encode_selected(
                    PcmInput::F32(&f32_buf[..n * c]),
                    n,
                    frame_size,
                    tmp,
                    Some(&i24_buf[..n * c]),
                ),
            }?;
            let mut rp = Repacketizer::new();
            rp.cat(&tmp[..len]).map_err(|_| INTERNAL_ERROR)?;
            let room = (max_data_bytes as usize).saturating_sub(tot_size);
            let out_len = room.min(out.len() - tot_size);
            let written = rp.out_range_impl(
                0,
                rp.nb_frames(),
                &mut out[tot_size..tot_size + out_len],
                !vbr && last,
                !last,
            )?;
            tot_size += written;
        }
        Ok(tot_size)
    }

    /// Sets the bitrate of the whole packet (libopus `OPUS_SET_BITRATE`):
    /// [`OpusEncoder::BITRATE_AUTO`], [`OpusEncoder::BITRATE_MAX`], or a
    /// rate clamped to 500 to 750000 b/s per channel; others are
    /// `error::BAD_ARG`.
    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<(), &'static str> {
        let ch = self.channels as i32;
        self.bitrate_bps = match bitrate {
            OpusEncoder::BITRATE_AUTO | OpusEncoder::BITRATE_MAX => bitrate,
            ..=0 => return Err(BAD_ARG),
            b => b.clamp(500 * ch, 750_000 * ch),
        };
        Ok(())
    }

    /// The sum of the streams' bitrates (libopus `OPUS_GET_BITRATE`).
    pub fn bitrate(&self) -> i32 {
        self.encoders.iter().map(OpusEncoder::bitrate).sum()
    }

    /// The frame duration to code (libopus
    /// `OPUS_SET_EXPERT_FRAME_DURATION` on the multistream encoder).
    pub fn set_expert_frame_duration(&mut self, duration: FrameDuration) {
        self.variable_duration = duration;
    }

    /// The frame duration setting.
    pub fn expert_frame_duration(&self) -> FrameDuration {
        self.variable_duration
    }

    /// Restarts every stream (libopus `OPUS_RESET_STATE`).
    pub fn reset(&mut self) {
        self.encoders.iter_mut().for_each(OpusEncoder::reset);
    }

    /// The XOR of the streams' final ranges (libopus
    /// `OPUS_GET_FINAL_RANGE`).
    pub fn final_range(&self) -> u32 {
        self.encoders.iter().fold(0, |r, e| r ^ e.final_range())
    }

    /// The encoder of stream `stream` (libopus
    /// `OPUS_MULTISTREAM_GET_ENCODER_STATE`), `error::BAD_ARG` past the
    /// last stream. Its getters are the multistream encoder's, which
    /// libopus answers from the first stream.
    pub fn encoder_state(&mut self, stream: usize) -> Result<&mut OpusEncoder, &'static str> {
        self.encoders.get_mut(stream).ok_or(BAD_ARG)
    }

    /// The first stream's encoder, whose values libopus's multistream
    /// queries report (lookahead, sample rate, complexity, ...).
    pub fn first_stream(&self) -> &OpusEncoder {
        &self.encoders[0]
    }

    /// Applies `f` to every stream until one fails, as libopus forwards a
    /// control (`OPUS_SET_COMPLEXITY`, `OPUS_SET_VBR`, ...).
    fn each(&mut self, mut f: impl FnMut(&mut OpusEncoder) -> Result<(), &'static str>) -> Result<(), &'static str> {
        self.encoders.iter_mut().try_for_each(|e| f(e))
    }

    /// `OPUS_SET_COMPLEXITY` on every stream.
    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), &'static str> {
        self.each(|e| e.set_complexity(complexity))
    }

    /// `OPUS_SET_VBR` on every stream.
    pub fn set_vbr(&mut self, vbr: bool) {
        let _ = self.each(|e| Ok(e.set_vbr(vbr)));
    }

    /// `OPUS_SET_VBR_CONSTRAINT` on every stream.
    pub fn set_vbr_constraint(&mut self, constrained: bool) {
        let _ = self.each(|e| Ok(e.set_vbr_constraint(constrained)));
    }

    /// `OPUS_SET_BANDWIDTH` on every stream.
    pub fn set_bandwidth(&mut self, bandwidth: Bandwidth) {
        let _ = self.each(|e| Ok(e.set_bandwidth(bandwidth)));
    }

    /// `OPUS_SET_MAX_BANDWIDTH` on every stream.
    pub fn set_max_bandwidth(&mut self, bandwidth: Bandwidth) {
        let _ = self.each(|e| Ok(e.set_max_bandwidth(bandwidth)));
    }

    /// `OPUS_SET_SIGNAL` on every stream.
    pub fn set_signal(&mut self, signal: Signal) {
        let _ = self.each(|e| Ok(e.set_signal(signal)));
    }

    /// `OPUS_SET_APPLICATION` on every stream; the multistream encoder
    /// keeps the one it was created with, as libopus does.
    pub fn set_application(&mut self, application: Application) -> Result<(), &'static str> {
        self.each(|e| e.set_application(application))
    }

    /// `OPUS_SET_INBAND_FEC` on every stream.
    pub fn set_inband_fec(&mut self, fec: i32) -> Result<(), &'static str> {
        self.each(|e| e.set_inband_fec(fec))
    }

    /// `OPUS_SET_PACKET_LOSS_PERC` on every stream.
    pub fn set_packet_loss_perc(&mut self, percent: i32) -> Result<(), &'static str> {
        self.each(|e| e.set_packet_loss_perc(percent))
    }

    /// `OPUS_SET_DTX` on every stream.
    pub fn set_dtx(&mut self, dtx: bool) {
        let _ = self.each(|e| Ok(e.set_dtx(dtx)));
    }

    /// `OPUS_SET_LSB_DEPTH` on every stream.
    pub fn set_lsb_depth(&mut self, depth: i32) -> Result<(), &'static str> {
        self.each(|e| e.set_lsb_depth(depth))
    }

    /// `OPUS_SET_FORCE_CHANNELS` on every stream.
    pub fn set_force_channels(&mut self, channels: Option<usize>) -> Result<(), &'static str> {
        self.each(|e| e.set_force_channels(channels))
    }

    /// `OPUS_SET_PREDICTION_DISABLED` on every stream.
    pub fn set_prediction_disabled(&mut self, disabled: bool) {
        let _ = self.each(|e| Ok(e.set_prediction_disabled(disabled)));
    }

    /// `OPUS_SET_PHASE_INVERSION_DISABLED` on every stream.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        let _ = self.each(|e| Ok(e.set_phase_inversion_disabled(disabled)));
    }

    /// The application the multistream encoder was created with.
    pub fn application(&self) -> Application {
        self.application
    }
}
