//! libopus's multistream encoder (`src/opus_multistream_encoder.c`): one
//! [`OpusEncoder`] per elementary stream, each coding one or two input
//! channels picked by the channel mapping, with the bitrate shared between
//! the streams and the stream packets joined, self-delimited, into one
//! multistream packet.

use crate::encoder_ctl::frame_size_select;
use crate::error::{BAD_ARG, BUFFER_TOO_SMALL, INTERNAL_ERROR, UNIMPLEMENTED};
use crate::analysis::AnalysisPcm;
use crate::mapping_matrix::MappingMatrix;
use crate::multistream::MultistreamLayout;
use crate::repacketizer::Repacketizer;
use crate::{
    Application, Bandwidth, FrameDuration, OpusEncoder, OpusMode, PcmInput, Signal, bits_to_bitrate,
    bitrate_to_bits,
};

/// libopus `MS_FRAME_TMP`: the most one stream codes into (six frames).
const MS_FRAME_TMP: usize = 6 * 1275 + 12;

/// The CELT MDCT overlap of the surround analysis.
const OVERLAP: usize = 120;

/// The input samples of one call.
#[derive(Clone, Copy)]
enum MsPcm<'a> {
    F32(&'a [f32]),
    I16(&'a [i16]),
    I24(&'a [i32]),
}

/// libopus `MappingType`: how the streams share the bitrate and code.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MappingType {
    None,
    /// Family 1 with more than two channels: surround masking and an LFE
    /// stream.
    Surround,
    /// Family 2: CELT-only streams of equal rate.
    Ambisonics,
}

/// libopus `vorbis_mappings`: streams, coupled streams and mapping of 1 to
/// 8 channels in channel mapping family 1.
const VORBIS_MAPPINGS: [(usize, usize, &[u8]); 8] = [
    (1, 0, &[0]),
    (1, 1, &[0, 1]),
    (2, 1, &[0, 2, 1]),
    (2, 2, &[0, 1, 2, 3]),
    (3, 2, &[0, 4, 1, 2, 3]),
    (4, 2, &[0, 4, 1, 2, 3, 5]),
    (4, 3, &[0, 4, 1, 2, 3, 5, 6]),
    (5, 3, &[0, 6, 1, 2, 3, 4, 5, 7]),
];

/// What a multistream encoder keeps beside its stream encoders: the
/// layout, the settings and the surround analysis memory (libopus
/// `OpusMSEncoder` without the streams). It has a fixed size, so a
/// multistream encoder is this header plus its streams.
#[derive(Clone, Copy)]
pub struct MultistreamEncoderHeader {
    layout: MultistreamLayout,
    /// libopus `bitrate_bps` of the whole packet.
    bitrate_bps: i32,
    application: Application,
    sampling_rate: i32,
    variable_duration: FrameDuration,
    /// The projection encoder's mixing matrix, which forms each stream's
    /// channels from all input channels (libopus
    /// `opus_projection_copy_channel_in_*`).
    pub(crate) mixing: Option<MappingMatrix<'static>>,
    /// The projection encoder's demixing matrix, for the decoder.
    pub(crate) demixing: Option<MappingMatrix<'static>>,
    mapping_type: MappingType,
    lfe_stream: Option<usize>,
    /// The surround analysis's pre-emphasis and MDCT overlap per channel
    /// (libopus `ms_get_preemph_mem`, `ms_get_window_mem`), which runs on
    /// up to 8 channels.
    preemph_mem: [f32; 8],
    window_mem: [f32; 8 * OVERLAP],
    /// libopus `OpusMSEncoder.arch`, chosen at init.
    arch: crate::Arch,
}

/// libopus `OpusMSEncoder`.
pub struct MultistreamEncoder {
    pub(crate) head: MultistreamEncoderHeader,
    encoders: Vec<OpusEncoder>,
}

/// A multistream encoder over a header and its stream encoders held
/// elsewhere, such as in one block of memory: what [`MultistreamEncoder`]
/// encodes with.
pub struct MultistreamEncoderView<'a> {
    head: &'a mut MultistreamEncoderHeader,
    encoders: &'a mut [OpusEncoder],
}

impl MultistreamEncoderHeader {
    /// The header of libopus `opus_multistream_encoder_create`: `channels` input
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
        Self::new_impl(sampling_rate, channels, streams, coupled, mapping, application, MappingType::None, None)
    }

    /// The header of libopus `opus_multistream_surround_encoder_create`: the layout of
    /// `channels` channels in channel mapping family `mapping_family` (0
    /// mono or stereo, 1 Vorbis order up to 8 channels with an LFE stream
    /// from 6, 2 ambisonics, 255 one mono stream per channel), with
    /// libopus's surround masking and rate allocation. The layout is
    /// [`Self::streams`], [`Self::coupled_streams`] and [`Self::mapping`].
    /// Other families and counts are `error::UNIMPLEMENTED` or
    /// `error::BAD_ARG`, as in libopus.
    pub fn new_surround(
        sampling_rate: i32,
        channels: usize,
        mapping_family: i32,
        application: Application,
    ) -> Result<Self, &'static str> {
        if !(1..=255).contains(&channels) {
            return Err(BAD_ARG);
        }
        let mut lfe_stream = None;
        let (streams, coupled, mapping): (usize, usize, Vec<u8>) = match mapping_family {
            0 if channels == 1 => (1, 0, vec![0]),
            0 if channels == 2 => (1, 1, vec![0, 1]),
            1 if channels <= 8 => {
                let (s, c, m) = VORBIS_MAPPINGS[channels - 1];
                if channels >= 6 {
                    lfe_stream = Some(s - 1);
                }
                (s, c, m.to_vec())
            }
            255 => (channels, 0, (0..channels as u8).collect()),
            2 => {
                // libopus validate_ambisonics.
                let order_plus_one = (1..=15).take_while(|n| n * n <= channels).last().unwrap_or(0);
                let nondiegetic = channels - order_plus_one * order_plus_one;
                if channels > 227 || (nondiegetic != 0 && nondiegetic != 2) {
                    return Err(BAD_ARG);
                }
                let s = order_plus_one * order_plus_one + usize::from(nondiegetic != 0);
                let c = usize::from(nondiegetic != 0);
                let mut m = vec![0u8; channels];
                for i in 0..s - c {
                    m[i] = (i + c * 2) as u8;
                }
                for i in 0..c * 2 {
                    m[i + s - c] = i as u8;
                }
                (s, c, m)
            }
            _ => return Err(UNIMPLEMENTED),
        };
        let mapping_type = match mapping_family {
            1 if channels > 2 => MappingType::Surround,
            2 => MappingType::Ambisonics,
            _ => MappingType::None,
        };
        if mapping_type != MappingType::Surround {
            lfe_stream = None;
        }
        Self::new_impl(sampling_rate, channels, streams, coupled, &mapping, application, mapping_type, lfe_stream)
    }

    /// libopus `opus_multistream_encoder_init_impl`'s checks and settings.
    #[allow(clippy::too_many_arguments)]
    fn new_impl(
        sampling_rate: i32,
        channels: usize,
        streams: usize,
        coupled: usize,
        mapping: &[u8],
        application: Application,
        mapping_type: MappingType,
        lfe_stream: Option<usize>,
    ) -> Result<Self, &'static str> {
        if streams + coupled > channels || ![8000, 12000, 16000, 24000, 48000].contains(&sampling_rate) {
            return Err(BAD_ARG);
        }
        let head = MultistreamEncoderHeader {
            layout: MultistreamLayout::new(channels, streams, coupled, mapping)?,
            bitrate_bps: OpusEncoder::BITRATE_AUTO,
            application,
            sampling_rate,
            variable_duration: FrameDuration::Argument,
            mixing: None,
            demixing: None,
            mapping_type,
            lfe_stream,
            preemph_mem: [0.0; 8],
            window_mem: [0.0; 8 * OVERLAP],
            arch: crate::Arch::select(),
        };
        // libopus validate_encoder_layout.
        if (0..streams).any(|s| head.stream_channels(s).is_none()) {
            return Err(BAD_ARG);
        }
        Ok(head)
    }

    /// A new encoder for stream `s`, an LFE stream in a surround layout's
    /// LFE place (libopus `opus_encoder_init` of each stream).
    pub fn stream_encoder(&self, s: usize) -> Result<OpusEncoder, &'static str> {
        let mut enc = OpusEncoder::new(self.sampling_rate, self.layout.stream_channels(s), self.application)
            .map_err(|_| BAD_ARG)?;
        enc.set_lfe(Some(s) == self.lfe_stream);
        Ok(enc)
    }

    /// [`Self::stream_encoder`] built in `slot` (`OpusEncoder::init`).
    pub fn init_stream_encoder<'a>(
        &self,
        s: usize,
        slot: &'a mut core::mem::MaybeUninit<OpusEncoder>,
    ) -> Result<&'a mut OpusEncoder, &'static str> {
        let enc = OpusEncoder::init(slot, self.sampling_rate, self.layout.stream_channels(s), self.application)
            .map_err(|_| BAD_ARG)?;
        enc.set_lfe(Some(s) == self.lfe_stream);
        Ok(enc)
    }

    /// The input channels.
    pub fn channels(&self) -> usize {
        self.layout.channels()
    }

    /// The streams of the packets.
    pub fn streams(&self) -> usize {
        self.layout.streams()
    }

    /// The coupled streams of the packets.
    pub fn coupled_streams(&self) -> usize {
        self.layout.coupled_streams()
    }

    /// The decoded channel of each input channel, for the decoder and the
    /// Ogg Opus header.
    pub fn mapping(&self) -> &[u8] {
        self.layout.mapping()
    }

    /// The application the multistream encoder was created with.
    pub fn application(&self) -> Application {
        self.application
    }

    /// libopus `frame_size_select` of a call of `frame_size` samples: the
    /// samples per channel it codes, `None` when that is no valid frame.
    pub fn select_frame_size(&self, frame_size: usize) -> Option<usize> {
        let fs = self.sampling_rate;
        frame_size_select(fs, frame_size, self.variable_duration)
            .filter(|&n| self.application != Application::RestrictedSilk || n >= fs as usize / 100)
    }

    /// The input channels of stream `s` (libopus `get_left_channel` and
    /// `get_right_channel`, or `get_mono_channel`).
    fn stream_channels(&self, s: usize) -> Option<(usize, Option<usize>)> {
        let find = |m: usize| self.layout.mapping().iter().position(|&x| x as usize == m);
        if s < self.layout.coupled_streams() {
            Some((find(2 * s)?, Some(find(2 * s + 1)?)))
        } else {
            Some((find(s + self.layout.coupled_streams())?, None))
        }
    }

    /// Sets the bitrate of the whole packet (libopus `OPUS_SET_BITRATE`):
    /// [`OpusEncoder::BITRATE_AUTO`], [`OpusEncoder::BITRATE_MAX`], or a
    /// rate clamped to 500 to 750000 b/s per channel; others are
    /// `error::BAD_ARG`.
    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<(), &'static str> {
        let ch = self.layout.channels() as i32;
        self.bitrate_bps = match bitrate {
            OpusEncoder::BITRATE_AUTO | OpusEncoder::BITRATE_MAX => bitrate,
            ..=0 => return Err(BAD_ARG),
            b => b.clamp(500 * ch, 750_000 * ch),
        };
        Ok(())
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
        Self::with_header(MultistreamEncoderHeader::new(sampling_rate, channels, streams, coupled, mapping, application)?)
    }

    /// libopus `opus_multistream_surround_encoder_create`, as
    /// [`MultistreamEncoderHeader::new_surround`].
    pub fn new_surround(
        sampling_rate: i32,
        channels: usize,
        mapping_family: i32,
        application: Application,
    ) -> Result<Self, &'static str> {
        Self::with_header(MultistreamEncoderHeader::new_surround(sampling_rate, channels, mapping_family, application)?)
    }

    /// The encoder of `head` with new stream encoders.
    pub(crate) fn with_header(head: MultistreamEncoderHeader) -> Result<Self, &'static str> {
        let encoders = (0..head.streams()).map(|s| head.stream_encoder(s)).collect::<Result<_, _>>()?;
        Ok(MultistreamEncoder { head, encoders })
    }

    /// The encoder as a [`MultistreamEncoderView`].
    pub fn view(&mut self) -> MultistreamEncoderView<'_> {
        MultistreamEncoderView { head: &mut self.head, encoders: &mut self.encoders }
    }

    /// The streams of the packets.
    pub fn streams(&self) -> usize {
        self.head.streams()
    }

    /// The coupled streams of the packets.
    pub fn coupled_streams(&self) -> usize {
        self.head.coupled_streams()
    }

    /// The decoded channel of each input channel, for the decoder and the
    /// Ogg Opus header.
    pub fn mapping(&self) -> &[u8] {
        self.head.mapping()
    }

    /// Encodes a frame of interleaved float input of `frame_size` samples
    /// per channel (libopus `opus_multistream_encode_float`) and returns the
    /// packet length.
    pub fn encode(&mut self, pcm: &[f32], frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        self.view().encode(pcm, frame_size, out)
    }

    /// [`Self::encode`] for 16-bit input (libopus `opus_multistream_encode`).
    pub fn encode_i16(&mut self, pcm: &[i16], frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        self.view().encode_i16(pcm, frame_size, out)
    }

    /// [`Self::encode`] for 24-bit input in 32-bit integers (libopus
    /// `opus_multistream_encode24`).
    pub fn encode24(&mut self, pcm: &[i32], frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        self.view().encode24(pcm, frame_size, out)
    }

    /// Sets the bitrate of the whole packet, as
    /// [`MultistreamEncoderHeader::set_bitrate`].
    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<(), &'static str> {
        self.head.set_bitrate(bitrate)
    }

    /// The sum of the streams' bitrates (libopus `OPUS_GET_BITRATE`).
    pub fn bitrate(&self) -> i32 {
        self.encoders.iter().map(OpusEncoder::bitrate).sum()
    }

    /// The frame duration to code (libopus
    /// `OPUS_SET_EXPERT_FRAME_DURATION` on the multistream encoder).
    pub fn set_expert_frame_duration(&mut self, duration: FrameDuration) {
        self.head.set_expert_frame_duration(duration);
    }

    /// The frame duration setting.
    pub fn expert_frame_duration(&self) -> FrameDuration {
        self.head.expert_frame_duration()
    }

    /// Restarts every stream and the surround analysis (libopus
    /// `OPUS_RESET_STATE`).
    pub fn reset(&mut self) {
        self.view().reset();
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
        self.head.application
    }
}

impl<'a> MultistreamEncoderView<'a> {
    /// The encoder of `head` with `encoders`, one per stream with the
    /// stream's channels; `error::BAD_ARG` otherwise.
    pub fn new(head: &'a mut MultistreamEncoderHeader, encoders: &'a mut [OpusEncoder]) -> Result<Self, &'static str> {
        if encoders.len() != head.streams()
            || encoders.iter().enumerate().any(|(s, e)| e.channels() != head.layout.stream_channels(s))
        {
            return Err(BAD_ARG);
        }
        Ok(MultistreamEncoderView { head, encoders })
    }

    /// [`MultistreamEncoder::encode`].
    pub fn encode(&mut self, pcm: &[f32], frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        self.encode_native(MsPcm::F32(pcm), frame_size, out)
    }

    /// [`MultistreamEncoder::encode_i16`].
    pub fn encode_i16(&mut self, pcm: &[i16], frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        self.encode_native(MsPcm::I16(pcm), frame_size, out)
    }

    /// [`MultistreamEncoder::encode24`].
    pub fn encode24(&mut self, pcm: &[i32], frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        self.encode_native(MsPcm::I24(pcm), frame_size, out)
    }

    /// [`MultistreamEncoder::reset`].
    pub fn reset(&mut self) {
        self.head.preemph_mem = [0.0; 8];
        self.head.window_mem = [0.0; 8 * OVERLAP];
        self.encoders.iter_mut().for_each(OpusEncoder::reset);
    }

    /// libopus `rate_allocation`: `surround_rate_allocation`, or
    /// `ambisonics_rate_allocation`, at least 500 b/s per stream.
    fn rate_allocation(&self, frame_size: usize) -> Vec<i32> {
        let fs = self.head.sampling_rate;
        let streams = self.encoders.len() as i32;
        let nb_coupled = self.head.layout.coupled_streams() as i32;
        let rates: Vec<i32> = if self.head.mapping_type == MappingType::Ambisonics {
            let total = match self.head.bitrate_bps {
                OpusEncoder::BITRATE_AUTO => {
                    (nb_coupled + streams) * (fs + 60 * fs / frame_size as i32) + streams * 15000
                }
                OpusEncoder::BITRATE_MAX => (streams + nb_coupled) * 750_000,
                b => b,
            };
            vec![total / streams; streams as usize]
        } else {
            let nb_lfe = i32::from(self.head.lfe_stream.is_some());
            let nb_uncoupled = streams - nb_coupled - nb_lfe;
            let nb_normal = 2 * nb_coupled + nb_uncoupled;
            let channel_offset = 40 * 50.max(fs / frame_size as i32);
            let bitrate = match self.head.bitrate_bps {
                OpusEncoder::BITRATE_AUTO => nb_normal * (channel_offset + fs + 10000) + 8000 * nb_lfe,
                OpusEncoder::BITRATE_MAX => nb_normal * 750_000 + nb_lfe * 128_000,
                b => b,
            };
            let lfe_offset = (bitrate / 20).min(3000) + 15 * 50.max(fs / frame_size as i32);
            let stream_offset = ((bitrate - channel_offset * nb_normal - lfe_offset * nb_lfe) / nb_normal / 2).clamp(0, 20000);
            let (coupled_ratio, lfe_ratio) = (512, 32);
            let total = (nb_uncoupled << 8) + coupled_ratio * nb_coupled + nb_lfe * lfe_ratio;
            let channel_rate = (256
                * (bitrate - lfe_offset * nb_lfe - stream_offset * (nb_coupled + nb_uncoupled) - channel_offset * nb_normal)
                    as i64
                / total as i64) as i32;
            (0..streams)
                .map(|s| {
                    if s < nb_coupled {
                        2 * channel_offset + 0.max(stream_offset + (channel_rate.wrapping_mul(coupled_ratio) >> 8))
                    } else if Some(s as usize) != self.head.lfe_stream {
                        channel_offset + 0.max(stream_offset + channel_rate)
                    } else {
                        0.max(lfe_offset + (channel_rate.wrapping_mul(lfe_ratio) >> 8))
                    }
                })
                .collect()
        };
        rates.into_iter().map(|r| r.max(500)).collect()
    }

    /// libopus `opus_multistream_encode_native`.
    fn encode_native(&mut self, pcm: MsPcm<'_>, analysis_frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        let fs = self.head.sampling_rate;
        let vbr = self.encoders[0].vbr();
        let frame_size = self.head.select_frame_size(analysis_frame_size).ok_or(BAD_ARG)?;
        let streams = self.encoders.len();
        let len = match pcm {
            MsPcm::F32(x) => x.len(),
            MsPcm::I16(x) => x.len(),
            MsPcm::I24(x) => x.len(),
        };
        if len < analysis_frame_size * self.head.layout.channels() {
            return Err(BAD_ARG);
        }
        let hundred_ms = fs / frame_size as i32 == 10;
        let smallest = streams * 2 - 1 + if hundred_ms { streams } else { 0 };
        if out.len() < smallest {
            return Err(BUFFER_TOO_SMALL);
        }
        // The masking between surround channels (libopus surround_analysis),
        // which SILK-only streams go without.
        let masking = self.head.mapping_type == MappingType::Surround && self.head.application != Application::RestrictedSilk;
        let band_smr = if masking {
            self.surround_analysis(pcm, frame_size)
        } else {
            Vec::new()
        };
        let rates = self.rate_allocation(frame_size);
        let rate_sum: i32 = rates.iter().sum();
        let mut max_data_bytes = out.len() as i32;
        if !vbr {
            if self.head.bitrate_bps == OpusEncoder::BITRATE_AUTO {
                max_data_bytes = max_data_bytes.min((bitrate_to_bits(rate_sum, fs, frame_size) + 4) / 8);
            } else if self.head.bitrate_bps != OpusEncoder::BITRATE_MAX {
                let bytes = (bitrate_to_bits(self.head.bitrate_bps, fs, frame_size) + 4) / 8;
                max_data_bytes = max_data_bytes.min((smallest as i32).max(bytes));
            }
        }
        let ch = self.head.layout.channels() as i32;
        for (s, (enc, &rate)) in self.encoders.iter_mut().zip(&rates).enumerate() {
            let _ = enc.set_bitrate(rate);
            match self.head.mapping_type {
                MappingType::Surround => {
                    let mut equiv_rate = self.head.bitrate_bps;
                    if frame_size * 50 < fs as usize {
                        equiv_rate -= 60 * (fs / frame_size as i32 - 50) * ch;
                    }
                    enc.set_bandwidth(if equiv_rate > 10000 * ch {
                        Bandwidth::Fullband
                    } else if equiv_rate > 7000 * ch {
                        Bandwidth::Superwideband
                    } else if equiv_rate > 5000 * ch {
                        Bandwidth::Wideband
                    } else {
                        Bandwidth::Narrowband
                    });
                    if s < self.head.layout.coupled_streams() {
                        // Stereo CELT keeps the spatial image.
                        enc.set_force_mode(Some(OpusMode::CeltOnly));
                        let _ = enc.set_force_channels(Some(2));
                    }
                }
                MappingType::Ambisonics => enc.set_force_mode(Some(OpusMode::CeltOnly)),
                MappingType::None => {}
            }
        }
        // libopus's bandLogE, which a mono stream fills in half.
        let mut band_log_e = [0.0f32; 42];
        let n = analysis_frame_size;
        let mut f32_buf = vec![0.0f32; 2 * n];
        // The stream's input channels for the signal analysis when the
        // coded ones are mixed.
        let mut raw_f32 = vec![0.0f32; 2 * n];
        let mut i16_buf = vec![0i16; 2 * n];
        let mut i24_buf = vec![0i32; 2 * n];
        let mut tmp = vec![0u8; MS_FRAME_TMP];
        let mut tot_size = 0usize;
        for s in 0..streams {
            let (left, right) = self.head.stream_channels(s).ok_or(INTERNAL_ERROR)?;
            let chans: &[usize] = &match right {
                Some(r) => [left, r],
                None => [left, left],
            }[..1 + right.is_some() as usize];
            let c = chans.len();
            // The stream's channels of the call's samples, the analysis's
            // too (libopus copy_channel_in, and downmix with c1 and c2).
            for (k, &ch) in chans.iter().enumerate() {
                for i in 0..n {
                    let at = i * self.head.layout.channels() + ch;
                    match (pcm, self.head.mixing.is_some()) {
                        (MsPcm::F32(x), false) => f32_buf[i * c + k] = x[at],
                        (MsPcm::F32(x), true) => raw_f32[i * c + k] = x[at],
                        (MsPcm::I16(x), _) => i16_buf[i * c + k] = x[at],
                        (MsPcm::I24(x), false) => {
                            i24_buf[i * c + k] = x[at];
                            f32_buf[i * c + k] = (1.0 / 32768.0 / 256.0) * x[at] as f32;
                        }
                        // opus_projection_encode24 analyses with
                        // downmix_int, which reads the 32-bit samples as
                        // 16-bit ones (little endian).
                        (MsPcm::I24(x), true) => {
                            let k16 = i * self.head.layout.channels() + ch;
                            let w = x[k16 / 2];
                            i16_buf[i * c + k] = if k16 % 2 == 0 { w as i16 } else { (w >> 16) as i16 };
                        }
                    }
                }
                if let Some(m) = &self.head.mixing {
                    let out = &mut f32_buf[k..];
                    match pcm {
                        MsPcm::F32(x) => m.in_float(x, self.head.layout.channels(), out, ch, c, n),
                        MsPcm::I16(x) => m.in_short(x, self.head.layout.channels(), out, ch, c, n),
                        MsPcm::I24(x) => m.in_int24(x, self.head.layout.channels(), out, ch, c, n),
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
            if masking {
                band_log_e[..21].copy_from_slice(&band_smr[21 * left..21 * left + 21]);
                if let Some(r) = right {
                    band_log_e[21..].copy_from_slice(&band_smr[21 * r..21 * r + 21]);
                }
            }
            let enc = &mut self.encoders[s];
            if masking {
                enc.set_energy_mask(band_log_e);
            }
            if !vbr && last {
                let _ = enc.set_bitrate(bits_to_bitrate(curr_max * 8, fs, frame_size));
            }
            if curr_max <= 0 {
                return Err(BAD_ARG);
            }
            let tmp = &mut tmp[..curr_max as usize];
            let coded = PcmInput::F32(&f32_buf[..n * c]);
            let len = match (pcm, self.head.mixing.is_some()) {
                (MsPcm::F32(_), false) => enc.encode_selected(coded, n, frame_size, tmp, 24, None),
                (MsPcm::I16(_), false) => enc.encode_selected(PcmInput::I16(&i16_buf[..n * c]), n, frame_size, tmp, 16, None),
                (MsPcm::I24(_), false) => {
                    enc.encode_selected(coded, n, frame_size, tmp, 24, Some(AnalysisPcm::I24(&i24_buf[..n * c])))
                }
                (MsPcm::F32(_), true) => {
                    enc.encode_selected(coded, n, frame_size, tmp, 24, Some(AnalysisPcm::F32(&raw_f32[..n * c])))
                }
                (MsPcm::I16(_), true) => {
                    enc.encode_selected(coded, n, frame_size, tmp, 16, Some(AnalysisPcm::I16(&i16_buf[..n * c])))
                }
                (MsPcm::I24(_), true) => {
                    enc.encode_selected(coded, n, frame_size, tmp, 24, Some(AnalysisPcm::I16(&i16_buf[..n * c])))
                }
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

    /// libopus `surround_analysis`: per input channel and band, how far
    /// the band's energy is above the masking of the other channels on its
    /// side, 21 values per channel.
    fn surround_analysis(&mut self, pcm: MsPcm<'_>, len: usize) -> Vec<f32> {
        let mode = crate::modes::default_mode();
        let channels = self.head.layout.channels();
        let overlap = mode.overlap;
        debug_assert_eq!(overlap, OVERLAP);
        let upsample = 48000 / self.head.sampling_rate as usize;
        let frame_size = len * upsample;
        let lm = (0..mode.max_lm)
            .find(|&lm| mode.short_mdct_size << lm == frame_size)
            .unwrap_or(mode.max_lm);
        let freq_size = mode.short_mdct_size << lm;
        let pos = channel_pos(channels);
        let mut mask_log_e = [[-28.0f32; 21]; 3];
        let mut band_log_e = vec![0.0f32; 21 * channels];
        let mut input = vec![0.0f32; frame_size + overlap];
        let mut freq = vec![0.0f32; freq_size];
        let coef = mode.preemph[0];
        for c in 0..channels {
            input[..overlap].copy_from_slice(&self.head.window_mem[c * overlap..(c + 1) * overlap]);
            // copy_channel_in, then celt_preemphasis without clipping.
            let mut m = self.head.preemph_mem[c];
            for i in 0..frame_size {
                let x = if i % upsample == 0 {
                    let at = (i / upsample) * channels + c;
                    32768.0
                        * match pcm {
                            MsPcm::F32(p) => p[at],
                            MsPcm::I16(p) => p[at] as f32 * (1.0 / 32768.0),
                            MsPcm::I24(p) => (1.0 / 32768.0 / 256.0) * p[at] as f32,
                        }
                } else {
                    0.0
                };
                input[overlap + i] = x - m;
                m = coef * x;
            }
            self.head.preemph_mem[c] = m;
            // celt_inner_prod, which x86-64 libopus runs as its SSE kernel.
            let sum = crate::pitch::inner_prod(&input, &input, input.len(), self.head.arch);
            if !(sum < 1e18) || sum.is_nan() {
                input.fill(0.0);
                self.head.preemph_mem[c] = 0.0;
            }
            let mut band_e = [0.0f32; 21];
            for frame in 0..frame_size / freq_size {
                let mut tmp_e = [0.0f32; 21];
                mode.mdct.forward(&input[freq_size * frame..], &mut freq, mode.window, overlap, mode.max_lm - lm, 1);
                if upsample != 1 {
                    let bound = freq_size / upsample;
                    for v in &mut freq[..bound] {
                        *v *= upsample as f32;
                    }
                    freq[bound..].fill(0.0);
                }
                crate::bands::compute_band_energies(mode, &freq, &mut tmp_e, 21, 1, lm, self.head.arch);
                for i in 0..21 {
                    band_e[i] = band_e[i].max(tmp_e[i]);
                }
            }
            let b = &mut band_log_e[21 * c..21 * c + 21];
            crate::bands::amp2log2(mode, 21, 21, &band_e, b, 1);
            // A spreading function of -6 dB per band up and -12 dB down.
            for i in 1..21 {
                b[i] = b[i].max(b[i - 1] - 1.0);
            }
            for i in (0..20).rev() {
                b[i] = b[i].max(b[i + 1] - 2.0);
            }
            match pos[c] {
                1 => (0..21).for_each(|i| mask_log_e[0][i] = log_sum(mask_log_e[0][i], b[i])),
                3 => (0..21).for_each(|i| mask_log_e[2][i] = log_sum(mask_log_e[2][i], b[i])),
                2 => (0..21).for_each(|i| {
                    mask_log_e[0][i] = log_sum(mask_log_e[0][i], b[i] - 0.5);
                    mask_log_e[2][i] = log_sum(mask_log_e[2][i], b[i] - 0.5);
                }),
                _ => {}
            }
            self.head.window_mem[c * overlap..(c + 1) * overlap].copy_from_slice(&input[frame_size..frame_size + overlap]);
        }
        for i in 0..21 {
            mask_log_e[1][i] = mask_log_e[0][i].min(mask_log_e[2][i]);
        }
        let channel_offset = 0.5 * crate::mathops::celt_log2(2.0 / (channels - 1) as f32);
        for row in &mut mask_log_e {
            for v in row.iter_mut() {
                *v += channel_offset;
            }
        }
        for c in 0..channels {
            for i in 0..21 {
                band_log_e[21 * c + i] = match pos[c] {
                    0 => 0.0,
                    p => band_log_e[21 * c + i] - mask_log_e[p - 1][i],
                };
            }
        }
        band_log_e
    }
}

/// libopus `channel_pos`: each channel's place in the mix, 0 not mixed, 1
/// left, 2 centre, 3 right.
fn channel_pos(channels: usize) -> [usize; 8] {
    match channels {
        4 => [1, 3, 1, 3, 0, 0, 0, 0],
        3 | 5 | 6 => [1, 2, 3, 1, 3, 0, 0, 0],
        7 => [1, 2, 3, 1, 3, 2, 0, 0],
        8 => [1, 2, 3, 1, 3, 1, 3, 0],
        _ => [0; 8],
    }
}

/// libopus `logSum`: about log2(2^a + 2^b).
fn log_sum(a: f32, b: f32) -> f32 {
    const DIFF_TABLE: [f32; 17] = [
        0.5, 0.2924813, 0.1609640, 0.0849625, 0.0437314, 0.0221971, 0.0111839, 0.0056136, 0.0028123, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let (max, diff) = if a > b { (a, a - b) } else { (b, b - a) };
    if !(diff < 8.0) {
        return max;
    }
    let low = (2.0 * diff).floor() as usize;
    let frac = 2.0 * diff - low as f32;
    max + DIFF_TABLE[low] + frac * (DIFF_TABLE[low + 1] - DIFF_TABLE[low])
}
