//! Multistream Opus decode (channel mapping family 1 — RFC 7845 §5.1.1).
//! Decode only. Every elementary stream is a plain 1- or 2-channel
//! `OpusDecoder`; this module just demuxes and re-routes, mirroring
//! libopus's `opus_multistream_decoder.c` (a wrapper, not a codec change).

use crate::decode_int::IntSample;
use crate::{OpusDecoder, frame_samples_from_toc, opus_packet_parse, parse_frame_size};

/// `channel_mapping[]` sentinel: output silence on this channel.
const SILENT_CHANNEL: u8 = 255;

/// RFC 7845 §5.1.1 channel mapping table (the bytes following OpusHead's
/// fixed 19-byte prefix, present when `mapping_family != 0`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelMappingTable {
    pub channels: u8,
    pub stream_count: u8,
    /// First `coupled_count` streams are stereo; the rest are mono.
    pub coupled_count: u8,
    /// One entry per output channel, indexing the concatenated per-stream
    /// decode buffer (coupled streams first, 2 channels each, then mono
    /// streams). `SILENT_CHANNEL` = output silence.
    pub mapping: Vec<u8>,
}

impl ChannelMappingTable {
    /// `tail` starts right after `output_gain` (OpusHead byte 19).
    /// `None` for `mapping_family == 0` or a truncated tail.
    pub fn parse(mapping_family: u8, channels: u8, tail: &[u8]) -> Option<Self> {
        if mapping_family == 0 || tail.len() < 2 + channels as usize {
            return None;
        }
        Some(Self {
            channels,
            stream_count: tail[0],
            coupled_count: tail[1],
            mapping: tail[2..2 + channels as usize].to_vec(),
        })
    }
}

/// Splits one leading self-delimited packet (RFC 6716 Appendix B) off
/// `data`, returning `(packet, bytes_consumed)`. `packet` is a normal
/// (non-self-delimited) Opus packet — decodable as-is by a standard
/// decoder — reconstructed by excising the extra length field(s)
/// Appendix B framing injects versus normal framing (code 0/1/CBR-3:
/// length is otherwise absent/implicit; code 2/VBR-3: the final frame's
/// length is otherwise implicit). `bytes_consumed` counts the original
/// self-delimited bytes, for advancing through the outer buffer.
/// `scratch` is cleared and reused for the reconstructed packet, so a
/// caller decoding many sub-streams per call (e.g. `MultistreamDecoder`)
/// need not allocate on every one.
pub fn split_self_delimited<'a>(
    data: &[u8],
    scratch: &'a mut Vec<u8>,
) -> Result<(&'a [u8], usize), &'static str> {
    let toc = *data.first().ok_or("empty multistream packet")?;
    let code = toc & 0x03;
    let mut cursor = 1usize;
    scratch.clear();
    scratch.push(toc);

    match code {
        0 => {
            let (n1, len_bytes) = parse_frame_size(&data[cursor..])?;
            cursor += len_bytes; // excised: normal code 0 has no length field
            cursor = push_frame(data, cursor, n1, scratch)?;
        }
        1 => {
            let (n1, len_bytes) = parse_frame_size(&data[cursor..])?;
            cursor += len_bytes; // excised: normal code 1 has no length field
            cursor = push_frame(data, cursor, n1, scratch)?;
            cursor = push_frame(data, cursor, n1, scratch)?;
        }
        2 => {
            let (n1, len1_bytes) = parse_frame_size(&data[cursor..])?;
            scratch.extend_from_slice(&data[cursor..cursor + len1_bytes]); // kept: normal code 2 has this one
            cursor += len1_bytes;
            let (n2, len2_bytes) = parse_frame_size(&data[cursor..])?;
            cursor += len2_bytes; // excised: Appendix B's extra length for frame 2
            cursor = push_frame(data, cursor, n1, scratch)?;
            cursor = push_frame(data, cursor, n2, scratch)?;
        }
        _ => {
            let fc = *data
                .get(cursor)
                .ok_or("truncated code-3 frame-count byte")?;
            scratch.push(fc);
            cursor += 1;
            let vbr = fc & 0x80 != 0;
            let has_padding = fc & 0x40 != 0;
            let m = (fc & 0x3F) as usize;
            if m == 0 {
                return Err("code-3 packet declares zero frames");
            }
            if frame_samples_from_toc(toc, 48000).is_none_or(|n| n * m > 5760) {
                return Err("code-3 packet longer than 120 ms");
            }

            let mut padding_bytes = 0usize;
            if has_padding {
                loop {
                    let b = *data.get(cursor).ok_or("truncated padding length chain")?;
                    scratch.push(b);
                    cursor += 1;
                    if b == 255 {
                        padding_bytes += 254;
                    } else {
                        padding_bytes += b as usize;
                        break;
                    }
                }
            }

            let mut sizes = Vec::with_capacity(m);
            if vbr {
                for _ in 0..m - 1 {
                    let (n, len_bytes) = parse_frame_size(&data[cursor..])?;
                    scratch.extend_from_slice(&data[cursor..cursor + len_bytes]); // kept: normal VBR has these
                    cursor += len_bytes;
                    sizes.push(n);
                }
                let (n_last, len_bytes) = parse_frame_size(&data[cursor..])?;
                cursor += len_bytes; // excised: Appendix B's extra length for the final frame
                sizes.push(n_last);
            } else {
                let (n, len_bytes) = parse_frame_size(&data[cursor..])?;
                cursor += len_bytes; // excised: normal CBR code 3 has no length field at all
                sizes.extend(std::iter::repeat_n(n, m));
            }

            for n in sizes {
                cursor = push_frame(data, cursor, n, scratch)?;
            }
            if data.len() < cursor + padding_bytes {
                return Err("truncated code-3 padding");
            }
            scratch.extend_from_slice(&data[cursor..cursor + padding_bytes]);
            cursor += padding_bytes;
        }
    }

    Ok((scratch.as_slice(), cursor))
}

#[inline]
fn push_frame(
    data: &[u8],
    cursor: usize,
    n: usize,
    out: &mut Vec<u8>,
) -> Result<usize, &'static str> {
    let end = cursor.checked_add(n).ok_or("frame length overflow")?;
    if end > data.len() {
        return Err("truncated frame in self-delimited packet");
    }
    out.extend_from_slice(&data[cursor..end]);
    Ok(end)
}

/// libopus `opus_multistream_packet_validate`: parses every stream of a
/// non-empty packet and returns their common duration in samples.
fn packet_validate(
    mut data: &[u8],
    nb_streams: usize,
    fs: i32,
    scratch: &mut Vec<u8>,
) -> Result<usize, &'static str> {
    let mut samples = 0;
    for s in 0..nb_streams {
        if data.is_empty() {
            return Err("truncated multistream packet");
        }
        let (packet, offset) = if s + 1 < nb_streams {
            split_self_delimited(data, scratch)?
        } else {
            (data, data.len())
        };
        let frame_samples =
            frame_samples_from_toc(packet[0], fs).ok_or("Invalid TOC for sampling rate")?;
        let n = opus_packet_parse(packet)?.len() * frame_samples;
        if s != 0 && n != samples {
            return Err("elementary streams differ in duration");
        }
        samples = n;
        data = &data[offset..];
    }
    Ok(samples)
}

/// A multistream layout (libopus `ChannelLayout`): `channels` channels
/// from `streams` streams, the first `coupled` of them stereo. Entry `c` of
/// the mapping is the decoded channel of channel `c` (coupled streams' left
/// and right, then mono streams), 255 for silence. It has a fixed size, so
/// a multistream state is this layout plus its streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultistreamLayout {
    channels: u8,
    streams: u8,
    coupled: u8,
    mapping: [u8; 255],
}

impl MultistreamLayout {
    /// libopus `opus_multistream_decoder_init`'s checks and
    /// `validate_layout`: `error::BAD_ARG` for 0 or more than 255
    /// channels, no stream, more coupled streams than streams, more than
    /// 255 decoded channels, a mapping shorter than `channels` or an entry
    /// past the decoded channels.
    pub fn new(channels: usize, streams: usize, coupled: usize, mapping: &[u8]) -> Result<Self, &'static str> {
        if !(1..=255).contains(&channels)
            || coupled > streams
            || streams < 1
            || streams > 255 - coupled
            || mapping.len() < channels
            || mapping[..channels].iter().any(|&m| m as usize >= streams + coupled && m != SILENT_CHANNEL)
        {
            return Err(crate::error::BAD_ARG);
        }
        let mut layout = MultistreamLayout { channels: channels as u8, streams: streams as u8, coupled: coupled as u8, mapping: [0; 255] };
        layout.mapping[..channels].copy_from_slice(&mapping[..channels]);
        Ok(layout)
    }

    /// The channels.
    pub fn channels(&self) -> usize {
        self.channels as usize
    }

    /// The streams.
    pub fn streams(&self) -> usize {
        self.streams as usize
    }

    /// The coupled streams.
    pub fn coupled_streams(&self) -> usize {
        self.coupled as usize
    }

    /// The decoded channel of each channel.
    pub fn mapping(&self) -> &[u8] {
        &self.mapping[..self.channels as usize]
    }

    /// The channels of stream `s`: 2 for a coupled stream, else 1.
    pub fn stream_channels(&self, s: usize) -> usize {
        if s < self.coupled as usize { 2 } else { 1 }
    }

    /// The channels the streams decode: two per coupled stream, one per
    /// mono stream.
    fn decoded_channels(&self) -> usize {
        (self.streams + self.coupled) as usize
    }
}

/// One [`OpusDecoder`] per elementary stream, fanned out per
/// [`ChannelMappingTable`]. CELT/SILK still only ever see 1–2 channels.
pub struct MultistreamDecoder {
    layout: MultistreamLayout,
    decoders: Vec<OpusDecoder>,
}

/// A multistream decoder over a layout and its stream decoders held
/// elsewhere, such as in one block of memory: what [`MultistreamDecoder`]
/// decodes with.
pub struct MultistreamDecoderView<'a> {
    layout: &'a MultistreamLayout,
    decoders: &'a mut [OpusDecoder],
}

impl MultistreamDecoder {
    pub fn new(sampling_rate: i32, table: ChannelMappingTable) -> Result<Self, &'static str> {
        if table.mapping.len() != table.channels as usize {
            return Err("channel mapping table length does not match declared channel count");
        }
        let layout = MultistreamLayout::new(
            table.channels as usize,
            table.stream_count as usize,
            table.coupled_count as usize,
            &table.mapping,
        )?;
        let decoders = (0..layout.streams())
            .map(|s| OpusDecoder::new(sampling_rate, layout.stream_channels(s)))
            .collect::<Result<_, _>>()?;
        Ok(Self { layout, decoders })
    }

    /// The decoder as a [`MultistreamDecoderView`].
    pub fn view(&mut self) -> MultistreamDecoderView<'_> {
        MultistreamDecoderView { layout: &self.layout, decoders: &mut self.decoders }
    }

    /// `frame_size` is per-channel capacity; `out` must hold at least
    /// `frame_size * table.channels`. Returns frames decoded (per channel).
    /// An empty packet conceals `frame_size` samples on every stream.
    pub fn decode(&mut self, packet: &[u8], frame_size: usize, out: &mut [f32]) -> Result<usize, &'static str> {
        self.view().decode(packet, frame_size, out)
    }

    /// Rebuilds a lost packet from the in-band FEC of the packet after it,
    /// as [`OpusDecoder::decode_fec`] does for each stream (libopus
    /// `opus_multistream_decode_float(..., decode_fec = 1)`).
    pub fn decode_fec(&mut self, packet: &[u8], frame_size: usize, out: &mut [f32]) -> Result<usize, &'static str> {
        self.view().decode_fec(packet, frame_size, out)
    }

    /// The XOR of every stream decoder's final range (libopus
    /// `OPUS_GET_FINAL_RANGE` on a multistream decoder).
    pub fn final_range(&self) -> u32 {
        self.decoders.iter().fold(0, |r, d| r ^ d.final_range())
    }

    /// Decodes into interleaved 16-bit PCM, soft clipped per stream
    /// (libopus `opus_multistream_decode` of the float build).
    pub fn decode_i16(&mut self, packet: &[u8], frame_size: usize, out: &mut [i16]) -> Result<usize, &'static str> {
        self.view().decode_i16(packet, frame_size, out)
    }

    /// [`Self::decode_i16`] with in-band FEC.
    pub fn decode_fec_i16(&mut self, packet: &[u8], frame_size: usize, out: &mut [i16]) -> Result<usize, &'static str> {
        self.view().decode_fec_i16(packet, frame_size, out)
    }

    /// Decodes into interleaved 24-bit PCM in 32-bit integers (libopus
    /// `opus_multistream_decode24`).
    pub fn decode24(&mut self, packet: &[u8], frame_size: usize, out: &mut [i32]) -> Result<usize, &'static str> {
        self.view().decode24(packet, frame_size, out)
    }

    /// [`Self::decode24`] with in-band FEC.
    pub fn decode_fec24(&mut self, packet: &[u8], frame_size: usize, out: &mut [i32]) -> Result<usize, &'static str> {
        self.view().decode_fec24(packet, frame_size, out)
    }

    /// Restarts every stream (libopus `OPUS_RESET_STATE`).
    pub fn reset(&mut self) {
        self.decoders.iter_mut().for_each(OpusDecoder::reset);
    }

    /// Sets the output gain of every stream (libopus `OPUS_SET_GAIN`).
    pub fn set_gain(&mut self, gain: i32) -> Result<(), &'static str> {
        self.decoders.iter_mut().try_for_each(|d| d.set_gain(gain))
    }

    /// Sets the complexity of every stream (libopus
    /// `OPUS_SET_COMPLEXITY`).
    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), &'static str> {
        self.decoders.iter_mut().try_for_each(|d| d.set_complexity(complexity))
    }

    /// Sets phase inversion of every stream (libopus
    /// `OPUS_SET_PHASE_INVERSION_DISABLED`).
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        self.decoders
            .iter_mut()
            .for_each(|d| d.set_phase_inversion_disabled(disabled));
    }

    /// The gain. This and the queries below report the first stream's
    /// value, as libopus's multistream queries do.
    pub fn gain(&self) -> i32 {
        self.decoders[0].gain()
    }

    /// The complexity (libopus `OPUS_GET_COMPLEXITY`).
    pub fn complexity(&self) -> i32 {
        self.decoders[0].complexity()
    }

    /// Whether phase inversion is disabled.
    pub fn phase_inversion_disabled(&self) -> bool {
        self.decoders[0].phase_inversion_disabled()
    }

    /// The bandwidth of the last packet.
    pub fn bandwidth(&self) -> Option<crate::Bandwidth> {
        self.decoders[0].bandwidth()
    }

    /// The sampling rate.
    pub fn sample_rate(&self) -> i32 {
        self.decoders[0].sample_rate()
    }

    /// The duration of the last packet.
    pub fn last_packet_duration(&self) -> usize {
        self.decoders[0].last_packet_duration()
    }

    /// The decoder of stream `stream` (libopus
    /// `OPUS_MULTISTREAM_GET_DECODER_STATE`), `error::BAD_ARG` past the
    /// last stream.
    pub fn decoder_state(&mut self, stream: usize) -> Result<&mut OpusDecoder, &'static str> {
        self.decoders.get_mut(stream).ok_or(crate::error::BAD_ARG)
    }
}

impl<'a> MultistreamDecoderView<'a> {
    /// The decoder of `layout` with `decoders`, one per stream with the
    /// stream's channels and one sampling rate; `error::BAD_ARG` otherwise.
    pub fn new(layout: &'a MultistreamLayout, decoders: &'a mut [OpusDecoder]) -> Result<Self, &'static str> {
        let fs = decoders.first().map(OpusDecoder::sample_rate);
        if decoders.len() != layout.streams()
            || decoders
                .iter()
                .enumerate()
                .any(|(s, d)| d.channels() != layout.stream_channels(s) || Some(d.sample_rate()) != fs)
        {
            return Err(crate::error::BAD_ARG);
        }
        Ok(MultistreamDecoderView { layout, decoders })
    }

    /// [`MultistreamDecoder::decode`].
    pub fn decode(&mut self, packet: &[u8], frame_size: usize, out: &mut [f32]) -> Result<usize, &'static str> {
        self.decode_native(packet, frame_size, out, false, false)
    }

    /// [`MultistreamDecoder::decode_fec`].
    pub fn decode_fec(&mut self, packet: &[u8], frame_size: usize, out: &mut [f32]) -> Result<usize, &'static str> {
        self.decode_native(packet, frame_size, out, true, false)
    }

    /// [`MultistreamDecoder::final_range`].
    pub fn final_range(&self) -> u32 {
        self.decoders.iter().fold(0, |r, d| r ^ d.final_range())
    }

    /// [`MultistreamDecoder::decode_i16`].
    pub fn decode_i16(&mut self, packet: &[u8], frame_size: usize, out: &mut [i16]) -> Result<usize, &'static str> {
        self.decode_int(packet, frame_size, out, false, true)
    }

    /// [`MultistreamDecoder::decode_fec_i16`].
    pub fn decode_fec_i16(&mut self, packet: &[u8], frame_size: usize, out: &mut [i16]) -> Result<usize, &'static str> {
        self.decode_int(packet, frame_size, out, true, true)
    }

    /// [`MultistreamDecoder::decode24`].
    pub fn decode24(&mut self, packet: &[u8], frame_size: usize, out: &mut [i32]) -> Result<usize, &'static str> {
        self.decode_int(packet, frame_size, out, false, false)
    }

    /// [`MultistreamDecoder::decode_fec24`].
    pub fn decode_fec24(&mut self, packet: &[u8], frame_size: usize, out: &mut [i32]) -> Result<usize, &'static str> {
        self.decode_int(packet, frame_size, out, true, false)
    }

    /// The channels the streams decode.
    pub(crate) fn decoded_channels(&self) -> usize {
        self.layout.decoded_channels()
    }

    fn decode_int<T: IntSample>(
        &mut self,
        packet: &[u8],
        frame_size: usize,
        out: &mut [T],
        decode_fec: bool,
        soft_clip: bool,
    ) -> Result<usize, &'static str> {
        let mut buf = vec![0.0f32; out.len()];
        let n = self.decode_native(packet, frame_size, &mut buf, decode_fec, soft_clip)?;
        let len = n * self.layout.channels();
        for (o, &x) in out.iter_mut().zip(&buf[..len]) {
            *o = T::from_float(x);
        }
        Ok(n)
    }

    /// libopus `opus_multistream_decode_native`, soft clipping each stream
    /// with `soft_clip`.
    fn decode_native(
        &mut self,
        packet: &[u8],
        frame_size: usize,
        out: &mut [f32],
        decode_fec: bool,
        soft_clip: bool,
    ) -> Result<usize, &'static str> {
        let out_channels = self.layout.channels();
        let (frames_decoded, decoded) = self.decode_checked(packet, frame_size, out.len(), decode_fec, soft_clip)?;
        let decoded_channels = self.layout.decoded_channels();
        for f in 0..frames_decoded {
            for (out_c, &src) in self.layout.mapping().iter().enumerate() {
                out[f * out_channels + out_c] = if src == SILENT_CHANNEL {
                    0.0
                } else {
                    decoded[f * decoded_channels + src as usize]
                };
            }
        }
        Ok(frames_decoded)
    }

    /// `opus_multistream_decode_native` up to the channel output: its
    /// checks for an output of `out_len` samples, then the streams decoded
    /// (see [`Self::decode_streams`]).
    pub(crate) fn decode_checked(
        &mut self,
        packet: &[u8],
        frame_size: usize,
        out_len: usize,
        decode_fec: bool,
        soft_clip: bool,
    ) -> Result<(usize, Vec<f32>), &'static str> {
        if frame_size == 0 {
            return Err(crate::error::BAD_ARG);
        }
        // libopus limits frame_size to 120 ms.
        let frame_size = frame_size.min(self.decoders[0].sampling_rate as usize / 25 * 3);
        let out_channels = self.layout.channels();
        if out_len < frame_size * out_channels {
            return Err("output buffer too small for frame_size * channels");
        }

        // libopus checks the whole packet before decoding any stream.
        if !packet.is_empty() {
            let nb_streams = self.decoders.len();
            if packet.len() < 2 * nb_streams - 1 {
                return Err(crate::error::INVALID_PACKET);
            }
            let samples = packet_validate(
                packet,
                nb_streams,
                self.decoders[0].sampling_rate,
                &mut Vec::new(),
            )
            .map_err(|_| crate::error::INVALID_PACKET)?;
            if samples > frame_size {
                return Err(crate::error::BUFFER_TOO_SMALL);
            }
        }

        self.decode_streams(packet, frame_size, decode_fec, soft_clip)
    }

    /// The decoding of every stream of `opus_multistream_decode_native`
    /// after its checks: the samples per channel and the decoded channels
    /// (coupled streams' left and right, then mono streams) interleaved.
    pub(crate) fn decode_streams(
        &mut self,
        packet: &[u8],
        frame_size: usize,
        decode_fec: bool,
        soft_clip: bool,
    ) -> Result<(usize, Vec<f32>), &'static str> {
        let decoded_channels = self.layout.decoded_channels();
        let mut decoded = vec![0.0f32; frame_size * decoded_channels];
        let mut decoded_offset_channels = 0usize;
        let mut frames_decoded = 0usize;

        let mut rest = packet;
        let num_streams = self.decoders.len();
        let (mut stream_scratch, mut packet_scratch) = (Vec::new(), Vec::new());
        for (i, decoder) in self.decoders.iter_mut().enumerate() {
            let ch = self.layout.stream_channels(i);
            stream_scratch.clear();
            stream_scratch.resize(frame_size * ch, 0.0);

            // RFC 6716 Appendix B: only the last sub-packet is unframed —
            // decode it directly. Every other sub-packet must first be
            // reconstructed into a normal packet (split_self_delimited
            // strips the injected Appendix B length field(s)). An empty
            // packet is a loss: every stream conceals frame_size samples.
            let n = if packet.is_empty() || i + 1 == num_streams {
                decoder.decode_native(rest, frame_size, &mut stream_scratch, decode_fec, soft_clip)?
            } else {
                let (pkt, consumed) = split_self_delimited(rest, &mut packet_scratch)?;
                let n =
                    decoder.decode_native(pkt, frame_size, &mut stream_scratch, decode_fec, soft_clip)?;
                rest = &rest[consumed..];
                n
            };
            if i == 0 {
                frames_decoded = n;
            } else if n != frames_decoded {
                return Err("elementary streams decoded to different frame counts");
            }

            for f in 0..n {
                for c in 0..ch {
                    decoded[f * decoded_channels + decoded_offset_channels + c] =
                        stream_scratch[f * ch + c];
                }
            }
            decoded_offset_channels += ch;
        }
        Ok((frames_decoded, decoded))
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn toc_byte(config: u8, stereo: bool, code: u8) -> u8 {
        (config << 3) | ((stereo as u8) << 2) | (code & 0x03)
    }

    #[test]
    fn mapping_table_parses_5_1_shape() {
        let tail = [4u8, 2, 0, 1, 4, 5, 2, 3];
        let table = ChannelMappingTable::parse(1, 6, &tail).unwrap();
        assert_eq!(table.stream_count, 4);
        assert_eq!(table.coupled_count, 2);
        assert_eq!(table.mapping, vec![0, 1, 4, 5, 2, 3]);
        let layout = MultistreamLayout::new(6, 4, 2, &table.mapping).unwrap();
        assert_eq!(layout.decoded_channels(), 6);
    }

    #[test]
    fn mapping_family_zero_has_no_table() {
        assert!(ChannelMappingTable::parse(0, 2, &[9, 9, 9, 9]).is_none());
    }

    #[test]
    fn mapping_table_rejects_truncated_tail() {
        assert!(ChannelMappingTable::parse(1, 6, &[4, 2, 0, 1]).is_none());
    }

    #[test]
    fn split_code0_single_frame() {
        let bytes = [toc_byte(0, false, 0), 3, 0xAA, 0xBB, 0xCC, 0xFF];
        let mut scratch = Vec::new();
        let (pkt, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 1 + 3);
        // reconstructed packet excises the length byte: TOC + frame data only
        assert_eq!(pkt, &[toc_byte(0, false, 0), 0xAA, 0xBB, 0xCC][..]);
    }

    #[test]
    fn split_code0_two_byte_length() {
        let mut bytes = vec![toc_byte(0, false, 0), 252, 0];
        bytes.extend(std::iter::repeat_n(0x7Fu8, 252));
        let mut scratch = Vec::new();
        let (_, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 2 + 252);
    }

    #[test]
    fn split_code1_equal_frames() {
        let bytes = [toc_byte(0, false, 1), 2, 0xAA, 0xBB, 0xCC, 0xDD, 0x11];
        let mut scratch = Vec::new();
        let (_, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 1 + 4);
    }

    #[test]
    fn split_code2_two_lengths() {
        let bytes = [toc_byte(0, false, 2), 3, 2, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE];
        let mut scratch = Vec::new();
        let (pkt, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 1 + 1 + 3 + 2);
        // inline length (3, for frame 1) is kept; the extra Appendix B
        // length (2, for frame 2) is excised
        assert_eq!(
            pkt,
            &[toc_byte(0, false, 2), 3, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE][..]
        );
    }

    #[test]
    fn split_code3_cbr_three_frames() {
        let bytes = [
            toc_byte(0, false, 3),
            3,
            2,
            0x10,
            0x11,
            0x20,
            0x21,
            0x30,
            0x31,
        ];
        let mut scratch = Vec::new();
        let (_, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 1 + 1 + 6);
    }

    #[test]
    fn split_code3_cbr_with_padding() {
        let fc = 0x40 | 2u8;
        let bytes = [
            toc_byte(0, false, 3),
            fc,
            2,
            3,
            0xAA,
            0xBB,
            0xCC,
            0xDD,
            0xEE,
            0xFF,
            0x55,
            0x66,
        ];
        let mut scratch = Vec::new();
        let (_, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 12);
    }

    #[test]
    fn split_code3_vbr_three_frames() {
        let fc = 0x80 | 3u8;
        let bytes = [
            toc_byte(0, false, 3),
            fc,
            2,
            3,
            1,
            0xAA,
            0xBB,
            0xCC,
            0xDD,
            0xEE,
            0xFF,
        ];
        let mut scratch = Vec::new();
        let (pkt, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 1 + 1 + 1 + 1 + 2 + 3 + 1);
        // inline lengths for frames 1-2 (2, 3) are kept; the extra
        // Appendix B length for frame 3 (1) is excised
        assert_eq!(
            pkt,
            &[
                toc_byte(0, false, 3),
                fc,
                2,
                3,
                0xAA,
                0xBB,
                0xCC,
                0xDD,
                0xEE,
                0xFF
            ][..]
        );
    }

    #[test]
    fn split_two_streams_chained() {
        let bytes = [
            toc_byte(0, false, 0),
            2,
            0xA0,
            0xA1,
            toc_byte(0, false, 1),
            1,
            0xB0,
            0xB1,
        ];
        let mut scratch = Vec::new();
        let a_consumed = {
            let (a, a_consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
            // reconstructed: TOC + frame data, length byte excised
            assert_eq!(a, &[toc_byte(0, false, 0), 0xA0, 0xA1][..]);
            a_consumed
        };
        assert_eq!(a_consumed, 4);
        let (_, b_consumed) = split_self_delimited(&bytes[a_consumed..], &mut scratch).unwrap();
        assert_eq!(b_consumed, 4);
    }

    #[test]
    fn split_truncated_frame_is_error() {
        let bytes = [toc_byte(0, false, 0), 4, 0xAA, 0xBB];
        let mut scratch = Vec::new();
        assert!(split_self_delimited(&bytes, &mut scratch).is_err());
    }

    #[test]
    fn split_code3_zero_frames_is_error() {
        let bytes = [toc_byte(0, false, 3), 0, 1, 0xAA];
        let mut scratch = Vec::new();
        assert!(split_self_delimited(&bytes, &mut scratch).is_err());
    }

    #[test]
    fn split_code3_over_120_ms_is_error() {
        // 48 CBR frames of 5 ms with 0 bytes each.
        let bytes = [toc_byte(29, false, 3), 48, 0];
        let mut scratch = Vec::new();
        assert!(split_self_delimited(&bytes, &mut scratch).is_err());
        assert!(split_self_delimited(&[toc_byte(29, false, 3), 24, 0], &mut scratch).is_ok());
    }

    #[test]
    fn multistream_decoder_rejects_bad_coupled_count() {
        let table = ChannelMappingTable {
            channels: 2,
            stream_count: 1,
            coupled_count: 2,
            mapping: vec![0, 1],
        };
        assert!(MultistreamDecoder::new(48000, table).is_err());
    }

    #[test]
    fn multistream_decoder_rejects_invalid_layouts() {
        let layout = |channels, stream_count, coupled_count, mapping: &[u8]| ChannelMappingTable {
            channels,
            stream_count,
            coupled_count,
            mapping: mapping.to_vec(),
        };
        for table in [
            layout(0, 1, 0, &[]),
            layout(1, 0, 0, &[0]),
            layout(1, 200, 100, &[0]),
            layout(3, 1, 1, &[0, 1, 2]),
            layout(3, 1, 1, &[0, 1, 7]),
        ] {
            assert!(MultistreamDecoder::new(48000, table).is_err());
        }
        assert!(MultistreamDecoder::new(48000, layout(3, 1, 1, &[0, 1, 255])).is_ok());
        assert!(MultistreamDecoder::new(48000, layout(1, 2, 1, &[2])).is_ok());
    }

    #[test]
    fn multistream_decoder_builds_expected_stream_shape() {
        let table = ChannelMappingTable {
            channels: 6,
            stream_count: 4,
            coupled_count: 2,
            mapping: vec![0, 1, 4, 5, 2, 3],
        };
        let ms = MultistreamDecoder::new(48000, table).unwrap();
        assert_eq!(ms.decoders.len(), 4);
    }
}
