//! Multistream Opus decode (channel mapping family 1 — RFC 7845 §5.1.1).
//! Decode only. Every elementary stream is a plain 1- or 2-channel
//! `OpusDecoder`; this module just demuxes and re-routes, mirroring
//! libopus's `opus_multistream_decoder.c` (a wrapper, not a codec change).

use crate::{parse_frame_size, OpusDecoder};

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

    fn decoded_channel_count(&self) -> usize {
        self.coupled_count as usize * 2 + (self.stream_count - self.coupled_count) as usize
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
            let fc = *data.get(cursor).ok_or("truncated code-3 frame-count byte")?;
            scratch.push(fc);
            cursor += 1;
            let vbr = fc & 0x80 != 0;
            let has_padding = fc & 0x40 != 0;
            let m = (fc & 0x3F) as usize;
            if m == 0 {
                return Err("code-3 packet declares zero frames");
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

    crate::frame_payloads(scratch)?;
    Ok((scratch.as_slice(), cursor))
}

#[inline]
fn push_frame(data: &[u8], cursor: usize, n: usize, out: &mut Vec<u8>) -> Result<usize, &'static str> {
    let end = cursor.checked_add(n).ok_or("frame length overflow")?;
    if end > data.len() {
        return Err("truncated frame in self-delimited packet");
    }
    out.extend_from_slice(&data[cursor..end]);
    Ok(end)
}

/// One [`OpusDecoder`] per elementary stream, fanned out per
/// [`ChannelMappingTable`]. CELT/SILK still only ever see 1–2 channels.
pub struct MultistreamDecoder {
    decoders: Vec<OpusDecoder>,
    table: ChannelMappingTable,
    stream_scratch: Vec<f32>,
    packet_scratch: Vec<u8>,
}

impl MultistreamDecoder {
    pub fn new(sampling_rate: i32, table: ChannelMappingTable) -> Result<Self, &'static str> {
        if table.mapping.len() != table.channels as usize {
            return Err("channel mapping table length does not match declared channel count");
        }
        if table.coupled_count > table.stream_count {
            return Err("coupled_count exceeds stream_count");
        }

        let mut decoders = Vec::with_capacity(table.stream_count as usize);
        for i in 0..table.stream_count {
            let ch = if i < table.coupled_count { 2 } else { 1 };
            decoders.push(OpusDecoder::new(sampling_rate, ch)?);
        }

        Ok(Self {
            decoders,
            table,
            stream_scratch: Vec::new(),
            packet_scratch: Vec::with_capacity(1500),
        })
    }

    /// `frame_size` is per-channel capacity; `out` must hold at least
    /// `frame_size * table.channels`. Returns frames decoded (per channel).
    pub fn decode(
        &mut self,
        packet: &[u8],
        frame_size: usize,
        out: &mut [f32],
    ) -> Result<usize, &'static str> {
        let out_channels = self.table.channels as usize;
        if out.len() < frame_size * out_channels {
            return Err("output buffer too small for frame_size * channels");
        }

        let decoded_channels = self.table.decoded_channel_count();
        let mut decoded = vec![0.0f32; frame_size * decoded_channels];
        let mut decoded_offset_channels = 0usize;
        let mut frames_decoded = 0usize;

        let mut rest = packet;
        let num_streams = self.decoders.len();
        for (i, decoder) in self.decoders.iter_mut().enumerate() {
            let ch = if (i as u8) < self.table.coupled_count { 2 } else { 1 };
            self.stream_scratch.clear();
            self.stream_scratch.resize(frame_size * ch, 0.0);

            // RFC 6716 Appendix B: only the last sub-packet is unframed —
            // decode it directly. Every other sub-packet must first be
            // reconstructed into a normal packet (split_self_delimited
            // strips the injected Appendix B length field(s)).
            let n = if i + 1 == num_streams {
                decoder.decode(rest, frame_size, &mut self.stream_scratch)?
            } else {
                let (pkt, consumed) = split_self_delimited(rest, &mut self.packet_scratch)?;
                let n = decoder.decode(pkt, frame_size, &mut self.stream_scratch)?;
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
                        self.stream_scratch[f * ch + c];
                }
            }
            decoded_offset_channels += ch;
        }

        for f in 0..frames_decoded {
            for (out_c, &src) in self.table.mapping.iter().enumerate() {
                out[f * out_channels + out_c] = if src == SILENT_CHANNEL {
                    0.0
                } else {
                    decoded[f * decoded_channels + src as usize]
                };
            }
        }

        Ok(frames_decoded)
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
        assert_eq!(table.decoded_channel_count(), 6);
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
        assert_eq!(pkt, &[toc_byte(0, false, 2), 3, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE][..]);
    }

    #[test]
    fn split_code3_cbr_three_frames() {
        let bytes = [toc_byte(0, false, 3), 3, 2, 0x10, 0x11, 0x20, 0x21, 0x30, 0x31];
        let mut scratch = Vec::new();
        let (_, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 1 + 1 + 6);
    }

    #[test]
    fn split_code3_cbr_with_padding() {
        let fc = 0x40 | 2u8;
        let bytes = [
            toc_byte(0, false, 3), fc, 2, 3,
            0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
            0x55, 0x66,
        ];
        let mut scratch = Vec::new();
        let (_, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 12);
    }

    #[test]
    fn split_code3_vbr_three_frames() {
        let fc = 0x80 | 3u8;
        let bytes = [
            toc_byte(0, false, 3), fc, 2, 3, 1,
            0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
        ];
        let mut scratch = Vec::new();
        let (pkt, consumed) = split_self_delimited(&bytes, &mut scratch).unwrap();
        assert_eq!(consumed, 1 + 1 + 1 + 1 + 1 + 2 + 3 + 1);
        // inline lengths for frames 1-2 (2, 3) are kept; the extra
        // Appendix B length for frame 3 (1) is excised
        assert_eq!(pkt, &[toc_byte(0, false, 3), fc, 2, 3, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF][..]);
    }

    #[test]
    fn split_two_streams_chained() {
        let bytes = [
            toc_byte(0, false, 0), 2, 0xA0, 0xA1,
            toc_byte(0, false, 1), 1, 0xB0, 0xB1,
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