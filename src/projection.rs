//! libopus's ambisonics projection encoder and decoder
//! (`src/opus_projection_encoder.c`, `src/opus_projection_decoder.c`):
//! a multistream codec whose coded channels are a fixed mix of the
//! ambisonic input channels (channel mapping family 3), demixed again by
//! the decoder with the matrix the encoder reports.

use crate::decode_int::IntSample;
use crate::error::{ALLOC_FAIL, BAD_ARG};
use crate::mapping_matrix::{self as mm, MappingMatrix};
use crate::multistream::{ChannelMappingTable, MultistreamDecoder};
use crate::multistream_encoder::MultistreamEncoder;
use crate::Application;

/// libopus `get_streams_from_channels` for mapping family 3: the streams,
/// the coupled streams and the ambisonic order plus one, from `(1 + n)^2 +
/// 2j` channels (n up to 14, j 0 or 1).
fn streams_from_channels(channels: usize, mapping_family: i32) -> Option<(usize, usize, usize)> {
    if mapping_family != 3 || !(1..=227).contains(&channels) {
        return None;
    }
    let order_plus_one = (1..=15).take_while(|n| n * n <= channels).last()?;
    let nondiegetic = channels - order_plus_one * order_plus_one;
    if nondiegetic != 0 && nondiegetic != 2 {
        return None;
    }
    Some((channels.div_ceil(2), channels / 2, order_plus_one))
}

/// libopus `OpusProjectionEncoder`: a [`MultistreamEncoder`] of the mixed
/// channels and the matching demixing matrix.
pub struct ProjectionEncoder {
    ms: MultistreamEncoder,
    demixing: MappingMatrix<'static>,
    streams: usize,
    coupled: usize,
    channels: usize,
}

impl ProjectionEncoder {
    /// libopus `opus_projection_ambisonics_encoder_create`: `channels`
    /// ambisonic channels (4, 9, 16, 25 or 36, each plus 2 non-diegetic
    /// ones) in mapping family 3. As in libopus, a channel count or family
    /// without a matrix is `error::ALLOC_FAIL`.
    pub fn new(
        sampling_rate: i32,
        channels: usize,
        mapping_family: i32,
        application: Application,
    ) -> Result<Self, &'static str> {
        let (streams, coupled, order_plus_one) =
            streams_from_channels(channels, mapping_family).ok_or(ALLOC_FAIL)?;
        let (mixing, demixing) = match order_plus_one {
            2 => (mm::FOA_MIXING, mm::FOA_DEMIXING),
            3 => (mm::SOA_MIXING, mm::SOA_DEMIXING),
            4 => (mm::TOA_MIXING, mm::TOA_DEMIXING),
            5 => (mm::FOURTHOA_MIXING, mm::FOURTHOA_DEMIXING),
            6 => (mm::FIFTHOA_MIXING, mm::FIFTHOA_DEMIXING),
            _ => return Err(ALLOC_FAIL),
        };
        if streams + coupled > mixing.rows
            || channels > mixing.cols
            || channels > demixing.rows
            || streams + coupled > demixing.cols
        {
            return Err(BAD_ARG);
        }
        let mapping: Vec<u8> = (0..channels as u8).collect();
        let mut ms = MultistreamEncoder::new(sampling_rate, channels, streams, coupled, &mapping, application)?;
        ms.mixing = Some(mixing);
        Ok(ProjectionEncoder { ms, demixing, streams, coupled, channels })
    }

    /// The streams of the coded packets.
    pub fn streams(&self) -> usize {
        self.streams
    }

    /// The coupled streams of the coded packets.
    pub fn coupled_streams(&self) -> usize {
        self.coupled
    }

    /// Encodes interleaved float ambisonic input (libopus
    /// `opus_projection_encode_float`).
    pub fn encode(&mut self, pcm: &[f32], frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        self.ms.encode(pcm, frame_size, out)
    }

    /// Encodes 16-bit input (libopus `opus_projection_encode`).
    pub fn encode_i16(&mut self, pcm: &[i16], frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        self.ms.encode_i16(pcm, frame_size, out)
    }

    /// Encodes 24-bit input in 32-bit integers (libopus
    /// `opus_projection_encode24`).
    pub fn encode24(&mut self, pcm: &[i32], frame_size: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        self.ms.encode24(pcm, frame_size, out)
    }

    /// The demixing matrix's gain in Q8 dB (libopus
    /// `OPUS_PROJECTION_GET_DEMIXING_MATRIX_GAIN`).
    pub fn demixing_matrix_gain(&self) -> i32 {
        self.demixing.gain
    }

    /// The bytes of the demixing matrix (libopus
    /// `OPUS_PROJECTION_GET_DEMIXING_MATRIX_SIZE`).
    pub fn demixing_matrix_size(&self) -> usize {
        self.channels * (self.streams + self.coupled) * 2
    }

    /// Writes the demixing matrix, 16-bit little-endian cells column by
    /// column, as [`ProjectionDecoder::new`] and the Ogg Opus header take
    /// it (libopus `OPUS_PROJECTION_GET_DEMIXING_MATRIX`);
    /// `error::BAD_ARG` unless `out` has [`Self::demixing_matrix_size`]
    /// bytes.
    pub fn demixing_matrix(&self, out: &mut [u8]) -> Result<(), &'static str> {
        if out.len() != self.demixing_matrix_size() {
            return Err(BAD_ARG);
        }
        let mut l = 0;
        for i in 0..self.streams + self.coupled {
            for j in 0..self.channels {
                let v = self.demixing.data[self.demixing.rows * i + j];
                out[2 * l..2 * l + 2].copy_from_slice(&v.to_le_bytes());
                l += 1;
            }
        }
        Ok(())
    }

    /// The multistream encoder, for the controls libopus's projection
    /// encoder forwards to it.
    pub fn multistream(&mut self) -> &mut MultistreamEncoder {
        &mut self.ms
    }
}

/// libopus `OpusProjectionDecoder`: a [`MultistreamDecoder`] whose decoded
/// channels the demixing matrix turns into the output channels.
pub struct ProjectionDecoder {
    ms: MultistreamDecoder,
    /// `channels` rows by `streams + coupled` columns, column by column.
    demixing: Vec<i16>,
    channels: usize,
}

impl ProjectionDecoder {
    /// libopus `opus_projection_decoder_create` with the demixing matrix
    /// the encoder reports. As in libopus, sizes no matrix or decoder can
    /// have are `error::ALLOC_FAIL`, a matrix of the wrong length or an
    /// invalid layout `error::BAD_ARG`.
    pub fn new(
        sampling_rate: i32,
        channels: usize,
        streams: usize,
        coupled: usize,
        demixing_matrix: &[u8],
    ) -> Result<Self, &'static str> {
        let inputs = streams + coupled;
        // opus_projection_decoder_get_size: mapping_matrix_get_size and
        // opus_multistream_decoder_get_size.
        if inputs > 255 || channels > 255 || inputs * channels * 2 > 65004 || streams < 1 || coupled > streams || streams > 255 - coupled {
            return Err(ALLOC_FAIL);
        }
        if demixing_matrix.len() != inputs * channels * 2 {
            return Err(BAD_ARG);
        }
        let demixing = demixing_matrix
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        let table = ChannelMappingTable {
            channels: u8::try_from(channels).map_err(|_| BAD_ARG)?,
            stream_count: streams as u8,
            coupled_count: coupled as u8,
            mapping: (0..channels as u8).collect(),
        };
        let ms = MultistreamDecoder::new(sampling_rate, table).map_err(|_| BAD_ARG)?;
        Ok(ProjectionDecoder { ms, demixing, channels })
    }

    fn matrix(&self) -> MappingMatrix<'_> {
        MappingMatrix {
            rows: self.channels,
            cols: self.demixing.len() / self.channels.max(1),
            gain: 0,
            data: &self.demixing,
        }
    }

    /// Decodes into interleaved float output (libopus
    /// `opus_projection_decode_float`).
    pub fn decode(&mut self, packet: &[u8], frame_size: usize, out: &mut [f32]) -> Result<usize, &'static str> {
        self.decode_float(packet, frame_size, out, false)
    }

    /// [`Self::decode`] with in-band FEC.
    pub fn decode_fec(&mut self, packet: &[u8], frame_size: usize, out: &mut [f32]) -> Result<usize, &'static str> {
        self.decode_float(packet, frame_size, out, true)
    }

    fn decode_float(&mut self, packet: &[u8], frame_size: usize, out: &mut [f32], fec: bool) -> Result<usize, &'static str> {
        let (n, decoded) = self.ms.decode_checked(packet, frame_size, out.len(), fec, false)?;
        let (ch, d) = (self.channels, self.ms.decoded_channels());
        out[..n * ch].fill(0.0);
        let m = self.matrix();
        for i in 0..d.min(ch) {
            m.out_float(&decoded[i..], i, d, out, ch, n);
        }
        Ok(n)
    }

    /// Decodes into interleaved 16-bit output, soft clipped per stream
    /// (libopus `opus_projection_decode`).
    pub fn decode_i16(&mut self, packet: &[u8], frame_size: usize, out: &mut [i16]) -> Result<usize, &'static str> {
        self.decode_int(packet, frame_size, out, false, true, |m, x, r, rs, o, oc, n| m.out_short(x, r, rs, o, oc, n))
    }

    /// [`Self::decode_i16`] with in-band FEC.
    pub fn decode_fec_i16(&mut self, packet: &[u8], frame_size: usize, out: &mut [i16]) -> Result<usize, &'static str> {
        self.decode_int(packet, frame_size, out, true, true, |m, x, r, rs, o, oc, n| m.out_short(x, r, rs, o, oc, n))
    }

    /// Decodes into interleaved 24-bit output in 32-bit integers (libopus
    /// `opus_projection_decode24`).
    pub fn decode24(&mut self, packet: &[u8], frame_size: usize, out: &mut [i32]) -> Result<usize, &'static str> {
        self.decode_int(packet, frame_size, out, false, false, |m, x, r, rs, o, oc, n| m.out_int24(x, r, rs, o, oc, n))
    }

    /// [`Self::decode24`] with in-band FEC.
    pub fn decode_fec24(&mut self, packet: &[u8], frame_size: usize, out: &mut [i32]) -> Result<usize, &'static str> {
        self.decode_int(packet, frame_size, out, true, false, |m, x, r, rs, o, oc, n| m.out_int24(x, r, rs, o, oc, n))
    }

    fn decode_int<T: IntSample + Default>(
        &mut self,
        packet: &[u8],
        frame_size: usize,
        out: &mut [T],
        fec: bool,
        soft_clip: bool,
        demix: impl Fn(&MappingMatrix<'_>, &[f32], usize, usize, &mut [T], usize, usize),
    ) -> Result<usize, &'static str> {
        let (n, decoded) = self.ms.decode_checked(packet, frame_size, out.len(), fec, soft_clip)?;
        let (ch, d) = (self.channels, self.ms.decoded_channels());
        out[..n * ch].fill(T::default());
        let m = self.matrix();
        for i in 0..d.min(ch) {
            demix(&m, &decoded[i..], i, d, out, ch, n);
        }
        Ok(n)
    }

    /// The multistream decoder, for the controls libopus's projection
    /// decoder forwards to it.
    pub fn multistream(&mut self) -> &mut MultistreamDecoder {
        &mut self.ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The demixing matrix has a column per decoded stream channel
    /// (libopus mapping_matrix_init with nb_input_streams).
    #[test]
    fn demixing_matrix_has_a_column_per_stream_channel() {
        let d = ProjectionDecoder::new(48000, 4, 2, 2, &[0u8; 32]).unwrap();
        let m = d.matrix();
        assert_eq!((m.rows, m.cols), (4, 4));
        assert_eq!(m.rows * m.cols, m.data.len());
    }
}
