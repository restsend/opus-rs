//! libopus's repacketizer (`src/repacketizer.c`): merging the frames of
//! packets with the same ToC configuration into one packet, splitting a
//! packet's frames into several, and padding and unpadding packets,
//! single-stream and multistream.
//!
//! Packets keep their extensions: a repacketized packet carries the
//! extensions of the frames it holds, renumbered. Unpadding drops them.

use crate::error::{ALLOC_FAIL, BAD_ARG, INVALID_PACKET};
use crate::extensions::{ExtensionData, ExtensionIterator, parse_packet_impl, write_packet};
use crate::packet::{nb_frames, samples_per_frame};

/// The most extensions a packet written by [`Repacketizer::out_range`] or
/// [`pad`] holds; more are `error::ALLOC_FAIL`.
pub const MAX_EXTENSIONS: usize = 256;

/// libopus `OpusRepacketizer`: up to 120 ms of frames that share a ToC
/// configuration, borrowed from the packets added with [`Self::cat`].
#[derive(Clone, Copy, Debug)]
pub struct Repacketizer<'a> {
    toc: u8,
    nb_frames: usize,
    frames: [&'a [u8]; 48],
    /// The padding of the packet that starts at each frame, empty for the
    /// other frames, and the frame count of that packet.
    paddings: [&'a [u8]; 48],
    padding_nb_frames: [u8; 48],
    /// Samples per frame at 8 kHz.
    framesize: i32,
}

impl Default for Repacketizer<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Repacketizer<'a> {
    /// An empty repacketizer (libopus `opus_repacketizer_create` and
    /// `opus_repacketizer_init`).
    pub const fn new() -> Self {
        Repacketizer {
            toc: 0,
            nb_frames: 0,
            frames: [&[]; 48],
            paddings: [&[]; 48],
            padding_nb_frames: [0; 48],
            framesize: 0,
        }
    }

    /// Empties the repacketizer (libopus `opus_repacketizer_init`).
    pub fn init(&mut self) {
        self.nb_frames = 0;
    }

    /// The frames held (libopus `opus_repacketizer_get_nb_frames`).
    pub fn nb_frames(&self) -> usize {
        self.nb_frames
    }

    /// Adds the frames of `data` (libopus `opus_repacketizer_cat`).
    /// `error::INVALID_PACKET` for an invalid packet, for one whose ToC
    /// differs from the first packet's in more than the frame count code,
    /// and when the frames would exceed 120 ms; the repacketizer is then
    /// unchanged.
    pub fn cat(&mut self, data: &'a [u8]) -> Result<(), &'static str> {
        self.cat_impl(data, false)
    }

    fn cat_impl(&mut self, data: &'a [u8], self_delimited: bool) -> Result<(), &'static str> {
        let &toc = data.first().ok_or(INVALID_PACKET)?;
        if self.nb_frames == 0 {
            self.toc = toc;
            self.framesize = samples_per_frame(toc, 8000);
        } else if self.toc & 0xfc != toc & 0xfc {
            return Err(INVALID_PACKET);
        }
        let count = nb_frames(data).map_err(|_| INVALID_PACKET)?;
        if count < 1 || (count + self.nb_frames) as i32 * self.framesize > 960 {
            return Err(INVALID_PACKET);
        }
        let parts = parse_packet_impl(data, self_delimited)?;
        let n = self.nb_frames;
        self.frames[n..n + parts.count].copy_from_slice(parts.frames());
        self.paddings[n..n + parts.count].fill(&[]);
        self.padding_nb_frames[n..n + parts.count].fill(0);
        self.paddings[n] = parts.padding;
        self.padding_nb_frames[n] = parts.count as u8;
        self.nb_frames += parts.count;
        Ok(())
    }

    /// Writes frames `begin..end` as one packet into `out`, at most
    /// `out.len()` bytes, and returns its length (libopus
    /// `opus_repacketizer_out_range`). `error::BAD_ARG` for an empty or
    /// out-of-range span, `error::BUFFER_TOO_SMALL` when the packet does
    /// not fit.
    pub fn out_range(&self, begin: usize, end: usize, out: &mut [u8]) -> Result<usize, &'static str> {
        self.out_range_impl(begin, end, out, false)
    }

    /// Writes all frames as one packet (libopus `opus_repacketizer_out`).
    pub fn out(&self, out: &mut [u8]) -> Result<usize, &'static str> {
        self.out_range_impl(0, self.nb_frames, out, false)
    }

    /// libopus `opus_repacketizer_out_range_impl` (not self-delimited,
    /// without added extensions): with `pad`, the packet fills `out`.
    fn out_range_impl(
        &self,
        begin: usize,
        end: usize,
        out: &mut [u8],
        pad: bool,
    ) -> Result<usize, &'static str> {
        if begin >= end || end > self.nb_frames {
            return Err(BAD_ARG);
        }
        // The extensions of the frames in the span, renumbered.
        let mut exts = [ExtensionData {
            id: 0,
            frame: 0,
            data: &[],
        }; MAX_EXTENSIONS];
        let mut n = 0;
        let mut overflow = false;
        for i in 0..end {
            let nb = self.padding_nb_frames[i] as usize;
            if self.paddings[i].is_empty() || i + nb <= begin {
                continue;
            }
            let mut iter = ExtensionIterator::new(self.paddings[i], nb);
            while let Some(ext) = iter.next().map_err(|_| INVALID_PACKET)? {
                let f = i + ext.frame;
                if f >= begin && f < end {
                    if n == MAX_EXTENSIONS {
                        overflow = true;
                        continue;
                    }
                    exts[n] = ExtensionData {
                        frame: f - begin,
                        ..ext
                    };
                    n += 1;
                }
            }
        }
        if overflow {
            return Err(ALLOC_FAIL);
        }
        let len = out.len();
        write_packet(out, self.toc, &self.frames[begin..end], &exts[..n], len, pad)
    }
}

/// libopus `opus_packet_pad`: `packet` padded to `out.len()` bytes into
/// `out`, as the same frames and extensions. `error::BAD_ARG` for an empty
/// packet or one longer than `out`.
pub fn pad(packet: &[u8], out: &mut [u8]) -> Result<(), &'static str> {
    if packet.is_empty() || packet.len() > out.len() {
        return Err(BAD_ARG);
    }
    if packet.len() == out.len() {
        out.copy_from_slice(packet);
        return Ok(());
    }
    let mut rp = Repacketizer::new();
    rp.cat(packet)?;
    rp.out_range_impl(0, rp.nb_frames, out, true).map(|_| ())
}

/// libopus `opus_packet_unpad`: removes the padding and extensions of the
/// packet in `data`, in place, and returns its new length.
pub fn unpad(data: &mut [u8]) -> Result<usize, &'static str> {
    if data.is_empty() {
        return Err(BAD_ARG);
    }
    let len = data.len();
    unpad_stream(data, 0, len, 0, false)
}

/// libopus `opus_multistream_packet_pad`: the last stream of a multistream
/// packet of `nb_streams` streams padded so that the packet fills `out`.
pub fn multistream_pad(packet: &[u8], nb_streams: usize, out: &mut [u8]) -> Result<(), &'static str> {
    if packet.is_empty() || packet.len() > out.len() {
        return Err(BAD_ARG);
    }
    if packet.len() == out.len() {
        out.copy_from_slice(packet);
        return Ok(());
    }
    let amount = out.len() - packet.len();
    let mut pos = 0;
    for _ in 1..nb_streams {
        if pos >= packet.len() {
            return Err(INVALID_PACKET);
        }
        pos += parse_packet_impl(&packet[pos..], true)?.packet_offset;
    }
    out[..pos].copy_from_slice(&packet[..pos]);
    let last = &packet[pos..];
    pad(last, &mut out[pos..pos + last.len() + amount])
}

/// libopus `opus_multistream_packet_unpad`: removes the padding and
/// extensions of every stream of a multistream packet of `nb_streams`
/// streams in `data`, in place, and returns its new length.
pub fn multistream_unpad(data: &mut [u8], nb_streams: usize) -> Result<usize, &'static str> {
    if data.is_empty() {
        return Err(BAD_ARG);
    }
    let (mut src, mut dst) = (0, 0);
    for s in 0..nb_streams {
        if src >= data.len() {
            return Err(INVALID_PACKET);
        }
        let self_delimited = s + 1 != nb_streams;
        let len = parse_packet_impl(&data[src..], self_delimited)?.packet_offset;
        dst += unpad_stream(data, src, len, dst, self_delimited)?;
        src += len;
    }
    Ok(dst)
}

/// libopus `opus_packet_unpad` of one stream in place: the packet at
/// `data[src..src + len]` without padding and extensions, written at
/// `dst <= src` (`opus_repacketizer_out_range_impl` without padding and
/// extensions). Returns the length written.
fn unpad_stream(
    data: &mut [u8],
    src: usize,
    len: usize,
    dst: usize,
    self_delimited: bool,
) -> Result<usize, &'static str> {
    let base = data.as_ptr() as usize;
    let mut rp = Repacketizer::new();
    rp.cat_impl(&data[src..src + len], self_delimited)?;
    let count = rp.nb_frames;
    let mut frames = [(0usize, 0usize); 48];
    for (f, frame) in frames.iter_mut().zip(&rp.frames[..count]) {
        *f = (frame.as_ptr() as usize - base, frame.len());
    }
    let toc = rp.toc & 0xfc;
    let size = |i: usize| frames[i].1;
    // The header: ToC, frame count and lengths, and the self-delimiting
    // length of the last frame.
    let mut header = [0u8; 2 + 2 * 48];
    let mut h = 1;
    if count == 1 {
        header[0] = toc;
    } else if count == 2 && size(0) == size(1) {
        header[0] = toc | 1;
    } else if count == 2 {
        header[0] = toc | 2;
        h += encode_size(size(0), &mut header[h..]);
    } else {
        let vbr = (1..count).any(|i| size(i) != size(0));
        header[0] = toc | 3;
        header[1] = count as u8 | if vbr { 0x80 } else { 0 };
        h = 2;
        if vbr {
            for i in 0..count - 1 {
                h += encode_size(size(i), &mut header[h..]);
            }
        }
    }
    if self_delimited {
        h += encode_size(size(count - 1), &mut header[h..]);
    }
    // The header is no longer than the packet's, so every frame moves to
    // the front or stays.
    data[dst..dst + h].copy_from_slice(&header[..h]);
    let mut ptr = dst + h;
    for &(at, n) in &frames[..count] {
        data.copy_within(at..at + n, ptr);
        ptr += n;
    }
    Ok(ptr - dst)
}

/// libopus `encode_size`.
fn encode_size(size: usize, data: &mut [u8]) -> usize {
    if size < 252 {
        data[0] = size as u8;
        1
    } else {
        let first = 252 + (size & 3);
        data[0] = first as u8;
        data[1] = ((size - first) >> 2) as u8;
        2
    }
}
