//! Opus packet extensions (libopus `src/extensions.c`), the parts of
//! packet parsing that locate them (`opus_packet_parse_impl` of
//! `src/opus.c`) and the packet writer that adds them
//! (`opus_repacketizer_out_range_impl` of `src/repacketizer.c`).
//!
//! Extensions live in the padding of a code 3 packet. Each starts with an
//! ID byte (ID in the upper 7 bits, the `L` flag in bit 0): ID 0 is
//! padding, 1 a frame separator, 2 "repeat these extensions", 3 to 31 short
//! extensions of 0 or 1 payload bytes, and 32 to 127 long extensions with a
//! lacing-coded length (or, with `L` = 0, the rest of the padding).

/// libopus `OPUS_INVALID_PACKET`: the error of the parsers.
pub const INVALID_PACKET: &str = "invalid packet";
/// libopus `OPUS_BAD_ARG`.
pub const BAD_ARG: &str = "invalid argument";
/// libopus `OPUS_BUFFER_TOO_SMALL`.
pub const BUFFER_TOO_SMALL: &str = "buffer too small";

/// libopus `opus_extension_data`: one extension of a packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtensionData<'a> {
    /// 3 to 127.
    pub id: u8,
    /// The index of the frame the extension belongs to.
    pub frame: usize,
    /// The payload.
    pub data: &'a [u8],
}

/// libopus `skip_extension_payload`: moves `pos` past the payload of an
/// extension whose ID byte is `id_byte` (and that is already skipped) and
/// returns the length left, or -1 when the payload does not fit. Sets
/// `header_size` to the length bytes it read. "Repeat these extensions"
/// does not move past the repeated payloads.
fn skip_extension_payload(
    data: &[u8],
    pos: &mut usize,
    mut len: i32,
    header_size: &mut i32,
    id_byte: u8,
    trailing_short_len: i32,
) -> i32 {
    let mut p = *pos;
    let mut hs = 0;
    let id = id_byte >> 1;
    let l = (id_byte & 1) as i32;
    if (id == 0 && l == 1) || id == 2 {
        // Nothing to do.
    } else if id > 0 && id < 32 {
        if len < l {
            return -1;
        }
        p += l as usize;
        len -= l;
    } else if l == 0 {
        if len < trailing_short_len {
            return -1;
        }
        p += (len - trailing_short_len) as usize;
        len = trailing_short_len;
    } else {
        let mut bytes = 0i32;
        loop {
            if len < 1 {
                return -1;
            }
            let lacing = data[p] as i32;
            p += 1;
            bytes += lacing;
            hs += 1;
            len -= lacing + 1;
            if lacing != 255 {
                break;
            }
        }
        if len < 0 {
            return -1;
        }
        p += bytes as usize;
    }
    *pos = p;
    *header_size = hs;
    len
}

/// libopus `skip_extension`: moves `pos` past one extension, its ID byte
/// included, and returns the length left, or -1 when it does not fit.
fn skip_extension(data: &[u8], pos: &mut usize, len: i32, header_size: &mut i32) -> i32 {
    if len == 0 {
        *header_size = 0;
        return 0;
    }
    if len < 1 {
        return -1;
    }
    let mut p = *pos;
    let id_byte = data[p];
    p += 1;
    let len = skip_extension_payload(data, &mut p, len - 1, header_size, id_byte, 0);
    if len >= 0 {
        *pos = p;
        *header_size += 1;
    }
    len
}

/// libopus `OpusExtensionIterator`: walks the extensions of a packet's
/// padding in bitstream order, repeated extensions included. The C
/// pointers are offsets into `data`.
#[derive(Clone, Debug)]
pub struct ExtensionIterator<'a> {
    data: &'a [u8],
    curr_data: usize,
    repeat_data: usize,
    last_long: Option<usize>,
    src_data: usize,
    curr_len: i32,
    repeat_len: i32,
    src_len: i32,
    trailing_short_len: i32,
    nb_frames: usize,
    frame_max: usize,
    curr_frame: usize,
    repeat_frame: usize,
    repeat_l: u8,
}

impl<'a> ExtensionIterator<'a> {
    /// `opus_extension_iterator_init`: the extensions in `data` (a packet's
    /// padding) of a packet with `nb_frames` frames (at most 48).
    pub fn new(data: &'a [u8], nb_frames: usize) -> Self {
        debug_assert!(nb_frames <= 48);
        ExtensionIterator {
            data,
            curr_data: 0,
            repeat_data: 0,
            last_long: None,
            src_data: 0,
            curr_len: data.len() as i32,
            repeat_len: 0,
            src_len: 0,
            trailing_short_len: 0,
            nb_frames,
            frame_max: nb_frames,
            curr_frame: 0,
            repeat_frame: 0,
            repeat_l: 0,
        }
    }

    /// `opus_extension_iterator_reset`: back to the first extension.
    pub fn reset(&mut self) {
        self.repeat_data = 0;
        self.curr_data = 0;
        self.last_long = None;
        self.curr_len = self.data.len() as i32;
        self.repeat_frame = 0;
        self.curr_frame = 0;
        self.trailing_short_len = 0;
    }

    /// `opus_extension_iterator_set_frame_max`: no extensions of frame
    /// `frame_max` or later.
    pub fn set_frame_max(&mut self, frame_max: usize) {
        self.frame_max = frame_max;
    }

    /// `opus_extension_iterator_next_repeat`: the next repeated extension,
    /// `Ok(None)` when the repetition is done.
    fn next_repeat(&mut self) -> Result<Option<ExtensionData<'a>>, &'static str> {
        let mut header_size = 0;
        debug_assert!(self.repeat_frame > 0);
        while self.repeat_frame < self.nb_frames {
            while self.src_len > 0 {
                let mut repeat_id_byte = self.data[self.src_data];
                self.src_len = skip_extension(self.data, &mut self.src_data, self.src_len, &mut header_size);
                // Skipped once before, so it fits now.
                debug_assert!(self.src_len >= 0);
                // Padding and frame separators are not repeated.
                if repeat_id_byte <= 3 {
                    continue;
                }
                // With L == 0 on the repeat indicator, the last repeated
                // long extension of the last frame takes the rest.
                if self.repeat_l == 0
                    && self.repeat_frame + 1 >= self.nb_frames
                    && Some(self.src_data) == self.last_long
                {
                    repeat_id_byte &= !1;
                }
                let curr_data0 = self.curr_data;
                self.curr_len = skip_extension_payload(
                    self.data,
                    &mut self.curr_data,
                    self.curr_len,
                    &mut header_size,
                    repeat_id_byte,
                    self.trailing_short_len,
                );
                if self.curr_len < 0 {
                    return Err(INVALID_PACKET);
                }
                debug_assert_eq!(self.curr_data as i32, self.data.len() as i32 - self.curr_len);
                if self.repeat_frame >= self.frame_max {
                    continue;
                }
                let start = curr_data0 + header_size as usize;
                return Ok(Some(ExtensionData {
                    id: repeat_id_byte >> 1,
                    frame: self.repeat_frame,
                    data: &self.data[start..self.curr_data],
                }));
            }
            // This frame's repetitions are done.
            self.src_data = self.repeat_data;
            self.src_len = self.repeat_len;
            self.repeat_frame += 1;
        }
        self.repeat_data = self.curr_data;
        self.last_long = None;
        // With L == 0, data a long extension left over moves on a frame.
        if self.repeat_l == 0 {
            self.curr_frame += 1;
            // Padding after the last frame is ignored.
            if self.curr_frame >= self.nb_frames {
                self.curr_len = 0;
            }
        }
        self.repeat_frame = 0;
        Ok(None)
    }

    /// `opus_extension_iterator_next`: the next extension (padding,
    /// separators and repeat indicators skipped, repeated extensions
    /// included), `Ok(None)` at the end.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<ExtensionData<'a>>, &'static str> {
        let mut header_size = 0;
        if self.curr_len < 0 {
            return Err(INVALID_PACKET);
        }
        if self.repeat_frame > 0
            && let Some(ext) = self.next_repeat()?
        {
            return Ok(Some(ext));
        }
        // Checked here so that set_frame_max works at any point.
        if self.curr_frame >= self.frame_max {
            return Ok(None);
        }
        while self.curr_len > 0 {
            let curr_data0 = self.curr_data;
            let id = self.data[curr_data0] >> 1;
            let l = self.data[curr_data0] & 1;
            self.curr_len = skip_extension(self.data, &mut self.curr_data, self.curr_len, &mut header_size);
            if self.curr_len < 0 {
                return Err(INVALID_PACKET);
            }
            debug_assert_eq!(self.curr_data as i32, self.data.len() as i32 - self.curr_len);
            if id == 1 {
                if l == 0 {
                    self.curr_frame += 1;
                } else {
                    let inc = self.data[curr_data0 + 1] as usize;
                    // An increment of 0 does nothing.
                    if inc == 0 {
                        continue;
                    }
                    self.curr_frame += inc;
                }
                if self.curr_frame >= self.nb_frames {
                    self.curr_len = -1;
                    return Err(INVALID_PACKET);
                }
                if self.curr_frame >= self.frame_max {
                    self.curr_len = 0;
                }
                self.repeat_data = self.curr_data;
                self.last_long = None;
                self.trailing_short_len = 0;
            } else if id == 2 {
                self.repeat_l = l;
                self.repeat_frame = self.curr_frame + 1;
                self.repeat_len = (curr_data0 - self.repeat_data) as i32;
                self.src_data = self.repeat_data;
                self.src_len = self.repeat_len;
                if let Some(ext) = self.next_repeat()? {
                    return Ok(Some(ext));
                }
            } else if id > 2 {
                // Where the last long extension ends, for a repetition with
                // L == 0; the payload bytes of short ones after it.
                if id >= 32 {
                    self.last_long = Some(self.curr_data);
                    self.trailing_short_len = 0;
                } else {
                    self.trailing_short_len += l as i32;
                }
                let start = curr_data0 + header_size as usize;
                return Ok(Some(ExtensionData {
                    id,
                    frame: self.curr_frame,
                    data: &self.data[start..self.curr_data],
                }));
            }
        }
        Ok(None)
    }

    /// `opus_extension_iterator_find`: the next extension with ID `id`.
    pub fn find(&mut self, id: u8) -> Result<Option<ExtensionData<'a>>, &'static str> {
        loop {
            match self.next()? {
                None => return Ok(None),
                Some(ext) if ext.id == id => return Ok(Some(ext)),
                Some(_) => {}
            }
        }
    }
}

/// The samples per frame of a packet at 48 kHz, from its ToC
/// (`opus_packet_get_samples_per_frame`).
pub fn samples_per_frame_48k(toc: u8) -> usize {
    if toc & 0x80 != 0 {
        (48000 << ((toc >> 3) & 3)) / 400
    } else if toc & 0x60 == 0x60 {
        if toc & 0x08 != 0 { 960 } else { 480 }
    } else {
        match (toc >> 3) & 3 {
            3 => 2880,
            d => (48000 << d) / 100,
        }
    }
}

/// libopus `parse_size`: a frame length and the bytes it takes.
fn parse_size(data: &[u8]) -> Option<(usize, usize)> {
    match *data {
        [] => None,
        [b, ..] if b < 252 => Some((b as usize, 1)),
        [_] => None,
        [b0, b1, ..] => Some((4 * b1 as usize + b0 as usize, 2)),
    }
}

/// The parts of a packet that `opus_packet_parse_impl` (without
/// self-delimiting) returns.
#[derive(Clone, Copy, Debug)]
pub struct PacketParts<'a> {
    pub toc: u8,
    /// The number of frames.
    pub count: usize,
    /// The frames, `frames[..count]`.
    pub frames: [&'a [u8]; 48],
    /// The padding of a code 3 packet, where extensions live.
    pub padding: &'a [u8],
}

/// libopus `opus_packet_parse_impl` without self-delimiting: the ToC,
/// frames and padding of `data`, or an error for an invalid packet.
pub fn parse_packet(data: &[u8]) -> Result<PacketParts<'_>, &'static str> {
    let (&toc, rest) = data.split_first().ok_or(INVALID_PACKET)?;
    let framesize = samples_per_frame_48k(toc);
    let mut sizes = [0usize; 48];
    let mut pos = 1usize;
    let mut len = rest.len() as i64;
    let mut last_size = len;
    let mut pad = 0usize;
    let count;
    match toc & 3 {
        0 => count = 1,
        1 => {
            count = 2;
            if len & 1 != 0 {
                return Err(INVALID_PACKET);
            }
            last_size = len / 2;
            sizes[0] = last_size as usize;
        }
        2 => {
            count = 2;
            let (s, bytes) = parse_size(&data[pos..]).ok_or(INVALID_PACKET)?;
            len -= bytes as i64;
            if s as i64 > len {
                return Err(INVALID_PACKET);
            }
            sizes[0] = s;
            pos += bytes;
            last_size = len - s as i64;
        }
        _ => {
            if len < 1 {
                return Err(INVALID_PACKET);
            }
            let ch = data[pos];
            pos += 1;
            count = (ch & 0x3f) as usize;
            if count == 0 || framesize * count > 5760 {
                return Err(INVALID_PACKET);
            }
            len -= 1;
            if ch & 0x40 != 0 {
                loop {
                    if len <= 0 {
                        return Err(INVALID_PACKET);
                    }
                    let p = data[pos] as usize;
                    pos += 1;
                    len -= 1;
                    let tmp = if p == 255 { 254 } else { p };
                    len -= tmp as i64;
                    pad += tmp;
                    if p != 255 {
                        break;
                    }
                }
            }
            if len < 0 {
                return Err(INVALID_PACKET);
            }
            if ch & 0x80 != 0 {
                // VBR: the lengths of all but the last frame.
                last_size = len;
                for size in sizes.iter_mut().take(count - 1) {
                    let (s, bytes) = parse_size(&data[pos..]).ok_or(INVALID_PACKET)?;
                    len -= bytes as i64;
                    if s as i64 > len {
                        return Err(INVALID_PACKET);
                    }
                    *size = s;
                    pos += bytes;
                    last_size -= (bytes + s) as i64;
                }
                if last_size < 0 {
                    return Err(INVALID_PACKET);
                }
            } else {
                last_size = len / count as i64;
                if last_size * count as i64 != len {
                    return Err(INVALID_PACKET);
                }
                for size in sizes.iter_mut().take(count - 1) {
                    *size = last_size as usize;
                }
            }
        }
    }
    // The last frame (all of them for CBR) has no explicit length, so it
    // can exceed 1275 bytes.
    if last_size > 1275 {
        return Err(INVALID_PACKET);
    }
    sizes[count - 1] = last_size as usize;
    let mut frames: [&[u8]; 48] = [&[]; 48];
    for i in 0..count {
        frames[i] = &data[pos..pos + sizes[i]];
        pos += sizes[i];
    }
    Ok(PacketParts { toc, count, frames, padding: &data[pos..pos + pad] })
}

/// `write_extension_payload`: writes the payload of `ext` at `pos` (with
/// its length unless `last`) and returns the new position. With `data`
/// `None` it only counts.
fn write_extension_payload(
    data: &mut Option<&mut [u8]>,
    len: usize,
    mut pos: usize,
    ext: &ExtensionData<'_>,
    last: bool,
) -> Result<usize, &'static str> {
    debug_assert!((3..=127).contains(&ext.id));
    if ext.id < 32 {
        if ext.data.len() > 1 {
            return Err(BAD_ARG);
        }
        if let Some(&b) = ext.data.first() {
            if len - pos < 1 {
                return Err(BUFFER_TOO_SMALL);
            }
            if let Some(d) = data {
                d[pos] = b;
            }
            pos += 1;
        }
    } else {
        let n = ext.data.len();
        let length_bytes = if last { 0 } else { 1 + n / 255 };
        if len - pos < length_bytes + n {
            return Err(BUFFER_TOO_SMALL);
        }
        if !last {
            for _ in 0..n / 255 {
                if let Some(d) = data {
                    d[pos] = 255;
                }
                pos += 1;
            }
            if let Some(d) = data {
                d[pos] = (n % 255) as u8;
            }
            pos += 1;
        }
        if let Some(d) = data {
            d[pos..pos + n].copy_from_slice(ext.data);
        }
        pos += n;
    }
    Ok(pos)
}

/// `write_extension`: the ID byte and the payload of `ext`.
fn write_extension(
    data: &mut Option<&mut [u8]>,
    len: usize,
    pos: usize,
    ext: &ExtensionData<'_>,
    last: bool,
) -> Result<usize, &'static str> {
    if len - pos < 1 {
        return Err(BUFFER_TOO_SMALL);
    }
    debug_assert!((3..=127).contains(&ext.id));
    if let Some(d) = data {
        d[pos] = (ext.id << 1) + if ext.id < 32 { ext.data.len() as u8 } else { u8::from(!last) };
    }
    write_extension_payload(data, len, pos + 1, ext, last)
}

/// libopus `opus_packet_extensions_generate`: writes `extensions` for a
/// packet of `nb_frames` frames into `data` (at most `len` bytes; with
/// `None` it only measures), with separators and the repeat mechanism,
/// and with `pad` fills the rest of `len` with padding in front. Returns
/// the bytes written.
pub fn generate_extensions(
    mut data: Option<&mut [u8]>,
    len: usize,
    extensions: &[ExtensionData<'_>],
    nb_frames: usize,
    pad: bool,
) -> Result<usize, &'static str> {
    if nb_frames > 48 {
        return Err(BAD_ARG);
    }
    let nb_extensions = extensions.len();
    let mut frame_min_idx = [nb_extensions; 48];
    let mut frame_max_idx = [0usize; 48];
    let mut frame_repeat_idx = [0usize; 48];
    for (i, ext) in extensions.iter().enumerate() {
        let f = ext.frame;
        if f >= nb_frames || !(3..=127).contains(&ext.id) {
            return Err(BAD_ARG);
        }
        frame_min_idx[f] = frame_min_idx[f].min(i);
        frame_max_idx[f] = frame_max_idx[f].max(i + 1);
    }
    frame_repeat_idx[..nb_frames].copy_from_slice(&frame_min_idx[..nb_frames]);
    let mut curr_frame = 0;
    let mut pos = 0;
    let mut written = 0;
    for f in 0..nb_frames {
        let mut last_long_idx: Option<usize> = None;
        let mut repeat_count = 0;
        if f + 1 < nb_frames {
            for i in frame_min_idx[f]..frame_max_idx[f] {
                if extensions[i].frame != f {
                    continue;
                }
                // Can this extension repeat in all later frames?
                let mut g = f + 1;
                while g < nb_frames {
                    if frame_repeat_idx[g] >= frame_max_idx[g] {
                        break;
                    }
                    let other = &extensions[frame_repeat_idx[g]];
                    debug_assert_eq!(other.frame, g);
                    if other.id != extensions[i].id {
                        break;
                    }
                    if other.id < 32 && other.data.len() != extensions[i].data.len() {
                        break;
                    }
                    g += 1;
                }
                if g < nb_frames {
                    break;
                }
                // It can. The last instance of a long one changes its L
                // flag.
                if extensions[i].id >= 32 {
                    last_long_idx = Some(frame_repeat_idx[nb_frames - 1]);
                }
                for g in f + 1..nb_frames {
                    let mut j = frame_repeat_idx[g] + 1;
                    while j < frame_max_idx[g] && extensions[j].frame != g {
                        j += 1;
                    }
                    frame_repeat_idx[g] = j;
                }
                repeat_count += 1;
                frame_repeat_idx[f] = i;
            }
        }
        for i in frame_min_idx[f]..frame_max_idx[f] {
            if extensions[i].frame != f {
                continue;
            }
            // A separator where the frame changes.
            if f != curr_frame {
                let diff = f - curr_frame;
                if len - pos < 2 {
                    return Err(BUFFER_TOO_SMALL);
                }
                if diff == 1 {
                    if let Some(d) = &mut data {
                        d[pos] = 0x02;
                    }
                    pos += 1;
                } else {
                    if let Some(d) = &mut data {
                        d[pos] = 0x03;
                        d[pos + 1] = diff as u8;
                    }
                    pos += 2;
                }
                curr_frame = f;
            }
            pos = write_extension(&mut data, len, pos, &extensions[i], written == nb_extensions - 1)?;
            written += 1;
            if repeat_count > 0 && frame_repeat_idx[f] == i {
                // The repeat indicator, then the repeated payloads.
                let nb_repeated = repeat_count * (nb_frames - (f + 1));
                let last = written + nb_repeated == nb_extensions
                    || (last_long_idx.is_none() && i + 1 >= frame_max_idx[f]);
                if len - pos < 1 {
                    return Err(BUFFER_TOO_SMALL);
                }
                if let Some(d) = &mut data {
                    d[pos] = 0x04 + u8::from(!last);
                }
                pos += 1;
                for g in f + 1..nb_frames {
                    let mut j = frame_min_idx[g];
                    while j < frame_repeat_idx[g] {
                        if extensions[j].frame == g {
                            pos = write_extension_payload(
                                &mut data,
                                len,
                                pos,
                                &extensions[j],
                                last && Some(j) == last_long_idx,
                            )?;
                            written += 1;
                        }
                        j += 1;
                    }
                    frame_min_idx[g] = j;
                }
                if last {
                    curr_frame += 1;
                }
            }
        }
    }
    debug_assert_eq!(written, nb_extensions);
    // Padding goes in front, as 0x01 bytes.
    if pad && pos < len {
        let padding = len - pos;
        if let Some(d) = &mut data {
            d.copy_within(..pos, padding);
            d[..padding].fill(0x01);
        }
        pos += padding;
    }
    Ok(pos)
}

/// libopus `encode_size`: a frame length in one or two bytes.
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

/// libopus `opus_repacketizer_out_range_impl` without self-delimiting:
/// writes a packet of `frames` (sharing the ToC `toc`, at most 48) with
/// `extensions` in its padding into `out`, at most `maxlen` bytes, and
/// with `pad` padded to `maxlen`. Returns the packet length.
pub fn write_packet(
    out: &mut [u8],
    toc: u8,
    frames: &[&[u8]],
    extensions: &[ExtensionData<'_>],
    maxlen: usize,
    pad: bool,
) -> Result<usize, &'static str> {
    let count = frames.len();
    if count == 0 || count > 48 {
        return Err(BAD_ARG);
    }
    let maxlen = maxlen.min(out.len());
    let len = |i: usize| frames[i].len();
    let mut tot_size = 0;
    let mut ptr = 0;
    if count == 1 {
        tot_size += len(0) + 1;
        if tot_size > maxlen {
            return Err(BUFFER_TOO_SMALL);
        }
        out[0] = toc & 0xfc;
        ptr = 1;
    } else if count == 2 {
        if len(1) == len(0) {
            tot_size += 2 * len(0) + 1;
            if tot_size > maxlen {
                return Err(BUFFER_TOO_SMALL);
            }
            out[0] = (toc & 0xfc) | 1;
            ptr = 1;
        } else {
            tot_size += len(0) + len(1) + 2 + usize::from(len(0) >= 252);
            if tot_size > maxlen {
                return Err(BUFFER_TOO_SMALL);
            }
            out[0] = (toc & 0xfc) | 2;
            ptr = 1 + encode_size(len(0), &mut out[1..]);
        }
    }
    let mut ext_len = 0;
    let mut ext_begin = 0;
    let (mut ones_begin, mut ones_end) = (0, 0);
    if count > 2 || (pad && tot_size < maxlen) || !extensions.is_empty() {
        // Code 3, also for the padding.
        let vbr = (1..count).any(|i| len(i) != len(0));
        if vbr {
            tot_size = 2 + (0..count - 1).map(|i| 1 + usize::from(len(i) >= 252) + len(i)).sum::<usize>() + len(count - 1);
            if tot_size > maxlen {
                return Err(BUFFER_TOO_SMALL);
            }
            out[0] = (toc & 0xfc) | 3;
            out[1] = count as u8 | 0x80;
        } else {
            tot_size = count * len(0) + 2;
            if tot_size > maxlen {
                return Err(BUFFER_TOO_SMALL);
            }
            out[0] = (toc & 0xfc) | 3;
            out[1] = count as u8;
        }
        ptr = 2;
        let mut pad_amount = if pad { maxlen - tot_size } else { 0 };
        if !extensions.is_empty() {
            ext_len = generate_extensions(None, maxlen - tot_size, extensions, count, false)?;
            if !pad {
                pad_amount = ext_len + if ext_len != 0 { ext_len.div_ceil(254) } else { 1 };
            }
        }
        if pad_amount != 0 {
            out[1] |= 0x40;
            let nb_255s = (pad_amount - 1) / 255;
            if tot_size + ext_len + nb_255s + 1 > maxlen {
                return Err(BUFFER_TOO_SMALL);
            }
            ext_begin = tot_size + pad_amount - ext_len;
            // 0x01 padding in front of the extensions.
            ones_begin = tot_size + nb_255s + 1;
            ones_end = tot_size + pad_amount - ext_len;
            out[ptr..ptr + nb_255s].fill(255);
            ptr += nb_255s;
            out[ptr] = (pad_amount - 255 * nb_255s - 1) as u8;
            ptr += 1;
            tot_size += pad_amount;
        }
        if vbr {
            for i in 0..count - 1 {
                ptr += encode_size(len(i), &mut out[ptr..]);
            }
        }
    }
    for f in frames {
        out[ptr..ptr + f.len()].copy_from_slice(f);
        ptr += f.len();
    }
    if ext_len > 0 {
        let n = generate_extensions(Some(&mut out[ext_begin..ext_begin + ext_len]), ext_len, extensions, count, false)?;
        debug_assert_eq!(n, ext_len);
    }
    out[ones_begin..ones_end].fill(0x01);
    if pad && extensions.is_empty() {
        out[ptr..maxlen].fill(0);
    }
    Ok(tot_size)
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn all(data: &[u8], nb_frames: usize) -> Result<std::vec::Vec<(u8, usize, std::vec::Vec<u8>)>, &'static str> {
        let mut it = ExtensionIterator::new(data, nb_frames);
        let mut out = std::vec::Vec::new();
        while let Some(e) = it.next()? {
            out.push((e.id, e.frame, e.data.to_vec()));
        }
        Ok(out)
    }

    #[test]
    fn short_long_and_separators() {
        // A short extension 3 with one byte, a long extension 33 with
        // "ab" (L = 1, length 2), a separator, then a long extension 126
        // taking the rest (L = 0).
        let data = [3 << 1 | 1, 7, 33 << 1 | 1, 2, b'a', b'b', 0x02, 126 << 1, b'D', 12, 1];
        assert_eq!(
            all(&data, 2).unwrap(),
            [(3, 0, vec![7]), (33, 0, b"ab".to_vec()), (126, 1, vec![b'D', 12, 1])]
        );
        // Without a second frame the separator is invalid.
        assert!(all(&data, 1).is_err());
    }

    #[test]
    fn repeat_extensions() {
        // Extension 3 (no payload) and long 40 with "x" in frame 0,
        // repeated in frames 1 and 2 with L = 0: the repeated payloads
        // follow, and the last repeated long one takes the rest.
        let data = [3 << 1, 40 << 1 | 1, 1, b'x', 0x04, 1, b'y', b'z', b'w'];
        assert_eq!(
            all(&data, 3).unwrap(),
            [(3, 0, vec![]), (40, 0, b"x".to_vec()), (3, 1, vec![]), (40, 1, b"y".to_vec()), (3, 2, vec![]), (40, 2, b"zw".to_vec())]
        );
    }

    #[test]
    fn parse_packet_padding() {
        // Code 3, one CBR frame of 2 bytes, 3 bytes of padding.
        let p = [0x7b, 0x41, 3, 9, 9, 1, 2, 3];
        let parts = parse_packet(&p).unwrap();
        assert_eq!((parts.count, parts.frames[0], parts.padding), (1, &[9u8, 9][..], &[1u8, 2, 3][..]));
        assert!(parse_packet(&[0x7b, 0x41, 7, 9]).is_err());
    }
}
