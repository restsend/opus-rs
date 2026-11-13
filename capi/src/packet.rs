//! Packet inspection (`opus_packet_*`) and `opus_pcm_soft_clip`, ported
//! from libopus's opus.c and opus_decoder.c.

use core::ffi::c_int;
use core::slice;

use crate::{
    OPUS_BAD_ARG, OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_MEDIUMBAND, OPUS_BANDWIDTH_NARROWBAND,
    OPUS_BANDWIDTH_SUPERWIDEBAND, OPUS_INVALID_PACKET,
};

/// A parsed packet (libopus `opus_packet_parse_impl`). Offsets are into the
/// parsed buffer.
#[derive(Clone, Copy)]
pub(crate) struct Parsed {
    pub toc: u8,
    pub count: usize,
    pub sizes: [i16; 48],
    pub frames: [usize; 48],
    pub payload_offset: usize,
    /// Bytes the packet takes, padding included (self-delimited packets are
    /// followed by the next stream's).
    pub packet_offset: usize,
    pub padding: usize,
    pub padding_len: usize,
}

pub(crate) fn samples_per_frame(toc: u8, fs: i32) -> i32 {
    if toc & 0x80 != 0 {
        (fs << ((toc >> 3) & 0x3)) / 400
    } else if toc & 0x60 == 0x60 {
        if toc & 0x08 != 0 { fs / 50 } else { fs / 100 }
    } else {
        match (toc >> 3) & 0x3 {
            3 => fs * 60 / 1000,
            s => (fs << s) / 100,
        }
    }
}

pub(crate) fn bandwidth(toc: u8) -> i32 {
    if toc & 0x80 != 0 {
        let bw = OPUS_BANDWIDTH_MEDIUMBAND + ((toc >> 5) & 0x3) as i32;
        if bw == OPUS_BANDWIDTH_MEDIUMBAND { OPUS_BANDWIDTH_NARROWBAND } else { bw }
    } else if toc & 0x60 == 0x60 {
        if toc & 0x10 != 0 { OPUS_BANDWIDTH_FULLBAND } else { OPUS_BANDWIDTH_SUPERWIDEBAND }
    } else {
        OPUS_BANDWIDTH_NARROWBAND + ((toc >> 5) & 0x3) as i32
    }
}

pub(crate) fn nb_frames(packet: &[u8]) -> i32 {
    match packet {
        [] => OPUS_BAD_ARG,
        [toc, ..] if toc & 0x3 == 0 => 1,
        [toc, ..] if toc & 0x3 != 3 => 2,
        [_] => OPUS_INVALID_PACKET,
        [_, n, ..] => (n & 0x3f) as i32,
    }
}

pub(crate) fn nb_samples(packet: &[u8], fs: i32) -> i32 {
    let count = nb_frames(packet);
    if count < 0 {
        return count;
    }
    let samples = count * samples_per_frame(packet[0], fs);
    // No more than 120 ms.
    if samples * 25 > fs * 3 { OPUS_INVALID_PACKET } else { samples }
}

/// Reads a frame length (one or two bytes): (bytes read, size), or `None`.
fn parse_size(data: &[u8]) -> Option<(usize, i16)> {
    match data {
        [] => None,
        [b, ..] if *b < 252 => Some((1, *b as i16)),
        [_] => None,
        [b0, b1, ..] => Some((2, 4 * *b1 as i16 + *b0 as i16)),
    }
}

pub(crate) fn encode_size(size: usize, out: &mut [u8]) -> usize {
    if size < 252 {
        out[0] = size as u8;
        1
    } else {
        out[0] = 252 + (size & 0x3) as u8;
        out[1] = ((size - out[0] as usize) >> 2) as u8;
        2
    }
}

/// libopus `opus_packet_parse_impl`, with its exact validation.
pub(crate) fn parse(packet: &[u8], self_delimited: bool) -> Result<Parsed, c_int> {
    if packet.is_empty() {
        return Err(OPUS_INVALID_PACKET);
    }
    let mut p = Parsed {
        toc: packet[0],
        count: 0,
        sizes: [0; 48],
        frames: [0; 48],
        payload_offset: 0,
        packet_offset: 0,
        padding: 0,
        padding_len: 0,
    };
    let framesize = samples_per_frame(packet[0], 48000);
    let mut pos = 1usize;
    let mut len = packet.len() as i32 - 1;
    let mut last_size = len;
    let mut cbr = false;
    let mut pad = 0i32;
    let count: usize;
    match p.toc & 0x3 {
        0 => count = 1,
        1 => {
            count = 2;
            cbr = true;
            if !self_delimited {
                if len & 1 != 0 {
                    return Err(OPUS_INVALID_PACKET);
                }
                last_size = len / 2;
                p.sizes[0] = last_size as i16;
            }
        }
        2 => {
            count = 2;
            let (bytes, size) = parse_size(&packet[pos..]).ok_or(OPUS_INVALID_PACKET)?;
            len -= bytes as i32;
            if size < 0 || size as i32 > len {
                return Err(OPUS_INVALID_PACKET);
            }
            p.sizes[0] = size;
            pos += bytes;
            last_size = len - size as i32;
        }
        _ => {
            if len < 1 {
                return Err(OPUS_INVALID_PACKET);
            }
            let ch = packet[pos];
            pos += 1;
            count = (ch & 0x3f) as usize;
            if count == 0 || framesize * count as i32 > 5760 {
                return Err(OPUS_INVALID_PACKET);
            }
            len -= 1;
            if ch & 0x40 != 0 {
                loop {
                    if len <= 0 {
                        return Err(OPUS_INVALID_PACKET);
                    }
                    let b = packet[pos] as i32;
                    pos += 1;
                    len -= 1;
                    let tmp = if b == 255 { 254 } else { b };
                    len -= tmp;
                    pad += tmp;
                    if b != 255 {
                        break;
                    }
                }
            }
            if len < 0 {
                return Err(OPUS_INVALID_PACKET);
            }
            cbr = ch & 0x80 == 0;
            if !cbr {
                last_size = len;
                for i in 0..count - 1 {
                    // libopus reads `len` bytes ahead of `data`; past the end
                    // of the packet is the padding it already subtracted.
                    let end = pos + len.max(0) as usize;
                    let (bytes, size) = parse_size(&packet[pos..end]).ok_or(OPUS_INVALID_PACKET)?;
                    len -= bytes as i32;
                    if size as i32 > len {
                        return Err(OPUS_INVALID_PACKET);
                    }
                    p.sizes[i] = size;
                    pos += bytes;
                    last_size -= bytes as i32 + size as i32;
                }
                if last_size < 0 {
                    return Err(OPUS_INVALID_PACKET);
                }
            } else if !self_delimited {
                last_size = len / count as i32;
                if last_size * count as i32 != len {
                    return Err(OPUS_INVALID_PACKET);
                }
                for i in 0..count - 1 {
                    p.sizes[i] = last_size as i16;
                }
            }
        }
    }
    if self_delimited {
        let end = pos + len.max(0) as usize;
        let (bytes, size) = parse_size(&packet[pos..end]).ok_or(OPUS_INVALID_PACKET)?;
        len -= bytes as i32;
        if size as i32 > len {
            return Err(OPUS_INVALID_PACKET);
        }
        p.sizes[count - 1] = size;
        pos += bytes;
        if cbr {
            if size as i32 * count as i32 > len {
                return Err(OPUS_INVALID_PACKET);
            }
            for i in 0..count - 1 {
                p.sizes[i] = size;
            }
        } else if bytes as i32 + size as i32 > last_size {
            return Err(OPUS_INVALID_PACKET);
        }
    } else {
        // The last frame's size is implicit; it may not exceed 1275.
        if last_size > 1275 {
            return Err(OPUS_INVALID_PACKET);
        }
        p.sizes[count - 1] = last_size as i16;
    }
    p.payload_offset = pos;
    for i in 0..count {
        p.frames[i] = pos;
        pos += p.sizes[i] as usize;
    }
    p.padding = pos;
    p.padding_len = pad as usize;
    p.packet_offset = pad as usize + pos;
    p.count = count;
    Ok(p)
}

/// The packet of a C (pointer, length) pair; `None` for a negative length.
pub(crate) unsafe fn packet<'a>(data: *const u8, len: i32) -> Option<&'a [u8]> {
    if len < 0 {
        None
    } else if len == 0 {
        Some(&[])
    } else {
        Some(unsafe { slice::from_raw_parts(data, len as usize) })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_parse(
    data: *const u8,
    len: i32,
    out_toc: *mut u8,
    frames: *mut *const u8,
    size: *mut i16,
    payload_offset: *mut c_int,
) -> c_int {
    let Some(packet) = (unsafe { crate::packet::packet(data, len) }) else {
        return OPUS_BAD_ARG;
    };
    if size.is_null() {
        return OPUS_BAD_ARG;
    }
    match parse(packet, false) {
        Err(e) => e,
        Ok(p) => unsafe {
            crate::set_out(out_toc, p.toc);
            crate::set_out(payload_offset, p.payload_offset as c_int);
            for i in 0..p.count {
                *size.add(i) = p.sizes[i];
                if !frames.is_null() {
                    *frames.add(i) = data.add(p.frames[i]);
                }
            }
            p.count as c_int
        },
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_bandwidth(data: *const u8) -> c_int {
    bandwidth(unsafe { *data })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_samples_per_frame(data: *const u8, fs: i32) -> c_int {
    samples_per_frame(unsafe { *data }, fs)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_nb_channels(data: *const u8) -> c_int {
    if unsafe { *data } & 0x4 != 0 { 2 } else { 1 }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_nb_frames(packet: *const u8, len: i32) -> c_int {
    match unsafe { crate::packet::packet(packet, len) } {
        Some(p) => nb_frames(p),
        None => OPUS_BAD_ARG,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_nb_samples(packet: *const u8, len: i32, fs: i32) -> c_int {
    match unsafe { crate::packet::packet(packet, len) } {
        Some(p) => nb_samples(p, fs),
        None => OPUS_BAD_ARG,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_has_lbrr(packet: *const u8, len: i32) -> c_int {
    let Some(packet) = (unsafe { crate::packet::packet(packet, len) }) else {
        return OPUS_BAD_ARG;
    };
    let Some(&toc) = packet.first() else {
        return OPUS_INVALID_PACKET;
    };
    // CELT-only packets carry no LBRR.
    if toc & 0x80 != 0 {
        return 0;
    }
    let frame_size = samples_per_frame(toc, 48000);
    let frames = if frame_size > 960 { frame_size / 960 } else { 1 };
    let p = match parse(packet, false) {
        Ok(p) => p,
        Err(e) => return e,
    };
    if p.sizes[0] == 0 {
        return 0;
    }
    let first = packet[p.frames[0]];
    let mut lbrr = (first >> (7 - frames)) & 1;
    if toc & 0x4 != 0 {
        lbrr |= (first >> (6 - 2 * frames)) & 1;
    }
    lbrr as c_int
}

/// libopus `opus_pcm_soft_clip_impl`: bends samples beyond ±1 back under it
/// smoothly, carrying the curve across calls in `declip_mem`.
pub(crate) fn soft_clip(x: &mut [f32], n: usize, c: usize, declip_mem: &mut [f32]) {
    if c < 1 || n < 1 {
        return;
    }
    // The non-linearity's domain is [-2, 2].
    for v in x[..n * c].iter_mut() {
        *v = v.clamp(-2.0, 2.0);
    }
    for ch in 0..c {
        let at = |i: usize| i * c + ch;
        let mut a = declip_mem[ch];
        // Continue the previous frame's curve to its first zero crossing.
        for i in 0..n {
            if x[at(i)] * a >= 0.0 {
                break;
            }
            x[at(i)] += a * x[at(i)] * x[at(i)];
        }
        let mut curr = 0;
        let x0 = x[at(0)];
        loop {
            let mut i = curr;
            while i < n && !(x[at(i)] > 1.0 || x[at(i)] < -1.0) {
                i += 1;
            }
            if i == n {
                a = 0.0;
                break;
            }
            let mut peak_pos = i;
            let mut start = i;
            let mut end = i;
            let mut maxval = x[at(i)].abs();
            // The zero crossings around the clipped run.
            while start > 0 && x[at(i)] * x[at(start - 1)] >= 0.0 {
                start -= 1;
            }
            while end < n && x[at(i)] * x[at(end)] >= 0.0 {
                if x[at(end)].abs() > maxval {
                    maxval = x[at(end)].abs();
                    peak_pos = end;
                }
                end += 1;
            }
            // Clipping before the frame's first zero crossing.
            let special = start == 0 && x[at(i)] * x[at(0)] >= 0.0;
            // maxval + a*maxval^2 = 1, boosted by 2^-22.
            a = (maxval - 1.0) / (maxval * maxval);
            a += a * 2.4e-7;
            if x[at(i)] > 0.0 {
                a = -a;
            }
            for j in start..end {
                x[at(j)] += a * x[at(j)] * x[at(j)];
            }
            if special && peak_pos >= 2 {
                // Ramp from the first sample to the peak, to avoid a
                // discontinuity at the frame's start.
                let mut offset = x0 - x[at(0)];
                let delta = offset / peak_pos as f32;
                for j in curr..peak_pos {
                    offset -= delta;
                    x[at(j)] = (x[at(j)] + offset).clamp(-1.0, 1.0);
                }
            }
            curr = end;
            if curr == n {
                break;
            }
        }
        declip_mem[ch] = a;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_pcm_soft_clip(pcm: *mut f32, frame_size: c_int, channels: c_int, softclip_mem: *mut f32) {
    if channels < 1 || frame_size < 1 || pcm.is_null() || softclip_mem.is_null() {
        return;
    }
    let (n, c) = (frame_size as usize, channels as usize);
    unsafe {
        soft_clip(
            slice::from_raw_parts_mut(pcm, n * c),
            n,
            c,
            slice::from_raw_parts_mut(softclip_mem, c),
        )
    }
}
