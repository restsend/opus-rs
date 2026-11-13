//! `opus_repacketizer_*` and `opus_packet_(un)pad`, after libopus's
//! repacketizer.c.
//!
//! libopus 1.6 carries the extensions it finds in packet padding over to the
//! packets it builds; this port drops them (opus-rs neither writes nor reads
//! extensions), so it only ever pads with zeros.

use core::ffi::c_int;
use core::ptr;
use core::slice;

use crate::packet::{self, encode_size, parse};
use crate::{OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_INVALID_PACKET, OPUS_OK};

/// libopus's `OpusRepacketizer`. The frames point into the packets given to
/// `opus_repacketizer_cat`, which must outlive it.
#[repr(C)]
pub struct OpusRepacketizer {
    toc: u8,
    nb_frames: c_int,
    frames: [*const u8; 48],
    len: [i16; 48],
    framesize: c_int,
}

impl OpusRepacketizer {
    pub(crate) const fn new() -> Self {
        Self { toc: 0, nb_frames: 0, frames: [ptr::null(); 48], len: [0; 48], framesize: 0 }
    }

    /// libopus `opus_repacketizer_cat_impl`.
    pub(crate) unsafe fn cat(&mut self, data: *const u8, len: i32, self_delimited: bool) -> c_int {
        if len < 1 {
            return OPUS_INVALID_PACKET;
        }
        let packet = unsafe { slice::from_raw_parts(data, len as usize) };
        if self.nb_frames == 0 {
            self.toc = packet[0];
            self.framesize = packet::samples_per_frame(packet[0], 8000);
        } else if self.toc & 0xfc != packet[0] & 0xfc {
            return OPUS_INVALID_PACKET;
        }
        let curr = packet::nb_frames(packet);
        if curr < 1 {
            return OPUS_INVALID_PACKET;
        }
        // At most 120 ms.
        if (curr + self.nb_frames) * self.framesize > 960 {
            return OPUS_INVALID_PACKET;
        }
        let p = match parse(packet, self_delimited) {
            Ok(p) => p,
            Err(e) => return e,
        };
        for i in 0..p.count {
            let n = self.nb_frames as usize + i;
            self.frames[n] = unsafe { data.add(p.frames[i]) };
            self.len[n] = p.sizes[i];
        }
        self.nb_frames += p.count as c_int;
        OPUS_OK
    }

    /// libopus `opus_repacketizer_out_range_impl`, without extensions.
    pub(crate) unsafe fn out_range(
        &self,
        begin: c_int,
        end: c_int,
        data: *mut u8,
        maxlen: i32,
        self_delimited: bool,
        pad: bool,
    ) -> i32 {
        if begin < 0 || begin >= end || end > self.nb_frames {
            return OPUS_BAD_ARG;
        }
        let count = (end - begin) as usize;
        let len = &self.len[begin as usize..end as usize];
        let frames = &self.frames[begin as usize..end as usize];
        let sd_bytes = if self_delimited { 1 + (len[count - 1] >= 252) as i32 } else { 0 };
        let mut tot_size = sd_bytes;
        let mut ptr = data;
        let toc = self.toc & 0xfc;
        // `ptr` and `data` stay within the caller's `maxlen` bytes: every
        // write below is preceded by the size check that covers it.
        let put = |ptr: &mut *mut u8, b: u8| unsafe {
            **ptr = b;
            *ptr = ptr.add(1);
        };
        let put_size = |ptr: &mut *mut u8, size: i16| unsafe {
            let mut tmp = [0u8; 2];
            let n = encode_size(size as usize, &mut tmp);
            ptr::copy_nonoverlapping(tmp.as_ptr(), *ptr, n);
            *ptr = ptr.add(n);
        };
        if count == 1 {
            tot_size += len[0] as i32 + 1;
            if tot_size > maxlen {
                return OPUS_BUFFER_TOO_SMALL;
            }
            put(&mut ptr, toc);
        } else if count == 2 {
            if len[1] == len[0] {
                tot_size += 2 * len[0] as i32 + 1;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                put(&mut ptr, toc | 0x1);
            } else {
                tot_size += len[0] as i32 + len[1] as i32 + 2 + (len[0] >= 252) as i32;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                put(&mut ptr, toc | 0x2);
                put_size(&mut ptr, len[0]);
            }
        }
        if count > 2 || (pad && tot_size < maxlen) {
            // Code 3: start over, for the padding case.
            ptr = data;
            tot_size = sd_bytes;
            let vbr = len[1..].iter().any(|&l| l != len[0]);
            if vbr {
                tot_size += 2;
                for &l in &len[..count - 1] {
                    tot_size += 1 + (l >= 252) as i32 + l as i32;
                }
                tot_size += len[count - 1] as i32;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                put(&mut ptr, toc | 0x3);
                put(&mut ptr, count as u8 | 0x80);
            } else {
                tot_size += count as i32 * len[0] as i32 + 2;
                if tot_size > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                put(&mut ptr, toc | 0x3);
                put(&mut ptr, count as u8);
            }
            let pad_amount = if pad { maxlen - tot_size } else { 0 };
            if pad_amount != 0 {
                unsafe { *data.add(1) |= 0x40 };
                let nb_255s = (pad_amount - 1) / 255;
                if tot_size + nb_255s + 1 > maxlen {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                for _ in 0..nb_255s {
                    put(&mut ptr, 255);
                }
                put(&mut ptr, (pad_amount - 255 * nb_255s - 1) as u8);
                tot_size += pad_amount;
            }
            if vbr {
                for &l in &len[..count - 1] {
                    put_size(&mut ptr, l);
                }
            }
        }
        if self_delimited {
            put_size(&mut ptr, len[count - 1]);
        }
        for (&frame, &l) in frames.iter().zip(len) {
            // A move: opus_packet_(un)pad repacketize in place.
            unsafe {
                ptr::copy(frame, ptr, l as usize);
                ptr = ptr.add(l as usize);
            }
        }
        if pad {
            // Zero padding.
            let end = unsafe { data.add(maxlen as usize) };
            while ptr < end {
                put(&mut ptr, 0);
            }
        }
        tot_size
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_repacketizer_get_size() -> c_int {
    core::mem::size_of::<OpusRepacketizer>() as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_init(rp: *mut OpusRepacketizer) -> *mut OpusRepacketizer {
    unsafe { rp.write(OpusRepacketizer::new()) };
    rp
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_create() -> *mut OpusRepacketizer {
    unsafe {
        let rp = crate::alloc_state(core::mem::size_of::<OpusRepacketizer>()) as *mut OpusRepacketizer;
        if rp.is_null() { rp } else { opus_repacketizer_init(rp) }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_destroy(rp: *mut OpusRepacketizer) {
    unsafe { crate::free_state(rp) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_cat(rp: *mut OpusRepacketizer, data: *const u8, len: i32) -> c_int {
    unsafe { (*rp).cat(data, len, false) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_get_nb_frames(rp: *mut OpusRepacketizer) -> c_int {
    unsafe { (*rp).nb_frames }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_out_range(
    rp: *mut OpusRepacketizer,
    begin: c_int,
    end: c_int,
    data: *mut u8,
    maxlen: i32,
) -> i32 {
    unsafe { (*rp).out_range(begin, end, data, maxlen, false, false) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_out(rp: *mut OpusRepacketizer, data: *mut u8, maxlen: i32) -> i32 {
    unsafe { (*rp).out_range(0, (*rp).nb_frames, data, maxlen, false, false) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_pad(data: *mut u8, len: i32, new_len: i32) -> c_int {
    if len < 1 {
        return OPUS_BAD_ARG;
    }
    if len == new_len {
        return OPUS_OK;
    } else if len > new_len {
        return OPUS_BAD_ARG;
    }
    // Repacketize from a copy, so the padded packet can overwrite `data`.
    let copy = unsafe { slice::from_raw_parts(data, len as usize) }.to_vec();
    let mut rp = OpusRepacketizer::new();
    let ret = unsafe { rp.cat(copy.as_ptr(), len, false) };
    if ret != OPUS_OK {
        return ret;
    }
    let ret = unsafe { rp.out_range(0, rp.nb_frames, data, new_len, false, true) };
    if ret > 0 { OPUS_OK } else { ret }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_unpad(data: *mut u8, len: i32) -> i32 {
    if len < 1 {
        return OPUS_BAD_ARG;
    }
    let mut rp = OpusRepacketizer::new();
    let ret = unsafe { rp.cat(data, len, false) };
    if ret < 0 {
        return ret;
    }
    unsafe { rp.out_range(0, rp.nb_frames, data, len, false, false) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_packet_pad(mut data: *mut u8, mut len: i32, new_len: i32, nb_streams: c_int) -> c_int {
    if len < 1 {
        return OPUS_BAD_ARG;
    }
    if len == new_len {
        return OPUS_OK;
    } else if len > new_len {
        return OPUS_BAD_ARG;
    }
    let amount = new_len - len;
    // Pad the last stream's packet.
    for _ in 0..nb_streams - 1 {
        if len <= 0 {
            return OPUS_INVALID_PACKET;
        }
        let p = match parse(unsafe { slice::from_raw_parts(data, len as usize) }, true) {
            Ok(p) => p,
            Err(e) => return e,
        };
        data = unsafe { data.add(p.packet_offset) };
        len -= p.packet_offset as i32;
    }
    unsafe { opus_packet_pad(data, len, len + amount) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_packet_unpad(mut data: *mut u8, mut len: i32, nb_streams: c_int) -> i32 {
    if len < 1 {
        return OPUS_BAD_ARG;
    }
    let mut dst = data;
    let mut dst_len = 0;
    for s in 0..nb_streams {
        let self_delimited = s != nb_streams - 1;
        if len <= 0 {
            return OPUS_INVALID_PACKET;
        }
        let p = match parse(unsafe { slice::from_raw_parts(data, len as usize) }, self_delimited) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let mut rp = OpusRepacketizer::new();
        let ret = unsafe { rp.cat(data, p.packet_offset as i32, self_delimited) };
        if ret < 0 {
            return ret;
        }
        let ret = unsafe { rp.out_range(0, rp.nb_frames, dst, len, self_delimited, false) };
        if ret < 0 {
            return ret;
        }
        dst_len += ret;
        unsafe {
            dst = dst.add(ret as usize);
            data = data.add(p.packet_offset);
        }
        len -= p.packet_offset as i32;
    }
    dst_len
}

/// The standard (not self-delimited) form of a self-delimited packet, as
/// opus-rs decodes, written to `out`; its length.
pub(crate) fn undelimit(packet: &[u8], out: &mut [u8]) -> Result<usize, c_int> {
    let mut rp = OpusRepacketizer::new();
    let ret = unsafe { rp.cat(packet.as_ptr(), packet.len() as i32, true) };
    if ret < 0 {
        return Err(ret);
    }
    let ret = unsafe { rp.out_range(0, rp.nb_frames, out.as_mut_ptr(), out.len() as i32, false, false) };
    if ret < 0 { Err(ret) } else { Ok(ret as usize) }
}
