//! `opus_packet_*` and `opus_pcm_soft_clip` over [`opus_rs::packet`].

use core::ffi::c_int;
use core::slice;

use opus_rs::packet as rs;

use crate::{OPUS_BAD_ARG, code, packet};

/// An `opus_int32` result or error code.
fn int(r: Result<usize, &str>) -> c_int {
    r.map_or_else(code, |n| n as c_int)
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
    if size.is_null() {
        return OPUS_BAD_ARG;
    }
    // SAFETY: `data` holds `len` bytes, `size` (and `frames`, if not NULL)
    // has room for 48 entries, `out_toc` and `payload_offset` are NULL or
    // valid for a write (opus.h).
    unsafe {
        let p = match packet(data, len).map(rs::parse) {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => return code(e),
            Err(e) => return e,
        };
        crate::set_out(out_toc, p.toc);
        crate::set_out(payload_offset, p.payload_offset as c_int);
        for (i, f) in p.frames().iter().enumerate() {
            *size.add(i) = f.len() as i16;
            if !frames.is_null() {
                *frames.add(i) = f.as_ptr();
            }
        }
        p.count as c_int
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_bandwidth(data: *const u8) -> c_int {
    // SAFETY: `data` points to a ToC byte (opus.h).
    rs::bandwidth(unsafe { *data }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_samples_per_frame(data: *const u8, fs: i32) -> c_int {
    // SAFETY: `data` points to a ToC byte (opus.h).
    rs::samples_per_frame(unsafe { *data }, fs)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_nb_channels(data: *const u8) -> c_int {
    // SAFETY: `data` points to a ToC byte (opus.h).
    rs::nb_channels(unsafe { *data }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_nb_frames(data: *const u8, len: i32) -> c_int {
    // SAFETY: `data` holds `len` bytes (opus.h).
    unsafe { packet(data, len) }.map_or_else(|e| e, |p| int(rs::nb_frames(p)))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_get_nb_samples(data: *const u8, len: i32, fs: i32) -> c_int {
    // SAFETY: `data` holds `len` bytes (opus.h).
    unsafe { packet(data, len) }.map_or_else(|e| e, |p| int(rs::nb_samples(p, fs)))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_has_lbrr(data: *const u8, len: i32) -> c_int {
    // SAFETY: `data` holds `len` bytes (opus.h).
    unsafe { packet(data, len) }.map_or_else(|e| e, |p| int(rs::has_lbrr(p).map(usize::from)))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_pcm_soft_clip(pcm: *mut f32, frame_size: c_int, channels: c_int, softclip_mem: *mut f32) {
    if channels < 1 || frame_size < 1 || pcm.is_null() || softclip_mem.is_null() {
        return;
    }
    let (n, c) = (frame_size as usize, channels as usize);
    // SAFETY: `pcm` holds `frame_size` samples of `channels` and
    // `softclip_mem` one value per channel (opus.h).
    let _ = unsafe {
        rs::soft_clip(slice::from_raw_parts_mut(pcm, n * c), n, c, slice::from_raw_parts_mut(softclip_mem, c))
    };
}
