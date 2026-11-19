//! `opus_repacketizer_*` and `opus_(multistream_)packet_(un)pad` over
//! [`opus_rs::repacketizer`].

use core::ffi::c_int;
use core::slice;

use opus_rs::repacketizer::{self as rs, Repacketizer};

use crate::{OPUS_BAD_ARG, OPUS_INVALID_PACKET, code, ok};

/// libopus's `OpusRepacketizer`. Its frames borrow the packets given to
/// `opus_repacketizer_cat`, which opus_repacketizer.h requires to stay
/// valid until the next `opus_repacketizer_init` or
/// `opus_repacketizer_destroy`; `opus_repacketizer_init` drops every
/// borrow, so `'static` stands for that span.
pub type OpusRepacketizer = Repacketizer<'static>;

/// A byte count, or the code of the error.
fn length(r: Result<usize, &str>) -> c_int {
    r.map_or_else(code, |n| n as c_int)
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_repacketizer_get_size() -> c_int {
    size_of::<OpusRepacketizer>() as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_init(rp: *mut OpusRepacketizer) -> *mut OpusRepacketizer {
    // SAFETY: `rp` points to opus_repacketizer_get_size() writable bytes
    // with malloc's alignment.
    unsafe { rp.write(Repacketizer::new()) };
    rp
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_create() -> *mut OpusRepacketizer {
    // SAFETY: alloc_state returns NULL or room for a repacketizer.
    unsafe {
        let rp = crate::alloc_state(size_of::<OpusRepacketizer>()).cast::<OpusRepacketizer>();
        if rp.is_null() { rp } else { opus_repacketizer_init(rp) }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_destroy(rp: *mut OpusRepacketizer) {
    // SAFETY: `rp` is NULL or came from opus_repacketizer_create.
    unsafe { crate::free_state(rp) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_cat(rp: *mut OpusRepacketizer, data: *const u8, len: i32) -> c_int {
    if len < 1 || data.is_null() {
        return OPUS_INVALID_PACKET;
    }
    // SAFETY: `rp` is an initialized repacketizer and `data` holds `len`
    // bytes that stay valid as `OpusRepacketizer` describes.
    unsafe { ok((*rp).cat(slice::from_raw_parts(data, len as usize))) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_get_nb_frames(rp: *mut OpusRepacketizer) -> c_int {
    // SAFETY: `rp` is an initialized repacketizer.
    unsafe { (*rp).nb_frames() as c_int }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_out_range(
    rp: *mut OpusRepacketizer,
    begin: c_int,
    end: c_int,
    data: *mut u8,
    maxlen: i32,
) -> i32 {
    if begin < 0 || end < 0 || data.is_null() {
        return OPUS_BAD_ARG;
    }
    // SAFETY: `rp` is an initialized repacketizer and `data` holds
    // `maxlen` bytes.
    unsafe {
        let out = slice::from_raw_parts_mut(data, maxlen.max(0) as usize);
        length((*rp).out_range(begin as usize, end as usize, out))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_repacketizer_out(rp: *mut OpusRepacketizer, data: *mut u8, maxlen: i32) -> i32 {
    // SAFETY: `rp` is an initialized repacketizer.
    unsafe { opus_repacketizer_out_range(rp, 0, (*rp).nb_frames() as c_int, data, maxlen) }
}

/// The `len` bytes at `data` padded in place to `new_len` with `pad`, as
/// libopus's pad functions check their lengths.
unsafe fn pad_with(data: *mut u8, len: i32, new_len: i32, pad: impl FnOnce(&[u8], &mut [u8]) -> Result<(), &'static str>) -> c_int {
    if len < 1 || len > new_len || data.is_null() {
        return OPUS_BAD_ARG;
    }
    // SAFETY: `data` holds `new_len` bytes, the packet its first `len`.
    unsafe {
        let packet = slice::from_raw_parts(data, len as usize).to_vec();
        ok(pad(&packet, slice::from_raw_parts_mut(data, new_len as usize)))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_pad(data: *mut u8, len: i32, new_len: i32) -> c_int {
    // SAFETY: `data` holds `new_len` bytes (opus.h).
    unsafe { pad_with(data, len, new_len, rs::pad) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_packet_pad(data: *mut u8, len: i32, new_len: i32, nb_streams: c_int) -> c_int {
    // SAFETY: `data` holds `new_len` bytes (opus_multistream.h).
    unsafe { pad_with(data, len, new_len, |p, out| rs::multistream_pad(p, nb_streams.max(1) as usize, out)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_packet_unpad(data: *mut u8, len: i32) -> i32 {
    if len < 1 || data.is_null() {
        return OPUS_BAD_ARG;
    }
    // SAFETY: `data` holds `len` bytes.
    length(rs::unpad(unsafe { slice::from_raw_parts_mut(data, len as usize) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_packet_unpad(data: *mut u8, len: i32, nb_streams: c_int) -> i32 {
    if len < 1 || data.is_null() {
        return OPUS_BAD_ARG;
    }
    // SAFETY: `data` holds `len` bytes.
    length(rs::multistream_unpad(unsafe { slice::from_raw_parts_mut(data, len as usize) }, nb_streams.max(0) as usize))
}
