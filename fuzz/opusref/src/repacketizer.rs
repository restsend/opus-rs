//! FFI to the reference's repacketizer (`src/repacketizer.c`) and packet
//! extensions (`src/extensions.c`).
use core::ffi::{c_int, c_uchar, c_void};

pub type OpusRepacketizer = c_void;

/// libopus `opus_extension_data` (src/opus_private.h).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OpusExtensionData {
    pub id: c_int,
    pub frame: c_int,
    pub data: *const c_uchar,
    pub len: i32,
}

unsafe extern "C" {
    pub fn opus_packet_extensions_parse(
        data: *const c_uchar,
        len: i32,
        extensions: *mut OpusExtensionData,
        nb_extensions: *mut i32,
        nb_frames: c_int,
    ) -> i32;
    pub fn opus_packet_extensions_generate(
        data: *mut c_uchar,
        len: i32,
        extensions: *const OpusExtensionData,
        nb_extensions: i32,
        nb_frames: c_int,
        pad: c_int,
    ) -> i32;
    pub fn opus_packet_parse_impl(
        data: *const c_uchar,
        len: i32,
        self_delimited: c_int,
        out_toc: *mut c_uchar,
        frames: *mut *const c_uchar,
        size: *mut i16,
        payload_offset: *mut c_int,
        packet_offset: *mut i32,
        padding: *mut *const c_uchar,
        padding_len: *mut i32,
    ) -> c_int;
}

unsafe extern "C" {
    pub fn opus_repacketizer_create() -> *mut OpusRepacketizer;
    pub fn opus_repacketizer_destroy(rp: *mut OpusRepacketizer);
    pub fn opus_repacketizer_cat(
        rp: *mut OpusRepacketizer,
        data: *const c_uchar,
        len: i32,
    ) -> c_int;
    pub fn opus_repacketizer_out_range_impl(
        rp: *mut OpusRepacketizer,
        begin: c_int,
        end: c_int,
        data: *mut c_uchar,
        maxlen: i32,
        self_delimited: c_int,
        pad: c_int,
        extensions: *const OpusExtensionData,
        nb_extensions: c_int,
    ) -> i32;
    pub fn opus_repacketizer_init(rp: *mut OpusRepacketizer) -> *mut OpusRepacketizer;
    pub fn opus_repacketizer_get_nb_frames(rp: *mut OpusRepacketizer) -> c_int;
    pub fn opus_repacketizer_out_range(
        rp: *mut OpusRepacketizer,
        begin: c_int,
        end: c_int,
        data: *mut c_uchar,
        maxlen: i32,
    ) -> i32;
    pub fn opus_repacketizer_out(rp: *mut OpusRepacketizer, data: *mut c_uchar, maxlen: i32) -> i32;
    pub fn opus_packet_pad(data: *mut c_uchar, len: i32, new_len: i32) -> c_int;
    pub fn opus_packet_unpad(data: *mut c_uchar, len: i32) -> i32;
    pub fn opus_multistream_packet_pad(
        data: *mut c_uchar,
        len: i32,
        new_len: i32,
        nb_streams: c_int,
    ) -> c_int;
    pub fn opus_multistream_packet_unpad(data: *mut c_uchar, len: i32, nb_streams: c_int) -> i32;
    pub fn opus_packet_pad_impl(
        data: *mut c_uchar,
        len: i32,
        new_len: i32,
        pad: c_int,
        extensions: *const OpusExtensionData,
        nb_extensions: c_int,
    ) -> i32;
}

/// The extensions of `data` (a padding of a packet with `nb_frames`
/// frames) by `opus_packet_extensions_parse`: (id, frame, payload) or the
/// error code.
pub fn extensions_parse(data: &[u8], nb_frames: usize) -> Result<Vec<(u8, usize, Vec<u8>)>, i32> {
    let mut exts = vec![
        OpusExtensionData {
            id: 0,
            frame: 0,
            data: core::ptr::null(),
            len: 0
        };
        512
    ];
    let mut n = exts.len() as i32;
    let ret = unsafe {
        opus_packet_extensions_parse(
            data.as_ptr(),
            data.len() as i32,
            exts.as_mut_ptr(),
            &mut n,
            nb_frames as c_int,
        )
    };
    if ret < 0 {
        return Err(ret);
    }
    Ok(exts[..n as usize]
        .iter()
        .map(|e| {
            let payload = if e.len == 0 {
                Vec::new()
            } else {
                unsafe { core::slice::from_raw_parts(e.data, e.len as usize) }.to_vec()
            };
            (e.id as u8, e.frame as usize, payload)
        })
        .collect())
}

