//! FFI to the reference's DRED API (feature `dnn`): the DRED decoder and
//! parser of `src/opus_decoder.c`, the DRED encoder control and the packet
//! extension functions of `src/extensions.c`.
use core::ffi::{c_int, c_uchar, c_void};

use crate::{OpusDecoder, OpusEncoder};

pub type OpusDREDDecoder = c_void;
pub type OpusRepacketizer = c_void;

/// `DRED_NUM_REDUNDANCY_FRAMES` (dnn/dred_config.h).
pub const DRED_NUM_REDUNDANCY_FRAMES: usize = 52;
/// `DRED_NUM_FEATURES`, `DRED_LATENT_DIM`, `DRED_STATE_DIM`
/// (dnn/dred_rdovae_constants.h).
pub const DRED_NUM_FEATURES: usize = 20;
pub const DRED_LATENT_DIM: usize = 25;
pub const DRED_STATE_DIM: usize = 50;

/// libopus `struct OpusDRED` (dnn/dred_decoder.h). `opus_dred_get_size`
/// checks the layout.
#[repr(C)]
#[derive(Clone)]
pub struct OpusDRED {
    pub fec_features: [f32; 2 * DRED_NUM_REDUNDANCY_FRAMES * DRED_NUM_FEATURES],
    pub state: [f32; DRED_STATE_DIM],
    pub latents: [f32; (DRED_NUM_REDUNDANCY_FRAMES / 2) * (DRED_LATENT_DIM + 1)],
    pub nb_latents: c_int,
    pub process_stage: c_int,
    pub dred_offset: c_int,
}

impl OpusDRED {
    /// All zero.
    pub fn zeroed() -> Box<Self> {
        Box::new(OpusDRED {
            fec_features: [0.0; 2 * DRED_NUM_REDUNDANCY_FRAMES * DRED_NUM_FEATURES],
            state: [0.0; DRED_STATE_DIM],
            latents: [0.0; (DRED_NUM_REDUNDANCY_FRAMES / 2) * (DRED_LATENT_DIM + 1)],
            nb_latents: 0,
            process_stage: 0,
            dred_offset: 0,
        })
    }
}

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
    pub fn opus_dred_decoder_create(error: *mut c_int) -> *mut OpusDREDDecoder;
    pub fn opus_dred_decoder_destroy(dec: *mut OpusDREDDecoder);
    pub fn opus_dred_get_size() -> c_int;
    pub fn opus_dred_parse(
        dred_dec: *mut OpusDREDDecoder,
        dred: *mut OpusDRED,
        data: *const c_uchar,
        len: i32,
        max_dred_samples: i32,
        sampling_rate: i32,
        dred_end: *mut c_int,
        defer_processing: c_int,
    ) -> c_int;
    pub fn opus_dred_process(
        dred_dec: *mut OpusDREDDecoder,
        src: *const OpusDRED,
        dst: *mut OpusDRED,
    ) -> c_int;
    pub fn opus_decoder_dred_decode(
        st: *mut OpusDecoder,
        dred: *const OpusDRED,
        dred_offset: i32,
        pcm: *mut i16,
        frame_size: i32,
    ) -> c_int;
    pub fn opus_decoder_dred_decode_float(
        st: *mut OpusDecoder,
        dred: *const OpusDRED,
        dred_offset: i32,
        pcm: *mut f32,
        frame_size: i32,
    ) -> c_int;

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
    pub fn opus_packet_pad_impl(
        data: *mut c_uchar,
        len: i32,
        new_len: i32,
        pad: c_int,
        extensions: *const OpusExtensionData,
        nb_extensions: c_int,
    ) -> i32;
}

/// `OPUS_SET_DRED_DURATION_REQUEST` and `OPUS_GET_DRED_DURATION_REQUEST`
/// (include/opus_defines.h).
pub const SET_DRED_DURATION: c_int = 4050;
pub const GET_DRED_DURATION: c_int = 4051;

/// Sets the DRED duration (in 10 ms frames) of a reference encoder.
pub unsafe fn set_dred_duration(enc: *mut OpusEncoder, frames: i32) -> c_int {
    unsafe { crate::opus_encoder_ctl(enc, SET_DRED_DURATION, frames) }
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

/// A reference DRED decoder with the built-in model.
pub struct RefDredDecoder(pub *mut OpusDREDDecoder);

impl RefDredDecoder {
    pub fn new() -> Self {
        let mut err = 0;
        let d = unsafe { opus_dred_decoder_create(&mut err) };
        assert!(err == 0 && !d.is_null(), "opus_dred_decoder_create: {err}");
        RefDredDecoder(d)
    }

    /// `opus_dred_parse` into `dred`: (return value, dred_end).
    pub fn parse(
        &self,
        dred: &mut OpusDRED,
        packet: &[u8],
        max_dred_samples: i32,
        fs: i32,
        defer: bool,
    ) -> (i32, i32) {
        let mut end = -1;
        let ret = unsafe {
            opus_dred_parse(
                self.0,
                dred,
                packet.as_ptr(),
                packet.len() as i32,
                max_dred_samples,
                fs,
                &mut end,
                defer as c_int,
            )
        };
        (ret, end)
    }

    /// `opus_dred_process` in place.
    pub fn process(&self, dred: &mut OpusDRED) -> i32 {
        let p: *mut OpusDRED = dred;
        unsafe { opus_dred_process(self.0, p, p) }
    }
}

impl Default for RefDredDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RefDredDecoder {
    fn drop(&mut self) {
        unsafe { opus_dred_decoder_destroy(self.0) }
    }
}
