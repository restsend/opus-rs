//! FFI to the reference's DRED API (feature `dnn`): the DRED decoder and
//! parser of `src/opus_decoder.c`, the DRED encoder control and the packet
//! extension functions of `src/extensions.c`.
use core::ffi::{c_int, c_uchar, c_void};

pub use crate::repacketizer::*;
use crate::{OpusDecoder, OpusEncoder};

pub type OpusDREDDecoder = c_void;

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
    pub fn opus_decoder_dred_decode24(
        st: *mut OpusDecoder,
        dred: *const OpusDRED,
        dred_offset: i32,
        pcm: *mut i32,
        frame_size: i32,
    ) -> c_int;
    pub fn opus_decoder_dred_decode_float(
        st: *mut OpusDecoder,
        dred: *const OpusDRED,
        dred_offset: i32,
        pcm: *mut f32,
        frame_size: i32,
    ) -> c_int;

}

/// `OPUS_SET_DRED_DURATION_REQUEST` and `OPUS_GET_DRED_DURATION_REQUEST`
/// (include/opus_defines.h).
pub use crate::{GET_DRED_DURATION, SET_DRED_DURATION};

/// Sets the DRED duration (in 10 ms frames) of a reference encoder.
pub unsafe fn set_dred_duration(enc: *mut OpusEncoder, frames: i32) -> c_int {
    unsafe { crate::opus_encoder_ctl(enc, SET_DRED_DURATION, frames) }
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
