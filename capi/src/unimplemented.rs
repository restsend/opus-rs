//! The parts of the libopus API opus-rs has no counterpart for yet:
//! multistream and projection encoding, and DRED. They answer like a libopus
//! built without them: `OPUS_UNIMPLEMENTED`, a NULL state, a size of 0.

use core::ffi::{c_int, c_void};
use core::ptr;

use crate::{OPUS_UNIMPLEMENTED, set_out};

/// Functions returning `OPUS_UNIMPLEMENTED`.
macro_rules! unimplemented_fns {
    ($( fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty; )*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name($(_: $ty),*) -> $ret {
            OPUS_UNIMPLEMENTED as $ret
        }
    )*};
}

/// Constructors: NULL, with `OPUS_UNIMPLEMENTED` in `*error`, the last
/// parameter, which the macro adds.
macro_rules! unimplemented_create {
    ($( fn $name:ident($($arg:ident: $ty:ty),* $(,)?); )*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name($(_: $ty,)* error: *mut c_int) -> *mut c_void {
            unsafe { set_out(error, OPUS_UNIMPLEMENTED) };
            ptr::null_mut()
        }
    )*};
}

/// Sizes: 0, as libopus reports for invalid arguments.
macro_rules! zero_size {
    ($( fn $name:ident($($arg:ident: $ty:ty),* $(,)?); )*) => {$(
        #[unsafe(no_mangle)]
        pub extern "C" fn $name($(_: $ty),*) -> i32 {
            0
        }
    )*};
}

/// Destructors of states the constructors never hand out.
macro_rules! free_fns {
    ($( fn $name:ident; )*) => {$(
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(st: *mut c_void) {
            unsafe { crate::free_state(st) }
        }
    )*};
}

type P = *mut c_void;

zero_size! {
    fn opus_multistream_encoder_get_size(streams: c_int, coupled_streams: c_int);
    fn opus_multistream_surround_encoder_get_size(channels: c_int, mapping_family: c_int);
    fn opus_projection_ambisonics_encoder_get_size(channels: c_int, mapping_family: c_int);
    fn opus_projection_decoder_get_size(channels: c_int, streams: c_int, coupled_streams: c_int);
    fn opus_dred_decoder_get_size();
    fn opus_dred_get_size();
}

unimplemented_create! {
    fn opus_multistream_encoder_create(fs: i32, channels: c_int, streams: c_int, coupled_streams: c_int, mapping: P, application: c_int);
    fn opus_multistream_surround_encoder_create(fs: i32, channels: c_int, mapping_family: c_int, streams: P, coupled_streams: P, mapping: P, application: c_int);
    fn opus_projection_ambisonics_encoder_create(fs: i32, channels: c_int, mapping_family: c_int, streams: P, coupled_streams: P, application: c_int);
    fn opus_projection_decoder_create(fs: i32, channels: c_int, streams: c_int, coupled_streams: c_int, demixing_matrix: P, demixing_matrix_size: i32);
    fn opus_dred_decoder_create();
    fn opus_dred_alloc();
}

unimplemented_fns! {
    fn opus_multistream_encoder_init(st: P, fs: i32, channels: c_int, streams: c_int, coupled_streams: c_int, mapping: P, application: c_int) -> c_int;
    fn opus_multistream_surround_encoder_init(st: P, fs: i32, channels: c_int, mapping_family: c_int, streams: P, coupled_streams: P, mapping: P, application: c_int) -> c_int;
    fn opus_multistream_encode(st: P, pcm: P, frame_size: c_int, data: P, max_data_bytes: i32) -> c_int;
    fn opus_multistream_encode24(st: P, pcm: P, frame_size: c_int, data: P, max_data_bytes: i32) -> c_int;
    fn opus_multistream_encode_float(st: P, pcm: P, frame_size: c_int, data: P, max_data_bytes: i32) -> c_int;
    fn opus_projection_ambisonics_encoder_init(st: P, fs: i32, channels: c_int, mapping_family: c_int, streams: P, coupled_streams: P, application: c_int) -> c_int;
    fn opus_projection_encode(st: P, pcm: P, frame_size: c_int, data: P, max_data_bytes: i32) -> c_int;
    fn opus_projection_encode24(st: P, pcm: P, frame_size: c_int, data: P, max_data_bytes: i32) -> c_int;
    fn opus_projection_encode_float(st: P, pcm: P, frame_size: c_int, data: P, max_data_bytes: i32) -> c_int;
    fn opus_projection_decoder_init(st: P, fs: i32, channels: c_int, streams: c_int, coupled_streams: c_int, demixing_matrix: P, demixing_matrix_size: i32) -> c_int;
    fn opus_projection_decode(st: P, data: P, len: i32, pcm: P, frame_size: c_int, decode_fec: c_int) -> c_int;
    fn opus_projection_decode24(st: P, data: P, len: i32, pcm: P, frame_size: c_int, decode_fec: c_int) -> c_int;
    fn opus_projection_decode_float(st: P, data: P, len: i32, pcm: P, frame_size: c_int, decode_fec: c_int) -> c_int;
    fn opus_dred_decoder_init(dec: P) -> c_int;
    fn opus_dred_parse(dred_dec: P, dred: P, data: P, len: i32, max_dred_samples: i32, sampling_rate: i32, dred_end: P, defer_processing: c_int) -> c_int;
    fn opus_dred_process(dred_dec: P, src: P, dst: P) -> c_int;
    fn opus_decoder_dred_decode(st: P, dred: P, dred_offset: i32, pcm: P, frame_size: i32) -> c_int;
    fn opus_decoder_dred_decode24(st: P, dred: P, dred_offset: i32, pcm: P, frame_size: i32) -> c_int;
    fn opus_decoder_dred_decode_float(st: P, dred: P, dred_offset: i32, pcm: P, frame_size: i32) -> c_int;
}

free_fns! {
    fn opus_multistream_encoder_destroy;
    fn opus_projection_encoder_destroy;
    fn opus_projection_decoder_destroy;
    fn opus_dred_decoder_destroy;
    fn opus_dred_free;
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encoder_ctl(_st: P, _request: c_int, _args: ...) -> c_int {
    OPUS_UNIMPLEMENTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_encoder_ctl(_st: P, _request: c_int, _args: ...) -> c_int {
    OPUS_UNIMPLEMENTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decoder_ctl(_st: P, _request: c_int, _args: ...) -> c_int {
    OPUS_UNIMPLEMENTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_dred_decoder_ctl(_st: P, _request: c_int, _args: ...) -> c_int {
    OPUS_UNIMPLEMENTED
}
