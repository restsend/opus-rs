//! The DRED calls of opus.h over opus-rs's `dred` feature (libopus
//! `--enable-dred`).

use core::ffi::c_int;
use core::ptr;

use opus_rs::dred::{OpusDred, OpusDredDecoder};

use crate::decoder::OpusDecoder;
use crate::{OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, OPUS_UNIMPLEMENTED, code, guard, packet, set_out};

#[unsafe(no_mangle)]
pub extern "C" fn opus_dred_decoder_get_size() -> c_int {
    crate::align_size(size_of::<OpusDredDecoder>()) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_dred_decoder_init(dec: *mut OpusDredDecoder) -> c_int {
    match OpusDredDecoder::new() {
        Ok(d) => {
            // SAFETY: `dec` points to opus_dred_decoder_get_size() writable
            // bytes with malloc's alignment.
            unsafe { dec.write(d) };
            OPUS_OK
        }
        Err(_) => OPUS_UNIMPLEMENTED,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_dred_decoder_create(error: *mut c_int) -> *mut OpusDredDecoder {
    // SAFETY: `create` hands `init` a block of the size it asks for; the
    // arguments are the caller's, as the header requires.
    unsafe { crate::create(opus_dred_decoder_get_size(), error, |dec| opus_dred_decoder_init(dec)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_dred_decoder_destroy(dec: *mut OpusDredDecoder) {
    // SAFETY: `dec` is NULL or came from opus_dred_decoder_create.
    unsafe { crate::free_state(dec) }
}

/// libopus's DRED decoder has requests only for a weights file build.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_dred_decoder_ctl(_dec: *mut OpusDredDecoder, _request: c_int, _args: ...) -> c_int {
    OPUS_UNIMPLEMENTED
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_dred_get_size() -> c_int {
    crate::align_size(size_of::<OpusDred>()) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_dred_alloc(error: *mut c_int) -> *mut OpusDred {
    // SAFETY: `create` hands the closure room for an OpusDred.
    unsafe {
        crate::create(opus_dred_get_size(), error, |dred: *mut OpusDred| {
            dred.write(OpusDred::default());
            OPUS_OK
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_dred_free(dred: *mut OpusDred) {
    // SAFETY: `dred` is NULL or came from opus_dred_alloc.
    unsafe { crate::free_state(dred) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_dred_parse(
    dred_dec: *mut OpusDredDecoder,
    dred: *mut OpusDred,
    data: *const u8,
    len: i32,
    max_dred_samples: i32,
    sampling_rate: i32,
    dred_end: *mut c_int,
    defer_processing: c_int,
) -> c_int {
    // SAFETY: `dred_dec` and `dred` are initialized, `data` holds `len`
    // bytes and `dred_end` is NULL or valid for a write (opus.h).
    unsafe {
        let data = match packet(data, len) {
            Ok(d) => d,
            Err(e) => return e,
        };
        let (dec, dred) = (&*dred_dec, &mut *dred);
        guard(OPUS_INTERNAL_ERROR, || {
            match dec.parse(dred, data, max_dred_samples, sampling_rate, defer_processing != 0) {
                Ok((samples, end)) => {
                    set_out(dred_end, end);
                    samples
                }
                Err(e) => code(e),
            }
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_dred_process(dred_dec: *mut OpusDredDecoder, src: *const OpusDred, dst: *mut OpusDred) -> c_int {
    if dred_dec.is_null() || src.is_null() || dst.is_null() {
        return OPUS_BAD_ARG;
    }
    // SAFETY: the three are initialized states (opus.h); `src` may be `dst`.
    unsafe {
        if !(1..=2).contains(&(*src).process_stage) {
            return OPUS_BAD_ARG;
        }
        if !ptr::eq(src, dst) {
            (*dst).clone_from(&*src);
        }
        guard(OPUS_INTERNAL_ERROR, || crate::ok((*dred_dec).process(&mut *dst)))
    }
}

/// The DRED decode calls: `st` and `dred` are initialized states, `pcm`
/// holds `frame_size` samples per channel (opus.h).
macro_rules! dred_decode_fn {
    ($name:ident, $t:ty, $decode:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            st: *mut OpusDecoder,
            dred: *const OpusDred,
            dred_offset: i32,
            pcm: *mut $t,
            frame_size: i32,
        ) -> c_int {
            if frame_size <= 0 || pcm.is_null() {
                return OPUS_BAD_ARG;
            }
            // SAFETY: see the macro.
            unsafe {
                let (st, dred) = (&mut *st, &*dred);
                let pcm = core::slice::from_raw_parts_mut(pcm, frame_size as usize * st.channels());
                guard(OPUS_INTERNAL_ERROR, || {
                    st.$decode(dred, dred_offset, pcm, frame_size as usize).map_or_else(code, |n| n as c_int)
                })
            }
        }
    };
}

dred_decode_fn!(opus_decoder_dred_decode, i16, dred_decode);
dred_decode_fn!(opus_decoder_dred_decode24, i32, dred_decode24);
dred_decode_fn!(opus_decoder_dred_decode_float, f32, dred_decode_float);
