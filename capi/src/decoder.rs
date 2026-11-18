//! `opus_decoder_*` / `opus_decode*` over [`opus_rs::OpusDecoder`].

use core::ffi::c_int;
use core::mem::{MaybeUninit, size_of};
use core::ptr;

use crate::{
    CtlArgs, OPUS_ALLOC_FAIL, OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, OPUS_UNIMPLEMENTED,
    decode_with, get, guard, ok, req, set_bool, set_out,
};

/// libopus's `OpusDecoder` is the opus-rs decoder: one flat block (opus-rs
/// is built without `heap`), so C callers may place it with
/// `opus_decoder_init` and copy it with `memcpy`.
pub type OpusDecoder = opus_rs::OpusDecoder;

const _: () = assert!(align_of::<OpusDecoder>() <= 16, "malloc alignment");

/// `opus_decoder_ctl` on `st` with the arguments in `ap`.
pub(crate) unsafe fn ctl(st: &mut OpusDecoder, request: c_int, mut ap: impl CtlArgs) -> c_int {
    let ap = &mut ap;
    // SAFETY: `ap` is the argument list of a ctl `request`, whose next
    // argument has the type opus_defines.h gives for `request`.
    unsafe {
        match request {
            req::GET_BANDWIDTH => get(ap, st.bandwidth().map_or(0, |b| b as i32)),
            req::SET_COMPLEXITY => ok(st.set_complexity(ap.int())),
            req::GET_COMPLEXITY => get(ap, st.complexity()),
            req::GET_FINAL_RANGE => get(ap, st.final_range()),
            req::RESET_STATE => {
                st.reset();
                OPUS_OK
            }
            req::GET_SAMPLE_RATE => get(ap, st.sample_rate()),
            req::GET_PITCH => get(ap, st.pitch()),
            req::GET_GAIN => get(ap, st.gain()),
            req::SET_GAIN => ok(st.set_gain(ap.int())),
            req::GET_LAST_PACKET_DURATION => get(ap, st.last_packet_duration() as i32),
            req::SET_PHASE_INVERSION_DISABLED => set_bool(ap, |v| st.set_phase_inversion_disabled(v)),
            req::GET_PHASE_INVERSION_DISABLED => get(ap, st.phase_inversion_disabled() as i32),
            req::SET_IGNORE_EXTENSIONS => set_bool(ap, |v| st.set_ignore_extensions(v)),
            req::GET_IGNORE_EXTENSIONS => get(ap, st.ignore_extensions() as i32),
            #[cfg(feature = "osce")]
            req::SET_OSCE_BWE => set_bool(ap, |v| st.set_osce_bwe(v)),
            #[cfg(feature = "osce")]
            req::GET_OSCE_BWE => get(ap, st.osce_bwe() as i32),
            _ => OPUS_UNIMPLEMENTED,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_decoder_get_size(channels: c_int) -> c_int {
    if !(1..=2).contains(&channels) {
        return 0;
    }
    crate::align_size(size_of::<OpusDecoder>()) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_init(st: *mut OpusDecoder, fs: i32, channels: c_int) -> c_int {
    if st.is_null() {
        return OPUS_BAD_ARG;
    }
    // SAFETY: `st` points to opus_decoder_get_size() writable bytes with
    // malloc's alignment (opus.h).
    let slot = unsafe { &mut *st.cast::<MaybeUninit<OpusDecoder>>() };
    guard(OPUS_INTERNAL_ERROR, || match OpusDecoder::init(slot, fs, channels as usize) {
        Ok(_) => OPUS_OK,
        Err(_) => OPUS_BAD_ARG,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_create(fs: i32, channels: c_int, error: *mut c_int) -> *mut OpusDecoder {
    let size = opus_decoder_get_size(channels);
    // SAFETY: `error` is NULL or valid for a write (opus.h); the state
    // comes from alloc_state with the size opus_decoder_init needs.
    unsafe {
        if size == 0 {
            set_out(error, OPUS_BAD_ARG);
            return ptr::null_mut();
        }
        let st = crate::alloc_state(size as usize).cast::<OpusDecoder>();
        if st.is_null() {
            set_out(error, OPUS_ALLOC_FAIL);
            return st;
        }
        let ret = opus_decoder_init(st, fs, channels);
        set_out(error, ret);
        if ret != OPUS_OK {
            crate::free_state(st);
            return ptr::null_mut();
        }
        st
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_destroy(st: *mut OpusDecoder) {
    // SAFETY: `st` is NULL or came from opus_decoder_create (opus.h).
    unsafe { crate::free_state(st) }
}

/// The decode calls: `st` is a decoder from opus_decoder_init or
/// opus_decoder_create (opus.h), the rest is checked by `decode_with`.
macro_rules! decode_fn {
    ($name:ident, $t:ty, $decode:ident, $decode_fec:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            st: *mut OpusDecoder,
            data: *const u8,
            len: i32,
            pcm: *mut $t,
            frame_size: c_int,
            decode_fec: c_int,
        ) -> c_int {
            // SAFETY: see the macro.
            let st = unsafe { &mut *st };
            guard(OPUS_INTERNAL_ERROR, || unsafe {
                decode_with(data, len, pcm, frame_size, decode_fec, st.channels(), |p, n, out, fec| {
                    if fec { st.$decode_fec(p, n, out) } else { st.$decode(p, n, out) }
                })
            })
        }
    };
}

decode_fn!(opus_decode, i16, decode_i16, decode_fec_i16);
decode_fn!(opus_decode24, i32, decode24, decode_fec24);
decode_fn!(opus_decode_float, f32, decode, decode_fec);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_ctl(st: *mut OpusDecoder, request: c_int, args: ...) -> c_int {
    // SAFETY: `st` is an initialized decoder and `args` the ctl's arguments
    // (opus.h).
    guard(OPUS_INTERNAL_ERROR, || unsafe { ctl(&mut *st, request, args) })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_get_nb_samples(dec: *const OpusDecoder, packet: *const u8, len: i32) -> c_int {
    // SAFETY: `dec` is an initialized decoder (opus.h).
    unsafe { crate::packet::opus_packet_get_nb_samples(packet, len, (*dec).sample_rate()) }
}
