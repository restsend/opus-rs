//! The libopus C API (`opus.h`, `opus_multistream.h`, `opus_projection.h`)
//! over opus-rs.
//!
//! Every function keeps libopus's name, signature and argument checking, so
//! the `libopus.so.0` / `libopus.a` this crate builds stand in for libopus
//! for C callers, and libopus's own test programs link against it
//! (`capi/test.sh`).
//!
//! Every state is one flat block of memory, as in libopus: callers may
//! size it with `opus_*_get_size()`, place it themselves with `opus_*_init()`,
//! release it with `free()` and copy it with `memcpy`. That is why opus-rs
//! is built without its `heap` feature here, and why the multistream and
//! projection states are a header followed by their streams, used through
//! opus-rs's views.
//!
//! The unsafe code is the C ABI: every unsafe block states the caller
//! contract of the libopus header it relies on.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]

use core::ffi::{c_char, c_int};
use core::ptr;

mod decoder;
#[cfg(feature = "dred")]
mod dred;
mod encoder;
mod multistream;
mod packet;
mod projection;
mod repacketizer;
#[cfg(test)]
mod tests;
#[cfg(not(feature = "dred"))]
mod unimplemented;

pub const OPUS_OK: c_int = 0;
pub const OPUS_BAD_ARG: c_int = -1;
pub const OPUS_INTERNAL_ERROR: c_int = -3;
pub const OPUS_INVALID_PACKET: c_int = -4;
pub const OPUS_UNIMPLEMENTED: c_int = -5;
pub const OPUS_ALLOC_FAIL: c_int = -7;

pub const OPUS_AUTO: i32 = -1000;

/// The ctl requests of `opus_defines.h`, and the private ones of
/// `opus_private.h` that libopus's tests use.
pub(crate) mod req {
    pub const SET_APPLICATION: i32 = 4000;
    pub const GET_APPLICATION: i32 = 4001;
    pub const SET_BITRATE: i32 = 4002;
    pub const GET_BITRATE: i32 = 4003;
    pub const SET_MAX_BANDWIDTH: i32 = 4004;
    pub const GET_MAX_BANDWIDTH: i32 = 4005;
    pub const SET_VBR: i32 = 4006;
    pub const GET_VBR: i32 = 4007;
    pub const SET_BANDWIDTH: i32 = 4008;
    pub const GET_BANDWIDTH: i32 = 4009;
    pub const SET_COMPLEXITY: i32 = 4010;
    pub const GET_COMPLEXITY: i32 = 4011;
    pub const SET_INBAND_FEC: i32 = 4012;
    pub const GET_INBAND_FEC: i32 = 4013;
    pub const SET_PACKET_LOSS_PERC: i32 = 4014;
    pub const GET_PACKET_LOSS_PERC: i32 = 4015;
    pub const SET_DTX: i32 = 4016;
    pub const GET_DTX: i32 = 4017;
    pub const SET_VBR_CONSTRAINT: i32 = 4020;
    pub const GET_VBR_CONSTRAINT: i32 = 4021;
    pub const SET_FORCE_CHANNELS: i32 = 4022;
    pub const GET_FORCE_CHANNELS: i32 = 4023;
    pub const SET_SIGNAL: i32 = 4024;
    pub const GET_SIGNAL: i32 = 4025;
    pub const GET_LOOKAHEAD: i32 = 4027;
    pub const RESET_STATE: i32 = 4028;
    pub const GET_SAMPLE_RATE: i32 = 4029;
    pub const GET_FINAL_RANGE: i32 = 4031;
    pub const GET_PITCH: i32 = 4033;
    pub const SET_GAIN: i32 = 4034;
    pub const GET_GAIN: i32 = 4045;
    pub const SET_LSB_DEPTH: i32 = 4036;
    pub const GET_LSB_DEPTH: i32 = 4037;
    pub const GET_LAST_PACKET_DURATION: i32 = 4039;
    pub const SET_EXPERT_FRAME_DURATION: i32 = 4040;
    pub const GET_EXPERT_FRAME_DURATION: i32 = 4041;
    pub const SET_PREDICTION_DISABLED: i32 = 4042;
    pub const GET_PREDICTION_DISABLED: i32 = 4043;
    pub const SET_PHASE_INVERSION_DISABLED: i32 = 4046;
    pub const GET_PHASE_INVERSION_DISABLED: i32 = 4047;
    pub const GET_IN_DTX: i32 = 4049;
    #[cfg(feature = "dred")]
    pub const SET_DRED_DURATION: i32 = 4050;
    #[cfg(feature = "dred")]
    pub const GET_DRED_DURATION: i32 = 4051;
    #[cfg(feature = "osce")]
    pub const SET_OSCE_BWE: i32 = 4054;
    #[cfg(feature = "osce")]
    pub const GET_OSCE_BWE: i32 = 4055;
    pub const SET_IGNORE_EXTENSIONS: i32 = 4058;
    pub const GET_IGNORE_EXTENSIONS: i32 = 4059;
    pub const SET_FORCE_MODE: i32 = 11002;
    pub const MULTISTREAM_GET_ENCODER_STATE: i32 = 5120;
    pub const MULTISTREAM_GET_DECODER_STATE: i32 = 5122;
}

/// Where a ctl handler reads its argument from: the C caller's `va_list`,
/// or one value a multistream ctl hands each of its streams.
pub(crate) trait CtlArgs {
    unsafe fn int(&mut self) -> i32;
    unsafe fn ptr<T>(&mut self) -> *mut T;
}

impl CtlArgs for core::ffi::VaList<'_> {
    unsafe fn int(&mut self) -> i32 {
        // SAFETY: the caller's next variadic argument is an `opus_int32`.
        unsafe { self.next_arg::<i32>() }
    }

    unsafe fn ptr<T>(&mut self) -> *mut T {
        // SAFETY: the caller's next variadic argument is a pointer.
        unsafe { self.next_arg::<*mut T>() }
    }
}

/// A ctl argument already read: an `opus_int32` or a pointer.
#[derive(Clone, Copy)]
pub(crate) enum Arg {
    Int(i32),
    Ptr(*mut core::ffi::c_void),
}

impl CtlArgs for Arg {
    unsafe fn int(&mut self) -> i32 {
        match *self {
            Arg::Int(v) => v,
            Arg::Ptr(p) => p as usize as i32,
        }
    }

    unsafe fn ptr<T>(&mut self) -> *mut T {
        match *self {
            Arg::Int(v) => v as usize as *mut T,
            Arg::Ptr(p) => p as *mut T,
        }
    }
}

/// The libopus code of an opus-rs error.
pub(crate) fn code(e: &str) -> c_int {
    opus_rs::error::code(e).unwrap_or(OPUS_INTERNAL_ERROR)
}

/// `OPUS_OK`, or the code of the error.
pub(crate) fn ok(r: Result<(), &str>) -> c_int {
    r.map_or_else(code, |()| OPUS_OK)
}

/// A getter ctl: writes `value` through the pointer argument, `OPUS_BAD_ARG`
/// for NULL.
pub(crate) unsafe fn get<T>(ap: &mut impl CtlArgs, value: T) -> c_int {
    // SAFETY: the caller passes the ctl's argument list, whose next argument
    // is the getter's `T *` (opus_defines.h), and the pointer is NULL or
    // valid for a write of `T`.
    let p = unsafe { ap.ptr::<T>() };
    if p.is_null() {
        return OPUS_BAD_ARG;
    }
    // SAFETY: a non-NULL getter pointer is valid for a write (opus_defines.h).
    unsafe { p.write(value) };
    OPUS_OK
}

/// A setter ctl's 0/1 argument, `OPUS_BAD_ARG` for other values.
pub(crate) unsafe fn set_bool(ap: &mut impl CtlArgs, set: impl FnOnce(bool)) -> c_int {
    // SAFETY: the caller passes the ctl's argument list, whose next argument
    // is the setter's `opus_int32`.
    match unsafe { ap.int() } {
        v @ 0..=1 => {
            set(v == 1);
            OPUS_OK
        }
        _ => OPUS_BAD_ARG,
    }
}

/// The packet of a decode call: empty for a loss (NULL or length 0),
/// `OPUS_BAD_ARG` for a negative length.
pub(crate) unsafe fn packet<'a>(data: *const u8, len: i32) -> Result<&'a [u8], c_int> {
    match len {
        _ if data.is_null() || len == 0 => Ok(&[]),
        ..0 => Err(OPUS_BAD_ARG),
        // SAFETY: the caller's `data` holds `len` bytes (opus.h).
        _ => Ok(unsafe { core::slice::from_raw_parts(data, len as usize) }),
    }
}

/// The checks every decode call shares, then `decode(packet, frame_size,
/// pcm, fec)` with the C output of `frame_size` samples of `channels`.
pub(crate) unsafe fn decode_with<T>(
    data: *const u8,
    len: i32,
    pcm: *mut T,
    frame_size: c_int,
    decode_fec: c_int,
    channels: usize,
    decode: impl FnOnce(&[u8], usize, &mut [T], bool) -> Result<usize, &'static str>,
) -> c_int {
    let frame_size = usize::try_from(frame_size).unwrap_or(0);
    let (Some(n @ 1..), false, 0..=1) = (frame_size.checked_mul(channels), pcm.is_null(), decode_fec) else {
        return OPUS_BAD_ARG;
    };
    // SAFETY: `data` holds `len` bytes (opus.h).
    let packet = match unsafe { packet(data, len) } {
        Ok(p) => p,
        Err(e) => return e,
    };
    // SAFETY: `pcm` holds `frame_size` samples per channel (opus.h).
    let pcm = unsafe { core::slice::from_raw_parts_mut(pcm, n) };
    match decode(packet, frame_size, pcm, decode_fec == 1) {
        Ok(n) => n as c_int,
        Err(e) => code(e),
    }
}

/// Runs `f`, turning a panic into `fallback` instead of unwinding into C.
pub(crate) fn guard<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or(fallback)
}

/// Stores `value` in an optional C out-parameter.
pub(crate) unsafe fn set_out<T>(out: *mut T, value: T) {
    if !out.is_null() {
        // SAFETY: the caller's out-parameter is NULL or valid for a write.
        unsafe { out.write(value) };
    }
}

unsafe extern "C" {
    fn malloc(size: usize) -> *mut core::ffi::c_void;
    fn free(p: *mut core::ffi::c_void);
}

/// libc `malloc`, as libopus's `opus_alloc`: C code may release the states
/// `opus_*_create` returns with `free`, like libopus's.
pub(crate) unsafe fn alloc_state(size: usize) -> *mut u8 {
    // SAFETY: malloc takes any size.
    unsafe { malloc(size) as *mut u8 }
}

pub(crate) unsafe fn free_state<T>(st: *mut T) {
    // SAFETY: the caller's `st` is NULL or came from alloc_state.
    unsafe { free(st as *mut core::ffi::c_void) }
}

/// libopus's `opus_*_create`: a state of `size` bytes from `alloc_state`
/// (`OPUS_BAD_ARG` for 0) set up by `init`, whose result goes to `error`;
/// NULL when either fails.
pub(crate) unsafe fn create<T>(size: c_int, error: *mut c_int, init: impl FnOnce(*mut T) -> c_int) -> *mut T {
    // SAFETY: `error` is NULL or valid for a write (opus.h); `init` gets
    // `size` bytes from malloc and the state is freed when it fails.
    unsafe {
        let st = if size > 0 { alloc_state(size as usize).cast::<T>() } else { ptr::null_mut() };
        let ret = match size {
            ..=0 => OPUS_BAD_ARG,
            _ if st.is_null() => OPUS_ALLOC_FAIL,
            _ => init(st),
        };
        set_out(error, ret);
        if ret != OPUS_OK {
            free_state(st);
            return ptr::null_mut();
        }
        st
    }
}

/// libopus's `align()` of state sizes.
pub(crate) const fn align_size(size: usize) -> usize {
    size.div_ceil(16) * 16
}

/// The size of a flat C state: a header `H`, then `n` values `T` from
/// `align_size(size_of::<H>())`, as libopus lays out its multistream
/// states.
pub(crate) const fn block_size<H, T>(n: usize) -> usize {
    align_size(size_of::<H>()) + n * size_of::<T>()
}

/// The `n` values `T` after the header of the flat state `st`.
///
/// # Safety
///
/// `st` points to `block_size::<H, T>(n)` bytes with malloc's alignment,
/// which `T`'s alignment does not exceed, and nothing else refers to the
/// values while the slice lives.
pub(crate) unsafe fn tail<'a, H, T>(st: *mut H, n: usize) -> &'a mut [T] {
    const { assert!(align_of::<T>() <= 16) };
    // SAFETY: the values start inside the block, at a multiple of 16.
    unsafe { core::slice::from_raw_parts_mut(st.cast::<u8>().add(align_size(size_of::<H>())).cast::<T>(), n) }
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_get_version_string() -> *const c_char {
    // libopus's tests and some callers look for the "libopus " prefix.
    c"libopus 1.6.1-opus-rs".as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_strerror(error: c_int) -> *const c_char {
    // opus_rs::error::strerror's texts of -7..=0 and "unknown error", as C strings.
    static TEXTS: std::sync::LazyLock<Vec<std::ffi::CString>> = std::sync::LazyLock::new(|| {
        (-8..=0).map(|c| std::ffi::CString::new(opus_rs::error::strerror(c)).unwrap()).collect()
    });
    let i = if (-7..=0).contains(&error) { error + 8 } else { 0 };
    TEXTS[i as usize].as_ptr()
}
