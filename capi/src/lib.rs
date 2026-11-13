//! The libopus C API (`opus.h`, `opus_multistream.h`) over opus-rs.
//!
//! Every function keeps libopus's name, signature and argument checking, so
//! the `libopus.so` / `libopus.a` this crate builds can stand in for libopus
//! for C callers, and libopus's own test programs can be linked against it
//! (`capi/run_libopus_tests.sh`).
//!
//! The codec state is one flat block of memory, as in libopus: callers may
//! size it with `opus_*_get_size()`, place it themselves with `opus_*_init()`,
//! and copy it with `memcpy`. That is why opus-rs is built without its `heap`
//! feature here.
//!
//! What opus-rs has no counterpart for (multistream encoding, projection,
//! DRED, extensions in repacketized padding, some ctls) returns
//! `OPUS_UNIMPLEMENTED`, as libopus does for features compiled out.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

use core::ffi::{c_char, c_int};

mod decoder;
mod encoder;
mod multistream;
mod packet;
mod repacketizer;
mod unimplemented;

pub const OPUS_OK: c_int = 0;
pub const OPUS_BAD_ARG: c_int = -1;
pub const OPUS_BUFFER_TOO_SMALL: c_int = -2;
pub const OPUS_INTERNAL_ERROR: c_int = -3;
pub const OPUS_INVALID_PACKET: c_int = -4;
pub const OPUS_UNIMPLEMENTED: c_int = -5;
pub const OPUS_INVALID_STATE: c_int = -6;
pub const OPUS_ALLOC_FAIL: c_int = -7;

pub const OPUS_AUTO: i32 = -1000;
pub const OPUS_BITRATE_MAX: i32 = -1;

pub const OPUS_APPLICATION_VOIP: c_int = 2048;
pub const OPUS_APPLICATION_AUDIO: c_int = 2049;
pub const OPUS_APPLICATION_RESTRICTED_LOWDELAY: c_int = 2051;
pub const OPUS_APPLICATION_RESTRICTED_SILK: c_int = 2052;
pub const OPUS_APPLICATION_RESTRICTED_CELT: c_int = 2053;

pub const OPUS_SIGNAL_VOICE: i32 = 3001;
pub const OPUS_SIGNAL_MUSIC: i32 = 3002;

pub const OPUS_BANDWIDTH_NARROWBAND: i32 = 1101;
pub const OPUS_BANDWIDTH_MEDIUMBAND: i32 = 1102;
pub const OPUS_BANDWIDTH_WIDEBAND: i32 = 1103;
pub const OPUS_BANDWIDTH_SUPERWIDEBAND: i32 = 1104;
pub const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;

pub const OPUS_FRAMESIZE_ARG: i32 = 5000;
pub const OPUS_FRAMESIZE_2_5_MS: i32 = 5001;
pub const OPUS_FRAMESIZE_40_MS: i32 = 5005;
pub const OPUS_FRAMESIZE_120_MS: i32 = 5009;

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
    pub const SET_IGNORE_EXTENSIONS: i32 = 4058;
    pub const GET_IGNORE_EXTENSIONS: i32 = 4059;
    pub const SET_VOICE_RATIO: i32 = 11018;
    pub const GET_VOICE_RATIO: i32 = 11019;
    pub const SET_FORCE_MODE: i32 = 11002;
    pub const SET_LFE: i32 = 10024;
    pub const SET_ENERGY_MASK: i32 = 10026;
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
        unsafe { self.next_arg::<i32>() }
    }

    unsafe fn ptr<T>(&mut self) -> *mut T {
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

/// Runs `f`, turning a panic into `fallback` instead of unwinding into C.
pub(crate) fn guard<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or(fallback)
}

/// Stores `value` in an optional C out-parameter.
pub(crate) unsafe fn set_out<T>(out: *mut T, value: T) {
    if !out.is_null() {
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
    unsafe { malloc(size) as *mut u8 }
}

pub(crate) unsafe fn free_state<T>(st: *mut T) {
    unsafe { free(st as *mut core::ffi::c_void) }
}

/// libopus's `align()` of state sizes.
pub(crate) const fn align_size(size: usize) -> usize {
    size.div_ceil(16) * 16
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_get_version_string() -> *const c_char {
    // libopus's tests and some callers look for the "libopus " prefix.
    c"libopus 1.6.1-opus-rs".as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_strerror(error: c_int) -> *const c_char {
    const STRINGS: [&core::ffi::CStr; 8] = [
        c"success",
        c"invalid argument",
        c"buffer too small",
        c"internal error",
        c"corrupted stream",
        c"request not implemented",
        c"invalid state",
        c"memory allocation failed",
    ];
    if (-7..=0).contains(&error) {
        STRINGS[(-error) as usize].as_ptr()
    } else {
        c"unknown error".as_ptr()
    }
}
