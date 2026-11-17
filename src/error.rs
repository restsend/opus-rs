//! libopus's error codes (`opus_defines.h`) as the error values of the
//! APIs that mirror a libopus function, with libopus's texts
//! (`opus_strerror`, celt/celt.c), and the version string.
//!
//! An API returns `Err(BAD_ARG)` where libopus returns `OPUS_BAD_ARG`, and
//! [`code`] gives back the libopus number.

/// `OPUS_BAD_ARG` (-1): one or more invalid or out-of-range arguments.
pub const BAD_ARG: &str = "invalid argument";
/// `OPUS_BUFFER_TOO_SMALL` (-2): not enough bytes allocated in the buffer.
pub const BUFFER_TOO_SMALL: &str = "buffer too small";
/// `OPUS_INTERNAL_ERROR` (-3).
pub const INTERNAL_ERROR: &str = "internal error";
/// `OPUS_INVALID_PACKET` (-4): the compressed data passed is corrupted.
pub const INVALID_PACKET: &str = "corrupted stream";
/// `OPUS_UNIMPLEMENTED` (-5): invalid or unsupported request number.
pub const UNIMPLEMENTED: &str = "request not implemented";
/// `OPUS_INVALID_STATE` (-6).
pub const INVALID_STATE: &str = "invalid state";
/// `OPUS_ALLOC_FAIL` (-7).
pub const ALLOC_FAIL: &str = "memory allocation failed";

const TEXTS: [&str; 8] = [
    "success",
    BAD_ARG,
    BUFFER_TOO_SMALL,
    INTERNAL_ERROR,
    INVALID_PACKET,
    UNIMPLEMENTED,
    INVALID_STATE,
    ALLOC_FAIL,
];

/// libopus `opus_strerror`: the text of an error code, "success" for 0
/// and "unknown error" outside -7..=0.
pub fn strerror(code: i32) -> &'static str {
    match code {
        -7..=0 => TEXTS[(-code) as usize],
        _ => "unknown error",
    }
}

/// The libopus code (-7 to -1) of one of this module's errors.
pub fn code(error: &str) -> Option<i32> {
    TEXTS[1..]
        .iter()
        .position(|&t| t == error)
        .map(|i| -(i as i32) - 1)
}

/// libopus `opus_get_version_string`: this crate's name and version, and
/// the libopus revision it follows.
pub fn version_string() -> &'static str {
    concat!("opus-rs ", env!("CARGO_PKG_VERSION"), " (libopus f1afc4222a)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_round_trip() {
        for c in -7..=-1 {
            assert_eq!(code(strerror(c)), Some(c));
        }
        assert_eq!(code("success"), None);
        assert_eq!(strerror(1), "unknown error");
        assert_eq!(strerror(-8), "unknown error");
    }
}
