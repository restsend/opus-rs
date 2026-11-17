//! opus_strerror and opus_get_version_string against opus-rs's `error`.
use core::ffi::CStr;
use opus_rs::error;
use opusref::*;

#[test]
fn strerror_matches_libopus() {
    for code in -10..=3 {
        let c = unsafe { CStr::from_ptr(opus_strerror(code)) }.to_str().unwrap();
        assert_eq!(error::strerror(code), c, "code {code}");
        if (-7..0).contains(&code) {
            assert_eq!(error::code(c), Some(code));
        }
    }
}

#[test]
fn version_string_names_the_libopus_revision() {
    let c = unsafe { CStr::from_ptr(opus_get_version_string()) }.to_str().unwrap();
    // "libopus f1afc4222a-ref" for the reference build.
    let rev = c.trim_start_matches("libopus ").trim_end_matches("-ref");
    assert!(error::version_string().contains(rev), "{c}");
}
