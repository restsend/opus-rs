//! The bodies of the differential fuzz targets. The libFuzzer binaries in
//! `fuzz_targets/` call them, and `tests/regressions.rs` replays the saved
//! reproducers through them in an ordinary `cargo test`.

#[macro_use]
pub mod common;

pub mod api_args;
pub mod diff_decoder;
pub mod diff_encoder;
pub mod diff_kernels;
pub mod diff_ms_encoder;
pub mod diff_multistream;
pub mod diff_repacketizer;
