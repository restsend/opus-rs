//! SILK-only packets of opus-rs against float libopus, whose SILK encoder
//! (silk/float/) the default build ports, over the grid of
//! common/silk_grid.rs. Runs with `cargo test --release --test silk_float`;
//! the dnn and simd references are left out (the neural features; the AVX2
//! inner product of float SILK).
#![cfg(not(any(feature = "fixed", feature = "dnn", feature = "simd")))]

#[path = "common/silk_grid.rs"]
mod grid;

#[test]
fn silk_packets_match_float_libopus() {
    grid::silk_packets();
}

#[test]
fn stereo_to_mono_switches_match_float_libopus() {
    grid::stereo_to_mono_switches();
}

#[test]
fn prediction_disabled_matches_float_libopus() {
    grid::prediction_disabled();
}
