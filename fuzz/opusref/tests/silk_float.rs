//! SILK-only packets of opus-rs against float libopus, whose SILK encoder
//! (silk/float/) the default build ports, over the grid of
//! common/silk_grid.rs. Runs with `cargo test --release --test silk_float`,
//! and with `--features simd` against the reference built as x86-64
//! configure builds it, opus-rs at the level OPUS_RS_ARCH names. The dnn
//! reference, which adds the neural features, is left out.
#![cfg(not(any(feature = "fixed", feature = "dnn")))]

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
