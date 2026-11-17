//! SILK-only packets of opus-rs with `silk-fixed` against fixed-point libopus,
//! whose SILK encoder (silk/fixed/) that feature ports, over the grid of
//! common/silk_grid.rs. Runs with
//! `cargo test --release --features fixed --test silk_fixed`.
#![cfg(feature = "fixed")]

#[path = "common/silk_grid.rs"]
mod grid;

#[test]
fn silk_packets_match_fixed_point_libopus() {
    grid::silk_packets();
}

#[test]
fn stereo_to_mono_switches_match_fixed_point_libopus() {
    grid::stereo_to_mono_switches();
}

#[test]
fn prediction_disabled_matches_fixed_point_libopus() {
    grid::prediction_disabled();
}
