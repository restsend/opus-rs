//! The neural network code of libopus 1.6.1 (its `dnn/` folder), in safe
//! Rust.
//!
//! - `vec`, `nnet`: the generic C kernels and layer functions;
//! - `weights`, `models`: weight blobs and the layers of each model;
//!
//! The arithmetic follows libopus's generic C build (no SIMD,
//! `-ffp-contract=off`) operation for operation, including its mix of
//! float and double. Nothing here allocates: the layers borrow their
//! weights from a blob, and the states are fixed-size arrays.

// The constants are copied digit for digit from libopus.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

#[rustfmt::skip]
pub mod models;
pub mod nnet;
pub mod vec;
pub mod weights;
