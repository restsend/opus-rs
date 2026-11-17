//! The neural network code of libopus 1.6.1 (its `dnn/` folder), in safe
//! Rust.
//!
//! - `vec`, `nnet`: the generic C kernels and layer functions;
//! - `weights`, `models`, `builtin`: weight blobs, the layers of each
//!   model and the embedded 1.6.1 weights;
//! - `fft`, `freq`, `burg`, `common`, `lpcnet_tables`, `pitchdnn`,
//!   `lpcnet_enc`: the LPCNet feature analysis;
//! - `fargan`, `lpcnet_plc`: deep packet loss concealment;
//! - `dred`: deep redundancy (RDOVAE and its coding);
//! - `osce`: the LACE/NoLACE enhancers and the BBWENet bandwidth extension.
//!
//! The arithmetic follows libopus's generic C build (no SIMD,
//! `-ffp-contract=off`) operation for operation, including its mix of
//! float and double. Nothing here allocates: the layers borrow their
//! weights from a blob, and the states are fixed-size arrays.

// The constants are copied digit for digit from libopus.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

pub mod builtin;
pub mod burg;
pub mod common;
#[cfg(feature = "dred")]
pub mod dred;
pub mod fargan;
pub mod fft;
pub mod freq;
pub mod lpcnet_enc;
pub mod lpcnet_plc;
#[rustfmt::skip]
pub mod lpcnet_tables;
#[rustfmt::skip]
pub mod models;
pub mod nnet;
#[cfg(feature = "osce")]
pub mod osce;
pub mod pitchdnn;
pub mod vec;
pub mod weights;
