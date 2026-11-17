//! Deep redundancy (DRED): libopus `dnn/dred_*.c`.
//!
//! - `encoder`: [`DredEnc`], the encoder state: resampling to 16 kHz,
//!   LPCNet features, RDOVAE latents and the coded DRED payload
//!   (`dred_encoder.c`).
//! - `decoder`: [`OpusDred`] and the reading of a DRED payload
//!   (`dred_decoder.c`).
//! - `rdovae_enc`, `rdovae_dec`: the RDOVAE networks
//!   (`dred_rdovae_enc.c`, `dred_rdovae_dec.c`).
//! - `coding`, `stats`, `laplace`: the quantizer schedule, the quantizer
//!   tables and the Laplace coder of the latents (`dred_coding.c`,
//!   `dred_rdovae_stats_data.c`, `ec_laplace_*_p0` of `celt/laplace.c`).

pub mod coding;
pub mod laplace;
pub mod stats;


/// `DRED_EXTENSION_ID`: the packet extension that carries DRED data.
pub const DRED_EXTENSION_ID: u8 = 126;
/// `DRED_EXPERIMENTAL_VERSION`: the second byte of the experimental DRED
/// extension payload (the first is `b'D'`).
pub const DRED_EXPERIMENTAL_VERSION: u8 = 12;
/// `DRED_EXPERIMENTAL_BYTES`: the length of that two-byte header.
pub const DRED_EXPERIMENTAL_BYTES: usize = 2;
/// `DRED_MIN_BYTES`.
pub const DRED_MIN_BYTES: usize = 8;
/// `DRED_SILK_ENCODER_DELAY`: the 16 kHz samples the encoder input buffer
/// starts with.
pub const DRED_SILK_ENCODER_DELAY: usize = 79 + 12 - 80;
/// `DRED_FRAME_SIZE`: 10 ms at 16 kHz.
pub const DRED_FRAME_SIZE: usize = 160;
/// `DRED_DFRAME_SIZE`: one RDOVAE frame, 20 ms at 16 kHz.
pub const DRED_DFRAME_SIZE: usize = 2 * DRED_FRAME_SIZE;
/// `DRED_MAX_DATA_SIZE`: the largest DRED payload in bytes.
pub const DRED_MAX_DATA_SIZE: usize = 1000;
/// `DRED_ENC_Q0`.
pub const DRED_ENC_Q0: i32 = 6;
/// `DRED_ENC_Q1`.
pub const DRED_ENC_Q1: i32 = 15;
/// `DRED_MAX_LATENTS`: 26 latents cover 1.04 s.
pub const DRED_MAX_LATENTS: usize = 26;
/// `DRED_NUM_REDUNDANCY_FRAMES`.
pub const DRED_NUM_REDUNDANCY_FRAMES: usize = 2 * DRED_MAX_LATENTS;
/// `DRED_MAX_FRAMES`: the latents and states the encoder keeps, and the
/// largest DRED duration in 10 ms frames.
pub const DRED_MAX_FRAMES: usize = 4 * DRED_MAX_LATENTS;
