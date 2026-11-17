//! The layers of each libopus 1.6.1 model, generated from its data files
//! by `scripts/gen_dnn_models.py`.

#[cfg(feature = "osce")]
pub mod bbwenet;
pub mod dred_rdovae_constants;
#[cfg(feature = "dred")]
pub mod dred_rdovae_dec;
#[cfg(feature = "dred")]
pub mod dred_rdovae_enc;
pub mod fargan;
#[cfg(feature = "osce")]
pub mod lace;
#[cfg(feature = "osce")]
pub mod nolace;
pub mod pitchdnn;
pub mod plc;
