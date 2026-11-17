//! libopus's float SILK encoder analysis (`silk/float/`), the encoder of the
//! default build. The noise shaping quantizers, the parameter quantizers,
//! the tables and the entropy coding are the shared fixed-point code.

pub mod find_pitch_lags_flp;
pub mod pitch_analysis_core_flp;
pub mod sigproc_flp;
pub mod structs_flp;
