//! libopus's float SILK encoder analysis (`silk/float/`), the encoder of the
//! default build. The noise shaping quantizers, the parameter quantizers,
//! the tables and the entropy coding are the shared fixed-point code.

pub mod sigproc_flp;
