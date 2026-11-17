//! Encoder controls of `opus_encoder_ctl` that the encoder keeps beside
//! its public fields, with libopus's checks and errors.

use crate::error::BAD_ARG;
use crate::{OpusEncoder, PcmInput};

impl OpusEncoder {
    /// The depth of the input signal in bits, 8 to 24 (libopus
    /// `OPUS_SET_LSB_DEPTH`, 24 by default): the encoder treats samples
    /// below it as noise, which lowers its rate on quiet signals, and
    /// frames without a sample above it as digital silence. The 16-bit
    /// entry point uses at most 16. Other values are `error::BAD_ARG`.
    pub fn set_lsb_depth(&mut self, depth: i32) -> Result<(), &'static str> {
        if !(8..=24).contains(&depth) {
            return Err(BAD_ARG);
        }
        self.lsb_depth = depth;
        Ok(())
    }

    /// The input depth (libopus `OPUS_GET_LSB_DEPTH`).
    pub fn lsb_depth(&self) -> i32 {
        self.lsb_depth
    }

    /// The channels to code stereo input with, 1 or 2, or `None` to choose
    /// them from the bitrate (libopus `OPUS_SET_FORCE_CHANNELS`, `None` by
    /// default). More channels than the input's or 0 are `error::BAD_ARG`.
    pub fn set_force_channels(&mut self, channels: Option<usize>) -> Result<(), &'static str> {
        if channels.is_some_and(|c| c == 0 || c > self.channels) {
            return Err(BAD_ARG);
        }
        self.force_channels = channels;
        Ok(())
    }

    /// The forced channels (libopus `OPUS_GET_FORCE_CHANNELS`).
    pub fn force_channels(&self) -> Option<usize> {
        self.force_channels
    }

    /// The depth the encoder assumes for `input`: the set depth, at most
    /// 16 for 16-bit input (libopus `IMIN(lsb_depth, st->lsb_depth)`).
    pub(crate) fn input_depth(&self, input: PcmInput<'_>) -> i32 {
        match input {
            PcmInput::F32(_) => self.lsb_depth,
            PcmInput::I16(_) => self.lsb_depth.min(16),
        }
    }
}
