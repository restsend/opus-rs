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

    /// Codes every frame without prediction from the frames before
    /// (libopus `OPUS_SET_PREDICTION_DISABLED`, off by default), so that
    /// each packet decodes without the ones before it, at a cost in
    /// quality: SILK codes each frame as the first after a reset and CELT
    /// without inter-frame prediction.
    pub fn set_prediction_disabled(&mut self, disabled: bool) {
        self.prediction_disabled = disabled;
    }

    /// Whether prediction is disabled (libopus
    /// `OPUS_GET_PREDICTION_DISABLED`).
    pub fn prediction_disabled(&self) -> bool {
        self.prediction_disabled
    }

    /// Disables the phase inversion of CELT's intensity stereo (libopus
    /// `OPUS_SET_PHASE_INVERSION_DISABLED`, off by default), which lets a
    /// stereo stream downmix to mono without the inverted bands cancelling,
    /// at a small cost in stereo quality.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        self.celt_enc.disable_inv = disabled;
    }

    /// Whether phase inversion is disabled (libopus
    /// `OPUS_GET_PHASE_INVERSION_DISABLED`).
    pub fn phase_inversion_disabled(&self) -> bool {
        self.celt_enc.disable_inv
    }

    /// libopus `silk_Encode` with `reducedDependency`: every channel of the
    /// input codes its next frame as the first after a reset.
    pub(crate) fn silk_reduced_dependency(&mut self) {
        if self.prediction_disabled {
            self.silk_enc.s_cmn.first_frame_after_reset = 1;
            if self.channels == 2 {
                self.silk_enc_side.s_cmn.first_frame_after_reset = 1;
            }
        }
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
