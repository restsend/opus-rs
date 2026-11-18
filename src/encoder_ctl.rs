//! Encoder controls of `opus_encoder_ctl` that the encoder keeps beside
//! its public fields, with libopus's checks and errors.

use crate::error::BAD_ARG;
use crate::OpusMode;
use crate::{Application, OpusEncoder, PcmInput, bits_to_bitrate, frame_rate_from_params};

/// The frame duration the encoder codes, as libopus
/// `OPUS_SET_EXPERT_FRAME_DURATION` (`OPUS_FRAMESIZE_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDuration {
    /// The `frame_size` of each call (`OPUS_FRAMESIZE_ARG`, the default).
    Argument = 5000,
    Ms2_5 = 5001,
    Ms5 = 5002,
    Ms10 = 5003,
    Ms20 = 5004,
    Ms40 = 5005,
    Ms60 = 5006,
    Ms80 = 5007,
    Ms100 = 5008,
    Ms120 = 5009,
}

impl OpusEncoder {
    /// Sets the application before the first frame (libopus
    /// `OPUS_SET_APPLICATION`): after it, and after a [`Self::reset`], only
    /// the current one is accepted, others are `error::BAD_ARG`.
    pub fn set_application(&mut self, application: Application) -> Result<(), &'static str> {
        let fixed = |a| matches!(a, Application::RestrictedSilk | Application::RestrictedCelt);
        if fixed(self.application) || fixed(application) || (!self.first && application != self.application) {
            return Err(BAD_ARG);
        }
        self.application = application;
        let low_delay = application == Application::RestrictedLowDelay;
        let fs = self.sampling_rate as usize;
        self.encoder_buffer = if low_delay { 0 } else { fs / 100 };
        self.delay_compensation = if low_delay { 0 } else { fs / 250 };
        Ok(())
    }

    /// The application (libopus `OPUS_GET_APPLICATION`).
    pub fn application(&self) -> Application {
        self.application
    }

    /// The sampling rate (libopus `OPUS_GET_SAMPLE_RATE`).
    pub fn sample_rate(&self) -> i32 {
        self.sampling_rate
    }

    /// The input channels, 1 or 2.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Sets [`Self::bitrate_bps`] as libopus `OPUS_SET_BITRATE` does:
    /// [`Self::BITRATE_AUTO`], [`Self::BITRATE_MAX`], or a rate clamped to
    /// 500 to 750000 b/s per channel; 0 and other negative values are
    /// `error::BAD_ARG`.
    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<(), &'static str> {
        self.bitrate_bps = match bitrate {
            Self::BITRATE_AUTO | Self::BITRATE_MAX => bitrate,
            ..=0 => return Err(BAD_ARG),
            b => b.clamp(500, 750_000 * self.channels as i32),
        };
        Ok(())
    }

    /// The bitrate in b/s (libopus `OPUS_GET_BITRATE`): the set rate, or
    /// what `Auto` and `Max` give for the last frame's duration (2.5 ms
    /// before the first frame), at most what 1276 bytes hold.
    pub fn bitrate(&self) -> i32 {
        let fs = self.sampling_rate;
        let frame = match self.prev_framesize {
            0 => fs as usize / 400,
            n => n,
        };
        let user = match self.bitrate_bps {
            Self::BITRATE_AUTO => 60 * fs / frame as i32 + fs * self.channels as i32,
            Self::BITRATE_MAX => 1_500_000,
            b => b,
        };
        user.min(bits_to_bitrate(1276 * 8, fs, frame))
    }

    /// Sets [`Self::complexity`], 0 to 10 (libopus `OPUS_SET_COMPLEXITY`);
    /// other values are `error::BAD_ARG`.
    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), &'static str> {
        if !(0..=10).contains(&complexity) {
            return Err(BAD_ARG);
        }
        self.complexity = complexity;
        Ok(())
    }

    /// Sets [`Self::packet_loss_perc`], 0 to 100 (libopus
    /// `OPUS_SET_PACKET_LOSS_PERC`); other values are `error::BAD_ARG`.
    pub fn set_packet_loss_perc(&mut self, percent: i32) -> Result<(), &'static str> {
        if !(0..=100).contains(&percent) {
            return Err(BAD_ARG);
        }
        self.packet_loss_perc = percent;
        Ok(())
    }

    /// In-band FEC as libopus `OPUS_SET_INBAND_FEC`: 0 off, 1 on (SILK is
    /// chosen when the expected loss calls for it), 2 on without switching
    /// to SILK for music; other values are `error::BAD_ARG`. Sets
    /// [`Self::use_inband_fec`].
    pub fn set_inband_fec(&mut self, fec: i32) -> Result<(), &'static str> {
        if !(0..=2).contains(&fec) {
            return Err(BAD_ARG);
        }
        self.use_inband_fec = fec != 0;
        self.fec_keeps_mode = fec == 2;
        Ok(())
    }

    /// The in-band FEC setting, 0, 1 or 2 (libopus `OPUS_GET_INBAND_FEC`).
    pub fn inband_fec(&self) -> i32 {
        match (self.use_inband_fec, self.fec_keeps_mode) {
            (false, _) => 0,
            (true, false) => 1,
            (true, true) => 2,
        }
    }

    /// Variable bitrate, on by default (libopus `OPUS_SET_VBR`); off sets
    /// [`Self::use_cbr`].
    pub fn set_vbr(&mut self, vbr: bool) {
        self.use_cbr = !vbr;
    }

    /// Whether the bitrate is variable (libopus `OPUS_GET_VBR`).
    pub fn vbr(&self) -> bool {
        !self.use_cbr
    }

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
    /// at a small cost in stereo quality. RestrictedSilk streams have no
    /// CELT and keep it off.
    pub fn set_phase_inversion_disabled(&mut self, disabled: bool) {
        if self.application != Application::RestrictedSilk {
            self.celt_enc.disable_inv = disabled;
        }
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

    /// The duration of the frames to code (libopus
    /// `OPUS_SET_EXPERT_FRAME_DURATION`, [`FrameDuration::Argument`] by
    /// default). With a fixed duration each call codes that much of its
    /// input, which has to be at least as long, and the signal analysis
    /// looks at all `frame_size` samples it is given.
    pub fn set_expert_frame_duration(&mut self, duration: FrameDuration) {
        self.variable_duration = duration;
    }

    /// The frame duration setting (libopus
    /// `OPUS_GET_EXPERT_FRAME_DURATION`).
    pub fn expert_frame_duration(&self) -> FrameDuration {
        self.variable_duration
    }

    /// libopus `frame_size_select`: the samples per channel a call of
    /// `frame_size` codes, `None` when that is no valid frame.
    pub fn select_frame_size(&self, frame_size: usize) -> Option<usize> {
        frame_size_select(self.sampling_rate, frame_size, self.variable_duration)
            .filter(|&n| self.application != Application::RestrictedSilk || n >= self.sampling_rate as usize / 100)
    }

    /// libopus `OPUS_SET_FORCE_MODE`: codes every frame in `mode`, or
    /// chooses (`None`).
    pub fn set_force_mode(&mut self, mode: Option<OpusMode>) {
        self.user_forced_mode = mode;
    }

    /// libopus `OPUS_SET_LFE`: the stream is a surround LFE channel,
    /// coded in narrowband CELT.
    #[cfg(feature = "std")]
    pub(crate) fn set_lfe(&mut self, lfe: bool) {
        self.lfe = lfe;
    }

    /// libopus `OPUS_SET_ENERGY_MASK`: the surround masking of the next
    /// frame, 21 bands per channel.
    #[cfg(feature = "std")]
    pub(crate) fn set_energy_mask(&mut self, mask: [f32; 42]) {
        self.energy_mask = Some(mask);
    }

    /// The depth the encoder assumes for the call's input: the set depth,
    /// at most the call's (16 for 16-bit input; libopus
    /// `IMIN(lsb_depth, st->lsb_depth)`).
    pub(crate) fn input_depth(&self, _input: PcmInput<'_>) -> i32 {
        self.lsb_depth.min(self.call_depth)
    }
}

/// libopus `frame_size_select` at `fs` Hz.
pub(crate) fn frame_size_select(fs: i32, frame_size: usize, duration: FrameDuration) -> Option<usize> {
    let rate = fs as usize;
    if frame_size < rate / 400 {
        return None;
    }
    let d = duration as usize;
    let new_size = match duration {
        FrameDuration::Argument => frame_size,
        FrameDuration::Ms2_5 | FrameDuration::Ms5 | FrameDuration::Ms10 | FrameDuration::Ms20 | FrameDuration::Ms40 => {
            (rate / 400) << (d - 5001)
        }
        _ => (d - 5001 - 2) * rate / 50,
    };
    if new_size > frame_size {
        return None;
    }
    frame_rate_from_params(fs, new_size).map(|_| new_size)
}
