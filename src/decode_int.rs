//! The integer decoder entry points, libopus `opus_decode` (16-bit, soft
//! clipped) and `opus_decode24` (24-bit in 32-bit integers) of the float
//! build: they decode to float and round.

use crate::error::{BAD_ARG, INVALID_PACKET};
use crate::{OpusDecoder, OpusMode, frame_samples_from_toc, mode_from_toc, opus_packet_parse};

/// The float buffer of a call: 120 ms of stereo at 48 kHz, which holds
/// every packet. Longer concealments go through it in pieces of whole
/// multiples of every frame duration (120 ms at 48 kHz), which conceal
/// what one call does.
const BUF: usize = 5760 * 2;

/// An integer sample of the integer entry points.
pub(crate) trait IntSample: Copy {
    /// libopus `FLOAT2INT16` / `FLOAT2INT24`.
    fn from_float(x: f32) -> Self;
}

impl IntSample for i16 {
    fn from_float(x: f32) -> Self {
        crate::float2int16(x)
    }
}

impl IntSample for i32 {
    /// Scaled by 2^23, clamped to ±2^24 (a NaN to -2^24) and rounded to
    /// nearest with ties to even.
    fn from_float(x: f32) -> Self {
        let x = x * 8_388_608.0;
        let x = if x > -16_777_216.0 { x } else { -16_777_216.0 };
        let x = if x < 16_777_216.0 { x } else { 16_777_216.0 };
        ((x as f64 + 6_755_399_441_055_744.0) - 6_755_399_441_055_744.0) as i32
    }
}

impl OpusDecoder {
    /// Decodes one packet into interleaved 16-bit PCM (libopus
    /// `opus_decode`), soft clipping the decoded audio into range across
    /// packets as libopus's float build does. Losses, `frame_size` and the
    /// result are as for [`Self::decode`]; `frame_size` 0 is
    /// `error::BAD_ARG`.
    pub fn decode_i16(
        &mut self,
        input: &[u8],
        frame_size: usize,
        output: &mut [i16],
    ) -> Result<usize, &'static str> {
        self.decode_int(input, frame_size, output, false, true)
    }

    /// [`Self::decode_i16`] for in-band FEC (libopus `opus_decode` with
    /// `decode_fec`), as [`Self::decode_fec`].
    pub fn decode_fec_i16(
        &mut self,
        input: &[u8],
        frame_size: usize,
        output: &mut [i16],
    ) -> Result<usize, &'static str> {
        self.decode_int(input, frame_size, output, true, true)
    }

    /// Decodes one packet into interleaved 24-bit PCM in 32-bit integers,
    /// ±2^23 for full scale (libopus `opus_decode24`), without soft
    /// clipping. Otherwise as [`Self::decode_i16`].
    pub fn decode24(
        &mut self,
        input: &[u8],
        frame_size: usize,
        output: &mut [i32],
    ) -> Result<usize, &'static str> {
        self.decode_int(input, frame_size, output, false, false)
    }

    /// [`Self::decode24`] for in-band FEC (libopus `opus_decode24` with
    /// `decode_fec`).
    pub fn decode_fec24(
        &mut self,
        input: &[u8],
        frame_size: usize,
        output: &mut [i32],
    ) -> Result<usize, &'static str> {
        self.decode_int(input, frame_size, output, true, false)
    }

    fn decode_int<T: IntSample>(
        &mut self,
        data: &[u8],
        mut frame_size: usize,
        output: &mut [T],
        fec: bool,
        soft_clip: bool,
    ) -> Result<usize, &'static str> {
        if frame_size == 0 {
            return Err(BAD_ARG);
        }
        if !data.is_empty() && !fec {
            let n = self.nb_samples(data).ok().filter(|&n| n > 0);
            frame_size = frame_size.min(n.ok_or(INVALID_PACKET)?);
        }
        let ch = self.channels;
        let mut buf = [0.0f32; BUF];
        if frame_size.checked_mul(ch).is_some_and(|n| n <= BUF) {
            // The float output the caller's buffer stands for.
            let len = output.len().min(frame_size * ch);
            let n = self.decode_native(data, frame_size, &mut buf[..len], fec, soft_clip)?;
            for (o, &x) in output.iter_mut().zip(&buf[..n * ch]) {
                *o = T::from_float(x);
            }
            return Ok(n);
        }
        // A concealment longer than the buffer, of a loss or before FEC:
        // libopus's checks of opus_decode_native, then the concealment in
        // pieces and the FEC frame.
        let fs = self.sampling_rate as usize;
        if !frame_size.is_multiple_of(fs / 400) {
            return Err(BAD_ARG);
        }
        if frame_size.checked_mul(ch).is_none_or(|n| output.len() < n) {
            return Err("Output buffer too small for packet");
        }
        let plc = if data.is_empty() {
            frame_size
        } else {
            let packet_frame_size =
                frame_samples_from_toc(data[0], self.sampling_rate).ok_or("Invalid TOC for sampling rate")?;
            opus_packet_parse(data)?;
            if packet_frame_size > frame_size
                || mode_from_toc(data[0]) == OpusMode::CeltOnly
                || self.mode == Some(OpusMode::CeltOnly)
            {
                frame_size
            } else {
                frame_size - packet_frame_size
            }
        };
        let duration = self.last_packet_duration;
        let piece = BUF / ch;
        let mut pos = 0;
        while pos < plc {
            let n = piece.min(plc - pos);
            let out = &mut buf[..n * ch];
            if let Err(e) = self.decode_native(&[], n, out, false, false) {
                self.last_packet_duration = duration;
                return Err(e);
            }
            for (o, &x) in output[pos * ch..].iter_mut().zip(out.iter()) {
                *o = T::from_float(x);
            }
            pos += n;
        }
        self.last_packet_duration = plc;
        if plc < frame_size {
            let n = frame_size - plc;
            let out = &mut buf[..n * ch];
            self.decode_native(data, n, out, true, false)?;
            for (o, &x) in output[plc * ch..].iter_mut().zip(out.iter()) {
                *o = T::from_float(x);
            }
        }
        self.last_packet_duration = frame_size;
        Ok(frame_size)
    }
}
