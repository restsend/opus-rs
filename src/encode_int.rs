//! libopus `opus_encode24` of the float build: 24-bit input in 32-bit
//! integers, converted to float for coding and read as integers by the
//! signal analysis.

use crate::{OpusEncoder, PcmInput};

/// 120 ms of stereo at 48 kHz, the longest call.
const BUF: usize = 5760 * 2;

impl OpusEncoder {
    /// Encodes a frame of interleaved 24-bit PCM in 32-bit integers, ±2^23
    /// for full scale (libopus `opus_encode24`), as [`Self::encode`] does
    /// with the samples scaled by 2^-23.
    pub fn encode24(
        &mut self,
        input: &[i32],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        let mut buf = [0.0f32; BUF];
        let n = input.len().min(BUF);
        for (f, &x) in buf.iter_mut().zip(&input[..n]) {
            // libopus INT24TORES.
            *f = (1.0 / 32768.0 / 256.0) * x as f32;
        }
        self.encode_native(PcmInput::F32(&buf[..n]), frame_size, output, Some(input))
    }
}
