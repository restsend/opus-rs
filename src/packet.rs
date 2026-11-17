//! Opus packet inspection and soft clipping (libopus `src/opus.c` and the
//! packet functions of `src/opus_decoder.c`): the ToC fields, the frames of
//! a packet, and `opus_pcm_soft_clip`.

use crate::error::{BAD_ARG, INVALID_PACKET};
use crate::extensions::parse_packet;
pub use crate::extensions::PacketParts as Packet;
use crate::{Bandwidth, OpusDecoder, OpusMode, bandwidth_from_toc, channels_from_toc, mode_from_toc};

/// libopus `opus_packet_parse`: the ToC, the frames and the offset of the
/// first frame's data in `data`, or `error::INVALID_PACKET`.
pub fn parse(data: &[u8]) -> Result<Packet<'_>, &'static str> {
    parse_packet(data)
}

/// libopus `opus_packet_get_bandwidth`: the bandwidth a packet's ToC byte
/// (its first byte) signals.
pub fn bandwidth(toc: u8) -> Bandwidth {
    bandwidth_from_toc(toc)
}

/// libopus `opus_packet_get_nb_channels`: 2 for a stereo ToC byte, else 1.
pub fn nb_channels(toc: u8) -> usize {
    channels_from_toc(toc)
}

/// libopus `opus_packet_get_samples_per_frame`: the samples per frame of a
/// ToC byte at `fs` Hz.
pub fn samples_per_frame(toc: u8, fs: i32) -> i32 {
    let d = ((toc >> 3) & 3) as u32;
    if toc & 0x80 != 0 {
        fs.wrapping_shl(d) / 400
    } else if toc & 0x60 == 0x60 {
        if toc & 0x08 != 0 { fs / 50 } else { fs / 100 }
    } else if d == 3 {
        fs.wrapping_mul(60) / 1000
    } else {
        fs.wrapping_shl(d) / 100
    }
}

/// libopus `opus_packet_get_nb_frames`: the frames of a packet, from its
/// first two bytes; `error::BAD_ARG` for an empty packet and
/// `error::INVALID_PACKET` for a code 3 packet without its count byte.
pub fn nb_frames(packet: &[u8]) -> Result<usize, &'static str> {
    match packet {
        [] => Err(BAD_ARG),
        [toc, ..] if toc & 3 == 0 => Ok(1),
        [toc, ..] if toc & 3 != 3 => Ok(2),
        [_] => Err(INVALID_PACKET),
        [_, count, ..] => Ok((count & 0x3f) as usize),
    }
}

/// libopus `opus_packet_get_nb_samples`: the samples per channel of a
/// packet at `fs` Hz, `error::INVALID_PACKET` above 120 ms.
pub fn nb_samples(packet: &[u8], fs: i32) -> Result<usize, &'static str> {
    let samples = nb_frames(packet)? as i64 * samples_per_frame(packet[0], fs) as i64;
    if samples * 25 > fs as i64 * 3 || samples < 0 {
        return Err(INVALID_PACKET);
    }
    Ok(samples as usize)
}

/// libopus `opus_packet_has_lbrr`: whether the first SILK frame of a packet
/// carries in-band FEC (LBRR) for the packet before; `false` for CELT-only
/// packets and an error for an invalid one.
pub fn has_lbrr(packet: &[u8]) -> Result<bool, &'static str> {
    let &toc = packet.first().ok_or(INVALID_PACKET)?;
    if mode_from_toc(toc) == OpusMode::CeltOnly {
        return Ok(false);
    }
    let nb = (samples_per_frame(toc, 48000) / 960).max(1) as u32;
    let frames = parse_packet(packet)?;
    let Some(&first) = frames.frames[0].first() else {
        return Ok(false);
    };
    let mut lbrr = (first >> (7 - nb)) & 1 != 0;
    if channels_from_toc(toc) == 2 {
        lbrr |= (first >> (6 - 2 * nb)) & 1 != 0;
    }
    Ok(lbrr)
}

impl OpusDecoder {
    /// libopus `opus_decoder_get_nb_samples`: the samples per channel of a
    /// packet at the decoder's rate.
    pub fn nb_samples(&self, packet: &[u8]) -> Result<usize, &'static str> {
        nb_samples(packet, self.sampling_rate)
    }
}

/// libopus `opus_pcm_soft_clip`: brings interleaved float samples of
/// `frame_size` samples per channel within [-1, 1] with a smooth
/// non-linearity that continues across calls through `declip_mem` (one
/// value per channel, zero for a new stream). `error::BAD_ARG` when `pcm`
/// or `declip_mem` is too short; nothing happens without samples or
/// channels.
pub fn soft_clip(
    pcm: &mut [f32],
    frame_size: usize,
    channels: usize,
    declip_mem: &mut [f32],
) -> Result<(), &'static str> {
    if channels == 0 || frame_size == 0 {
        return Ok(());
    }
    let len = frame_size.checked_mul(channels).ok_or(BAD_ARG)?;
    if pcm.len() < len || declip_mem.len() < channels {
        return Err(BAD_ARG);
    }
    soft_clip_impl(&mut pcm[..len], channels, declip_mem);
    Ok(())
}

/// libopus `opus_pcm_soft_clip_impl` with the generic
/// `opus_limit2_checkwithin1` (which clamps to [-2, 2] and gives no hint).
pub(crate) fn soft_clip_impl(pcm: &mut [f32], channels: usize, declip_mem: &mut [f32]) {
    let n = pcm.len() / channels;
    if n == 0 {
        return;
    }
    for x in pcm.iter_mut() {
        *x = x.clamp(-2.0, 2.0);
    }
    for c in 0..channels {
        let at = |i: usize| i * channels + c;
        let mut a = declip_mem[c];
        // Continue the previous frame's non-linearity up to the first zero
        // crossing.
        for i in 0..n {
            let x = pcm[at(i)];
            if x * a >= 0.0 {
                break;
            }
            pcm[at(i)] = x + a * x * x;
        }
        let mut curr = 0;
        let x0 = pcm[at(0)];
        loop {
            let Some(i) = (curr..n).find(|&i| pcm[at(i)] > 1.0 || pcm[at(i)] < -1.0) else {
                a = 0.0;
                break;
            };
            let xi = pcm[at(i)];
            let mut peak_pos = i;
            let (mut start, mut end) = (i, i);
            let mut maxval = xi.abs();
            // The zero crossings around the peak.
            while start > 0 && xi * pcm[at(start - 1)] >= 0.0 {
                start -= 1;
            }
            while end < n && xi * pcm[at(end)] >= 0.0 {
                if pcm[at(end)].abs() > maxval {
                    maxval = pcm[at(end)].abs();
                    peak_pos = end;
                }
                end += 1;
            }
            // Clipping before the first zero crossing.
            let special = start == 0 && xi * pcm[at(0)] >= 0.0;
            // maxval + a*maxval^2 = 1, boosted by 2^-22.
            a = (maxval - 1.0) / (maxval * maxval);
            a += a * 2.4e-7;
            if xi > 0.0 {
                a = -a;
            }
            for j in start..end {
                let x = pcm[at(j)];
                pcm[at(j)] = x + a * x * x;
            }
            if special && peak_pos >= 2 {
                // A linear ramp from the first sample to the peak.
                let mut offset = x0 - pcm[at(0)];
                let delta = offset / peak_pos as f32;
                for j in curr..peak_pos {
                    offset -= delta;
                    pcm[at(j)] = (pcm[at(j)] + offset).clamp(-1.0, 1.0);
                }
            }
            curr = end;
            if curr == n {
                break;
            }
        }
        declip_mem[c] = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_and_samples() {
        assert_eq!(nb_frames(&[]), Err(BAD_ARG));
        assert_eq!(nb_frames(&[0x03]), Err(INVALID_PACKET));
        assert_eq!(nb_frames(&[0x03, 0x45]), Ok(5));
        // 6 x 20 ms is 120 ms; 7 is too long.
        assert_eq!(nb_samples(&[0xfb, 6], 48000), Ok(5760));
        assert_eq!(nb_samples(&[0xfb, 7], 48000), Err(INVALID_PACKET));
        assert_eq!(samples_per_frame(0x18, 16000), 960);
    }

    #[test]
    fn soft_clip_bounds_the_samples() {
        let mut x: [f32; 8] = [0.5, 1.5, 3.0, 0.2, -0.1, -1.7, -0.4, 0.3];
        let mut mem = [0.0f32];
        soft_clip(&mut x, 8, 1, &mut mem).unwrap();
        assert!(x.iter().all(|v| v.abs() <= 1.0));
        assert_eq!(soft_clip(&mut x, 9, 1, &mut mem), Err(BAD_ARG));
    }
}
