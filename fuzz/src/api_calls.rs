//! The libopus API calls of `api_args`: arbitrary arguments, buffer
//! lengths and frame sizes for the encoder, decoder, multistream,
//! projection, packet and repacketizer functions. They must not panic (a
//! panic is a libFuzzer crash here), and where libopus has the same call
//! with valid buffers, they return its error code.

use crate::common::*;
use arbitrary::Arbitrary;
use opus_rs::multistream::{ChannelMappingTable, MultistreamDecoder};
use opus_rs::multistream_encoder::MultistreamEncoder;
use opus_rs::projection::{ProjectionDecoder, ProjectionEncoder};
use opus_rs::repacketizer::{self, Repacketizer};
use opus_rs::{Application, OpusDecoder, OpusEncoder, error, packet};
use opusref as c;

#[derive(Arbitrary, Debug)]
pub enum ApiOp {
    /// An encoder control with any value, on a valid encoder.
    EncoderCtl { setup: u8, ctl: u8, value: i32 },
    /// An encode call with any frame size and buffer lengths.
    Encode { setup: u8, entry: u8, frame: u32, input: u16, output: u16 },
    /// A decode call with any packet, frame size and output length.
    Decode { setup: u8, entry: u8, packet: Vec<u8>, frame: u32, output: u16 },
    /// The packet functions with any packet and rate.
    Packet { packet: Vec<u8>, fs: i32, n: u16, ch: u8, mem: u8 },
    /// The repacketizer and padding with any spans and lengths.
    Repacketize { packets: Vec<Vec<u8>>, begin: u8, end: u8, len: u16, streams: u8 },
    /// Multistream and projection codecs with any layout and lengths.
    Multistream { fs: i32, channels: u8, streams: u8, coupled: u8, family: i8, mapping: Vec<u8>, frame: u16, len: u16, packet: Vec<u8> },
}

const RATES: [i32; 5] = [8000, 12000, 16000, 24000, 48000];

fn code<T>(r: &Result<T, &str>) -> i32 {
    r.as_ref().map_or_else(|e| error::code(e).unwrap_or(1), |_| 0)
}

/// IJON: transitions between the error codes of one kind of call.
fn reward(kind: u64, prev: &mut [i32; 8], r: i32) {
    let k = kind as usize % 8;
    ijon_set(5 << 32 | kind << 16 | ((prev[k] + 8) as u64 & 0xff) << 8 | (r + 8) as u64 & 0xff);
    prev[k] = r;
}

pub fn run(op: &ApiOp, prev: &mut [i32; 8]) {
    match op {
        ApiOp::EncoderCtl { setup, ctl, value } => {
            let fs = RATES[*setup as usize % 5];
            let ch = 1 + (*setup as usize / 5) % 2;
            let mut rs = OpusEncoder::new(fs, ch, Application::Audio).unwrap();
            let mut re = RefEnc::new(fs, ch, 2049);
            let v = *value;
            let (a, req) = match ctl % 6 {
                0 => (code(&rs.set_bitrate(v)), c::SET_BITRATE),
                1 => (code(&rs.set_complexity(v)), c::SET_COMPLEXITY),
                2 => (code(&rs.set_packet_loss_perc(v)), c::SET_PACKET_LOSS_PERC),
                3 => (code(&rs.set_inband_fec(v)), c::SET_INBAND_FEC),
                4 => (code(&rs.set_lsb_depth(v)), c::SET_LSB_DEPTH),
                _ => (
                    code(&rs.set_force_channels(if v == -1000 { None } else { usize::try_from(v).ok().or(Some(usize::MAX)) })),
                    c::SET_FORCE_CHANNELS,
                ),
            };
            let b = re.try_ctl(req, v);
            assert_eq!(a, b, "encoder control {req} = {v}");
            reward(0, prev, a);
        }
        ApiOp::Encode { setup, entry, frame, input, output } => {
            let fs = RATES[*setup as usize % 5];
            let ch = 1 + (*setup as usize / 5) % 2;
            let mut rs = OpusEncoder::new(fs, ch, Application::Voip).unwrap();
            let (frame, input, output) = (*frame as usize % 20000, *input as usize % 12000, *output as usize % 2000);
            let mut out = vec![0u8; output];
            let r = match entry % 3 {
                0 => rs.encode(&vec![0.25; input], frame, &mut out),
                1 => rs.encode_i16(&vec![999; input], frame, &mut out),
                _ => rs.encode24(&vec![99999; input], frame, &mut out),
            };
            if input >= frame * ch && output > 0 {
                // libopus with buffers that hold the call's samples.
                let mut re = RefEnc::new(fs, ch, 2048);
                let b = re.encode(&vec![0.25; input], frame, &mut vec![0u8; output]);
                assert_eq!(r.is_ok(), b > 0, "encode {frame} samples into {output} bytes: {r:?} vs {b}");
            }
            reward(1, prev, code(&r));
        }
        ApiOp::Decode { setup, entry, packet, frame, output } => {
            let fs = RATES[*setup as usize % 5];
            let ch = 1 + (*setup as usize / 5) % 2;
            let mut d = OpusDecoder::new(fs, ch).unwrap();
            let (frame, output) = (*frame as usize % 30000, *output as usize % 30000);
            let r = match entry % 7 {
                0 => d.decode(packet, frame, &mut vec![0.0; output]),
                1 => d.decode_fec(packet, frame, &mut vec![0.0; output]),
                2 => d.decode_i16(packet, frame, &mut vec![0; output]),
                3 => d.decode_fec_i16(packet, frame, &mut vec![0; output]),
                4 => d.decode24(packet, frame, &mut vec![0; output]),
                5 => d.decode_fec24(packet, frame, &mut vec![0; output]),
                _ => d.nb_samples(packet),
            };
            reward(2, prev, code(&r));
        }
        ApiOp::Packet { packet: p, fs, n, ch, mem } => {
            let parsed = packet::parse(p);
            let _ = (packet::nb_frames(p), packet::nb_samples(p, *fs), packet::has_lbrr(p));
            if let Some(&t) = p.first() {
                let _ = (packet::bandwidth(t), packet::nb_channels(t), packet::samples_per_frame(t, *fs));
            }
            let mut x: Vec<f32> = p.iter().map(|&b| (b as f32 - 128.0) / 40.0).collect();
            let mut m = vec![0.5f32; *mem as usize % 4];
            let r = packet::soft_clip(&mut x, *n as usize % 300, *ch as usize % 4, &mut m);
            reward(3, prev, code(&parsed));
            reward(4, prev, code(&r));
        }
        ApiOp::Repacketize { packets, begin, end, len, streams } => {
            let mut rp = Repacketizer::new();
            for p in packets.iter().take(8) {
                let _ = rp.cat(p);
            }
            let mut out = vec![0u8; *len as usize % 3000];
            let r = rp.out_range(*begin as usize % 60, *end as usize % 60, &mut out);
            reward(5, prev, code(&r));
            if let Some(p) = packets.first() {
                let _ = repacketizer::pad(p, &mut out);
                let _ = repacketizer::multistream_pad(p, *streams as usize % 6, &mut out);
                let _ = repacketizer::unpad(&mut p.clone());
                let r = repacketizer::multistream_unpad(&mut p.clone(), *streams as usize % 6);
                reward(6, prev, code(&r));
            }
        }
        ApiOp::Multistream { fs, channels, streams, coupled, family, mapping, frame, len, packet } => {
            let fs = if *fs % 3 == 0 { RATES[fs.unsigned_abs() as usize % 5] } else { *fs };
            // Few streams, so that a call stays fast.
            let (ch, s, cp) = (*channels as usize % 12, *streams as usize % 9, *coupled as usize % 9);
            let (frame, len) = (*frame as usize % 6000, *len as usize % 20000);
            let mut map = mapping.clone();
            map.resize(ch.max(mapping.len()), 0);
            let mut out = vec![0u8; len % 4000];
            if let Ok(mut e) = MultistreamEncoder::new(fs, ch, s, cp, &map, Application::Audio) {
                let _ = e.encode(&vec![0.1; len], frame, &mut out);
            }
            let r = MultistreamEncoder::new_surround(fs, ch, *family as i32, Application::Voip);
            reward(7, prev, code(&r));
            if let Ok(mut e) = r {
                let _ = e.encode_i16(&vec![77; len], frame, &mut out);
            }
            if let Ok(mut e) = ProjectionEncoder::new(fs, ch, *family as i32, Application::Audio) {
                let _ = e.encode24(&vec![1 << 20; len], frame, &mut out);
                let mut m = vec![0u8; len % 3000];
                let _ = e.demixing_matrix(&mut m);
            }
            let table = ChannelMappingTable { channels: ch as u8, stream_count: s as u8, coupled_count: cp as u8, mapping: map.clone() };
            if let Ok(mut d) = MultistreamDecoder::new(fs, table) {
                let _ = d.decode_i16(packet, frame, &mut vec![0; len]);
                let _ = d.decode24(packet, frame, &mut vec![0; len]);
                let _ = d.decoder_state(s);
            }
            if let Ok(mut d) = ProjectionDecoder::new(fs, ch, s, cp, &map) {
                let _ = d.decode(packet, frame, &mut vec![0.0; len]);
                let _ = d.decode_i16(packet, frame, &mut vec![0; len]);
            }
        }
    }
}
