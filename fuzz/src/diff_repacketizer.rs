//! Differential fuzzer of the packet functions: the repacketizer
//! (`cat`, `out_range`, `out`, `init`), `opus_packet_pad`/`unpad` and their
//! multistream variants, `opus_packet_parse` and the packet queries, and
//! `opus_pcm_soft_clip`, on arbitrary packets and libopus-encoded ones with
//! extensions, against libopus.
//!
//! Oracles: every call has libopus's result (`result`), the written bytes
//! are identical (`bytes`), the parsed frames, counts and queries agree
//! (`parse`), and soft clipping is bit-identical with the same state
//! (`soft-clip`).

use crate::common::*;
use arbitrary::Arbitrary;
use core::ffi::c_int;
use opus_rs::{error, packet, repacketizer as rs};
use opusref as c;
use opusref::repacketizer as crp;

#[derive(Arbitrary, Debug)]
struct Input {
    ops: Vec<Op>,
}

#[derive(Arbitrary, Debug)]
enum Op {
    Cat(Pkt),
    OutRange { begin: u8, end: u8, max: u16 },
    Out { max: u16 },
    Init,
    Pad { pkt: Pkt, extra: u16 },
    Unpad(Pkt),
    MultistreamPad { pkt: Pkt, streams: u8, extra: u16 },
    MultistreamUnpad { pkt: Pkt, streams: u8 },
    Parse(Pkt),
    SoftClip { ch: u8, scale: u8, samples: Vec<i16> },
}

/// A packet: bytes as given, or from the libopus encoder with extensions.
#[derive(Arbitrary, Debug)]
enum Pkt {
    Raw(Vec<u8>),
    Encoded {
        config: u8,
        frames: u8,
        ext: Vec<(u8, u8, Vec<u8>)>,
    },
}

/// The libFuzzer entry point's body.
pub fn fuzz(data: &[u8]) {
    let Ok(inp) = Input::arbitrary_take_rest(arbitrary::Unstructured::new(data)) else {
        return;
    };
    if parse_only(&inp) {
        return;
    }
    report("diff_repacketizer", data, guarded(|| run(&inp)));
}

fn code<T: TryInto<i64>>(r: Result<T, &str>) -> i32 {
    match r {
        Ok(v) => v.try_into().ok().unwrap_or(i64::MAX) as i32,
        Err(e) => error::code(e).unwrap_or(1),
    }
}

/// The bytes of `p`: an encoded packet comes from a libopus encoder of one
/// of a few configurations, its frames padded with the given extensions.
fn bytes(p: &Pkt) -> Vec<u8> {
    let (config, frames, ext) = match p {
        Pkt::Raw(b) => return b.clone(),
        Pkt::Encoded { config, frames, ext } => (config, frames, ext),
    };
    // (fs, channels, 2.5 ms units, mode)
    const CONFIGS: [(i32, i32, usize, i32); 6] = [
        (48000, 2, 4, 1002),
        (48000, 1, 1, 1002),
        (16000, 1, 8, 1000),
        (16000, 2, 24, 1000),
        (48000, 2, 8, 1001),
        (8000, 1, 16, 1000),
    ];
    let (fs, ch, units, mode) = CONFIGS[*config as usize % CONFIGS.len()];
    let frame = fs as usize / 400 * units * (1 + *frames as usize % 2);
    let mut enc = RefEnc::new(fs, ch as usize, 2049);
    enc.ctl(c::SET_FORCE_MODE, mode);
    let pcm: Vec<f32> = (0..frame * ch as usize)
        .map(|i| (i as f32 * 0.031 + *config as f32).sin() * 0.3)
        .collect();
    let mut out = vec![0u8; 1500];
    let n = enc.encode(&pcm, frame, &mut out);
    if n <= 0 {
        return Vec::new();
    }
    out.truncate(n as usize);
    let nb = packet::nb_frames(&out).unwrap_or(1) as c_int;
    let exts: Vec<crp::OpusExtensionData> = ext
        .iter()
        .take(16)
        .map(|(id, f, d)| crp::OpusExtensionData {
            id: 3 + *id as c_int % 125,
            frame: *f as c_int % nb,
            data: d.as_ptr(),
            len: if *id % 125 < 29 { d.len().min(1) } else { d.len() } as i32,
        })
        .collect();
    if exts.is_empty() {
        return out;
    }
    let len = out.len();
    out.resize(len + 1200, 0);
    let r = unsafe {
        crp::opus_packet_pad_impl(out.as_mut_ptr(), len as i32, out.len() as i32, 0, exts.as_ptr(), exts.len() as c_int)
    };
    out.truncate(if r > 0 { r as usize } else { len });
    out
}

fn run(inp: &Input) -> Result<(), Finding> {
    // The packets the repacketizers borrow, kept until the end.
    let pkts: Vec<Vec<u8>> = inp
        .ops
        .iter()
        .take(32)
        .map(|op| match op {
            Op::Cat(p) => bytes(p),
            _ => Vec::new(),
        })
        .collect();
    let mut r = rs::Repacketizer::new();
    let crep = unsafe { crp::opus_repacketizer_create() };
    let _guard = Guard(crep);
    for (k, op) in inp.ops.iter().take(32).enumerate() {
        ijon_set(k as u64 * 16 + discriminant(op));
        match op {
            Op::Cat(_) => {
                let p = &pkts[k];
                let a = code(r.cat(p).map(|()| 0));
                let b = unsafe { crp::opus_repacketizer_cat(crep, p.as_ptr(), p.len() as i32) };
                if a != b {
                    fail!("result", "cat", "op {k} cat {p:02x?}: opus-rs {a}, libopus {b}");
                }
                let n = unsafe { crp::opus_repacketizer_get_nb_frames(crep) };
                if r.nb_frames() as i32 != n {
                    fail!("result", "nb_frames", "op {k}: opus-rs {}, libopus {n}", r.nb_frames());
                }
                // IJON: the frames held per ToC configuration and code, and
                // the result.
                let toc = p.first().map_or(0, |&t| u64::from(t >> 3) << 2 | u64::from(t & 3));
                ijon_set(1 << 40 | toc << 16 | (n as u64) << 4 | (b & 15) as u64);
            }
            Op::OutRange { begin, end, max } => {
                let (begin, end, max) = (*begin as usize % 50, *end as usize % 50, *max as usize % 4000);
                let (mut x, mut y) = (vec![0u8; max], vec![0u8; max]);
                let a = code(r.out_range(begin, end, &mut x));
                let b = unsafe {
                    crp::opus_repacketizer_out_range(crep, begin as c_int, end as c_int, y.as_mut_ptr(), max as i32)
                };
                compare("out_range", k, a, b, &x, &y)?;
            }
            Op::Out { max } => {
                let max = *max as usize % 4000;
                let (mut x, mut y) = (vec![0u8; max], vec![0u8; max]);
                let a = code(r.out(&mut x));
                let b = unsafe { crp::opus_repacketizer_out(crep, y.as_mut_ptr(), max as i32) };
                compare("out", k, a, b, &x, &y)?;
            }
            Op::Init => {
                r.init();
                unsafe { crp::opus_repacketizer_init(crep) };
            }
            Op::Pad { pkt, extra } => {
                let p = bytes(pkt);
                let new_len = (p.len() + *extra as usize % 2000).saturating_sub(usize::from(*extra % 7 == 0));
                let mut x = vec![0u8; new_len];
                let a = code(rs::pad(&p, &mut x).map(|()| 0));
                let mut y = p.clone();
                y.resize(new_len.max(p.len()), 0);
                let b = unsafe { crp::opus_packet_pad(y.as_mut_ptr(), p.len() as i32, new_len as i32) };
                compare("pad", k, a, b, &x, &y[..new_len])?;
                // IJON: padding sizes by magnitude, per result.
                ijon_max(2 << 8 | (b & 15) as u64, (new_len - p.len().min(new_len)) as u64);
            }
            Op::Unpad(pkt) => {
                let p = bytes(pkt);
                let (mut x, mut y) = (p.clone(), p.clone());
                let a = code(rs::unpad(&mut x));
                let b = unsafe { crp::opus_packet_unpad(y.as_mut_ptr(), y.len() as i32) };
                compare("unpad", k, a, b, &x, &y)?;
                // IJON: the padding removed, by magnitude.
                ijon_max(4 << 8 | (b & 15) as u64, (p.len() as i64 - b as i64).max(0) as u64);
            }
            Op::MultistreamPad { pkt, streams, extra } => {
                let p = bytes(pkt);
                let s = *streams as usize % 5;
                let new_len = (p.len() + *extra as usize % 2000).saturating_sub(usize::from(*extra % 7 == 0));
                let mut x = vec![0u8; new_len];
                let a = code(rs::multistream_pad(&p, s, &mut x).map(|()| 0));
                let mut y = p.clone();
                y.resize(new_len.max(p.len()), 0);
                let b = unsafe {
                    crp::opus_multistream_packet_pad(y.as_mut_ptr(), p.len() as i32, new_len as i32, s as c_int)
                };
                compare("multistream_pad", k, a, b, &x, &y[..new_len])?;
                ijon_max(3 << 8 | (s as u64) << 4 | (b & 15) as u64, (new_len - p.len().min(new_len)) as u64);
            }
            Op::MultistreamUnpad { pkt, streams } => {
                let p = bytes(pkt);
                let s = *streams as usize % 5;
                let (mut x, mut y) = (p.clone(), p.clone());
                let a = code(rs::multistream_unpad(&mut x, s));
                let b = unsafe { crp::opus_multistream_packet_unpad(y.as_mut_ptr(), y.len() as i32, s as c_int) };
                compare("multistream_unpad", k, a, b, &x, &y)?;
            }
            Op::Parse(pkt) => parse(k, &bytes(pkt))?,
            Op::SoftClip { ch, scale, samples } => {
                let ch = 1 + *ch as usize % 3;
                let n = samples.len() / ch;
                let scale = *scale as f32 / 4096.0;
                let mut x: Vec<f32> = samples.iter().map(|&s| s as f32 * scale).collect();
                let mut y = x.clone();
                let (mut ma, mut mb) = (vec![0.1f32; ch], vec![0.1f32; ch]);
                let _ = packet::soft_clip(&mut x, n, ch, &mut ma);
                unsafe { c::opus_pcm_soft_clip(y.as_mut_ptr(), n as c_int, ch as c_int, mb.as_mut_ptr()) };
                let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits());
                if !same(&x, &y) || !same(&ma, &mb) {
                    fail!("soft-clip", format!("ch{ch}"), "op {k}: samples or state differ");
                }
            }
        }
    }
    Ok(())
}

struct Guard(*mut core::ffi::c_void);
impl Drop for Guard {
    fn drop(&mut self) {
        unsafe { crp::opus_repacketizer_destroy(self.0) }
    }
}

fn discriminant(op: &Op) -> u64 {
    match op {
        Op::Cat(_) => 0,
        Op::OutRange { .. } => 1,
        Op::Out { .. } => 2,
        Op::Init => 3,
        Op::Pad { .. } => 4,
        Op::Unpad(_) => 5,
        Op::MultistreamPad { .. } => 6,
        Op::MultistreamUnpad { .. } => 7,
        Op::Parse(_) => 8,
        Op::SoftClip { .. } => 9,
    }
}

/// Same result, and for a success the same bytes up to its length (a
/// result of 0 compares the whole buffer, as pad leaves it).
fn compare(what: &str, k: usize, a: i32, b: i32, x: &[u8], y: &[u8]) -> Result<(), Finding> {
    if a != b {
        fail!("result", what, "op {k} {what}: opus-rs {a}, libopus {b}");
    }
    let n = if b == 0 { x.len() } else { b.max(0) as usize };
    if b >= 0 && x[..n] != y[..n] {
        let i = (0..n).find(|&i| x[i] != y[i]).unwrap();
        fail!("bytes", what, "op {k} {what}: byte {i} of {n}: opus-rs {:02x?}, libopus {:02x?}", &x[..n], &y[..n]);
    }
    Ok(())
}

/// `opus_packet_parse` and the queries of one packet.
fn parse(k: usize, p: &[u8]) -> Result<(), Finding> {
    let (ptr, len) = (p.as_ptr(), p.len() as i32);
    let mut toc = 0u8;
    let mut frames = [core::ptr::null(); 48];
    let mut size = [0i16; 48];
    let mut offset = 0;
    let n = if p.is_empty() {
        // libopus reads no byte of an empty packet here.
        error::code(error::INVALID_PACKET).unwrap()
    } else {
        unsafe { c::opus_packet_parse(ptr, len, &mut toc, frames.as_mut_ptr(), size.as_mut_ptr(), &mut offset) }
    };
    let got = match packet::parse(p) {
        Ok(parts) => {
            let same = parts.toc == toc
                && parts.payload_offset as i32 == offset
                && parts.frames().iter().enumerate().all(|(i, f)| f.as_ptr() == frames[i] && f.len() as i16 == size[i]);
            if n > 0 && !same {
                fail!("parse", "frames", "op {k} {p:02x?}: frames or offsets differ");
            }
            parts.count as i32
        }
        Err(e) => error::code(e).unwrap_or(1),
    };
    if got != n {
        fail!("parse", "count", "op {k} {p:02x?}: opus-rs {got}, libopus {n}");
    }
    let rs_q = (code(packet::nb_frames(p)), code(packet::nb_samples(p, 48000)), code(packet::nb_samples(p, 8000)));
    let c_q = unsafe {
        (
            c::opus_packet_get_nb_frames(ptr, len),
            c::opus_packet_get_nb_samples(ptr, len, 48000),
            c::opus_packet_get_nb_samples(ptr, len, 8000),
        )
    };
    if rs_q != c_q {
        fail!("parse", "queries", "op {k} {p:02x?}: opus-rs {rs_q:?}, libopus {c_q:?}");
    }
    if let Some(&t) = p.first() {
        let a = (
            packet::bandwidth(t) as i32,
            packet::nb_channels(t) as i32,
            packet::samples_per_frame(t, 48000),
            code(packet::has_lbrr(p).map(i64::from)),
        );
        let b = unsafe {
            (
                c::opus_packet_get_bandwidth(ptr),
                c::opus_packet_get_nb_channels(ptr),
                c::opus_packet_get_samples_per_frame(ptr, 48000),
                c::opus_packet_has_lbrr(ptr, len),
            )
        };
        if a != b {
            fail!("parse", "toc", "op {k} {p:02x?}: opus-rs {a:?}, libopus {b:?}");
        }
    }
    Ok(())
}
