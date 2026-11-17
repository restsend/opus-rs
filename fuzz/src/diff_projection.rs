//! Differential projection fuzzer: ambisonics encoders of any channel
//! count and mapping family, and decoders with arbitrary layouts and
//! demixing matrices, on generated signals and arbitrary and
//! libopus-encoded packets, against libopus's opus_projection_*.
//!
//! Oracles:
//! - `create`: both accept or reject the encoder or decoder with the same
//!   error code, the same streams and the same demixing matrix.
//! - `result`: each call has the same result or error code.
//! - `bytes`: identical packets where `common::strict_bytes` holds.
//! - `pcm`: both decoders decode libopus's packets, arbitrary packets and
//!   losses alike, within `OFZ_DEC_TOL` of max(1, peak) per decoded
//!   channel, plus one step of an integer output per decoded channel.

use crate::common::*;
use crate::diff_decoder::Format;
use arbitrary::Arbitrary;
use core::ffi::{c_int, c_void};
use opus_rs::projection::{ProjectionDecoder, ProjectionEncoder};
use opus_rs::{Application, error};
use opusref as c;

#[derive(Arbitrary, Debug)]
struct Input {
    rate: u8,
    channels: u8,
    family: u8,
    app: u8,
    /// A decoder of its own layout and matrix bytes instead of the
    /// encoder's.
    decoder: Option<(u8, u8, Vec<u8>)>,
    ops: Vec<Op>,
}

#[derive(Arbitrary, Debug)]
enum Op {
    Encode { signal: Gen, duration: u8, budget: u16, entry: u8 },
    Packet(Vec<u8>),
    Lost,
    Format(Format),
    Bitrate(i32),
}

const DURATIONS: [usize; 9] = [1, 2, 4, 8, 16, 24, 32, 40, 48];
const APPS: [Application; 5] = [
    Application::Voip,
    Application::Audio,
    Application::RestrictedLowDelay,
    Application::RestrictedSilk,
    Application::RestrictedCelt,
];

struct RefEnc(*mut c_void);
impl Drop for RefEnc {
    fn drop(&mut self) {
        unsafe { c::opus_projection_encoder_destroy(self.0) }
    }
}
struct RefDec(*mut c_void);
impl Drop for RefDec {
    fn drop(&mut self) {
        unsafe { c::opus_projection_decoder_destroy(self.0) }
    }
}

/// The libFuzzer entry point's body.
pub fn fuzz(data: &[u8]) {
    let Ok(inp) = Input::arbitrary_take_rest(arbitrary::Unstructured::new(data)) else {
        return;
    };
    report("diff_projection", data, guarded(|| run(inp)));
}

fn code<T>(r: &Result<T, &str>) -> i32 {
    r.as_ref().map_or_else(|e| error::code(e).unwrap_or(1), |_| 0)
}

fn run(inp: Input) -> Result<(), Finding> {
    let fs = RATES[inp.rate as usize % 5];
    // Mostly valid ambisonic channel counts, sometimes any.
    let channels = match inp.channels % 16 {
        0 => inp.channels as usize / 16,
        k => [4, 6, 9, 11, 16, 18, 25, 27, 36, 38, 1, 3, 49, 2, 5][k as usize - 1],
    };
    let family = if inp.family % 8 == 0 { inp.family as i32 / 8 } else { 3 };
    let app = APPS[inp.app as usize % APPS.len()];
    let (mut s, mut cp, mut err) = (0, 0, 0);
    let re = unsafe {
        c::opus_projection_ambisonics_encoder_create(fs, channels as c_int, family, &mut s, &mut cp, app as c_int, &mut err)
    };
    let rs = ProjectionEncoder::new(fs, channels, family, app);
    if code(&rs) != err {
        fail!("create", "encoder", "fs{fs} ch{channels} family {family}: opus-rs {:?}, libopus {err}", rs.err());
    }
    ijon_set(1 << 32 | (channels as u64) << 8 | family as u64 & 0xff);
    let Ok(mut rs) = rs else { return Ok(()) };
    let re = RefEnc(re);
    if (rs.streams() as i32, rs.coupled_streams() as i32) != (s, cp) {
        fail!("create", "streams", "ch{channels}: opus-rs {} {}, libopus {s} {cp}", rs.streams(), rs.coupled_streams());
    }
    let mut matrix = vec![0u8; rs.demixing_matrix_size()];
    rs.demixing_matrix(&mut matrix).unwrap();
    let mut cm = vec![0u8; matrix.len()];
    unsafe { c::opus_projection_encoder_ctl(re.0, c::PROJECTION_GET_DEMIXING_MATRIX, cm.as_mut_ptr(), cm.len() as i32) };
    if matrix != cm {
        fail!("create", "matrix", "ch{channels}: demixing matrices differ");
    }
    // The decoder: the encoder's layout and matrix, or arbitrary ones.
    let (dch, ds, dc, mut dm) = match &inp.decoder {
        None => (channels, s as usize, cp as usize, matrix.clone()),
        Some((st, cs, m)) => {
            let (st, cs) = (1 + *st as usize % 40, *cs as usize % 40);
            (channels, st, cs, m.clone())
        }
    };
    let mut derr = 0;
    let rd = unsafe {
        c::opus_projection_decoder_create(fs, dch as c_int, ds as c_int, dc as c_int, dm.as_mut_ptr(), dm.len() as i32, &mut derr)
    };
    let d = ProjectionDecoder::new(fs, dch, ds, dc, &dm);
    if code(&d) != derr {
        fail!("create", "decoder", "ch{dch} s{ds} c{dc} {} matrix bytes: opus-rs {:?}, libopus {derr}", dm.len(), d.err());
    }
    let mut d = d.ok();
    let rd = (!rd.is_null()).then_some(RefDec(rd));
    let mut synth = Synth::new(fs);
    let mut fmt = Format::F32;
    let tol = env_f64("OFZ_DEC_TOL", 4e-6) as f32;
    let ch = channels;
    for (k, op) in inp.ops.iter().take(16).enumerate() {
        let pkt: Vec<u8> = match op {
            Op::Format(f) => {
                fmt = *f;
                continue;
            }
            Op::Bitrate(b) => {
                let a = code(&rs.multistream().set_bitrate(*b));
                let r = unsafe { c::opus_projection_encoder_ctl(re.0, c::SET_BITRATE, *b) };
                if a != r {
                    fail!("result", "bitrate", "op {k}: opus-rs {a}, libopus {r}");
                }
                continue;
            }
            Op::Packet(p) => p.clone(),
            Op::Lost => Vec::new(),
            Op::Encode { signal, duration, budget, entry } => {
                let frame = fs as usize / 400 * DURATIONS[*duration as usize % 9];
                let budget = *budget as usize % 8000;
                let mut mono = vec![0f32; frame];
                synth.fill(signal, &mut mono, 1, 0, 1);
                let pcm: Vec<f32> = (0..frame * ch)
                    .map(|i| mono[(i / ch).saturating_sub(i % ch)] * (1.0 - 0.02 * (i % ch) as f32))
                    .collect();
                let (mut x, mut y) = (vec![0u8; budget], vec![0u8; budget]);
                let (a, b) = match entry % 3 {
                    0 => (rs.encode(&pcm, frame, &mut x), unsafe {
                        c::opus_projection_encode_float(re.0, pcm.as_ptr(), frame as c_int, y.as_mut_ptr(), budget as i32)
                    }),
                    1 => {
                        let q: Vec<i16> = pcm.iter().map(|&v| (v * 32768.0).clamp(-32768.0, 32767.0) as i16).collect();
                        (rs.encode_i16(&q, frame, &mut x), unsafe {
                            c::opus_projection_encode(re.0, q.as_ptr(), frame as c_int, y.as_mut_ptr(), budget as i32)
                        })
                    }
                    _ => {
                        let q: Vec<i32> = pcm.iter().map(|&v| (v * 8_388_608.0) as i32).collect();
                        (rs.encode24(&q, frame, &mut x), unsafe {
                            c::opus_projection_encode24(re.0, q.as_ptr(), frame as c_int, y.as_mut_ptr(), budget as i32)
                        })
                    }
                };
                let ctx = format!("op {k}: ch{ch} {app:?} {frame} samples budget {budget} entry {}", entry % 3);
                let n = match a {
                    Ok(n) if b > 0 => n,
                    Err(e) if b <= 0 && error::code(e) == Some(b) => continue,
                    r => fail!("result", "encode", "{ctx}: opus-rs {r:?}, libopus {b}"),
                };
                if strict_bytes(app) && x[..n] != y[..b as usize] {
                    fail!("bytes", format!("ch{ch}"), "{ctx}: opus-rs {n} bytes, libopus {b}");
                }
                ijon_max(2 << 8 | ch as u64, n as u64);
                y.truncate(b as usize);
                y
            }
        };
        let (Some(d), Some(rd)) = (d.as_mut(), rd.as_ref()) else { continue };
        let size = fs as usize * 120 / 1000;
        let n = size * dch;
        let (ptr, len) = if pkt.is_empty() { (core::ptr::null(), 0) } else { (pkt.as_ptr(), pkt.len() as i32) };
        let (r, m, a, b): (_, _, Vec<f32>, Vec<f32>) = match fmt {
            Format::F32 => {
                let (mut x, mut y) = (vec![0f32; n], vec![0f32; n]);
                let r = d.decode(&pkt, size, &mut x);
                let m = unsafe { c::opus_projection_decode_float(rd.0, ptr, len, y.as_mut_ptr(), size as c_int, 0) };
                (r, m, x, y)
            }
            Format::I16 => {
                let (mut x, mut y) = (vec![0i16; n], vec![0i16; n]);
                let r = d.decode_i16(&pkt, size, &mut x);
                let m = unsafe { c::opus_projection_decode(rd.0, ptr, len, y.as_mut_ptr(), size as c_int, 0) };
                let f = |v: Vec<i16>| v.iter().map(|&s| s as f32 / 32768.0).collect();
                (r, m, f(x), f(y))
            }
            Format::I24 => {
                let (mut x, mut y) = (vec![0i32; n], vec![0i32; n]);
                let r = d.decode24(&pkt, size, &mut x);
                let m = unsafe { c::opus_projection_decode24(rd.0, ptr, len, y.as_mut_ptr(), size as c_int, 0) };
                let f = |v: Vec<i32>| v.iter().map(|&s| s as f32 / 8_388_608.0).collect();
                (r, m, f(x), f(y))
            }
        };
        let k_n = match (r, m) {
            (Ok(v), m) if m >= 0 && v == m as usize => v,
            (Err(_), m) if m < 0 => continue,
            (r, m) => fail!("result", "decode", "op {k}: opus-rs {r:?}, libopus {m} ({} bytes)", pkt.len()),
        };
        // Demixing sums every decoded channel into each output, so the
        // rounding of an integer output adds a step per input channel.
        let step = fmt.step() * (ds + dc) as f32;
        let (dmax, peak, at) = max_diff(&a[..k_n * dch], &b[..k_n * dch]);
        if !(dmax <= tol * peak * (ds + dc) as f32 + step) {
            fail!("pcm", format!("{fmt:?}"), "op {k}: |opus-rs - libopus| = {dmax:e} at {at} (opus-rs {}, libopus {})", a[at], b[at]);
        }
    }
    Ok(())
}
