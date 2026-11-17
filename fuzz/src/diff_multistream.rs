//! Differential multistream decoder fuzzer: random channel layouts
//! (channels, streams, coupled streams, mapping), random and
//! libopus-encoded multistream packets and losses, decoded by opus-rs's
//! MultistreamDecoder and libopus's opus_multistream_decoder.
//!
//! Oracles: both accept or both reject the layout (`layout`); per call, both
//! succeed with the same sample count or both fail (`result`); the samples
//! agree within `OFZ_DEC_TOL` relative to max(1, peak) (`pcm`, default 4e-6),
//! plus one step of an integer output; the final range and the first
//! stream's queries agree (`get`); controls have libopus's result (`ctl`).

use crate::common::*;
use arbitrary::Arbitrary;
use core::ffi::{c_int, c_void};
use opus_rs::multistream::{ChannelMappingTable, MultistreamDecoder};
use crate::diff_decoder::Format;
use opusref as c;

#[derive(Arbitrary, Debug)]
struct Input {
    rate: u8,
    channels: u8,
    streams: u8,
    coupled: u8,
    mapping: [u8; 8],
    ops: Vec<Op>,
}

#[derive(Debug)]
enum Op {
    Packet(Vec<u8>),
    /// A packet from libopus's multistream encoder for the same layout.
    Encoded {
        signal: Gen,
        kbps: u8,
    },
    Lost,
    /// `decode_fec` with the packet.
    Fec(Vec<u8>),
    /// `reset()` and `OPUS_RESET_STATE`.
    Reset,
    /// A decoder control on both sides.
    Set(Setting),
}

#[derive(Arbitrary, Debug, Clone, Copy)]
enum Setting {
    Gain(i32),
    Complexity(i8),
    PhaseInversionDisabled(bool),
    /// The output of the calls from here on.
    Format(crate::diff_decoder::Format),
}

/// The first three variants keep the derived encoding, which the saved
/// reproducers use; the top sixteenth of the variant word, which selected
/// `Lost` there, selects the new operations.
impl<'a> Arbitrary<'a> for Op {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let x = u32::arbitrary(u)?;
        if x >= 0xf000_0000 {
            return Ok(match x % 3 {
                0 => Op::Fec(Arbitrary::arbitrary(u)?),
                1 => Op::Reset,
                _ => Op::Set(Arbitrary::arbitrary(u)?),
            });
        }
        Ok(match (u64::from(x) * 3) >> 32 {
            0 => Op::Packet(Arbitrary::arbitrary(u)?),
            1 => Op::Encoded {
                signal: Arbitrary::arbitrary(u)?,
                kbps: Arbitrary::arbitrary(u)?,
            },
            _ => Op::Lost,
        })
    }
}

struct RefMsDec(*mut c_void);
impl Drop for RefMsDec {
    fn drop(&mut self) {
        unsafe { c::opus_multistream_decoder_destroy(self.0) }
    }
}
struct RefMsEnc(*mut c_void);
impl Drop for RefMsEnc {
    fn drop(&mut self) {
        unsafe { c::opus_multistream_encoder_destroy(self.0) }
    }
}

/// The libFuzzer entry point's body.
pub fn fuzz(data: &[u8]) {
    let Ok(inp) = Input::arbitrary_take_rest(arbitrary::Unstructured::new(data)) else {
        return;
    };
    report("diff_multistream", data, guarded(|| run(inp)));
}

fn run(inp: Input) -> Result<(), Finding> {
    let fs = RATES[inp.rate as usize % 5];
    let channels = 1 + inp.channels % 8;
    let streams = inp.streams % 9;
    let coupled = inp.coupled % 9;
    // Mapping entries below streams+coupled index a decoded channel, 255
    // is silence; other values are invalid and both decoders must say so.
    let mapping: Vec<u8> = inp.mapping[..channels as usize].to_vec();
    let mut err = 0;
    let rd = unsafe {
        c::opus_multistream_decoder_create(
            fs,
            channels as i32,
            streams as i32,
            coupled as i32,
            mapping.as_ptr(),
            &mut err,
        )
    };
    let table = ChannelMappingTable {
        channels,
        stream_count: streams,
        coupled_count: coupled,
        mapping: mapping.clone(),
    };
    let rs = MultistreamDecoder::new(fs, table);
    let layout = format!("ch{channels} s{streams} c{coupled}");
    let (mut rs, rd) = match (rs, rd.is_null()) {
        (Ok(d), false) => (d, RefMsDec(rd)),
        (Err(_), true) => return Ok(()),
        (r, null) => fail!(
            "layout",
            if null { "rs-accepts" } else { "rs-rejects" },
            "{layout} mapping {mapping:?}: opus-rs {}, libopus error {err}",
            if r.is_ok() { "accepts" } else { "rejects" }
        ),
    };
    let enc = {
        let mut e = 0;
        let p = unsafe {
            c::opus_multistream_encoder_create(
                fs,
                channels as i32,
                streams as i32,
                coupled as i32,
                mapping.as_ptr(),
                2049,
                &mut e,
            )
        };
        (!p.is_null()).then_some(RefMsEnc(p))
    };
    let mut synth = Synth::new(fs);
    let frame = fs as usize / 50;
    let max_frame = fs as usize * 120 / 1000;
    let ch = channels as usize;
    let tol = env_f64("OFZ_DEC_TOL", 4e-6) as f32;
    // IJON: the layout's shape.
    ijon_set(u64::from(streams) << 8 | u64::from(coupled) << 4 | u64::from(channels));
    let mut fmt = Format::F32;
    let mut prev = (0u8, 0u8);
    for (k, op) in inp.ops.iter().take(24).enumerate() {
        let pkt: Vec<u8> = match op {
            Op::Packet(p) | Op::Fec(p) => p.clone(),
            Op::Lost => Vec::new(),
            Op::Reset => {
                rs.reset();
                unsafe { c::opus_multistream_decoder_ctl(rd.0, c::RESET_STATE) };
                continue;
            }
            Op::Set(st) => {
                let ctl = |req: c_int, v: i32| unsafe { c::opus_multistream_decoder_ctl(rd.0, req, v) };
                let code = |r: Result<(), &str>| r.map_or_else(|e| opus_rs::error::code(e).unwrap_or(1), |()| 0);
                let (a, b) = match *st {
                    Setting::Gain(g) => (code(rs.set_gain(g)), ctl(c::SET_GAIN, g)),
                    Setting::Complexity(x) => (code(rs.set_complexity(x as i32)), ctl(c::SET_COMPLEXITY, x as i32)),
                    Setting::PhaseInversionDisabled(d) => {
                        rs.set_phase_inversion_disabled(d);
                        (0, ctl(c::SET_PHASE_INVERSION_DISABLED, d as i32))
                    }
                    Setting::Format(f) => {
                        fmt = f;
                        (0, 0)
                    }
                };
                if a != b {
                    fail!("ctl", format!("{st:?}"), "op {k}: opus-rs {a}, libopus {b} {layout}");
                }
                continue;
            }
            Op::Encoded { signal, kbps } => {
                let Some(e) = &enc else { continue };
                let mut pcm = vec![0f32; frame * ch];
                synth.fill(signal, &mut pcm, ch, 0, 1);
                let mut out = vec![0u8; 4000];
                let _ = unsafe {
                    c::opus_multistream_encoder_ctl(
                        e.0,
                        c::SET_BITRATE,
                        (16 + *kbps as i32) * 1000 * streams.max(1) as i32,
                    )
                };
                let n = unsafe {
                    c::opus_multistream_encode_float(
                        e.0,
                        pcm.as_ptr(),
                        frame as i32,
                        out.as_mut_ptr(),
                        out.len() as i32,
                    )
                };
                if n <= 0 {
                    continue;
                }
                out.truncate(n as usize);
                out
            }
        };
        let kind = match op {
            Op::Packet(_) => "packet",
            Op::Lost => "lost",
            Op::Encoded { .. } => "valid",
            Op::Fec(_) => "fec",
            Op::Reset | Op::Set(_) => unreachable!(),
        };
        let fec = matches!(op, Op::Fec(_));
        // IJON: (previous op, op, first stream's ToC configuration).
        let now = (kind.len() as u8 + fmt as u8 * 8, pkt.first().map_or(0, |t| 1 + (t >> 3)));
        ijon_set(u64::from(prev.0) << 24 | u64::from(prev.1) << 16 | u64::from(now.0) << 8 | u64::from(now.1));
        prev = now;
        let size = if pkt.is_empty() { frame } else { max_frame };
        let mut o_ref = vec![0f32; size * ch];
        let mut o_rs = vec![0f32; size * ch];
        let (ptr, len) = if pkt.is_empty() {
            (core::ptr::null(), 0)
        } else {
            (pkt.as_ptr(), pkt.len() as i32)
        };
        let (r_ref, r_rs) = decode_both(&mut rs, rd.0, fmt, &pkt, ptr, len, size, fec, &mut o_rs, &mut o_ref);
        let n = match (r_rs, r_ref) {
            (Ok(n), r) if r >= 0 && n == r as usize => n,
            (Err(_), r) if r < 0 => continue,
            (r, c) => fail!(
                "result",
                format!(
                    "{kind} {}",
                    match &r {
                        Ok(_) => "rs-ok",
                        Err(m) => m,
                    }
                ),
                "op {k} {kind}: opus-rs {r:?}, libopus {c} ({} bytes) mapping {mapping:?}",
                pkt.len()
            ),
        };
        let (d, peak, at) = max_diff(&o_rs[..n * ch], &o_ref[..n * ch]);
        if !(d <= tol * peak + fmt.step()) {
            fail!(
                "pcm",
                format!("{kind} s{streams} c{coupled}"),
                "op {k} {kind}: |opus-rs - libopus| = {d:e} at sample {at} (opus-rs {}, libopus {}) {layout} mapping {mapping:?}",
                o_rs[at],
                o_ref[at]
            );
        }
        // `get`: the final range and the first stream's queries.
        let get = |req| {
            let mut v = 0i32;
            unsafe { c::opus_multistream_decoder_ctl(rd.0, req, &mut v as *mut i32) };
            v
        };
        let want = (
            get(c::GET_FINAL_RANGE) as u32,
            get(c::GET_BANDWIDTH),
            get(c::GET_LAST_PACKET_DURATION),
            get(c::GET_GAIN),
            get(c::GET_COMPLEXITY),
            get(c::GET_PHASE_INVERSION_DISABLED),
            get(c::GET_SAMPLE_RATE),
        );
        let got = (
            rs.final_range(),
            rs.bandwidth().map_or(0, |b| b as i32),
            rs.last_packet_duration() as i32,
            rs.gain(),
            rs.complexity(),
            rs.phase_inversion_disabled() as i32,
            rs.sample_rate(),
        );
        if got != want {
            fail!("get", kind, "op {k} {kind}: opus-rs {got:?}, libopus {want:?} {layout}");
        }
    }
    Ok(())
}

/// One multistream call on both decoders in `fmt`, the output scaled back
/// to float.
#[allow(clippy::too_many_arguments)]
fn decode_both(
    rs: &mut MultistreamDecoder,
    rd: *mut c_void,
    fmt: Format,
    pkt: &[u8],
    ptr: *const u8,
    len: i32,
    size: usize,
    fec: bool,
    o_rs: &mut [f32],
    o_ref: &mut [f32],
) -> (i32, Result<usize, &'static str>) {
    let f = fec as i32;
    match fmt {
        Format::F32 => {
            let r = unsafe { c::opus_multistream_decode_float(rd, ptr, len, o_ref.as_mut_ptr(), size as i32, f) };
            let a = if fec { rs.decode_fec(pkt, size, o_rs) } else { rs.decode(pkt, size, o_rs) };
            (r, a)
        }
        Format::I16 => {
            let (mut x, mut y) = (vec![0i16; o_rs.len()], vec![0i16; o_ref.len()]);
            let r = unsafe { c::opus_multistream_decode(rd, ptr, len, y.as_mut_ptr(), size as i32, f) };
            let a = if fec { rs.decode_fec_i16(pkt, size, &mut x) } else { rs.decode_i16(pkt, size, &mut x) };
            o_rs.iter_mut().zip(&x).for_each(|(o, &v)| *o = v as f32 / 32768.0);
            o_ref.iter_mut().zip(&y).for_each(|(o, &v)| *o = v as f32 / 32768.0);
            (r, a)
        }
        Format::I24 => {
            let (mut x, mut y) = (vec![0i32; o_rs.len()], vec![0i32; o_ref.len()]);
            let r = unsafe { c::opus_multistream_decode24(rd, ptr, len, y.as_mut_ptr(), size as i32, f) };
            let a = if fec { rs.decode_fec24(pkt, size, &mut x) } else { rs.decode24(pkt, size, &mut x) };
            o_rs.iter_mut().zip(&x).for_each(|(o, &v)| *o = v as f32 / 8_388_608.0);
            o_ref.iter_mut().zip(&y).for_each(|(o, &v)| *o = v as f32 / 8_388_608.0);
            (r, a)
        }
    }
}
