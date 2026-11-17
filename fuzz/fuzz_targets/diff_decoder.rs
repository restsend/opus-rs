//! Differential decoder fuzzer: sequences of API calls (decode, decode_fec,
//! loss concealment, reset) with arbitrary packets and frame sizes, run on
//! opus-rs and on the scalar libopus 1.6.1 reference.
//!
//! Oracles, after every call:
//! - `result`: same success/error and sample count as libopus.
//! - `range`: same final range as libopus (`OPUS_GET_FINAL_RANGE`).
//! - `pcm`: opus-rs with SIMD off matches libopus within `OFZ_DEC_TOL`
//!   (relative to max(1, peak); default 4e-6: CELT float rounding
//!   differences stay below it, one SILK LSB, 3e-5, is above it), and
//!   `OFZ_PLC_TOL` for concealed frames.
//! - `simd`: opus-rs with SIMD on matches opus-rs with SIMD off within
//!   `OFZ_SIMD_TOL` (default 1e-5) and gives the same result and range.
//! - Absurd frame sizes must not panic.
#![no_main]

#[macro_use]
mod common;

use arbitrary::Arbitrary;
use common::*;
use libfuzzer_sys::fuzz_target;
use opus_rs::OpusDecoder;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

fn hash<T: Hash>(v: &T) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    v.hash(&mut h);
    h.finish()
}

fn discriminant(op: &Op) -> u8 {
    match op {
        Op::Decode { .. } => 0,
        Op::Fec { .. } => 1,
        Op::Lost { .. } => 2,
        Op::Reset => 3,
        Op::Encoded { .. } => 4,
    }
}

#[derive(Arbitrary, Debug)]
struct Input {
    rate: u8,
    stereo: bool,
    /// Only libopus-encoded packets and losses: conformant streams, where
    /// SILK must decode bit-exactly.
    valid_only: bool,
    /// With the `dnn` feature: the decoder complexity (deep PLC from 5,
    /// LACE at 6, NoLACE from 7) and the bandwidth extension.
    #[cfg(feature = "dnn")]
    complexity: u8,
    #[cfg(feature = "dnn")]
    bwe: bool,
    ops: Vec<Op>,
}

/// A new opus-rs decoder with the input's neural network settings.
fn new_dec(fs: i32, ch: usize, inp: &Input) -> OpusDecoder {
    #[allow(unused_mut)]
    let mut d = OpusDecoder::new(fs, ch).unwrap();
    #[cfg(feature = "dnn")]
    {
        d.set_complexity(inp.complexity as i32 % 11).unwrap();
        d.set_osce_bwe(inp.bwe);
    }
    let _ = inp;
    d
}

#[derive(Arbitrary, Debug)]
enum Op {
    Decode {
        pkt: Vec<u8>,
        size: Size,
    },
    /// A valid packet from the libopus encoder; `force` selects SILK, hybrid
    /// or CELT so mode transitions occur in otherwise conformant streams.
    Encoded {
        signal: Gen,
        kbps: u8,
        ms: u8,
        force: u8,
    },
    Fec {
        pkt: Vec<u8>,
        size: Size,
    },
    Lost {
        size: Size,
    },
    Reset,
}

#[derive(Arbitrary, Debug, Clone, Copy)]
enum Size {
    /// 120 ms, the usual "whole buffer" request.
    Max,
    /// Multiple of 2.5 ms.
    Ms(u8),
    /// Any value, including nonsense.
    Raw(u64),
}

const CAP: usize = 5760 * 4;

struct Tol {
    dec: f32,
    plc: f32,
    simd: f32,
}
fn tol() -> &'static Tol {
    static T: OnceLock<Tol> = OnceLock::new();
    T.get_or_init(|| Tol {
        dec: env_f64("OFZ_DEC_TOL", 4e-6) as f32,
        plc: env_f64("OFZ_PLC_TOL", 4e-6) as f32,
        simd: env_f64("OFZ_SIMD_TOL", 1e-5) as f32,
    })
}

fn call(d: &mut OpusDecoder, op: &Op, fs: usize, out: &mut [f32]) -> Result<usize, &'static str> {
    match op {
        Op::Decode { pkt, .. } => d.decode(pkt, fs, out),
        Op::Fec { pkt, .. } => d.decode_fec(pkt, fs, out),
        Op::Lost { .. } => d.decode(&[], fs, out),
        Op::Reset | Op::Encoded { .. } => unreachable!(),
    }
}

/// Encodes one frame with the reference encoder for an `Op::Encoded`.
fn encode(
    enc: &mut RefEnc,
    synth: &mut Synth,
    fs: i32,
    ch: usize,
    signal: &Gen,
    kbps: u8,
    ms: u8,
    force: u8,
) -> Vec<u8> {
    const KBPS: [i32; 10] = [6, 9, 12, 16, 20, 24, 32, 48, 64, 128];
    enc.ctl(opusref::SET_BITRATE, KBPS[kbps as usize % 10] * 1000);
    // OPUS_SET_FORCE_MODE (opus_private.h): auto, SILK, hybrid, CELT.
    enc.ctl(11002, [-1000, 1000, 1001, 1002][force as usize % 4]);
    let frame = fs as usize / 400 * [4, 8, 16, 24][ms as usize % 4];
    let mut pcm = vec![0f32; frame * ch];
    synth.fill(signal, &mut pcm, ch, 0, 1);
    let mut out = [0u8; 1500];
    let n = enc.encode(&pcm, frame, &mut out);
    out[..n.max(0) as usize].to_vec()
}

fuzz_target!(|data: &[u8]| {
    let Ok(inp) = Input::arbitrary_take_rest(arbitrary::Unstructured::new(data)) else {
        return;
    };
    report("diff_decoder", data, guarded(|| run(inp)));
});

/// Mode/bandwidth/duration class of a packet, for grouping findings.
fn class(pkt: &[u8]) -> String {
    match pkt.first() {
        None => "empty".into(),
        Some(_) if pkt.len() == 1 => "dtx".into(),
        Some(&t) => {
            let cfg = t >> 3;
            let mode = if cfg < 12 {
                "silk"
            } else if cfg < 16 {
                "hybrid"
            } else {
                "celt"
            };
            format!("{mode}{}c{}", if t & 4 != 0 { "S" } else { "M" }, t & 3)
        }
    }
}

fn run(inp: Input) -> Result<(), Finding> {
    let fs = RATES[inp.rate as usize % RATES.len()];
    let ch = 1 + inp.stereo as usize;
    let mut scalar = new_dec(fs, ch, &inp);
    let mut simd = new_dec(fs, ch, &inp);
    let mut refd = RefDec::new(fs, ch);
    // OPUS_SET_COMPLEXITY and OPUS_SET_OSCE_BWE, which a reset keeps.
    #[cfg(feature = "dnn")]
    {
        refd.ctl(4010, inp.complexity as i32 % 11);
        refd.ctl(4054, inp.bwe as i32);
    }
    let t = tol();
    let mut prev_cls = String::new();
    let mut enc = None;
    let mut synth = Synth::new(fs);
    let mut valid = true;

    if std::env::var("OFZ_DEBUG").is_ok() {
        eprintln!("script {fs} {ch}");
    }
    for (k, op) in inp.ops.iter().take(48).enumerate() {
        let generated;
        let op = match op {
            Op::Encoded {
                signal,
                kbps,
                ms,
                force,
            } => {
                let e = enc.get_or_insert_with(|| RefEnc::new(fs, ch, 2049));
                generated = Op::Decode {
                    pkt: encode(e, &mut synth, fs, ch, signal, *kbps, *ms, *force),
                    size: Size::Max,
                };
                &generated
            }
            Op::Decode { .. } | Op::Fec { .. } if inp.valid_only => continue,
            op => {
                valid &= matches!(op, Op::Lost { .. } | Op::Reset);
                op
            }
        };
        if std::env::var("OFZ_DEBUG").is_ok() {
            // One line per call, replayable: `fs ch` then `d|f <hex> <n>`,
            // `l <n>`, `r`.
            let hex = |p: &[u8]| p.iter().map(|b| format!("{b:02x}")).collect::<String>();
            let n = |s: &Size| match *s {
                Size::Max => fs as usize * 120 / 1000,
                Size::Ms(m) => (m as usize % 48 + 1) * fs as usize / 400,
                Size::Raw(r) => r as usize,
            };
            match op {
                Op::Decode { pkt, size } => eprintln!("script d {} {}", hex(pkt), n(size)),
                Op::Fec { pkt, size } => eprintln!("script f {} {}", hex(pkt), n(size)),
                Op::Lost { size } => eprintln!("script l {}", n(size)),
                Op::Reset => eprintln!("script r"),
                Op::Encoded { .. } => {}
            }
        }
        let size = match op {
            Op::Reset => {
                // opus-rs has no reset; a new decoder is the same state.
                scalar = new_dec(fs, ch, &inp);
                simd = new_dec(fs, ch, &inp);
                refd.reset();
                continue;
            }
            Op::Decode { size, .. } | Op::Fec { size, .. } | Op::Lost { size } => *size,
            Op::Encoded { .. } => unreachable!(),
        };
        let frame = match size {
            Size::Max => fs as usize * 120 / 1000,
            Size::Ms(m) => (m as usize % 48 + 1) * fs as usize / 400,
            Size::Raw(r) => r as usize,
        };
        if frame > CAP {
            // Only the no-panic property is checked; libopus takes an int and
            // a caller-sized buffer, so it has no equivalent call.
            let mut out = vec![0f32; CAP * ch];
            set_simd(false);
            let _ = call(&mut scalar, op, frame, &mut out);
            return Ok(());
        }
        // Known divergences (see fuzz/DIFFERENTIAL.md); skipping ends the
        // sequence, since the decoders' states may differ afterwards.
        let pkt = match op {
            Op::Decode { pkt, .. } | Op::Fec { pkt, .. } => Some(pkt.as_slice()),
            _ => None,
        };
        let parsed = pkt.map(parse_packet);
        if let Some(p) = &parsed {
            // opus-rs treats a 0/1-byte packet as loss even when libopus
            // rejects it, and decodes 0/1-byte frames inside longer packets
            // that libopus conceals.
            let short = pkt.unwrap().len() <= 1;
            let tiny = (short && p.is_none())
                || (!short && p.as_ref().is_some_and(|f| f.iter().any(|&s| s <= 1)));
            if tiny && suppressed("tiny-frame") {
                return Ok(());
            }
            if p.is_none() && matches!(op, Op::Fec { .. }) && suppressed("fec-invalid") {
                return Ok(());
            }
        }
        // IJON: reward new (previous packet class, call, packet class) triples
        // and multi-frame packets, so mode/rate transitions get explored.
        let cls_now = pkt.map(class).unwrap_or_default();
        ijon_set(hash(&(
            prev_cls.clone(),
            discriminant(op),
            cls_now.clone(),
            frame / 120,
        )));
        let prev_mode = std::mem::replace(&mut prev_cls, cls_now);
        let mut o_ref = vec![0f32; frame.max(1) * ch];
        let mut o_sc = o_ref.clone();
        let mut o_si = o_ref.clone();
        let r_ref = match op {
            Op::Decode { pkt, .. } => refd.decode(Some(pkt), &mut o_ref, frame, false),
            Op::Fec { pkt, .. } => refd.decode(Some(pkt), &mut o_ref, frame, true),
            _ => refd.decode(None, &mut o_ref, frame, false),
        };
        set_simd(false);
        let r_sc = call(&mut scalar, op, frame, &mut o_sc);
        set_simd(true);
        let r_si = call(&mut simd, op, frame, &mut o_si);

        let (kind, cls) = match op {
            Op::Decode { pkt, .. } => ("decode", class(pkt)),
            Op::Fec { pkt, .. } => ("decode_fec", class(pkt)),
            _ => ("lost", String::new()),
        };
        // Group by call, previous and current packet mode (not rate or size).
        let mode = |c: &str| {
            c.trim_end_matches(|ch: char| ch.is_ascii_digit())
                .trim_end_matches(['M', 'S', 'c'])
                .to_string()
        };
        let stream = if valid { "valid " } else { "" };
        let sig =
            |what: &str| format!("{stream}{kind} {}->{} {what}", mode(&prev_mode), mode(&cls));
        let n = match (r_sc, r_ref) {
            (Ok(n), c) if c >= 0 && n == c as usize => n,
            (Err(_), c) if c < 0 => {
                if r_si.is_ok() {
                    fail!(
                        "simd",
                        sig("ok-vs-err"),
                        "op {k} {kind}: SIMD Ok, scalar Err"
                    );
                }
                continue;
            }
            (r, c) => fail!(
                "result",
                sig(if r.is_ok() { "rs-ok" } else { "rs-err" }),
                "op {k} {kind} frame={frame}: opus-rs {r:?}, libopus {c}"
            ),
        };
        if r_si != r_sc {
            fail!(
                "simd",
                sig("result"),
                "op {k} {kind}: SIMD {r_si:?}, scalar {r_sc:?}"
            );
        }
        let (rr, rs) = (refd.final_range(), scalar.final_range());
        if rr != rs {
            fail!(
                "range",
                sig(""),
                "op {k} {kind}: opus-rs {rs:#010x}, libopus {rr:#010x}"
            );
        }
        if simd.final_range() != rs {
            fail!("simd", sig("range"), "op {k} {kind}: SIMD range differs");
        }
        if std::env::var("OFZ_DEBUG").is_ok() {
            let pk = |v: &[f32]| v.iter().fold(0f32, |a, x| a.max(x.abs()));
            if let Op::Decode { pkt, .. } | Op::Fec { pkt, .. } = op {
                eprintln!(
                    "pkt {}",
                    pkt.iter().map(|b| format!("{b:02x}")).collect::<String>()
                );
            }
            eprintln!(
                "op {k} {kind} {cls} n={n} ref_peak={} rs_peak={} ref[..8]={:?} rs[..8]={:?}",
                pk(&o_ref[..n * ch]),
                pk(&o_sc[..n * ch]),
                &o_ref[..8.min(n * ch)],
                &o_sc[..8.min(n * ch)]
            );
        }
        let (d, peak, at) = max_diff(&o_sc[..n * ch], &o_ref[..n * ch]);
        let lim = if kind == "decode" { t.dec } else { t.plc } * peak;
        if !(d <= lim) {
            // Where the differences are: a transition window shows up as a
            // short span at the start of the frame.
            let bad: Vec<usize> = (0..n * ch)
                .filter(|&i| !((o_sc[i] - o_ref[i]).abs() <= lim))
                .collect();
            let span = format!(
                "{} samples differ in [{}, {}] of {}",
                bad.len(),
                bad[0] / ch,
                bad[bad.len() - 1] / ch,
                n
            );
            let head = bad[bad.len() - 1] / ch < fs as usize / 400;
            fail!(
                "pcm",
                sig(
                    if o_ref[at].abs() > 1.0
                        && o_sc[at].abs() == 1.0
                        && o_sc[at].signum() == o_ref[at].signum()
                    {
                        "clamp"
                    } else if head {
                        "first-2.5ms"
                    } else {
                        ""
                    }
                ),
                "op {k} {kind}: |opus-rs - libopus| = {d:e} at sample {at} (opus-rs {}, libopus {}), limit {lim:e}; {span}",
                o_sc[at],
                o_ref[at]
            );
        }
        let (d, peak, at) = max_diff(&o_si[..n * ch], &o_sc[..n * ch]);
        if !(d <= t.simd * peak) {
            fail!(
                "simd",
                sig("pcm"),
                "op {k} {kind}: |SIMD - scalar| = {d:e} at sample {at}"
            );
        }
    }
    Ok(())
}
