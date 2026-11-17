//! Differential encoder fuzzer: generated signals (silence, DC, clipping,
//! low pitch, impulses, voice-like, raw) with settings that change between
//! segments, encoded by opus-rs and by the scalar libopus 1.6.1 reference.
//!
//! opus-rs runs twice: with the x86 AVX/FMA kernels off (`scalar`) and on.
//! Oracles per frame:
//! - `result`: both encoders succeed.
//! - `decode`: libopus decodes every opus-rs packet to a full frame, and its
//!   final range equals the opus-rs encoder's.
//! - `cbr`: in CBR, opus-rs packets have libopus's length.
//! - `lookahead`: same lookahead as libopus.
//! - `toc`: same mode, bandwidth, frame duration and channel flag
//!   (suppress with `OFZ_SUPPRESS=toc`).
//! - `bytes` (`OFZ_STRICT_BYTES=1`, scalar only): byte-identical packets in
//!   CELT-only streams (`RestrictedLowDelay`), where opus-rs and libopus run
//!   the same float algorithm. The first differing packet is reported.
//! - `silk-input`: the 16-bit samples handed to SILK are identical.
//! - `fidelity`: per-frame SNR against the input, both streams decoded by
//!   libopus. Flags frames where opus-rs is below 3 dB while libopus is at
//!   least 6 dB better.
//!
//! With `--features dnn` the settings also choose a DRED duration (both
//! encoders, so the packets carry the DRED extension and its bits).
#![no_main]

#[macro_use]
mod common;

use arbitrary::Arbitrary;
use common::*;
use libfuzzer_sys::fuzz_target;
use opus_rs::{Application, Bandwidth, OpusEncoder, Signal};
use opusref as c;

#[derive(Arbitrary, Debug)]
struct Input {
    rate: u8,
    stereo: bool,
    app: u8,
    frame: u8,
    segs: Vec<Seg>,
}

#[derive(Arbitrary, Debug)]
struct Seg {
    frames: u8,
    signal: Gen,
    set: Option<Settings>,
    call: Call,
}

/// How the segment's frames are encoded: API arguments that vary per call.
#[derive(Arbitrary, Debug)]
struct Call {
    /// A new frame duration from this segment on (index into DURATIONS).
    frame: Option<u8>,
    /// The output buffer size in bytes for this segment's calls (0-1599);
    /// 1500 otherwise.
    budget: Option<u16>,
    /// Encode through the 16-bit entry points (`encode_i16`/`opus_encode`).
    int16: bool,
    /// First try one call with an arbitrary frame size.
    bad_frame: Option<u32>,
}

/// 2.5, 5, 10, 20, 40, 60, 80, 100 and 120 ms in units of 2.5 ms.
const DURATIONS: [usize; 9] = [1, 2, 4, 8, 16, 24, 32, 40, 48];

#[derive(Arbitrary, Debug, Clone)]
struct Settings {
    bitrate: Bitrate,
    complexity: u8,
    /// 0 VBR, 1 constrained VBR, 2 CBR.
    rc: u8,
    fec: bool,
    loss: u8,
    dtx: bool,
    signal: u8,
    bandwidth: u8,
    max_bandwidth: u8,
    /// With the `dnn` feature: the DRED duration in 10 ms frames (0-104).
    #[cfg(feature = "dnn")]
    dred: u8,
}

#[derive(Arbitrary, Debug, Clone, Copy)]
enum Bitrate {
    Common(u8),
    /// Anything libopus accepts, 500 b/s to 4 Mb/s; libopus clamps above
    /// 750 kb/s per channel.
    Any(u32),
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            bitrate: Bitrate::Common(7),
            complexity: 10,
            rc: 1,
            fec: false,
            loss: 0,
            dtx: false,
            signal: 0,
            bandwidth: 0,
            max_bandwidth: 0,
            #[cfg(feature = "dnn")]
            dred: 0,
        }
    }
}

const COMMON_KBPS: [i32; 16] = [
    6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 192, 256, 510,
];
const BW: [Bandwidth; 6] = [
    Bandwidth::Auto,
    Bandwidth::Narrowband,
    Bandwidth::Mediumband,
    Bandwidth::Wideband,
    Bandwidth::Superwideband,
    Bandwidth::Fullband,
];

fn bitrate(b: Bitrate) -> i32 {
    match b {
        Bitrate::Common(i) => COMMON_KBPS[i as usize % 16] * 1000,
        Bitrate::Any(v) => 500 + (v % 3_999_500) as i32,
    }
}

fn apply(rs: &mut [&mut OpusEncoder; 2], re: &mut RefEnc, s: &Settings) {
    let br = bitrate(s.bitrate);
    let cx = (s.complexity % 11) as i32;
    let loss = (s.loss % 101) as i32;
    let sig = [Signal::Auto, Signal::Voice, Signal::Music][s.signal as usize % 3];
    let bw = BW[s.bandwidth as usize % 6];
    let mbw = BW[1 + s.max_bandwidth as usize % 5];
    for e in rs.iter_mut() {
        e.bitrate_bps = br;
        e.complexity = cx;
        e.use_cbr = s.rc % 3 == 2;
        e.set_vbr_constraint(s.rc % 3 == 1);
        e.use_inband_fec = s.fec;
        e.packet_loss_perc = loss;
        e.set_dtx(s.dtx);
        e.set_signal(sig);
        e.set_bandwidth(bw);
        e.set_max_bandwidth(mbw);
    }
    re.ctl(c::SET_BITRATE, br);
    re.ctl(c::SET_COMPLEXITY, cx);
    re.ctl(c::SET_VBR, (s.rc % 3 != 2) as i32);
    re.ctl(c::SET_VBR_CONSTRAINT, (s.rc % 3 == 1) as i32);
    re.ctl(c::SET_INBAND_FEC, s.fec as i32);
    re.ctl(c::SET_PACKET_LOSS_PERC, loss);
    re.ctl(c::SET_DTX, s.dtx as i32);
    re.ctl(c::SET_SIGNAL, sig as i32);
    re.ctl(c::SET_BANDWIDTH, bw as i32);
    re.ctl(c::SET_MAX_BANDWIDTH, mbw as i32);
    // OPUS_SET_DRED_DURATION; the packet loss above sets DRED's bitrate.
    #[cfg(feature = "dnn")]
    {
        let dred = (s.dred % 105) as i32;
        for e in rs.iter_mut() {
            e.set_dred_duration(dred).unwrap();
        }
        re.ctl(4050, dred);
    }
}

fn mode_name(toc: u8) -> &'static str {
    match toc >> 3 {
        0..=11 => "silk",
        12..=15 => "hybrid",
        _ => "celt",
    }
}

/// Per-frame SNR (dB) of `dec` against `inp`, both `n` interleaved samples.
fn snr(inp: &[f32], dec: &[f32]) -> f64 {
    let s: f64 = inp.iter().map(|&x| (x as f64).powi(2)).sum();
    let e: f64 = inp
        .iter()
        .zip(dec)
        .map(|(&x, &y)| ((x - y) as f64).powi(2))
        .sum();
    10.0 * (s / e.max(1e-20)).log10()
}

fuzz_target!(|data: &[u8]| {
    let Ok(inp) = Input::arbitrary_take_rest(arbitrary::Unstructured::new(data)) else {
        return;
    };
    report("diff_encoder", data, guarded(|| run(inp)));
});

fn run(inp: Input) -> Result<(), Finding> {
    let fs = RATES[inp.rate as usize % 5];
    let ch = 1 + inp.stereo as usize;
    let (app, capp) = [
        (Application::Voip, 2048),
        (Application::Audio, 2049),
        (Application::RestrictedLowDelay, 2051),
    ][inp.app as usize % 3];
    let mut frame = fs as usize / 400 * DURATIONS[inp.frame as usize % 9];
    let strict = std::env::var("OFZ_STRICT_BYTES").is_ok() && capp == 2051;

    let mut e_sc = OpusEncoder::new(fs, ch, app).unwrap();
    let mut e_si = OpusEncoder::new(fs, ch, app).unwrap();
    let mut re = RefEnc::new(fs, ch, capp);
    let (mut d_sc, mut d_si, mut d_ref) = (
        RefDec::new(fs, ch),
        RefDec::new(fs, ch),
        RefDec::new(fs, ch),
    );
    apply(&mut [&mut e_sc, &mut e_si], &mut re, &Settings::default());
    let la = re.get(c::GET_LOOKAHEAD) as usize;
    if e_sc.lookahead() != la {
        fail!(
            "lookahead",
            format!("app{capp} fs{fs}"),
            "opus-rs {}, libopus {la}",
            e_sc.lookahead()
        );
    }

    let mut synth = Synth::new(fs);
    // Input history, so decoded frames can be aligned by the lookahead.
    let mut hist: Vec<f32> = vec![0.0; la * ch];
    let mut identical = true;
    let mut cur = Settings::default();
    let mut total = 0;
    let mut pcm = vec![0f32; frame * ch];
    for seg in inp.segs.iter().take(12) {
        if let Some(s) = &seg.set {
            apply(&mut [&mut e_sc, &mut e_si], &mut re, s);
            cur = s.clone();
        }
        if let Some(f) = seg.call.frame {
            frame = fs as usize / 400 * DURATIONS[f as usize % 9];
            pcm.resize(frame * ch, 0.0);
        }
        let budget = seg.call.budget.map_or(1500, |b| b as usize % 1600);
        // An invalid frame size must fail in both encoders without changing
        // their state.
        if let Some(b) = seg.call.bad_frame {
            let bad = b as usize % 20000;
            if !DURATIONS.iter().any(|&d| d * fs as usize / 400 == bad) {
                let x = vec![0.1f32; bad.max(1) * ch];
                let (mut o1, mut o2) = ([0u8; 1500], [0u8; 1500]);
                let r = re.encode(&x, bad, &mut o1);
                set_simd(false);
                let rs = e_sc.encode(&x, bad, &mut o2);
                if (r > 0) != rs.is_ok() {
                    fail!(
                        "result",
                        "bad frame size",
                        "frame size {bad}: opus-rs {rs:?}, libopus {r} fs{fs} ch{ch}"
                    );
                }
                if r > 0 {
                    // libopus accepted it (e.g. a multiple it supports): the
                    // encoders' states moved on; stop comparing this input.
                    return Ok(());
                }
            }
        }
        let n = 1 + seg.frames as usize % 8;
        for i in 0..n {
            total += 1;
            if total > 48 {
                return Ok(());
            }
            synth.fill(&seg.signal, &mut pcm, ch, i, n);
            hist.extend_from_slice(&pcm);
            let (mut p_sc, mut p_si, mut p_ref) = ([0u8; 1600], [0u8; 1600], [0u8; 1600]);
            let pcm16: Vec<i16> = pcm
                .iter()
                .map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
                .collect();
            let r_ref = if seg.call.int16 {
                re.encode_i16(&pcm16, frame, &mut p_ref[..budget])
            } else {
                re.encode(&pcm, frame, &mut p_ref[..budget])
            };
            set_simd(false);
            let r_sc = if seg.call.int16 {
                e_sc.encode_i16(&pcm16, frame, &mut p_sc[..budget])
            } else {
                e_sc.encode(&pcm, frame, &mut p_sc[..budget])
            };
            // `silk-input`: both encoders hand SILK the same 16-bit samples
            // (after the high-pass or DC filter and the conversion).
            let (si_rs, si_ref) = (opus_rs::fuzzing::take_silk_input(), c::take_silk_input());
            if !si_rs.is_empty() && !si_ref.is_empty() && !suppressed("silk-input") {
                if let Some(i) = (0..si_rs.len().min(si_ref.len())).find(|&i| si_rs[i] != si_ref[i])
                {
                    let worst = si_rs
                        .iter()
                        .zip(&si_ref)
                        .map(|(a, b)| (*a as i32 - *b as i32).abs())
                        .max()
                        .unwrap_or(0);
                    fail!(
                        "silk-input",
                        format!("app{capp} {}", if worst <= 1 { "lsb" } else { "large" }),
                        "frame {total}: sample {i} opus-rs {}, libopus {} (largest difference {worst}) fs{fs} ch{ch} gen {:?}",
                        si_rs[i],
                        si_ref[i],
                        seg.signal
                    );
                }
            }
            // `OFZ_SUPPRESS=simd` skips the SIMD encoder (see fuzz/DIFFERENTIAL.md).
            let r_si = if suppressed("simd") {
                p_si = p_sc;
                r_sc
            } else {
                set_simd(true);
                if seg.call.int16 {
                    e_si.encode_i16(&pcm16, frame, &mut p_si[..budget])
                } else {
                    e_si.encode(&pcm, frame, &mut p_si[..budget])
                }
            };
            let br = bitrate(cur.bitrate);
            let ctx = format!(
                "fs{fs} ch{ch} app{capp} {}ms rc{} br{}k budget{budget}{}",
                frame * 1000 / fs as usize,
                cur.rc % 3,
                br / 1000,
                if seg.call.int16 { " i16" } else { "" }
            );
            // `result`: both encoders accept or both reject the call.
            let n_ref = match (r_ref, &r_sc) {
                (n, Ok(_)) if n > 0 => n as usize,
                (n, Err(_)) if n <= 0 => {
                    // Rejected by both (e.g. a too small buffer): nothing was
                    // coded, so this frame leaves the history.
                    hist.truncate(hist.len() - frame * ch);
                    continue;
                }
                (n, r) => fail!(
                    "result",
                    format!(
                        "{} budget {}",
                        if n > 0 { "rs-err" } else { "rs-ok" },
                        if budget < 3 {
                            "tiny"
                        } else if budget < 1275 {
                            "short"
                        } else {
                            "full"
                        }
                    ),
                    "libopus {n}, opus-rs {r:?} at frame {total} {ctx}"
                ),
            };
            let mut o_ref = vec![0f32; frame * ch];
            let (_, ref_syms) =
                c::trace(|| d_ref.decode(Some(&p_ref[..n_ref]), &mut o_ref, frame, false));
            let mut packets = [
                ("scalar", r_sc, &p_sc, &mut e_sc, &mut d_sc),
                ("simd", r_si, &p_si, &mut e_si, &mut d_si),
            ];
            let mut decoded = Vec::new();
            for (tag, r, p, e, d) in packets.iter_mut() {
                let len = match r {
                    Ok(n) => *n,
                    Err(m) => fail!(
                        "result",
                        format!("{tag} {m}"),
                        "opus-rs {tag}: {m} at frame {total} {ctx}"
                    ),
                };
                let pkt = &p[..len];
                let mut out = vec![0f32; frame * ch];
                let (got, syms) = c::trace(|| d.decode(Some(pkt), &mut out, frame, false));
                if got != frame as i32 {
                    fail!(
                        "decode",
                        format!("{tag} {}", mode_name(pkt[0])),
                        "libopus returns {got} for opus-rs {tag} packet {total} ({len} bytes) {ctx}"
                    );
                }
                let skip = *tag == "simd" && suppressed("simd");
                if skip {
                    decoded.push(out);
                    continue;
                }
                if len > 1 && d.final_range() != e.final_range() {
                    fail!(
                        "decode",
                        format!("{tag} range {}", mode_name(pkt[0])),
                        "frame {total}: encoder range {:#x}, decoder {:#x} {ctx}",
                        e.final_range(),
                        d.final_range()
                    );
                }
                if cur.rc % 3 == 2 && !cur.dtx && len != n_ref {
                    fail!(
                        "cbr",
                        format!(
                            "{tag} {}ms {}",
                            frame * 1000 / fs as usize,
                            if br > 750_000 * ch as i32 {
                                ">750k/ch"
                            } else {
                                ""
                            }
                        ),
                        "frame {total}: opus-rs {tag} {len} bytes, libopus {n_ref} {ctx}"
                    );
                }
                if *tag == "scalar" && !suppressed("toc") && pkt[0] & 0xfc != p_ref[0] & 0xfc {
                    fail!(
                        "toc",
                        format!("{}->{}", mode_name(p_ref[0]), mode_name(pkt[0])),
                        "frame {total}: opus-rs toc {:#04x}, libopus {:#04x} {ctx}",
                        pkt[0],
                        p_ref[0]
                    );
                }
                if *tag == "scalar" && strict && identical && pkt != &p_ref[..n_ref] {
                    // Localize: the first symbol the reference decoder reads
                    // differently from the two packets is the first coded
                    // decision where the encoders disagree.
                    let at = syms.iter().zip(&ref_syms).position(|(a, b)| a != b);
                    let site = match at {
                        Some(i) => format!("{}:{}", ref_syms[i].file, ref_syms[i].line),
                        None if len != n_ref => "length".into(),
                        None => "tail".into(),
                    };
                    let detail = at
                        .map(|i| {
                            let before: Vec<String> = ref_syms[i.saturating_sub(4)..i]
                                .iter()
                                .map(|s| format!("{}:{}={}", s.file, s.line, s.val))
                                .collect();
                            format!(
                                "symbol {i}: opus-rs {:?}, libopus {:?}, after [{}]",
                                syms[i],
                                ref_syms[i],
                                before.join(" ")
                            )
                        })
                        .unwrap_or_default();
                    if std::env::var("OFZ_DEBUG").is_ok() {
                        eprintln!("opus-rs {pkt:02x?}\nlibopus {:02x?}", &p_ref[..n_ref]);
                    }
                    fail!(
                        "bytes",
                        format!("{} {site}", mode_name(p_ref[0])),
                        "frame {total}: {detail}; opus-rs {len} bytes, libopus {n_ref} {ctx} gen {:?}",
                        seg.signal
                    );
                }
                ijon_set(hash3(
                    mode_name(pkt[0]),
                    (pkt[0] >> 3) as u64,
                    (len.min(1275) / 64) as u64,
                ));
                decoded.push(out);
            }
            identical &= p_sc[..r_sc.unwrap()] == p_ref[..n_ref];
            // Fidelity, once the lookahead has passed.
            let start = hist.len() - (la + frame) * ch;
            if total > 2 {
                let x = &hist[start..start + frame * ch];
                let energy = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
                if energy > 1e-5 {
                    let s_ref = snr(x, &o_ref);
                    for (tag, o) in ["scalar", "simd"].iter().zip(&decoded) {
                        let s = snr(x, o);
                        if s < 3.0 && s_ref - s > 6.0 {
                            fail!(
                                "fidelity",
                                format!("{tag} {}", mode_name(p_ref[0])),
                                "frame {total}: opus-rs {tag} {s:.1} dB, libopus {s_ref:.1} dB {ctx} gen {:?}",
                                seg.signal
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn hash3(a: &str, b: u64, c: u64) -> u64 {
    a.len() as u64 * 0x1_0000_0000 ^ b << 16 ^ c
}
