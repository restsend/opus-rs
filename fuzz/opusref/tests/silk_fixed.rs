//! SILK-only packets of opus-rs against fixed-point libopus, whose SILK
//! encoder (silk/fixed/) opus-rs ports, over a grid of SILK rates (fixed, or
//! switching every 200 ms), input rates, channels, frame durations,
//! complexities, VBR/CBR, FEC and DTX, on a generated signal with voiced,
//! high-pitched, loud, noisy and silent parts. Runs with
//! `cargo test --release --features fixed --test silk_fixed`.
//!
//! Each configuration reports the first frame where the TOCs differ (`toc`),
//! where the 16-bit samples handed to SILK differ (`input`), where one encoder
//! sends a DTX packet (`dtx`) or adds a redundant CELT frame (`redundancy`),
//! or where SILK-only packets differ (`packet`). The first four are
//! Opus-layer decisions, printed; the test fails on `packet`. Packets match
//! when they and the final ranges are identical, or when libopus's decoder,
//! tracing its range-decoder calls, reads the same SILK symbols from both:
//! packets around a mode or internal rate switch carry a CELT frame, which
//! libopus codes with fixed-point CELT. CELT-only packets of both encoders
//! are skipped.
#![cfg(feature = "fixed")]
use opus_rs::{Application, Bandwidth, OpusEncoder, Signal};
use opusref::*;

/// The traced libopus files that decode SILK symbols (wrap/).
const SILK_DECODER: [&str; 6] = [
    "dec_API.c",
    "decode_indices.c",
    "decode_pulses.c",
    "code_signs.c",
    "shell_coder.c",
    "stereo_decode_pred.c",
];

const BW: [Bandwidth; 3] = [
    Bandwidth::Narrowband,
    Bandwidth::Mediumband,
    Bandwidth::Wideband,
];

/// One second of interleaved test signal at `fs`: 200 ms each of a voiced
/// sweep, noise, silence, a high-pitched voice and a loud voice.
fn signal(fs: usize, ch: usize) -> Vec<i16> {
    let mut seed = 1u32;
    let mut noise = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 16) as f32 / 32768.0 - 1.0
    };
    let mut phase = 0f32;
    let mut out = Vec::with_capacity(fs * ch);
    for i in 0..fs {
        let t = i as f32 / fs as f32;
        let (f0, amp, hiss) = match i * 5 / fs {
            0 => (100.0 + 750.0 * t, 6000.0, 0.02),
            1 => (0.0, 0.0, 0.3),
            2 => (0.0, 0.0, 0.0),
            3 => (480.0 + 40.0 * (t * 30.0).sin(), 8000.0, 0.01),
            _ => (140.0, 30000.0, 0.0),
        };
        phase = (phase + core::f32::consts::TAU * f0 / fs as f32) % core::f32::consts::TAU;
        let voiced: f32 = (1..12).map(|k| (phase * k as f32).sin() / k as f32).sum();
        let x = amp * voiced + 8000.0 * hiss * noise();
        out.push(x.clamp(-32768.0, 32767.0) as i16);
        if ch == 2 {
            out.push((0.6 * x + 1500.0 * noise()).clamp(-32768.0, 32767.0) as i16);
        }
    }
    out
}

#[derive(Debug, Clone, Copy)]
struct Cfg {
    /// Input rate.
    fs: i32,
    /// Bandwidth (index into BW) of each 200 ms part of the signal, repeated.
    bws: &'static [usize],
    ch: usize,
    ms: usize,
    cx: i32,
    cbr: bool,
    fec: bool,
    dtx: bool,
    /// Bitrate of each 200 ms part, alternating.
    rates: [i32; 2],
}

/// The SILK input hooks and the symbol trace are process-wide: one test at a time.
static HOOKS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Encodes the signal with both encoders; the first difference, if any.
fn run(c: Cfg, pcm: &[i16]) -> Option<(usize, &'static str)> {
    let _hooks = HOOKS.lock().unwrap_or_else(|e| e.into_inner());
    let n = c.fs as usize * c.ms / 1000;
    let mut rs = OpusEncoder::new(c.fs, c.ch, Application::Voip).unwrap();
    rs.bitrate_bps = c.rates[0];
    rs.complexity = c.cx;
    rs.use_cbr = c.cbr;
    rs.use_inband_fec = c.fec;
    rs.packet_loss_perc = if c.fec { 20 } else { 0 };
    rs.set_dtx(c.dtx);
    rs.set_signal(Signal::Voice);
    let mut err = 0;
    let st = unsafe { opus_encoder_create(c.fs, c.ch as i32, 2048, &mut err) };
    assert_eq!(err, 0);
    unsafe {
        opus_encoder_ctl(st, SET_BITRATE, c.rates[0]);
        opus_encoder_ctl(st, SET_COMPLEXITY, c.cx);
        opus_encoder_ctl(st, SET_VBR, !c.cbr as i32);
        opus_encoder_ctl(st, SET_INBAND_FEC, c.fec as i32);
        opus_encoder_ctl(st, SET_PACKET_LOSS_PERC, if c.fec { 20 } else { 0 });
        opus_encoder_ctl(st, SET_DTX, c.dtx as i32);
        opus_encoder_ctl(st, SET_SIGNAL, 3001);
    }
    let (d_rs, d_c) = unsafe {
        (
            opus_decoder_create(c.fs, c.ch as i32, &mut err),
            opus_decoder_create(c.fs, c.ch as i32, &mut err),
        )
    };
    let mut out = vec![0i16; n * c.ch];
    // The SILK symbols libopus's decoder reads from a packet, and the
    // redundancy flag of opus_decoder.c, read when a CELT frame follows.
    let mut silk_symbols = |d: *mut OpusDecoder, p: &[u8]| {
        let (_, syms) = trace(|| unsafe {
            opus_decode(d, p.as_ptr(), p.len() as i32, out.as_mut_ptr(), n as i32, 0)
        });
        syms.into_iter()
            .filter(|s| SILK_DECODER.contains(&s.file) || s.file == "opus_decoder.c")
            .collect::<Vec<_>>()
    };
    let redundant = |s: &[Symbol]| s.iter().any(|s| s.file == "opus_decoder.c");
    let mut res = None;
    for (f, frame) in pcm.chunks_exact(n * c.ch).enumerate() {
        let bw = BW[c.bws[f * c.ms / 200 % c.bws.len()]];
        let rate = c.rates[f * c.ms / 200 % 2];
        rs.set_bandwidth(bw);
        rs.bitrate_bps = rate;
        unsafe {
            opus_encoder_ctl(st, SET_BANDWIDTH, bw as i32);
            opus_encoder_ctl(st, SET_BITRATE, rate);
        }
        let (mut a, mut b) = ([0u8; 1500], [0u8; 1500]);
        let la = rs.encode_i16(frame, n, &mut a).unwrap();
        let lb = unsafe { opus_encode(st, frame.as_ptr(), n as i32, b.as_mut_ptr(), 1500) };
        assert!(lb > 0);
        let mut range = 0u32;
        unsafe { opus_encoder_ctl(st, GET_FINAL_RANGE, &mut range as *mut u32) };
        let input_rs = opus_rs::fuzzing::take_silk_input();
        let input_c = take_silk_input();
        let (a, b) = (&a[..la], &b[..lb as usize]);
        // Both decoders decode every packet, which keeps their states.
        let (s_rs, s_c) = (silk_symbols(d_rs, a), silk_symbols(d_c, b));
        // A DTX packet of one encoder only; analysis-driven DTX decisions
        // follow the float build in opus-rs.
        let kind = if (a.len() == 1) != (b.len() == 1) {
            "dtx"
        } else if a[0] != b[0] {
            "toc"
        } else if input_rs != input_c {
            "input"
        } else if a[0] >> 3 >= 12 {
            // A CELT-only packet of both encoders (TOC configs 16-31).
            continue;
        } else if redundant(&s_rs) != redundant(&s_c) {
            // One encoder starts a mode transition.
            "redundancy"
        } else if s_rs != s_c || (a == b && rs.final_range() != range) {
            "packet"
        } else {
            continue;
        };
        res = Some((f, kind));
        break;
    }
    unsafe {
        opus_encoder_destroy(st);
        opus_decoder_destroy(d_rs);
        opus_decoder_destroy(d_c);
    }
    res
}

#[test]
fn silk_packets_match_fixed_point_libopus() {
    // Fixed SILK rates at their own and at 48 kHz input, then switches.
    let mut grid: Vec<(i32, &'static [usize])> = Vec::new();
    for (bw, bws) in [&[0usize][..], &[1], &[2]].into_iter().enumerate() {
        grid.extend([([8000, 12000, 16000][bw], bws), (48000, bws)]);
    }
    let switches: &'static [usize] = &[2, 0, 1, 2, 0];
    grid.extend([(16000, switches), (48000, switches)]);
    let (mut total, mut fails) = (0, Vec::new());
    for (fs, bws) in grid {
        for ch in 1..=2 {
            let pcm = signal(fs as usize, ch);
            for ms in [10, 20, 40, 60] {
                for cx in 0..=10 {
                    for flags in 0..8 {
                        let (cbr, fec, dtx) = (flags & 1 != 0, flags & 2 != 0, flags & 4 != 0);
                        // libopus codes 10 ms CBR below 9 kb/s as CELT
                        // (opus_encoder.c:1526).
                        let rate = [8000, 12000, 20000, 32000][(cx as usize + flags) % 4]
                            .max(if ms == 10 { 10000 } else { 0 })
                            * ch as i32;
                        let c = Cfg {
                            fs,
                            bws,
                            ch,
                            ms,
                            cx,
                            cbr,
                            fec,
                            dtx,
                            rates: [rate; 2],
                        };
                        total += 1;
                        if let Some(d) = run(c, &pcm) {
                            fails.push((c, d));
                        }
                    }
                }
            }
        }
    }
    for (c, (f, kind)) in &fails {
        eprintln!("{kind} frame {f}: {c:?}");
    }
    let count = |k: &str| fails.iter().filter(|(_, (_, kind))| *kind == k).count();
    eprintln!(
        "{} of {total} configurations match; toc {}, input {}, dtx {}, redundancy {}, packet {}",
        total - fails.len(),
        count("toc"),
        count("input"),
        count("dtx"),
        count("redundancy"),
        count("packet")
    );
    assert_eq!(count("packet"), 0);
}

#[test]
fn stereo_to_mono_switches_match_fixed_point_libopus() {
    // Stereo above the stereo threshold, then mono below it, every 200 ms:
    // SILK delays each switch to mono by two frames with toMono set.
    let mut fails = Vec::new();
    for fs in [16000, 48000] {
        let pcm = signal(fs as usize, 2);
        for ms in [10, 20, 40, 60] {
            for cx in [0, 2, 5, 8, 10] {
                for cbr in [false, true] {
                    let c = Cfg {
                        fs,
                        bws: &[2],
                        ch: 2,
                        ms,
                        cx,
                        cbr,
                        fec: false,
                        dtx: false,
                        rates: [32000, 10000],
                    };
                    if let Some(d) = run(c, &pcm) {
                        fails.push((c, d));
                    }
                }
            }
        }
    }
    for (c, (f, kind)) in &fails {
        eprintln!("{kind} frame {f}: {c:?}");
    }
    assert!(fails.is_empty(), "{} configurations differ", fails.len());
}
