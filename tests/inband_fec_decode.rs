//! In-band FEC decoding (`OpusDecoder::decode_fec`) checked against libopus's
//! `opus_decode(..., decode_fec = 1)`.
//!
//! A SILK or Hybrid packet coded with in-band FEC carries, after its header,
//! an LBRR copy of the audio before it. With packet N lost and packet N+1 in
//! hand, libopus (`opus_decode_native`) conceals all but the last frame of
//! the requested duration, then decodes the LBRR of N+1's first Opus frame
//! (`silk_Decode` with `FLAG_DECODE_LBRR`): per SILK frame, the LBRR when its
//! flag is set and concealment when not; mid and side for stereo; Hybrid's
//! CELT layer concealed. CELT-only packets, packets that follow CELT, and
//! packets longer than the request carry no usable FEC and are concealed.
//!
//! Each test drops packets from a stream encoded with FEC on, runs opus-rs
//! and libopus through the same loss — `decode_fec` on the packet after it,
//! then `decode` on that packet — and compares the two outputs, one loss per
//! pass so each starts from decoders in step.

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Decoder as CDec,
    Encoder as CEnc, Signal as CSignal, packet,
};
use opus_rs::range_coder::RangeCoder;
use opus_rs::silk::{SILK_LBRR_FLAGS_2_ICDF, SILK_LBRR_FLAGS_3_ICDF};
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;
use std::f64::consts::PI as PI64;

/// Largest Opus packet per channel: 120 ms at 48 kHz.
const MAX_FRAME: usize = 5760;
/// The decoders' rate.
const RATE: usize = 48000;
/// 20 ms at 48 kHz.
const FRAME_20MS: usize = 960;

fn c_channels(ch: usize) -> CCh {
    if ch == 2 { CCh::Stereo } else { CCh::Mono }
}

fn noise(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
    (*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

/// Two seconds of a voiced harmonic series (150 Hz, drifting) with a little
/// broadband noise, under a syllable-rate envelope, at `sr`. The right
/// channel is a phase-shifted, quieter copy, so the side is not zero.
fn voiced(sr: usize, ch: usize) -> Vec<f32> {
    let n = 2 * sr;
    let mut seed = 11u32;
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(n * ch);
    for i in 0..n {
        let t = i as f32 / sr as f32;
        phase += 2.0 * PI * (150.0 + 20.0 * (2.0 * PI * 0.7 * t).sin()) / sr as f32;
        let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin();
        let (mut l, mut r) = (0.0, 0.0);
        for h in 1..=10 {
            let a = 0.2 / h as f32;
            l += a * (h as f32 * phase).sin();
            r += a * (h as f32 * phase + 0.3 * h as f32).sin();
        }
        let hiss = 0.03 * noise(&mut seed);
        out.push((l + hiss) * env * 0.5);
        if ch == 2 {
            out.push((r + hiss) * env * 0.4);
        }
    }
    out
}

/// The packet's mode from its TOC: 'S' SILK-only, 'H' Hybrid, 'C' CELT-only.
fn mode(pkt: &[u8]) -> char {
    match pkt[0] >> 3 {
        0..=11 => 'S',
        12..=15 => 'H',
        _ => 'C',
    }
}

/// libopus packets of `ms` ms at 48 kHz, `bitrate` b/s and `bw`, with
/// in-band FEC on (30% expected loss) when `fec`.
fn libopus_packets(
    ch: usize,
    ms: usize,
    bitrate: i32,
    bw: CBw,
    signal: CSignal,
    fec: bool,
) -> Vec<Vec<u8>> {
    let frame = RATE * ms / 1000;
    let app = match signal {
        CSignal::Music => CApp::Audio,
        _ => CApp::Voip,
    };
    let mut enc = CEnc::new(RATE as u32, c_channels(ch), app).unwrap();
    enc.set_bitrate(CBitrate::Bits(bitrate)).unwrap();
    enc.set_bandwidth(bw).unwrap();
    enc.set_signal(signal).unwrap();
    enc.set_inband_fec(fec).unwrap();
    enc.set_packet_loss_perc(if fec { 30 } else { 0 }).unwrap();
    let mut pkt = vec![0u8; 1500];
    voiced(RATE, ch)
        .chunks_exact(frame * ch)
        .map(|f| {
            let n = enc.encode_float(f, &mut pkt).unwrap();
            pkt[..n].to_vec()
        })
        .collect()
}

/// opus-rs packets of 20 ms at 16 kHz (SILK wideband), 32 kb/s, with
/// in-band FEC on (30% expected loss).
fn opus_rs_packets(ch: usize) -> Vec<Vec<u8>> {
    let sr = 16000;
    let frame = sr / 50;
    let mut enc = OpusEncoder::new(sr as i32, ch, Application::Voip).unwrap();
    enc.bitrate_bps = 32000;
    enc.use_inband_fec = true;
    enc.packet_loss_perc = 30;
    let mut pkt = vec![0u8; 1500];
    voiced(sr, ch)
        .chunks_exact(frame * ch)
        .map(|f| {
            let n = enc.encode(f, frame, &mut pkt).unwrap();
            pkt[..n].to_vec()
        })
        .collect()
}

/// Whether FEC rebuilds all of the audio before `pkt` from LBRR: its first
/// frame's SILK header (dec_API.c) sets the LBRR flag of every SILK frame,
/// for every channel. Otherwise some of it is concealed.
fn fully_covered(pkt: &[u8], ch: usize) -> bool {
    let frame = packet::parse(pkt).unwrap().frames[0];
    let silk_frames = match (mode(pkt), (pkt[0] >> 3) & 3) {
        ('S', 2) => 2,
        ('S', 3) => 3,
        _ => 1,
    };
    let mut rc = RangeCoder::new_decoder(frame);
    let mut lbrr = [false; 2];
    for flag in &mut lbrr[..ch] {
        for _ in 0..silk_frames {
            rc.decode_bit_logp(1); // VAD
        }
        *flag = rc.decode_bit_logp(1);
    }
    lbrr[..ch].iter().all(|&on| {
        on && (silk_frames == 1 || {
            let icdf: &[u8] = if silk_frames == 2 {
                &SILK_LBRR_FLAGS_2_ICDF
            } else {
                &SILK_LBRR_FLAGS_3_ICDF
            };
            rc.decode_icdf(icdf, 8) + 1 == (1 << silk_frames) - 1
        })
    })
}

/// A stretch of output: first sample and length, per channel.
type Span = (usize, usize);

/// The outputs of one pass through a stream with one burst of losses.
struct Pass {
    /// libopus's output, interleaved.
    c: Vec<f32>,
    /// opus-rs's output with `decode_fec`.
    rs: Vec<f32>,
    /// opus-rs's output with the loss concealed instead.
    rs_plc: Vec<f32>,
    /// What FEC conceals before the LBRR, when more was lost than it covers.
    head: Span,
    /// What FEC rebuilds from the LBRR.
    tail: Span,
    /// The packet decoded right after the recovery.
    next: Span,
}

/// Decode `packets` with libopus and opus-rs, losing `burst` packets from
/// `first` on. The packet after the loss is first decoded with FEC for the
/// whole lost duration, then decoded normally.
fn pass(packets: &[Vec<u8>], ch: usize, first: usize, burst: usize) -> Pass {
    let mut c_dec = CDec::new(RATE as u32, c_channels(ch)).unwrap();
    let mut rs_dec = OpusDecoder::new(RATE as i32, ch).unwrap();
    let mut plc_dec = OpusDecoder::new(RATE as i32, ch).unwrap();
    let (mut c, mut rs, mut rs_plc) = (Vec::new(), Vec::new(), Vec::new());
    let (mut head, mut tail, mut next) = ((0, 0), (0, 0), (0, 0));
    let mut buf = vec![0f32; MAX_FRAME * ch];
    let mut gap = 0usize;
    for (p, pkt) in packets.iter().enumerate() {
        let n = packet::get_nb_samples(pkt, RATE as u32).unwrap();
        if (first..first + burst).contains(&p) {
            gap += n;
            continue;
        }
        if gap > 0 {
            let len = gap * ch;
            let got = c_dec.decode_float(pkt, &mut buf[..len], true).unwrap();
            assert_eq!(got, gap, "packet {p}: libopus FEC length");
            let covered = packet::get_samples_per_frame(pkt, RATE as u32).unwrap();
            let from = c.len() / ch;
            head = (from, gap - covered);
            tail = (from + gap - covered, covered);
            next = (from + gap, n);
            c.extend_from_slice(&buf[..len]);

            let got = rs_dec
                .decode_fec(pkt, gap, &mut buf[..len])
                .unwrap_or_else(|e| panic!("packet {p}: decode_fec failed: {e}"));
            assert_eq!(got, gap, "packet {p}: opus-rs FEC length");
            rs.extend_from_slice(&buf[..len]);

            let got = plc_dec.decode(&[], gap, &mut buf[..len]).unwrap();
            assert_eq!(got, gap, "packet {p}: opus-rs PLC length");
            rs_plc.extend_from_slice(&buf[..len]);
            gap = 0;
        }
        let got = c_dec.decode_float(pkt, &mut buf, false).unwrap();
        assert_eq!(got, n, "packet {p}: libopus length");
        c.extend_from_slice(&buf[..n * ch]);
        for (dec, pcm) in [(&mut rs_dec, &mut rs), (&mut plc_dec, &mut rs_plc)] {
            let got = dec
                .decode(pkt, MAX_FRAME, &mut buf)
                .unwrap_or_else(|e| panic!("packet {p}: decode failed: {e}"));
            assert_eq!(got, n, "packet {p}: opus-rs length");
            pcm.extend_from_slice(&buf[..n * ch]);
        }
    }
    Pass {
        c,
        rs,
        rs_plc,
        head,
        tail,
        next,
    }
}

/// The interleaved samples of `span`.
fn samples(pcm: &[f32], ch: usize, (start, len): Span) -> &[f32] {
    &pcm[start * ch..(start + len) * ch]
}

/// `pcm` (interleaved, `ch` channels) over `span`, low-passed at 6 kHz with
/// a 255-tap Blackman-windowed sinc: the SILK band of a Hybrid frame,
/// without the CELT layer above 8 kHz.
fn silk_band(pcm: &[f32], ch: usize, (start, len): Span) -> Vec<f32> {
    const TAPS: usize = 255;
    let fc = 6000.0 / RATE as f64;
    let h: Vec<f64> = (0..TAPS)
        .map(|i| {
            let m = i as f64 - (TAPS / 2) as f64;
            let sinc = if m == 0.0 {
                2.0 * fc
            } else {
                (2.0 * PI64 * fc * m).sin() / (PI64 * m)
            };
            let w = 2.0 * PI64 * i as f64 / (TAPS - 1) as f64;
            sinc * (0.42 - 0.5 * w.cos() + 0.08 * (2.0 * w).cos())
        })
        .collect();
    let n = pcm.len() / ch;
    let mut out = Vec::with_capacity(len * ch);
    for i in start..start + len {
        for c in 0..ch {
            let mut acc = 0f64;
            for (k, hk) in h.iter().enumerate() {
                let j = (i + k).checked_sub(TAPS / 2).filter(|&j| j < n);
                if let Some(j) = j {
                    acc += hk * pcm[j * ch + c] as f64;
                }
            }
            out.push(acc as f32);
        }
    }
    out
}

/// The error of `test` against `reference`, in dB below `level` (a mean
/// square).
fn db_below(level: f64, reference: &[f32], test: &[f32]) -> f64 {
    let err = reference
        .iter()
        .zip(test)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        / reference.len() as f64;
    10.0 * (level / err.max(1e-30)).log10()
}

/// `test`'s error against `reference` over `span`, in dB below the whole
/// `reference`'s level; in the SILK band only when `low`.
fn db(reference: &[f32], test: &[f32], ch: usize, span: Span, low: bool) -> f64 {
    let level = reference.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / reference.len() as f64;
    if low {
        let (r, t) = (silk_band(reference, ch, span), silk_band(test, ch, span));
        db_below(level, &r, &t)
    } else {
        db_below(level, samples(reference, ch, span), samples(test, ch, span))
    }
}

/// The losses tried in a stream of `n` packets, each in a pass of its own.
fn loss_positions(n: usize, burst: usize) -> impl Iterator<Item = usize> {
    (2..n.saturating_sub(burst + 1)).step_by(3)
}

fn worst(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Lose single packets from `packets` and recover them with FEC. Where the
/// next packet carries LBRR for all of the lost audio, the recovery and the
/// packet after it must match libopus's. Hybrid's CELT layer is concealed
/// (not decoded) by FEC, so Hybrid compares the SILK band.
fn check(name: &str, packets: &[Vec<u8>], ch: usize, expect: char) {
    let modes: String = packets.iter().map(|p| mode(p)).collect();
    assert!(
        modes[2..].chars().all(|m| m == expect),
        "{name}: expected only '{expect}' packets: {modes}"
    );
    let low = expect == 'H';
    let (mut fec, mut next, mut plc, mut full) = (vec![], vec![], vec![], vec![]);
    let (mut partial, mut concealed) = (0, 0);
    for first in loss_positions(packets.len(), 1) {
        let r = pass(packets, ch, first, 1);
        if !fully_covered(&packets[first + 1], ch) {
            if samples(&r.rs, ch, r.tail) == samples(&r.rs_plc, ch, r.tail) {
                concealed += 1;
            } else {
                partial += 1;
            }
            continue;
        }
        fec.push(db(&r.c, &r.rs, ch, r.tail, low));
        next.push(db(&r.c, &r.rs, ch, r.next, low));
        plc.push(db(&r.c, &r.rs_plc, ch, r.tail, low));
        full.push(db(&r.c, &r.rs, ch, r.tail, false));
    }
    let band = if low { " (0-6 kHz)" } else { "" };
    println!(
        "{name}: {} losses from LBRR, {partial} partly, {concealed} concealed. \
         vs libopus{band}: FEC worst {:.1} dB, next packet worst {:.1} dB; \
         concealing instead {:.1} dB on average; FEC full band worst {:.1} dB",
        fec.len(),
        worst(&fec),
        worst(&next),
        mean(&plc),
        worst(&full),
    );
    assert!(
        fec.len() >= 3,
        "{name}: only {} losses fully covered",
        fec.len()
    );
    let floor = if low { 90.0 } else { 100.0 };
    assert!(
        worst(&fec) > floor,
        "{name}: FEC is {:.1} dB from libopus",
        worst(&fec)
    );
    assert!(
        worst(&next) > floor,
        "{name}: the packet after FEC is {:.1} dB from libopus",
        worst(&next)
    );
}

#[test]
fn silk_wb_mono_20ms_fec_matches_libopus() {
    let p = libopus_packets(1, 20, 24000, CBw::Wideband, CSignal::Voice, true);
    check("SILK WB mono 20 ms", &p, 1, 'S');
}

#[test]
fn silk_wb_stereo_20ms_fec_matches_libopus() {
    let p = libopus_packets(2, 20, 40000, CBw::Wideband, CSignal::Voice, true);
    check("SILK WB stereo 20 ms", &p, 2, 'S');
}

#[test]
fn silk_nb_mono_10ms_fec_matches_libopus() {
    let p = libopus_packets(1, 10, 20000, CBw::Narrowband, CSignal::Voice, true);
    check("SILK NB mono 10 ms", &p, 1, 'S');
}

#[test]
fn silk_mb_stereo_10ms_fec_matches_libopus() {
    let p = libopus_packets(2, 10, 32000, CBw::Mediumband, CSignal::Voice, true);
    check("SILK MB stereo 10 ms", &p, 2, 'S');
}

/// A 40/60 ms SILK packet holds 2/3 SILK frames, each with its own LBRR flag
/// and LBRR frame, coded conditionally on the LBRR frame before it.
#[test]
fn silk_40ms_mono_fec_matches_libopus() {
    let p = libopus_packets(1, 40, 24000, CBw::Wideband, CSignal::Voice, true);
    check("SILK WB mono 40 ms", &p, 1, 'S');
}

#[test]
fn silk_60ms_stereo_fec_matches_libopus() {
    let p = libopus_packets(2, 60, 40000, CBw::Wideband, CSignal::Voice, true);
    check("SILK WB stereo 60 ms", &p, 2, 'S');
}

/// Hybrid's FEC is its SILK layer's LBRR; the CELT layer is concealed.
#[test]
fn hybrid_fb_mono_20ms_fec_matches_libopus() {
    let p = libopus_packets(1, 20, 40000, CBw::Fullband, CSignal::Voice, true);
    check("Hybrid FB mono 20 ms", &p, 1, 'H');
}

#[test]
fn hybrid_swb_stereo_10ms_fec_matches_libopus() {
    let p = libopus_packets(2, 10, 56000, CBw::Superwideband, CSignal::Voice, true);
    check("Hybrid SWB stereo 10 ms", &p, 2, 'H');
}

/// The crate's own encoder: its LBRR must decode in opus-rs as in libopus.
#[test]
fn opus_rs_encoded_fec_matches_libopus() {
    for ch in [1, 2] {
        let p = opus_rs_packets(ch);
        check(&format!("opus-rs SILK WB {ch} ch 20 ms"), &p, ch, 'S');
    }
}

/// Lose `burst` packets at a time from `packets`, more than the FEC of the
/// next packet's first frame covers. FEC conceals the head — sample for
/// sample as `decode(&[], ..)` would — and decodes the LBRR for the tail on
/// from the concealed state.
fn check_concealed_head(name: &str, packets: &[Vec<u8>], ch: usize, burst: usize, low: bool) {
    let (mut fec, mut plc) = (Vec::new(), Vec::new());
    for first in loss_positions(packets.len(), burst) {
        let r = pass(packets, ch, first, burst);
        assert!(r.head.1 > 0, "{name}: nothing to conceal");
        assert_eq!(
            samples(&r.rs, ch, r.head),
            samples(&r.rs_plc, ch, r.head),
            "{name}: lost {first}: the head is not the concealment"
        );
        if fully_covered(&packets[first + burst], ch) {
            fec.push(db(&r.c, &r.rs, ch, r.tail, low));
            plc.push(db(&r.c, &r.rs_plc, ch, r.tail, low));
        }
    }
    // opus-rs's concealment is not bit-exact with libopus's, so the LBRR
    // decoded on from it drifts too; it must still land far closer to
    // libopus's FEC than concealing all of the loss does.
    println!(
        "{name}: {} losses from LBRR; the LBRR tail vs libopus {:.1} dB on average, \
         concealing instead {:.1} dB",
        fec.len(),
        mean(&fec),
        mean(&plc)
    );
    assert!(
        fec.len() >= 3,
        "{name}: only {} losses fully covered",
        fec.len()
    );
    assert!(
        mean(&fec) > mean(&plc) + 10.0,
        "{name}: FEC {:.1} dB vs concealment {:.1} dB",
        mean(&fec),
        mean(&plc)
    );
}

/// Two 20 ms packets lost: FEC covers only the second.
#[test]
fn silk_fec_after_two_losses_conceals_the_first() {
    let p = libopus_packets(1, 20, 24000, CBw::Wideband, CSignal::Voice, true);
    check_concealed_head("SILK WB mono 20 ms, 2 lost", &p, 1, 2, false);
}

/// A 40 ms Hybrid packet is two 20 ms Opus frames: FEC takes the first
/// frame's LBRR, so a 40 ms loss gets 20 ms concealed, then 20 ms of LBRR.
#[test]
fn hybrid_two_frame_packet_fec_conceals_the_first_frame() {
    let p = libopus_packets(1, 40, 40000, CBw::Fullband, CSignal::Voice, true);
    assert!(
        p[2..].iter().all(|p| mode(p) == 'H' && p[0] & 3 != 0),
        "two-frame Hybrid packets"
    );
    check_concealed_head("Hybrid FB mono 2x20 ms", &p, 1, 1, true);
}

/// With no LBRR in the packet, or a CELT-only packet, FEC conceals: the
/// output is the concealment's, sample for sample.
fn check_conceals(name: &str, packets: &[Vec<u8>], ch: usize) {
    for first in loss_positions(packets.len(), 1) {
        let r = pass(packets, ch, first, 1);
        assert_eq!(
            r.rs, r.rs_plc,
            "{name}: FEC after packet {first} did not conceal"
        );
    }
}

#[test]
fn silk_without_lbrr_conceals() {
    for ch in [1, 2] {
        let p = libopus_packets(ch, 20, 24000, CBw::Wideband, CSignal::Voice, false);
        check_conceals(&format!("SILK WB {ch} ch, no FEC"), &p, ch);
    }
}

#[test]
fn celt_only_conceals() {
    let p = libopus_packets(1, 20, 64000, CBw::Fullband, CSignal::Music, false);
    assert!(p[2..].iter().all(|p| mode(p) == 'C'), "CELT-only packets");
    check_conceals("CELT FB mono", &p, 1);
}

/// A packet longer than the requested duration can't be used: libopus
/// conceals instead, and so must opus-rs.
#[test]
fn a_packet_longer_than_the_request_conceals() {
    let p = libopus_packets(1, 40, 24000, CBw::Wideband, CSignal::Voice, true);
    let mut fec = OpusDecoder::new(RATE as i32, 1).unwrap();
    let mut plc = OpusDecoder::new(RATE as i32, 1).unwrap();
    let mut buf = vec![0f32; MAX_FRAME];
    for pkt in &p[..4] {
        fec.decode(pkt, MAX_FRAME, &mut buf).unwrap();
        plc.decode(pkt, MAX_FRAME, &mut buf).unwrap();
    }
    let mut a = vec![0f32; FRAME_20MS];
    let mut b = vec![0f32; FRAME_20MS];
    assert_eq!(fec.decode_fec(&p[5], FRAME_20MS, &mut a), Ok(FRAME_20MS));
    assert_eq!(plc.decode(&[], FRAME_20MS, &mut b), Ok(FRAME_20MS));
    assert_eq!(a, b);
}
