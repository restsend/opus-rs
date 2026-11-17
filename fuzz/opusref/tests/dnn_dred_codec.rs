//! DRED in the codec: opus-rs's packet extension parsing, DRED parser,
//! DRED decoding and DRED encoding against the DNN-enabled libopus.
#![cfg(feature = "dnn")]
use opusref::dnn::{Rng, same_bits, speechlike};
use opusref::dnn_dred_api::*;
use opusref::*;

/// opus-rs's extensions of a padding, as (id, frame, payload).
fn rs_extensions(data: &[u8], nb_frames: usize) -> Result<Vec<(u8, usize, Vec<u8>)>, &'static str> {
    let mut it = opus_rs::extensions::ExtensionIterator::new(data, nb_frames);
    let mut out = Vec::new();
    while let Some(e) = it.next()? {
        out.push((e.id, e.frame, e.data.to_vec()));
    }
    Ok(out)
}

/// Random extension lists written by libopus's
/// opus_packet_extensions_generate (which uses separators and the repeat
/// mechanism), plus random bytes: opus-rs's iterator returns what
/// opus_packet_extensions_parse returns, or fails where it fails.
#[test]
fn extension_iterator_matches_libopus() {
    let mut r = Rng(0x5eed);
    let mut generated = 0;
    let mut valid_random = 0;
    for case in 0..20000 {
        let nb_frames = 1 + r.below(if case % 2 == 0 { 4 } else { 48 }) as usize;
        let data: Vec<u8> = if case % 3 == 0 {
            // Random bytes, biased to small IDs and lengths.
            let n = r.below(40) as usize;
            (0..n).map(|_| if r.below(2) == 0 { r.below(16) as u8 } else { r.next_u32() as u8 }).collect()
        } else {
            let n = r.below(8) as usize;
            let payloads: Vec<Vec<u8>> = (0..n)
                .map(|_| {
                    let len = match r.below(4) {
                        0 => 0,
                        1 => 1,
                        2 => r.below(20) as usize,
                        _ => r.below(600) as usize,
                    };
                    (0..len).map(|_| r.next_u32() as u8).collect()
                })
                .collect();
            let mut exts: Vec<OpusExtensionData> = payloads
                .iter()
                .map(|p| {
                    let id = if r.below(2) == 0 { 3 + r.below(29) } else { 32 + r.below(96) } as i32;
                    let len = if id < 32 { p.len().min(1) } else { p.len() } as i32;
                    OpusExtensionData { id, frame: r.below(nb_frames as u32) as i32, data: p.as_ptr(), len }
                })
                .collect();
            // Mostly in frame order, which lets repeats happen.
            if r.below(4) != 0 {
                exts.sort_by_key(|e| e.frame);
            }
            let mut buf = vec![0u8; 8000];
            let pad = r.below(2) as i32;
            let cap = if pad != 0 { 1 + r.below(7000) as i32 } else { 8000 };
            let len = unsafe {
                opus_packet_extensions_generate(buf.as_mut_ptr(), cap, exts.as_ptr(), exts.len() as i32, nb_frames as i32, pad)
            };
            if len < 0 {
                continue;
            }
            generated += 1;
            buf.truncate(len as usize);
            // Sometimes truncated or with a byte changed.
            match r.below(6) {
                0 if !buf.is_empty() => {
                    let k = r.below(buf.len() as u32) as usize;
                    buf.truncate(k);
                }
                1 if !buf.is_empty() => {
                    let k = r.below(buf.len() as u32) as usize;
                    buf[k] = r.next_u32() as u8;
                }
                _ => {}
            }
            buf
        };
        let c = extensions_parse(&data, nb_frames);
        let rs = rs_extensions(&data, nb_frames);
        match (&c, &rs) {
            (Ok(c), Ok(rs)) => {
                assert_eq!(c, rs, "case {case}: {data:?} with {nb_frames} frames");
                if case % 3 == 0 {
                    valid_random += 1;
                }
            }
            (Err(_), Err(_)) => {}
            _ => panic!("case {case}: libopus {c:?}, opus-rs {rs:?} for {data:?} with {nb_frames} frames"),
        }
    }
    assert!(generated > 5000 && valid_random > 1000, "{generated} generated, {valid_random} valid random");
}

/// opus-rs's packet parsing returns the frame count and padding of
/// libopus's opus_packet_parse_impl, and fails where it fails.
#[test]
fn packet_padding_matches_libopus() {
    let mut r = Rng(77);
    for case in 0..50000 {
        let n = 1 + r.below(if case % 4 == 0 { 2000 } else { 40 }) as usize;
        let mut p: Vec<u8> = (0..n).map(|_| r.next_u32() as u8).collect();
        if case % 2 == 0 {
            // Code 3 with a small frame count and padding.
            p[0] |= 3;
            if n > 1 {
                p[1] = (p[1] & 0xc0) | (1 + r.below(6)) as u8;
            }
            if n > 2 && r.below(2) == 0 {
                p[2] = r.below(n as u32) as u8;
            }
        }
        let mut toc = 0u8;
        let mut frames = [core::ptr::null::<u8>(); 48];
        let mut size = [0i16; 48];
        let mut padding = core::ptr::null::<u8>();
        let mut padding_len = 0i32;
        let ret = unsafe {
            opus_packet_parse_impl(
                p.as_ptr(),
                p.len() as i32,
                0,
                &mut toc,
                frames.as_mut_ptr(),
                size.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                &mut padding,
                &mut padding_len,
            )
        };
        match opus_rs::extensions::parse_packet(&p) {
            Ok(parts) => {
                assert!(ret > 0, "case {case}: libopus {ret} for {p:?}");
                assert_eq!(parts.count, ret as usize, "case {case}");
                for i in 0..parts.count {
                    let off = unsafe { frames[i].offset_from(p.as_ptr()) } as usize;
                    assert_eq!((parts.frames[i].as_ptr() as usize - p.as_ptr() as usize, parts.frames[i].len()), (off, size[i] as usize), "case {case} frame {i}");
                }
                let off = unsafe { padding.offset_from(p.as_ptr()) } as usize;
                assert_eq!((parts.padding.as_ptr() as usize - p.as_ptr() as usize, parts.padding.len()), (off, padding_len as usize), "case {case}");
            }
            Err(_) => assert!(ret < 0, "case {case}: libopus {ret}, opus-rs failed for {p:?}"),
        }
    }
}

/// Bursts of 1 to `max_burst` lost packets, a burst starting with
/// probability `rate` per packet; the first second is never lost.
pub fn with_losses(packets: Vec<Vec<u8>>, seed: u64, rate: u32, max_burst: u32, frame_ms: usize) -> Vec<Option<Vec<u8>>> {
    let mut r = Rng(seed);
    let mut left = 0;
    packets
        .into_iter()
        .enumerate()
        .map(|(k, p)| {
            if k * frame_ms >= 1000 && left == 0 && r.below(100) < rate {
                left = 1 + r.below(max_burst);
            }
            if left > 0 {
                left -= 1;
                None
            } else {
                Some(p)
            }
        })
        .collect()
}

/// One reference encoder setup with DRED.
#[derive(Clone, Copy, Debug)]
pub struct Setup {
    pub fs: i32,
    pub channels: usize,
    pub app: i32,
    pub bitrate: i32,
    pub max_bandwidth: i32,
    pub frame_ms: usize,
    /// OPUS_SET_PACKET_LOSS_PERC, which sets the DRED bitrate.
    pub loss: i32,
    /// OPUS_SET_DRED_DURATION in 10 ms frames.
    pub dred: i32,
    pub vbr: bool,
}

pub const SILK_WB_16K: Setup = Setup {
    fs: 16000,
    channels: 1,
    app: 2048,
    bitrate: 32000,
    max_bandwidth: 1103,
    frame_ms: 20,
    loss: 20,
    dred: 100,
    vbr: true,
};
pub const SILK_WB_48K: Setup = Setup { fs: 48000, ..SILK_WB_16K };
/// At 32 kb/s, 10 ms frames leave too few bits for two DRED chunks.
pub const SILK_WB_16K_CBR_10MS: Setup = Setup { frame_ms: 10, vbr: false, dred: 50, bitrate: 48000, ..SILK_WB_16K };
pub const HYBRID_48K: Setup = Setup { fs: 48000, max_bandwidth: 1105, bitrate: 40000, ..SILK_WB_16K };
pub const CELT_48K: Setup = Setup { fs: 48000, app: 2051, bitrate: 64000, max_bandwidth: 1105, ..SILK_WB_16K };
pub const CELT_48K_STEREO: Setup = Setup { channels: 2, bitrate: 96000, ..CELT_48K };

/// `seconds` of a speech-like signal at `s.fs` (interleaved; the second
/// channel is the first scaled).
pub fn signal(s: Setup, seconds: usize, seed: u64) -> Vec<f32> {
    let sig16 = speechlike(16000 * seconds, seed);
    let n = s.fs as usize * seconds;
    let mut pcm = vec![0f32; n * s.channels];
    for i in 0..n {
        let v = sig16[(i * 16000 / s.fs as usize).min(sig16.len() - 1)] / 32768.0;
        for c in 0..s.channels {
            pcm[i * s.channels + c] = if c == 0 { v } else { 0.8 * v };
        }
    }
    pcm
}

/// Packets of the reference encoder with DRED for `pcm`.
pub fn ref_encode(s: Setup, pcm: &[f32]) -> Vec<Vec<u8>> {
    let frame = s.fs as usize * s.frame_ms / 1000;
    let mut out = Vec::new();
    unsafe {
        let mut err = 0;
        let enc = opus_encoder_create(s.fs, s.channels as i32, s.app, &mut err);
        assert_eq!(err, 0);
        opus_encoder_ctl(enc, SET_BITRATE, s.bitrate);
        opus_encoder_ctl(enc, SET_MAX_BANDWIDTH, s.max_bandwidth);
        opus_encoder_ctl(enc, SET_COMPLEXITY, 9);
        opus_encoder_ctl(enc, SET_VBR, s.vbr as i32);
        opus_encoder_ctl(enc, SET_PACKET_LOSS_PERC, s.loss);
        assert_eq!(set_dred_duration(enc, s.dred), 0);
        for f in pcm.chunks_exact(frame * s.channels) {
            let mut pkt = [0u8; 1500];
            let len = opus_encode_float(enc, f.as_ptr(), frame as i32, pkt.as_mut_ptr(), 1500);
            assert!(len > 0);
            out.push(pkt[..len as usize].to_vec());
        }
        opus_encoder_destroy(enc);
    }
    out
}

/// Compares libopus's OpusDRED with opus-rs's, bit for bit.
fn same_dred(c: &OpusDRED, r: &opus_rs::dred::OpusDred) -> Result<(), String> {
    let ints = (c.process_stage, c.nb_latents, c.dred_offset);
    if ints != (r.process_stage, r.nb_latents, r.dred_offset) {
        return Err(format!("(stage, latents, offset) {ints:?} vs {:?}", (r.process_stage, r.nb_latents, r.dred_offset)));
    }
    for (name, a, b) in [
        ("fec_features", &c.fec_features[..], &r.fec_features[..]),
        ("state", &c.state[..], &r.state[..]),
        ("latents", &c.latents[..], &r.latents[..]),
    ] {
        if let Some(i) = same_bits(a, b) {
            return Err(format!("{name}[{i}]: {} vs {}", a[i], b[i]));
        }
    }
    Ok(())
}

/// Parses every packet with libopus's opus_dred_parse and opus-rs's
/// OpusDredDecoder::parse (deferred and not, for several sample counts and
/// rates) and checks that both return the same and fill the same OpusDRED.
/// Returns the number of packets with DRED data.
fn compare_parse(s: Setup, packets: &[Vec<u8>]) -> usize {
    let cdec = RefDredDecoder::new();
    let rdec = opus_rs::dred::OpusDredDecoder::new().unwrap();
    let mut with_dred = 0;
    for (k, p) in packets.iter().enumerate() {
        for (fs, max) in [(s.fs, s.fs / 50), (s.fs, s.fs), (s.fs, 2 * s.fs), (16000, 4000), (48000, 30000)] {
            for defer in [false, true] {
                let mut c = OpusDRED::zeroed();
                let mut r = Box::<opus_rs::dred::OpusDred>::default();
                let (cret, cend) = cdec.parse(&mut c, p, max, fs, defer);
                let rres = rdec.parse(&mut r, p, max, fs, defer);
                assert!(cret >= 0, "packet {k}: {cret}");
                assert_eq!(rres, Ok((cret, cend)), "packet {k}, {max} samples at {fs} Hz");
                if let Err(e) = same_dred(&c, &r) {
                    panic!("packet {k}, {max} samples at {fs} Hz, defer {defer}: {e}");
                }
                if defer && c.process_stage == 1 {
                    assert_eq!(cdec.process(&mut c), 0);
                    rdec.process(&mut r).unwrap();
                    if let Err(e) = same_dred(&c, &r) {
                        panic!("packet {k} processed: {e}");
                    }
                }
                if fs == s.fs && max == s.fs && !defer && c.process_stage == 2 {
                    with_dred += 1;
                }
            }
        }
    }
    with_dred
}

/// libopus's encoder with DRED (SILK wideband at 16 and 48 kHz, 10 and
/// 20 ms, VBR and CBR, hybrid, CELT mono and stereo): opus-rs parses
/// every packet's DRED like opus_dred_parse.
#[test]
fn dred_parse_matches_libopus() {
    for (name, s) in [
        ("silk wb 16k", SILK_WB_16K),
        ("silk wb 48k", SILK_WB_48K),
        ("silk wb 16k cbr 10ms", SILK_WB_16K_CBR_10MS),
        ("hybrid 48k", HYBRID_48K),
        ("celt 48k", CELT_48K),
        ("celt 48k stereo", CELT_48K_STEREO),
    ] {
        let packets = ref_encode(s, &signal(s, 4, 5));
        let n = compare_parse(s, &packets);
        assert!(n > packets.len() / 2, "{name}: only {n} of {} packets carry DRED", packets.len());
    }
}

/// Packets without DRED or broken ones: both parsers report no DRED (or
/// an error) alike.
#[test]
fn dred_parse_edge_cases_match_libopus() {
    let cdec = RefDredDecoder::new();
    let rdec = opus_rs::dred::OpusDredDecoder::new().unwrap();
    let s = SILK_WB_16K;
    let packets = ref_encode(s, &signal(s, 2, 9));
    let mut r = Rng(3);
    for (k, p) in packets.iter().enumerate() {
        let mut q = p.clone();
        match k % 4 {
            0 => {}
            1 => {
                let i = r.below(q.len() as u32) as usize;
                q[i] = r.next_u32() as u8;
            }
            2 => q.truncate(1 + r.below(q.len() as u32) as usize),
            _ => {
                let i = 1 + r.below(q.len() as u32 - 1) as usize;
                q[i] ^= 1 << r.below(8);
            }
        }
        let mut c = OpusDRED::zeroed();
        let mut rd = Box::<opus_rs::dred::OpusDred>::default();
        let (cret, cend) = cdec.parse(&mut c, &q, s.fs, s.fs, false);
        let rres = rdec.parse(&mut rd, &q, s.fs, s.fs, false);
        if cret < 0 {
            assert!(rres.is_err(), "packet {k}: libopus {cret}, opus-rs {rres:?}");
            assert_eq!((c.process_stage, rd.process_stage), (-1, -1));
        } else {
            assert_eq!(rres, Ok((cret, cend)), "packet {k}");
            same_dred(&c, &rd).unwrap_or_else(|e| panic!("packet {k}: {e}"));
        }
    }
}

/// How a lost frame is concealed.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lost {
    /// One DRED decode per lost frame, at libopus opus_demo's offsets.
    Frame,
    /// Each lost frame in two halves, with their own offsets.
    Halves,
}

/// Decodes `packets` (None = lost) with libopus and opus-rs at
/// `complexity`, concealing losses before a packet with DRED from it as
/// opus_demo does. Returns the largest difference, the first differing
/// frame, and the number of frames concealed from DRED.
fn dred_decode_compare(s: Setup, packets: &[Option<Vec<u8>>], complexity: i32, how: Lost) -> (f32, Option<usize>, usize) {
    let frame = s.fs as usize * s.frame_ms / 1000;
    let ch = s.channels;
    let mut rs = opus_rs::OpusDecoder::new(s.fs, ch).unwrap();
    rs.set_complexity(complexity).unwrap();
    let cdred = RefDredDecoder::new();
    let rdred = opus_rs::dred::OpusDredDecoder::new().unwrap();
    let mut maxd = 0f32;
    let mut first = None;
    let mut from_dred = 0;
    let mut check = |k: usize, a: &[f32], b: &[f32], maxd: &mut f32| {
        for (x, y) in a.iter().zip(b) {
            let d = (x - y).abs();
            if d > 0.0 && first.is_none() {
                first = Some(k);
            }
            *maxd = maxd.max(d);
        }
    };
    unsafe {
        let mut err = 0;
        let dec = opus_decoder_create(s.fs, ch as i32, &mut err);
        opus_decoder_ctl(dec, SET_COMPLEXITY, complexity);
        let mut k = 0;
        while k < packets.len() {
            let Some(p) = &packets[k] else {
                k += 1;
                continue;
            };
            // The losses before this packet.
            let lost = (0..k).rev().take_while(|&j| packets[j].is_none()).count();
            if lost > 0 {
                let mut c = OpusDRED::zeroed();
                let mut r = Box::<opus_rs::dred::OpusDred>::default();
                let want = ((lost * frame) as i32).min(s.fs);
                let (cret, cend) = cdred.parse(&mut c, p, want, s.fs, false);
                assert_eq!(rdred.parse(&mut r, p, want, s.fs, false), Ok((cret, cend)), "packet {k}");
                same_dred(&c, &r).unwrap();
                for fr in 0..lost {
                    let j = k - lost + fr;
                    let parts: &[(usize, usize)] = match how {
                        Lost::Frame => &[(frame, (lost - fr) * frame)],
                        Lost::Halves => &[(frame / 2, (lost - fr) * frame), (frame / 2, (lost - fr) * frame - frame / 2)],
                    };
                    for &(n, offset) in parts {
                        let mut a = vec![0f32; n * ch];
                        let mut b = vec![0f32; n * ch];
                        let (na, nb) = if cret > 0 {
                            from_dred += 1;
                            (
                                opus_decoder_dred_decode_float(dec, &*c, offset as i32, a.as_mut_ptr(), n as i32),
                                rs.dred_decode_float(&r, offset as i32, &mut b, n).unwrap(),
                            )
                        } else {
                            (
                                opus_decode_float(dec, core::ptr::null(), 0, a.as_mut_ptr(), n as i32, 0),
                                rs.decode(&[], n, &mut b).unwrap(),
                            )
                        };
                        assert_eq!(na as usize, nb, "frame {j}");
                        check(j, &a, &b, &mut maxd);
                    }
                }
            }
            let mut a = vec![0f32; frame * ch];
            let mut b = vec![0f32; frame * ch];
            let na = opus_decode_float(dec, p.as_ptr(), p.len() as i32, a.as_mut_ptr(), frame as i32, 0);
            let nb = rs.decode(p, frame, &mut b).unwrap();
            assert_eq!(na as usize, nb, "frame {k}");
            check(k, &a, &b, &mut maxd);
            k += 1;
        }
        opus_decoder_destroy(dec);
    }
    (maxd, first, from_dred)
}

/// The same losses concealed by plain PLC, for the effect of DRED.
fn plc_output(s: Setup, packets: &[Option<Vec<u8>>], complexity: i32, dred: bool) -> Vec<f32> {
    let frame = s.fs as usize * s.frame_ms / 1000;
    let mut rs = opus_rs::OpusDecoder::new(s.fs, s.channels).unwrap();
    rs.set_complexity(complexity).unwrap();
    let rdred = opus_rs::dred::OpusDredDecoder::new().unwrap();
    let mut out = Vec::new();
    for (k, p) in packets.iter().enumerate() {
        let mut b = vec![0f32; frame * s.channels];
        match p {
            Some(p) => {
                rs.decode(p, frame, &mut b).unwrap();
            }
            None => {
                let next = (k..packets.len()).find(|&j| packets[j].is_some());
                let mut r = Box::<opus_rs::dred::OpusDred>::default();
                if let (true, Some(m)) = (dred, next) {
                    rdred.parse(&mut r, packets[m].as_ref().unwrap(), s.fs, s.fs, false).unwrap();
                }
                rs.dred_decode_float(&r, ((next.unwrap_or(k + 1) - k) * frame) as i32, &mut b, frame).unwrap();
            }
        }
        out.extend(b);
    }
    out
}

/// Losses concealed from DRED in SILK wideband (16 and 48 kHz, CBR
/// 10 ms), Hybrid and CELT (mono and stereo): bit-exact with libopus at
/// decoder complexity 0 and 5, with one DRED decode per frame and with
/// halves, and different from plain concealment (DRED was used).
#[test]
fn dred_decode_matches_libopus() {
    for (name, s) in [
        ("silk wb 16k", SILK_WB_16K),
        ("silk wb 48k", SILK_WB_48K),
        ("silk wb 16k cbr 10ms", SILK_WB_16K_CBR_10MS),
        ("hybrid 48k", HYBRID_48K),
        ("celt 48k", CELT_48K),
        ("celt 48k stereo", CELT_48K_STEREO),
    ] {
        let packets = with_losses(ref_encode(s, &signal(s, 6, 7)), 3, 15, 5, s.frame_ms);
        for complexity in [0, 5] {
            for how in [Lost::Frame, Lost::Halves] {
                let (d, f, n) = dred_decode_compare(s, &packets, complexity, how);
                assert_eq!((d, f), (0.0, None), "{name} complexity {complexity} {how:?}");
                assert!(n > 20, "{name}: only {n} DRED decodes");
            }
            let plain = plc_output(s, &packets, complexity, false);
            let dred = plc_output(s, &packets, complexity, true);
            let effect = plain.iter().zip(&dred).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            assert!(effect > 0.01, "{name} complexity {complexity}: DRED changed nothing ({effect})");
        }
    }
}

/// opus_decoder_dred_decode (16-bit output) against
/// OpusDecoder::dred_decode, also for losses concealed in one call of
/// 45 ms and 22.5 ms, which opus-rs conceals in pieces.
#[test]
fn dred_decode_i16_matches_libopus() {
    let s = SILK_WB_48K;
    let frame = s.fs as usize * s.frame_ms / 1000;
    let packets = ref_encode(s, &signal(s, 4, 11));
    let cdred = RefDredDecoder::new();
    let rdred = opus_rs::dred::OpusDredDecoder::new().unwrap();
    let mut rs = opus_rs::OpusDecoder::new(s.fs, 1).unwrap();
    let mut n_dred = 0;
    unsafe {
        let mut err = 0;
        let dec = opus_decoder_create(s.fs, 1, &mut err);
        let mut k = 0;
        let mut conceal = [frame, 2 * frame + frame / 4, frame + frame / 8].into_iter().cycle();
        while k < packets.len() {
            let p = &packets[k];
            if k > 50 && k % 7 == 0 && k + 3 < packets.len() {
                // Lose packets k and k + 1 and conceal them from packet k + 2.
                let q = &packets[k + 2];
                let mut c = OpusDRED::zeroed();
                let mut r = Box::<opus_rs::dred::OpusDred>::default();
                let (cret, cend) = cdred.parse(&mut c, q, 2 * frame as i32, s.fs, false);
                assert_eq!(rdred.parse(&mut r, q, 2 * frame as i32, s.fs, false), Ok((cret, cend)));
                let mut left = 2 * frame;
                while left > 0 {
                    let n = conceal.next().unwrap().min(left);
                    let mut a = vec![0i16; n];
                    let mut b = vec![0i16; n];
                    let na = opus_decoder_dred_decode(dec, &*c, left as i32, a.as_mut_ptr(), n as i32);
                    let nb = rs.dred_decode(&r, left as i32, &mut b, n).unwrap();
                    assert_eq!((na as usize, &a), (nb, &b), "packet {k}, {left} samples left");
                    left -= n;
                    n_dred += usize::from(cret > 0);
                }
                k += 2;
                continue;
            }
            let mut a = vec![0f32; frame];
            let mut b = vec![0f32; frame];
            opus_decode_float(dec, p.as_ptr(), p.len() as i32, a.as_mut_ptr(), frame as i32, 0);
            rs.decode(p, frame, &mut b).unwrap();
            assert_eq!(same_bits(&a, &b), None, "packet {k}");
            k += 1;
        }
        opus_decoder_destroy(dec);
    }
    assert!(n_dred > 10, "{n_dred}");
}
