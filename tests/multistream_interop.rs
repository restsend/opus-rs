//! Differential tests: compare `MultistreamDecoder` against a real libopus
//! decode of a mapping-family-1 (5.1/7.1) Ogg-Opus file.
//!
//! The `multistream` module is `std`-gated, so this whole test crate is
//! skipped in `no_std` builds.
#![cfg(feature = "std")]
//! Requires, per case, in tmp/opus_ref/ (relative to the crate root — run `cargo test` from there):
//!   `<name>.pkts` — raw container packets, length-prefixed u16 LE, fed
//!                   whole (don't pre-split by stream).
//!   `<name>.pcm`  — libopus reference decode, interleaved f32 LE, e.g.
//!                   `ffmpeg -i in.opus -f f32le -acodec pcm_f32le ref.pcm`.
//!   `<name>.map`  — channels(1) | stream_count(1) | coupled_count(1) |
//!                   mapping[channels] — i.e. OpusHead byte 9 + bytes 19..
//! Tests are skipped if the reference files are not present.

use opus_rs::OpusDecoder;
use opus_rs::multistream::{ChannelMappingTable, MultistreamDecoder, split_self_delimited};
use std::fs;

const OPUS_DECODE_RATE: i32 = 48_000;
const MAX_FRAME_SIZE: usize = 5760 * 3; // RFC 6716: 120ms @ 48kHz, up to 3 frames/packet

struct RefData {
    packets: Vec<Vec<u8>>,
    ref_pcm: Vec<f32>,
    table: ChannelMappingTable,
    pre_skip: u16,
}

fn load_ref(path_prefix: &str) -> Option<RefData> {
    let pkt_raw = fs::read(format!("tmp/opus_ref/{}.pkts", path_prefix)).ok()?;
    let pcm_raw = fs::read(format!("tmp/opus_ref/{}.pcm", path_prefix)).ok()?;
    let map_raw = fs::read(format!("tmp/opus_ref/{}.map", path_prefix)).ok()?;

    // .map layout: pre_skip(2 LE) | channels(1) | stream_count(1) | coupled_count(1) | mapping[channels]
    if map_raw.len() < 5 {
        return None;
    }
    let pre_skip = u16::from_le_bytes([map_raw[0], map_raw[1]]);
    let table = ChannelMappingTable::parse(1, map_raw[2], &map_raw[3..])?;

    let mut packets = Vec::new();
    let mut pos = 0;
    while pos + 2 <= pkt_raw.len() {
        let n = u16::from_le_bytes([pkt_raw[pos], pkt_raw[pos + 1]]) as usize;
        pos += 2;
        if pos + n > pkt_raw.len() {
            break;
        }
        packets.push(pkt_raw[pos..pos + n].to_vec());
        pos += n;
    }

    let ref_pcm: Vec<f32> = pcm_raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    Some(RefData {
        packets,
        ref_pcm,
        table,
        pre_skip,
    })
}

fn goertzel_power(samples: &[f32], sample_rate: f64, freq: f64) -> f64 {
    let n = samples.len();
    if n == 0 {
        return 0.0;
    }
    let k = (0.5 + (n as f64 * freq / sample_rate)) as usize;
    let w = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
    let coeff = 2.0 * w.cos();
    let (mut q1, mut q2) = (0.0f64, 0.0f64);
    for &s in samples {
        let q0 = coeff * q1 - q2 + s as f64;
        q2 = q1;
        q1 = q0;
    }
    let real = q1 - q2 * w.cos();
    let imag = q2 * w.sin();
    (real * real + imag * imag).sqrt()
}

fn run_diff(label: &str, data: &RefData) {
    let channels = data.table.channels as usize;
    let mut dec = MultistreamDecoder::new(
        OPUS_DECODE_RATE,
        ChannelMappingTable {
            channels: data.table.channels,
            stream_count: data.table.stream_count,
            coupled_count: data.table.coupled_count,
            mapping: data.table.mapping.clone(),
        },
    )
    .unwrap_or_else(|e| panic!("{}: failed to build MultistreamDecoder: {}", label, e));

    // Frame-boundary diagnostic: independently re-run split_self_delimited on
    // the first couple of real packets to inspect exactly what byte range and
    // TOC each stream gets, without going through the decoder at all. Also
    // decode each sub-packet standalone (fresh OpusDecoder per stream, per
    // packet) to isolate whether corruption is in per-stream decode itself
    // vs. MultistreamDecoder's aggregation/scatter logic.
    for (pkt_idx, pkt) in data.packets.iter().take(2).enumerate() {
        eprintln!(
            "{}: packet {} total_len={} toc={:02x}",
            label,
            pkt_idx,
            pkt.len(),
            pkt[0]
        );
        let mut rest = pkt.as_slice();
        let mut split_scratch = Vec::new();
        for s in 0..data.table.stream_count as usize {
            let is_last = s + 1 == data.table.stream_count as usize;
            let stream_ch = if (s as u8) < data.table.coupled_count {
                2
            } else {
                1
            };
            let sub: Vec<u8> = if is_last {
                rest.to_vec()
            } else {
                match split_self_delimited(rest, &mut split_scratch) {
                    Ok((sub, consumed)) => {
                        let sub = sub.to_vec();
                        rest = &rest[consumed..];
                        sub
                    }
                    Err(e) => {
                        eprintln!(
                            "{}:   stream {}: split_self_delimited ERROR: {}",
                            label, s, e
                        );
                        continue;
                    }
                }
            };

            eprintln!(
                "{}:   stream {}: len={} toc={:02x} ch={}",
                label,
                s,
                sub.len(),
                sub.first().copied().unwrap_or(0),
                stream_ch
            );

            match OpusDecoder::new(OPUS_DECODE_RATE, stream_ch) {
                Ok(mut standalone_dec) => {
                    let mut buf = vec![0.0f32; MAX_FRAME_SIZE * stream_ch];
                    match standalone_dec.decode(&sub, MAX_FRAME_SIZE, &mut buf) {
                        Ok(n) => {
                            let preview: Vec<f32> =
                                buf[..(6 * stream_ch).min(n * stream_ch)].to_vec();
                            eprintln!(
                                "{}:     standalone decode: n={} preview={:?}",
                                label, n, preview
                            );
                        }
                        Err(e) => eprintln!("{}:     standalone decode ERROR: {:?}", label, e),
                    }
                }
                Err(e) => eprintln!("{}:     standalone OpusDecoder::new ERROR: {:?}", label, e),
            }
        }
    }

    let mut rust_pcm: Vec<f32> = Vec::new();
    for pkt in &data.packets {
        let mut buf = vec![0.0f32; MAX_FRAME_SIZE * channels];
        match dec.decode(pkt, MAX_FRAME_SIZE, &mut buf) {
            Ok(n) => rust_pcm.extend_from_slice(&buf[..n * channels]),
            Err(e) => panic!(
                "{}: decode error on packet (toc={:02x}): {}",
                label, pkt[0], e
            ),
        }
    }

    // MultistreamDecoder does no pre-skip trimming itself (same as plain
    // OpusDecoder) — that's OpusSource's job in Audion, not this crate's.
    // ffmpeg's reference decode already trims it, so drain it here to compare
    // like-for-like instead of a constant-offset mismatch across every window.
    let skip_samples = data.pre_skip as usize * channels;
    if skip_samples < rust_pcm.len() {
        rust_pcm.drain(0..skip_samples);
    }

    // rust_pcm is in the RFC 7845 Table 2 fixed Vorbis-order layout
    // (6ch: L, C, R, RL, RR, LFE; 8ch: L, C, R, SL, SR, RL, RR, LFE).
    // ffmpeg's raw f32le export instead uses WAVE/SMPTE order for its own
    // "5.1"/"7.1" channel-layout tags (5.1: FL,FR,FC,LFE,BL,BR; 7.1 adds
    // SL,SR: FL,FR,FC,LFE,BL,BR,SL,SR), so reorder rust_pcm to match before
    // comparing — this is ffmpeg's own remapping on the reference side, not
    // a decoder bug.
    let vorbis_to_wave: Option<&[usize]> = match channels {
        6 => Some(&[0, 2, 1, 5, 3, 4]),
        8 => Some(&[0, 2, 1, 7, 5, 6, 3, 4]),
        _ => None,
    };
    if let Some(perm) = vorbis_to_wave {
        let frames = rust_pcm.len() / channels;
        let mut reordered = vec![0.0f32; rust_pcm.len()];
        for f in 0..frames {
            for (wave_c, &vorbis_c) in perm.iter().enumerate() {
                reordered[f * channels + wave_c] = rust_pcm[f * channels + vorbis_c];
            }
        }
        rust_pcm = reordered;
    }

    eprintln!(
        "{}: decoded {} frames ({} samples), ref has {} samples",
        label,
        rust_pcm.len() / channels,
        rust_pcm.len(),
        data.ref_pcm.len()
    );
    eprintln!(
        "{}: rust[0..12]  = {:?}",
        label,
        &rust_pcm[..12.min(rust_pcm.len())]
    );
    eprintln!(
        "{}: ref[0..12]   = {:?}",
        label,
        &data.ref_pcm[..12.min(data.ref_pcm.len())]
    );

    // A second sample point well past any startup transient/preskip-adjacent
    // region, so a channel-order bug shows up clearly against steady-state signal.
    let mid = (rust_pcm.len() / 2 / channels) * channels;
    if mid + 12 <= rust_pcm.len() && mid + 12 <= data.ref_pcm.len() {
        eprintln!(
            "{}: rust[mid..mid+12] = {:?}",
            label,
            &rust_pcm[mid..mid + 12]
        );
        eprintln!(
            "{}: ref[mid..mid+12]  = {:?}",
            label,
            &data.ref_pcm[mid..mid + 12]
        );
    }

    // Per-channel dominant-frequency check: identifies which known tone lives
    // in each channel column of rust_pcm vs. ref_pcm independently, so a
    // channel-order mismatch shows up directly without relying on sample-exact
    // alignment between the two decodes.
    const CANDIDATE_FREQS: [f64; 8] = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];
    let analysis_start = (48_000usize / 10) * channels; // skip 100ms of onset
    let analysis_len = 4_800 * channels; // 100ms window
    if analysis_start + analysis_len <= rust_pcm.len()
        && analysis_start + analysis_len <= data.ref_pcm.len()
    {
        for ch_idx in 0..channels {
            let rust_ch: Vec<f32> = rust_pcm[analysis_start..analysis_start + analysis_len]
                .iter()
                .skip(ch_idx)
                .step_by(channels)
                .copied()
                .collect();
            let ref_ch: Vec<f32> = data.ref_pcm[analysis_start..analysis_start + analysis_len]
                .iter()
                .skip(ch_idx)
                .step_by(channels)
                .copied()
                .collect();

            let rust_dom = CANDIDATE_FREQS
                .iter()
                .map(|&f| (f, goertzel_power(&rust_ch, 48_000.0, f)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            let ref_dom = CANDIDATE_FREQS
                .iter()
                .map(|&f| (f, goertzel_power(&ref_ch, 48_000.0, f)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            eprintln!(
                "{}: channel {} — rust dominant: {:.0}Hz ({:.4}), ref dominant: {:.0}Hz ({:.4})",
                label, ch_idx, rust_dom.0, rust_dom.1, ref_dom.0, ref_dom.1
            );
        }
    }

    // Windowed SNR against the reference, lag-searched to absorb small
    // alignment jitter between independent decode pipelines.
    const WINDOW_FRAMES: usize = 960; // 20ms @ 48kHz
    let window = WINDOW_FRAMES * channels;
    let mut pos = 0usize;
    let mut n_windows = 0usize;
    let mut n_exact = 0usize;
    let mut n_good = 0usize;
    let mut n_bad = 0usize;
    let mut worst_snr = f64::MAX;
    let mut worst_window = 0usize;

    while pos + window <= rust_pcm.len() && pos + window <= data.ref_pcm.len() {
        let r_win = &rust_pcm[pos..pos + window];
        let f_win = &data.ref_pcm[pos..pos + window];

        let mut best_snr = f64::MIN;
        for lag in -30i32..=30i32 {
            let lag = lag as isize;
            let (rs, fs): (&[f32], &[f32]) = if lag >= 0 {
                let l = lag as usize;
                if l >= window {
                    continue;
                }
                (&r_win[l..], &f_win[..window - l])
            } else {
                let l = (-lag) as usize;
                if l >= window {
                    continue;
                }
                (&r_win[..window - l], &f_win[l..])
            };
            let m = rs.len().min(fs.len());
            if m < 10 {
                continue;
            }
            let mut ef = 0.0f64;
            let mut err = 0.0f64;
            for i in 0..m {
                let r = rs[i] as f64;
                let f = fs[i] as f64;
                ef += f * f;
                err += (r - f) * (r - f);
            }
            if ef > 0.0 {
                let snr = if err > 1e-30 {
                    10.0 * (ef / err).log10()
                } else {
                    999.0
                };
                if snr > best_snr {
                    best_snr = snr;
                }
            }
        }

        if best_snr > 100.0 {
            n_exact += 1;
        } else if best_snr > 40.0 {
            n_good += 1;
        } else {
            n_bad += 1;
            if best_snr < worst_snr {
                worst_snr = best_snr;
                worst_window = n_windows;
            }
        }

        pos += window;
        n_windows += 1;
    }

    let worst_desc = if n_bad > 0 {
        format!("worst {:.1}dB @ window {}", worst_snr, worst_window)
    } else {
        "worst n/a (no bad windows)".to_string()
    };
    println!(
        "{}: {}ch, {} windows — exact:{} good:{} bad:{} | {}",
        label, channels, n_windows, n_exact, n_good, n_bad, worst_desc
    );

    assert!(
        n_bad == 0,
        "{}: {} of {} windows below 40dB SNR (worst {:.1}dB @ window {})",
        label,
        n_bad,
        n_windows,
        worst_snr,
        worst_window
    );
}

/// No fixture needed: a `channel_mapping[]` entry of 255 must always zero
/// that output channel regardless of decoded content.
#[test]
fn multistream_silence_mapping_produces_silence() {
    let table = ChannelMappingTable {
        channels: 3,
        stream_count: 1,
        coupled_count: 0,
        mapping: vec![0, 0, 255],
    };
    let mut dec = MultistreamDecoder::new(OPUS_DECODE_RATE, table).unwrap();

    let packet = [0u8]; // not a real encoded frame; swap for a captured mono packet if rejected
    let frame_size = 960;
    let mut out = vec![1.0f32; frame_size * 3];

    if let Ok(frames_decoded) = dec.decode(&packet, frame_size, &mut out) {
        for f in 0..frames_decoded {
            assert_eq!(out[f * 3 + 2], 0.0);
        }
    }
}

#[test]
fn diff_multistream_5_1() {
    match load_ref("surround_5_1") {
        Some(data) => run_diff("multistream 5.1 (family 1)", &data),
        None => {
            eprintln!("diff_multistream_5_1: SKIPPED (no fixtures at tmp/opus_ref/surround_5_1.*)")
        }
    }
}

#[test]
fn diff_multistream_7_1() {
    match load_ref("surround_7_1") {
        Some(data) => run_diff("multistream 7.1 (family 1)", &data),
        None => {
            eprintln!("diff_multistream_7_1: SKIPPED (no fixtures at tmp/opus_ref/surround_7_1.*)")
        }
    }
}
