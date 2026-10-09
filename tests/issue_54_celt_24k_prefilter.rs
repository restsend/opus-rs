//! Issue #54: CELT from 24 kHz input topped out at ~20 dB SNR at any bitrate
//! (20.1 / 20.9 / 21.1 dB at 32 / 64 / 128 kbps on the issue's chirp), while
//! 48 kHz input and libopus's own encoder at 24 kHz kept improving with bits.
//!
//! Root cause: `run_prefilter`'s cancel path. libopus 1.6 reverts the pitch
//! pre-filter for a frame when filtering makes it louder (`cancel_pitch`),
//! but it still fades the *previous* frame's comb filter out over the first
//! `overlap` samples (celt_encoder.c:1572-1584), because the decoder's
//! post-filter always cross-fades from the old gain to the new one. opus-rs
//! restored the unfiltered input over the whole frame instead, so after every
//! "on" frame that was cancelled, the decoder post-filtered 120 samples the
//! encoder never pre-filtered: an error burst no bitrate can remove.
//!
//! 48 kHz input rarely cancels: the comb filter attenuates the tone itself.
//! From 24 kHz, the zero-stuffed, pre-emphasized frame is dominated by its
//! image above 12 kHz, which the comb filter barely touches, so whether the
//! filter "helps" flips with the parity of the pitch period as the chirp
//! sweeps, and the cancel fired on about every other frame.
//!
//! The harness is the issue's: encode in 20 ms `Audio` frames, decode with
//! libopus (the `opus` dev-dependency) at 48 kHz, drop the 312-sample
//! pre-skip and take the SNR against the analytic chirp. On the unfixed tree
//! opus-rs scored ~21 dB at 64 and 128 kbps against libopus's 37-40. libopus
//! 1.5.2 has no `cancel_pitch` at all, so it is a reference for quality, not
//! for bits: the assertions are relative to libopus's own encoder and to
//! opus-rs's lower bitrates, never absolute dB.

use opus::{Application as CApp, Bitrate, Channels as CCh, Decoder as CDec, Encoder as CEnc};
use opus_rs::{Application, OpusEncoder};

const PRE_SKIP: usize = 312;

/// The issue's chirp: 200 Hz rising at 1200 Hz/s, amplitude 0.4 × `level`.
fn chirp(rate: usize, n: usize, level: f64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            ((std::f64::consts::TAU * (200.0 * t + 600.0 * t * t)).sin() * 0.4 * level) as f32
        })
        .collect()
}

/// 1.3 s of chirp, zero-padded to whole 20 ms frames covering the pre-skip.
fn input(rate: usize, level: f64) -> Vec<f32> {
    let (frame, len) = (rate / 50, rate * 13 / 10);
    let mut x = chirp(rate, len, level);
    x.resize(
        (len + PRE_SKIP * rate / 48_000).div_ceil(frame) * frame,
        0.0,
    );
    x
}

fn opus_rs_packets(rate: usize, bitrate: i32, level: f64) -> Vec<Vec<u8>> {
    let mut enc = OpusEncoder::new(rate as i32, 1, Application::Audio).unwrap();
    enc.bitrate_bps = bitrate;
    let frame = rate / 50;
    let mut pkt = [0u8; 4000];
    input(rate, level)
        .chunks(frame)
        .map(|f| {
            let n = enc.encode(f, frame, &mut pkt).unwrap();
            pkt[..n].to_vec()
        })
        .collect()
}

fn libopus_packets(rate: usize, bitrate: i32, level: f64) -> Vec<Vec<u8>> {
    let mut enc = CEnc::new(rate as u32, CCh::Mono, CApp::Audio).unwrap();
    enc.set_bitrate(Bitrate::Bits(bitrate)).unwrap();
    // Complexity 6 keeps libopus's tonality analysis off (opus_encoder.c:1252).
    // opus-rs does not port it, and from 24 kHz its leakage boosts are worth
    // another ~3 dB at 64 kbps; with it off the CELT paths compare like-for-like.
    enc.set_complexity(6).unwrap();
    // Same look-ahead as opus-rs, so the same pre-skip applies to both.
    assert_eq!(
        enc.get_lookahead().unwrap() as usize * 48_000 / rate,
        PRE_SKIP
    );
    let mut pkt = [0u8; 4000];
    input(rate, level)
        .chunks(rate / 50)
        .map(|f| {
            let n = enc.encode_float(f, &mut pkt).unwrap();
            pkt[..n].to_vec()
        })
        .collect()
}

/// SNR (dB) of the libopus 48 kHz decode against the analytic chirp.
fn snr(rate: usize, packets: &[Vec<u8>], level: f64) -> f64 {
    let mut dec = CDec::new(48_000, CCh::Mono).unwrap();
    let (mut out, mut pcm) = (Vec::new(), vec![0f32; 5760]);
    for p in packets {
        let n = dec.decode_float(p, &mut pcm, false).unwrap();
        out.extend_from_slice(&pcm[..n]);
    }
    let len = rate * 13 / 10 * (48_000 / rate);
    let got = &out[PRE_SKIP..PRE_SKIP + len];
    let want = chirp(48_000, len, level);
    let (mut sig, mut err) = (0f64, 0f64);
    for (&g, &w) in got.iter().zip(&want) {
        sig += f64::from(w) * f64::from(w);
        err += f64::from(g - w) * f64::from(g - w);
    }
    10.0 * (sig / err).log10()
}

/// TOC configs that are not CELT super-wideband 20 ms (config 27).
fn non_celt_swb(packets: &[Vec<u8>]) -> usize {
    packets.iter().filter(|p| p[0] >> 3 != 27).count()
}

#[test]
fn celt_24k_tracks_libopus() {
    for level in [0.9, 1.0, 1.1] {
        for bitrate in [64_000, 128_000] {
            let ours = opus_rs_packets(24_000, bitrate, level);
            let theirs = libopus_packets(24_000, bitrate, level);
            assert_eq!(non_celt_swb(&ours), 0, "opus-rs left CELT SWB at {bitrate}");
            assert_eq!(
                non_celt_swb(&theirs),
                0,
                "libopus left CELT SWB at {bitrate}"
            );
            let (rs, c) = (snr(24_000, &ours, level), snr(24_000, &theirs, level));
            assert!(
                rs >= c - 3.0,
                "24 kHz CELT at {} kbps (level {level}): opus-rs {rs:.1} dB vs libopus {c:.1} dB",
                bitrate / 1000
            );
        }
    }
}

#[test]
fn celt_24k_snr_rises_with_bitrate() {
    for level in [0.9, 1.0, 1.1] {
        let mut prev = f64::NEG_INFINITY;
        for bitrate in [32_000, 64_000, 128_000] {
            let packets = opus_rs_packets(24_000, bitrate, level);
            assert_eq!(
                non_celt_swb(&packets),
                0,
                "opus-rs left CELT SWB at {bitrate}"
            );
            let db = snr(24_000, &packets, level);
            assert!(
                db >= prev + 3.0,
                "24 kHz SNR stopped rising at {} kbps (level {level}): {db:.1} dB after {prev:.1} dB",
                bitrate / 1000
            );
            prev = db;
        }
    }
}
