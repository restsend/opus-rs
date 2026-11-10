//! A switch into CELT without redundancy must be bridged like libopus.
//!
//! When a CELT frame follows a SILK or Hybrid one that did not end in a
//! redundant CELT frame, libopus (`opus_decode_frame`) first conceals the
//! start of the new frame in the old mode, SILK with its resampler and, for
//! Hybrid, the CELT layer above SILK's band from its own state (5 ms, or
//! 2.5 ms for a 2.5 ms frame: `pcm_transition`). Only then does it reset CELT
//! and decode the frame, and it crossfades the concealed audio into it.
//!
//! opus-rs reset CELT before the bridge and built it from a separate SILK
//! concealment: no CELT layer, the left channel on both sides, no resampler at
//! equal rates (so the bridge came early by the resampler's delay), and no
//! bridge at all before a 2.5 ms frame.
//!
//! The streams come from a libopus encoder whose frames drop below 10 ms,
//! which it can code only in CELT and then switches without redundancy, and
//! are spliced from two libopus encoders held in different modes, as when a
//! sender restarts its encoder. The oracle is libopus 1.6.1 (`opusic-sys`).

use opus_rs::OpusDecoder;
use opusic_sys as ffi;
use std::f32::consts::PI;
use std::ffi::c_int;

/// `OPUS_SET_FORCE_MODE` and its modes (libopus `opus_private.h`).
const OPUS_SET_FORCE_MODE_REQUEST: c_int = 11002;
const MODE_SILK_ONLY: c_int = 1000;
const MODE_HYBRID: c_int = 1001;
const MODE_CELT_ONLY: c_int = 1002;

const RATE: usize = 48000;
/// 20 ms SILK or Hybrid packets before the switch.
const BEFORE: usize = 25;

struct Encoder(*mut ffi::OpusEncoder, usize);

impl Encoder {
    /// A 48 kHz encoder of `channels` for `application`, set up by `ctls`.
    fn new(application: c_int, channels: usize, ctls: &[(c_int, c_int)]) -> Self {
        let mut err: c_int = 0;
        let ptr = unsafe {
            ffi::opus_encoder_create(RATE as i32, channels as c_int, application, &mut err)
        };
        assert!(!ptr.is_null() && err == 0, "opus_encoder_create: {err}");
        for &(request, value) in ctls {
            assert_eq!(unsafe { ffi::opus_encoder_ctl(ptr, request, value) }, 0);
        }
        Self(ptr, channels)
    }

    /// An encoder held in `mode` at `bitrate`.
    fn forced(mode: c_int, bitrate: i32, channels: usize) -> Self {
        let bandwidth = if mode == MODE_SILK_ONLY {
            ffi::OPUS_BANDWIDTH_WIDEBAND
        } else {
            ffi::OPUS_BANDWIDTH_FULLBAND
        };
        Self::new(
            ffi::OPUS_APPLICATION_AUDIO,
            channels,
            &[
                (OPUS_SET_FORCE_MODE_REQUEST, mode),
                (ffi::OPUS_SET_BITRATE_REQUEST, bitrate),
                (ffi::OPUS_SET_BANDWIDTH_REQUEST, bandwidth),
            ],
        )
    }

    /// Packets of `pcm` in frames of `frame` samples per channel.
    fn encode(&mut self, pcm: &[f32], frame: usize) -> Vec<Vec<u8>> {
        pcm.chunks_exact(frame * self.1)
            .map(|x| {
                let mut out = [0u8; 1500];
                let n = unsafe {
                    ffi::opus_encode_float(
                        self.0,
                        x.as_ptr(),
                        frame as c_int,
                        out.as_mut_ptr(),
                        1500,
                    )
                };
                assert!(n > 0, "opus_encode_float: {n}");
                out[..n as usize].to_vec()
            })
            .collect()
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        unsafe { ffi::opus_encoder_destroy(self.0) }
    }
}

struct Decoder(*mut ffi::OpusDecoder, usize);

impl Decoder {
    fn new(rate: usize, channels: usize) -> Self {
        let mut err: c_int = 0;
        let ptr = unsafe { ffi::opus_decoder_create(rate as i32, channels as c_int, &mut err) };
        assert!(!ptr.is_null() && err == 0, "opus_decoder_create: {err}");
        Self(ptr, channels)
    }

    fn decode(&mut self, data: &[u8], out: &mut [f32]) -> usize {
        let n = unsafe {
            ffi::opus_decode_float(
                self.0,
                data.as_ptr(),
                data.len() as c_int,
                out.as_mut_ptr(),
                (out.len() / self.1) as c_int,
                0,
            )
        };
        assert!(n > 0, "opus_decode_float: {n}");
        n as usize
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe { ffi::opus_decoder_destroy(self.0) }
    }
}

/// A voiced signal (harmonics of a gliding 140 Hz pitch, a little noise,
/// under a syllable-rate envelope), interleaved in `channels`.
fn signal(n: usize, channels: usize) -> Vec<f32> {
    let mut seed = 7u32;
    let mut phase = 0.0f32;
    let mut out = Vec::with_capacity(n * channels);
    for i in 0..n {
        let t = i as f32 / RATE as f32;
        phase += 2.0 * PI * (140.0 + 30.0 * (2.0 * PI * 0.7 * t).sin()) / RATE as f32;
        let mut v = 0.0;
        for h in 1..=40 {
            v += 0.15 / h as f32 * (h as f32 * phase).sin();
        }
        let mut noise = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        };
        let x = (v + 0.04 * noise()) * (0.55 + 0.45 * (2.0 * PI * 3.0 * t).sin());
        out.push(x);
        if channels == 2 {
            out.push(0.7 * x + 0.03 * noise());
        }
    }
    out
}

/// `BEFORE` 20 ms packets of an encoder held in `mode`, then CELT packets of
/// `celt_frame` samples from an encoder that has coded CELT all along.
fn spliced(mode: c_int, bitrate: i32, channels: usize, celt_frame: usize) -> Vec<Vec<u8>> {
    let pcm = signal(960 * (BEFORE + 5), channels);
    let head = Encoder::forced(mode, bitrate, channels).encode(&pcm, 960);
    let celt = Encoder::forced(MODE_CELT_ONLY, 64000, channels).encode(&pcm, celt_frame);
    let first_celt = BEFORE * 960 / celt_frame;
    head[..BEFORE]
        .iter()
        .chain(&celt[first_celt..])
        .cloned()
        .collect()
}

fn snr(want: &[f32], got: &[f32]) -> f64 {
    let (mut s, mut e) = (0f64, 0f64);
    for (x, y) in want.iter().zip(got) {
        s += (*x as f64).powi(2);
        e += (*x as f64 - *y as f64).powi(2);
    }
    10.0 * (s / e.max(1e-30)).log10()
}

/// The lowest SNR against libopus 1.6.1 of the switch packet and the `after`
/// packets that follow it, decoded at `rate` in `channels`.
fn worst_at_switch(packets: &[Vec<u8>], rate: usize, channels: usize, after: usize) -> f64 {
    let mut c_dec = Decoder::new(rate, channels);
    let mut rs_dec = OpusDecoder::new(rate as i32, channels).unwrap();
    let (mut a, mut b) = (vec![0f32; 5760 * 2], vec![0f32; 5760 * 2]);
    let mut worst = f64::INFINITY;
    for (i, p) in packets.iter().enumerate().take(BEFORE + 1 + after) {
        let n = c_dec.decode(p, &mut a);
        assert_eq!(rs_dec.decode(p, 5760, &mut b).unwrap(), n, "packet {i}");
        if i >= BEFORE {
            worst = worst.min(snr(&a[..n * channels], &b[..n * channels]));
        }
    }
    worst
}

#[test]
fn switch_into_celt_without_redundancy_bridges_like_libopus() {
    let mut worst = (f64::INFINITY, String::new());
    for (mode, bitrate, name) in [
        (MODE_HYBRID, 24000, "Hybrid"),
        (MODE_SILK_ONLY, 32000, "SILK"),
    ] {
        for channels in [1usize, 2] {
            let packets = spliced(mode, bitrate, channels, 960);
            assert_eq!(
                packets[BEFORE - 1][0] & 0x04 != 0,
                channels == 2,
                "{name}: the stream is coded in {channels} channels"
            );
            for rate in [48000usize, 24000, 16000] {
                // A stereo stream is decoded in stereo only: this compares
                // the switch, not the downmix.
                for out in channels..=2 {
                    let db = worst_at_switch(&packets, rate, out, 1);
                    if db < worst.0 {
                        worst = (
                            db,
                            format!("{name} to CELT, {channels} to {out} ch, {rate} Hz"),
                        );
                    }
                }
            }
        }
    }
    println!("worst: {}: {:.1} dB from libopus", worst.1, worst.0);
    assert!(
        worst.0 > 40.0,
        "{}: {:.1} dB from libopus",
        worst.1,
        worst.0
    );
}

#[test]
fn short_celt_frames_after_the_switch_bridge_like_libopus() {
    // 2.5 ms frames get the 2.5 ms bridge faded into them; 5 and 10 ms ones
    // the full one.
    let mut worst = (f64::INFINITY, String::new());
    for (mode, bitrate, name) in [
        (MODE_HYBRID, 24000, "Hybrid"),
        (MODE_SILK_ONLY, 32000, "SILK"),
    ] {
        for celt_frame in [120usize, 240, 480] {
            let packets = spliced(mode, bitrate, 1, celt_frame);
            for rate in [48000usize, 24000, 16000] {
                let db = worst_at_switch(&packets, rate, 1, 2);
                if db < worst.0 {
                    let ms = celt_frame as f64 / 48.0;
                    worst = (db, format!("{name} to {ms} ms CELT, {rate} Hz"));
                }
            }
        }
    }
    println!("worst: {}: {:.1} dB from libopus", worst.1, worst.0);
    assert!(
        worst.0 > 40.0,
        "{}: {:.1} dB from libopus",
        worst.1,
        worst.0
    );
}

#[test]
fn libopus_switch_to_5_ms_frames_bridges_like_libopus() {
    // libopus's encoder codes no redundancy when it has to switch to CELT
    // for frames under 10 ms (opus_encode_frame_native): here an
    // application moves from 20 ms SILK or Hybrid frames to 5 ms ones.
    let mut worst = (f64::INFINITY, String::new());
    for (application, bitrate, name) in [
        (ffi::OPUS_APPLICATION_VOIP, 12000, "SILK"),
        (ffi::OPUS_APPLICATION_VOIP, 32000, "Hybrid"),
        (ffi::OPUS_APPLICATION_AUDIO, 24000, "Hybrid"),
    ] {
        for channels in [1usize, 2] {
            let pcm = signal(960 * (BEFORE + 5), channels);
            let mut enc = Encoder::new(
                application,
                channels,
                &[(ffi::OPUS_SET_BITRATE_REQUEST, bitrate)],
            );
            let split = BEFORE * 960 * channels;
            let mut packets = enc.encode(&pcm[..split], 960);
            packets.extend(enc.encode(&pcm[split..], 240));
            let mode = |p: &[u8]| match p[0] >> 3 {
                0..=11 => "SILK",
                12..=15 => "Hybrid",
                _ => "CELT",
            };
            assert_eq!(mode(&packets[BEFORE - 1]), name);
            assert_eq!(mode(&packets[BEFORE]), "CELT");
            for rate in [48000usize, 24000, 16000] {
                let db = worst_at_switch(&packets, rate, channels, 2);
                if db < worst.0 {
                    worst = (db, format!("{name} to CELT, {channels} ch, {rate} Hz"));
                }
            }
        }
    }
    println!("worst: {}: {:.1} dB from libopus", worst.1, worst.0);
    assert!(
        worst.0 > 40.0,
        "{}: {:.1} dB from libopus",
        worst.1,
        worst.0
    );
}
