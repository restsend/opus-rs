//! MultistreamDecoder against libopus's opus_multistream_decoder on
//! directed packets and losses.
use core::ffi::c_void;
use opus_rs::multistream::{ChannelMappingTable, MultistreamDecoder};
use opusref::*;

/// Both decoders for one layout at 48 kHz.
struct Pair {
    rs: MultistreamDecoder,
    c: *mut c_void,
    channels: usize,
}

impl Pair {
    fn new(streams: u8, coupled: u8, mapping: &[u8]) -> Self {
        let table = ChannelMappingTable {
            channels: mapping.len() as u8,
            stream_count: streams,
            coupled_count: coupled,
            mapping: mapping.to_vec(),
        };
        let mut err = 0;
        let c = unsafe {
            opus_multistream_decoder_create(
                48000,
                mapping.len() as i32,
                streams as i32,
                coupled as i32,
                mapping.as_ptr(),
                &mut err,
            )
        };
        assert!(!c.is_null(), "libopus error {err}");
        let rs = MultistreamDecoder::new(48000, table).unwrap();
        Self {
            rs,
            c,
            channels: mapping.len(),
        }
    }

    /// Decodes with both and returns (opus-rs, libopus) results and PCM.
    fn decode(
        &mut self,
        p: &[u8],
        frame_size: usize,
    ) -> (Result<usize, &'static str>, i32, Vec<f32>, Vec<f32>) {
        self.decode_with(p, frame_size, false)
    }

    fn decode_with(
        &mut self,
        p: &[u8],
        frame_size: usize,
        fec: bool,
    ) -> (Result<usize, &'static str>, i32, Vec<f32>, Vec<f32>) {
        let mut o_rs = vec![0f32; frame_size * self.channels];
        let mut o_c = vec![0f32; frame_size * self.channels];
        let ptr = if p.is_empty() {
            core::ptr::null()
        } else {
            p.as_ptr()
        };
        let n = unsafe {
            opus_multistream_decode_float(
                self.c,
                ptr,
                p.len() as i32,
                o_c.as_mut_ptr(),
                frame_size as i32,
                fec as i32,
            )
        };
        let r = if fec {
            self.rs.decode_fec(p, frame_size, &mut o_rs)
        } else {
            self.rs.decode(p, frame_size, &mut o_rs)
        };
        (r, n, o_rs, o_c)
    }

    fn final_range(&self) -> u32 {
        let mut r = 0u32;
        unsafe { opus_multistream_decoder_ctl(self.c, GET_FINAL_RANGE, &mut r as *mut u32) };
        r
    }
}

impl Drop for Pair {
    fn drop(&mut self) {
        unsafe { opus_multistream_decoder_destroy(self.c) }
    }
}

#[test]
fn concealment_is_limited_to_120_ms() {
    let mut d = Pair::new(1, 0, &[0]);
    let (r, n, ..) = d.decode(&[], 10000);
    assert_eq!(n, 5760, "libopus");
    assert_eq!(r, Ok(5760), "opus-rs");
}

/// A CELT-only frame: ToC `toc` (code 0) and `n` pseudo-random bytes.
fn celt(toc: u8, n: usize, seed: u32) -> Vec<u8> {
    let mut x = seed;
    let mut p = vec![toc];
    p.extend((0..n).map(|_| {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        (x >> 24) as u8
    }));
    p
}

/// Two mono streams: `a` self-delimited, then `b`.
fn two_streams(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut p = vec![a[0], (a.len() - 1) as u8];
    p.extend_from_slice(&a[1..]);
    p.extend_from_slice(b);
    p
}

fn assert_same(d: &mut Pair, p: &[u8], frame_size: usize) {
    assert_same_with(d, p, frame_size, false);
}

fn assert_same_with(d: &mut Pair, p: &[u8], frame_size: usize, fec: bool) {
    let (r, n, o_rs, o_c) = d.decode_with(p, frame_size, fec);
    assert!(n > 0, "libopus {n}");
    assert_eq!(r, Ok(n as usize));
    let peak = o_c.iter().fold(1f32, |m, x| m.max(x.abs()));
    let diff = o_rs
        .iter()
        .zip(&o_c)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(diff <= 4e-6 * peak, "PCM differs by {diff}");
    assert_eq!(d.rs.final_range(), d.final_range(), "final range");
}

/// libopus rejects `bad` before any stream decodes it, so the packet after
/// it decodes the same.
fn rejected_without_state_change(bad: &[u8]) {
    const FB20: u8 = 31 << 3;
    let mut d = Pair::new(2, 0, &[0, 1]);
    assert_same(
        &mut d,
        &two_streams(&celt(FB20, 60, 1), &celt(FB20, 60, 2)),
        960,
    );
    let (r, n, ..) = d.decode(bad, 960);
    assert!(n < 0, "libopus {n}");
    assert!(r.is_err(), "opus-rs {r:?}");
    assert_same(
        &mut d,
        &two_streams(&celt(FB20, 60, 5), &celt(FB20, 60, 6)),
        960,
    );
}

#[test]
fn an_unparsable_second_stream_is_invalid() {
    // The last stream is a ToC-only code-2 packet, which is invalid.
    rejected_without_state_change(&two_streams(&celt(31 << 3, 60, 3), &[(31 << 3) | 2]));
}

#[test]
fn streams_of_different_durations_are_invalid() {
    rejected_without_state_change(&two_streams(&celt(31 << 3, 60, 3), &celt(30 << 3, 40, 4)));
}

#[test]
fn an_empty_last_stream_is_invalid() {
    let p = two_streams(&celt(31 << 3, 60, 3), &[0]);
    rejected_without_state_change(&p[..p.len() - 1]);
}

#[test]
fn an_empty_packet_conceals_every_stream() {
    const FB20: u8 = 31 << 3;
    let mut d = Pair::new(2, 0, &[1, 0]);
    assert_same(
        &mut d,
        &two_streams(&celt(FB20, 60, 1), &celt(FB20, 60, 2)),
        960,
    );
    assert_same(&mut d, &[], 960);
    assert_same(&mut d, &[], 480);
    assert_same(
        &mut d,
        &two_streams(&celt(FB20, 60, 5), &celt(FB20, 60, 6)),
        960,
    );
}

#[test]
fn fec_recovers_every_stream() {
    let mapping = [0u8, 1, 2];
    let (streams, coupled) = (2u8, 1u8);
    let mut err = 0;
    let enc = unsafe {
        opus_multistream_encoder_create(
            48000,
            3,
            streams as i32,
            coupled as i32,
            mapping.as_ptr(),
            2048,
            &mut err,
        )
    };
    assert!(!enc.is_null(), "libopus error {err}");
    unsafe {
        opus_multistream_encoder_ctl(enc, SET_BITRATE, 40000);
        opus_multistream_encoder_ctl(enc, SET_INBAND_FEC, 1);
        opus_multistream_encoder_ctl(enc, SET_PACKET_LOSS_PERC, 20);
    }
    let mut packets = Vec::new();
    for k in 0..30 {
        let pcm: Vec<f32> = (0..960 * 3)
            .map(|i| {
                let t = (k * 960 + i / 3) as f32 / 48000.0;
                0.3 * (t * 220.0 * (1.0 + (i % 3) as f32) * core::f32::consts::TAU).sin()
            })
            .collect();
        let mut out = vec![0u8; 1500];
        let n = unsafe {
            opus_multistream_encode_float(enc, pcm.as_ptr(), 960, out.as_mut_ptr(), 1500)
        };
        assert!(n > 0);
        out.truncate(n as usize);
        packets.push(out);
    }
    unsafe { opus_multistream_encoder_destroy(enc) };

    let mut d = Pair::new(streams, coupled, &mapping);
    for (k, p) in packets.iter().enumerate() {
        match k % 5 {
            3 => continue,
            4 => assert_same_with(&mut d, p, 960, true),
            _ => {}
        }
        assert_same(&mut d, p, 960);
    }
}

/// libopus-encoded multistream packets of a 5.1-like layout.
fn encoded(ch: usize, streams: i32, coupled: i32, count: usize) -> Vec<Vec<u8>> {
    let mapping: Vec<u8> = (0..ch as u8).collect();
    let mut err = 0;
    let enc = unsafe {
        opus_multistream_encoder_create(48000, ch as i32, streams, coupled, mapping.as_ptr(), 2049, &mut err)
    };
    assert!(!enc.is_null());
    unsafe { opus_multistream_encoder_ctl(enc, SET_BITRATE, 48000 * streams) };
    let out = (0..count)
        .map(|f| {
            let pcm: Vec<f32> = (0..960 * ch)
                .map(|i| ((f * 960 + i / ch) as f32 * 0.01 * (1 + i % ch) as f32).sin() * 0.9)
                .collect();
            let mut p = vec![0u8; 4000];
            let n = unsafe { opus_multistream_encode_float(enc, pcm.as_ptr(), 960, p.as_mut_ptr(), 4000) };
            p.truncate(n as usize);
            p
        })
        .collect();
    unsafe { opus_multistream_encoder_destroy(enc) };
    out
}

/// The controls, queries, reset and integer outputs of the multistream
/// decoder against libopus's: gain, complexity and phase inversion reach
/// every stream and survive a reset, and the queries report the first
/// stream.
#[test]
fn controls_queries_and_integer_output_match_libopus() {
    let mut d = Pair::new(4, 2, &[0, 1, 2, 3, 4, 5]);
    let pkts = encoded(6, 4, 2, 12);
    let get = |d: &Pair, req| {
        let mut v = 0i32;
        unsafe { opus_multistream_decoder_ctl(d.c, req, &mut v as *mut i32) };
        v
    };
    let code = |r: Result<(), &str>| r.map_or_else(|e| opus_rs::error::code(e).unwrap(), |()| 0);
    assert_eq!(code(d.rs.set_gain(40000)), unsafe { opus_multistream_decoder_ctl(d.c, SET_GAIN, 40000) });
    assert_eq!(code(d.rs.set_gain(-700)), unsafe { opus_multistream_decoder_ctl(d.c, SET_GAIN, -700) });
    assert_eq!(code(d.rs.set_complexity(11)), unsafe { opus_multistream_decoder_ctl(d.c, SET_COMPLEXITY, 11) });
    assert_eq!(code(d.rs.set_complexity(6)), unsafe { opus_multistream_decoder_ctl(d.c, SET_COMPLEXITY, 6) });
    d.rs.set_phase_inversion_disabled(true);
    unsafe { opus_multistream_decoder_ctl(d.c, SET_PHASE_INVERSION_DISABLED, 1) };
    for (f, p) in pkts.iter().enumerate() {
        if f == 6 {
            d.rs.reset();
            unsafe { opus_multistream_decoder_ctl(d.c, RESET_STATE) };
        }
        // 16-bit and 24-bit output in turn, as integers within one step
        // of the float tolerance.
        let (a, b, x, y): (_, _, Vec<i64>, Vec<i64>) = if f % 2 == 0 {
            let (mut x, mut y) = (vec![0i16; 960 * 6], vec![0i16; 960 * 6]);
            let a = d.rs.decode_i16(p, 960, &mut x);
            let b = unsafe { opus_multistream_decode(d.c, p.as_ptr(), p.len() as i32, y.as_mut_ptr(), 960, 0) };
            (a, b, x.iter().map(|&v| v as i64).collect(), y.iter().map(|&v| v as i64).collect())
        } else {
            let (mut x, mut y) = (vec![0i32; 960 * 6], vec![0i32; 960 * 6]);
            let a = d.rs.decode24(p, 960, &mut x);
            let b = unsafe { opus_multistream_decode24(d.c, p.as_ptr(), p.len() as i32, y.as_mut_ptr(), 960, 0) };
            (a, b, x.iter().map(|&v| v as i64 / 256).collect(), y.iter().map(|&v| v as i64 / 256).collect())
        };
        assert_eq!(a, Ok(b as usize), "frame {f}");
        let worst = x.iter().zip(&y).map(|(a, b)| (a - b).abs()).max().unwrap();
        assert!(worst <= 1, "frame {f}: {worst}");
        assert_eq!(d.rs.final_range(), d.final_range(), "frame {f}");
        let got = (
            d.rs.gain(),
            d.rs.complexity(),
            d.rs.phase_inversion_disabled() as i32,
            d.rs.bandwidth().map_or(0, |b| b as i32),
            d.rs.sample_rate(),
            d.rs.last_packet_duration() as i32,
        );
        let want = (
            get(&d, GET_GAIN),
            get(&d, GET_COMPLEXITY),
            get(&d, GET_PHASE_INVERSION_DISABLED),
            get(&d, GET_BANDWIDTH),
            get(&d, GET_SAMPLE_RATE),
            get(&d, GET_LAST_PACKET_DURATION),
        );
        assert_eq!(got, want, "frame {f}");
    }
    assert!(d.rs.decoder_state(3).is_ok());
    assert_eq!(d.rs.decoder_state(4).err(), Some(opus_rs::error::BAD_ARG));
}

/// The errors of opus_multistream_decode_native: OPUS_BAD_ARG for a frame
/// size of 0, OPUS_INVALID_PACKET for
/// a packet too short for its streams or with a malformed stream,
/// OPUS_BUFFER_TOO_SMALL for a frame size below the packet's.
#[test]
fn errors_match_libopus() {
    const FB20: u8 = 31 << 3;
    let good = two_streams(&celt(FB20, 60, 1), &celt(FB20, 60, 2));
    for (p, frame) in [
        (vec![FB20], 960),
        (two_streams(&celt(FB20, 60, 1), &[FB20 | 1, 0, 0]), 960),
        (two_streams(&celt(FB20, 60, 1), &celt(FB20 - 8, 60, 2)), 960),
        (good.clone(), 480),
        (good.clone(), 0),
        (vec![], 0),
    ] {
        let mut d = Pair::new(2, 0, &[0, 1]);
        let (r, n, ..) = d.decode(&p, frame);
        assert!(n < 0, "libopus {n}");
        assert_eq!(r.map_err(opus_rs::error::code), Err(Some(n)), "{p:02x?} frame {frame}");
    }
}
