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
        Self { rs, c, channels: mapping.len() }
    }

    /// Decodes with both and returns (opus-rs, libopus) results and PCM.
    fn decode(&mut self, p: &[u8], frame_size: usize) -> (Result<usize, &'static str>, i32, Vec<f32>, Vec<f32>) {
        let mut o_rs = vec![0f32; frame_size * self.channels];
        let mut o_c = vec![0f32; frame_size * self.channels];
        let ptr = if p.is_empty() { core::ptr::null() } else { p.as_ptr() };
        let n = unsafe {
            opus_multistream_decode_float(self.c, ptr, p.len() as i32, o_c.as_mut_ptr(), frame_size as i32, 0)
        };
        let r = self.rs.decode(p, frame_size, &mut o_rs);
        (r, n, o_rs, o_c)
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
    let (r, n, o_rs, o_c) = d.decode(p, frame_size);
    assert!(n > 0, "libopus {n}");
    assert_eq!(r, Ok(n as usize));
    let peak = o_c.iter().fold(1f32, |m, x| m.max(x.abs()));
    let diff = o_rs.iter().zip(&o_c).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(diff <= 4e-6 * peak, "PCM differs by {diff}");
}

/// libopus rejects `bad` before any stream decodes it, so the packet after
/// it decodes the same.
fn rejected_without_state_change(bad: &[u8]) {
    const FB20: u8 = 31 << 3;
    let mut d = Pair::new(2, 0, &[0, 1]);
    assert_same(&mut d, &two_streams(&celt(FB20, 60, 1), &celt(FB20, 60, 2)), 960);
    let (r, n, ..) = d.decode(bad, 960);
    assert!(n < 0, "libopus {n}");
    assert!(r.is_err(), "opus-rs {r:?}");
    assert_same(&mut d, &two_streams(&celt(FB20, 60, 5), &celt(FB20, 60, 6)), 960);
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
