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
