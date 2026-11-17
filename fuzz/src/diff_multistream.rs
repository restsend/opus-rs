//! Differential multistream decoder fuzzer: random channel layouts
//! (channels, streams, coupled streams, mapping), random and
//! libopus-encoded multistream packets and losses, decoded by opus-rs's
//! MultistreamDecoder and libopus's opus_multistream_decoder.
//!
//! Oracles: both accept or both reject the layout (`layout`); per call, both
//! succeed with the same sample count or both fail (`result`); the samples
//! agree within `OFZ_DEC_TOL` relative to max(1, peak) (`pcm`, default 4e-6).

use crate::common::*;
use arbitrary::Arbitrary;
use core::ffi::c_void;
use opus_rs::multistream::{ChannelMappingTable, MultistreamDecoder};
use opusref as c;

#[derive(Arbitrary, Debug)]
struct Input {
    rate: u8,
    channels: u8,
    streams: u8,
    coupled: u8,
    mapping: [u8; 8],
    ops: Vec<Op>,
}

#[derive(Arbitrary, Debug)]
enum Op {
    Packet(Vec<u8>),
    /// A packet from libopus's multistream encoder for the same layout.
    Encoded {
        signal: Gen,
        kbps: u8,
    },
    Lost,
}

struct RefMsDec(*mut c_void);
impl Drop for RefMsDec {
    fn drop(&mut self) {
        unsafe { c::opus_multistream_decoder_destroy(self.0) }
    }
}
struct RefMsEnc(*mut c_void);
impl Drop for RefMsEnc {
    fn drop(&mut self) {
        unsafe { c::opus_multistream_encoder_destroy(self.0) }
    }
}

/// The libFuzzer entry point's body.
pub fn fuzz(data: &[u8]) {
    let Ok(inp) = Input::arbitrary_take_rest(arbitrary::Unstructured::new(data)) else {
        return;
    };
    report("diff_multistream", data, guarded(|| run(inp)));
}

fn run(inp: Input) -> Result<(), Finding> {
    let fs = RATES[inp.rate as usize % 5];
    let channels = 1 + inp.channels % 8;
    let streams = inp.streams % 9;
    let coupled = inp.coupled % 9;
    // Mapping entries below streams+coupled index a decoded channel, 255
    // is silence; other values are invalid and both decoders must say so.
    let mapping: Vec<u8> = inp.mapping[..channels as usize].to_vec();
    let mut err = 0;
    let rd = unsafe {
        c::opus_multistream_decoder_create(
            fs,
            channels as i32,
            streams as i32,
            coupled as i32,
            mapping.as_ptr(),
            &mut err,
        )
    };
    let table = ChannelMappingTable {
        channels,
        stream_count: streams,
        coupled_count: coupled,
        mapping: mapping.clone(),
    };
    let rs = MultistreamDecoder::new(fs, table);
    let layout = format!("ch{channels} s{streams} c{coupled}");
    let (mut rs, rd) = match (rs, rd.is_null()) {
        (Ok(d), false) => (d, RefMsDec(rd)),
        (Err(_), true) => return Ok(()),
        (r, null) => fail!(
            "layout",
            if null { "rs-accepts" } else { "rs-rejects" },
            "{layout} mapping {mapping:?}: opus-rs {}, libopus error {err}",
            if r.is_ok() { "accepts" } else { "rejects" }
        ),
    };
    let enc = {
        let mut e = 0;
        let p = unsafe {
            c::opus_multistream_encoder_create(
                fs,
                channels as i32,
                streams as i32,
                coupled as i32,
                mapping.as_ptr(),
                2049,
                &mut e,
            )
        };
        (!p.is_null()).then_some(RefMsEnc(p))
    };
    let mut synth = Synth::new(fs);
    let frame = fs as usize / 50;
    let max_frame = fs as usize * 120 / 1000;
    let ch = channels as usize;
    let tol = env_f64("OFZ_DEC_TOL", 4e-6) as f32;
    for (k, op) in inp.ops.iter().take(24).enumerate() {
        let pkt: Vec<u8> = match op {
            Op::Packet(p) => p.clone(),
            Op::Lost => Vec::new(),
            Op::Encoded { signal, kbps } => {
                let Some(e) = &enc else { continue };
                let mut pcm = vec![0f32; frame * ch];
                synth.fill(signal, &mut pcm, ch, 0, 1);
                let mut out = vec![0u8; 4000];
                let _ = unsafe {
                    c::opus_multistream_encoder_ctl(
                        e.0,
                        c::SET_BITRATE,
                        (16 + *kbps as i32) * 1000 * streams.max(1) as i32,
                    )
                };
                let n = unsafe {
                    c::opus_multistream_encode_float(
                        e.0,
                        pcm.as_ptr(),
                        frame as i32,
                        out.as_mut_ptr(),
                        out.len() as i32,
                    )
                };
                if n <= 0 {
                    continue;
                }
                out.truncate(n as usize);
                out
            }
        };
        let kind = match op {
            Op::Packet(_) => "packet",
            Op::Lost => "lost",
            Op::Encoded { .. } => "valid",
        };
        let size = if pkt.is_empty() { frame } else { max_frame };
        let mut o_ref = vec![0f32; size * ch];
        let mut o_rs = vec![0f32; size * ch];
        let (ptr, len) = if pkt.is_empty() {
            (core::ptr::null(), 0)
        } else {
            (pkt.as_ptr(), pkt.len() as i32)
        };
        let r_ref = unsafe {
            c::opus_multistream_decode_float(rd.0, ptr, len, o_ref.as_mut_ptr(), size as i32, 0)
        };
        let r_rs = rs.decode(&pkt, size, &mut o_rs);
        let n = match (r_rs, r_ref) {
            (Ok(n), r) if r >= 0 && n == r as usize => n,
            (Err(_), r) if r < 0 => continue,
            (r, c) => fail!(
                "result",
                format!(
                    "{kind} {}",
                    match &r {
                        Ok(_) => "rs-ok",
                        Err(m) => m,
                    }
                ),
                "op {k} {kind}: opus-rs {r:?}, libopus {c} ({} bytes) mapping {mapping:?}",
                pkt.len()
            ),
        };
        let (d, peak, at) = max_diff(&o_rs[..n * ch], &o_ref[..n * ch]);
        if !(d <= tol * peak) {
            fail!(
                "pcm",
                format!("{kind} s{streams} c{coupled}"),
                "op {k} {kind}: |opus-rs - libopus| = {d:e} at sample {at} (opus-rs {}, libopus {}) {layout} mapping {mapping:?}",
                o_rs[at],
                o_ref[at]
            );
        }
    }
    Ok(())
}
