//! Differential multistream encoder fuzzer: random layouts (1-8 channels,
//! streams, coupled streams, mapping), generated signals, frame durations,
//! output budgets, the float, 16-bit and 24-bit entry points and the
//! multistream controls, encoded by opus-rs's MultistreamEncoder and
//! libopus's opus_multistream_encoder.
//!
//! Oracles:
//! - `layout`: both accept or reject the layout, with the same error code.
//! - `result`: both accept or reject each call and control, with the same
//!   error code.
//! - `decode`: libopus's multistream decoder decodes every opus-rs packet
//!   to its duration, with the encoder's final range.
//! - `toc`: the first stream's ToC agrees, and every stream's where
//!   `strict_bytes` holds (suppress with `OFZ_SUPPRESS=toc`).
//! - `get`: the same bitrate query after each call.
//! - `bytes`: byte-identical packets where `common::strict_bytes` says the
//!   encoders run the same algorithm (`OFZ_STRICT_BYTES=1`).

use crate::common::*;
use arbitrary::Arbitrary;
use core::ffi::{c_int, c_void};
use opus_rs::multistream_encoder::MultistreamEncoder;
use opus_rs::{Application, Bandwidth, FrameDuration, Signal, error};
use opusref as c;

#[derive(Arbitrary, Debug)]
struct Input {
    rate: u8,
    channels: u8,
    streams: u8,
    coupled: u8,
    mapping: [u8; 8],
    app: u8,
    /// `opus_multistream_surround_encoder_create` with this mapping family
    /// (an index into `FAMILIES`) in place of the given layout.
    surround: Option<u8>,
    segs: Vec<Seg>,
}

const FAMILIES: [i32; 6] = [0, 1, 2, 255, 1, 3];

#[derive(Arbitrary, Debug)]
struct Seg {
    frames: u8,
    signal: Gen,
    /// 2.5 to 120 ms in units of 2.5 ms, an index into `DURATIONS`.
    duration: u8,
    /// The output buffer in bytes (`None`: 4000).
    budget: Option<u16>,
    /// 0 float, 1 16-bit, 2 24-bit.
    entry: u8,
    ctl: Option<Ctl>,
}

#[derive(Arbitrary, Debug, Clone, Copy)]
enum Ctl {
    Bitrate(i32),
    Vbr(bool),
    VbrConstraint(bool),
    Complexity(i8),
    PacketLoss(i8),
    InbandFec(i8),
    Dtx(bool),
    Bandwidth(u8),
    MaxBandwidth(u8),
    Signal(u8),
    LsbDepth(i8),
    ForceChannels(i8),
    PredictionDisabled(bool),
    PhaseInversionDisabled(bool),
    FrameDuration(u8),
    Reset,
}

const DURATIONS: [usize; 9] = [1, 2, 4, 8, 16, 24, 32, 40, 48];
const APPS: [Application; 5] = [
    Application::Voip,
    Application::Audio,
    Application::RestrictedLowDelay,
    Application::RestrictedSilk,
    Application::RestrictedCelt,
];
const BW: [Bandwidth; 6] = [
    Bandwidth::Auto,
    Bandwidth::Narrowband,
    Bandwidth::Mediumband,
    Bandwidth::Wideband,
    Bandwidth::Superwideband,
    Bandwidth::Fullband,
];
const FRAME_DURATIONS: [FrameDuration; 10] = [
    FrameDuration::Argument,
    FrameDuration::Ms2_5,
    FrameDuration::Ms5,
    FrameDuration::Ms10,
    FrameDuration::Ms20,
    FrameDuration::Ms40,
    FrameDuration::Ms60,
    FrameDuration::Ms80,
    FrameDuration::Ms100,
    FrameDuration::Ms120,
];

struct RefMsEnc(*mut c_void);
impl Drop for RefMsEnc {
    fn drop(&mut self) {
        unsafe { c::opus_multistream_encoder_destroy(self.0) }
    }
}
struct RefMsDec(*mut c_void);
impl Drop for RefMsDec {
    fn drop(&mut self) {
        unsafe { c::opus_multistream_decoder_destroy(self.0) }
    }
}

/// The libFuzzer entry point's body.
pub fn fuzz(data: &[u8]) {
    let Ok(inp) = Input::arbitrary_take_rest(arbitrary::Unstructured::new(data)) else {
        return;
    };
    report("diff_ms_encoder", data, guarded(|| run(inp)));
}

fn code(r: Result<(), &str>) -> i32 {
    r.map_or_else(|e| error::code(e).unwrap_or(1), |()| 0)
}

/// Applies `ctl` to both encoders; their results must agree.
fn apply(rs: &mut MultistreamEncoder, re: *mut c_void, ctl: Ctl) -> Result<(), Finding> {
    let set = |req: c_int, v: i32| unsafe { c::opus_multistream_encoder_ctl(re, req, v) };
    let (a, b) = match ctl {
        Ctl::Bitrate(v) => (code(rs.set_bitrate(v)), set(c::SET_BITRATE, v)),
        Ctl::Vbr(v) => (code(Ok(rs.set_vbr(v))), set(c::SET_VBR, v as i32)),
        Ctl::VbrConstraint(v) => (code(Ok(rs.set_vbr_constraint(v))), set(c::SET_VBR_CONSTRAINT, v as i32)),
        Ctl::Complexity(v) => (code(rs.set_complexity(v as i32)), set(c::SET_COMPLEXITY, v as i32)),
        Ctl::PacketLoss(v) => (code(rs.set_packet_loss_perc(v as i32)), set(c::SET_PACKET_LOSS_PERC, v as i32)),
        Ctl::InbandFec(v) => (code(rs.set_inband_fec(v as i32)), set(c::SET_INBAND_FEC, v as i32)),
        Ctl::Dtx(v) => (code(Ok(rs.set_dtx(v))), set(c::SET_DTX, v as i32)),
        Ctl::Bandwidth(i) => {
            let bw = BW[i as usize % 6];
            (code(Ok(rs.set_bandwidth(bw))), set(c::SET_BANDWIDTH, bw as i32))
        }
        Ctl::MaxBandwidth(i) => {
            let bw = BW[1 + i as usize % 5];
            (code(Ok(rs.set_max_bandwidth(bw))), set(c::SET_MAX_BANDWIDTH, bw as i32))
        }
        Ctl::Signal(i) => {
            let s = [Signal::Auto, Signal::Voice, Signal::Music][i as usize % 3];
            (code(Ok(rs.set_signal(s))), set(c::SET_SIGNAL, s as i32))
        }
        Ctl::LsbDepth(v) => (code(rs.set_lsb_depth(v as i32)), set(c::SET_LSB_DEPTH, v as i32)),
        Ctl::ForceChannels(v) => {
            let f = (v >= 0).then_some(v as usize);
            (code(rs.set_force_channels(f)), set(c::SET_FORCE_CHANNELS, f.map_or(-1000, |f| f as i32)))
        }
        Ctl::PredictionDisabled(v) => {
            (code(Ok(rs.set_prediction_disabled(v))), set(c::SET_PREDICTION_DISABLED, v as i32))
        }
        Ctl::PhaseInversionDisabled(v) => (
            code(Ok(rs.set_phase_inversion_disabled(v))),
            set(c::SET_PHASE_INVERSION_DISABLED, v as i32),
        ),
        Ctl::FrameDuration(i) => {
            let d = FRAME_DURATIONS[i as usize % 10];
            (code(Ok(rs.set_expert_frame_duration(d))), set(c::SET_EXPERT_FRAME_DURATION, d as i32))
        }
        Ctl::Reset => {
            rs.reset();
            (0, unsafe { c::opus_multistream_encoder_ctl(re, c::RESET_STATE) })
        }
    };
    if a != b {
        fail!("result", format!("ctl {ctl:?}"), "opus-rs {a}, libopus {b}");
    }
    Ok(())
}

/// The ToC configuration of each stream, or `None` if a stream does not
/// parse.
fn tocs(p: &[u8], streams: usize) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    let mut pos = 0;
    for s in 0..streams {
        let parts = opus_rs::extensions::parse_packet_impl(p.get(pos..)?, s + 1 < streams).ok()?;
        out.push(parts.toc & 0xfc);
        pos += parts.packet_offset;
    }
    Some(out)
}

fn run(inp: Input) -> Result<(), Finding> {
    let fs = RATES[inp.rate as usize % 5];
    let channels = 1 + inp.channels as usize % 8;
    let mut streams = inp.streams as usize % 9;
    let mut coupled = inp.coupled as usize % 9;
    let mut mapping = inp.mapping[..channels].to_vec();
    let app = APPS[inp.app as usize % APPS.len()];
    let mut err = 0;
    let (re, rs) = match inp.surround {
        None => (
            unsafe {
                c::opus_multistream_encoder_create(
                    fs,
                    channels as c_int,
                    streams as c_int,
                    coupled as c_int,
                    mapping.as_ptr(),
                    app as c_int,
                    &mut err,
                )
            },
            MultistreamEncoder::new(fs, channels, streams, coupled, &mapping, app),
        ),
        Some(f) => {
            let family = FAMILIES[f as usize % FAMILIES.len()];
            let (mut s, mut cp) = (0, 0);
            let mut m = [0u8; 255];
            let re = unsafe {
                c::opus_multistream_surround_encoder_create(
                    fs,
                    channels as c_int,
                    family,
                    &mut s,
                    &mut cp,
                    m.as_mut_ptr(),
                    app as c_int,
                    &mut err,
                )
            };
            let rs = MultistreamEncoder::new_surround(fs, channels, family, app);
            if let Ok(e) = &rs {
                if (e.streams() as i32, e.coupled_streams() as i32, e.mapping()) != (s, cp, &m[..channels]) {
                    fail!("layout", "surround", "fs{fs} ch{channels} family {family}: streams, coupled or mapping differ");
                }
                (streams, coupled, mapping) = (s as usize, cp as usize, m[..channels].to_vec());
            }
            ijon_set(4 << 32 | (family as u64 & 0xff) << 8 | channels as u64);
            (re, rs)
        }
    };
    let mapping = &mapping[..];
    let layout = format!("fs{fs} ch{channels} s{streams} c{coupled} {app:?} {:?}", inp.surround);
    let mut rs = match (rs, re.is_null()) {
        (Ok(e), false) => e,
        (Err(e), true) if error::code(e) == Some(err) => return Ok(()),
        (r, _) => fail!(
            "layout",
            if r.is_ok() { "rs-accepts" } else { "rs-rejects" },
            "{layout} mapping {mapping:?}: opus-rs {:?}, libopus {err}",
            r.err()
        ),
    };
    let re = RefMsEnc(re);
    let rd = RefMsDec(unsafe {
        c::opus_multistream_decoder_create(fs, channels as c_int, streams as c_int, coupled as c_int, mapping.as_ptr(), &mut err)
    });
    if rd.0.is_null() {
        fail!("layout", "decoder", "{layout}: libopus's decoder rejects the encoder's layout");
    }
    // IJON: the layout's shape.
    ijon_set(1 << 32 | (streams as u64) << 8 | (coupled as u64) << 4 | channels as u64);
    let mut synth = Synth::new(fs);
    let mut prev_tocs = Vec::new();
    let mut total = 0;
    for seg in inp.segs.iter().take(10) {
        if let Some(ctl) = seg.ctl {
            apply(&mut rs, re.0, ctl)?;
        }
        let frame = fs as usize / 400 * DURATIONS[seg.duration as usize % 9];
        let budget = seg.budget.map_or(4000, |b| b as usize % 6000);
        let entry = seg.entry % 3;
        for i in 0..1 + seg.frames as usize % 4 {
            total += 1;
            if total > 24 {
                return Ok(());
            }
            // One generated channel, the others scaled and delayed copies.
            let mut mono = vec![0f32; frame];
            synth.fill(&seg.signal, &mut mono, 1, i, 4);
            let pcm: Vec<f32> = (0..frame * channels)
                .map(|k| {
                    let (t, c) = (k / channels, k % channels);
                    mono[t.saturating_sub(3 * c)] * (1.0 - 0.15 * c as f32)
                })
                .collect();
            let (mut a, mut b) = (vec![0u8; budget], vec![0u8; budget]);
            let (r_rs, r_c) = match entry {
                0 => (rs.encode(&pcm, frame, &mut a), unsafe {
                    c::opus_multistream_encode_float(re.0, pcm.as_ptr(), frame as c_int, b.as_mut_ptr(), budget as i32)
                }),
                1 => {
                    let x: Vec<i16> = pcm.iter().map(|&v| (v * 32768.0).clamp(-32768.0, 32767.0) as i16).collect();
                    (rs.encode_i16(&x, frame, &mut a), unsafe {
                        c::opus_multistream_encode(re.0, x.as_ptr(), frame as c_int, b.as_mut_ptr(), budget as i32)
                    })
                }
                _ => {
                    let x: Vec<i32> = pcm.iter().map(|&v| (v * 8_388_608.0) as i32).collect();
                    (rs.encode24(&x, frame, &mut a), unsafe {
                        c::opus_multistream_encode24(re.0, x.as_ptr(), frame as c_int, b.as_mut_ptr(), budget as i32)
                    })
                }
            };
            let ctx = format!("{layout} call {total} {} ms budget {budget} entry {entry}", frame as f32 * 1000.0 / fs as f32);
            let n = match r_rs {
                Ok(n) if r_c > 0 => n,
                Err(e) if r_c <= 0 && error::code(e) == Some(r_c) => continue,
                r => fail!(
                    "result",
                    if r.is_ok() { "rs-ok" } else { "rs-err" },
                    "{ctx}: opus-rs {r:?}, libopus {r_c}"
                ),
            };
            let (pa, pb) = (&a[..n], &b[..r_c as usize]);
            let (ta, tb) = (tocs(pa, streams), tocs(pb, streams));
            let Some(ta) = ta else {
                fail!("decode", "parse", "{ctx}: an opus-rs stream does not parse: {pa:02x?}");
            };
            // IJON: (previous ToCs, ToCs) per stream configuration.
            ijon_set(2 << 32 | (ta.iter().fold(0u64, |h, &t| h * 31 + t as u64) ^ prev_tocs.iter().fold(7u64, |h, &t| h * 37 + t as u64)) & 0xffff_ffff);
            prev_tocs = ta.clone();
            // The later streams get what the earlier ones left of the
            // budget, which follows their sizes: they agree where the bytes
            // do. The first stream's ToC always agrees.
            let same = match &tb {
                Some(tb) if strict_bytes(app) => &ta == tb,
                Some(tb) => ta[0] == tb[0],
                None => false,
            };
            if !same && !suppressed("toc") {
                fail!("toc", format!("s{streams}"), "{ctx}: opus-rs {ta:02x?}, libopus {tb:02x?}");
            }
            if strict_bytes(app) && pa != pb {
                fail!("bytes", format!("s{streams} c{coupled}"), "{ctx}: opus-rs {} bytes, libopus {}", pa.len(), pb.len());
            }
            let mut out = vec![0f32; 5760 * channels];
            let d = unsafe {
                c::opus_multistream_decode_float(rd.0, pa.as_ptr(), pa.len() as i32, out.as_mut_ptr(), 5760, 0)
            };
            let coded = opus_rs::packet::nb_samples(pa, fs).unwrap_or(0) as i32;
            if d != coded || d <= 0 {
                fail!("decode", "samples", "{ctx}: libopus decodes {d} samples of {coded}");
            }
            let mut r = 0u32;
            unsafe { c::opus_multistream_decoder_ctl(rd.0, c::GET_FINAL_RANGE, &mut r as *mut u32) };
            if r != rs.final_range() {
                fail!("decode", "range", "{ctx}: encoder range {:#x}, decoder {r:#x}", rs.final_range());
            }
            let mut br = 0i32;
            unsafe { c::opus_multistream_encoder_ctl(re.0, c::GET_BITRATE, &mut br as *mut i32) };
            // The query reads each stream's rate and last frame duration,
            // which follow the streams' modes and, under CBR, the sizes of
            // the streams before; those agree where the bytes do.
            if rs.bitrate() != br && ((rs.first_stream().vbr() && Some(&ta) == tb.as_ref()) || strict_bytes(app)) {
                fail!("get", "bitrate", "{ctx}: opus-rs {}, libopus {br}", rs.bitrate());
            }
            ijon_max(3 << 8 | streams as u64, n as u64);
        }
    }
    Ok(())
}
