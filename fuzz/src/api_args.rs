//! Soundness fuzzer for the public API: arbitrary call sequences and
//! arguments on `RangeCoder` and the CELT rate functions, and the libopus
//! API calls of `api_calls` (encoder, decoder, multistream, projection,
//! packet and repacketizer functions).
//!
//! The low-level functions may panic on bad arguments; those panics are
//! caught and ignored. The libopus API calls must not panic, and match
//! libopus's error codes where it has the same call (see `api_calls`). What remains is undefined behaviour: built with debug assertions
//! (the cargo-fuzz default), the standard library aborts on a violated
//! `get_unchecked`/`unchecked_*` precondition, and AddressSanitizer reports
//! out-of-bounds accesses. Both are libFuzzer crashes.

use arbitrary::Arbitrary;
use opus_rs::range_coder::RangeCoder;

#[derive(Debug)]
enum Op {
    EncUint(u32, u32),
    EncBits(u32, u32),
    Encode(u32, u32, u32),
    EncodeBitLogp(bool, u32),
    EncodeIcdf(i32, Vec<u8>),
    Laplace(i32, u32, i32),
    PadToBits(i32),
    PatchInitialBits(u32, u32),
    Shrink(u32),
    Done,
    DecUint(u32),
    DecBits(u32),
    Decode(u32),
    Update(u32, u32, u32),
    DecodeBitLogp(u32),
    DecodeIcdf(Vec<u8>),
    LaplaceDecode(u32, i32),
    Tell,
    Bits2Pulses(u8, i32, i32),
    Pulses2Bits(u8, i32, i32),
    /// A libopus API call (`api_calls`).
    Api(crate::api_calls::ApiOp),
}

/// The derived encoding of the first twenty variants, which the saved
/// reproducers use; `Api` takes the top 256th of the variant word, which
/// selected `Pulses2Bits` there.
impl<'a> Arbitrary<'a> for Op {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        use Op::*;
        let x = u32::arbitrary(u)?;
        if x >= 0xff00_0000 {
            return Ok(Api(Arbitrary::arbitrary(u)?));
        }
        let a = |u: &mut arbitrary::Unstructured<'a>| u32::arbitrary(u);
        Ok(match (u64::from(x) * 20) >> 32 {
            0 => EncUint(a(u)?, a(u)?),
            1 => EncBits(a(u)?, a(u)?),
            2 => Encode(a(u)?, a(u)?, a(u)?),
            3 => EncodeBitLogp(Arbitrary::arbitrary(u)?, a(u)?),
            4 => EncodeIcdf(Arbitrary::arbitrary(u)?, Arbitrary::arbitrary(u)?),
            5 => Laplace(Arbitrary::arbitrary(u)?, a(u)?, Arbitrary::arbitrary(u)?),
            6 => PadToBits(Arbitrary::arbitrary(u)?),
            7 => PatchInitialBits(a(u)?, a(u)?),
            8 => Shrink(a(u)?),
            9 => Done,
            10 => DecUint(a(u)?),
            11 => DecBits(a(u)?),
            12 => Decode(a(u)?),
            13 => Update(a(u)?, a(u)?, a(u)?),
            14 => DecodeBitLogp(a(u)?),
            15 => DecodeIcdf(Arbitrary::arbitrary(u)?),
            16 => LaplaceDecode(a(u)?, Arbitrary::arbitrary(u)?),
            17 => Tell,
            18 => Bits2Pulses(Arbitrary::arbitrary(u)?, Arbitrary::arbitrary(u)?, Arbitrary::arbitrary(u)?),
            _ => Pulses2Bits(Arbitrary::arbitrary(u)?, Arbitrary::arbitrary(u)?, Arbitrary::arbitrary(u)?),
        })
    }
}

#[derive(Arbitrary, Debug)]
pub struct Input {
    // Small buffers reach the end-of-buffer handling within a few calls.
    size: u8,
    data: [u8; 32],
    ops: Vec<Op>,
}

/// A valid 8-bit iCDF table (strictly decreasing, ending in 0) from fuzz
/// bytes; a non-decreasing table zeroes the range and renormalizing never
/// ends (n6). The symbol index stays arbitrary.
fn icdf(t: &[u8]) -> Vec<u8> {
    let mut t = t.to_vec();
    t.sort_unstable_by(|a, b| b.cmp(a));
    t.dedup();
    t.retain(|&v| v > 0);
    t.push(0);
    t
}

/// Runs `data` as libfuzzer-sys's `fuzz_target!(|inp: Input| ...)` does: too
/// short or unparsable inputs do nothing.
pub fn fuzz(data: &[u8]) {
    if data.len() < Input::size_hint(0).0 {
        return;
    }
    if let Ok(inp) = Input::arbitrary_take_rest(arbitrary::Unstructured::new(data)) {
        run(inp);
    }
}

/// The libFuzzer entry point's body.
pub fn run(inp: Input) {
    if crate::common::parse_only(&inp) {
        return;
    }
    // Print only the aborting UB-check panics; ordinary panics are expected.
    std::panic::set_hook(Box::new(|info| {
        let msg = info.payload().downcast_ref::<&str>().copied().unwrap_or("");
        if msg.contains("unsafe precondition") {
            eprintln!("UB CHECK at {:?}: {msg}", info.location());
        }
    }));
    let mode = opus_rs::modes::default_mode();
    let mut enc = RangeCoder::new_encoder(inp.size as u32);
    let mut dec = RangeCoder::new_decoder(&inp.data);
    // `OFZ_SUPPRESS=rate,icdf` skips those calls to reach other sites.
    let sup = |n: &str| std::env::var("OFZ_SUPPRESS").is_ok_and(|v| v.split(',').any(|s| s == n));
    let (skip_rate, skip_icdf) = (sup("rate"), sup("icdf"));
    let mut prev_codes = [0i32; 8];
    // The libopus API calls build codecs; a few per input keep it fast.
    let mut api_calls = 0;
    for op in inp.ops.iter().take(256) {
        if let Op::Api(api) = op {
            api_calls += 1;
            if api_calls <= 4 {
                crate::api_calls::run(api, &mut prev_codes);
            }
            continue;
        }
        if skip_rate && matches!(op, Op::Bits2Pulses(..) | Op::Pulses2Bits(..))
            || skip_icdf && matches!(op, Op::EncodeIcdf(..) | Op::DecodeIcdf(..))
        {
            continue;
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match op {
            // Outside libopus's argument domain (fl < fh <= ft <= 2^16 for
            // encode/update/decode, fl < ft and ft >= 2 for the uint calls)
            // the range can reach zero and renormalizing never ends (n6).
            Op::EncUint(a, b) if *b >= 2 && a < b => enc.enc_uint(*a, *b),
            Op::EncUint(..) => {}
            Op::EncBits(a, b) if *b <= 25 => enc.enc_bits(*a, *b),
            Op::EncBits(..) => {}
            // An empty interval (fl >= fh) zeroes the range, and renormalizing
            // never ends; skip it, as libopus's callers never pass one.
            Op::Encode(a, b, c) if a < b && b <= c && *c <= 1 << 16 => enc.encode(*a, *b, *c),
            Op::Encode(..) => {}
            // logp 0 or above 23 can zero the range, and renormalizing never ends;
            // libopus's callers pass at most 15.
            Op::EncodeBitLogp(a, b) if (1..=23).contains(b) => enc.encode_bit_logp(*a, *b),
            Op::EncodeBitLogp(..) => {}
            Op::EncodeIcdf(s, t) => enc.encode_icdf(*s, &icdf(t), 8),
            Op::Laplace(v, f, d) => {
                let mut v = *v % 1024;
                enc.laplace_encode(&mut v, *f, *d)
            }
            // Unbounded targets loop for minutes; keep them near the buffer size.
            Op::PadToBits(b) => enc.pad_to_bits(*b % 4096),
            Op::PatchInitialBits(v, n) => enc.patch_initial_bits(*v, *n),
            Op::Shrink(n) => enc.shrink(*n),
            // Coding after done() leaves the range at zero (n6); start over.
            Op::Done => {
                enc.done();
                enc = RangeCoder::new_encoder(inp.size as u32);
            }
            Op::DecUint(a) if *a < 2 => {}
            Op::DecUint(a) => {
                std::hint::black_box(dec.dec_uint(*a));
            }
            Op::DecBits(a) if *a > 25 => {}
            Op::DecBits(a) => {
                std::hint::black_box(dec.dec_bits(*a));
            }
            Op::Decode(a) if *a == 0 || *a > 1 << 16 => {}
            Op::Decode(a) => {
                std::hint::black_box(dec.decode(*a));
            }
            Op::Update(a, b, c) if a < b && b <= c && *c <= 1 << 16 => dec.update(*a, *b, *c),
            Op::Update(..) => {}
            Op::DecodeBitLogp(a) => {
                std::hint::black_box(dec.decode_bit_logp(*a));
            }
            Op::DecodeIcdf(t) => {
                std::hint::black_box(dec.decode_icdf(&icdf(t), 8));
            }
            Op::LaplaceDecode(f, d) => {
                std::hint::black_box(dec.laplace_decode(*f, *d));
            }
            Op::Tell => {
                std::hint::black_box((enc.tell_frac(), dec.tell_frac()));
            }
            Op::Bits2Pulses(b, lm, bits) => {
                std::hint::black_box(opus_rs::rate::bits2pulses(mode, *b as usize, *lm, *bits));
            }
            Op::Pulses2Bits(b, lm, p) => {
                std::hint::black_box(opus_rs::rate::pulses2bits(mode, *b as usize, *lm, *p));
            }
            Op::Api(_) => {}
        }));
        // Safe code may keep using a coder after catching a panic, so the
        // sequence continues with the same state.
        let _ = r;
    }
}
