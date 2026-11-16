//! Soundness fuzzer for the public low-level API: arbitrary call sequences
//! and arguments on `RangeCoder` and the CELT rate functions.
//!
//! Safe functions may panic on bad arguments; those panics are caught and
//! ignored. What remains is undefined behaviour: built with debug assertions
//! (the cargo-fuzz default), the standard library aborts on a violated
//! `get_unchecked`/`unchecked_*` precondition, and AddressSanitizer reports
//! out-of-bounds accesses. Both are libFuzzer crashes.
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use opus_rs::range_coder::RangeCoder;

#[derive(Arbitrary, Debug)]
enum Op {
    EncUint(u32, u32),
    EncBits(u32, u32),
    Encode(u32, u32, u32),
    EncodeBitLogp(bool, u32),
    EncodeIcdf(i32, Vec<u8>, u32),
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
    DecodeIcdf(Vec<u8>, u32),
    LaplaceDecode(u32, i32),
    Tell,
    Bits2Pulses(u8, i32, i32),
    Pulses2Bits(u8, i32, i32),
}

#[derive(Arbitrary, Debug)]
struct Input {
    // Small buffers reach the end-of-buffer handling within a few calls.
    size: u8,
    data: [u8; 32],
    ops: Vec<Op>,
}

fuzz_target!(|inp: Input| {
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
    for op in inp.ops.iter().take(256) {
        if skip_rate && matches!(op, Op::Bits2Pulses(..) | Op::Pulses2Bits(..))
            || skip_icdf && matches!(op, Op::EncodeIcdf(..) | Op::DecodeIcdf(..))
        {
            continue;
        }
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match op {
            Op::EncUint(a, b) => enc.enc_uint(*a, *b),
            Op::EncBits(a, b) => enc.enc_bits(*a, *b),
            // An empty interval (fl >= fh) zeroes the range, and renormalizing
            // never ends; skip it, as libopus's callers never pass one.
            Op::Encode(a, b, c) if a < b => enc.encode(*a, *b, *c),
            Op::Encode(..) => {}
            Op::EncodeBitLogp(a, b) => enc.encode_bit_logp(*a, *b),
            Op::EncodeIcdf(s, t, f) => enc.encode_icdf(*s, t, *f),
            Op::Laplace(v, f, d) => {
                // Outside fs <= 32768 and 0 <= decay < 16384 the loop does not end.
                let mut v = *v % 1024;
                enc.laplace_encode(&mut v, *f % 32769, d.rem_euclid(16384))
            }
            // Unbounded targets loop for minutes; keep them near the buffer size.
            Op::PadToBits(b) => enc.pad_to_bits(*b % 4096),
            Op::PatchInitialBits(v, n) => enc.patch_initial_bits(*v, *n),
            Op::Shrink(n) => enc.shrink(*n),
            Op::Done => enc.done(),
            Op::DecUint(a) => {
                std::hint::black_box(dec.dec_uint(*a));
            }
            Op::DecBits(a) => {
                std::hint::black_box(dec.dec_bits(*a));
            }
            Op::Decode(a) => {
                std::hint::black_box(dec.decode(*a));
            }
            Op::Update(a, b, c) if a < b => dec.update(*a, *b, *c),
            Op::Update(..) => {}
            Op::DecodeBitLogp(a) => {
                std::hint::black_box(dec.decode_bit_logp(*a));
            }
            Op::DecodeIcdf(t, f) => {
                std::hint::black_box(dec.decode_icdf(t, *f));
            }
            Op::LaplaceDecode(f, d) => {
                std::hint::black_box(dec.laplace_decode(*f % 32769, d.rem_euclid(16384)));
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
        }));
        // Safe code may keep using a coder after catching a panic, so the
        // sequence continues with the same state.
        let _ = r;
    }
});
