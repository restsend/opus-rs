//! Stack-usage and struct-size coverage for the `heap` feature (issue #12).
//!
//! Since 0.1.27 the codec state lives inline in `OpusEncoder`/`OpusDecoder`
//! (~250 KB / ~175 KB without `heap`), so constructing them required ~850 KiB of
//! stack (Windows main threads default to 1 MiB). With the `heap` feature the
//! state is `Box`-allocated, the structs shrink to ~1 KB, and construction plus
//! a full encode/decode round-trip must fit comfortably on a 256 KiB stack.

use opus_rs::{Application, OpusDecoder, OpusEncoder};

/// `size_of` returns bytes.
#[cfg(feature = "heap")]
#[test]
fn encoder_and_decoder_structs_are_small_with_heap() {
    assert!(
        std::mem::size_of::<OpusEncoder>() < 4 * 1024,
        "OpusEncoder is {} bytes with heap; expected < 4 KiB",
        std::mem::size_of::<OpusEncoder>()
    );
    assert!(
        std::mem::size_of::<OpusDecoder>() < 4 * 1024,
        "OpusDecoder is {} bytes with heap; expected < 4 KiB",
        std::mem::size_of::<OpusDecoder>()
    );
}

/// Without `heap` the state is inline, so the structs must be large — this
/// keeps the tradeoff visible and guards against accidentally making the
/// heap-free build tiny (which would break its `static`-placement model).
#[cfg(not(feature = "heap"))]
#[test]
fn structs_are_large_without_heap() {
    assert!(
        std::mem::size_of::<OpusEncoder>() > 100_000,
        "OpusEncoder is {} bytes without heap; expected > 100 KiB",
        std::mem::size_of::<OpusEncoder>()
    );
    assert!(
        std::mem::size_of::<OpusDecoder>() > 100_000,
        "OpusDecoder is {} bytes without heap; expected > 100 KiB",
        std::mem::size_of::<OpusDecoder>()
    );
}

#[test]
fn report_struct_sizes() {
    println!(
        "size_of::<OpusEncoder>() = {} B",
        std::mem::size_of::<OpusEncoder>()
    );
    println!(
        "size_of::<OpusDecoder>() = {} B",
        std::mem::size_of::<OpusDecoder>()
    );
}

/// Full encode + decode round-trip on a small-stack thread.
#[cfg(feature = "heap")]
fn roundtrip_on_stack(stack_size: usize) -> bool {
    std::thread::Builder::new()
        .stack_size(stack_size)
        .spawn(|| {
            let mut enc = OpusEncoder::new(48000, 2, Application::Audio).unwrap();
            enc.bitrate_bps = 128_000;

            let frame_size = 960; // 20 ms @ 48 kHz
            let mut input = vec![0.0f32; frame_size * 2];
            for (i, x) in input.iter_mut().enumerate() {
                let t = i as f32 / 48000.0;
                *x = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
            }
            let mut packet = vec![0u8; 2048];
            let n = enc.encode(&input, frame_size, &mut packet).unwrap();

            let mut dec = OpusDecoder::new(48000, 2).unwrap();
            let mut pcm = vec![0.0f32; frame_size * 2];
            let samples = dec.decode(&packet[..n], frame_size, &mut pcm).unwrap();
            samples > 0
        })
        .unwrap()
        .join()
        .unwrap()
}

/// Constructing, encoding and decoding must all fit on a 768 KiB stack.
///
/// Without `heap` this overflows even at 2 MiB in debug builds (the 0.1.26-era
/// inline-state design needed ~850 KiB in release / ~2 MiB in debug). 768 KiB is
/// comfortably below a Windows main-thread's default 1 MiB stack.
#[cfg(feature = "heap")]
#[test]
fn construction_and_roundtrip_fit_in_768kb_stack() {
    assert!(
        roundtrip_on_stack(768 * 1024),
        "constructing + encoding + decoding overflowed a 768 KiB stack"
    );
}
