//! The C API called from Rust: round trips through the exported functions
//! and the flat state layout (a state copied byte for byte keeps working).
//! The ctl calls are C-variadic and run outside Miri; the rest runs under
//! `cargo miri test`.

use core::ffi::c_int;
use core::ptr;

use crate::decoder::*;
use crate::encoder::*;
use crate::multistream::*;
use crate::packet::*;
use crate::repacketizer::*;
use crate::{OPUS_OK, alloc_state, free_state};

const VOIP: c_int = 2048;

/// 20 ms of a 440 Hz tone at 8 kHz, `channels` interleaved.
fn tone(channels: usize) -> Vec<f32> {
    (0..160 * channels).map(|i| 0.5 * (i as f32 / channels as f32 * 440.0 / 8000.0 * 6.283).sin()).collect()
}

fn encode_frames(n: usize) -> Vec<Vec<u8>> {
    let mut err = 0;
    let enc = unsafe { opus_encoder_create(8000, 1, VOIP, &mut err) };
    assert_eq!(err, OPUS_OK);
    let pcm = tone(1);
    let packets = (0..n)
        .map(|_| {
            let mut out = vec![0u8; 400];
            let len = unsafe { opus_encode_float(enc, pcm.as_ptr(), 160, out.as_mut_ptr(), 400) };
            assert!(len > 0, "{len}");
            out.truncate(len as usize);
            out
        })
        .collect();
    unsafe { opus_encoder_destroy(enc) };
    packets
}

#[test]
fn encode_decode() {
    let packets = encode_frames(2);
    let mut err = 0;
    let dec = unsafe { opus_decoder_create(8000, 1, &mut err) };
    assert_eq!(err, OPUS_OK);
    for p in &packets {
        assert_eq!(unsafe { opus_decoder_get_nb_samples(dec, p.as_ptr(), p.len() as i32) }, 160);
        let mut pcm = [0i16; 160];
        assert_eq!(unsafe { opus_decode(dec, p.as_ptr(), p.len() as i32, pcm.as_mut_ptr(), 160, 0) }, 160);
    }
    // A loss.
    let mut pcm = [0.0f32; 160];
    assert_eq!(unsafe { opus_decode_float(dec, ptr::null(), 0, pcm.as_mut_ptr(), 160, 0) }, 160);
    unsafe { opus_decoder_destroy(dec) };
}

/// A decoder copied byte for byte into another block decodes as the
/// original does, and the original block is freed without a destroy call.
#[test]
fn decoder_copies() {
    let packets = encode_frames(2);
    let size = opus_decoder_get_size(1) as usize;
    unsafe {
        let a = alloc_state(size).cast::<OpusDecoder>();
        assert_eq!(opus_decoder_init(a, 8000, 1), OPUS_OK);
        let mut out = [[0.0f32; 160]; 2];
        opus_decode_float(a, packets[0].as_ptr(), packets[0].len() as i32, out[0].as_mut_ptr(), 160, 0);
        let b = alloc_state(size).cast::<OpusDecoder>();
        ptr::copy_nonoverlapping(a.cast::<u8>(), b.cast::<u8>(), size);
        for (st, out) in [a, b].into_iter().zip(&mut out) {
            assert_eq!(opus_decode_float(st, packets[1].as_ptr(), packets[1].len() as i32, out.as_mut_ptr(), 160, 0), 160);
            free_state(st);
        }
        assert_eq!(out[0], out[1]);
    }
}

/// A two-stream encoder and decoder in caller blocks, the decoder copied.
#[test]
fn multistream_copies() {
    let mapping = [0u8, 1, 2];
    unsafe {
        let enc = alloc_state(opus_multistream_encoder_get_size(2, 1) as usize).cast::<OpusMSEncoder>();
        assert_eq!(opus_multistream_encoder_init(enc, 8000, 3, 2, 1, mapping.as_ptr(), VOIP), OPUS_OK);
        let pcm = tone(3);
        let mut packet = [0u8; 1000];
        let len = opus_multistream_encode_float(enc, pcm.as_ptr(), 160, packet.as_mut_ptr(), 1000);
        assert!(len > 0, "{len}");
        free_state(enc);

        let size = opus_multistream_decoder_get_size(2, 1) as usize;
        let a = alloc_state(size).cast::<OpusMSDecoder>();
        assert_eq!(opus_multistream_decoder_init(a, 8000, 3, 2, 1, mapping.as_ptr()), OPUS_OK);
        let b = alloc_state(size).cast::<OpusMSDecoder>();
        ptr::copy_nonoverlapping(a.cast::<u8>(), b.cast::<u8>(), size);
        let mut out = [[0.0f32; 480]; 2];
        for (st, out) in [a, b].into_iter().zip(&mut out) {
            assert_eq!(opus_multistream_decode_float(st, packet.as_ptr(), len, out.as_mut_ptr(), 160, 0), 160);
            free_state(st);
        }
        assert_eq!(out[0], out[1]);
    }
}

#[test]
fn repacketize_and_pad() {
    let packets = encode_frames(2);
    unsafe {
        let rp = opus_repacketizer_create();
        for p in &packets {
            assert_eq!(opus_repacketizer_cat(rp, p.as_ptr(), p.len() as i32), OPUS_OK);
        }
        assert_eq!(opus_repacketizer_get_nb_frames(rp), 2);
        let mut out = [0u8; 1000];
        let len = opus_repacketizer_out(rp, out.as_mut_ptr(), 1000);
        assert!(len > 0);
        opus_repacketizer_destroy(rp);

        let (mut toc, mut frames, mut sizes, mut offset) = (0u8, [ptr::null(); 48], [0i16; 48], 0);
        let n = opus_packet_parse(out.as_ptr(), len, &mut toc, frames.as_mut_ptr(), sizes.as_mut_ptr(), &mut offset);
        assert_eq!(n, 2);
        assert_eq!(sizes[..2].iter().map(|&s| s as usize).collect::<Vec<_>>(), packets.iter().map(|p| p.len() - 1).collect::<Vec<_>>());

        assert_eq!(opus_packet_pad(out.as_mut_ptr(), len, len + 20), OPUS_OK);
        assert_eq!(opus_packet_unpad(out.as_mut_ptr(), len + 20), len);
    }
}

#[test]
#[cfg_attr(miri, ignore = "C-variadic calls")]
fn ctls() {
    let mut err = 0;
    unsafe {
        let enc = opus_encoder_create(8000, 1, VOIP, &mut err);
        assert_eq!(opus_encoder_ctl(enc, 4010, 3), OPUS_OK);
        let mut v = 0i32;
        assert_eq!(opus_encoder_ctl(enc, 4011, &mut v), OPUS_OK);
        assert_eq!(v, 3);
        assert_eq!(opus_encoder_ctl(enc, 4010, 11), crate::OPUS_BAD_ARG);
        opus_encoder_destroy(enc);
    }
}
