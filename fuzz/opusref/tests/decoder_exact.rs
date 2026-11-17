//! The float decoder's output is bit-identical with libopus's float build
//! for libopus-encoded streams. Each test decodes a stream whose frames
//! reach one rounding rule of the float build that the differential
//! fuzzer's tolerance does not see.
use opus_rs::OpusDecoder;
use opusref::*;

/// Encodes 100 frames of a voiced signal (silent for frames 40-59) with
/// libopus in `force` mode (`OPUS_SET_FORCE_MODE`), drops every `loss`-th
/// packet (0: none), decodes both streams and asserts identical samples.
fn assert_exact(fs: i32, ch: usize, frame: usize, kbps: i32, force: i32, loss: usize) {
    let mut err = 0;
    let enc = unsafe { opus_encoder_create(fs, ch as i32, 2049, &mut err) };
    assert!(!enc.is_null() && err == 0);
    unsafe {
        opus_encoder_ctl(enc, SET_BITRATE, kbps * 1000);
        opus_encoder_ctl(enc, 11002, force);
    }
    let mut rs = OpusDecoder::new(fs, ch).unwrap();
    let c = unsafe { opus_decoder_create(fs, ch as i32, &mut err) };
    let (mut a, mut b) = (vec![0f32; frame * ch], vec![0f32; frame * ch]);
    for f in 0..100usize {
        let pitch = 150.0 + 40.0 * ((f as f32) * 0.3).sin();
        let pcm: Vec<f32> = (0..frame * ch)
            .map(|i| {
                if (40..60).contains(&f) {
                    return 0.0;
                }
                let t = (f * frame + i / ch) as f32 / fs as f32;
                (1..12)
                    .map(|h| ((h as f32) * pitch * t * 6.2832).sin() / h as f32)
                    .sum::<f32>()
                    * 0.2
            })
            .collect();
        let mut pkt = [0u8; 1500];
        let len =
            unsafe { opus_encode_float(enc, pcm.as_ptr(), frame as i32, pkt.as_mut_ptr(), 1500) };
        assert!(len > 0);
        let lost = loss > 0 && f % loss == loss - 1;
        let p: &[u8] = if lost { &[] } else { &pkt[..len as usize] };
        let n = rs.decode(p, frame, &mut a).unwrap();
        let ptr = if lost { core::ptr::null() } else { p.as_ptr() };
        let m =
            unsafe { opus_decode_float(c, ptr, p.len() as i32, b.as_mut_ptr(), frame as i32, 0) };
        assert_eq!(n as i32, m, "frame {f}");
        if let Some(i) = (0..a.len()).find(|&i| a[i].to_bits() != b[i].to_bits()) {
            panic!(
                "frame {f} sample {i}: opus-rs {:e}, libopus {:e}",
                a[i], b[i]
            );
        }
    }
    unsafe {
        opus_encoder_destroy(enc);
        opus_decoder_destroy(c);
    }
}

const CELT: i32 = 1002;

/// denormalise_bands and anti_collapse take 2^x as the float build's
/// celt_exp2, exp() in double (mathops.h:347-350).
#[test]
fn band_gains_use_the_float_celt_exp2() {
    assert_exact(48000, 1, 960, 64, CELT, 0);
    assert_exact(48000, 2, 120, 128, CELT, 0);
    assert_exact(24000, 1, 480, 48, CELT, 0);
}
