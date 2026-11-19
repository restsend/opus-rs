//! The frame after a CELT concealment folds the concealed overlap with
//! prefilter_and_fold, whose comb filter libopus runs without a window
//! (window NULL, overlap 0): every sample takes the current postfilter.
//! After a 2.5 ms frame the old and current postfilters differ, so a
//! crossfade there changes the next frame. The tolerance is loose because
//! the concealment amplifies CELT's remaining float rounding differences
//! (up to about 1e-4 here); the bug this guards against is 0.1.
use opus_rs::OpusDecoder;
use opusref::*;

#[test]
fn frame_after_a_loss_folds_without_crossfade() {
    let (fs, frame) = (48000, 120usize);
    let mut err = 0;
    let enc = unsafe { opus_encoder_create(fs, 1, 2049, &mut err) };
    unsafe {
        opus_encoder_ctl(enc, SET_BITRATE, 64000);
        opus_encoder_ctl(enc, 11002, 1002); // OPUS_SET_FORCE_MODE: CELT
    }
    let mut rs = OpusDecoder::new(fs, 1).unwrap();
    let c = unsafe { opus_decoder_create(fs, 1, &mut err) };
    let (mut a, mut b) = (vec![0f32; frame], vec![0f32; frame]);
    for f in 0..60usize {
        let pitch = 150.0 + 40.0 * ((f as f32) * 0.3).sin();
        let pcm: Vec<f32> = (0..frame)
            .map(|i| {
                let t = (f * frame + i) as f32 / fs as f32;
                (1..12)
                    .map(|h| ((h as f32) * pitch * t * 6.2832).sin() / h as f32)
                    .sum::<f32>()
                    * 0.2
            })
            .collect();
        let mut pkt = [0u8; 1500];
        let len =
            unsafe { opus_encode_float(enc, pcm.as_ptr(), frame as i32, pkt.as_mut_ptr(), 1500) };
        let lost = f % 7 == 6;
        let p: &[u8] = if lost { &[] } else { &pkt[..len as usize] };
        rs.decode(p, frame, &mut a).unwrap();
        let ptr = if lost { core::ptr::null() } else { p.as_ptr() };
        unsafe { opus_decode_float(c, ptr, p.len() as i32, b.as_mut_ptr(), frame as i32, 0) };
        let d = a
            .iter()
            .zip(&b)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(d <= 1e-3, "frame {f}: max difference {d}");
    }
    unsafe { opus_encoder_destroy(enc) };
    unsafe { opus_decoder_destroy(c) };
}
