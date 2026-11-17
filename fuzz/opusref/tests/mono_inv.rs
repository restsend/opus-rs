//! A mono decoder playing a stereo CELT stream: libopus's CELT decoder sets
//! disable_inv for one output channel (celt_decoder_init), so intensity
//! bands ignore the coded inversion before the downmix. opus-rs must too.
use opus_rs::OpusDecoder;
use opusref::*;

#[test]
fn mono_decoder_ignores_intensity_inversion() {
    let (fs, frame) = (48000, 960usize);
    let mut err = 0;
    let enc = unsafe { opus_encoder_create(fs, 2, 2049, &mut err) };
    unsafe {
        opus_encoder_ctl(enc, SET_BITRATE, 32000);
        opus_encoder_ctl(enc, 11002, 1002); // OPUS_SET_FORCE_MODE: CELT
    }
    let mut rs = OpusDecoder::new(fs, 1).unwrap();
    let c = unsafe { opus_decoder_create(fs, 1, &mut err) };
    let (mut a, mut b) = (vec![0f32; frame], vec![0f32; frame]);
    let mut seed = 1u32;
    for f in 0..20 {
        // Right is the inverted left: intensity stereo with inversion.
        let mut pcm = vec![0f32; 2 * frame];
        for i in 0..frame {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let n = ((seed >> 9) as f32 / 8388608.0 - 0.5) * 0.3;
            let v = n + (((f * frame + i) as f32) * 0.02).sin() * 0.2;
            pcm[2 * i] = v;
            pcm[2 * i + 1] = -0.8 * v;
        }
        let mut pkt = [0u8; 1500];
        let len = unsafe { opus_encode_float(enc, pcm.as_ptr(), frame as i32, pkt.as_mut_ptr(), 1500) };
        let p = &pkt[..len as usize];
        rs.decode(p, frame, &mut a).unwrap();
        unsafe { opus_decode_float(c, p.as_ptr(), len, b.as_mut_ptr(), frame as i32, 0) };
        let d = a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(d <= 4e-6, "frame {f}: max difference {d}");
    }
    unsafe { opus_encoder_destroy(enc) };
    unsafe { opus_decoder_destroy(c) };
}
