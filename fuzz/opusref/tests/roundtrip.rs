//! The reference build encodes and decodes a tone.
use opusref::*;

#[test]
fn tone_round_trip() {
    unsafe {
        let mut err = 0;
        let enc = opus_encoder_create(48000, 1, 2049, &mut err);
        let dec = opus_decoder_create(48000, 1, &mut err);
        let mut peak = 0f32;
        for f in 0..20 {
            let pcm: Vec<f32> = (0..960)
                .map(|i| (((f * 960 + i) as f32) * 0.05).sin() * 0.5)
                .collect();
            let mut pkt = [0u8; 1500];
            let n = opus_encode_float(enc, pcm.as_ptr(), 960, pkt.as_mut_ptr(), 1500);
            assert!(n > 0);
            let mut out = [0f32; 960];
            let m = opus_decode_float(dec, pkt.as_ptr(), n, out.as_mut_ptr(), 960, 0);
            assert_eq!(m, 960);
            peak = out.iter().fold(peak, |a, x| a.max(x.abs()));
        }
        assert!(peak > 0.3, "peak {peak}");
    }
}
