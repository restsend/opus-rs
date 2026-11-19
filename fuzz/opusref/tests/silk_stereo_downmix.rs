//! Stereo input coded as mono SILK: silk_Encode sums the two channels rounded
//! to 16 bits each and halves the sum (enc_API.c), so a loud in-phase signal
//! keeps its level. Both encoders' packets decode to the same peak.
use opus_rs::{Application, Bandwidth, OpusEncoder};
use opusref::*;

#[test]
fn loud_stereo_coded_mono_keeps_its_level() {
    let (fs, n) = (48000, 960usize);
    let mut rs = OpusEncoder::new(fs, 2, Application::Audio).unwrap();
    rs.bitrate_bps = 12000;
    rs.set_bandwidth(Bandwidth::Wideband);
    let mut err = 0;
    let c = unsafe { opus_encoder_create(fs, 2, 2049, &mut err) };
    unsafe {
        opus_encoder_ctl(c, SET_BITRATE, 12000);
        opus_encoder_ctl(c, SET_BANDWIDTH, Bandwidth::Wideband as i32);
    }
    let (d_rs, d_c) = unsafe {
        (
            opus_decoder_create(fs, 1, &mut err),
            opus_decoder_create(fs, 1, &mut err),
        )
    };
    let (mut peak_rs, mut peak_c) = (0i16, 0i16);
    let mut out = vec![0i16; n];
    for f in 0..50 {
        let pcm: Vec<f32> = (0..n)
            .flat_map(|i| {
                let v = 0.9 * ((f * n + i) as f32 * 0.03).sin();
                [v, v]
            })
            .collect();
        let (mut a, mut b) = ([0u8; 1500], [0u8; 1500]);
        let la = rs.encode(&pcm, n, &mut a).unwrap();
        let lb = unsafe { opus_encode_float(c, pcm.as_ptr(), n as i32, b.as_mut_ptr(), 1500) };
        // SILK-only, one coded channel.
        assert_eq!((a[0] >> 2, b[0] >> 2), (0x09 << 1, 0x09 << 1), "frame {f}");
        for (d, p, peak) in [
            (d_rs, &a[..la], &mut peak_rs),
            (d_c, &b[..lb as usize], &mut peak_c),
        ] {
            unsafe { opus_decode(d, p.as_ptr(), p.len() as i32, out.as_mut_ptr(), n as i32, 0) };
            if f >= 10 {
                *peak = out.iter().fold(*peak, |m, &x| m.max(x.saturating_abs()));
            }
        }
    }
    unsafe {
        opus_encoder_destroy(c);
        opus_decoder_destroy(d_rs);
        opus_decoder_destroy(d_c);
    }
    assert!(peak_c > 25000, "libopus peak {peak_c}");
    assert!(
        (peak_rs - peak_c).abs() < peak_c / 20,
        "peaks {peak_rs} and {peak_c}"
    );
}
