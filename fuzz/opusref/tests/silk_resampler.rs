//! SilkResampler against libopus's silk_resampler_init (silk/resampler.c):
//! the decoder-direction rate check.
use opus_rs::silk::SilkResampler;
use opusref::*;
use std::os::raw::c_int;

unsafe extern "C" {
    fn silk_resampler_init(s: *mut u64, fs_in: i32, fs_out: i32, for_enc: c_int) -> c_int;
}

/// Room for libopus's `silk_resampler_state_struct`.
type CState = [u64; 128];

#[test]
fn decoder_rates_match_libopus() {
    let _ = unsafe { silk_lin2log(1) };
    let ins = [8000, 11025, 12000, 16000, 24000, 48000];
    let outs = [
        4000, 8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000, 96000,
    ];
    for &fs_in in &ins {
        for &fs_out in &outs {
            let mut c: CState = [0; 128];
            let rc = unsafe { silk_resampler_init(c.as_mut_ptr(), fs_in, fs_out, 0) };
            let rr = SilkResampler::default().init(fs_in, fs_out);
            assert_eq!(rr, rc, "decoder init {fs_in} -> {fs_out}");
        }
    }
}
