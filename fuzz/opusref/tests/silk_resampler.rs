//! SilkResampler against libopus's silk_resampler_init and silk_resampler
//! (silk/resampler.c): the decoder-direction rate check and the outputs of
//! successive calls, also with lengths that leave the down-FIR decimator a
//! one-sample remainder.
use opus_rs::silk::SilkResampler;
use opusref::*;
use std::os::raw::c_int;

unsafe extern "C" {
    fn silk_resampler_init(s: *mut u64, fs_in: i32, fs_out: i32, for_enc: c_int) -> c_int;
    fn silk_resampler(s: *mut u64, out: *mut i16, input: *const i16, in_len: i32) -> c_int;
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

#[test]
fn outputs_match_libopus() {
    let _ = unsafe { silk_lin2log(1) };
    let mut seed = 1u32;
    let pairs = [
        (8000, 8000, 0),
        (8000, 16000, 0),
        (8000, 48000, 0),
        (12000, 48000, 0),
        (16000, 24000, 0),
        (16000, 48000, 0),
        (48000, 8000, 1),
        (48000, 12000, 1),
        (48000, 16000, 1),
        (24000, 8000, 1),
        (24000, 12000, 1),
        (24000, 16000, 1),
        (16000, 8000, 1),
        (16000, 12000, 1),
        (12000, 8000, 1),
    ];
    for &(fs_in, fs_out, for_enc) in &pairs {
        let mut c: CState = [0; 128];
        assert_eq!(
            unsafe { silk_resampler_init(c.as_mut_ptr(), fs_in, fs_out, for_enc) },
            0
        );
        let mut r = SilkResampler::default();
        let rr = if for_enc == 1 {
            r.init_for_enc(fs_in, fs_out)
        } else {
            r.init(fs_in, fs_out)
        };
        assert_eq!(rr, 0);
        let k = (fs_in / 1000) as usize;
        // Whole 10 ms blocks, and the delay buffer plus whole batches plus
        // one sample (a one-sample remainder for the down-FIR decimator).
        for &len in &[
            10 * k,
            k + 10 * k + 1,
            20 * k,
            k + 20 * k + 1,
            k + 1,
            10 * k,
        ] {
            let x: Vec<i16> = (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 16) as i16 / 2
                })
                .collect();
            let n_out = len * fs_out as usize / fs_in as usize + 4;
            let (mut a, mut b) = (vec![0x5555i16; n_out], vec![0x5555i16; n_out]);
            r.process(&mut a, &x, len as i32);
            unsafe { silk_resampler(c.as_mut_ptr(), b.as_mut_ptr(), x.as_ptr(), len as i32) };
            assert_eq!(a, b, "{fs_in} -> {fs_out} len {len}");
        }
    }
}
