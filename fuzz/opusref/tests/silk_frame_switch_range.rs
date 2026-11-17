//! libopus's decoder ends every opus-rs packet with the encoder's final
//! range, also when SILK switches from longer frames to 10 ms frames.
use opus_rs::{Application, Bandwidth, OpusEncoder};
use opusref as c;

fn voiced(t: usize, fs: usize) -> f32 {
    let t = t as f32 / fs as f32;
    let mut x = 0.0;
    for h in 1..10 {
        x += (2.0 * std::f32::consts::PI * 150.0 * h as f32 * t).sin() / h as f32;
    }
    0.3 * x * (0.6 + 0.4 * (5.0 * t).sin())
}

/// The first packet whose final range differs, if any.
fn first_mismatch(first_ms: usize, cx: i32, cbr: bool) -> Option<usize> {
    let fs = 48000;
    let mut enc = OpusEncoder::new(fs as i32, 1, Application::Audio).unwrap();
    enc.bitrate_bps = 20000;
    enc.complexity = cx;
    enc.use_cbr = cbr;
    enc.set_max_bandwidth(Bandwidth::Mediumband);
    let mut err = 0;
    let dec = unsafe { c::opus_decoder_create(fs as i32, 1, &mut err) };
    assert_eq!(err, 0);
    let (mut t, mut res) = (0, None);
    for k in 0..60 {
        let n = fs / 1000 * if k < 20 { first_ms } else { 10 };
        let pcm: Vec<f32> = (t..t + n).map(|i| voiced(i, fs)).collect();
        t += n;
        let mut pkt = [0u8; 1500];
        let len = enc.encode(&pcm, n, &mut pkt).unwrap();
        let mut out = vec![0f32; 5760];
        let r = unsafe {
            c::opus_decode_float(dec, pkt.as_ptr(), len as i32, out.as_mut_ptr(), 5760, 0)
        };
        assert_eq!(r, n as i32);
        let mut range = 0u32;
        unsafe { c::opus_decoder_ctl(dec, c::GET_FINAL_RANGE, &mut range as *mut u32) };
        if range != enc.final_range() {
            res = Some(k);
            break;
        }
    }
    unsafe { c::opus_decoder_destroy(dec) };
    res
}

#[test]
fn ten_ms_silk_frames_after_longer_frames_keep_the_range() {
    let mut fails = Vec::new();
    for first_ms in [10, 20, 40, 60] {
        for cx in [3, 10] {
            for cbr in [false, true] {
                if let Some(k) = first_mismatch(first_ms, cx, cbr) {
                    fails.push((first_ms, cx, cbr, k));
                }
            }
        }
    }
    assert!(
        fails.is_empty(),
        "(first ms, complexity, cbr, packet): {fails:?}"
    );
}
