//! CELT packet loss concealment against libopus: harmonic signals with
//! pitch periods from 160 to 690 samples, bursts of 1 to 45 lost frames
//! (periodic concealment, the switch to noise after 100 ms, consecutive
//! losses, the first good frame after a loss), every frame size, mono and
//! stereo. The tolerance covers CELT's float rounding, which concealment
//! amplifies.
use opus_rs::OpusDecoder;
use opusref::*;

const BURSTS: [usize; 6] = [1, 2, 4, 9, 3, 45];

fn run(frame: usize, channels: usize, f0: f32) -> Result<(), String> {
    let fs = 48000;
    let mut err = 0;
    let enc = unsafe { opus_encoder_create(fs, channels as i32, 2049, &mut err) };
    unsafe {
        opus_encoder_ctl(enc, SET_BITRATE, 64000 * channels as i32);
        opus_encoder_ctl(enc, 11002, 1002); // OPUS_SET_FORCE_MODE: CELT
    }
    let mut rs = OpusDecoder::new(fs, channels).unwrap();
    let c = unsafe { opus_decoder_create(fs, channels as i32, &mut err) };
    let n = frame * channels;
    let (mut a, mut b) = (vec![0f32; n], vec![0f32; n]);
    let mut lost_left = 0;
    let mut burst = 0;
    let mut good = 0;
    let mut result = Ok(());
    for f in 0..400usize {
        let pcm: Vec<f32> = (0..n)
            .map(|k| {
                let t = (f * frame + k / channels) as f32 / fs as f32;
                let pitch = f0 * (1.0 + 0.02 * (t * 3.0).sin());
                (1..16)
                    .map(|h| {
                        ((h as f32) * pitch * t * 6.2832 + (k % channels) as f32).sin() / h as f32
                    })
                    .sum::<f32>()
                    * 0.2
            })
            .collect();
        let mut pkt = [0u8; 1500];
        let len =
            unsafe { opus_encode_float(enc, pcm.as_ptr(), frame as i32, pkt.as_mut_ptr(), 1500) };
        // Eight good frames, then the next burst.
        if lost_left == 0 && good >= 8 {
            lost_left = BURSTS[burst % BURSTS.len()];
            burst += 1;
            good = 0;
        }
        let lost = lost_left > 0;
        if lost {
            lost_left -= 1;
        } else {
            good += 1;
        }
        let p: &[u8] = if lost { &[] } else { &pkt[..len as usize] };
        rs.decode(p, frame, &mut a).unwrap();
        let ptr = if lost { core::ptr::null() } else { p.as_ptr() };
        unsafe { opus_decode_float(c, ptr, p.len() as i32, b.as_mut_ptr(), frame as i32, 0) };
        let d = a
            .iter()
            .zip(&b)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        if d > 1e-3 {
            result = Err(format!(
                "frame {frame} channels {channels} f0 {f0}: frame {f} (lost {lost}) differs by {d}"
            ));
            break;
        }
    }
    unsafe { opus_encoder_destroy(enc) };
    unsafe { opus_decoder_destroy(c) };
    result
}

#[test]
fn concealment_matches_libopus() {
    let mut failures = Vec::new();
    for frame in [120, 240, 480, 960] {
        for channels in [1, 2] {
            for f0 in [70.0, 85.0, 140.0, 300.0] {
                if let Err(e) = run(frame, channels, f0) {
                    failures.push(e);
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
