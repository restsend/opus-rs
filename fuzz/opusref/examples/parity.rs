//! Byte-identity baseline: how often opus-rs (x86 SIMD off) and the scalar
//! libopus reference produce identical CELT-only packets, per configuration.
//! The `bytes` oracle of `diff_encoder` is only informative where this is high.
use opus_rs::{Application, OpusEncoder};
use opusref::*;

fn main() {
    opus_rs::fuzzing::SIMD_ENABLED.store(false, core::sync::atomic::Ordering::Relaxed);
    let fs = 48000;
    for ch in [1usize, 2] {
        for ms in [10usize, 20] {
            for kbps in [24, 32, 64, 96, 128, 192, 256, 510] {
                for rc in 0..3 {
                    for sig in ["voice", "noise", "silence+voice"] {
                        let frame = fs * ms / 1000;
                        let mut e =
                            OpusEncoder::new(fs as i32, ch, Application::RestrictedLowDelay)
                                .unwrap();
                        e.bitrate_bps = kbps * 1000;
                        e.complexity = 10;
                        e.use_cbr = rc == 2;
                        e.set_vbr_constraint(rc == 1);
                        let mut err = 0;
                        let r =
                            unsafe { opus_encoder_create(fs as i32, ch as i32, 2051, &mut err) };
                        unsafe {
                            opus_encoder_ctl(r, SET_BITRATE, kbps * 1000);
                            opus_encoder_ctl(r, SET_COMPLEXITY, 10);
                            opus_encoder_ctl(r, SET_VBR, (rc != 2) as i32);
                            opus_encoder_ctl(r, SET_VBR_CONSTRAINT, (rc == 1) as i32);
                        }
                        let (mut same, mut first) = (0, None);
                        let frames = 50;
                        let mut seed = 1u32;
                        for f in 0..frames {
                            let pcm: Vec<f32> = (0..frame * ch)
                                .map(|i| {
                                    let t = (f * frame + i / ch) as f32 / fs as f32;
                                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                                    let noise = (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5;
                                    let voice = (0..20)
                                        .map(|h| {
                                            ((h + 1) as f32 * 140.0 * t * 6.2832).sin()
                                                / (h + 1) as f32
                                        })
                                        .sum::<f32>()
                                        * 0.2;
                                    match sig {
                                        "voice" => voice,
                                        "noise" => noise * 0.5,
                                        _ => {
                                            if f < 10 {
                                                0.0
                                            } else {
                                                voice
                                            }
                                        }
                                    }
                                })
                                .collect();
                            let mut a = [0u8; 1500];
                            let mut b = [0u8; 1500];
                            let n = e.encode(&pcm, frame, &mut a).unwrap();
                            let m = unsafe {
                                opus_encode_float(
                                    r,
                                    pcm.as_ptr(),
                                    frame as i32,
                                    b.as_mut_ptr(),
                                    1500,
                                )
                            } as usize;
                            if a[..n] == b[..m] {
                                same += 1
                            } else if first.is_none() {
                                first = Some(f)
                            }
                        }
                        unsafe { opus_encoder_destroy(r) };
                        println!(
                            "ch{ch} {ms}ms {kbps:>3}k rc{rc} {sig:<14} identical {same:>2}/{frames} first diff {first:?}"
                        );
                    }
                }
            }
        }
    }
}
