//! The ambisonics projection encoder and decoder against libopus's
//! opus_projection_*: channel counts and families, the demixing matrix,
//! packets (byte-identical in RestrictedLowDelay) and decoding, also of
//! lost packets, in every output format.
use core::ffi::c_void;
use opus_rs::projection::{ProjectionDecoder, ProjectionEncoder};
use opus_rs::{Application, error};
use opusref::*;

fn code<T>(r: &Result<T, &str>) -> i32 {
    r.as_ref().map_or_else(|e| error::code(e).unwrap(), |_| 0)
}

/// The ambisonic test signal: a rotating source and some diffuse noise.
fn signal(ch: usize, frame: usize, f: usize) -> Vec<f32> {
    let mut seed = (f as u32).wrapping_mul(2654435761);
    (0..frame * ch)
        .map(|i| {
            let t = (f * frame + i / ch) as f32 / 48000.0;
            let k = (i % ch) as f32;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            0.3 * (t * 330.0 * 6.2832 + k).sin() * (t * 2.0 + 0.7 * k).cos() + 0.02 * ((seed >> 9) as f32 / 8388608.0 - 1.0)
        })
        .collect()
}

#[test]
fn encoders_are_created_like_libopus() {
    for family in [2, 3, 1] {
        for ch in 0..=40 {
            let (mut s, mut c, mut err) = (0, 0, 0);
            let p = unsafe { opus_projection_ambisonics_encoder_create(48000, ch, family, &mut s, &mut c, 2049, &mut err) };
            let rs = ProjectionEncoder::new(48000, ch as usize, family, Application::Audio);
            assert_eq!(code(&rs), err, "family {family} channels {ch}");
            if let Ok(e) = &rs {
                assert_eq!((e.streams() as i32, e.coupled_streams() as i32), (s, c));
                let (mut gain, mut size) = (0i32, 0i32);
                unsafe {
                    opus_projection_encoder_ctl(p, PROJECTION_GET_DEMIXING_MATRIX_GAIN, &mut gain as *mut i32);
                    opus_projection_encoder_ctl(p, PROJECTION_GET_DEMIXING_MATRIX_SIZE, &mut size as *mut i32);
                }
                assert_eq!((e.demixing_matrix_gain(), e.demixing_matrix_size() as i32), (gain, size));
                let (mut a, mut b) = (vec![0u8; size as usize], vec![0u8; size as usize]);
                e.demixing_matrix(&mut a).unwrap();
                unsafe { opus_projection_encoder_ctl(p, PROJECTION_GET_DEMIXING_MATRIX, b.as_mut_ptr(), size) };
                assert_eq!(a, b);
                assert_eq!(e.demixing_matrix(&mut vec![0; size as usize + 1]), Err(error::BAD_ARG));
                unsafe { opus_projection_encoder_destroy(p) };
            }
        }
    }
}

struct Dec {
    rs: ProjectionDecoder,
    c: *mut c_void,
    ch: usize,
}

impl Dec {
    fn new(ch: usize, s: usize, c: usize, matrix: &mut [u8]) -> Self {
        let mut err = 0;
        let p = unsafe {
            opus_projection_decoder_create(48000, ch as i32, s as i32, c as i32, matrix.as_mut_ptr(), matrix.len() as i32, &mut err)
        };
        assert!(!p.is_null(), "{err}");
        Dec { rs: ProjectionDecoder::new(48000, ch, s, c, matrix).unwrap(), c: p, ch }
    }

    /// Decodes in `format` (0 float, 1 16-bit, 2 24-bit) and compares.
    fn decode(&mut self, p: &[u8], frame: usize, format: usize, what: &str) {
        let n = frame * self.ch;
        let (ptr, len) = if p.is_empty() { (core::ptr::null(), 0) } else { (p.as_ptr(), p.len() as i32) };
        let (r, m, a, b, step): (_, _, Vec<f32>, Vec<f32>, f32) = match format {
            0 => {
                let (mut x, mut y) = (vec![0f32; n], vec![0f32; n]);
                let r = self.rs.decode(p, frame, &mut x);
                let m = unsafe { opus_projection_decode_float(self.c, ptr, len, y.as_mut_ptr(), frame as i32, 0) };
                (r, m, x, y, 0.0)
            }
            1 => {
                let (mut x, mut y) = (vec![0i16; n], vec![0i16; n]);
                let r = self.rs.decode_i16(p, frame, &mut x);
                let m = unsafe { opus_projection_decode(self.c, ptr, len, y.as_mut_ptr(), frame as i32, 0) };
                let f = |v: Vec<i16>| v.iter().map(|&s| s as f32 / 32768.0).collect();
                (r, m, f(x), f(y), 1.0 / 32768.0)
            }
            _ => {
                let (mut x, mut y) = (vec![0i32; n], vec![0i32; n]);
                let r = self.rs.decode24(p, frame, &mut x);
                let m = unsafe { opus_projection_decode24(self.c, ptr, len, y.as_mut_ptr(), frame as i32, 0) };
                let f = |v: Vec<i32>| v.iter().map(|&s| s as f32 / 8388608.0).collect();
                (r, m, f(x), f(y), 64.0 / 8388608.0)
            }
        };
        assert_eq!(r.map_or_else(|e| error::code(e).unwrap_or(0), |v| v as i32), m, "{what}: {r:?}");
        let Ok(k) = r else { return };
        assert_eq!(k as i32, m, "{what}");
        for i in 0..k * self.ch {
            assert!((a[i] - b[i]).abs() <= 2e-5 + step, "{what}: sample {i}: {} vs {}", a[i], b[i]);
        }
    }
}

impl Drop for Dec {
    fn drop(&mut self) {
        unsafe { opus_projection_decoder_destroy(self.c) }
    }
}

#[test]
fn packets_and_decoding_match_libopus() {
    for app in [Application::RestrictedLowDelay, Application::Audio] {
        for ch in [4usize, 6, 9, 11, 16, 25, 36] {
            let (mut s, mut c, mut err) = (0, 0, 0);
            let p = unsafe { opus_projection_ambisonics_encoder_create(48000, ch as i32, 3, &mut s, &mut c, app as i32, &mut err) };
            let mut e = ProjectionEncoder::new(48000, ch, 3, app).unwrap();
            let rate = 32000 * s;
            e.multistream().set_bitrate(rate).unwrap();
            unsafe { opus_projection_encoder_ctl(p, SET_BITRATE, rate) };
            let mut matrix = vec![0u8; e.demixing_matrix_size()];
            e.demixing_matrix(&mut matrix).unwrap();
            let mut d = Dec::new(ch, s as usize, c as usize, &mut matrix);
            for f in 0..10 {
                let what = format!("{app:?} {ch} channels frame {f}");
                let frame = [960, 480, 960, 1920][f % 4];
                let pcm = signal(ch, frame, f);
                let (mut x, mut y) = (vec![0u8; 8000], vec![0u8; 8000]);
                let (n, m) = match f % 3 {
                    0 => (e.encode(&pcm, frame, &mut x), unsafe {
                        opus_projection_encode_float(p, pcm.as_ptr(), frame as i32, y.as_mut_ptr(), 8000)
                    }),
                    1 => {
                        let q: Vec<i16> = pcm.iter().map(|&v| (v * 32768.0) as i16).collect();
                        (e.encode_i16(&q, frame, &mut x), unsafe {
                            opus_projection_encode(p, q.as_ptr(), frame as i32, y.as_mut_ptr(), 8000)
                        })
                    }
                    _ => {
                        let q: Vec<i32> = pcm.iter().map(|&v| (v * 8388608.0) as i32).collect();
                        (e.encode24(&q, frame, &mut x), unsafe {
                            opus_projection_encode24(p, q.as_ptr(), frame as i32, y.as_mut_ptr(), 8000)
                        })
                    }
                };
                let n = n.unwrap();
                assert!(m > 0, "{what}");
                if app == Application::RestrictedLowDelay {
                    assert_eq!(x[..n], y[..m as usize], "{what}: bytes");
                }
                // libopus's packets decode alike on both decoders.
                let pkt = &y[..m as usize];
                d.decode(if f == 5 { &[] } else { pkt }, 1920, f % 3, &what);
                // A frame size of 0 is an invalid argument.
                d.decode(if f == 5 { &[] } else { pkt }, 0, f % 3, &what);
            }
            unsafe { opus_projection_encoder_destroy(p) };
        }
    }
}
