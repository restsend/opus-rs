//! The packet functions of opus.c and opus_decoder.c against opus-rs's
//! `packet` module: every ToC byte, generated packets of every framing
//! code, libopus-encoded packets with and without LBRR, and soft clipping.
use opus_rs::{OpusDecoder, error, packet};
use opusref::*;

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 >> 8
    }
}

fn code<T>(r: Result<T, &str>) -> Result<T, i32> {
    r.map_err(|e| error::code(e).unwrap())
}

#[test]
fn toc_fields_match_libopus() {
    for toc in 0..=255u8 {
        let p = [toc];
        assert_eq!(packet::bandwidth(toc) as i32, unsafe { opus_packet_get_bandwidth(p.as_ptr()) });
        assert_eq!(packet::nb_channels(toc) as i32, unsafe { opus_packet_get_nb_channels(p.as_ptr()) });
        for fs in [8000, 12000, 16000, 24000, 48000, 44100, 1, 96000] {
            let c = unsafe { opus_packet_get_samples_per_frame(p.as_ptr(), fs) };
            assert_eq!(packet::samples_per_frame(toc, fs), c, "toc {toc:#x} fs {fs}");
        }
    }
}

/// A packet of `len` bytes: ToC `toc`, then bytes from `rng` biased to
/// small values so that frame lengths parse.
fn gen_packet(rng: &mut Lcg, toc: u8, len: usize) -> Vec<u8> {
    let mut p = vec![toc];
    for i in 1..len {
        let r = rng.next();
        p.push(if i <= 3 && r % 3 == 0 { (r >> 8) as u8 % 8 } else { (r >> 4) as u8 });
    }
    p
}

#[test]
fn parse_and_counts_match_libopus() {
    let mut rng = Lcg(7);
    let dec = unsafe { opus_decoder_create(16000, 1, &mut 0) };
    let rs_dec = OpusDecoder::new(16000, 1).unwrap();
    for n in 0..200_000 {
        let toc = rng.next() as u8;
        let len = [0, 1, 2, 3, 5, 9, 40, 300, 1400, 2000][n % 10] + rng.next() as usize % 4;
        let p = gen_packet(&mut rng, toc, len);
        let (ptr, l) = (p.as_ptr(), p.len() as i32);
        let what = format!("packet {p:02x?}");

        let mut ctoc = 0u8;
        let mut frames = [core::ptr::null(); 48];
        let mut size = [0i16; 48];
        let mut offset = -1;
        let c = unsafe {
            opus_packet_parse(ptr, l, &mut ctoc, frames.as_mut_ptr(), size.as_mut_ptr(), &mut offset)
        };
        match code(packet::parse(&p)) {
            Ok(parts) => {
                assert_eq!(parts.count as i32, c, "{what}");
                assert_eq!(parts.toc, ctoc, "{what}");
                assert_eq!(parts.payload_offset as i32, offset, "{what}");
                for (i, f) in parts.frames().iter().enumerate() {
                    assert_eq!(f.len() as i16, size[i], "{what} frame {i}");
                    assert_eq!(f.as_ptr(), frames[i], "{what} frame {i}");
                }
            }
            Err(e) => assert_eq!(e, c, "{what}"),
        }
        let c = unsafe { opus_packet_get_nb_frames(ptr, l) };
        assert_eq!(code(packet::nb_frames(&p)).map(|v| v as i32), if c < 0 { Err(c) } else { Ok(c) }, "{what}");
        for fs in [8000, 48000] {
            let c = unsafe { opus_packet_get_nb_samples(ptr, l, fs) };
            let r = code(packet::nb_samples(&p, fs)).map(|v| v as i32);
            assert_eq!(r, if c < 0 { Err(c) } else { Ok(c) }, "{what} fs {fs}");
        }
        let c = unsafe { opus_decoder_get_nb_samples(dec, ptr, l) };
        let r = code(rs_dec.nb_samples(&p)).map(|v| v as i32);
        assert_eq!(r, if c < 0 { Err(c) } else { Ok(c) }, "{what}");
        if !p.is_empty() {
            let c = unsafe { opus_packet_has_lbrr(ptr, l) };
            let r = code(packet::has_lbrr(&p)).map(|v| v as i32);
            assert_eq!(r, if c < 0 { Err(c) } else { Ok(c) }, "{what}");
        }
    }
    unsafe { opus_decoder_destroy(dec) };
}

/// LBRR in libopus-encoded SILK and Hybrid packets of 10 to 60 ms.
#[test]
fn has_lbrr_on_encoded_packets() {
    let mut lbrr = 0;
    for (fs, ch, force) in [(16000, 1, 1000), (16000, 2, 1000), (48000, 2, 1001)] {
        for ms in [10, 20, 40, 60] {
            let frame = fs as usize * ms / 1000;
            let enc = unsafe { opus_encoder_create(fs, ch, 2048, &mut 0) };
            unsafe {
                opus_encoder_ctl(enc, SET_FORCE_MODE, force);
                opus_encoder_ctl(enc, SET_INBAND_FEC, 1);
                opus_encoder_ctl(enc, SET_PACKET_LOSS_PERC, 30);
                opus_encoder_ctl(enc, SET_BITRATE, 32000);
            }
            for f in 0..20 {
                let pcm: Vec<f32> = (0..frame * ch as usize)
                    .map(|i| ((f * frame + i) as f32 * 0.05).sin() * 0.3)
                    .collect();
                let mut out = [0u8; 1500];
                let n = unsafe { opus_encode_float(enc, pcm.as_ptr(), frame as i32, out.as_mut_ptr(), 1500) };
                let p = &out[..n as usize];
                let c = unsafe { opus_packet_has_lbrr(p.as_ptr(), n) };
                assert_eq!(packet::has_lbrr(p).unwrap() as i32, c, "{fs} {ch} {ms} ms frame {f}");
                lbrr += c;
            }
            unsafe { opus_encoder_destroy(enc) };
        }
    }
    assert!(lbrr > 50, "{lbrr} packets with LBRR");
}

#[test]
fn soft_clip_matches_libopus() {
    let mut rng = Lcg(3);
    for ch in 1..=3usize {
        let (mut m_rs, mut m_c) = (vec![0f32; ch], vec![0f32; ch]);
        for k in 0..400 {
            let n = 1 + rng.next() as usize % 300;
            let scale = [0.5, 1.2, 3.0, 8.0][k % 4];
            let x: Vec<f32> = (0..n * ch)
                .map(|i| {
                    let t = (i / ch) as f32;
                    scale * ((t * 0.07 + k as f32).sin() + (rng.next() % 1000) as f32 / 4000.0)
                })
                .collect();
            let (mut a, mut b) = (x.clone(), x);
            packet::soft_clip(&mut a, n, ch, &mut m_rs).unwrap();
            unsafe { opus_pcm_soft_clip(b.as_mut_ptr(), n as i32, ch as i32, m_c.as_mut_ptr()) };
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "ch {ch} call {k}"
            );
            assert_eq!(m_rs, m_c);
        }
    }
    let mut x = [0.5f32; 4];
    assert_eq!(packet::soft_clip(&mut x, 3, 2, &mut [0.0; 2]), Err(error::BAD_ARG));
    assert_eq!(packet::soft_clip(&mut x, 2, 2, &mut [0.0; 1]), Err(error::BAD_ARG));
    assert_eq!(packet::soft_clip(&mut x, 0, 2, &mut []), Ok(()));
}
