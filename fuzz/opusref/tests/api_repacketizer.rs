//! The repacketizer and packet padding against libopus's
//! `src/repacketizer.c`: libopus-encoded packets with and without
//! extensions, merged, split, padded and unpadded, single-stream and
//! multistream, and generated invalid packets.
use core::ffi::c_int;
use opus_rs::{error, repacketizer as rs};
use opusref::repacketizer::*;
use opusref::*;

fn code<T>(r: Result<T, &str>) -> i32
where
    T: Into<i64>,
{
    match r {
        Ok(v) => v.into() as i32,
        Err(e) => error::code(e).unwrap(),
    }
}

/// `count` packets of `ms` ms from libopus in `force` mode, each padded
/// with extensions in every frame when `exts`.
fn packets(fs: i32, ch: i32, ms: usize, force: i32, count: usize, exts: bool) -> Vec<Vec<u8>> {
    let frame = fs as usize * ms / 1000;
    let enc = unsafe { opus_encoder_create(fs, ch, 2049, &mut 0) };
    unsafe {
        opus_encoder_ctl(enc, SET_FORCE_MODE, force);
        opus_encoder_ctl(enc, SET_BITRATE, 48000);
    }
    let mut out = Vec::new();
    for f in 0..count {
        let pcm: Vec<f32> = (0..frame * ch as usize)
            .map(|i| ((f * frame + i) as f32 * 0.031).sin() * 0.3)
            .collect();
        let mut p = vec![0u8; 1500];
        let n = unsafe { opus_encode_float(enc, pcm.as_ptr(), frame as i32, p.as_mut_ptr(), 1500) };
        assert!(n > 0);
        p.truncate(n as usize);
        if exts {
            let frames = opus_rs::packet::nb_frames(&p).unwrap();
            let payload: Vec<Vec<u8>> = (0..frames).map(|i| vec![f as u8, i as u8, 7]).collect();
            let mut e = Vec::new();
            for (i, d) in payload.iter().enumerate() {
                e.push(OpusExtensionData { id: 5, frame: i as c_int, data: d.as_ptr(), len: 1 });
                e.push(OpusExtensionData { id: 40, frame: i as c_int, data: d.as_ptr(), len: 3 });
            }
            let len = p.len();
            p.resize(len + 40, 0);
            let r = unsafe {
                opus_packet_pad_impl(p.as_mut_ptr(), len as i32, len as i32 + 40, 0, e.as_ptr(), e.len() as c_int)
            };
            assert!(r > 0, "{r}");
            p.truncate(r as usize);
        }
        out.push(p);
    }
    unsafe { opus_encoder_destroy(enc) };
    out
}

/// Merges `pkts` in both repacketizers and compares every span written
/// into buffers of several sizes.
fn compare_repacketizer(pkts: &[Vec<u8>], what: &str) {
    let c = unsafe { opus_repacketizer_create() };
    let mut r = rs::Repacketizer::new();
    for (k, p) in pkts.iter().enumerate() {
        let a = code(r.cat(p).map(|()| 0));
        let b = unsafe { opus_repacketizer_cat(c, p.as_ptr(), p.len() as i32) };
        assert_eq!(a, b, "{what}: cat {k}");
        let n = unsafe { opus_repacketizer_get_nb_frames(c) };
        assert_eq!(r.nb_frames() as i32, n, "{what}: frames after cat {k}");
    }
    let n = r.nb_frames();
    for begin in 0..=n {
        for end in begin..=n + 1 {
            for max in [0, 1, 3, 60, 400, 1500, 4000] {
                let (mut x, mut y) = (vec![0u8; max], vec![0u8; max]);
                let a = code(r.out_range(begin, end, &mut x).map(|v| v as i64));
                let b = unsafe {
                    opus_repacketizer_out_range(c, begin as c_int, end as c_int, y.as_mut_ptr(), max as i32)
                };
                assert_eq!(a, b, "{what}: out_range {begin}..{end} into {max}");
                if b > 0 {
                    assert_eq!(x[..b as usize], y[..b as usize], "{what}: {begin}..{end}");
                }
            }
        }
    }
    let (mut x, mut y) = (vec![0u8; 5000], vec![0u8; 5000]);
    let a = code(r.out(&mut x).map(|v| v as i64));
    let b = unsafe { opus_repacketizer_out(c, y.as_mut_ptr(), 5000) };
    assert_eq!(a, b, "{what}: out");
    unsafe { opus_repacketizer_destroy(c) };
}

/// Pads and unpads `p` in both, to every length in `lens`.
fn compare_pad(p: &[u8], lens: &[usize], what: &str) {
    for &new_len in lens {
        let mut out = vec![0u8; new_len];
        let a = code(rs::pad(p, &mut out).map(|()| 0));
        let mut y = p.to_vec();
        y.resize(new_len.max(p.len()), 0);
        let b = unsafe { opus_packet_pad(y.as_mut_ptr(), p.len() as i32, new_len as i32) };
        assert_eq!(a, b, "{what}: pad to {new_len}");
        if b == 0 {
            assert_eq!(out, y[..new_len], "{what}: pad to {new_len}");
        }
    }
    let mut x = p.to_vec();
    let mut y = p.to_vec();
    let a = code(rs::unpad(&mut x).map(|v| v as i64));
    let b = unsafe { opus_packet_unpad(y.as_mut_ptr(), y.len() as i32) };
    assert_eq!(a, b, "{what}: unpad");
    if b > 0 {
        assert_eq!(x[..b as usize], y[..b as usize], "{what}: unpad");
    }
}

#[test]
fn repacketizer_matches_libopus() {
    for exts in [false, true] {
        for (fs, ch, ms, force, n) in [
            (48000, 2, 10, 1002, 12),
            (48000, 1, 20, 1002, 7),
            (16000, 1, 20, 1000, 7),
            (16000, 2, 60, 1000, 3),
            (48000, 2, 20, 1001, 7),
        ] {
            let pkts = packets(fs, ch, ms, force, n, exts);
            let what = format!("{fs} {ch} {ms} ms mode {force} exts {exts}");
            compare_repacketizer(&pkts, &what);
            // A packet of another configuration in between.
            let mut mixed = pkts[..2].to_vec();
            mixed.extend(packets(8000, 1, 20, 1000, 1, false));
            mixed.push(pkts[2].clone());
            compare_repacketizer(&mixed, &what);
            for p in &pkts {
                compare_pad(p, &[0, 1, p.len() - 1, p.len(), p.len() + 1, p.len() + 2, p.len() + 300, 2000], &what);
            }
        }
    }
}

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 >> 8
    }
}

/// Generated packets, mostly invalid, for every function.
#[test]
fn generated_packets_match_libopus() {
    let mut rng = Lcg(11);
    for k in 0..20_000 {
        let len = 1 + rng.next() as usize % [3, 12, 80, 700][k % 4];
        let mut p: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        if k % 3 == 0 && len > 2 {
            // A code 3 header with padding, so extensions get parsed.
            p[0] |= 3;
            p[1] = (p[1] & 0x0f).max(1) | 0x40;
        }
        let what = format!("packet {p:02x?}");
        compare_pad(&p, &[len, len + 1, len + 9, len + 600], &what);
        compare_repacketizer(&[p.clone(), p.clone()], &what);
        for streams in [0, 1, 2, 3] {
            compare_multistream(&p, streams, &what);
        }
    }
}

/// libopus multistream packets of `streams` streams, padded and unpadded.
fn compare_multistream(p: &[u8], streams: usize, what: &str) {
    for new_len in [p.len(), p.len() + 1, p.len() + 70, p.len().saturating_sub(1)] {
        let mut out = vec![0u8; new_len];
        let a = code(rs::multistream_pad(p, streams, &mut out).map(|()| 0));
        let mut y = p.to_vec();
        y.resize(new_len.max(p.len()), 0);
        let b = unsafe {
            opus_multistream_packet_pad(y.as_mut_ptr(), p.len() as i32, new_len as i32, streams as c_int)
        };
        assert_eq!(a, b, "{what}: multistream pad {streams} streams to {new_len}");
        if b == 0 {
            assert_eq!(out, y[..new_len], "{what}: multistream pad to {new_len}");
        }
    }
    let (mut x, mut y) = (p.to_vec(), p.to_vec());
    let a = code(rs::multistream_unpad(&mut x, streams).map(|v| v as i64));
    let b = unsafe { opus_multistream_packet_unpad(y.as_mut_ptr(), y.len() as i32, streams as c_int) };
    assert_eq!(a, b, "{what}: multistream unpad {streams} streams");
    if b > 0 {
        assert_eq!(x[..b as usize], y[..b as usize], "{what}: multistream unpad");
    }
}

#[test]
fn multistream_packets_match_libopus() {
    for (ch, streams, coupled) in [(2u8, 1, 1), (4, 3, 1), (6, 4, 2), (3, 3, 0)] {
        let mapping: Vec<u8> = (0..ch).collect();
        let mut err = 0;
        let enc = unsafe {
            opus_multistream_encoder_create(48000, ch as c_int, streams, coupled, mapping.as_ptr(), 2049, &mut err)
        };
        assert!(!enc.is_null() && err == 0);
        unsafe { opus_multistream_encoder_ctl(enc, SET_BITRATE, 64000 * streams) };
        for f in 0..10 {
            let pcm: Vec<f32> = (0..960 * ch as usize)
                .map(|i| ((f * 960 + i) as f32 * 0.02).sin() * 0.3)
                .collect();
            let mut out = vec![0u8; 4000];
            let n = unsafe { opus_multistream_encode_float(enc, pcm.as_ptr(), 960, out.as_mut_ptr(), 4000) };
            assert!(n > 0);
            // Padded by libopus first, so unpadding has padding to remove.
            let m = n as usize + 37 * f;
            unsafe { opus_multistream_packet_pad(out.as_mut_ptr(), n, m as i32, streams) };
            let p = &out[..m];
            compare_multistream(p, streams as usize, &format!("{ch} channels frame {f}"));
        }
        unsafe { opus_multistream_encoder_destroy(enc) };
    }
}
