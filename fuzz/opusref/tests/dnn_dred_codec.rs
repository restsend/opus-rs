//! DRED in the codec: opus-rs's packet extension parsing, DRED parser,
//! DRED decoding and DRED encoding against the DNN-enabled libopus.
#![cfg(feature = "dnn")]
use opusref::dnn::Rng;
use opusref::dnn_dred_api::*;

/// opus-rs's extensions of a padding, as (id, frame, payload).
fn rs_extensions(data: &[u8], nb_frames: usize) -> Result<Vec<(u8, usize, Vec<u8>)>, &'static str> {
    let mut it = opus_rs::extensions::ExtensionIterator::new(data, nb_frames);
    let mut out = Vec::new();
    while let Some(e) = it.next()? {
        out.push((e.id, e.frame, e.data.to_vec()));
    }
    Ok(out)
}

/// Random extension lists written by libopus's
/// opus_packet_extensions_generate (which uses separators and the repeat
/// mechanism), plus random bytes: opus-rs's iterator returns what
/// opus_packet_extensions_parse returns, or fails where it fails.
#[test]
fn extension_iterator_matches_libopus() {
    let mut r = Rng(0x5eed);
    let mut generated = 0;
    let mut valid_random = 0;
    for case in 0..20000 {
        let nb_frames = 1 + r.below(if case % 2 == 0 { 4 } else { 48 }) as usize;
        let data: Vec<u8> = if case % 3 == 0 {
            // Random bytes, biased to small IDs and lengths.
            let n = r.below(40) as usize;
            (0..n).map(|_| if r.below(2) == 0 { r.below(16) as u8 } else { r.next_u32() as u8 }).collect()
        } else {
            let n = r.below(8) as usize;
            let payloads: Vec<Vec<u8>> = (0..n)
                .map(|_| {
                    let len = match r.below(4) {
                        0 => 0,
                        1 => 1,
                        2 => r.below(20) as usize,
                        _ => r.below(600) as usize,
                    };
                    (0..len).map(|_| r.next_u32() as u8).collect()
                })
                .collect();
            let mut exts: Vec<OpusExtensionData> = payloads
                .iter()
                .map(|p| {
                    let id = if r.below(2) == 0 { 3 + r.below(29) } else { 32 + r.below(96) } as i32;
                    let len = if id < 32 { p.len().min(1) } else { p.len() } as i32;
                    OpusExtensionData { id, frame: r.below(nb_frames as u32) as i32, data: p.as_ptr(), len }
                })
                .collect();
            // Mostly in frame order, which lets repeats happen.
            if r.below(4) != 0 {
                exts.sort_by_key(|e| e.frame);
            }
            let mut buf = vec![0u8; 8000];
            let pad = r.below(2) as i32;
            let cap = if pad != 0 { 1 + r.below(7000) as i32 } else { 8000 };
            let len = unsafe {
                opus_packet_extensions_generate(buf.as_mut_ptr(), cap, exts.as_ptr(), exts.len() as i32, nb_frames as i32, pad)
            };
            if len < 0 {
                continue;
            }
            generated += 1;
            buf.truncate(len as usize);
            // Sometimes truncated or with a byte changed.
            match r.below(6) {
                0 if !buf.is_empty() => {
                    let k = r.below(buf.len() as u32) as usize;
                    buf.truncate(k);
                }
                1 if !buf.is_empty() => {
                    let k = r.below(buf.len() as u32) as usize;
                    buf[k] = r.next_u32() as u8;
                }
                _ => {}
            }
            buf
        };
        let c = extensions_parse(&data, nb_frames);
        let rs = rs_extensions(&data, nb_frames);
        match (&c, &rs) {
            (Ok(c), Ok(rs)) => {
                assert_eq!(c, rs, "case {case}: {data:?} with {nb_frames} frames");
                if case % 3 == 0 {
                    valid_random += 1;
                }
            }
            (Err(_), Err(_)) => {}
            _ => panic!("case {case}: libopus {c:?}, opus-rs {rs:?} for {data:?} with {nb_frames} frames"),
        }
    }
    assert!(generated > 5000 && valid_random > 1000, "{generated} generated, {valid_random} valid random");
}

/// opus-rs's packet parsing returns the frame count and padding of
/// libopus's opus_packet_parse_impl, and fails where it fails.
#[test]
fn packet_padding_matches_libopus() {
    let mut r = Rng(77);
    for case in 0..50000 {
        let n = 1 + r.below(if case % 4 == 0 { 2000 } else { 40 }) as usize;
        let mut p: Vec<u8> = (0..n).map(|_| r.next_u32() as u8).collect();
        if case % 2 == 0 {
            // Code 3 with a small frame count and padding.
            p[0] |= 3;
            if n > 1 {
                p[1] = (p[1] & 0xc0) | (1 + r.below(6)) as u8;
            }
            if n > 2 && r.below(2) == 0 {
                p[2] = r.below(n as u32) as u8;
            }
        }
        let mut toc = 0u8;
        let mut frames = [core::ptr::null::<u8>(); 48];
        let mut size = [0i16; 48];
        let mut padding = core::ptr::null::<u8>();
        let mut padding_len = 0i32;
        let ret = unsafe {
            opus_packet_parse_impl(
                p.as_ptr(),
                p.len() as i32,
                0,
                &mut toc,
                frames.as_mut_ptr(),
                size.as_mut_ptr(),
                core::ptr::null_mut(),
                core::ptr::null_mut(),
                &mut padding,
                &mut padding_len,
            )
        };
        match opus_rs::extensions::parse_packet(&p) {
            Ok(parts) => {
                assert!(ret > 0, "case {case}: libopus {ret} for {p:?}");
                assert_eq!(parts.count, ret as usize, "case {case}");
                for i in 0..parts.count {
                    let off = unsafe { frames[i].offset_from(p.as_ptr()) } as usize;
                    assert_eq!((parts.frames[i].as_ptr() as usize - p.as_ptr() as usize, parts.frames[i].len()), (off, size[i] as usize), "case {case} frame {i}");
                }
                let off = unsafe { padding.offset_from(p.as_ptr()) } as usize;
                assert_eq!((parts.padding.as_ptr() as usize - p.as_ptr() as usize, parts.padding.len()), (off, padding_len as usize), "case {case}");
            }
            Err(_) => assert!(ret < 0, "case {case}: libopus {ret}, opus-rs failed for {p:?}"),
        }
    }
}
