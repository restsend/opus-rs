//! ec_laplace_encode and ec_laplace_decode (celt/laplace.c) compute in
//! unsigned 32-bit arithmetic. They return for every value and input with
//! fs from 1 to 32736 and decay from 0 to 16384, and the codec calls them
//! with fs at most 32640. opus-rs's public RangeCoder codes those arguments
//! as libopus does and flags an error for the others, where libopus's
//! coder can loop forever.
use opus_rs::range_coder::RangeCoder;
use opusref::*;
use std::sync::mpsc;
use std::time::Duration;

/// opus-rs's two encodes into 64 bytes, on a thread, so that a loop that
/// never ends fails the test.
fn encode(v: i32, fs: u32, decay: i32) -> (i32, Vec<u8>) {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut x = v;
        let mut rc = RangeCoder::new_encoder(64);
        rc.laplace_encode(&mut x, fs, decay);
        rc.laplace_encode(&mut x.clone(), fs, decay);
        let mut buf = rc.finish();
        buf.resize(64, 0);
        let _ = tx.send((x, buf));
    });
    rx.recv_timeout(Duration::from_secs(10))
        .unwrap_or_else(|_| panic!("laplace_encode({v}, {fs}, {decay}) does not return"))
}

fn check(v: i32, fs: u32, decay: i32) {
    let (a, ra) = encode(v, fs, decay);
    let (mut b, mut buf) = (v, vec![0u8; 64]);
    unsafe { ref_laplace_encode(&mut b, fs, decay, buf.as_mut_ptr(), 64) };
    assert_eq!((a, &ra), (b, &buf), "value and bytes of {v} at fs {fs}, decay {decay}");
    let da = RangeCoder::new_decoder(&buf).laplace_decode(fs, decay);
    let db = unsafe { ref_laplace_decode(fs, decay, buf.as_mut_ptr(), 64) };
    assert_eq!(da, db, "decoded value at fs {fs}, decay {decay}");
}

#[test]
#[cfg_attr(target_os = "wasi", ignore = "WASI has no threads")]
fn laplace_codes_like_libopus() {
    for fs in [1, 2, 100, 8000, 16000, 24000, 32640, 32700, 32735, 32736] {
        for decay in [0, 1, 100, 6000, 11456, 16383, 16384] {
            for v in -40..=40 {
                check(v, fs, decay);
            }
        }
    }
}

/// The arguments RangeCoder refuses, among them calls where libopus's
/// coder never returns: the error flag is set and nothing is coded.
#[test]
#[cfg_attr(target_os = "wasi", ignore = "WASI has no threads")]
fn laplace_refuses_arguments_libopus_loops_on() {
    let libopus_loops = [
        (-1, 32737, 1), (1, 32737, 1), (-1, 32737, 100), (1, 32737, 100),
        (-1, 32737, 16383), (1, 32737, 16383), (-1, 32737, -5), (1, 32737, -5),
        (-1, 32750, 16383), (1, 32750, 16383), (-1, 32767, 16383), (1, 32767, 16383),
        (-1, 32768, 16383), (1, 32768, 16383), (-1, 33000, 16383), (1, 33000, 16383),
    ];
    let others = [(0, 0, 1), (5, 0, 6000), (5, 100, -1), (5, 100, 16385), (-3, 32768, 1), (5, u32::MAX, i32::MIN)];
    for (v, fs, decay) in libopus_loops.into_iter().chain(others) {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut rc = RangeCoder::new_encoder(64);
            let mut x = v;
            rc.laplace_encode(&mut x, fs, decay);
            let mut d = RangeCoder::new_decoder(&[0x5a; 16]);
            let dv = d.laplace_decode(fs, decay);
            let _ = tx.send((x, rc.error, rc.tell(), dv, d.error, d.tell()));
        });
        let r = rx
            .recv_timeout(Duration::from_secs(10))
            .unwrap_or_else(|_| panic!("laplace coding of {v} at fs {fs}, decay {decay} does not return"));
        assert_eq!(r, (v, 1, 1, 0, 1, 1), "value {v} at fs {fs}, decay {decay}");
    }
}
