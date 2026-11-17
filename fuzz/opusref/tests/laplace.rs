//! ec_laplace_encode and ec_laplace_decode (celt/laplace.c) compute in
//! unsigned 32-bit arithmetic. The codec calls them with fs at most 32640
//! and decay below 16384; opus-rs's public RangeCoder takes any arguments
//! and must code them as libopus does, also where libopus's sums wrap.
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
fn laplace_codes_like_libopus() {
    for fs in [1, 2, 100, 8000, 16000, 24000, 32640, 32700, 32735, 32736] {
        for decay in [1, 100, 6000, 11456, 16383] {
            for v in -40..=40 {
                check(v, fs, decay);
            }
        }
    }
    // Arguments outside the codec's where libopus's loops end.
    for (v, fs, decay) in [
        (-3, 32737, 1),
        (100, 32737, 16384),
        (-3, 32768, 1),
        (100, 32768, 1),
        (1, 32768, 16384),
        (100, 32768, 20000),
        (100, 32768, -5),
        (1, 65535, 16384),
        (100, 4294967295, 1),
        (100, 4294967295, 16384),
    ] {
        check(v, fs, decay);
    }
}
