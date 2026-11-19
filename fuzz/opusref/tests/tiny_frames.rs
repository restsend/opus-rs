//! Frames of 0 or 1 byte inside a longer packet: libopus conceals each one
//! in the previous mode (opus_decode_frame takes `len <= 1` as a loss) and
//! decodes the frames around it normally. opus-rs must give the same PCM.
use opus_rs::OpusDecoder;
use opusref::*;

const FS: i32 = 48000;
const FRAME: usize = 960;

fn packets(force: i32, n: usize) -> Vec<Vec<u8>> {
    let mut err = 0;
    let enc = unsafe { opus_encoder_create(FS, 1, 2049, &mut err) };
    unsafe {
        opus_encoder_ctl(enc, SET_BITRATE, 32000);
        opus_encoder_ctl(enc, 11002, force); // OPUS_SET_FORCE_MODE
    }
    let out = (0..n)
        .map(|f| {
            let pcm: Vec<f32> = (0..FRAME)
                .map(|i| {
                    (((f * FRAME + i) as f32) * 0.031).sin() * 0.3
                        + (((f * FRAME + i) as f32) * 0.0071).sin() * 0.2
                })
                .collect();
            let mut b = [0u8; 1500];
            let len =
                unsafe { opus_encode_float(enc, pcm.as_ptr(), FRAME as i32, b.as_mut_ptr(), 1500) };
            b[..len as usize].to_vec()
        })
        .collect();
    unsafe { opus_encoder_destroy(enc) };
    out
}

/// A code-2 packet: a `tiny`-byte first frame, then the payload of `p`.
fn with_tiny_first(p: &[u8], tiny: usize) -> Vec<u8> {
    let mut out = vec![(p[0] & 0xfc) | 2, tiny as u8];
    out.extend_from_slice(&p[1..1 + tiny]);
    out.extend_from_slice(&p[1..]);
    out
}

/// Decodes `pkts` on both decoders; returns the largest PCM difference.
fn max_diff(pkts: &[Vec<u8>]) -> f32 {
    let mut rs = OpusDecoder::new(FS, 1).unwrap();
    let mut err = 0;
    let c = unsafe { opus_decoder_create(FS, 1, &mut err) };
    let (mut a, mut b) = (vec![0f32; 5760], vec![0f32; 5760]);
    let mut worst = 0f32;
    for p in pkts {
        let n = rs.decode(p, 5760, &mut a).unwrap();
        let m =
            unsafe { opus_decode_float(c, p.as_ptr(), p.len() as i32, b.as_mut_ptr(), 5760, 0) };
        assert_eq!(n as i32, m);
        for i in 0..n {
            worst = worst.max((a[i] - b[i]).abs());
        }
    }
    unsafe { opus_decoder_destroy(c) };
    worst
}

fn check(force: i32) {
    let p = packets(force, 6);
    for tiny in [0, 1] {
        let mut seq = p[..3].to_vec();
        seq.push(with_tiny_first(&p[3], tiny));
        seq.extend_from_slice(&p[4..]);
        let d = max_diff(&seq);
        assert!(
            d <= 4e-6,
            "mode {force}, {tiny}-byte frame: max difference {d}"
        );
    }
}

#[test]
fn tiny_frame_in_a_silk_packet() {
    check(1000);
}

#[test]
fn tiny_frame_in_a_hybrid_packet() {
    check(1001);
}
