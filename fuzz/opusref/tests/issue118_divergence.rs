//! Replays the audio-divergence inputs of issue #118: fuzzer-made packet
//! sequences with losses, `decode_fec` and mode switches that decoded
//! differently from libopus 1.6.1. opus-rs now follows libopus main, and
//! every step of every input must decode sample-identically with the
//! pinned reference.
//!
//! Input format (issue #118): byte 0 selects the rate (`& 0x0f`, index into
//! 8/12/16/24/48 kHz) and stereo (`& 0x10`); then operations: `op % 8 == 0`
//! is a loss of `[2.5, 5, 10, 20, 40, 60][(op >> 3) % 6]` ms, `op % 8 == 1`
//! is `decode_fec` with that duration, anything else decodes a packet into
//! a 120 ms buffer; a packet is preceded by its length (one byte below 128,
//! otherwise two bytes `0x80 | len >> 8`, `len & 0xff`). The files under
//! `data/issue118/` are canonicalized copies of the issue's inputs, each
//! step checked against the author's replay log.
use opus_rs::OpusDecoder;
use opusref::*;

const INPUTS: [&str; 7] = [
    "diff-celt-fb-2.5ms-quiet.bin",
    "diff-celt-swb-20ms-steady.bin",
    "diff-celt-to-silk-switch.bin",
    "diff-fec-after-hybrid.bin",
    "diff-hybrid-to-celt-switch.bin",
    "diff-loss-after-hybrid.bin",
    "diff-silk-wb-40ms-steady.bin",
];

const RATES: [usize; 5] = [8000, 12000, 16000, 24000, 48000];
const DUR_2_5MS: [usize; 6] = [1, 2, 4, 8, 16, 24];

/// Decodes one step with both decoders and requires the same sample count
/// and identical samples.
fn step(
    rs: &mut OpusDecoder,
    refd: *mut core::ffi::c_void,
    fs: usize,
    ch: usize,
    pkt: &[u8],
    out: usize,
    fec: bool,
) {
    let mut buf_rs = vec![0f32; out * ch];
    let mut buf_c = vec![0f32; out * ch];
    let got_rs = if fec {
        rs.decode_fec(pkt, out, &mut buf_rs)
    } else {
        rs.decode(pkt, out, &mut buf_rs)
    };
    let got_c = unsafe {
        opus_decode_float(
            refd,
            pkt.as_ptr(),
            pkt.len() as i32,
            buf_c.as_mut_ptr(),
            out as i32,
            fec as i32,
        )
    };
    match (got_rs, got_c) {
        // Both decoders reject the packet: agreement, like the issue's replay.
        (Err(_), nc) if nc < 0 => {}
        (Ok(nr), nc) if nc >= 0 => {
            let nc = nc as usize;
            assert_eq!(nr, nc, "{fs} Hz {ch} ch: sample counts differ");
            assert_eq!(
                &buf_rs[..nc * ch],
                &buf_c[..nc * ch],
                "{fs} Hz {ch} ch: samples differ from libopus"
            );
        }
        (rs, c) => panic!("{fs} Hz {ch} ch: the decoders disagree: rs {rs:?} c {c}"),
    }
}

/// The steps of one input: `(packet, frame size, fec)` per step, `None` for
/// a loss.
fn steps(data: &[u8], fs: usize) -> Vec<(Option<&[u8]>, usize, bool)> {
    let mut out = Vec::new();
    let mut pos = 1; // the header byte selects the rate and stereo
    while pos < data.len() {
        let op = data[pos];
        pos += 1;
        let (out_n, fec) = match op % 8 {
            0 | 1 => (DUR_2_5MS[((op >> 3) % 6) as usize] * fs / 400, op % 8 == 1),
            _ => (fs * 120 / 1000, false),
        };
        if op % 8 == 0 {
            out.push((None, out_n, false));
            continue;
        }
        let b = data[pos] as usize;
        let (len, hdr) = if b & 0x80 != 0 {
            (((b & 0x7f) << 8) | data[pos + 1] as usize, 2)
        } else {
            (b, 1)
        };
        pos += hdr;
        out.push((Some(&data[pos..pos + len]), out_n, fec));
        pos += len;
    }
    out
}

#[test]
fn issue118_inputs_decode_sample_identically_with_libopus() {
    for name in INPUTS {
        let data: &[u8] = match name {
            "diff-celt-fb-2.5ms-quiet.bin" => {
                include_bytes!("data/issue118/diff-celt-fb-2.5ms-quiet.bin")
            }
            "diff-celt-swb-20ms-steady.bin" => {
                include_bytes!("data/issue118/diff-celt-swb-20ms-steady.bin")
            }
            "diff-celt-to-silk-switch.bin" => {
                include_bytes!("data/issue118/diff-celt-to-silk-switch.bin")
            }
            "diff-fec-after-hybrid.bin" => include_bytes!("data/issue118/diff-fec-after-hybrid.bin"),
            "diff-hybrid-to-celt-switch.bin" => {
                include_bytes!("data/issue118/diff-hybrid-to-celt-switch.bin")
            }
            "diff-loss-after-hybrid.bin" => {
                include_bytes!("data/issue118/diff-loss-after-hybrid.bin")
            }
            "diff-silk-wb-40ms-steady.bin" => {
                include_bytes!("data/issue118/diff-silk-wb-40ms-steady.bin")
            }
            _ => unreachable!(),
        };
        eprintln!("replaying {name}");
        let fs = RATES[(data[0] & 0x0f) as usize % RATES.len()];
        let ch = if data[0] & 0x10 != 0 { 2 } else { 1 };
        let mut err = 0;
        let refd = unsafe { opus_decoder_create(fs as i32, ch as i32, &mut err) as *mut core::ffi::c_void };
        assert_eq!(err, 0);
        let mut rs = OpusDecoder::new(fs as i32, ch).unwrap();
        for (pkt, out, fec) in steps(data, fs) {
            let (pkt, out) = match pkt {
                Some(p) => (p, out),
                None => (&[][..], out),
            };
            step(&mut rs, refd, fs, ch, pkt, out, fec);
        }
        unsafe { opus_decoder_destroy(refd.cast()) };
    }
}
