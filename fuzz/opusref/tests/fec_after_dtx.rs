//! decode_fec after a ToC-only (DTX) packet: libopus takes the mode of the
//! last accepted packet, ToC-only ones included, when it decides whether FEC
//! can be present. opus-rs must conceal or decode LBRR exactly when libopus does.
use opus_rs::OpusDecoder;
use opusref::*;

const FS: i32 = 16000;
const FRAME: usize = 320;

fn packets(force: i32, fec: bool, n: usize) -> Vec<Vec<u8>> {
    let mut err = 0;
    let enc = unsafe { opus_encoder_create(FS, 1, 2048, &mut err) };
    unsafe {
        opus_encoder_ctl(enc, SET_BITRATE, 24000);
        opus_encoder_ctl(enc, 11002, force); // OPUS_SET_FORCE_MODE
        opus_encoder_ctl(enc, SET_INBAND_FEC, fec as i32);
        opus_encoder_ctl(enc, SET_PACKET_LOSS_PERC, 30);
    }
    let out = (0..n)
        .map(|f| {
            let pcm: Vec<f32> = (0..FRAME)
                .map(|i| (((f * FRAME + i) as f32) * 0.07).sin() * 0.4)
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

/// Decodes `first` packets, a ToC-only packet `dtx`, then `decode_fec` of
/// `fec_pkt`, on both decoders; returns (opus-rs, libopus) final ranges.
fn run(first: &[Vec<u8>], dtx: u8, fec_pkt: &[u8]) -> (u32, u32) {
    let mut rs = OpusDecoder::new(FS, 1).unwrap();
    let mut err = 0;
    let c = unsafe { opus_decoder_create(FS, 1, &mut err) };
    let mut out = vec![0f32; FRAME];
    for p in first.iter().chain([&vec![dtx]]) {
        rs.decode(p, FRAME, &mut out).unwrap();
        unsafe {
            opus_decode_float(
                c,
                p.as_ptr(),
                p.len() as i32,
                out.as_mut_ptr(),
                FRAME as i32,
                0,
            )
        };
    }
    rs.decode_fec(fec_pkt, FRAME, &mut out).unwrap();
    unsafe {
        opus_decode_float(
            c,
            fec_pkt.as_ptr(),
            fec_pkt.len() as i32,
            out.as_mut_ptr(),
            FRAME as i32,
            1,
        )
    };
    let mut r = 0u32;
    unsafe { opus_decoder_ctl(c, GET_FINAL_RANGE, &mut r as *mut u32) };
    unsafe { opus_decoder_destroy(c) };
    (rs.final_range(), r)
}

#[test]
fn fec_after_silk_dtx_following_celt() {
    let celt = packets(1002, false, 3);
    let silk = packets(1000, true, 4);
    // ToC of a SILK WB 20 ms mono packet, without payload.
    let (rs, c) = run(&celt, silk[3][0] & 0xfc, &silk[3]);
    assert_eq!(
        rs, c,
        "libopus decodes the LBRR after a SILK ToC-only packet"
    );
}

#[test]
fn fec_after_celt_dtx_following_silk() {
    let silk = packets(1000, true, 4);
    let celt = packets(1002, false, 1);
    let (rs, c) = run(&silk[..3], celt[0][0] & 0xfc, &silk[3]);
    assert_eq!(rs, c, "libopus conceals after a CELT ToC-only packet");
}
