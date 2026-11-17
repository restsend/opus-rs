//! A Hybrid frame after a SILK-to-CELT redundant frame that turned the
//! postfilter on: libopus keeps CELT's postfilter period and gain on frames
//! from band 17 (it only skips parsing new ones), so the first short block
//! fades the old postfilter out. opus-rs must do the same.
use opus_rs::OpusDecoder;
use opusref::*;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn hybrid_frame_keeps_the_postfilter_of_a_redundant_frame() {
    // libopus Hybrid FB 20 ms packets; the fourth one's tail decodes a
    // SILK-to-CELT redundant frame with the postfilter on.
    let pkts = [
        "78845630cd2572fd38511e194b3b65793e921e993385a92d688e7286bddbf4fd9ddfc67a7e7dfb9ea792a4bfc335e62861f29cb7dd660b039b3d59b936407b22478194202dc55b1908569b27142b775c1dc55e4b96f7",
        "78b993af1bde65053b66431e2a6d2f08597194e0b9abf58e2ebf6c9a3596c61e65338da2e8be7ac203c2136ae3725929661fd782f8497e8c2d66f38bfe808cfa04be1391960e84e54b523a55ab6e9bacaa6fcb588abd0131e6e62e",
        "78b99faa73f550f987f761cc84207dbedbf2d686ab918cc6d3d9c1a02cc1fde8dded63e82afec221c3968ad28182ef82a56176083f6ac61e1d8fbe8b79df914c76acd99806681204ec8da97260806b3ad3730e63371837a5f1a8",
        "78b993af1bde3caae7ae43fd2082ed38709655bc16640ab6804e916cbe5a30ac49b9b6ac06c3e061b7049f9faad31277127c093894e0feff0a0c2878b79a57d046edac0b39a86628aae755d58613a59da5691aca6182eb659070da1b0d4d2c",
        "78b9a7c740e2021f58c0863013f72d6a64ba814c3e35c14d30639dd471a76705f4b6e850c1252c90a6d4b28569e35fe1cb87b905e4ac427f72d93926c53fd8649f9c5d82176a61ceb221c7972f86317c42eb538c371d666eafa0",
    ];
    let mut rs = OpusDecoder::new(48000, 1).unwrap();
    let mut err = 0;
    let c = unsafe { opus_decoder_create(48000, 1, &mut err) };
    let (mut a, mut b) = (vec![0f32; 960], vec![0f32; 960]);
    for (k, p) in pkts.iter().map(|p| unhex(p)).enumerate() {
        rs.decode(&p, 960, &mut a).unwrap();
        unsafe { opus_decode_float(c, p.as_ptr(), p.len() as i32, b.as_mut_ptr(), 960, 0) };
        let d = a
            .iter()
            .zip(&b)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(d <= 4e-6, "packet {k}: max difference {d}");
    }
    unsafe { opus_decoder_destroy(c) };
}
