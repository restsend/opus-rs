//! A Hybrid frame whose redundancy size runs past the packet: libopus's
//! sanity check sets the CELT length to 0, and celt_decode_with_ec conceals
//! a CELT layer of at most one byte. opus-rs must conceal it the same way.
use opus_rs::OpusDecoder;
use opusref::*;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn hybrid_frame_with_an_empty_celt_layer() {
    // Three libopus Hybrid FB 20 ms packets of noise, then one whose tail
    // decodes a CELT-to-SILK redundancy of 139 bytes in an 81-byte packet.
    let pkts = [
        "7884c1b0c7b173d93ba8008bf29a6c5a9d25eb530a864fa99ad5f705067b22bf5a95b5a8c9d1c92162b0836e86b014524598e5631d23524d42fe46d46add571c18c842c50ad69b270b675dc55e4b97c1",
        "789741667c025e42ea67f8f93e1bd2eb96a5161d45403c4affbac295634813649b0513eb40ec6c3f85ed9471fc1f98bbd420fe9c7c49222aa716b3cba7be701a88d0e05364dd6cc97e8e80f9a1552cb3af1ea4",
        "789741379e83b37251dc9ddec65fda1c35579e5fb04059f23693dd8fcf0d4de8489a7fe9bebebbae8e2d9d17a30578de1b2044bb69d462c15ed30fed1e0bae0fd9e2aa2b23a07e7165e330d2bded499d",
        "789748ac748a0e9cb91166abcf63e781a4f33dd110d3b8d564145b007eabf1295778fe5f311f6bb0c6ff297dde176ab8b67e70f4f66bdc051aabcf5831d7984a52a8bc173e238938ea9898360e9ce05ee2",
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
