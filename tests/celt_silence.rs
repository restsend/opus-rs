use opus_rs::celt::{CeltDecoder, CeltEncoder};
use opus_rs::modes::default_mode;
use opus_rs::range_coder::RangeCoder;

#[test]
fn test_celt_silence() {
    let mode = default_mode();
    let frame_size = 960;
    let channels = 1;

    // Create a proper silence packet
    // In CELT, silence is indicated by the first bit being 1 with logp=15
    let mut rc_enc = RangeCoder::new_encoder(100);
    // Encode silence bit
    rc_enc.encode_bit_logp(true, 15);
    rc_enc.done();
    let silence_packet = rc_enc.finish();

    println!(
        "Silence packet: {:?}, len={}",
        silence_packet,
        silence_packet.len()
    );

    // Decode silence packet
    let mut decoder = CeltDecoder::new(mode, channels);
    let mut output = vec![0.0f32; frame_size];
    decoder.decode(&silence_packet, frame_size, &mut output);

    let energy: f32 = output.iter().map(|&x| x * x).sum();
    println!("Decoded silence energy: {}", energy);

    // Should be very close to zero
    assert!(energy < 1.0, "Silence energy too high: {}", energy);
}
