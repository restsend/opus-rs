// Test with pulse debugging
use opus_rs::silk::enc_api::silk_encode;
use opus_rs::silk::init_encoder::silk_init_encoder;
use opus_rs::silk::control_codec::silk_control_encoder;
use opus_rs::silk::dec_api::SilkDecoder;
use opus_rs::silk::init_decoder::silk_decoder_set_fs;
use opus_rs::range_coder::RangeCoder;

fn main() {
    let sample_rate = 16000u32;
    let frame_samples = 320usize;

    let mut input: Vec<i16> = Vec::with_capacity(frame_samples);
    for i in 0..frame_samples {
        let t = i as f64 / sample_rate as f64;
        let sample = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 8000.0;
        input.push(sample as i16);
    }

    // Initialize encoder
    let mut enc_state = Default::default();
    silk_init_encoder(&mut enc_state, 0);
    silk_control_encoder(&mut enc_state, 16, 20, 20000, 2);

    // Encode
    let mut rc = RangeCoder::new_encoder(1024);
    let mut n_bytes: i32 = 0;
    silk_encode(
        &mut enc_state,
        &input,
        frame_samples,
        &mut rc,
        &mut n_bytes,
        20000,
        2500,
        0,
        1,
    );
    rc.done();
    let payload = rc.finish();

    println!("Encoded {} bytes", payload.len());
    println!("Encoder pulses (non-zero count): {}", enc_state.pulses.iter().filter(|&&x| x != 0).count());
    println!("Encoder pulses [0..20]: {:?}", &enc_state.pulses[..20]);

    // Decode
    let mut dec = SilkDecoder::new();
    silk_decoder_set_fs(&mut dec.channel_state[0], 16, 16000);

    let mut dec_rc = RangeCoder::new_decoder(payload);
    let mut output: Vec<i16> = vec![0; frame_samples];
    let n = dec.decode(&mut dec_rc, &mut output, 0, true, 20, 16000);

    println!("\nDecoded {} samples", n);
    println!("Decoder exc_q14 (non-zero count): {}", dec.channel_state[0].exc_q14.iter().filter(|&&x| x != 0).count());
    println!("Decoder exc_q14 [0..20]: {:?}", &dec.channel_state[0].exc_q14[..20]);

    println!("\nFirst 10 input:  {:?}", &input[..10]);
    println!("First 10 output: {:?}", &output[..10]);
}
