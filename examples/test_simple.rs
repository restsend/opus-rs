// Simple test: encode and decode a single frame with detailed logging
use opus_rs::silk::enc_api::silk_encode;
use opus_rs::silk::init_encoder::silk_init_encoder;
use opus_rs::silk::control_codec::silk_control_encoder;
use opus_rs::silk::dec_api::SilkDecoder;
use opus_rs::silk::init_decoder::silk_decoder_set_fs;
use opus_rs::range_coder::RangeCoder;

fn main() {
    // Generate a simple sine wave: 440Hz, 20ms, 16kHz
    let sample_rate = 16000u32;
    let frame_samples = 320usize; // 20ms at 16kHz

    let mut input: Vec<i16> = Vec::with_capacity(frame_samples);
    for i in 0..frame_samples {
        let t = i as f64 / sample_rate as f64;
        let sample = (2.0 * std::f64::consts::PI * 440.0 * t).sin() * 8000.0;
        input.push(sample as i16);
    }

    println!("Input samples [0..10]: {:?}", &input[..10]);

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

    println!("\nEncoded {} bytes", payload.len());
    println!("Payload: {:02x?}", &payload[..payload.len().min(30)]);

    // Decode
    let mut dec = SilkDecoder::new();
    silk_decoder_set_fs(&mut dec.channel_state[0], 16, 16000);

    let mut dec_rc = RangeCoder::new_decoder(payload);
    let mut output: Vec<i16> = vec![0; frame_samples];
    let n = dec.decode(&mut dec_rc, &mut output, 0, true, 20, 16000);

    println!("\nDecoded {} samples", n);
    println!("Output samples [0..10]: {:?}", &output[..10]);

    // Compare - calculate correlation
    let mut sum_xy = 0i64;
    let mut sum_x2 = 0i64;
    let mut sum_y2 = 0i64;
    for i in 0..n as usize {
        let x = input[i] as i64;
        let y = output[i] as i64;
        sum_xy += x * y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }

    let denom = ((sum_x2 as f64) * (sum_y2 as f64)).sqrt();
    let correlation = if denom > 0.0 { sum_xy as f64 / denom } else { 0.0 };

    println!("\nCorrelation: {:.4}", correlation);
    if correlation > 0.9 {
        println!("SUCCESS: Good correlation!");
    } else if correlation > 0.5 {
        println!("PARTIAL: Some correlation, but may have issues");
    } else {
        println!("FAILURE: Poor correlation - likely encoding/decoding mismatch");
    }
}
