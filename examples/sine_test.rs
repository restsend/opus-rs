// Simple sine wave test for SILK encode/decode
use opus_rs::silk::enc_api::silk_encode_frame;
use opus_rs::silk::init_encoder::silk_init_encoder;
use opus_rs::silk::control_codec::silk_control_encoder;
use opus_rs::silk::dec_api::SilkDecoder;
use opus_rs::silk::init_decoder::silk_decoder_set_fs;
use opus_rs::range_coder::RangeCoder;
use std::fs::File;
use std::io::Write;

fn main() {
    // Generate a simple sine wave test signal (440Hz, 1 second, 16kHz)
    let sample_rate = 16000u32;
    let frequency = 440u32;
    let duration_ms = 1000u32;
    let num_samples = (sample_rate * duration_ms / 1000) as usize;

    println!("Generating {} Hz sine wave, {} ms, {} samples", frequency, duration_ms, num_samples);

    let mut input_samples: Vec<i16> = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let sample = (2.0 * std::f64::consts::PI * frequency as f64 * t).sin() * 8000.0;
        input_samples.push(sample as i16);
    }

    println!("Input first 10 samples: {:?}", &input_samples[..10]);

    // Encode parameters
    let frame_size_ms = 20;
    let frame_samples = (sample_rate as usize) * frame_size_ms / 1000;
    let bitrate = 20000; // 20 kbps

    // Initialize encoder
    let mut enc_state = Default::default();
    silk_init_encoder(&mut enc_state, 0);
    silk_control_encoder(&mut enc_state, (sample_rate / 1000) as i32, frame_size_ms as i32, bitrate, 2);

    println!("\n=== Encoding ===");

    // Encode first frame only for testing
    let frame_data = &input_samples[..frame_samples];
    let mut rc = RangeCoder::new_encoder(1024);

    // Encode VAD flag (1 bit) - assume active speech
    rc.encode_bit_logp(true, 1);
    // Encode LBRR flag (1 bit) - no LBRR
    rc.encode_bit_logp(false, 1);

    let mut n_bytes: i32 = 0;
    silk_encode_frame(
        &mut enc_state,
        &frame_data,
        &mut rc,
        &mut n_bytes,
        0,
        bitrate,
        0,
    );

    rc.done();
    let payload = rc.finish();

    println!("Encoded {} bytes", payload.len());
    println!("Payload first 20 bytes: {:02x?}", &payload[..payload.len().min(20)]);

    // Decode
    println!("\n=== Decoding ===");

    let mut dec = SilkDecoder::new();
    silk_decoder_set_fs(&mut dec.channel_state[0], (sample_rate / 1000) as i32, sample_rate as i32);

    let mut dec_rc = RangeCoder::new_decoder(payload);
    let mut output: Vec<i16> = vec![0; frame_samples];
    let n = dec.decode(&mut dec_rc, &mut output, 0, true, 20, 16000);

    println!("Decoded {} samples", n);
    println!("Output first 10 samples: {:?}", &output[..10]);

    // Save output
    let mut file = File::create("sine_test.wav").unwrap();
    let data_size = (output.len() * 2) as u32;
    let file_size = 36 + data_size;

    file.write_all(b"RIFF").unwrap();
    file.write_all(&file_size.to_le_bytes()).unwrap();
    file.write_all(b"WAVE").unwrap();
    file.write_all(b"fmt ").unwrap();
    file.write_all(&16u32.to_le_bytes()).unwrap();
    file.write_all(&1u16.to_le_bytes()).unwrap();
    file.write_all(&1u16.to_le_bytes()).unwrap();
    file.write_all(&sample_rate.to_le_bytes()).unwrap();
    let byte_rate = sample_rate * 2;
    file.write_all(&byte_rate.to_le_bytes()).unwrap();
    file.write_all(&2u16.to_le_bytes()).unwrap();
    file.write_all(&16u16.to_le_bytes()).unwrap();
    file.write_all(b"data").unwrap();
    file.write_all(&data_size.to_le_bytes()).unwrap();

    for sample in &output {
        file.write_all(&sample.to_le_bytes()).unwrap();
    }

    println!("\nSaved to sine_test.wav");
}
