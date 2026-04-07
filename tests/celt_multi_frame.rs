use opus_rs::celt::{CeltDecoder, CeltEncoder};
use opus_rs::modes::default_mode;
use opus_rs::range_coder::RangeCoder;

#[test]
fn test_celt_multi_frame() {
    let mode = default_mode();
    let frame_size = 960;
    let channels = 1;

    // Create encoder and decoder (reused across frames)
    let mut encoder = CeltEncoder::new(mode, channels);
    let mut decoder = CeltDecoder::new(mode, channels);

    for frame in 0..5 {
        // Create sine wave for this frame
        let mut pcm: Vec<f32> = Vec::with_capacity(frame_size);
        for i in 0..frame_size {
            let t = (frame * frame_size + i) as f64 / 48000.0;
            let sample = f64::sin(2.0 * std::f64::consts::PI * 1000.0 * t) as f32 * 0.5;
            pcm.push(sample);
        }

        // Encode
        let mut rc_enc = RangeCoder::new_encoder(1000);
        encoder.encode(&pcm, frame_size, &mut rc_enc);
        let encoded = rc_enc.finish();

        // Decode
        let mut output = vec![0.0f32; frame_size];
        decoder.decode(&encoded, frame_size, &mut output);

        // Check
        let input_rms = pcm.iter().map(|&x| x * x).sum::<f32>().sqrt() / frame_size as f32;
        let output_rms = output.iter().map(|&x| x * x).sum::<f32>().sqrt() / frame_size as f32;
        let ratio = output_rms / input_rms.max(1e-10);

        println!(
            "Frame {}: input_rms={:.6}, output_rms={:.6}, ratio={:.4}",
            frame, input_rms, output_rms, ratio
        );

        // Print first few samples
        if frame < 2 {
            println!("  Input:  {:?}", &pcm[..5]);
            println!("  Output: {:?}", &output[..5]);
        }
    }
}
