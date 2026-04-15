use opus_rs::{Application, OpusDecoder, OpusEncoder};

#[test]
fn test_48k_audio_roundtrip() {
    let sample_rate = 48000;
    let frame_size = 960;

    let mut enc = OpusEncoder::new(sample_rate as i32, 1, Application::Audio).unwrap();
    enc.bitrate_bps = 64_000;
    enc.complexity = 0;
    let mut dec = OpusDecoder::new(sample_rate as i32, 1).unwrap();

    let mut rng: u32 = 12345;
    let input: Vec<f32> = (0..frame_size)
        .map(|_| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            ((rng >> 16) as i16) as f32 / 32768.0 * 0.3
        })
        .collect();

    let mut output_buf = vec![0u8; 1024];
    let mut pcm_out = vec![0.0f32; frame_size];

    for frame in 0..20 {
        let len = enc.encode(&input, frame_size, &mut output_buf).unwrap();
        eprintln!("Frame {}: {} bytes", frame, len);
        dec.decode(&output_buf[..len], frame_size, &mut pcm_out)
            .unwrap();
    }
}
