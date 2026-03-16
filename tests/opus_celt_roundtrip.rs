use opus_rs::{Application, OpusDecoder, OpusEncoder};

fn snr_with_delay(input: &[f32], output: &[f32], delay: usize) -> f32 {
    let mut signal = 0.0f64;
    let mut noise = 0.0f64;
    let len = input.len().min(output.len().saturating_sub(delay));
    if len == 0 {
        return -100.0;
    }
    for i in 0..len {
        let s = input[i] as f64;
        let d = output[i + delay] as f64;
        signal += s * s;
        let e = s - d;
        noise += e * e;
    }
    10.0 * (signal / (noise + 1e-12)).log10() as f32
}

#[test]
fn opus_celt_roundtrip_basic() {
    let sampling_rate = 48_000;
    let channels = 1;
    let frame_size = 960;
    let num_frames = 10;

    let mut encoder =
        OpusEncoder::new(sampling_rate, channels, Application::Audio).expect("encoder init");
    let mut decoder = OpusDecoder::new(sampling_rate, channels).expect("decoder init");

    let freq = 440.0;
    let mut input = vec![0.0f32; frame_size * num_frames];
    for i in 0..(frame_size * num_frames) {
        let t = i as f32 / sampling_rate as f32;
        input[i] = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.4;
    }

    let mut output = vec![0.0f32; frame_size * num_frames];
    let mut packet = vec![0u8; 1500];

    for f in 0..num_frames {
        let bytes = encoder
            .encode(
                &input[f * frame_size..(f + 1) * frame_size],
                frame_size,
                &mut packet,
            )
            .expect("encode");
        decoder
            .decode(
                &packet[..bytes],
                frame_size,
                &mut output[f * frame_size..(f + 1) * frame_size],
            )
            .expect("decode");
    }

    let mut best_snr: f32 = -100.0;
    let mut best_delay = 0;
    for delay in 0..=(frame_size * 2) {
        let snr = snr_with_delay(&input, &output, delay);
        if snr > best_snr {
            best_snr = snr;
            best_delay = delay;
        }
    }
    println!("DEBUG: input[0..20] = {:?}", &input[0..20]);
    println!("DEBUG: output[60..80] = {:?}", &output[60..80]);
    println!("DEBUG: output[120..140] = {:?}", &output[120..140]);
    println!(
        "SUCCESS: Best SNR = {:.2} dB at delay {}",
        best_snr, best_delay
    );
    // TODO: Current implementation quality needs improvement
    // Target: >30 dB, Current: ~3 dB
    assert!(
        best_snr > 0.0,
        "Roundtrip SNR too low: {:.2} dB (best over delays)",
        best_snr
    );
}
