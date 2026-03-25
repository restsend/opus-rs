use opus_rs::{Application, OpusDecoder, OpusEncoder};

fn main() {
    println!("Opus-RS High-level API Demo");

    let channels = 1;
    let sampling_rate = 48000;
    let frame_size = 480;
    let mut encoder = OpusEncoder::new(sampling_rate, channels, Application::Audio)
        .expect("Failed to create encoder");
    let mut decoder = OpusDecoder::new(sampling_rate, channels).expect("Failed to create decoder");

    let num_frames = 20;
    let mut total_input = Vec::new();
    let mut total_output = Vec::new();

    println!(
        "Encoding {} frames ({}ms total)...",
        num_frames,
        num_frames * 10
    );

    for f in 0..num_frames {
        let mut input = vec![0.0f32; frame_size];
        let freq = 440.0;
        for i in 0..frame_size {
            let t = (f * frame_size + i) as f32 / 48000.0;
            input[i] = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5;
        }
        total_input.extend_from_slice(&input);

        let mut compressed = vec![0u8; 127];
        let bytes = encoder
            .encode(&input, frame_size, &mut compressed)
            .expect("Encoding failed");
        if f == 0 {
            println!(
                "Encoded first frame: {} bytes, TOC={:02X}",
                bytes, compressed[0]
            );
        }

        let mut output = vec![0.0f32; frame_size];
        decoder
            .decode(&compressed[..bytes], frame_size, &mut output)
            .expect("Decoding failed");
        total_output.extend_from_slice(&output);
    }

    println!("Finished encoding/decoding.");

    let mut best_snr = -100.0;
    let mut best_delay = 0;
    let overlap = 120;

    for delay in (0..frame_size + overlap).step_by(1) {
        let mut signal_pow = 0.0;
        let mut noise_pow = 0.0;
        if total_output.len() <= delay {
            continue;
        }
        let len = (total_input.len()).min(total_output.len() - delay);
        for i in 0..len {
            signal_pow += total_input[i] * total_input[i];
            let noise = total_input[i] - total_output[i + delay];
            noise_pow += noise * noise;
        }
        let snr = 10.0 * (signal_pow / (noise_pow + 1e-10)).log10();
        if snr > best_snr {
            best_snr = snr;
            best_delay = delay;
        }
    }

    println!(
        "Best SNR: {:.2} dB at delay {} samples",
        best_snr, best_delay
    );

    if best_snr < 20.0 {
        println!("Warning: Best SNR is still low!");
    }

    println!("First 10 samples of frame 2 (input vs output):");
    for i in 0..10 {
        let idx = frame_size + i;
        println!(
            "  {:>2}: input={:>10.6}, output={:>10.6}",
            i, total_input[idx], total_output[idx]
        );
    }
}
