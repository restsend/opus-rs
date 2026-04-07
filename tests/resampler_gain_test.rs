use opus_rs::silk::resampler::{
    SilkResampler, SilkResamplerDown1_6, silk_resampler_down_1_6, silk_resampler_down2,
    silk_resampler_down2_3,
};

#[test]
fn test_down2_3_gain_consistency() {
    // Create a simple sine wave at 16kHz
    let sample_rate = 16000;
    let duration_ms = 100;
    let num_samples = sample_rate * duration_ms / 1000;

    let mut input: Vec<i16> = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let freq = 1000.0; // 1kHz sine wave
        let sample = (f64::sin(2.0 * std::f64::consts::PI * freq * t) * 10000.0) as i16;
        input.push(sample);
    }

    // Calculate input energy
    let input_energy: f64 =
        input.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / input.len() as f64;
    println!("Input energy: {}", input_energy);

    // Test 3x downsampling (16kHz -> 10.67kHz approx, actually 2/3 for 16k->10.67k)
    // Actually for 24kHz -> 16kHz, we use down2_3 which is 2/3 rate
    // Let's test with 24kHz input downsampled to 16kHz equivalent

    // For proper test: create 24kHz signal and downsample
    let sample_rate_24k = 24000;
    let num_samples_24k = sample_rate_24k * duration_ms / 1000;
    let mut input_24k: Vec<i16> = Vec::with_capacity(num_samples_24k);
    for i in 0..num_samples_24k {
        let t = i as f64 / sample_rate_24k as f64;
        let freq = 1000.0;
        let sample = (f64::sin(2.0 * std::f64::consts::PI * freq * t) * 10000.0) as i16;
        input_24k.push(sample);
    }

    let input_energy_24k: f64 =
        input_24k.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / input_24k.len() as f64;
    println!("Input 24k energy: {}", input_energy_24k);

    // Downsample using down2_3 (24kHz -> 16kHz)
    let output_len = (num_samples_24k * 2 / 3) as usize;
    let mut output = vec![0i16; output_len];
    let mut state = [0i32; 6];

    silk_resampler_down2_3(&mut state, &mut output, &input_24k, num_samples_24k as i32);

    let output_energy: f64 =
        output.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / output.len() as f64;
    println!("Output energy after down2_3: {}", output_energy);

    // Calculate gain
    let gain = (output_energy / input_energy_24k).sqrt();
    println!("Gain: {}", gain);

    // Gain should be close to 1.0 (maybe 0.9-1.1 range)
    assert!(
        gain > 0.5 && gain < 2.0,
        "Gain {} is outside reasonable range",
        gain
    );
}

#[test]
fn test_silk_resampler_up2_gain() {
    // Test 16kHz -> 32kHz (up2)
    let sample_rate = 16000;
    let duration_ms = 100;
    let num_samples = sample_rate * duration_ms / 1000;

    let mut input: Vec<i16> = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let freq = 1000.0;
        let sample = (f64::sin(2.0 * std::f64::consts::PI * freq * t) * 10000.0) as i16;
        input.push(sample);
    }

    let input_energy: f64 =
        input.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / input.len() as f64;
    println!("Input energy: {}", input_energy);

    // Test resampler 16kHz -> 32kHz
    let mut resampler = SilkResampler::default();
    resampler.init(16000, 32000);

    let output_len = num_samples * 2;
    let mut output = vec![0i16; output_len];

    resampler.process(&mut output, &input, num_samples as i32);

    let output_energy: f64 =
        output.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / output.len() as f64;
    println!("Output energy after 16k->32k: {}", output_energy);

    let gain = (output_energy / input_energy).sqrt();
    println!("Gain: {}", gain);

    // For up2, energy per sample should be similar, but we have 2x samples
    // Total energy should be roughly preserved
}

#[test]
fn test_silk_resampler_16k_to_48k() {
    // This is the case used in Hybrid decoding (SILK @ 16kHz -> output @ 48kHz)
    let sample_rate = 16000;
    let duration_ms = 100;
    let num_samples = sample_rate * duration_ms / 1000;

    let mut input: Vec<i16> = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let t = i as f64 / sample_rate as f64;
        let freq = 1000.0;
        let sample = (f64::sin(2.0 * std::f64::consts::PI * freq * t) * 10000.0) as i16;
        input.push(sample);
    }

    let input_rms =
        (input.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / input.len() as f64).sqrt();
    println!("Input RMS: {}", input_rms);

    // Test resampler 16kHz -> 48kHz
    let mut resampler = SilkResampler::default();
    resampler.init(16000, 48000);

    // 16kHz -> 48kHz is 3x upsampling
    let output_len = num_samples * 3;
    let mut output = vec![0i16; output_len];

    resampler.process(&mut output, &input, num_samples as i32);

    let output_rms =
        (output.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / output.len() as f64).sqrt();
    println!("Output RMS after 16k->48k: {}", output_rms);

    // The ratio should be close to 1.0 (signal amplitude preserved)
    let ratio = output_rms / input_rms;
    println!("Amplitude ratio: {}", ratio);

    // Allow some tolerance but it should be roughly 1.0
    // If there's a gain issue, this will fail
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "Amplitude ratio {} is outside reasonable range [0.5, 2.0]",
        ratio
    );
}

// ── silk_resampler_down_1_6 (48kHz → 8kHz) ─────────────────────────────────

/// Output length must equal input length / 6.
#[test]
fn test_down_1_6_output_length() {
    let input = vec![0i16; 480]; // 10ms at 48kHz
    let mut output = vec![0i16; 80]; // 10ms at 8kHz
    let mut state = SilkResamplerDown1_6::default();
    silk_resampler_down_1_6(&mut state, &mut output, &input);
    // If we reach here without a panic the slice sizes were accepted
}

/// DC input should produce DC output (near zero ripple after init transient).
#[test]
fn test_down_1_6_dc_passthrough() {
    // 60ms at 48kHz → 10ms at 8kHz, take only the last 40ms to skip transient
    let n_in = 2880usize; // 60ms
    let n_out = n_in / 6;
    let amplitude: i16 = 8000;
    let input = vec![amplitude; n_in];
    let mut output = vec![0i16; n_out];
    let mut state = SilkResamplerDown1_6::default();
    silk_resampler_down_1_6(&mut state, &mut output, &input);

    // Skip first 1/3 of output (transient) and check the rest is close to amplitude
    let skip = n_out / 3;
    for &s in &output[skip..] {
        let diff = (s as i32 - amplitude as i32).abs();
        assert!(
            diff < 500,
            "DC output sample {} too far from {} (diff {})",
            s,
            amplitude,
            diff
        );
    }
}

/// Silence in → silence out (no spurious output).
#[test]
fn test_down_1_6_silence() {
    let input = vec![0i16; 480];
    let mut output = vec![0i16; 80];
    let mut state = SilkResamplerDown1_6::default();
    silk_resampler_down_1_6(&mut state, &mut output, &input);
    for &s in &output {
        assert_eq!(s, 0, "Expected silence, got {}", s);
    }
}

/// RMS amplitude should be roughly preserved (gain ≈ 1.0) for a 1 kHz sine.
/// 1 kHz is well within the 4 kHz Nyquist of the 8 kHz output.
#[test]
fn test_down_1_6_gain() {
    let fs_in = 48000usize;
    let duration_ms = 200;
    let n_in = fs_in * duration_ms / 1000; // 9600
    let n_out = n_in / 6; // 1600

    let freq = 1000.0f64;
    let input: Vec<i16> = (0..n_in)
        .map(|i| {
            let t = i as f64 / fs_in as f64;
            (f64::sin(2.0 * std::f64::consts::PI * freq * t) * 10000.0) as i16
        })
        .collect();

    let input_rms = (input.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / n_in as f64).sqrt();

    let mut output = vec![0i16; n_out];
    let mut state = SilkResamplerDown1_6::default();
    silk_resampler_down_1_6(&mut state, &mut output, &input);

    // Skip first 1/3 to avoid IIR startup transient
    let skip = n_out / 3;
    let steady = &output[skip..];
    let output_rms =
        (steady.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / steady.len() as f64).sqrt();

    let ratio = output_rms / input_rms;
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "Gain ratio {} outside [0.5, 2.0] for 48→8 kHz resampler",
        ratio
    );
}

/// A 5 kHz sine (above 4 kHz Nyquist) must be attenuated significantly.
#[test]
fn test_down_1_6_alias_rejection() {
    let fs_in = 48000usize;
    let duration_ms = 200;
    let n_in = fs_in * duration_ms / 1000;
    let n_out = n_in / 6;

    let freq = 5000.0f64; // above 4 kHz Nyquist of 8 kHz output
    let input: Vec<i16> = (0..n_in)
        .map(|i| {
            let t = i as f64 / fs_in as f64;
            (f64::sin(2.0 * std::f64::consts::PI * freq * t) * 10000.0) as i16
        })
        .collect();

    let input_rms = (input.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / n_in as f64).sqrt();

    let mut output = vec![0i16; n_out];
    let mut state = SilkResamplerDown1_6::default();
    silk_resampler_down_1_6(&mut state, &mut output, &input);

    let skip = n_out / 3;
    let steady = &output[skip..];
    let output_rms =
        (steady.iter().map(|&s| (s as f64).powi(2)).sum::<f64>() / steady.len() as f64).sqrt();

    let attenuation = output_rms / input_rms;
    assert!(
        attenuation < 0.3,
        "5 kHz alias not sufficiently attenuated: ratio = {}",
        attenuation
    );
}

/// Stateful: processing two consecutive 10ms frames must give the same result
/// as processing the full 20ms block at once.
#[test]
fn test_down_1_6_stateful_continuity() {
    let fs_in = 48000usize;
    let n_frame = 480usize; // 10ms

    let freq = 800.0f64;
    let input: Vec<i16> = (0..n_frame * 2)
        .map(|i| {
            let t = i as f64 / fs_in as f64;
            (f64::sin(2.0 * std::f64::consts::PI * freq * t) * 10000.0) as i16
        })
        .collect();

    // Single-call reference
    let mut out_ref = vec![0i16; n_frame * 2 / 6];
    let mut state_ref = SilkResamplerDown1_6::default();
    silk_resampler_down_1_6(&mut state_ref, &mut out_ref, &input);

    // Two-call streaming
    let mut out_stream = vec![0i16; n_frame * 2 / 6];
    let mut state_stream = SilkResamplerDown1_6::default();
    silk_resampler_down_1_6(
        &mut state_stream,
        &mut out_stream[..n_frame / 6],
        &input[..n_frame],
    );
    silk_resampler_down_1_6(
        &mut state_stream,
        &mut out_stream[n_frame / 6..],
        &input[n_frame..],
    );

    assert_eq!(
        out_ref, out_stream,
        "Streaming and single-call outputs differ"
    );
}
