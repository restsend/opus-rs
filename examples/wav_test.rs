// WAV file encoder/decoder test using OpusEncoder/OpusDecoder
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

// WAV file structures
#[derive(Debug, Clone)]
struct WavHeader {
    sample_rate: u32,
    #[allow(unused)]
    bits_per_sample: u16,
    #[allow(unused)]
    num_channels: u16,
    #[allow(unused)]
    data_size: u32,
}

fn read_wav(path: &Path) -> (WavHeader, Vec<i16>) {
    let mut file = File::open(path).expect("Failed to open WAV file");

    // Read RIFF header
    let mut riff = [0u8; 12];
    file.read_exact(&mut riff).expect("Failed to read RIFF header");
    assert!(&riff[0..4] == b"RIFF", "Not a valid WAV file");
    assert!(&riff[8..12] == b"WAVE", "Not a valid WAV file");

    // Read fmt chunk
    let mut fmt = [0u8; 24];
    file.read_exact(&mut fmt).expect("Failed to read fmt chunk");
    assert!(&fmt[0..4] == b"fmt ", "Invalid fmt chunk");

    let audio_format = u16::from_le_bytes([fmt[8], fmt[9]]);
    assert!(
        audio_format == 1 || audio_format == 3,
        "Only PCM format supported"
    );

    let num_channels = u16::from_le_bytes([fmt[10], fmt[11]]);
    let sample_rate = u32::from_le_bytes([fmt[12], fmt[13], fmt[14], fmt[15]]);
    let bits_per_sample = u16::from_le_bytes([fmt[22], fmt[23]]);

    // Read data chunk
    let mut data_header = [0u8; 8];
    file.read_exact(&mut data_header).expect("Failed to read data header");
    assert!(&data_header[0..4] == b"data", "Invalid data chunk");

    let data_size = u32::from_le_bytes([
        data_header[4],
        data_header[5],
        data_header[6],
        data_header[7],
    ]);

    // Read audio data
    let mut data = vec![0u8; data_size as usize];
    file.read_exact(&mut data).expect("Failed to read data");

    // Convert to i16 samples (mono)
    let mut samples: Vec<i16> = Vec::new();
    if bits_per_sample == 16 {
        for chunk in data.chunks(2 * num_channels as usize) {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            samples.push(sample);
        }
    } else if bits_per_sample == 8 {
        for chunk in data.chunks(num_channels as usize) {
            let sample = (chunk[0] as i16) - 128;
            samples.push(sample);
        }
    }

    let header = WavHeader {
        sample_rate,
        bits_per_sample,
        num_channels,
        data_size,
    };

    println!(
        "Loaded WAV: {} Hz, {} bit, {} channels, {} samples",
        sample_rate,
        bits_per_sample,
        num_channels,
        samples.len()
    );

    (header, samples)
}

fn write_wav(path: &Path, sample_rate: u32, num_channels: u16, samples: &[i16]) {
    let mut file = File::create(path).expect("Failed to create WAV file");

    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;

    // RIFF header
    file.write_all(b"RIFF").unwrap();
    file.write_all(&file_size.to_le_bytes()).unwrap();
    file.write_all(b"WAVE").unwrap();

    // fmt chunk
    file.write_all(b"fmt ").unwrap();
    file.write_all(&16u32.to_le_bytes()).unwrap();
    file.write_all(&1u16.to_le_bytes()).unwrap();
    file.write_all(&num_channels.to_le_bytes()).unwrap();
    file.write_all(&sample_rate.to_le_bytes()).unwrap();
    let byte_rate = sample_rate * num_channels as u32 * 2;
    file.write_all(&byte_rate.to_le_bytes()).unwrap();
    let block_align = num_channels * 2;
    file.write_all(&block_align.to_le_bytes()).unwrap();
    file.write_all(&16u16.to_le_bytes()).unwrap();

    // data chunk
    file.write_all(b"data").unwrap();
    file.write_all(&data_size.to_le_bytes()).unwrap();

    // Write samples
    for sample in samples {
        file.write_all(&sample.to_le_bytes()).unwrap();
    }

    println!("Wrote WAV: {} samples to {:?}", samples.len(), path);
}

// Simple linear interpolation resampler
// Converts samples from src_rate to dst_rate
fn resample_linear(samples: &[i16], src_rate: u32, dst_rate: u32) -> Vec<i16> {
    if src_rate == dst_rate {
        return samples.to_vec();
    }

    // Calculate the ratio
    let ratio = src_rate as f64 / dst_rate as f64;
    let dst_len = (samples.len() as f64 / ratio).round() as usize;

    let mut resampled = Vec::with_capacity(dst_len);

    for i in 0..dst_len {
        let src_pos = i as f64 * ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;

        if src_idx >= samples.len() {
            break;
        }

        let sample = if src_idx + 1 < samples.len() {
            let s0 = samples[src_idx] as f64;
            let s1 = samples[src_idx + 1] as f64;
            (s0 + (s1 - s0) * frac).round() as i16
        } else {
            samples[src_idx]
        };

        resampled.push(sample);
    }

    resampled
}

fn main() {
    let input_arg = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "fixtures/answer_16k.wav".to_string());
    let input_path = Path::new(&input_arg);
    let encoded_path = Path::new("fixtures/encoded.opus");
    let decoded_path = Path::new("fixtures/decoded.wav");

    println!("Input file: {:?}", input_path);

    // Read input WAV
    println!("\n=== Reading input WAV ===");
    let (header, samples) = read_wav(input_path);

    // Determine target sample rate for Opus (must be 8000/12000/16000/24000/48000)
    let src_rate = header.sample_rate;
    let target_rate: u32 = if [8000, 12000, 16000, 24000, 48000].contains(&src_rate) {
        src_rate
    } else {
        // For arbitrary rates, resample to 16kHz (good for speech)
        println!(
            "Note: Sample rate {} not natively supported by Opus, will resample to 16000 Hz",
            src_rate
        );
        16000
    };

    // Resample if necessary
    let input_samples: Vec<i16> = if src_rate != target_rate {
        resample_linear(&samples, src_rate, target_rate)
    } else {
        samples.clone()
    };

    // Take only first 10 seconds for testing
    let max_samples = target_rate as usize * 10;
    let input_samples = if input_samples.len() > max_samples {
        println!("Truncating to {} samples (10 seconds)", max_samples);
        &input_samples[..max_samples]
    } else {
        &input_samples[..]
    };

    // Convert i16 to f32 for OpusEncoder (normalized to [-1, 1])
    let frame_size = (target_rate as usize) * 20 / 1000; // 20ms frame
    let bitrate = 20000; // 20 kbps - better quality than 10kbps

    println!("\n=== Encoding ===");
    println!("Target sample rate: {} Hz", target_rate);
    println!("Frame size: {} samples (20ms)", frame_size);
    println!("Bitrate: {} bps", bitrate);

    // Initialize encoder
    let mut encoder = OpusEncoder::new(target_rate as i32, 1, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = bitrate;
    encoder.use_cbr = true;
    encoder.complexity = 5; // Medium complexity for better quality

    // Initialize decoder
    let mut decoder = OpusDecoder::new(target_rate as i32, 1)
        .expect("Failed to create decoder");

    // Encode frame by frame
    let mut all_payload: Vec<u8> = Vec::new();
    let mut frame_count = 0usize;

    let mut sample_offset = 0usize;
    while sample_offset + frame_size <= input_samples.len() {
        // Convert i16 to f32
        let frame: Vec<f32> = input_samples[sample_offset..sample_offset + frame_size]
            .iter()
            .map(|&s| s as f32 / 32768.0)
            .collect();

        let mut encoded = vec![0u8; 512];
        let len = encoder.encode(&frame, frame_size, &mut encoded).unwrap();
        encoded.truncate(len);

        // Store as [len:u16][payload...]
        let len_u16 = len as u16;
        all_payload.write_all(&len_u16.to_le_bytes()).unwrap();
        all_payload.write_all(&encoded).unwrap();

        frame_count += 1;
        sample_offset += frame_size;
    }

    println!(
        "Encoded {} frames, {} bytes",
        frame_count,
        all_payload.len()
    );

    // Save encoded data
    std::fs::write(encoded_path, &all_payload).expect("Failed to write encoded file");
    println!("Saved encoded data to {:?}", encoded_path);

    // Decode
    println!("\n=== Decoding ===");

    let mut decoded_samples: Vec<i16> = Vec::new();
    let mut pos = 0usize;
    let mut decoded_frames = 0usize;

    while pos + 2 <= all_payload.len() {
        let len = u16::from_le_bytes([all_payload[pos], all_payload[pos + 1]]) as usize;
        pos += 2;

        if pos + len > all_payload.len() {
            break;
        }

        let payload = &all_payload[pos..pos + len];
        pos += len;

        let mut output = vec![0.0f32; frame_size];
        let n = decoder.decode(payload, frame_size, &mut output).unwrap();

        // Convert f32 back to i16
        for &s in &output[..n] {
            decoded_samples.push((s * 32768.0).clamp(-32768.0, 32767.0) as i16);
        }

        decoded_frames += 1;
    }

    println!(
        "Decoded {} frames, {} samples ({:.1} s)",
        decoded_frames,
        decoded_samples.len(),
        decoded_samples.len() as f64 / target_rate as f64
    );

    // SNR calculation with cross-correlation delay search (BEFORE resampling)
    let compare_len = input_samples.len().min(decoded_samples.len());
    let active_start = 63 * frame_size;
    let active_end = (80 * frame_size).min(compare_len);

    let mut best_corr = f64::NEG_INFINITY;
    let mut best_delay = 0i32;
    let max_delay = 320i32;

    for delay in -max_delay..=max_delay {
        let mut corr = 0.0f64;
        let mut count = 0usize;
        for i in active_start..active_end {
            let j = i as i32 + delay;
            if j >= 0 && (j as usize) < decoded_samples.len() {
                corr += input_samples[i] as f64 * decoded_samples[j as usize] as f64;
                count += 1;
            }
        }
        if count > 0 {
            corr /= count as f64;
        }
        if corr > best_corr {
            best_corr = corr;
            best_delay = delay;
        }
    }

    println!(
        "Best delay (cross-correlation): {} samples ({:.1} ms)",
        best_delay,
        best_delay as f64 * 1000.0 / target_rate as f64
    );

    // Compute delay-compensated SNR
    let delay = best_delay;
    let mut signal_energy = 0.0f64;
    let mut noise_energy = 0.0f64;

    for i in 0..compare_len {
        let j = i as i32 + delay;
        if j >= 0 && (j as usize) < compare_len {
            let s = input_samples[i] as f64;
            let d = decoded_samples[j as usize] as f64;
            let err = d - s;
            signal_energy += s * s;
            noise_energy += err * err;
        }
    }

    let snr = if noise_energy > 0.0 {
        10.0 * (signal_energy / noise_energy).log10()
    } else {
        999.0
    };
    println!("Delay-compensated SNR: {:.2} dB", snr);

    // Sample dump for frame 65
    let dump_frame = 65usize;
    let dump_start = dump_frame * frame_size;
    if dump_start + 20 <= input_samples.len().min(decoded_samples.len()) {
        println!("\nFrame {} sample dump (first 20 samples):", dump_frame);
        println!("  i  |  input  |  output  |  error");
        for i in 0..20 {
            let idx = dump_start + i;
            let inp = input_samples[idx];
            let out = decoded_samples[idx];
            println!(
                "  {:3} | {:6} | {:6} | {:6}",
                i,
                inp,
                out,
                out as i32 - inp as i32
            );
        }
    }

    // If we resampled, convert decoded output back to original rate
    let final_samples: Vec<i16> = if src_rate != target_rate {
        resample_linear(&decoded_samples, target_rate, src_rate)
    } else {
        decoded_samples
    };

    // Save decoded WAV
    write_wav(decoded_path, src_rate, 1, &final_samples);

    // Summary
    println!("\n=== Summary ===");
    println!("Input:  {} samples ({} Hz)", input_samples.len(), target_rate);
    println!("Output: {} samples ({} Hz)", final_samples.len(), src_rate);
}
