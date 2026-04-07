// WAV file encoder/decoder test using OpusEncoder/OpusDecoder
use opus_rs::silk::resampler::SilkResampler;
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
    file.read_exact(&mut riff)
        .expect("Failed to read RIFF header");
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
    file.read_exact(&mut data_header)
        .expect("Failed to read data header");
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

// High-quality resampler using SILK's resampler
// Converts samples from src_rate to dst_rate with proper anti-aliasing
fn resample_silk(samples: &[i16], src_rate: i32, dst_rate: i32) -> Vec<i16> {
    if src_rate == dst_rate {
        return samples.to_vec();
    }

    // Calculate output length
    let out_len = ((samples.len() as i64 * dst_rate as i64) / src_rate as i64) as usize;

    let mut resampler = SilkResampler::default();
    resampler.init(src_rate, dst_rate);

    let mut output = vec![0i16; out_len];
    resampler.process(&mut output, samples, samples.len() as i32);

    output
}

struct ModeConfig {
    app_name: &'static str,
    rate_name: &'static str,
    target_rate: u32,
    app_mode: Application,
    bitrate: i32,
}

fn process_mode(config: ModeConfig, src_samples: &[i16], src_rate: u32) {
    let ModeConfig {
        app_name,
        rate_name,
        target_rate,
        app_mode,
        bitrate,
    } = config;

    println!("\n{}", "=".repeat(60));
    println!("=== {} + {} ===", app_name, rate_name);
    println!("{}", "=".repeat(60));

    // Resample if necessary
    let input_samples: Vec<i16> = if src_rate != target_rate {
        println!(
            "Resampling {} Hz -> {} Hz (using SILK resampler)",
            src_rate, target_rate
        );
        resample_silk(src_samples, src_rate as i32, target_rate as i32)
    } else {
        println!("Input already at target rate {} Hz", target_rate);
        src_samples.to_vec()
    };

    let frame_size = (target_rate as usize) * 20 / 1000; // 20ms frame

    println!("\n--- Encoding ---");
    println!("Frame size: {} samples (20ms)", frame_size);
    println!("Bitrate: {} bps", bitrate);
    println!("Application mode: {:?}", app_mode);

    // Initialize encoder
    let mut encoder =
        OpusEncoder::new(target_rate as i32, 1, app_mode).expect("Failed to create encoder");
    encoder.bitrate_bps = bitrate;
    encoder.use_cbr = true;
    encoder.complexity = 5;

    // Initialize decoder
    let mut decoder = OpusDecoder::new(target_rate as i32, 1).expect("Failed to create decoder");

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

    // Decode
    println!("\n--- Decoding ---");

    let mut decoded_samples: Vec<i16> = Vec::new();
    let mut pos = 0usize;

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
    }

    println!(
        "Decoded {} frames, {} samples ({:.1} s)",
        frame_count,
        decoded_samples.len(),
        decoded_samples.len() as f64 / target_rate as f64
    );

    // SNR calculation
    let compare_len = input_samples.len().min(decoded_samples.len());
    let active_start = 63 * frame_size;
    let active_end = (80 * frame_size).min(compare_len);

    let mut best_corr = f64::NEG_INFINITY;
    let mut best_delay = 0i32;
    let max_delay = 2000i32; // Search a wide range for delay

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

    // Compute delay-compensated SNR (only over active region to exclude initialization effects)
    let delay = best_delay;
    let mut signal_energy = 0.0f64;
    let mut noise_energy = 0.0f64;

    for i in active_start..active_end {
        let j = i as i32 + delay;
        if j >= 0 && (j as usize) < decoded_samples.len() {
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
    println!("SNR: {:.2} dB (delay: {} samples)", snr, best_delay);

    // Print SNR at specific delays for debugging
    for &check_delay in &[0i32, 120, 316, 960, 1251, 1320, 1566, 1863, best_delay] {
        let mut se = 0.0f64;
        let mut ne = 0.0f64;
        let mut cnt = 0usize;
        for i in active_start..active_end {
            let j = i as i32 + check_delay;
            if j >= 0 && (j as usize) < decoded_samples.len() {
                let s = input_samples[i] as f64;
                let d = decoded_samples[j as usize] as f64;
                se += s * s;
                ne += (d - s) * (d - s);
                cnt += 1;
            }
        }
        if cnt > 0 && se > 0.0 {
            let check_snr = 10.0 * (se / ne.max(1e-10)).log10();
            println!(
                "  SNR at delay {:5}: {:.2} dB ({} samples)",
                check_delay, check_snr, cnt
            );
        }
    }

    // Save output
    let output_path = format!(
        "fixtures/decoded_{}_{}.wav",
        app_name.to_lowercase(),
        rate_name
    );
    write_wav(Path::new(&output_path), target_rate, 1, &decoded_samples);

    println!("Encoded size: {} bytes", all_payload.len());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_arg = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "fixtures/answer_16k.wav".to_string());
    let input_path = Path::new(&input_arg);

    println!("Input file: {:?}", input_path);

    // Read input WAV
    println!("\n=== Reading input WAV ===");
    let (header, samples) = read_wav(input_path);
    let src_rate = header.sample_rate;

    // Process all 4 combinations
    let modes = [
        ModeConfig {
            app_name: "voip",
            rate_name: "16k",
            target_rate: 16000,
            app_mode: Application::Voip,
            bitrate: 20000,
        },
        ModeConfig {
            app_name: "voip",
            rate_name: "48k",
            target_rate: 48000,
            app_mode: Application::Voip,
            bitrate: 32000,
        },
        ModeConfig {
            app_name: "audio",
            rate_name: "16k",
            target_rate: 16000,
            app_mode: Application::Audio,
            bitrate: 24000,
        },
        ModeConfig {
            app_name: "audio",
            rate_name: "48k",
            target_rate: 48000,
            app_mode: Application::Audio,
            bitrate: 32000,
        },
    ];

    for config in modes {
        process_mode(config, &samples, src_rate);
    }

    println!("\n=== Done ===");
    println!("Output files:");
    println!("  - fixtures/decoded_voip_16k.wav");
    println!("  - fixtures/decoded_voip_48k.wav");
    println!("  - fixtures/decoded_audio_16k.wav");
    println!("  - fixtures/decoded_audio_48k.wav");
}
