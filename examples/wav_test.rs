// WAV file encoder/decoder test
use opus_rs::range_coder::RangeCoder;
use opus_rs::silk::control_codec::silk_control_encoder;
use opus_rs::silk::dec_api::SilkDecoder;
use opus_rs::silk::enc_api::silk_encode;
use opus_rs::silk::init_decoder::silk_decoder_set_fs;
use opus_rs::silk::init_encoder::silk_init_encoder;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

// WAV file structures
#[derive(Debug, Clone)]
struct WavHeader {
    sample_rate: u32,
    #[allow(unused)]
    bits_per_sample: u16,
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

fn write_wav(path: &Path, header: &WavHeader, samples: &[i16]) {
    let mut file = File::create(path).expect("Failed to create WAV file");

    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;

    // RIFF header
    file.write_all(b"RIFF").unwrap();
    file.write_all(&file_size.to_le_bytes()).unwrap();
    file.write_all(b"WAVE").unwrap();

    // fmt chunk
    file.write_all(b"fmt ").unwrap();
    file.write_all(&16u32.to_le_bytes()).unwrap(); // chunk size
    file.write_all(&1u16.to_le_bytes()).unwrap(); // audio format (PCM)
    file.write_all(&header.num_channels.to_le_bytes()).unwrap();
    file.write_all(&header.sample_rate.to_le_bytes()).unwrap();
    let byte_rate = header.sample_rate * header.num_channels as u32 * 2;
    file.write_all(&byte_rate.to_le_bytes()).unwrap();
    let block_align = header.num_channels * 2;
    file.write_all(&block_align.to_le_bytes()).unwrap();
    file.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample

    // data chunk
    file.write_all(b"data").unwrap();
    file.write_all(&data_size.to_le_bytes()).unwrap();

    // Write samples
    for sample in samples {
        file.write_all(&sample.to_le_bytes()).unwrap();
    }

    println!("Wrote WAV: {} samples to {:?}", samples.len(), path);
}

fn main() {
    let input_path = Path::new("fixtures/hello_book_course_zh_16k.wav");
    let encoded_path = Path::new("fixtures/encoded.silk");
    let decoded_path = Path::new("fixtures/decoded.wav");

    // Read input WAV
    println!("\n=== Reading input WAV ===");
    let (header, samples) = read_wav(input_path);

    // Resample to 16kHz if needed
    let target_sample_rate = 16000u32;
    let sample_rate = header.sample_rate;

    let mut input_samples: Vec<i16> = if sample_rate != target_sample_rate {
        println!(
            "Warning: Sample rate {} != {}, using as-is",
            sample_rate, target_sample_rate
        );
        samples
    } else {
        samples
    };

    // Take only first 10 seconds for testing
    let max_samples = target_sample_rate as usize * 10;
    if input_samples.len() > max_samples {
        println!("Truncating to {} samples (10 seconds)", max_samples);
        input_samples.truncate(max_samples);
    }

    // Encode parameters
    let frame_size_ms = 20;
    let frame_samples = (target_sample_rate as usize) * frame_size_ms / 1000;
    let bitrate = 10000; // 10 kbps
    let complexity = 1;

    println!("\n=== Encoding ===");
    println!(
        "Frame size: {} ms ({} samples)",
        frame_size_ms, frame_samples
    );
    println!("Bitrate: {} bps", bitrate);
    println!("Complexity: {}", complexity);

    // Initialize encoder
    let mut enc_state = Default::default();
    silk_init_encoder(&mut enc_state, 0);
    silk_control_encoder(
        &mut enc_state,
        (target_sample_rate / 1000) as i32,
        frame_size_ms as i32,
        bitrate,
        complexity,
    );
    println!("Encoder frame_length = {}", enc_state.s_cmn.frame_length);
    println!("Encoder nb_subfr = {}", enc_state.s_cmn.nb_subfr);

    // Initialize decoder
    let mut dec = SilkDecoder::new();
    silk_decoder_set_fs(
        &mut dec.channel_state[0],
        (target_sample_rate / 1000) as i32,
        target_sample_rate as i32,
    );
    println!(
        "Decoder frame_length = {}",
        dec.channel_state[0].frame_length
    );
    println!("Decoder nb_subfr = {}", dec.channel_state[0].nb_subfr);

    // Encode frame by frame
    let mut all_payload: Vec<u8> = Vec::new();
    let mut frame_count = 0;

    // Encode frame by frame using silk_encode
    // Each call to silk_encode processes one packet worth of frames
    let mut sample_offset = 0;
    while sample_offset < input_samples.len() {
        let remaining = input_samples.len() - sample_offset;
        if remaining < frame_samples {
            break; // Not enough samples for a full frame
        }

        let frame_data = &input_samples[sample_offset..sample_offset + frame_samples];

        let mut rc = RangeCoder::new_encoder(1024);
        let mut n_bytes: i32 = 0;
        silk_encode(
            &mut enc_state,
            frame_data,
            frame_samples,
            &mut rc,
            &mut n_bytes,
            bitrate,
            (bitrate * frame_size_ms as i32) / 8,
            0,
            1, // activity = 1 (active speech)
        );

        rc.done();
        let payload = rc.finish();

        if !payload.is_empty() {
            // Write frame length (2 bytes) + payload
            let len = payload.len() as u16;
            all_payload.write_all(&len.to_le_bytes()).unwrap();
            all_payload.write_all(&payload).unwrap();
            frame_count += 1;
        }

        sample_offset += frame_samples;
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
    let mut pos = 0;
    let mut decoded_frames = 0;

    while pos + 2 <= all_payload.len() {
        let len = u16::from_le_bytes([all_payload[pos], all_payload[pos + 1]]) as usize;
        pos += 2;

        if pos + len > all_payload.len() {
            break;
        }

        let payload = &all_payload[pos..pos + len];
        pos += len;

        // Decode
        let mut dec_rc = RangeCoder::new_decoder(payload.to_vec());
        let mut output: Vec<i16> = vec![0; frame_samples];
        let n = dec.decode(&mut dec_rc, &mut output, 0, true, 20, 16000);

        if n > 0 {
            decoded_samples.extend_from_slice(&output[..n as usize]);
            decoded_frames += 1;
        }
    }

    println!(
        "Decoded {} frames, {} samples",
        decoded_frames,
        decoded_samples.len()
    );

    // Save decoded WAV
    let mut out_header = header.clone();
    out_header.sample_rate = target_sample_rate;
    write_wav(decoded_path, &out_header, &decoded_samples);

    // Summary and SNR calculation
    println!("\n=== Summary ===");
    println!("Input:  {} samples", input_samples.len());
    println!("Output: {} samples", decoded_samples.len());

    // Dump samples from frame 65 (active) to see the actual signal shape
    let dump_frame = 65;
    let dump_start = dump_frame * frame_samples;
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

    // Cross-correlation to find delay
    let compare_len = input_samples.len().min(decoded_samples.len());
    let active_start = 63 * frame_samples; // first active region
    let active_end = (80 * frame_samples).min(compare_len);
    let active_inp: Vec<f64> = input_samples[active_start..active_end]
        .iter()
        .map(|&s| s as f64)
        .collect();
    let active_out: Vec<f64> = decoded_samples[active_start..active_end]
        .iter()
        .map(|&s| s as f64)
        .collect();

    let max_delay = 320; // check up to 320 samples delay
    let mut best_corr = f64::NEG_INFINITY;
    let mut best_delay: i32 = 0;
    for delay in -max_delay..=max_delay {
        let mut corr = 0.0;
        let mut count = 0usize;
        for i in 0..active_inp.len() {
            let j = i as i32 + delay;
            if j >= 0 && (j as usize) < active_out.len() {
                corr += active_inp[i] * active_out[j as usize];
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
        "\nBest delay (cross-correlation): {} samples ({:.1} ms)",
        best_delay,
        best_delay as f64 / 16.0
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
    let snr_compensated = if noise_energy > 0.0 {
        10.0 * (signal_energy / noise_energy).log10()
    } else {
        999.0
    };
    println!("Delay-compensated SNR: {:.2} dB", snr_compensated);

    // Also compute gain ratio
    let mut sum_inp_sq = 0.0f64;
    let mut sum_out_sq = 0.0f64;
    for i in active_start..active_end {
        sum_inp_sq += (input_samples[i] as f64).powi(2);
        sum_out_sq += (decoded_samples[i] as f64).powi(2);
    }
    let gain_ratio = (sum_out_sq / sum_inp_sq.max(1.0)).sqrt();
    println!("Output/input gain ratio (active region): {:.3}", gain_ratio);
}
