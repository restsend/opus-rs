/// wav_test_c: encode/decode a WAV file using the raw libopus C API (opusic-sys)
/// Outputs decoded_c.wav and compares against decoded.wav (Rust implementation)
use std::ffi::CStr;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

// ── WAV helpers ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct WavHeader {
    sample_rate: u32,
    bits_per_sample: u16,
    num_channels: u16,
    data_size: u32,
}

fn read_wav(path: &Path) -> (WavHeader, Vec<i16>) {
    let mut file = File::open(path).unwrap_or_else(|e| panic!("Cannot open {:?}: {}", path, e));

    let mut riff = [0u8; 12];
    file.read_exact(&mut riff).expect("read RIFF header");
    assert!(&riff[0..4] == b"RIFF", "not a RIFF file");
    assert!(&riff[8..12] == b"WAVE", "not a WAVE file");

    let mut fmt = [0u8; 24];
    file.read_exact(&mut fmt).expect("read fmt chunk");
    assert!(&fmt[0..4] == b"fmt ", "missing fmt chunk");

    let audio_format = u16::from_le_bytes([fmt[8], fmt[9]]);
    assert!(audio_format == 1 || audio_format == 3, "only PCM supported");

    let num_channels = u16::from_le_bytes([fmt[10], fmt[11]]);
    let sample_rate = u32::from_le_bytes([fmt[12], fmt[13], fmt[14], fmt[15]]);
    let bits_per_sample = u16::from_le_bytes([fmt[22], fmt[23]]);

    let mut data_header = [0u8; 8];
    file.read_exact(&mut data_header).expect("read data header");
    assert!(&data_header[0..4] == b"data", "missing data chunk");
    let data_size = u32::from_le_bytes([
        data_header[4],
        data_header[5],
        data_header[6],
        data_header[7],
    ]);

    let mut raw = vec![0u8; data_size as usize];
    file.read_exact(&mut raw).expect("read PCM data");

    let mut samples: Vec<i16> = Vec::with_capacity(raw.len() / 2);
    if bits_per_sample == 16 {
        for chunk in raw.chunks(2 * num_channels as usize) {
            samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
    } else if bits_per_sample == 8 {
        for chunk in raw.chunks(num_channels as usize) {
            samples.push((chunk[0] as i16) - 128);
        }
    }

    println!(
        "[wav] loaded {:?}: {} Hz, {} bit, {} ch, {} samples",
        path,
        sample_rate,
        bits_per_sample,
        num_channels,
        samples.len()
    );

    let header = WavHeader {
        sample_rate,
        bits_per_sample,
        num_channels,
        data_size,
    };
    (header, samples)
}

fn write_wav(path: &Path, sample_rate: u32, num_channels: u16, samples: &[i16]) {
    let mut file = File::create(path).unwrap_or_else(|e| panic!("Cannot create {:?}: {}", path, e));

    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;
    let byte_rate = sample_rate * num_channels as u32 * 2;
    let block_align = num_channels * 2;

    file.write_all(b"RIFF").unwrap();
    file.write_all(&file_size.to_le_bytes()).unwrap();
    file.write_all(b"WAVE").unwrap();

    file.write_all(b"fmt ").unwrap();
    file.write_all(&16u32.to_le_bytes()).unwrap();
    file.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
    file.write_all(&num_channels.to_le_bytes()).unwrap();
    file.write_all(&sample_rate.to_le_bytes()).unwrap();
    file.write_all(&byte_rate.to_le_bytes()).unwrap();
    file.write_all(&block_align.to_le_bytes()).unwrap();
    file.write_all(&16u16.to_le_bytes()).unwrap();

    file.write_all(b"data").unwrap();
    file.write_all(&data_size.to_le_bytes()).unwrap();
    for s in samples {
        file.write_all(&s.to_le_bytes()).unwrap();
    }

    println!(
        "[wav] wrote {:?}: {} samples ({:.1} s)",
        path,
        samples.len(),
        samples.len() as f64 / sample_rate as f64
    );
}

// ── Main ───────────────────────────────────────────────────────────────────────

fn main() {
    let input_path = Path::new("fixtures/hello_book_course_zh_16k.wav");
    let encoded_path = Path::new("fixtures/encoded_c.opus");
    let decoded_c_path = Path::new("fixtures/decoded_c.wav");
    let decoded_rs_path = Path::new("fixtures/decoded.wav");

    // ── print libopus version ──────────────────────────────────────────────────
    let version_str = unsafe {
        CStr::from_ptr(opusic_sys::opus_get_version_string())
            .to_str()
            .unwrap_or("unknown")
    };
    println!("libopus version: {}", version_str);

    // ── read input WAV ─────────────────────────────────────────────────────────
    println!("\n=== Reading input WAV ===");
    let (header, samples) = read_wav(input_path);

    const SAMPLE_RATE: i32 = 16_000;
    const CHANNELS: i32 = 1;
    const FRAME_MS: usize = 20;
    const FRAME_SAMPLES: usize = (SAMPLE_RATE as usize) * FRAME_MS / 1000; // 320
    const BITRATE: i32 = 10_000; // 10 kbps – matches wav_test.rs

    assert_eq!(
        header.sample_rate, SAMPLE_RATE as u32,
        "input WAV must be 16 kHz"
    );

    // truncate to 10 s
    let mut input_samples: Vec<i16> = samples;
    let max_samples = SAMPLE_RATE as usize * 10;
    if input_samples.len() > max_samples {
        println!("Truncating to {} samples (10 s)", max_samples);
        input_samples.truncate(max_samples);
    }

    // ── create Opus encoder ────────────────────────────────────────────────────
    println!("\n=== Encoding (libopus C) ===");
    println!("Frame size: {} ms ({} samples)", FRAME_MS, FRAME_SAMPLES);
    println!("Bitrate: {} bps", BITRATE);

    let enc = unsafe {
        let mut err: i32 = 0;
        let p = opusic_sys::opus_encoder_create(
            SAMPLE_RATE,
            CHANNELS,
            opusic_sys::OPUS_APPLICATION_VOIP,
            &mut err,
        );
        assert_eq!(
            err,
            opusic_sys::OPUS_OK,
            "opus_encoder_create failed: {}",
            err
        );

        // Force CBR + SILK-only bandwidth to match Rust implementation
        opusic_sys::opus_encoder_ctl(p, opusic_sys::OPUS_SET_BITRATE_REQUEST, BITRATE as i32);
        opusic_sys::opus_encoder_ctl(p, opusic_sys::OPUS_SET_VBR_REQUEST, 0i32); // CBR
        opusic_sys::opus_encoder_ctl(
            p,
            opusic_sys::OPUS_SET_BANDWIDTH_REQUEST,
            opusic_sys::OPUS_BANDWIDTH_WIDEBAND as i32,
        );
        opusic_sys::opus_encoder_ctl(
            p,
            opusic_sys::OPUS_SET_SIGNAL_REQUEST,
            opusic_sys::OPUS_SIGNAL_VOICE as i32,
        );
        opusic_sys::opus_encoder_ctl(p, opusic_sys::OPUS_SET_COMPLEXITY_REQUEST, 1i32);
        p
    };

    // encode frame-by-frame; store as [len:u16][payload…]
    let mut encoded_frames: Vec<u8> = Vec::new();
    let mut frame_count = 0usize;
    let mut sample_offset = 0usize;
    let max_packet = 4000usize;

    while sample_offset + FRAME_SAMPLES <= input_samples.len() {
        let frame = &input_samples[sample_offset..sample_offset + FRAME_SAMPLES];
        let mut buf = vec![0u8; max_packet];

        let n = unsafe {
            opusic_sys::opus_encode(
                enc,
                frame.as_ptr() as *const opusic_sys::opus_int16,
                FRAME_SAMPLES as i32,
                buf.as_mut_ptr(),
                max_packet as i32,
            )
        };
        assert!(n > 0, "opus_encode error: {}", n);

        let len = n as u16;
        encoded_frames.write_all(&len.to_le_bytes()).unwrap();
        encoded_frames.write_all(&buf[..n as usize]).unwrap();
        frame_count += 1;
        sample_offset += FRAME_SAMPLES;
    }

    println!(
        "Encoded {} frames → {} bytes",
        frame_count,
        encoded_frames.len()
    );
    std::fs::write(encoded_path, &encoded_frames).expect("write encoded_c.opus");
    println!("Saved → {:?}", encoded_path);

    unsafe { opusic_sys::opus_encoder_destroy(enc) };

    // ── create Opus decoder ────────────────────────────────────────────────────
    println!("\n=== Decoding (libopus C) ===");

    let dec = unsafe {
        let mut err: i32 = 0;
        let p = opusic_sys::opus_decoder_create(SAMPLE_RATE, CHANNELS, &mut err);
        assert_eq!(
            err,
            opusic_sys::OPUS_OK,
            "opus_decoder_create failed: {}",
            err
        );
        p
    };

    let mut decoded: Vec<i16> = Vec::new();
    let mut pos = 0usize;
    let mut decoded_frame_count = 0usize;

    while pos + 2 <= encoded_frames.len() {
        let len = u16::from_le_bytes([encoded_frames[pos], encoded_frames[pos + 1]]) as usize;
        pos += 2;
        if pos + len > encoded_frames.len() {
            break;
        }
        let payload = &encoded_frames[pos..pos + len];
        pos += len;

        let mut out = vec![0i16; FRAME_SAMPLES];
        let n = unsafe {
            opusic_sys::opus_decode(
                dec,
                payload.as_ptr(),
                len as i32,
                out.as_mut_ptr() as *mut opusic_sys::opus_int16,
                FRAME_SAMPLES as i32,
                0i32, // no FEC
            )
        };
        assert!(n > 0, "opus_decode error: {}", n);
        decoded.extend_from_slice(&out[..n as usize]);
        decoded_frame_count += 1;
    }

    println!(
        "Decoded {} frames → {} samples ({:.1} s)",
        decoded_frame_count,
        decoded.len(),
        decoded.len() as f64 / SAMPLE_RATE as f64
    );

    unsafe { opusic_sys::opus_decoder_destroy(dec) };

    write_wav(decoded_c_path, SAMPLE_RATE as u32, 1, &decoded);

    // ── per-frame sample dump (frame 65, first 20 samples) ────────────────────
    let dump_frame = 65usize;
    let dump_start = dump_frame * FRAME_SAMPLES;
    if dump_start + 20 <= input_samples.len().min(decoded.len()) {
        println!("\nFrame {} sample dump (first 20):", dump_frame);
        println!("  i  |  input  | dec_c  |  error");
        for i in 0..20 {
            let idx = dump_start + i;
            let inp = input_samples[idx];
            let out = decoded[idx];
            println!(
                "  {:3} | {:6} | {:6} | {:6}",
                i,
                inp,
                out,
                out as i32 - inp as i32
            );
        }
    }

    // ── SNR vs original ────────────────────────────────────────────────────────
    println!("\n=== SNR: decoded_c vs original ===");
    print_snr_with_delay(&input_samples, &decoded, FRAME_SAMPLES, "dec_c");

    // ── compare decoded_c.wav vs decoded.wav (Rust implementation) ────────────
    if decoded_rs_path.exists() {
        println!("\n=== Comparing decoded_c.wav vs decoded.wav (Rust) ===");
        let (_, decoded_rs) = read_wav(decoded_rs_path);
        let cmp_len = decoded.len().min(decoded_rs.len());

        let mut sum_sq_diff = 0.0f64;
        let mut sum_sq_rs = 0.0f64;
        let mut sum_sq_c = 0.0f64;
        for i in 0..cmp_len {
            let c = decoded[i] as f64;
            let r = decoded_rs[i] as f64;
            sum_sq_diff += (c - r) * (c - r);
            sum_sq_rs += r * r;
            sum_sq_c += c * c;
        }
        let rms_diff = (sum_sq_diff / cmp_len as f64).sqrt();
        let rms_rs = (sum_sq_rs / cmp_len as f64).sqrt();
        let rms_c = (sum_sq_c / cmp_len as f64).sqrt();

        println!("Compared {} samples", cmp_len);
        println!("RMS  decoded (Rust): {:.2}", rms_rs);
        println!("RMS  decoded (C)   : {:.2}", rms_c);
        println!("RMS  difference    : {:.2}", rms_diff);
        if sum_sq_rs > 0.0 {
            let snr_vs_rs = 10.0 * (sum_sq_rs / sum_sq_diff.max(1e-9)).log10();
            println!("SNR dec_c vs Rust  : {:.2} dB", snr_vs_rs);
        }

        // per-frame dump comparison
        let dump_start2 = dump_frame * FRAME_SAMPLES;
        if dump_start2 + 20 <= cmp_len {
            println!("\nFrame {} comparison (first 20):", dump_frame);
            println!("  i  |  input  | dec_c  | dec_rs | Δ(c-rs)");
            for i in 0..20 {
                let idx = dump_start2 + i;
                let inp = input_samples[idx];
                let dc = decoded[idx];
                let dr = decoded_rs[idx];
                println!(
                    "  {:3} | {:6} | {:6} | {:6} | {:6}",
                    i,
                    inp,
                    dc,
                    dr,
                    dc as i32 - dr as i32
                );
            }
        }
    } else {
        println!(
            "\n[skip] {:?} not found — run `cargo run --example wav_test` first",
            decoded_rs_path
        );
    }
}

// ── helpers ────────────────────────────────────────────────────────────────────

fn print_snr_with_delay(original: &[i16], decoded: &[i16], frame_samples: usize, label: &str) {
    let compare_len = original.len().min(decoded.len());
    let active_start = 63 * frame_samples;
    let active_end = (80 * frame_samples).min(compare_len);

    // cross-correlation delay search
    let max_delay = 320i32;
    let mut best_corr = f64::NEG_INFINITY;
    let mut best_delay = 0i32;

    let win_orig: Vec<f64> = original[active_start..active_end]
        .iter()
        .map(|&s| s as f64)
        .collect();
    let win_dec: Vec<f64> = decoded[active_start..active_end]
        .iter()
        .map(|&s| s as f64)
        .collect();

    for delay in -max_delay..=max_delay {
        let mut corr = 0.0f64;
        let mut count = 0usize;
        for i in 0..win_orig.len() {
            let j = i as i32 + delay;
            if j >= 0 && (j as usize) < win_dec.len() {
                corr += win_orig[i] * win_dec[j as usize];
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
        "Best delay (cross-corr): {} samples ({:.1} ms)",
        best_delay,
        best_delay as f64 / 16.0
    );

    // delay-compensated SNR
    let delay = best_delay;
    let mut sig_e = 0.0f64;
    let mut noise_e = 0.0f64;
    for i in 0..compare_len {
        let j = i as i32 + delay;
        if j >= 0 && (j as usize) < compare_len {
            let s = original[i] as f64;
            let d = decoded[j as usize] as f64;
            sig_e += s * s;
            noise_e += (d - s) * (d - s);
        }
    }
    let snr = if noise_e > 0.0 {
        10.0 * (sig_e / noise_e).log10()
    } else {
        999.0
    };
    println!("Delay-compensated SNR [{label}]: {:.2} dB", snr);
}
