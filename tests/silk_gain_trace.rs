/// Trace gain computation through the SILK encoder
/// Goal: understand why Rust produces gain_index=40 while C produces gain_index=15
use opus_rs::{Application, OpusEncoder};
use std::f64::consts::PI;

/// Generate the same PCM that opus_demo reads:
/// i16 = int(sin(2*pi*440*i/8000) * 16383)
/// Then converted to f32 = i16_val / 32768.0
fn gen_silk_test_pcm(n_samples: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let val_f64 = (2.0 * PI * 440.0 * i as f64 / 8000.0).sin();
            let i16_val = (val_f64 * 16383.0) as i16;
            i16_val as f32 / 32768.0
        })
        .collect()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Reference frame 0 from opus_demo
const REF_FRAME_0: &str = "0b018455a4e3c2206bd13d16c0f1d332bbe6fca978f3eac09e538202d180";

fn hex_to_bytes(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn decode_silk_gain_index(silk_data: &[u8]) -> (i32, i32, i32) {
    let mut offs = 0usize;
    let mut rng = 0x8000_ffffu32;
    let mut val =
        ((silk_data[0] as u32) << 16) | ((silk_data[1] as u32) << 8) | (silk_data[2] as u32);

    let decode_bit_logp = |rng: &mut u32, val: &mut u32, offs: &mut usize, _logp: i32| -> i32 {
        *rng = rng.wrapping_shr(1);
        let bit = if *val >= *rng {
            *val -= *rng;
            1
        } else {
            0
        };
        while *rng < 0x8000 {
            *rng = rng.wrapping_shl(8);
            *offs += 1;
            if *offs < silk_data.len() {
                *val = ((*val) << 8) | (silk_data[*offs] as u32);
            } else {
                *val = (*val) << 8;
            }
        }
        bit
    };

    // Helper to decode using iCDF
    let decode_icdf =
        |rng: &mut u32, val: &mut u32, offs: &mut usize, icdf: &[u8], ft: i32| -> i32 {
            let r = *rng / (ft as u32);
            let mut tmp = r.wrapping_mul(icdf[0] as u32);
            let mut k = 0;
            while *val < tmp {
                k += 1;
                tmp = r.wrapping_mul(icdf[k] as u32);
            }
            *val -= tmp;
            *rng = if k == 0 {
                r.wrapping_mul((icdf[0] - icdf[1]) as u32)
            } else {
                r.wrapping_mul((icdf[k - 1] - icdf[k]) as u32)
            };

            while *rng < 0x8000 {
                *rng = rng.wrapping_shl(8);
                *offs += 1;
                if *offs < silk_data.len() {
                    *val = ((*val) << 8) | (silk_data[*offs] as u32);
                } else {
                    *val = (*val) << 8;
                }
            }
            k as i32
        };

    // Decode VAD (logp=1)
    let vad = decode_bit_logp(&mut rng, &mut val, &mut offs, 1);
    println!("  Decoded VAD: {}", vad);

    // Decode LBRR (logp=1)
    let lbrr = decode_bit_logp(&mut rng, &mut val, &mut offs, 1);
    println!("  Decoded LBRR: {}", lbrr);

    // Decode type/offset (iCDF for VAD=1)
    let type_offset_icdf: &[u8] = &[224, 112, 44, 15, 3, 2, 1, 0];
    let type_offset = decode_icdf(&mut rng, &mut val, &mut offs, type_offset_icdf, 8);
    let type_offset_adj = type_offset + 2;
    let signal_type = type_offset_adj / 2;
    let quant_offset_type = type_offset_adj % 2;
    println!(
        "  Decoded type_offset: {} -> signal_type={}, quant_offset_type={}",
        type_offset, signal_type, quant_offset_type
    );

    // Decode first gain MSB
    // SILK_GAIN_ICDF for signal_type>>1 = 0 (UNVOICED): [224, 112, 44, 15, 3, 2, 1, 0]
    let gain_icdf: &[u8] = &[224, 112, 44, 15, 3, 2, 1, 0];
    let gain_msb = decode_icdf(&mut rng, &mut val, &mut offs, gain_icdf, 8);
    println!("  Decoded gain MSB: {}", gain_msb);

    // Decode first gain LSB (uniform iCDF)
    let uniform8_icdf: &[u8] = &[224, 192, 160, 128, 96, 64, 32, 0];
    let gain_lsb = decode_icdf(&mut rng, &mut val, &mut offs, uniform8_icdf, 8);
    println!("  Decoded gain LSB: {}", gain_lsb);

    let first_gain_idx = (gain_msb << 3) | gain_lsb;
    println!("  First gain index: {}", first_gain_idx);

    (vad, signal_type, first_gain_idx)
}

#[test]
fn test_trace_gain_computation() {
    let sample_rate = 8000i32;
    let channels = 1;
    let frame_size = 160usize;

    // Generate first frame's PCM
    let pcm = gen_silk_test_pcm(frame_size);

    // Create encoder with same settings as reference
    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 12000;
    encoder.use_cbr = true;
    encoder.complexity = 10;

    // Encode first frame
    let mut pkt_buf = vec![0u8; 200];
    let pkt_len = encoder
        .encode(&pcm, frame_size, &mut pkt_buf)
        .expect("Encode failed");
    let pkt = &pkt_buf[..pkt_len];

    println!("\n=== Frame 0 Gain Trace ===");
    println!("Rust packet: {}", hex_encode(pkt));
    println!("Ref packet:  {}", REF_FRAME_0);

    let ref_bytes = hex_to_bytes(REF_FRAME_0);

    // Parse both packets
    // Format: TOC (1) + count (1) + SILK data
    println!("\nRust SILK data:");
    let rust_silk = &pkt[2..];
    let (rust_vad, rust_signal_type, rust_gain_idx) = decode_silk_gain_index(rust_silk);

    println!("\nReference SILK data:");
    let ref_silk = &ref_bytes[2..];
    let (ref_vad, ref_signal_type, ref_gain_idx) = decode_silk_gain_index(ref_silk);

    println!("\n=== Comparison ===");
    println!("VAD:         rust={} ref={}", rust_vad, ref_vad);
    println!(
        "Signal type: rust={} ref={}",
        rust_signal_type, ref_signal_type
    );
    println!(
        "Gain index:  rust={} ref={} (diff: {})",
        rust_gain_idx,
        ref_gain_idx,
        rust_gain_idx - ref_gain_idx
    );

    // The gain index difference is the key issue
    if rust_gain_idx != ref_gain_idx {
        println!("\n>>> GAIN INDEX MISMATCH <<<");
        println!(
            "The gain index difference of {} levels causes all subsequent bits to differ.",
            rust_gain_idx - ref_gain_idx
        );

        // Convert gain indices back to approximate gain_q16 values
        // gain_index = silk_smulwb(SCALE_Q16, silk_lin2log(gain_q16) - OFFSET)
        // silk_lin2log(gain_q16) = OFFSET + (gain_index / SCALE_Q16) * 65536
        // OFFSET = (MIN_QGAIN_DB * 128) / 6 + 16 * 128 = 42 + 2048 = 2090
        // SCALE_Q16 = (65536 * 63) / ((86 * 128) / 6) = 4128768 / 1834 ≈ 2251

        const OFFSET: f64 = 2090.0;
        const SCALE_Q16: f64 = 2251.0;

        let rust_log = OFFSET + (rust_gain_idx as f64 / SCALE_Q16) * 65536.0;
        let ref_log = OFFSET + (ref_gain_idx as f64 / SCALE_Q16) * 65536.0;

        // log2lin approximation
        let rust_lin = 2f64.powf((rust_log / 128.0 - 16.0) / 6.0);
        let ref_lin = 2f64.powf((ref_log / 128.0 - 16.0) / 6.0);

        println!("\nApproximate gain_q16 values:");
        println!(
            "  Rust gain_q16 ≈ {:.0} (log value: {:.1})",
            rust_lin * 65536.0,
            rust_log
        );
        println!(
            "  Ref  gain_q16 ≈ {:.0} (log value: {:.1})",
            ref_lin * 65536.0,
            ref_log
        );
        println!("  Ratio: {:.2}x", rust_lin / ref_lin);
    }
}

#[test]
fn test_decode_all_reference_frames() {
    /// Reference frames from opus_demo
    const REF_FRAMES: &[&str] = &[
        "0b018455a4e3c2206bd13d16c0f1d332bbe6fca978f3eac09e538202d180",
        "0b4101ac3140db3d937238af06f7e79b1c5633a1a31b781a8b390a5fd000",
        "0b41069b2b3431aeb2ee203024f749c535df096f03f0f5ac000000000000",
        "0b41049b2b3431af52bc8194161b42a77fd294ddc3315dd6804000000000",
        "0b41089b2b3431af52bc7d809b6a10fadb98075eaeb50000000000000000",
        "0b41019b276b5871305f453d4cb0b49d2ad5b4cdbd357ea9352d1c988000",
        "0b41069b2b3431af52bc0e19c44e4561a70230a88cae5860000000000000",
        "0b41049b2b3431aeb2ee1e34a16fe019f229a13dac28fc6d27f800000000",
        "0b41089b2b3431aeb2ee2ae92eedad4354553414c43c0000000000000000",
        "0b41089b2b34328f5dad8e3708b5d032e78caf628a1c0000000000000000",
        "0b41029b276b5871305f453db8d4eb89648b7d36982708a35e422b800000",
        "0b41079b2b3431af52baf6c101c730de54ae05d373486000000000000000",
        "0b41069b2b3431af52bc723ab88758626a78c382a1d37d40000000000000",
        "0b41089b2b343290cfc624d4cf89e71ad87e27c695800000000000000000",
        "0b41059b2b3431aeb2ee2030ec0576b66f838323636982f3260000000000",
        "0b41049b276b5871305f453db7c3ed5969f92c44a1a88702cdc000000000",
        "0b41099b2b3431af52bb0ca83d46e238c1953dac10000000000000000000",
        "0b41039b276b5871306072396b99a552e51681a9ef7728f530f49f000000",
        "0b41099b2b3431af52bb0c674d140714b1f180e2c0000000000000000000",
        "0b41089b2b3431aeb2ee20293241111e581ffafc224c0000000000000000",
        "0b41020626a68331b641f2c3963214b60448946eb4c91b2bb8d8fae00000",
    ];

    println!("\n=== Decode all reference frames ===");
    for (i, frame_hex) in REF_FRAMES.iter().enumerate() {
        let bytes = hex_to_bytes(frame_hex);
        let silk = &bytes[2..]; // Skip TOC and count

        // Simple decode
        println!("\nFrame {}:", i);
        println!(
            "  First 4 bytes of SILK: {:02x?}",
            &silk[..4.min(silk.len())]
        );

        // First byte: VAD/LBRR
        let vad = (silk[0] >> 7) & 1;
        let lbrr = (silk[0] >> 6) & 1;
        println!("  VAD={}, LBRR={}", vad, lbrr);
    }
}
