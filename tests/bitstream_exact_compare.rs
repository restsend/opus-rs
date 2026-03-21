use opus_rs::{Application, OpusEncoder};

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn gen_silk_test_pcm(n_samples: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let val_f64 = (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 8000.0).sin();
            let i16_val = (val_f64 * 16383.0) as i16;
            i16_val as f32 / 32768.0
        })
        .collect()
}

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
];

fn hex_to_bytes(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn test_silk_bitstream_exact_match() {
    let sample_rate = 8000i32;
    let channels = 1;
    let frame_size = 160usize; // 20ms at 8kHz
    let n_frames = 20;

    let all_pcm = gen_silk_test_pcm(
        n_frames * frame_size + frame_size, /* extra for lookahead */
    );

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 12000;
    encoder.use_cbr = true;
    encoder.complexity = 10;

    let mut mismatches = 0usize;
    let mut matches = 0usize;

    for frame_idx in 0..n_frames {
        let frame = &all_pcm[frame_idx * frame_size..(frame_idx + 1) * frame_size];
        let mut pkt_buf = vec![0u8; 200];
        let pkt_len = encoder
            .encode(frame, frame_size, &mut pkt_buf)
            .expect("Encode failed");
        let pkt = &pkt_buf[..pkt_len];

        let ref_bytes = hex_to_bytes(REF_FRAMES[frame_idx]);
        let ref_len = ref_bytes.len();

        if pkt == ref_bytes.as_slice() {
            matches += 1;
            println!("Frame {:2}: MATCH {} bytes", frame_idx, pkt_len);
        } else {
            mismatches += 1;
            println!(
                "Frame {:2}: MISMATCH rust={} ref={} bytes",
                frame_idx, pkt_len, ref_len
            );
            println!("  rust: {}", hex_encode(pkt));
            println!("  ref:  {}", hex_encode(&ref_bytes));
            // Find first diff byte
            let min_len = pkt_len.min(ref_len);
            for b in 0..min_len {
                if pkt[b] != ref_bytes[b] {
                    println!(
                        "  first diff at byte {}: rust={:02x} ref={:02x}",
                        b, pkt[b], ref_bytes[b]
                    );
                    break;
                }
            }
        }
    }

    println!("\nResult: {}/{} frames match", matches, n_frames);

    if mismatches > 0 {
        panic!(
            "{}/{} frames mismatched — see output above",
            mismatches, n_frames
        );
    }
}

#[test]
fn test_silk_bitstream_toc_and_size() {
    let sample_rate = 8000i32;
    let channels = 1;
    let frame_size = 160usize;
    let n_frames = 20;

    let all_pcm = gen_silk_test_pcm(n_frames * frame_size + frame_size);

    let mut encoder = OpusEncoder::new(sample_rate, channels, Application::Voip)
        .expect("Failed to create encoder");
    encoder.bitrate_bps = 12000;
    encoder.use_cbr = true;
    encoder.complexity = 10;

    for frame_idx in 0..n_frames {
        let frame = &all_pcm[frame_idx * frame_size..(frame_idx + 1) * frame_size];
        let mut pkt_buf = vec![0u8; 200];
        let pkt_len = encoder
            .encode(frame, frame_size, &mut pkt_buf)
            .expect("Encode failed");
        let pkt = &pkt_buf[..pkt_len];
        let ref_bytes = hex_to_bytes(REF_FRAMES[frame_idx]);

        assert_eq!(
            pkt_len,
            ref_bytes.len(),
            "Frame {}: packet size mismatch: rust={} ref={}",
            frame_idx,
            pkt_len,
            ref_bytes.len()
        );
        assert_eq!(
            pkt[0], ref_bytes[0],
            "Frame {}: TOC mismatch: rust={:02x} ref={:02x}",
            frame_idx, pkt[0], ref_bytes[0]
        );
    }
}
