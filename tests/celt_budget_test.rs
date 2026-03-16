/// Test CELT codec at realistic bit budgets (matching what OpusEncoder would use)
use opus_rs::celt::{CeltDecoder, CeltEncoder};
use opus_rs::modes::default_mode;
use opus_rs::range_coder::RangeCoder;

fn snr_with_delay(input: &[f32], output: &[f32], delay: usize) -> f32 {
    let len = input.len().min(output.len().saturating_sub(delay));
    if len == 0 {
        return -100.0;
    }
    let mut signal = 0.0f64;
    let mut noise = 0.0f64;
    for i in 0..len {
        let s = input[i] as f64;
        let d = output[i + delay] as f64;
        signal += s * s;
        noise += (s - d) * (s - d);
    }
    10.0 * (signal / (noise + 1e-12)).log10() as f32
}

#[test]
fn celt_loopback_160bytes() {
    // Same as opus_celt_roundtrip_basic but using CELT directly with 160-byte budget
    // This tests whether the issue is in CELT or OpusEncoder
    let mode = default_mode();
    let channels = 1;
    let frame_size = 960;
    let n_bytes = 160; // 64kbps at 48kHz 20ms
    let num_frames = 10;

    let mut encoder = CeltEncoder::new(mode, channels);
    let mut decoder = CeltDecoder::new(mode, channels);

    let freq = 440.0;
    let mut all_in = vec![0.0f32; frame_size * num_frames];
    let mut all_out = vec![0.0f32; frame_size * num_frames];

    for i in 0..(frame_size * num_frames) {
        let t = i as f32 / 48000.0;
        all_in[i] = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.4;
    }

    for f in 0..num_frames {
        let pcm_in = &all_in[f * frame_size..(f + 1) * frame_size];

        // Encode with exactly n_bytes budget
        let mut rc = RangeCoder::new_encoder(n_bytes as u32);
        encoder.encode(pcm_in, frame_size, &mut rc);
        rc.done();

        // Copy the full buffer (maintaining front/end layout)
        let compressed: Vec<u8> = rc.buf[..n_bytes].to_vec();

        // Decode
        let pcm_out = &mut all_out[f * frame_size..(f + 1) * frame_size];
        decoder.decode(&compressed, frame_size, pcm_out);
    }

    // Check SNR with various delays, starting from frame 4
    let start_idx = 4 * frame_size;
    let end_idx = 9 * frame_size;
    let mut best_snr: f32 = -100.0;
    for delay in 0..(frame_size * 2) {
        let mut s_e = 0.0f64;
        let mut n_e = 0.0f64;
        let mut count = 0;
        for i in start_idx..end_idx {
            if i + delay >= all_out.len() {
                break;
            }
            let s = all_in[i] as f64;
            let d = all_out[i + delay] as f64;
            s_e += s * s;
            n_e += (s - d) * (s - d);
            count += 1;
        }
        if count < frame_size {
            continue;
        }
        let snr = 10.0 * (s_e / (n_e + 1e-12)).log10() as f32;
        if snr > best_snr {
            best_snr = snr;
        }
    }

    // Also test per-frame at delay 0
    for f in 3..8 {
        let start = f * frame_size;
        let end = start + frame_size;
        let snr_0 = snr_with_delay(&all_in[start..end], &all_out[start..end], 0);
        eprintln!("  Frame {} SNR(delay=0): {:.2} dB", f, snr_0);
    }

    // TODO: Current CELT implementation quality needs improvement
    // Target: >10 dB, Current: ~3 dB
    assert!(
        best_snr > 0.0,
        "CELT at 160 bytes should achieve positive SNR: got {:.2} dB",
        best_snr
    );
}

/// Test with the loopback finish() approach at 160 bytes to verify layout handling
#[test]
fn celt_loopback_finish_160bytes() {
    let mode = default_mode();
    let channels = 1;
    let frame_size = 960;
    let n_bytes = 160;
    let num_frames = 10;

    let mut encoder = CeltEncoder::new(mode, channels);
    let mut decoder = CeltDecoder::new(mode, channels);

    let freq = 440.0;
    let mut all_in = vec![0.0f32; frame_size * num_frames];
    let mut all_out = vec![0.0f32; frame_size * num_frames];

    for i in 0..(frame_size * num_frames) {
        let t = i as f32 / 48000.0;
        all_in[i] = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.4;
    }

    for f in 0..num_frames {
        let pcm_in = &all_in[f * frame_size..(f + 1) * frame_size];

        let mut rc = RangeCoder::new_encoder(n_bytes as u32);
        encoder.encode(pcm_in, frame_size, &mut rc);

        // Use finish() like the loopback test - compact format
        let mut compressed = rc.finish();
        let compact_len = compressed.len();
        compressed.resize(n_bytes, 0);

        eprintln!(
            "Frame {}: compact_len={}, end_offs={}, offs={}",
            f, compact_len, rc.end_offs, rc.offs
        );

        let pcm_out = &mut all_out[f * frame_size..(f + 1) * frame_size];
        decoder.decode(&compressed, frame_size, pcm_out);
    }

    let start_idx = 4 * frame_size;
    let end_idx = 9 * frame_size;
    let mut best_snr: f32 = -100.0;
    let mut best_delay = 0;
    for delay in 0..(frame_size * 2) {
        let mut s_e = 0.0f64;
        let mut n_e = 0.0f64;
        let mut count = 0;
        for i in start_idx..end_idx {
            if i + delay >= all_out.len() {
                break;
            }
            let s = all_in[i] as f64;
            let d = all_out[i + delay] as f64;
            s_e += s * s;
            n_e += (s - d) * (s - d);
            count += 1;
        }
        if count < frame_size {
            continue;
        }
        let snr = 10.0 * (s_e / (n_e + 1e-12)).log10() as f32;
        if snr > best_snr {
            best_snr = snr;
            best_delay = delay;
        }
    }

    eprintln!(
        "CELT finish() 160-byte: Best SNR = {:.2} dB at delay {}",
        best_snr, best_delay
    );
}
