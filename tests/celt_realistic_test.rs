use opus_rs::celt::{CeltDecoder, CeltEncoder};
use opus_rs::modes::default_mode;
use opus_rs::range_coder::RangeCoder;

/// Test CELT loopback at a realistic bitrate (160 bytes per frame @ 960 samples)
/// to isolate whether the CELT codec itself works, vs the OpusEncoder wrapper.
#[test]
fn test_celt_realistic_bitrate() {
    let mode = default_mode();
    let channels = 1;
    let frame_size = 960;
    let budget = 160; // Same as OpusEncoder @ 64kbps

    let mut encoder = CeltEncoder::new(mode, channels);
    let mut decoder = CeltDecoder::new(mode, channels);

    let freq = 440.0;
    let num_frames = 10;
    let sr = 48000.0;

    let mut all_in = Vec::new();
    let mut all_out = Vec::new();

    for f in 0..num_frames {
        let mut pcm_in = vec![0.0f32; frame_size * channels];
        for i in 0..frame_size {
            let idx = f * frame_size + i;
            let t = idx as f32 / sr;
            pcm_in[i] = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.4;
        }
        all_in.extend_from_slice(&pcm_in);

        // Approach A: done() + copy full buffer (correct layout)
        let mut rc = RangeCoder::new_encoder(budget as u32);
        encoder.encode(&pcm_in, frame_size, &mut rc);
        rc.done();
        let compressed = rc.buf[..budget].to_vec();

        let mut pcm_out = vec![0.0f32; frame_size * channels];
        decoder.decode(&compressed, frame_size, &mut pcm_out);
        all_out.extend_from_slice(&pcm_out);
    }

    // Check SNR at various delays
    let start_idx = 4 * frame_size;
    let end_idx = 9 * frame_size;
    let mut best_snr: f32 = -100.0;
    let mut best_delay = 0;

    for delay in 0..2000 {
        let mut sig = 0.0f64;
        let mut noise = 0.0f64;
        let mut count = 0;
        for i in start_idx..end_idx {
            if i + delay >= all_out.len() {
                break;
            }
            let s = all_in[i] as f64;
            let d = all_out[i + delay] as f64;
            sig += s * s;
            noise += (s - d) * (s - d);
            count += 1;
        }
        if count < frame_size {
            continue;
        }
        let snr = 10.0 * (sig / (noise + 1e-10)).log10() as f32;
        if snr > best_snr {
            best_snr = snr;
            best_delay = delay;
        }
    }

    println!("CELT realistic bitrate ({} bytes): Best SNR = {:.2} dB at delay {}", budget, best_snr, best_delay);
    assert!(
        best_snr > 10.0,
        "CELT roundtrip SNR too low: {:.2} dB (best over delays)",
        best_snr
    );
}
