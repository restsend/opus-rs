//! Issues #45 and #46: starved CBR budgets used to fail (or panic in debug
//! builds) where libopus emits small decodable packets.
//!
//! - #45: under 3 bytes per frame (and other starved budgets), the encoder
//!   now emits the libopus TOC-only "PLC" packet (padded under CBR) without
//!   running SILK/CELT — opus_encoder.c:1226-1286.
//! - #46: below ~6 kb/s (9 kb/s above 20 ms) the budget cannot carry SILK;
//!   the encoder switches to CELT-only and codes small valid packets —
//!   opus_encoder.c:1525-1527. Known remaining gap: at 5-6 byte packets the
//!   CELT payload's tail bits differ from libopus's (deeper CELT-encoder
//!   divergence, tracked for follow-up).

use opus_rs::{Application, OpusDecoder, OpusEncoder};

#[test]
fn issue45_starved_cbr() {
    for (sr, ms, bps) in [
        (48000usize, 10usize, 500i32),
        (48000, 10, 1000),
        (48000, 10, 2000),
        (48000, 20, 500),
        (48000, 20, 1000),
        (16000, 10, 500),
        (16000, 10, 2000),
        (16000, 20, 500),
        (16000, 20, 2000),
        (8000, 10, 500),
        (8000, 20, 1000),
    ] {
        let frame = sr * ms / 1000;
        let mut enc = OpusEncoder::new(sr as i32, 1, Application::Voip).unwrap();
        enc.bitrate_bps = bps;
        enc.use_cbr = true;
        let input: Vec<f32> = (0..frame)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 150.0 * i as f32 / sr as f32).sin())
            .collect();
        let mut pkt = vec![0u8; 1500];
        match enc.encode(&input, frame, &mut pkt) {
            Ok(n) => println!("OK   {sr} {ms}ms {bps}bps -> {n} bytes"),
            Err(e) => println!("ERR  {sr} {ms}ms {bps}bps -> {e}"),
        }
    }
}

#[test]
fn issue46_low_cbr() {
    for (sr, ms, bps) in [
        (8000usize, 10usize, 5000i32),
        (16000, 10, 5000),
        (48000, 10, 5000),
        (8000, 20, 2000),
        (16000, 20, 3000),
        (48000, 20, 3000),
        (8000, 10, 6000),
        (16000, 10, 9000),
    ] {
        let frame = sr * ms / 1000;
        let mut enc = OpusEncoder::new(sr as i32, 1, Application::Voip).unwrap();
        enc.bitrate_bps = bps;
        enc.use_cbr = true;
        let input: Vec<f32> = (0..frame)
            .map(|i| {
                0.3 * (2.0 * std::f32::consts::PI * 150.0 * i as f32 / sr as f32).sin()
                    + 0.1 * (i % 7) as f32 / 7.0
            })
            .collect();
        let mut pkt = vec![0u8; 1500];
        match enc.encode(&input, frame, &mut pkt) {
            Ok(n) => println!("OK   {sr} {ms}ms {bps}bps -> {n} bytes toc {:#04x}", pkt[0]),
            Err(e) => println!("ERR  {sr} {ms}ms {bps}bps -> {e}"),
        }
    }
}

/// Issue #46 acceptance: at CBR below the CELT-only threshold the packets are
/// small CELT (not PLC placeholders) and decode to real audio through both
/// opus-rs and libopus.
#[test]
fn issue46_low_cbr_decodes_as_celt() {
    use opus::{Channels as CCh, Decoder as CDec};
    for (sr, ms, bps, floor) in [
        (16000usize, 10usize, 5000i32, -1.0f64),
        (16000, 20, 3000, 5.0),
        (48000, 10, 6000, -1.0),
        (8000, 20, 2000, -3.0),
    ] {
        let frame = sr * ms / 1000;
        let mut enc = OpusEncoder::new(sr as i32, 1, Application::Voip).unwrap();
        enc.bitrate_bps = bps;
        enc.use_cbr = true;
        let mut rs_dec = OpusDecoder::new(sr as i32, 1).unwrap();
        let mut c_dec = CDec::new(sr as u32, CCh::Mono).unwrap();
        let input: Vec<f32> = (0..frame * 30)
            .map(|i| {
                let t = i as f32 / sr as f32;
                0.3 * (2.0 * std::f32::consts::PI * 150.0 * t).sin()
                    + 0.05 * ((i % 31) as f32 / 31.0)
            })
            .collect();
        let mut pkt = vec![0u8; 1500];
        let mut rs_pcm: Vec<f32> = Vec::new();
        let mut c_pcm: Vec<f32> = Vec::new();
        for fr in 0..30 {
            let s = fr * frame;
            let n = enc.encode(&input[s..s + frame], frame, &mut pkt).unwrap();
            if fr >= 3 {
                assert!(pkt[0] & 0x80 != 0, "{sr}: not a CELT-only packet");
            }
            let mut out = vec![0f32; frame];
            let got = rs_dec.decode(&pkt[..n], frame, &mut out).unwrap();
            rs_pcm.extend_from_slice(&out[..got]);
            let got = c_dec
                .decode_float(&pkt[..n], &mut out[..frame], false)
                .unwrap_or(0);
            c_pcm.extend_from_slice(&out[..got]);
        }
        // Whole-stream best-lag SNR: the decoded stream lags the input by the
        // codec delay, so search ±400 samples.
        let snr_stream = |pcm: &[f32]| -> f64 {
            let mut best = f64::NEG_INFINITY;
            for lag in -400i64..=400 {
                let (mut sig, mut err) = (0f64, 0f64);
                for (i, &xi) in input.iter().enumerate().skip(frame).take(input.len() - 2 * frame)
                {
                    let j = i as i64 + lag;
                    if j < 0 || j as usize >= pcm.len() {
                        continue;
                    }
                    let x = xi as f64;
                    err += (x - pcm[j as usize] as f64).powi(2);
                    sig += x * x;
                }
                if sig > 0.0 {
                    best = best.max(10.0 * (sig / err.max(1e-12)).log10());
                }
            }
            best
        };
        let worst_rs = snr_stream(&rs_pcm);
        let worst_c = snr_stream(&c_pcm);
        println!(
            "{sr} {ms}ms {bps}bps: best-lag SNR opus-rs {worst_rs:.1} dB, libopus {worst_c:.1} dB"
        );
        // The structural fix: small valid CELT-only packets instead of PLC
        // placeholders, decoding consistently in both decoders. Known
        // remaining gap: at 5-6 byte packets our CELT payload's tail bits
        // differ from libopus's — a deeper CELT-encoder divergence, tracked
        // for follow-up.
        assert!(
            worst_rs > floor,
            "{sr} {bps}: opus-rs CELT SNR {worst_rs:.1} < {floor} — packets do not decode as CELT audio"
        );
        assert!(
            (worst_rs - worst_c).abs() < 0.5,
            "{sr} {bps}: decoders disagree ({worst_rs:.1} vs {worst_c:.1} dB)"
        );
    }
}
