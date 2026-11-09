//! SILK decoded at an API rate below its internal rate must match libopus.
//!
//! A decoder at 8 or 12 kHz receiving mediumband or wideband SILK has to
//! downsample. libopus's `silk_resampler_init` uses the private down-FIR
//! (anti-aliasing AR2 + FIR decimator) for every downsampling ratio, in the
//! decoder as in the encoder; the decoder's resampler instead ran the IIR/FIR
//! interpolator meant for upsampling, so such output was aliased and
//! uncorrelated with libopus's (worst windows near 0 dB).

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Decoder as CDec,
    Encoder as CEnc, Signal as CSignal,
};
use opus_rs::OpusDecoder;
use std::f32::consts::PI;

/// Largest Opus frame per channel: 120 ms at 48 kHz.
const MAX_FRAME: usize = 5760;

fn c_channels(ch: usize) -> CCh {
    if ch == 2 { CCh::Stereo } else { CCh::Mono }
}

/// One second of a voiced-speech-like signal with harmonics up to the
/// encoder's band edge, so that downsampling has content to reject.
fn voiced(sr: usize, ch: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(sr * ch);
    let mut phase = 0.0f32;
    for i in 0..sr {
        let t = i as f32 / sr as f32;
        let f0 = 150.0 + 20.0 * (2.0 * PI * 0.7 * t).sin();
        phase += 2.0 * PI * f0 / sr as f32;
        let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin();
        let (mut l, mut r) = (0.0, 0.0);
        for h in 1..=40 {
            if h as f32 * f0 > sr as f32 * 0.45 {
                break;
            }
            let a = 0.3 / h as f32;
            l += a * (h as f32 * phase).sin();
            r += a * (h as f32 * phase + 0.3 * h as f32).sin();
        }
        out.push(l * env * 0.4);
        if ch == 2 {
            out.push(r * env * 0.3);
        }
    }
    out
}

/// libopus-encoded 20 ms SILK packets at `rate` in `bandwidth`.
fn silk_packets(rate: usize, bandwidth: CBw, ch: usize) -> Vec<Vec<u8>> {
    let frame = rate / 50;
    let mut enc = CEnc::new(rate as u32, c_channels(ch), CApp::Voip).unwrap();
    enc.set_bitrate(CBitrate::Bits(32000)).unwrap();
    enc.set_bandwidth(bandwidth).unwrap();
    enc.set_signal(CSignal::Voice).unwrap();
    enc.set_complexity(9).unwrap();
    if ch == 2 {
        enc.set_force_channels(Some(CCh::Stereo)).unwrap();
    }
    let mut pkt = vec![0u8; 1500];
    voiced(rate, ch)
        .chunks_exact(frame * ch)
        .map(|f| {
            let n = enc.encode_float(f, &mut pkt).unwrap();
            assert!(
                pkt[0] & 0x80 == 0 && pkt[0] & 0x60 != 0x60,
                "expected SILK-only"
            );
            pkt[..n].to_vec()
        })
        .collect()
}

/// Worst lag-0 SNR (dB) of opus-rs against libopus over 20 ms windows and
/// channels, both decoding `packets` at `api_rate`; the first packet is
/// warm-up.
fn worst_window_snr(packets: &[Vec<u8>], ch: usize, api_rate: usize) -> f64 {
    let mut c_dec = CDec::new(api_rate as u32, c_channels(ch)).unwrap();
    let mut rs_dec = OpusDecoder::new(api_rate as i32, ch).unwrap();
    let (mut c_pcm, mut rs_pcm) = (Vec::new(), Vec::new());
    let mut c_buf = vec![0f32; MAX_FRAME * ch];
    let mut rs_buf = vec![0f32; MAX_FRAME * ch];
    let frame = api_rate / 50;
    for (p, pkt) in packets.iter().enumerate() {
        let c_n = c_dec.decode_float(pkt, &mut c_buf, false).unwrap();
        let rs_n = rs_dec.decode(pkt, MAX_FRAME, &mut rs_buf).unwrap();
        assert_eq!((c_n, rs_n), (frame, frame), "packet {p}: decoded lengths");
        c_pcm.extend_from_slice(&c_buf[..c_n * ch]);
        rs_pcm.extend_from_slice(&rs_buf[..rs_n * ch]);
    }
    let mut worst = f64::INFINITY;
    for c in 0..ch {
        let mut start = frame;
        while start + frame <= c_pcm.len() / ch {
            let (mut sig, mut err) = (0f64, 0f64);
            for i in start..start + frame {
                let r = c_pcm[i * ch + c] as f64;
                let e = r - rs_pcm[i * ch + c] as f64;
                sig += r * r;
                err += e * e;
            }
            worst = worst.min(10.0 * (sig / err.max(1e-20)).log10());
            start += frame;
        }
    }
    worst
}

#[test]
fn silk_decoded_below_its_internal_rate_matches_libopus() {
    for (rate, bandwidth, name, api_rates) in [
        (16000, CBw::Wideband, "WB", &[12000, 8000][..]),
        (12000, CBw::Mediumband, "MB", &[8000][..]),
    ] {
        for ch in [1, 2] {
            let packets = silk_packets(rate, bandwidth, ch);
            for &api_rate in api_rates {
                let snr = worst_window_snr(&packets, ch, api_rate);
                println!("{name} SILK, {ch} ch, decoded at {api_rate} Hz: {snr:.1} dB");
                assert!(
                    snr > 40.0,
                    "{name} SILK, {ch} ch, decoded at {api_rate} Hz: worst window {snr:.1} dB"
                );
            }
        }
    }
}
