//! Issue #53 regression: mediumband (12 kHz) SILK must decode like libopus.
//!
//! `silk_decoder_set_fs` gave a 12 kHz decoder the NB/MB NLSF codebook
//! (order 10) but the wideband LPC order (16). The decoder read ten NLSF
//! indices, then built and interpolated a 16-tap synthesis filter from them,
//! so every mediumband packet decoded to noise, at every output rate and from
//! either encoder. libopus (`silk/decoder_set_fs.c`) gives 8 and 12 kHz
//! `MIN_LPC_ORDER` alongside `silk_NLSF_CB_NB_MB`.
//!
//! These tests decode mediumband SILK from libopus's encoder and from
//! opus-rs's with both decoders, and require agreement at lag 0. Before the
//! fix every window after the first packet sat near 0 dB.

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Decoder as CDec,
    Encoder as CEnc, Signal as CSignal,
};
use opus_rs::silk::init_decoder::{silk_create_decoder, silk_decoder_set_fs};
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

/// Largest Opus frame per channel: 120 ms at 48 kHz.
const MAX_FRAME: usize = 5760;

/// Mediumband SILK's internal rate, and the encoders' input rate here.
const MB_RATE: usize = 12000;

fn c_channels(ch: usize) -> CCh {
    if ch == 2 { CCh::Stereo } else { CCh::Mono }
}

/// Voiced-speech-like test signal: a 140 Hz harmonic series with slow pitch
/// drift and a syllable-rate envelope. The right channel is a scaled, phase
/// shifted copy so the stereo side signal is not trivially zero.
fn voiced(sr: usize, ch: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * ch);
    let mut phase = 0.0f32;
    for i in 0..n {
        let t = i as f32 / sr as f32;
        let f0 = 140.0 + 15.0 * (2.0 * PI * 0.7 * t).sin();
        phase += 2.0 * PI * f0 / sr as f32;
        let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin();
        let mut l = 0.0;
        let mut r = 0.0;
        for h in 1..=8 {
            let a = 0.25 / h as f32;
            l += a * (h as f32 * phase).sin();
            r += a * (h as f32 * phase + 0.3 * h as f32).sin();
        }
        out.push(l * env * 0.5);
        if ch == 2 {
            out.push(r * env * 0.4);
        }
    }
    out
}

/// TOC check: SILK-only mediumband (configs 4..=7) with `ch` channels.
fn assert_mediumband_silk(pkt: &[u8], ch: usize, who: &str) {
    let config = pkt[0] >> 3;
    assert!(
        (4..=7).contains(&config),
        "{who}: expected a SILK MB TOC, got config {config}"
    );
    assert_eq!((pkt[0] >> 2) & 1 == 1, ch == 2, "{who}: TOC stereo bit");
}

/// ~1 s of libopus-encoded mediumband SILK in `ms` frames.
fn libopus_packets(ch: usize, ms: usize) -> Vec<Vec<u8>> {
    let frame = MB_RATE * ms / 1000;
    let input = voiced(MB_RATE, ch, MB_RATE);
    let mut enc = CEnc::new(MB_RATE as u32, c_channels(ch), CApp::Voip).unwrap();
    enc.set_bitrate(CBitrate::Bits(24000)).unwrap();
    enc.set_bandwidth(CBw::Mediumband).unwrap();
    // A steady harmonic tone can read as music and pull libopus into CELT.
    enc.set_signal(CSignal::Voice).unwrap();
    enc.set_complexity(9).unwrap();
    if ch == 2 {
        enc.set_force_channels(Some(CCh::Stereo)).unwrap();
    }
    let mut pkt = vec![0u8; 1500];
    input
        .chunks_exact(frame * ch)
        .map(|f| {
            let n = enc.encode_float(f, &mut pkt).unwrap();
            assert_mediumband_silk(&pkt[..n], ch, "libopus");
            pkt[..n].to_vec()
        })
        .collect()
}

/// ~1 s of opus-rs-encoded 20 ms packets from 12 kHz input, which the
/// encoder codes as mediumband SILK (the issue's own path).
fn opus_rs_packets(ch: usize) -> Vec<Vec<u8>> {
    let frame = MB_RATE / 50;
    let input = voiced(MB_RATE, ch, MB_RATE);
    let mut enc = OpusEncoder::new(MB_RATE as i32, ch, Application::Voip).unwrap();
    enc.bitrate_bps = 24000;
    let mut pkt = vec![0u8; 1500];
    input
        .chunks_exact(frame * ch)
        .map(|f| {
            let n = enc.encode(f, frame, &mut pkt).unwrap();
            assert_mediumband_silk(&pkt[..n], ch, "opus-rs");
            pkt[..n].to_vec()
        })
        .collect()
}

/// Decode `packets` with libopus and with opus-rs at `api_rate`, and return
/// the worst lag-0 SNR (dB) of opus-rs against libopus over 20 ms windows
/// and channels, after skipping the first packet (`ms` long) as warm-up.
fn worst_window_snr(packets: &[Vec<u8>], ch: usize, ms: usize, api_rate: usize) -> f64 {
    let mut c_dec = CDec::new(api_rate as u32, c_channels(ch)).unwrap();
    let mut rs_dec = OpusDecoder::new(api_rate as i32, ch).unwrap();
    let (mut c_pcm, mut rs_pcm) = (Vec::new(), Vec::new());
    let mut c_buf = vec![0f32; MAX_FRAME * ch];
    let mut rs_buf = vec![0f32; MAX_FRAME * ch];
    let frame = api_rate * ms / 1000;
    for (p, pkt) in packets.iter().enumerate() {
        let c_n = c_dec.decode_float(pkt, &mut c_buf, false).unwrap();
        let rs_n = rs_dec
            .decode(pkt, MAX_FRAME, &mut rs_buf)
            .unwrap_or_else(|e| panic!("packet {p}: opus-rs decode failed: {e}"));
        assert_eq!(c_n, frame, "packet {p}: libopus decoded length");
        assert_eq!(rs_n, frame, "packet {p}: opus-rs decoded length");
        c_pcm.extend_from_slice(&c_buf[..c_n * ch]);
        rs_pcm.extend_from_slice(&rs_buf[..rs_n * ch]);
    }

    let win = api_rate / 50;
    let mut worst = f64::INFINITY;
    for c in 0..ch {
        let mut start = frame;
        while start + win <= c_pcm.len() / ch {
            let (mut sig, mut err) = (0f64, 0f64);
            for i in start..start + win {
                let r = c_pcm[i * ch + c] as f64;
                let e = r - rs_pcm[i * ch + c] as f64;
                sig += r * r;
                err += e * e;
            }
            worst = worst.min(10.0 * (sig / err.max(1e-20)).log10());
            start += win;
        }
    }
    worst
}

/// Before the fix every configuration's worst window measured -14 to -18 dB.
/// Mono now decodes bit-exactly (~190 dB). libopus-encoded stereo stops at
/// 49-60 dB, at wideband too, because the decoder's stereo predictor step is
/// one Q16 unit off libopus's (6553 vs 6554). With issue #42's decoder fixes
/// stereo is bit-exact as well, so the bar sits well clear of both.
const AGREE_DB: f64 = 40.0;

#[test]
fn libopus_mediumband_decodes_like_libopus() {
    for ch in [1, 2] {
        for ms in [20, 60] {
            let packets = libopus_packets(ch, ms);
            for api_rate in [12000, 16000, 48000] {
                let snr = worst_window_snr(&packets, ch, ms, api_rate);
                println!("libopus MB, {ch} ch, {ms} ms, decoded at {api_rate} Hz: {snr:.1} dB");
                assert!(
                    snr > AGREE_DB,
                    "libopus MB, {ch} ch, {ms} ms, at {api_rate} Hz: worst window {snr:.1} dB"
                );
            }
        }
    }
}

#[test]
fn opus_rs_mediumband_decodes_like_libopus() {
    for ch in [1, 2] {
        let packets = opus_rs_packets(ch);
        let snr = worst_window_snr(&packets, ch, 20, 48000);
        println!("opus-rs MB, {ch} ch, decoded at 48000 Hz: {snr:.1} dB");
        assert!(
            snr > AGREE_DB,
            "opus-rs MB, {ch} ch, at 48000 Hz: worst window {snr:.1} dB"
        );
    }
}

/// libopus sets the LPC order and the NLSF codebook from one branch; the
/// decoder reads `ps_nlsf_cb.order` indices and filters with `lpc_order`
/// coefficients, so the two must agree at every rate and across switches.
#[test]
fn set_fs_pairs_lpc_order_with_its_nlsf_codebook() {
    let mut st = silk_create_decoder();
    for (fs_khz, order) in [(8, 10), (12, 10), (16, 16), (12, 10), (8, 10)] {
        assert_eq!(silk_decoder_set_fs(&mut st, fs_khz, 48000), 0);
        let cb_order = st.ps_nlsf_cb.expect("NLSF codebook").order as i32;
        assert_eq!(
            (st.lpc_order, cb_order),
            (order, order),
            "{fs_khz} kHz: (lpc_order, NLSF codebook order)"
        );
    }
}
