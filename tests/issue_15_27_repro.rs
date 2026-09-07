//! Regression tests for GitHub issues #15 and #27, plus the panic-risk
//! hardening items found by the accompanying deep scan.
//!
//! Issue #15: silk resampler panic ("len is 64 but the index is 64") when the
//! first real packet after a run of lost packets triggers the mode-transition
//! PLC bridge through an 8k->16k (Up2HQ) resampler with a truncated buffer.
//!
//! Issue #27:
//!   1. 40 ms stereo SILK encoding panicked (`FixedVec` capacity exceeded).
//!   2. The encoder omitted per-frame / LBRR stereo headers -> bitstream desync.
//!   3. Multi-frame (40/60 ms) SILK packets decoded only their first frame.

use opus_rs::silk::SilkResampler;
use opus_rs::silk::dec_api::SilkDecoder;
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

/// Delay-compensated SNR: SILK decode introduces an algorithmic delay, so
/// align by cross-correlation over a small search window before comparing.
fn aligned_snr(inp: &[f32], out: &[f32], start: usize, len: usize, max_delay: usize) -> f32 {
    let mut best = f64::NEG_INFINITY;
    for d in 0..max_delay {
        let mut sig = 0f64;
        let mut err = 0f64;
        for (i, &s) in inp.iter().enumerate().skip(start).take(len) {
            let oi = start + i - start + d; // output index = i + d
            if oi >= out.len() {
                break;
            }
            let s = s as f64;
            let e = s - out[oi] as f64;
            sig += s * s;
            err += e * e;
        }
        if sig > 0.0 {
            best = best.max(10.0 * (sig / err.max(1e-20)).log10());
        }
    }
    if best == f64::NEG_INFINITY {
        -100.0
    } else {
        best as f32
    }
}

fn sine_input(frame: usize, sample_rate: usize, channels: usize) -> Vec<f32> {
    (0..frame)
        .map(|i| {
            let s = (2.0 * PI * 440.0 * (i as f32 / sample_rate as f32)).sin() * 0.3;
            let mut pair = [0.0f32; 2];
            pair[0] = s;
            pair[1] = if channels == 2 { s } else { 0.0 };
            pair
        })
        .flat_map(|p| p)
        .take(frame * channels)
        .collect()
}

fn half_snrs(pcm: &[f32], input: &[f32], samples: usize, channels: usize) -> (f32, f32) {
    half_snrs_delay(pcm, input, samples, channels, 200)
}

fn half_snrs_delay(
    pcm: &[f32],
    input: &[f32],
    samples: usize,
    channels: usize,
    max_delay: usize,
) -> (f32, f32) {
    // Mid signal for stereo (mid-only stereo decodes L≈mid+side, R≈mid-side,
    // so (L+R)/2 recovers mid); identity for mono.
    let mid: Vec<f32> = if channels == 2 {
        (0..samples)
            .map(|i| (pcm[i * 2] + pcm[i * 2 + 1]) * 0.5)
            .collect()
    } else {
        pcm[..samples].to_vec()
    };
    let inp_mid: Vec<f32> = if channels == 2 {
        (0..samples).map(|i| input[i * 2]).collect()
    } else {
        input[..samples].to_vec()
    };
    let half = samples / 2;
    (
        aligned_snr(&inp_mid, &mid, 0, half, max_delay),
        aligned_snr(&inp_mid, &mid, half, half, max_delay),
    )
}

// ---------------------------------------------------------------------------
// Issue #15
// ---------------------------------------------------------------------------

/// After a run of lost packets, the first packet that triggers a
/// SILK->CELT mode transition generates a 5 ms PLC bridge through the
/// 8 kHz->16 kHz resampler. The bridge used to allocate a truncated output
/// buffer, panicking with "index out of bounds: the len is 64 but the index
/// is 64" inside `silk_resampler_private_up2_hq`.
#[test]
fn issue_15_plc_bridge_after_loss_no_panic() {
    let mut dec = OpusDecoder::new(16000, 1).unwrap();
    let mut pcm = vec![0.0f32; 320];

    // 1. A SILK-only NB packet (TOC 0x00: NB SILK, 10 ms, mono): puts the
    //    decoder on the 8 kHz internal rate with an 8k->16k Up2HQ resampler.
    let silk_pkt = [0x00u8, 0x10, 0x22, 0x33];
    dec.decode(&silk_pkt, 160, &mut pcm)
        .expect("silk nb decode");

    // 2. A series of lost packets (1-byte payloads trigger PLC), as in the
    //    issue report.
    for _ in 0..5 {
        let _ = dec.decode(&[0u8], 160, &mut pcm);
    }

    // 3. First real packet is CELT (TOC 0xB0: CELT NB, 10 ms, mono) -> mode
    //    transition. The 5 ms PLC bridge resamples 80 samples 8k->16k; the
    //    bridge buffer used to be clamped to 80 samples instead of 160.
    let celt_pkt = [0xB0u8, 0x55, 0x66, 0x77];
    let n = dec
        .decode(&celt_pkt, 160, &mut pcm)
        .expect("mode-transition decode must not panic");
    assert_eq!(n, 160);
}

/// Direct hardening: `SilkResampler::process` must never panic on a
/// mis-sized output buffer (Up2HQ and Copy modes are now clamped).
#[test]
fn issue_15_resampler_short_output_no_panic() {
    // Up2HQ mode (8k -> 16k): 80 input samples need 160 output samples.
    let mut res = SilkResampler::default();
    assert_eq!(res.init(8000, 16000), 0);
    let input = vec![100i16; 160];
    let mut out = vec![0i16; 80]; // deliberately too small
    res.process(&mut out, &input, 160);
    // Degenerate: smaller than one millisecond of output.
    let mut tiny = vec![0i16; 4];
    res.process(&mut tiny, &input, 160);

    // Copy mode (16k -> 16k): output must hold `in_len` samples.
    let mut res2 = SilkResampler::default();
    assert_eq!(res2.init(16000, 16000), 0);
    let mut out2 = vec![0i16; 10]; // deliberately too small
    res2.process(&mut out2, &input, 160);
}

// ---------------------------------------------------------------------------
// Issue #27-1: 40 ms stereo SILK encoding panicked
// ---------------------------------------------------------------------------

/// 16 kHz stereo VoIP with 40 ms frames used to panic immediately in
/// `silk_enc.stereo.side.resize(640)` (`FixedVec<i16, 320>` capacity).
#[test]
fn issue_27_1_stereo_silk_40ms_encode_no_panic() {
    let mut enc = OpusEncoder::new(16000, 2, Application::Voip).unwrap();
    enc.bitrate_bps = 24000;
    enc.use_cbr = true;
    let frame = 640; // 40 ms @ 16 kHz
    let input = sine_input(frame, 16000, 2);
    let mut pkt = vec![0u8; 1500];
    let n = enc
        .encode(&input, frame, &mut pkt)
        .expect("40 ms stereo SILK encode");
    assert!(n > 2, "packet too short: {n}");
}

// ---------------------------------------------------------------------------
// Issue #27-2a: per-frame stereo headers on multi-frame packets
// ---------------------------------------------------------------------------

/// A 40 ms stereo packet carries two SILK frames; libopus writes the stereo
/// header before every frame. The Rust encoder used to write it only before
/// frame 0, desyncing frame 1 (garbage in the second half, decode SNR of the
/// second half collapsing to ~-8 dB).
#[test]
fn issue_27_2a_stereo_40ms_roundtrip_both_frames_consistent() {
    let mut enc = OpusEncoder::new(16000, 2, Application::Voip).unwrap();
    enc.bitrate_bps = 24000;
    enc.use_cbr = true;
    let frame = 640;

    let mut dec = OpusDecoder::new(16000, 2).unwrap();
    for pkt_idx in 0..3 {
        let input = sine_input(frame, 16000, 2);
        let mut pkt = vec![0u8; 1500];
        let n = enc.encode(&input, frame, &mut pkt).expect("encode");
        let mut pcm = vec![0.0f32; frame * 2];
        let samples = dec
            .decode(&pkt[..n], frame, &mut pcm)
            .expect("decode 40 ms stereo");
        assert_eq!(samples, frame);

        let (snr_first, snr_second) = half_snrs(&pcm, &input, samples, 2);
        assert!(
            snr_first > 8.0 && snr_second > 8.0,
            "pkt {pkt_idx}: stereo desync — snr_first={snr_first:.1} dB, snr_second={snr_second:.1} dB"
        );
    }
}

// ---------------------------------------------------------------------------
// Issue #27-2b: stereo headers in the LBRR (in-band FEC) section
// ---------------------------------------------------------------------------

/// With in-band FEC enabled, the decoder's LBRR skip path expects a stereo
/// header before every LBRR payload; the encoder used to write none, so any
/// stereo packet carrying LBRR desynced immediately. The LBRR section must
/// also gracefully degrade when it does not fit the packet budget.
#[test]
fn issue_27_2b_stereo_fec_roundtrip_consistent() {
    let mut enc = OpusEncoder::new(16000, 2, Application::Voip).unwrap();
    enc.bitrate_bps = 24000;
    enc.use_cbr = true;
    enc.use_inband_fec = true;
    enc.packet_loss_perc = 40;
    let frame = 320; // 20 ms

    let mut dec = OpusDecoder::new(16000, 2).unwrap();
    for pkt_idx in 0..8 {
        let input = sine_input(frame, 16000, 2);
        let mut pkt = vec![0u8; 1500];
        let n = enc
            .encode(&input, frame, &mut pkt)
            .unwrap_or_else(|e| panic!("packet {pkt_idx}: encode failed: {e}"));
        let mut pcm = vec![0.0f32; frame * 2];
        let samples = dec
            .decode(&pkt[..n], frame, &mut pcm)
            .unwrap_or_else(|e| panic!("packet {pkt_idx}: decode failed: {e}"));
        assert_eq!(samples, frame);

        let (snr_first, snr_second) = half_snrs(&pcm, &input, samples, 2);
        assert!(
            snr_first > 8.0 && snr_second > 8.0,
            "packet {pkt_idx}: LBRR stereo desync — snr_first={snr_first:.1} dB, snr_second={snr_second:.1} dB"
        );
    }
}

// ---------------------------------------------------------------------------
// Issue #27-3: multi-frame SILK packets decoded only their first frame
// ---------------------------------------------------------------------------

/// 40 ms SILK mono at the same API rate: the second 20 ms half used to be
/// exact zeros although decode() reported the full length.
#[test]
fn issue_27_3_silk_40ms_mono_full_decode() {
    let mut enc = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
    enc.bitrate_bps = 24000;
    enc.use_cbr = true;
    let frame = 640;

    let mut dec = OpusDecoder::new(16000, 1).unwrap();
    for pkt_idx in 0..3 {
        let input = sine_input(frame, 16000, 1);
        let mut pkt = vec![0u8; 1500];
        let n = enc.encode(&input, frame, &mut pkt).expect("encode");
        let mut pcm = vec![0.0f32; frame];
        let samples = dec
            .decode(&pkt[..n], frame, &mut pcm)
            .expect("decode 40 ms mono");
        assert_eq!(samples, frame);

        let (snr_first, snr_second) = half_snrs(&pcm, &input, samples, 1);
        assert!(
            snr_first > 5.0 && snr_second > 5.0,
            "pkt {pkt_idx}: second SILK frame not decoded — snr_first={snr_first:.1} dB, snr_second={snr_second:.1} dB"
        );
    }
}

/// 40 ms SILK decoded at a different API rate (16 kHz stream -> 48 kHz
/// decoder): exercises the multi-frame loop through the resampling path.
#[test]
fn issue_27_3_silk_40ms_resampled_full_decode() {
    let mut enc = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
    enc.bitrate_bps = 24000;
    enc.use_cbr = true;
    let enc_frame = 640; // 40 ms @ 16 kHz

    let mut dec = OpusDecoder::new(48000, 1).unwrap();
    let dec_frame = 1920; // 40 ms @ 48 kHz
    for pkt_idx in 0..3 {
        let input = sine_input(enc_frame, 16000, 1);
        let mut pkt = vec![0u8; 1500];
        let n = enc.encode(&input, enc_frame, &mut pkt).expect("encode");
        let mut pcm = vec![0.0f32; dec_frame];
        let samples = dec
            .decode(&pkt[..n], dec_frame, &mut pcm)
            .expect("decode 40 ms @ 48 kHz");
        assert_eq!(samples, dec_frame);

        // Issue #27-3 fix target: the second half must contain actual decoded
        // content (it used to be exact zeros), and the reported length must
        // cover the whole packet. (The residual SNR of the 16k->48k resampled
        // multi-frame path is a separate pre-existing quality issue: the
        // baseline also scored only ~0.2 dB here.)
        let rms_second: f64 = pcm[dec_frame / 2..dec_frame]
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>();
        let rms_second = (rms_second / (dec_frame / 2) as f64).sqrt();
        assert!(
            rms_second > 1e-2,
            "pkt {pkt_idx}: second half zeros (rms={rms_second:.5}): only first SILK frame decoded"
        );
    }
}

// ---------------------------------------------------------------------------
// Deep-scan hardening regressions
// ---------------------------------------------------------------------------

/// encode() must reject short input instead of panicking inside the HP
/// filter / float conversion loops.
#[test]
fn deep_scan_encode_short_input_returns_err() {
    let mut enc = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
    let input = vec![0.0f32; 100]; // < 960
    let mut pkt = vec![0u8; 1500];
    assert!(enc.encode(&input, 960, &mut pkt).is_err());
}

/// CELT decode with a frame size in the 2049..=2168 gap (the old guard
/// allowed `DECODE_BUFFER_SIZE + overlap`) used to underflow/panic; it must
/// now be rejected cleanly.
#[test]
fn deep_scan_celt_decode_oversized_frame_rejected() {
    use opus_rs::{celt::CeltDecoder, modes};
    let mut dec = CeltDecoder::new(modes::default_mode(), 1, 48000);
    let pkt = [0xABu8; 32];
    let mut pcm = vec![0.0f32; 2160]; // 45 ms: lands in the old guard gap
    let n = dec.decode(&pkt, 2160, &mut pcm);
    assert_eq!(n, 0, "oversized frame must be rejected, not panic");
}

/// FFT sizes beyond `KISS_MAX_N` (twiddle/bitrev table capacity) must be
/// rejected gracefully by `KissFftState::new` (`kf_factor` bound), not by a
/// mid-construction panic. The `FixedVec::push` assert remains as the
/// internal fail-fast backstop.
#[test]
fn deep_scan_fixedvec_push_overflow_guarded() {
    // 960 = 2^6 * 3 * 5 is a valid 5-smooth FFT size but exceeds
    // KISS_MAX_N = 480.
    assert!(opus_rs::kiss_fft::KissFftState::new(960).is_none());
    // Sanity: normal sizes still construct.
    assert!(opus_rs::kiss_fft::KissFftState::new(480).is_some());
}

/// A `SilkDecoder` used without `init()` must fail gracefully instead of
/// unwrapping a missing NLSF codebook.
#[test]
fn deep_scan_silk_decoder_without_init_no_panic() {
    let mut dec = SilkDecoder::new();
    let mut out = vec![0i16; 320];
    // fs_khz == 0 -> invalid internal sample rate -> graceful -1, no panic.
    let ret = dec.decode_bytes(&[0x11, 0x22, 0x33], &mut out, true);
    assert_eq!(ret, -1);
}
