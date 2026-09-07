//! Regression tests for the panic-surface hardening found by the second
//! deep scan (issue #27 follow-up).
//!
//! P0: cross-bandwidth `lag_prev` underflow in `silk_decode_core` — a WB
//!     voiced packet followed by packet loss and then an NB packet used to
//!     index out of bounds because `silk_decoder_set_fs` (unlike libopus)
//!     did not reset rate-tied state on a bandwidth switch.
//!
//! P1: guards for direct calls into the low-level public components.

use opus_rs::silk::SilkResampler;
use opus_rs::silk::dec_api::SilkDecoder;
use opus_rs::silk::init_decoder::silk_decoder_set_fs;
use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// P0: cross-bandwidth PLC lag_prev panic
// ---------------------------------------------------------------------------

/// Encodes low-pitched voiced WB audio so that the SILK pitch lag lands
/// near the 16 kHz maximum (a 120 Hz tone yields lag_prev ≈ 229 via octave
/// doubling), then plays the sequence [WB voiced, 1-byte lost (PLC), NB
/// unvoiced] through one decoder. Before the `silk_decoder_set_fs` state
/// reset this panicked with a usize underflow in `silk_decode_core`.
#[test]
fn p0_cross_bandwidth_plc_lag_no_panic() {
    // 120 Hz tone at 16 kHz: decodes voiced with a large pitch lag.
    let wb_frame = 320usize; // 20 ms @ 16 kHz
    let mut wb_enc = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
    wb_enc.bitrate_bps = 24000;
    wb_enc.use_cbr = true;
    let wb_input: Vec<f32> = (0..wb_frame)
        .map(|i| (2.0 * PI * 120.0 * (i as f32 / 16000.0)).sin() * 0.4)
        .collect();

    // 8 kHz encoder: silence produces unvoiced NB packets.
    let nb_frame = 160usize; // 20 ms @ 8 kHz
    let mut nb_enc = OpusEncoder::new(8000, 1, Application::Voip).unwrap();
    nb_enc.bitrate_bps = 12000;
    nb_enc.use_cbr = true;
    let nb_input = vec![0.0f32; nb_frame];

    let mut dec = OpusDecoder::new(16000, 1).unwrap();

    // 1. WB voiced packets (establish lag_prev near the WB max).
    let mut wb_pkt = vec![0u8; 1500];
    let n = wb_enc
        .encode(&wb_input, wb_frame, &mut wb_pkt)
        .expect("wb encode");
    let mut pcm = vec![0.0f32; 2 * wb_frame];
    dec.decode(&wb_pkt[..n], wb_frame, &mut pcm)
        .expect("wb decode");

    // 2. A lost WB packet (1-byte TOC = WB SILK 20 ms) -> PLC, loss_cnt > 0,
    //    prev_signal_type stays voiced.
    let _ = dec.decode(&[0x48u8], wb_frame, &mut pcm);

    // 3. First packet after loss is NB and unvoiced -> the loss-recovery
    //    path surfaces the stale WB lag_prev in an NB frame.
    //    (The decoder runs at 16 kHz, so its 20 ms frame is 320 samples.)
    let mut nb_pkt = vec![0u8; 1500];
    let m = nb_enc
        .encode(&nb_input, nb_frame, &mut nb_pkt)
        .expect("nb encode");
    let res = dec.decode(&nb_pkt[..m], 2 * nb_frame, &mut pcm);
    assert!(
        res.is_ok(),
        "cross-bandwidth recovery frame must not panic: {res:?}"
    );
}

/// `silk_decoder_set_fs` must reset the rate-tied state on a bandwidth
/// switch (libopus decoder_set_fs.c) and leave it untouched otherwise.
#[test]
fn p0_set_fs_resets_state_on_bandwidth_change() {
    use opus_rs::silk::define::TYPE_NO_VOICE_ACTIVITY;

    let mut dec = opus_rs::silk::dec_api::SilkDecoder::new();
    assert_eq!(dec.init(16000, 1), 0);

    // Simulate voiced WB state with a maximal lag.
    dec.channel_state[0].lag_prev = 288;
    dec.channel_state[0].prev_signal_type = 2; // TYPE_VOICED
    dec.channel_state[0].first_frame_after_reset = 0;
    dec.channel_state[0].out_buf[10] = 1234;

    // Same-rate set_fs: state must be preserved.
    assert_eq!(silk_decoder_set_fs(&mut dec.channel_state[0], 16, 16000), 0);
    assert_eq!(dec.channel_state[0].lag_prev, 288);

    // Bandwidth switch to NB: rate-tied state must reset.
    assert_eq!(silk_decoder_set_fs(&mut dec.channel_state[0], 8, 16000), 0);
    assert_eq!(dec.channel_state[0].lag_prev, 100);
    assert_eq!(
        dec.channel_state[0].prev_signal_type,
        TYPE_NO_VOICE_ACTIVITY
    );
    assert_eq!(dec.channel_state[0].first_frame_after_reset, 1);
    assert!(dec.channel_state[0].out_buf.iter().all(|&v| v == 0));
    assert_eq!(dec.channel_state[0].lpc_order, 10); // MIN_LPC_ORDER at NB
    assert_eq!(dec.channel_state[0].ltp_mem_length, 160);
}

/// SILK-layer reproduction of the exact decoder call sequence: WB voiced
/// packet (lag_prev ≈ 229) -> PLC frame (loss_cnt = 1, lag drifts to 238) ->
/// `silk_decoder_set_fs(8 kHz)` -> NB unvoiced packet. Before the fix the
/// loss-recovery path used the stale 238-sample lag inside an NB frame
/// (`ltp_mem_length` = 160) and panicked with a usize underflow in
/// `silk_decode_core`. This test crashes on the pre-fix code.
#[test]
fn p0_silk_layer_cross_bandwidth_recovery() {
    use opus_rs::range_coder::RangeCoder;
    use opus_rs::silk::decode_frame::{FLAG_DECODE_NORMAL, FLAG_PACKET_LOST};

    let wb_frame = 320usize;
    let nb_frame = 160usize;

    let mut wb_enc = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
    wb_enc.bitrate_bps = 24000;
    wb_enc.use_cbr = true;
    let wb_input: Vec<f32> = (0..wb_frame)
        .map(|i| (2.0 * PI * 120.0 * (i as f32 / 16000.0)).sin() * 0.4)
        .collect();
    let mut wb_pkt = vec![0u8; 1500];
    let n = wb_enc
        .encode(&wb_input, wb_frame, &mut wb_pkt)
        .expect("enc");

    let mut nb_enc = OpusEncoder::new(8000, 1, Application::Voip).unwrap();
    nb_enc.bitrate_bps = 12000;
    nb_enc.use_cbr = true;
    let nb_input = vec![0.0f32; nb_frame];
    let mut nb_pkt = vec![0u8; 1500];
    let m = nb_enc
        .encode(&nb_input, nb_frame, &mut nb_pkt)
        .expect("enc2");

    let mut dec = SilkDecoder::new();
    assert_eq!(dec.init(16000, 1), 0);
    let mut out = vec![0i16; wb_frame];

    // Step 1: WB packet (dec_api passes internal_sample_rate = 16000).
    let mut rc1 = RangeCoder::new_decoder(&wb_pkt[2..n]);
    let r1 = dec.decode(&mut rc1, &mut out, FLAG_DECODE_NORMAL, true, 20, 16000);
    assert_eq!(r1, wb_frame as i32);
    assert_eq!(dec.channel_state[0].prev_signal_type, 2); // TYPE_VOICED
    assert!(
        dec.channel_state[0].lag_prev > 160,
        "test requires a large WB lag"
    );

    // Step 2: lost WB packet (PLC keeps lag_prev, loss_cnt = 1).
    let mut rc2 = RangeCoder::new_decoder(&[]);
    let r2 = dec.decode(&mut rc2, &mut out, FLAG_PACKET_LOST, true, 20, 16000);
    assert_eq!(r2, wb_frame as i32);
    assert_eq!(dec.channel_state[0].loss_cnt, 1);

    // Step 3: dec_api calls silk_decoder_set_fs(8, 16000) for the NB packet;
    // the fix resets lag_prev to 100 here.
    assert_eq!(silk_decoder_set_fs(&mut dec.channel_state[0], 8, 16000), 0);
    assert_eq!(dec.channel_state[0].fs_khz, 8);
    assert_eq!(dec.channel_state[0].lag_prev, 100);

    // Step 4: NB unvoiced packet — must decode cleanly.
    let mut rc3 = RangeCoder::new_decoder(&nb_pkt[2..m]);
    let r3 = dec.decode(&mut rc3, &mut out, FLAG_DECODE_NORMAL, true, 20, 8000);
    assert_eq!(r3, nb_frame as i32);
}

// ---------------------------------------------------------------------------
// P1: direct low-level API guards
// ---------------------------------------------------------------------------

/// `SilkResampler::init` with a zero output rate used to fault on an i64
/// division; it must now return -1.
#[test]
fn p1_resampler_init_zero_out_rate() {
    let mut res = SilkResampler::default();
    assert_eq!(res.init(16000, 0), -1);
    let mut res2 = SilkResampler::default();
    assert_eq!(res2.init(0, 16000), -1);
}

/// `hp_cutoff` with fs < 1000 used to divide by zero; it must now return
/// without touching the output.
#[test]
fn p1_hp_cutoff_invalid_fs_no_panic() {
    let input = vec![0.5f32; 160];
    let mut output = vec![0i16; 160];
    let mut hp_mem = [0i32; 2];
    opus_rs::hp_cutoff::hp_cutoff(
        &input,
        60,
        &mut output,
        &mut hp_mem,
        160,
        1,
        500, // invalid: < 1000
    );
}

/// `exp_rotation` with stride == 0 used to divide by zero; it must now be a
/// no-op.
#[test]
fn p1_exp_rotation_zero_stride_no_panic() {
    let mut x = vec![0.5f32; 64];
    opus_rs::pvq::exp_rotation(&mut x, 64, -1, 0, 2, 1);
}

/// CELT constructors reject invalid channel counts with a clear message
/// instead of a cryptic capacity panic later.
#[test]
#[should_panic(expected = "channels must be 1 or 2")]
fn p1_celt_encoder_new_rejects_channels() {
    let _ = opus_rs::celt::CeltEncoder::new(opus_rs::modes::default_mode(), 3);
}

#[test]
#[should_panic(expected = "channels must be 1 or 2")]
fn p1_celt_decoder_new_rejects_channels() {
    let _ = opus_rs::celt::CeltDecoder::new(opus_rs::modes::default_mode(), 3, 48000);
}

/// CELT decode with a frame size that does not match the MDCT geometry is
/// rejected cleanly (return 0) instead of silently decoding garbage or
/// panicking deep inside the transform.
#[test]
fn p1_celt_decode_invalid_frame_size_rejected() {
    let mut dec = opus_rs::celt::CeltDecoder::new(opus_rs::modes::default_mode(), 1, 48000);
    let pkt = [0xABu8; 64];
    // 100 samples @ 48 kHz matches no `120 << lm` geometry.
    let mut pcm = vec![0.0f32; 100];
    assert_eq!(dec.decode(&pkt, 100, &mut pcm), 0);
    // Undersized output buffer: rejected before decoding.
    let mut tiny = vec![0.0f32; 8];
    assert_eq!(dec.decode(&pkt, 960, &mut tiny), 0);
}

/// Range coder encode/decode reject ft == 0 with an assert instead of an
/// arithmetic division fault.
#[test]
#[should_panic(expected = "encode: ft must be > 0")]
fn p1_range_coder_encode_ft_zero_panics() {
    use opus_rs::range_coder::RangeCoder;
    let mut rc = RangeCoder::new_encoder(16);
    rc.encode(0, 0, 0);
}

/// `pitch_xcorr` clamps `max_pitch` when `y` is shorter than the window
/// contract instead of reading past the slice (SIMD kernels peek up to 3
/// elements past each window).
#[test]
fn p1_pitch_xcorr_short_y_no_panic() {
    let x = vec![0.25f32; 40];
    let y = vec![0.5f32; 50]; // contract would need len + max_pitch - 1 = 99
    let mut xcorr = vec![0.0f32; 60];
    opus_rs::pitch::pitch_xcorr(&x, &y, &mut xcorr, 40, 60);
}

// ---------------------------------------------------------------------------
// P2: encoder control-field clamps
// ---------------------------------------------------------------------------

/// Absurd pub-field values must not overflow the mode-selection / bitrate
/// arithmetic; encode() clamps them the way libopus ctl does.
#[test]
fn p2_encoder_extreme_fields_no_panic() {
    let mut enc = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
    enc.bitrate_bps = i32::MAX;
    enc.complexity = i32::MAX;
    enc.packet_loss_perc = i32::MAX;
    let input = vec![0.0f32; 960];
    let mut pkt = vec![0u8; 1500];
    let res = enc.encode(&input, 960, &mut pkt);
    assert!(res.is_ok(), "encode with extreme fields must not panic");

    let mut enc2 = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
    enc2.bitrate_bps = i32::MIN;
    enc2.complexity = i32::MIN;
    enc2.packet_loss_perc = i32::MIN;
    let res2 = enc2.encode(&input, 960, &mut pkt);
    assert!(res2.is_ok(), "encode with extreme fields must not panic");
}
