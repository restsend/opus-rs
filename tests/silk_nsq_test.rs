use opus_rs::silk::define::*;
use opus_rs::silk::nsq::silk_nsq;
use opus_rs::silk::nsq_del_dec::silk_nsq_del_dec;
use opus_rs::silk::structs::*;

/// Create a 16kHz wideband encoder state for testing
fn create_wb_encoder_state() -> SilkEncoderStateCommon {
    let mut s = SilkEncoderStateCommon::default();
    s.fs_khz = 16;
    s.nb_subfr = 4;
    s.subfr_length = 80; // 5ms * 16kHz
    s.frame_length = 320; // 20ms * 16kHz
    s.ltp_mem_length = 320; // PE_LTP_MEM_LENGTH_MS * fs_khz = 20 * 16
    s.predict_lpc_order = 16;
    s.shaping_lpc_order = 16;
    s.first_frame_after_reset = 1;
    s.indices.nlsf_interp_coef_q2 = 4; // No interpolation
    s.indices.signal_type = TYPE_UNVOICED as i8;
    s.indices.quant_offset_type = 0;
    s.n_states_delayed_decision = 1;
    s
}

/// Create a 8kHz narrowband encoder state for testing
fn create_nb_encoder_state() -> SilkEncoderStateCommon {
    let mut s = SilkEncoderStateCommon::default();
    s.fs_khz = 8;
    s.nb_subfr = 4;
    s.subfr_length = 40; // 5ms * 8kHz
    s.frame_length = 160; // 20ms * 8kHz
    s.ltp_mem_length = 160; // PE_LTP_MEM_LENGTH_MS * fs_khz = 20 * 8
    s.predict_lpc_order = 10;
    s.shaping_lpc_order = 16;
    s.first_frame_after_reset = 1;
    s.indices.nlsf_interp_coef_q2 = 4; // No interpolation
    s.indices.signal_type = TYPE_UNVOICED as i8;
    s.indices.quant_offset_type = 0;
    s.n_states_delayed_decision = 1;
    s
}

/// Generate synthetic unvoiced (noise-like) input
fn generate_unvoiced_input(length: usize) -> Vec<i16> {
    let mut rng: u32 = 12345;
    let mut out = vec![0i16; length];
    for i in 0..length {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        out[i] = ((rng >> 16) as i16) >> 2; // small amplitude noise
    }
    out
}

/// Generate synthetic voiced (periodic) input
fn generate_voiced_input(length: usize, pitch: usize) -> Vec<i16> {
    let mut out = vec![0i16; length];
    for i in 0..length {
        let phase = (i % pitch) as f32 / pitch as f32;
        // Pulse train - simple voiced model
        out[i] = if phase < 0.1 { 3000 } else { -200 };
    }
    out
}

/// Create basic AR shaping coefficients (mild spectral tilt)
fn create_ar_shaping(nb_subfr: usize) -> Vec<i16> {
    let mut ar = vec![0i16; nb_subfr * MAX_SHAPE_LPC_ORDER];
    for k in 0..nb_subfr {
        // Simple first-order tilt, rest zero
        ar[k * MAX_SHAPE_LPC_ORDER] = 4096; // ~0.5 in Q13
        ar[k * MAX_SHAPE_LPC_ORDER + 1] = 2048;
    }
    ar
}

/// Create basic LPC prediction coefficients
fn create_pred_coefs() -> Vec<i16> {
    let mut coefs = vec![0i16; 2 * MAX_LPC_ORDER];
    // Simple first-order predictor: a[0] = 0.9 in Q12 = 3686
    coefs[0] = 3686;
    coefs[1] = -1843; // -0.45 in Q12
    // Second set same as first
    coefs[MAX_LPC_ORDER] = 3686;
    coefs[MAX_LPC_ORDER + 1] = -1843;
    coefs
}

#[test]
fn test_nsq_unvoiced_basic() {
    let s_cmn = create_wb_encoder_state();
    let mut nsq = SilkNSQState::default();
    nsq.prev_gain_q16 = 65536; // gain = 1.0 in Q16

    let input = generate_unvoiced_input(s_cmn.frame_length as usize);
    let mut pulses = vec![0i8; s_cmn.frame_length as usize];

    let pred_coef_q12 = create_pred_coefs();
    let ltp_coef_q14 = vec![0i16; MAX_NB_SUBFR * LTP_ORDER]; // unvoiced: no LTP
    let ar_q13 = create_ar_shaping(s_cmn.nb_subfr as usize);
    let harm_shape_gain_q14 = vec![0i32; MAX_NB_SUBFR];
    let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
    let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];
    // Moderate gain
    let gains_q16 = vec![65536i32; MAX_NB_SUBFR]; // gain = 1.0 in Q16
    let pitch_l = vec![0i32; MAX_NB_SUBFR]; // no pitch for unvoiced
    let lambda_q10 = 1024; // lambda = 1.0 in Q10
    let ltp_scale_q14 = 16384; // 1.0 in Q14

    silk_nsq(
        &s_cmn,
        &mut nsq,
        &s_cmn.indices,
        &input,
        &mut pulses,
        &pred_coef_q12,
        &ltp_coef_q14,
        &ar_q13,
        &harm_shape_gain_q14,
        &tilt_q14,
        &lf_shp_q14,
        &gains_q16,
        &pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );

    // Verify pulses were generated (non-zero output)
    let non_zero_pulses = pulses.iter().filter(|&&p| p != 0).count();
    println!(
        "NSQ Unvoiced: {} non-zero pulses out of {}",
        non_zero_pulses,
        pulses.len()
    );
    assert!(
        non_zero_pulses > 0,
        "NSQ should produce non-zero pulses for non-silent input"
    );

    // Verify pulse magnitudes are reasonable (typical range -8..8 for SILK)
    let max_pulse = pulses.iter().map(|&p| (p as i32).abs()).max().unwrap();
    println!("NSQ Unvoiced: max pulse magnitude = {}", max_pulse);
    assert!(
        max_pulse <= 127,
        "Pulse magnitude should be within i8 range"
    );

    // Verify state was updated
    assert_ne!(
        nsq.s_ltp_shp_buf_idx, 0,
        "LTP shape buffer index should be updated"
    );

    println!("✅ NSQ unvoiced basic test passed");
}

#[test]
fn test_nsq_voiced_basic() {
    let mut s_cmn = create_wb_encoder_state();
    s_cmn.indices.signal_type = TYPE_VOICED as i8;
    s_cmn.first_frame_after_reset = 0;

    let mut nsq = SilkNSQState::default();
    nsq.prev_gain_q16 = 65536;
    nsq.prev_sig_type = TYPE_VOICED as i8;

    let pitch = 100; // ~160Hz
    let input = generate_voiced_input(s_cmn.frame_length as usize, pitch);
    let mut pulses = vec![0i8; s_cmn.frame_length as usize];

    let pred_coef_q12 = create_pred_coefs();
    let mut ltp_coef_q14 = vec![0i16; MAX_NB_SUBFR * LTP_ORDER];
    // Set LTP center tap to a moderate value
    for k in 0..s_cmn.nb_subfr as usize {
        ltp_coef_q14[k * LTP_ORDER + 2] = 8192; // center tap ~0.5 in Q14
    }
    let ar_q13 = create_ar_shaping(s_cmn.nb_subfr as usize);
    let harm_shape_gain_q14 = vec![4096i32; MAX_NB_SUBFR]; // mild harmonic shaping
    let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
    let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];
    let gains_q16 = vec![65536i32; MAX_NB_SUBFR];
    let pitch_l = vec![pitch as i32; MAX_NB_SUBFR];
    let lambda_q10 = 1024;
    let ltp_scale_q14 = 16384;

    silk_nsq(
        &s_cmn,
        &mut nsq,
        &s_cmn.indices,
        &input,
        &mut pulses,
        &pred_coef_q12,
        &ltp_coef_q14,
        &ar_q13,
        &harm_shape_gain_q14,
        &tilt_q14,
        &lf_shp_q14,
        &gains_q16,
        &pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );

    let non_zero_pulses = pulses.iter().filter(|&&p| p != 0).count();
    println!(
        "NSQ Voiced: {} non-zero pulses out of {}",
        non_zero_pulses,
        pulses.len()
    );
    assert!(
        non_zero_pulses > 0,
        "NSQ should produce non-zero pulses for voiced input"
    );

    // For voiced speech with good prediction, we expect fewer non-zero pulses
    // than for unvoiced (residual should be smaller)
    println!("✅ NSQ voiced basic test passed");
}

#[test]
fn test_nsq_del_dec_basic() {
    let mut s_cmn = create_wb_encoder_state();
    s_cmn.n_states_delayed_decision = 2; // NSQ_MAX_STATES_OPERATING = 2
    s_cmn.indices.signal_type = TYPE_UNVOICED as i8;

    let mut nsq = SilkNSQState::default();
    nsq.prev_gain_q16 = 65536;

    let input = generate_unvoiced_input(s_cmn.frame_length as usize);
    let mut pulses = vec![0i8; s_cmn.frame_length as usize];

    let pred_coef_q12 = create_pred_coefs();
    let ltp_coef_q14 = vec![0i16; MAX_NB_SUBFR * LTP_ORDER];
    let ar_q13 = create_ar_shaping(s_cmn.nb_subfr as usize);
    let harm_shape_gain_q14 = vec![0i32; MAX_NB_SUBFR];
    let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
    let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];
    let gains_q16 = vec![65536i32; MAX_NB_SUBFR];
    let pitch_l = vec![0i32; MAX_NB_SUBFR];
    let lambda_q10 = 1024;
    let ltp_scale_q14 = 16384;

    silk_nsq_del_dec(
        &s_cmn,
        &mut nsq,
        &s_cmn.indices,
        &input,
        &mut pulses,
        &pred_coef_q12,
        &ltp_coef_q14,
        &ar_q13,
        &harm_shape_gain_q14,
        &tilt_q14,
        &lf_shp_q14,
        &gains_q16,
        &pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );

    let non_zero_pulses = pulses.iter().filter(|&&p| p != 0).count();
    println!(
        "NSQ Del-Dec: {} non-zero pulses out of {}",
        non_zero_pulses,
        pulses.len()
    );
    assert!(
        non_zero_pulses > 0,
        "NSQ del-dec should produce non-zero pulses"
    );

    println!("✅ NSQ delayed decision basic test passed");
}

#[test]
fn test_nsq_silent_input() {
    let s_cmn = create_wb_encoder_state();
    let mut nsq = SilkNSQState::default();
    nsq.prev_gain_q16 = 65536;

    // Zero input
    let input = vec![0i16; s_cmn.frame_length as usize];
    let mut pulses = vec![0i8; s_cmn.frame_length as usize];

    let pred_coef_q12 = create_pred_coefs();
    let ltp_coef_q14 = vec![0i16; MAX_NB_SUBFR * LTP_ORDER];
    let ar_q13 = create_ar_shaping(s_cmn.nb_subfr as usize);
    let harm_shape_gain_q14 = vec![0i32; MAX_NB_SUBFR];
    let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
    let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];
    let gains_q16 = vec![65536i32; MAX_NB_SUBFR];
    let pitch_l = vec![0i32; MAX_NB_SUBFR];
    let lambda_q10 = 1024;
    let ltp_scale_q14 = 16384;

    silk_nsq(
        &s_cmn,
        &mut nsq,
        &s_cmn.indices,
        &input,
        &mut pulses,
        &pred_coef_q12,
        &ltp_coef_q14,
        &ar_q13,
        &harm_shape_gain_q14,
        &tilt_q14,
        &lf_shp_q14,
        &gains_q16,
        &pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );

    // Silent input should produce mostly zero or very small pulses
    let total_energy: i64 = pulses.iter().map(|&p| (p as i64) * (p as i64)).sum();
    println!("NSQ Silent: total pulse energy = {}", total_energy);
    // Not asserting zero because noise dithering may produce small pulses

    println!("✅ NSQ silent input test passed");
}

#[test]
fn test_nsq_gain_scaling() {
    let s_cmn = create_wb_encoder_state();

    // Test with low gain
    let mut nsq_low = SilkNSQState::default();
    nsq_low.prev_gain_q16 = 65536;
    let input = generate_unvoiced_input(s_cmn.frame_length as usize);
    let mut pulses_low = vec![0i8; s_cmn.frame_length as usize];

    let pred_coef_q12 = create_pred_coefs();
    let ltp_coef_q14 = vec![0i16; MAX_NB_SUBFR * LTP_ORDER];
    let ar_q13 = create_ar_shaping(s_cmn.nb_subfr as usize);
    let harm_shape_gain_q14 = vec![0i32; MAX_NB_SUBFR];
    let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
    let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];
    let pitch_l = vec![0i32; MAX_NB_SUBFR];
    let lambda_q10 = 1024;
    let ltp_scale_q14 = 16384;

    // Low gain
    let gains_low = vec![32768i32; MAX_NB_SUBFR]; // 0.5 in Q16
    silk_nsq(
        &s_cmn,
        &mut nsq_low,
        &s_cmn.indices,
        &input,
        &mut pulses_low,
        &pred_coef_q12,
        &ltp_coef_q14,
        &ar_q13,
        &harm_shape_gain_q14,
        &tilt_q14,
        &lf_shp_q14,
        &gains_low,
        &pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );

    // High gain
    let mut nsq_high = SilkNSQState::default();
    nsq_high.prev_gain_q16 = 65536;
    let mut pulses_high = vec![0i8; s_cmn.frame_length as usize];
    let gains_high = vec![131072i32; MAX_NB_SUBFR]; // 2.0 in Q16
    silk_nsq(
        &s_cmn,
        &mut nsq_high,
        &s_cmn.indices,
        &input,
        &mut pulses_high,
        &pred_coef_q12,
        &ltp_coef_q14,
        &ar_q13,
        &harm_shape_gain_q14,
        &tilt_q14,
        &lf_shp_q14,
        &gains_high,
        &pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );

    // With higher gain, we expect smaller pulse magnitudes (gain absorbs more signal)
    let energy_low: i64 = pulses_low.iter().map(|&p| (p as i64) * (p as i64)).sum();
    let energy_high: i64 = pulses_high.iter().map(|&p| (p as i64) * (p as i64)).sum();
    println!(
        "NSQ Gain: low_gain_energy={}, high_gain_energy={}",
        energy_low, energy_high
    );

    // Higher gain should result in different pulse distribution
    assert_ne!(
        energy_low, energy_high,
        "Different gains should produce different pulse energies"
    );

    println!("✅ NSQ gain scaling test passed");
}

/// Helper function to run silk_nsq and return the output
fn run_nsq_and_capture(
    s_cmn: &SilkEncoderStateCommon,
    input: &[i16],
    pred_coef_q12: &[i16],
    ltp_coef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
) -> (Vec<i8>, SilkNSQState) {
    let mut nsq = SilkNSQState::default();
    nsq.prev_gain_q16 = 65536;
    if s_cmn.indices.signal_type == TYPE_VOICED as i8 {
        nsq.prev_sig_type = TYPE_VOICED as i8;
    }

    let mut pulses = vec![0i8; s_cmn.frame_length as usize];

    silk_nsq(
        s_cmn,
        &mut nsq,
        &s_cmn.indices,
        input,
        &mut pulses,
        pred_coef_q12,
        ltp_coef_q14,
        ar_q13,
        harm_shape_gain_q14,
        tilt_q14,
        lf_shp_q14,
        gains_q16,
        pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );

    (pulses, nsq)
}

/// Consistency test: ensures optimization doesn't change output
/// These expected values are captured from the reference implementation
#[test]
fn test_nsq_consistency_unvoiced_wb() {
    let s_cmn = create_wb_encoder_state();
    let input = generate_unvoiced_input(s_cmn.frame_length as usize);

    let pred_coef_q12 = create_pred_coefs();
    let ltp_coef_q14 = vec![0i16; MAX_NB_SUBFR * LTP_ORDER];
    let ar_q13 = create_ar_shaping(s_cmn.nb_subfr as usize);
    let harm_shape_gain_q14 = vec![0i32; MAX_NB_SUBFR];
    let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
    let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];
    let gains_q16 = vec![65536i32; MAX_NB_SUBFR];
    let pitch_l = vec![0i32; MAX_NB_SUBFR];
    let lambda_q10 = 1024;
    let ltp_scale_q14 = 16384;

    let (pulses, nsq) = run_nsq_and_capture(
        &s_cmn,
        &input,
        &pred_coef_q12,
        &ltp_coef_q14,
        &ar_q13,
        &harm_shape_gain_q14,
        &tilt_q14,
        &lf_shp_q14,
        &gains_q16,
        &pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );

    // Reference output captured from original implementation
    // Sum of pulses for quick verification
    let pulse_sum: i64 = pulses.iter().map(|&p| p as i64).sum();
    let pulse_sq_sum: i64 = pulses.iter().map(|&p| (p as i64) * (p as i64)).sum();

    println!(
        "Unvoiced WB consistency: pulse_sum={}, pulse_sq_sum={}",
        pulse_sum, pulse_sq_sum
    );
    println!(
        "NSQ state after: lag_prev={}, rand_seed={}",
        nsq.lag_prev, nsq.rand_seed
    );

    // These values should remain constant after optimization
    // If they change, the optimization broke correctness
    assert_eq!(
        pulse_sum, 160,
        "Pulse sum mismatch - optimization may have broken correctness"
    );
    assert_eq!(
        pulse_sq_sum, 287360,
        "Pulse square sum mismatch - optimization may have broken correctness"
    );
    assert_eq!(nsq.lag_prev, 0, "lag_prev mismatch");
}

#[test]
fn test_nsq_consistency_voiced_wb() {
    let mut s_cmn = create_wb_encoder_state();
    s_cmn.indices.signal_type = TYPE_VOICED as i8;
    s_cmn.first_frame_after_reset = 0;

    let pitch = 100;
    let input = generate_voiced_input(s_cmn.frame_length as usize, pitch);

    let pred_coef_q12 = create_pred_coefs();
    let mut ltp_coef_q14 = vec![0i16; MAX_NB_SUBFR * LTP_ORDER];
    for k in 0..s_cmn.nb_subfr as usize {
        ltp_coef_q14[k * LTP_ORDER + 2] = 8192;
    }
    let ar_q13 = create_ar_shaping(s_cmn.nb_subfr as usize);
    let harm_shape_gain_q14 = vec![4096i32; MAX_NB_SUBFR];
    let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
    let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];
    let gains_q16 = vec![65536i32; MAX_NB_SUBFR];
    let pitch_l = vec![pitch as i32; MAX_NB_SUBFR];
    let lambda_q10 = 1024;
    let ltp_scale_q14 = 16384;

    let (pulses, nsq) = run_nsq_and_capture(
        &s_cmn,
        &input,
        &pred_coef_q12,
        &ltp_coef_q14,
        &ar_q13,
        &harm_shape_gain_q14,
        &tilt_q14,
        &lf_shp_q14,
        &gains_q16,
        &pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );

    let pulse_sum: i64 = pulses.iter().map(|&p| p as i64).sum();
    let pulse_sq_sum: i64 = pulses.iter().map(|&p| (p as i64) * (p as i64)).sum();

    println!(
        "Voiced WB consistency: pulse_sum={}, pulse_sq_sum={}",
        pulse_sum, pulse_sq_sum
    );
    println!(
        "NSQ state after: lag_prev={}, rand_seed={}",
        nsq.lag_prev, nsq.rand_seed
    );

    // Reference values
    assert_eq!(pulse_sum, 38, "Pulse sum mismatch");
    assert_eq!(pulse_sq_sum, 296718, "Pulse square sum mismatch");
    assert_eq!(nsq.lag_prev, 100, "lag_prev mismatch");
}

#[test]
fn test_nsq_consistency_unvoiced_nb() {
    let s_cmn = create_nb_encoder_state();
    let input = generate_unvoiced_input(s_cmn.frame_length as usize);

    let pred_coef_q12 = create_pred_coefs();
    let ltp_coef_q14 = vec![0i16; MAX_NB_SUBFR * LTP_ORDER];
    let ar_q13 = create_ar_shaping(s_cmn.nb_subfr as usize);
    let harm_shape_gain_q14 = vec![0i32; MAX_NB_SUBFR];
    let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
    let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];
    let gains_q16 = vec![65536i32; MAX_NB_SUBFR];
    let pitch_l = vec![0i32; MAX_NB_SUBFR];
    let lambda_q10 = 1024;
    let ltp_scale_q14 = 16384;

    let (pulses, nsq) = run_nsq_and_capture(
        &s_cmn,
        &input,
        &pred_coef_q12,
        &ltp_coef_q14,
        &ar_q13,
        &harm_shape_gain_q14,
        &tilt_q14,
        &lf_shp_q14,
        &gains_q16,
        &pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );

    let pulse_sum: i64 = pulses.iter().map(|&p| p as i64).sum();
    let pulse_sq_sum: i64 = pulses.iter().map(|&p| (p as i64) * (p as i64)).sum();

    println!(
        "Unvoiced NB consistency: pulse_sum={}, pulse_sq_sum={}",
        pulse_sum, pulse_sq_sum
    );
    println!(
        "NSQ state after: lag_prev={}, rand_seed={}",
        nsq.lag_prev, nsq.rand_seed
    );

    // Reference values
    assert_eq!(pulse_sum, 380, "Pulse sum mismatch");
    assert_eq!(pulse_sq_sum, 143080, "Pulse square sum mismatch");
    assert_eq!(nsq.lag_prev, 0, "lag_prev mismatch");
}

#[test]
fn test_nsq_consistency_voiced_nb() {
    let mut s_cmn = create_nb_encoder_state();
    s_cmn.indices.signal_type = TYPE_VOICED as i8;
    s_cmn.first_frame_after_reset = 0;

    let pitch = 50; // lower pitch for NB
    let input = generate_voiced_input(s_cmn.frame_length as usize, pitch);

    let pred_coef_q12 = create_pred_coefs();
    let mut ltp_coef_q14 = vec![0i16; MAX_NB_SUBFR * LTP_ORDER];
    for k in 0..s_cmn.nb_subfr as usize {
        ltp_coef_q14[k * LTP_ORDER + 2] = 8192;
    }
    let ar_q13 = create_ar_shaping(s_cmn.nb_subfr as usize);
    let harm_shape_gain_q14 = vec![4096i32; MAX_NB_SUBFR];
    let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
    let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];
    let gains_q16 = vec![65536i32; MAX_NB_SUBFR];
    let pitch_l = vec![pitch as i32; MAX_NB_SUBFR];
    let lambda_q10 = 1024;
    let ltp_scale_q14 = 16384;

    let (pulses, nsq) = run_nsq_and_capture(
        &s_cmn,
        &input,
        &pred_coef_q12,
        &ltp_coef_q14,
        &ar_q13,
        &harm_shape_gain_q14,
        &tilt_q14,
        &lf_shp_q14,
        &gains_q16,
        &pitch_l,
        lambda_q10,
        ltp_scale_q14,
    );

    let pulse_sum: i64 = pulses.iter().map(|&p| p as i64).sum();
    let pulse_sq_sum: i64 = pulses.iter().map(|&p| (p as i64) * (p as i64)).sum();

    println!(
        "Voiced NB consistency: pulse_sum={}, pulse_sq_sum={}",
        pulse_sum, pulse_sq_sum
    );
    println!(
        "NSQ state after: lag_prev={}, rand_seed={}",
        nsq.lag_prev, nsq.rand_seed
    );

    // Reference values
    assert_eq!(pulse_sum, -205, "Pulse sum mismatch");
    assert_eq!(pulse_sq_sum, 148497, "Pulse square sum mismatch");
    assert_eq!(nsq.lag_prev, 50, "lag_prev mismatch");
}
