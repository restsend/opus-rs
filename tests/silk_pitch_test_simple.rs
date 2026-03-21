use opus_rs::silk::define::*;
use opus_rs::silk::pitch_analysis::silk_pitch_analysis_core;
use std::f32::consts::PI;

#[test]
fn test_pitch_analysis_voiced_periodic() {
    let fs_khz = 16;
    let nb_subfr = 4;
    let frame_samples = (20 + nb_subfr * 5) * fs_khz;
    let pitch_period = 100;

    let mut frame = vec![0i16; frame_samples];
    for i in 0..frame_samples {
        let phase = (i % pitch_period) as f32 / pitch_period as f32;
        frame[i] = ((phase - 0.5) * 10000.0) as i16;
    }

    let mut pitch_out = [0i32; MAX_NB_SUBFR];
    let mut lag_index: i16 = 0;
    let mut contour_index: i8 = 0;
    let mut ltp_corr_q15: i32 = 0;

    let voicing = silk_pitch_analysis_core(
        &frame,
        &mut pitch_out,
        &mut lag_index,
        &mut contour_index,
        &mut ltp_corr_q15,
        100,
        3932,
        983,
        fs_khz as i32,
        2,
        nb_subfr,
    );

    println!(
        "Voiced: voicing={}, lags={:?}",
        voicing,
        &pitch_out[..nb_subfr]
    );
    assert_eq!(voicing, 0, "Periodic signal should be voiced");
    for i in 0..nb_subfr {
        assert!(pitch_out[i] > 0 && pitch_out[i] < 500);
    }
}

#[test]
fn test_pitch_analysis_unvoiced() {
    let fs_khz = 16;
    let nb_subfr = 4;
    let frame_samples = (20 + nb_subfr * 5) * fs_khz;

    let mut frame = vec![0i16; frame_samples];
    let mut seed = 54321u32;
    for i in 0..frame_samples {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let noise = ((seed >> 16) as i16) as i32;
        frame[i] = ((noise * 5000) >> 16) as i16;
    }

    let mut pitch_out = [0i32; MAX_NB_SUBFR];
    let mut lag_index: i16 = 0;
    let mut contour_index: i8 = 0;
    let mut ltp_corr_q15: i32 = 0;

    let voicing = silk_pitch_analysis_core(
        &frame,
        &mut pitch_out,
        &mut lag_index,
        &mut contour_index,
        &mut ltp_corr_q15,
        100,
        3932,
        983,
        fs_khz as i32,
        2,
        nb_subfr,
    );

    println!("Unvoiced: voicing={} (1=unvoiced)", voicing);
}

#[test]
fn test_pitch_analysis_integration() {
    let fs_khz = 16;
    let nb_subfr = 4;
    let frame_samples = (20 + nb_subfr * 5) * fs_khz;

    let mut frame = vec![0i16; frame_samples];
    for i in 0..frame_samples {
        let phase = (i % 80) as f32 / 80.0;
        frame[i] = (phase.sin() * 15000.0) as i16;
    }

    let mut pitch_out = [0i32; MAX_NB_SUBFR];
    let mut lag_index: i16 = 0;
    let mut contour_index: i8 = 0;
    let mut ltp_corr_q15: i32 = 0;

    let voicing = silk_pitch_analysis_core(
        &frame,
        &mut pitch_out,
        &mut lag_index,
        &mut contour_index,
        &mut ltp_corr_q15,
        100,
        3932,
        983,
        fs_khz as i32,
        2,
        nb_subfr,
    );

    println!(
        "Integration: voicing={}, lags={:?}",
        voicing,
        &pitch_out[..nb_subfr]
    );
    assert!(voicing >= 0);
}

#[test]
fn test_pitch_analysis_sine_wave() {
    let fs = 16000;
    let fs_khz = 16;
    let nb_subfr = 4;
    let frame_samples = (20 + nb_subfr * 5) * fs_khz;

    let mut frame = vec![0i16; frame_samples];
    for i in 0..frame_samples {
        let t = i as f32 / fs as f32;
        frame[i] = (12000.0 * (2.0 * PI * 150.0 * t).sin()) as i16;
    }

    let mut pitch_out = [0i32; MAX_NB_SUBFR];
    let mut lag_index: i16 = 0;
    let mut contour_index: i8 = 0;
    let mut ltp_corr_q15: i32 = 0;

    let voicing = silk_pitch_analysis_core(
        &frame,
        &mut pitch_out,
        &mut lag_index,
        &mut contour_index,
        &mut ltp_corr_q15,
        100,
        3932,
        983,
        fs_khz as i32,
        2,
        nb_subfr,
    );

    println!(
        "Sine: voicing={}, lags={:?}",
        voicing,
        &pitch_out[..nb_subfr]
    );
    assert_eq!(voicing, 0, "Sine wave should be voiced");
}
