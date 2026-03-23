use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

// Configure Criterion for faster benchmarks
fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(20) // Reduce from default 100 to 20 iterations
        .measurement_time(std::time::Duration::from_millis(500)) // 500ms per bench
        .warm_up_time(std::time::Duration::from_millis(100)) // 100ms warmup
}
use opus_rs::silk::define::*;
use opus_rs::celt_lpc::autocorr;
use opus_rs::kiss_fft::{KissCpx, KissFftState, opus_fft_impl};
use opus_rs::modes::default_mode;
use opus_rs::pvq::{alg_quant, encode_pulses, pvq_search};
use opus_rs::range_coder::RangeCoder;
use opus_rs::silk::lpc_analysis::silk_burg_modified_fix;
use opus_rs::silk::nsq::silk_nsq;
use opus_rs::silk::pitch_analysis::silk_pitch_analysis_core;
use opus_rs::silk::sigproc_fix::{
    silk_autocorr, silk_inner_prod_aligned, silk_lpc_analysis_filter,
};
use opus_rs::silk::structs::*;
use opus_rs::{Application, OpusDecoder, OpusEncoder};

fn sine_i16(samples: usize, sample_rate: u32, freq: u32) -> Vec<i16> {
    (0..samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            (f64::sin(2.0 * std::f64::consts::PI * freq as f64 * t) * 8000.0) as i16
        })
        .collect()
}

/// 440 Hz sine wave as f32, normalized to [-1, 1].
fn sine_f32(samples: usize, sample_rate: u32, freq: u32) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            f64::sin(2.0 * std::f64::consts::PI * freq as f64 * t) as f32 * 0.25
        })
        .collect()
}

fn bench_opus_encode_silk(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_encode_silk");

    for &(sample_rate, frame_ms) in &[(8000u32, 20usize), (16000, 20), (16000, 10)] {
        let frame_size = sample_rate as usize * frame_ms / 1000;
        let input = sine_f32(frame_size, sample_rate, 440);
        let mut output = vec![0u8; 256];

        group.throughput(Throughput::Bytes(
            frame_size as u64 * 2, /* i16 bytes */
        ));
        group.bench_with_input(
            BenchmarkId::new(format!("{sample_rate}Hz/{frame_ms}ms"), "voip"),
            &(sample_rate, frame_size),
            |b, &(sr, fs)| {
                let mut enc = OpusEncoder::new(sr as i32, 1, Application::Voip).unwrap();
                enc.bitrate_bps = 20_000;
                b.iter(|| {
                    enc.encode(black_box(&input), fs, black_box(&mut output))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_burg_modified(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_burg_modified_fix");

    for &(d, subfr_length, nb_subfr) in &[
        (10usize, 50usize, 4usize), // 8 kHz NB
        (16, 96, 4),                // 16 kHz WB
        (16, 96, 2),                // 16 kHz WB, half frame
    ] {
        let total = subfr_length * nb_subfr;
        let x = sine_i16(total, 16000, 440);
        let min_inv_gain_q30: i32 = 0;

        group.throughput(Throughput::Elements(total as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("d{d}/subfr{subfr_length}/nb{nb_subfr}"), ""),
            &(d, subfr_length, nb_subfr),
            |b, &(d, sflen, nb)| {
                let mut res_nrg = 0i32;
                let mut res_nrg_q = 0i32;
                let mut a_q16 = [0i32; 16];
                b.iter(|| {
                    silk_burg_modified_fix(
                        black_box(&mut res_nrg),
                        black_box(&mut res_nrg_q),
                        black_box(&mut a_q16),
                        black_box(&x),
                        black_box(min_inv_gain_q30),
                        black_box(sflen),
                        black_box(nb),
                        black_box(d),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_autocorr(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_autocorr");

    for &(n, lags) in &[(320usize, 17usize), (640, 17), (88, 13)] {
        let x = sine_i16(n, 16000, 440);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("n{n}/lags{lags}"), ""),
            &(n, lags),
            |b, &(ns, lgs)| {
                let mut results = vec![0i32; lgs];
                let mut scale = 0i32;
                b.iter(|| {
                    silk_autocorr(
                        black_box(&mut results),
                        black_box(&mut scale),
                        black_box(&x),
                        black_box(ns),
                        black_box(lgs),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_inner_prod(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_inner_prod_aligned");

    for &len in &[64usize, 128, 320, 640] {
        let a = sine_i16(len, 16000, 440);
        let b_vec = sine_i16(len, 16000, 880);

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, &l| {
            b.iter(|| silk_inner_prod_aligned(black_box(&a), black_box(&b_vec), black_box(l)));
        });
    }

    group.finish();
}

fn bench_lpc_analysis_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_lpc_analysis_filter");

    for &(order, len) in &[(10usize, 320usize), (16, 320), (16, 640)] {
        let x = sine_i16(len + order, 16000, 440);
        let a_q12: Vec<i16> = (0..order).map(|i| (i as i16 * 128) as i16).collect();

        group.throughput(Throughput::Elements(len as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("order{order}/len{len}"), ""),
            &(order, len),
            |b, &(ord, l)| {
                let mut out = vec![0i16; l];
                b.iter(|| {
                    silk_lpc_analysis_filter(
                        black_box(&mut out),
                        black_box(&x[..l + ord]),
                        black_box(&a_q12),
                        black_box(l),
                        black_box(ord),
                        0,
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_opus_encode_celt(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_encode_celt");

    for &(sample_rate, frame_ms) in &[(48000u32, 20usize), (48000, 10), (48000, 5)] {
        let frame_size = sample_rate as usize * frame_ms / 1000;
        let input = sine_f32(frame_size, sample_rate, 440);
        let mut output = vec![0u8; 1024];

        group.throughput(Throughput::Bytes(frame_size as u64 * 2));
        group.bench_with_input(
            BenchmarkId::new(format!("{sample_rate}Hz/{frame_ms}ms"), "audio"),
            &(sample_rate, frame_size),
            |b, &(sr, fs)| {
                let mut enc = OpusEncoder::new(sr as i32, 1, Application::Audio).unwrap();
                enc.bitrate_bps = 64_000;
                b.iter(|| {
                    enc.encode(black_box(&input), fs, black_box(&mut output))
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_silk_vs_c(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_vs_c");

    for &(sample_rate, frame_ms) in &[(8000u32, 20usize), (16000u32, 20usize), (16000u32, 10usize)]
    {
        let frame_size = sample_rate as usize * frame_ms / 1000;
        let input = sine_f32(frame_size, sample_rate, 440);

        group.throughput(Throughput::Bytes(frame_size as u64 * 2));

        group.bench_with_input(
            BenchmarkId::new(format!("rust/{sample_rate}Hz/{frame_ms}ms"), "cx0"),
            &(sample_rate, frame_size),
            |b, &(sr, fs)| {
                let mut enc = OpusEncoder::new(sr as i32, 1, Application::Voip).unwrap();
                enc.bitrate_bps = 20_000;
                enc.complexity = 0;
                let mut output = vec![0u8; 256];
                b.iter(|| {
                    enc.encode(black_box(&input), fs, black_box(&mut output))
                        .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("c_cx0/{sample_rate}Hz/{frame_ms}ms"), "cx0"),
            &(sample_rate, frame_size),
            |b, &(sr, fs)| {
                use opusic_sys::*;
                let mut err = 0i32;
                let enc =
                    unsafe { opus_encoder_create(sr as i32, 1, OPUS_APPLICATION_VOIP, &mut err) };
                assert_eq!(err, OPUS_OK);
                unsafe {
                    opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, 20_000i32);
                    opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, 0i32);
                }
                let mut output = vec![0u8; 256];
                b.iter(|| unsafe {
                    opus_encode_float(
                        enc,
                        black_box(input.as_ptr()),
                        fs as i32,
                        output.as_mut_ptr(),
                        output.len() as i32,
                    )
                });
                unsafe { opus_encoder_destroy(enc) };
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("c_cx9/{sample_rate}Hz/{frame_ms}ms"), "cx9"),
            &(sample_rate, frame_size),
            |b, &(sr, fs)| {
                use opusic_sys::*;
                let mut err = 0i32;
                let enc =
                    unsafe { opus_encoder_create(sr as i32, 1, OPUS_APPLICATION_VOIP, &mut err) };
                assert_eq!(err, OPUS_OK);
                unsafe {
                    opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, 20_000i32);
                    opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, 9i32);
                }
                let mut output = vec![0u8; 256];
                b.iter(|| unsafe {
                    opus_encode_float(
                        enc,
                        black_box(input.as_ptr()),
                        fs as i32,
                        output.as_mut_ptr(),
                        output.len() as i32,
                    )
                });
                unsafe { opus_encoder_destroy(enc) };
            },
        );
    }

    group.finish();
}

fn generate_unvoiced_input(length: usize) -> Vec<i16> {
    let mut rng: u32 = 12345;
    let mut out = vec![0i16; length];
    for i in 0..length {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        out[i] = ((rng >> 16) as i16) >> 2; // small amplitude noise
    }
    out
}

fn generate_voiced_input(length: usize, pitch: usize) -> Vec<i16> {
    let mut out = vec![0i16; length];
    for i in 0..length {
        let phase = (i % pitch) as f32 / pitch as f32;
        // Pulse train - simple voiced model
        out[i] = if phase < 0.1 { 3000 } else { -200 };
    }
    out
}

fn create_ar_shaping(nb_subfr: usize) -> Vec<i16> {
    let mut ar = vec![0i16; nb_subfr * MAX_SHAPE_LPC_ORDER];
    for k in 0..nb_subfr {
        ar[k * MAX_SHAPE_LPC_ORDER] = 4096; // ~0.5 in Q13
        ar[k * MAX_SHAPE_LPC_ORDER + 1] = 2048;
    }
    ar
}

fn create_pred_coefs() -> Vec<i16> {
    let mut coefs = vec![0i16; 2 * MAX_LPC_ORDER];
    coefs[0] = 3686; // 0.9 in Q12
    coefs[1] = -1843; // -0.45 in Q12
    coefs[MAX_LPC_ORDER] = 3686;
    coefs[MAX_LPC_ORDER + 1] = -1843;
    coefs
}

fn bench_silk_nsq(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_nsq");

    let configs: &[(i32, &str, usize)] = &[
        (16, "unvoiced", 20),
        (16, "voiced", 20),
        (8, "unvoiced", 20),
        (8, "voiced", 20),
    ];

    for &(fs_khz, sig_type, frame_ms) in configs {
        let frame_size = fs_khz as usize * frame_ms;
        let nb_subfr = 4;
        let subfr_length = frame_size / nb_subfr;

        let input: Vec<i16>;
        let pitch: i32;
        let ltp_coef_val: i16;
        let signal_type_val: i8;

        if sig_type == "voiced" {
            signal_type_val = TYPE_VOICED as i8;
            pitch = 100;
            input = generate_voiced_input(frame_size, pitch as usize);
            ltp_coef_val = 8192; // center tap ~0.5 in Q14
        } else {
            signal_type_val = TYPE_UNVOICED as i8;
            pitch = 0;
            input = generate_unvoiced_input(frame_size);
            ltp_coef_val = 0;
        }

        let pred_coef_q12 = create_pred_coefs();
        let mut ltp_coef_q14 = vec![0i16; nb_subfr * LTP_ORDER];
        if sig_type == "voiced" {
            for k in 0..nb_subfr {
                ltp_coef_q14[k * LTP_ORDER + 2] = ltp_coef_val;
            }
        }
        let ar_q13 = create_ar_shaping(nb_subfr);
        let harm_shape_gain_q14 = if sig_type == "voiced" {
            vec![4096i32; nb_subfr]
        } else {
            vec![0i32; nb_subfr]
        };
        let tilt_q14 = vec![0i32; nb_subfr];
        let lf_shp_q14 = vec![0i32; nb_subfr];
        let gains_q16 = vec![65536i32; nb_subfr]; // gain = 1.0
        let pitch_l = vec![pitch; nb_subfr];
        let lambda_q10 = 1024;
        let ltp_scale_q14 = 16384;

        group.throughput(Throughput::Elements(frame_size as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("{}kHz/{}ms", fs_khz, frame_ms), sig_type),
            &(fs_khz, frame_size, nb_subfr, subfr_length, signal_type_val),
            |b, &(fs_khz, frame_size, nb_subfr, subfr_length, signal_type_val)| {
                b.iter(|| {
                    // Create fresh state each iteration
                    let mut s_cmn = SilkEncoderStateCommon::default();
                    s_cmn.fs_khz = fs_khz;
                    s_cmn.nb_subfr = nb_subfr as i32;
                    s_cmn.subfr_length = subfr_length as i32;
                    s_cmn.frame_length = frame_size as i32;
                    s_cmn.ltp_mem_length = 20 * fs_khz;
                    s_cmn.predict_lpc_order = if fs_khz == 16 { 16 } else { 10 };
                    s_cmn.shaping_lpc_order = 16;
                    s_cmn.first_frame_after_reset = 1;
                    s_cmn.indices.nlsf_interp_coef_q2 = 4;
                    s_cmn.indices.quant_offset_type = 0;
                    s_cmn.indices.signal_type = signal_type_val;
                    s_cmn.n_states_delayed_decision = 1;

                    let mut nsq = SilkNSQState::default();
                    nsq.prev_gain_q16 = 65536;
                    if signal_type_val == TYPE_VOICED as i8 {
                        nsq.prev_sig_type = TYPE_VOICED as i8;
                    }

                    let mut pulses = vec![0i8; frame_size];

                    silk_nsq(
                        black_box(&s_cmn),
                        black_box(&mut nsq),
                        black_box(&s_cmn.indices),
                        black_box(&input),
                        black_box(&mut pulses),
                        black_box(&pred_coef_q12),
                        black_box(&ltp_coef_q14),
                        black_box(&ar_q13),
                        black_box(&harm_shape_gain_q14),
                        black_box(&tilt_q14),
                        black_box(&lf_shp_q14),
                        black_box(&gains_q16),
                        black_box(&pitch_l),
                        black_box(lambda_q10),
                        black_box(ltp_scale_q14),
                    );

                    pulses
                });
            },
        );
    }

    group.finish();
}

// ── 9. SILK Pitch Analysis Core ───────────────────────────────────────────────

/// Generate voiced signal with specific pitch period
fn generate_pitch_test_signal(length: usize, pitch_period: usize, _fs_khz: i32) -> Vec<i16> {
    let mut frame = vec![0i16; length];
    for i in 0..length {
        let phase = (i % pitch_period) as f32 / pitch_period as f32;
        // Pulse train with decay - more realistic voiced signal
        let pulse = if phase < 0.1 {
            10000.0 * (1.0 - phase * 10.0)
        } else {
            -300.0 * (1.0 - (phase - 0.1) / 0.9)
        };
        frame[i] = pulse as i16;
    }
    frame
}

fn bench_silk_pitch_analysis_core(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_pitch_analysis_core");

    // Test configurations: (fs_khz, nb_subfr, signal_type)
    let configs: &[(i32, usize, &str)] = &[
        (16, 4, "voiced"),
        (16, 4, "unvoiced"),
        (16, 2, "voiced"),
        (8, 4, "voiced"),
        (12, 4, "voiced"),
    ];

    for &(fs_khz, nb_subfr, sig_type) in configs {
        let frame_samples =
            (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_khz as usize;

        let input: Vec<i16> = if sig_type == "voiced" {
            // Pitch period in samples (e.g., 100Hz fundamental at 16kHz = 160 samples)
            let pitch_period = (fs_khz as usize * 1000) / 100;
            generate_pitch_test_signal(frame_samples, pitch_period, fs_khz)
        } else {
            generate_unvoiced_input(frame_samples)
        };

        let prev_lag = if sig_type == "voiced" { 160 } else { 0 };
        let search_thres1_q16: i32 = 3932; // 0.06 in Q16
        let search_thres2_q13: i32 = 983; // 0.12 in Q13
        let complexity = 2; // SILK_PE_MAX_COMPLEX

        group.throughput(Throughput::Elements(frame_samples as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("{}kHz/{}subfr", fs_khz, nb_subfr), sig_type),
            &(
                fs_khz,
                nb_subfr,
                prev_lag,
                search_thres1_q16,
                search_thres2_q13,
                complexity,
            ),
            |b, &(fs_khz, nb_subfr, prev_lag, thres1, thres2, cx)| {
                b.iter(|| {
                    let mut pitch_out = [0i32; MAX_NB_SUBFR];
                    let mut lag_index: i16 = 0;
                    let mut contour_index: i8 = 0;
                    let mut ltp_corr_q15: i32 = 0;

                    silk_pitch_analysis_core(
                        black_box(&input),
                        black_box(&mut pitch_out),
                        black_box(&mut lag_index),
                        black_box(&mut contour_index),
                        black_box(&mut ltp_corr_q15),
                        black_box(prev_lag),
                        black_box(thres1),
                        black_box(thres2),
                        black_box(fs_khz),
                        black_box(cx),
                        black_box(nb_subfr),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_opus_vs_c(c: &mut Criterion) {
    let mut group = c.benchmark_group("opus_vs_c");

    for &(sample_rate, frame_ms, app_str) in &[
        (8000u32, 20usize, "voip"),
        (16000u32, 20usize, "voip"),
        (16000u32, 10usize, "voip"),
        (48000u32, 20usize, "audio"),
        (48000u32, 10usize, "audio"),
    ] {
        let frame_size = sample_rate as usize * frame_ms / 1000;
        let input = sine_f32(frame_size, sample_rate, 440);

        group.throughput(Throughput::Bytes(frame_size as u64 * 2));

        group.bench_with_input(
            BenchmarkId::new(format!("rust/{sample_rate}Hz/{frame_ms}ms"), app_str),
            &(sample_rate, frame_size),
            |b, &(sr, fs)| {
                let app = if sr == 48000 {
                    Application::Audio
                } else {
                    Application::Voip
                };
                let mut enc = OpusEncoder::new(sr as i32, 1, app).unwrap();
                enc.bitrate_bps = if sr == 48000 { 64_000 } else { 20_000 };
                enc.complexity = 0;
                let mut dec = OpusDecoder::new(sr as i32, 1).unwrap();
                let mut output = vec![0u8; 1024];
                let mut pcm = vec![0.0f32; fs];
                b.iter(|| {
                    let len = enc
                        .encode(black_box(&input), fs, black_box(&mut output))
                        .unwrap();
                    dec.decode(black_box(&output[..len]), fs, black_box(&mut pcm))
                        .unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("c/{sample_rate}Hz/{frame_ms}ms"), app_str),
            &(sample_rate, frame_size),
            |b, &(sr, fs)| {
                use opusic_sys::*;
                let mut err = 0i32;
                let enc =
                    unsafe { opus_encoder_create(sr as i32, 1, OPUS_APPLICATION_VOIP, &mut err) };
                assert_eq!(err, OPUS_OK);
                let dec = unsafe { opus_decoder_create(sr as i32, 1, &mut err) };
                assert_eq!(err, OPUS_OK);
                unsafe {
                    opus_encoder_ctl(enc, OPUS_SET_BITRATE_REQUEST, 20_000i32);
                    opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY_REQUEST, 0i32);
                }
                let mut output = vec![0u8; 1024];
                let mut pcm = vec![0.0f32; fs];
                b.iter(|| unsafe {
                    let len = opus_encode_float(
                        enc,
                        black_box(input.as_ptr()),
                        fs as i32,
                        output.as_mut_ptr(),
                        output.len() as i32,
                    );
                    if len > 0 {
                        opus_decode_float(
                            dec,
                            black_box(output.as_ptr()),
                            len,
                            pcm.as_mut_ptr(),
                            fs as i32,
                            0,
                        );
                    }
                });
                unsafe {
                    opus_encoder_destroy(enc);
                    opus_decoder_destroy(dec);
                }
            },
        );
    }

    group.finish();
}

fn bench_celt_autocorr(c: &mut Criterion) {
    let mut group = c.benchmark_group("celt_autocorr");

    for &(n, lag) in &[(320usize, 24usize), (640, 24), (960, 24)] {
        let x = sine_f32(n, 48000, 1000);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("n{n}/lag{lag}"), ""),
            &(n, lag),
            |b, &(ns, lg)| {
                let mut ac = vec![0.0f32; lg + 1];
                b.iter(|| {
                    autocorr(
                        black_box(&x),
                        black_box(&mut ac),
                        None,
                        0,
                        lg,
                        ns,
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft");

    for &nfft in &[60usize, 120, 240, 480] {
        let st = KissFftState::new(nfft).unwrap();
        let mut fout: Vec<KissCpx> = (0..nfft)
            .map(|i| KissCpx::new(i as f32, -(i as f32)))
            .collect();

        group.throughput(Throughput::Elements(nfft as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("opus_fft_impl/n{nfft}"), ""),
            &nfft,
            |b, _| {
                b.iter(|| {
                    // Re-initialize each time to avoid accumulating overflow
                    for (i, v) in fout.iter_mut().enumerate() {
                        *v = KissCpx::new(i as f32, -(i as f32));
                    }
                    opus_fft_impl(black_box(&st), black_box(&mut fout));
                });
            },
        );
    }

    group.finish();
}

fn bench_mdct(c: &mut Criterion) {
    let mut group = c.benchmark_group("mdct");
    let mode = default_mode();

    // CELT 48kHz/20ms: N=1920, lm=3, shift=0 → nfft=480
    // We benchmark forward MDCT (the encode path)
    for &(shift, n) in &[(0usize, 1920usize), (1, 960), (2, 480), (3, 240)] {
        let frame_size = n;
        let overlap = mode.overlap;
        let window = &mode.window[..overlap];
        let input: Vec<f32> = (0..frame_size + overlap)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();
        let mut output = vec![0.0f32; frame_size];

        group.throughput(Throughput::Elements(frame_size as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("forward/shift{shift}/n{frame_size}"), ""),
            &(shift, frame_size),
            |b, &(sh, _fs)| {
                b.iter(|| {
                    mode.mdct.forward(
                        black_box(&input),
                        black_box(&mut output),
                        black_box(window),
                        overlap,
                        sh,
                        1,
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_pvq(c: &mut Criterion) {
    let mut group = c.benchmark_group("pvq");

    // Typical high-band: n=16, k=8 (moderate case)
    // High-freq bands: n=8, k=4
    // Large bands: n=64, k=16
    for &(n, k) in &[(8usize, 4i32), (16, 8), (32, 8), (64, 16)] {
        let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).sin()).collect();
        let mut y = vec![0i32; n];

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("pvq_search/n{n}/k{k}"), ""),
            &(n, k),
            |b, &(ns, ks)| {
                b.iter(|| {
                    pvq_search(black_box(&x), black_box(&mut y), ks, ns);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("encode_pulses/n{n}/k{k}"), ""),
            &(n, k),
            |b, &(ns, ks)| {
                let mut rc = RangeCoder::new_encoder(1024);
                b.iter(|| {
                    rc = RangeCoder::new_encoder(1024);
                    encode_pulses(black_box(&y), ns as u32, ks as u32, &mut rc);
                });
            },
        );
    }

    // alg_quant (combined pvq_search + encode_pulses + exp_rotation)
    for &(n, k) in &[(16usize, 8i32), (64, 16)] {
        let mut x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).sin()).collect();
        let mode = default_mode();

        group.bench_with_input(
            BenchmarkId::new(format!("alg_quant/n{n}/k{k}"), ""),
            &(n, k),
            |b, &(ns, ks)| {
                let mut rc = RangeCoder::new_encoder(1024);
                b.iter(|| {
                    // Reset x to avoid degenerate inputs
                    for (i, v) in x.iter_mut().enumerate() {
                        *v = (i as f32 * 0.5).sin();
                    }
                    rc = RangeCoder::new_encoder(1024);
                    alg_quant(
                        black_box(&mut x),
                        ns,
                        ks,
                        2, // SPREAD_NORMAL
                        1,
                        &mut rc,
                        1.0,
                        true,
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets =
        bench_opus_encode_silk,
        bench_burg_modified,
        bench_autocorr,
        bench_inner_prod,
        bench_lpc_analysis_filter,
        bench_silk_vs_c,
        bench_opus_encode_celt,
        bench_silk_nsq,
        bench_silk_pitch_analysis_core,
        bench_opus_vs_c,
        bench_celt_autocorr,
        bench_fft,
        bench_mdct,
        bench_pvq,
}
criterion_main!(benches);
