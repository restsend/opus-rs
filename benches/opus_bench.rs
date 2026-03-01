use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use opus_rs::silk::lpc_analysis::silk_burg_modified_fix;
use opus_rs::silk::sigproc_fix::{
    silk_autocorr, silk_inner_prod_aligned, silk_lpc_analysis_filter,
};
use opus_rs::{Application, OpusEncoder};

// ── helpers ──────────────────────────────────────────────────────────────────

/// 440 Hz sine wave at the given sample rate.
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

// ── 1. Full SILK-only encoder (most representative) ──────────────────────────

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

// ── 2. Burg modified LPC analysis ────────────────────────────────────────────

fn bench_burg_modified(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_burg_modified_fix");

    // Typical SILK 16 kHz: d=16, subfr_length=80+16=96, nb_subfr=4  (320+64 samples)
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

// ── 3. Autocorrelation ────────────────────────────────────────────────────────

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

// ── 4. Inner product ──────────────────────────────────────────────────────────

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

// ── 5. LPC analysis filter ────────────────────────────────────────────────────

fn bench_lpc_analysis_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_lpc_analysis_filter");

    // Typical params: order=16, len=320
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

// ── 6. CELT-only encoder ──────────────────────────────────────────────────────

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

// ── 7. SILK encoder vs C (opusic-sys) side-by-side ───────────────────────────
// Rust encoder uses complexity=0 (fast mode); two C variants are benchmarked:
//   "c_cx0"  – C at complexity=0 (same as Rust)
//   "c_cx9"  – C at complexity=9 (default production quality)

fn bench_silk_vs_c(c: &mut Criterion) {
    let mut group = c.benchmark_group("silk_vs_c");

    for &(sample_rate, frame_ms) in &[(8000u32, 20usize), (16000u32, 20usize), (16000u32, 10usize)]
    {
        let frame_size = sample_rate as usize * frame_ms / 1000;
        let input = sine_f32(frame_size, sample_rate, 440);

        group.throughput(Throughput::Bytes(frame_size as u64 * 2));

        // ── Rust encoder (complexity=0) ───────────────────────────────────────
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

        // ── C encoder – complexity=0 (fair comparison) ────────────────────────
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

        // ── C encoder – complexity=9 (default production quality) ─────────────
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

criterion_group!(
    benches,
    bench_opus_encode_silk,
    bench_burg_modified,
    bench_autocorr,
    bench_inner_prod,
    bench_lpc_analysis_filter,
    bench_silk_vs_c,
);
criterion_main!(benches);
