//! Miri replay: drives opus-rs through the paths the differential fuzzers
//! cover, with small deterministic inputs (Miri runs about 1000x slower).
//! Miri reports undefined behaviour: out-of-bounds or unaligned accesses in
//! unsafe/SIMD code, reads of uninitialized memory (the MaybeUninit-built
//! codec structs), aliasing violations. Build with
//! `RUSTFLAGS=-Ctarget-feature=+avx2,+fma` to take the AVX2/FMA kernels.
use opus_rs::range_coder::RangeCoder;
use opus_rs::{Application, OpusDecoder, OpusEncoder};

/// Small deterministic PRNG (LCG), so runs are reproducible.
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 23) as f32 * 2.0 - 1.0
    }
}

fn scale() -> usize {
    if cfg!(miri) { 1 } else { 20 }
}

#[test]
fn decoder_random_packets() {
    let mut r = Rng(1);
    for (fs, ch) in [(48000, 2), (16000, 1), (8000, 2)] {
        let mut d = OpusDecoder::new(fs, ch).unwrap();
        let mut out = vec![0f32; 5760 * ch];
        for _ in 0..4 * scale() {
            let len = 2 + r.byte() as usize % 60;
            let pkt: Vec<u8> = (0..len).map(|_| r.byte()).collect();
            let _ = d.decode(&pkt, 5760 * fs as usize / 48000, &mut out);
            if r.byte() & 3 == 0 {
                let _ = d.decode_fec(&pkt, fs as usize / 50, &mut out);
            }
            if r.byte() & 7 == 0 {
                let _ = d.decode(&[], fs as usize / 50, &mut out);
            }
        }
    }
}

#[test]
fn decoder_encoded_mode_switches() {
    // Valid streams through SILK, hybrid and CELT, including the transitions.
    let mut r = Rng(2);
    for simd in [false, true] {
        let arch = if simd { opus_rs::Arch::native() } else { opus_rs::Arch::Scalar };
        opus_rs::fuzzing::force_arch(Some(arch));
        let (fs, ch) = (48000, 1);
        let mut e = OpusEncoder::new(fs, ch, Application::Audio).unwrap();
        let mut d = OpusDecoder::new(fs, ch).unwrap();
        let frame = 960;
        for (i, kbps) in [8, 24, 64, 12, 128].iter().cycle().take(5 * scale()).enumerate() {
            e.bitrate_bps = kbps * 1000;
            e.use_inband_fec = i % 2 == 0;
            e.packet_loss_perc = 20;
            let pcm: Vec<f32> = (0..frame * ch).map(|_| r.f() * 0.3).collect();
            let mut pkt = [0u8; 1500];
            let n = e.encode(&pcm, frame, &mut pkt).unwrap();
            let mut out = vec![0f32; frame * ch];
            d.decode(&pkt[..n], frame, &mut out).unwrap();
            let _ = d.decode_fec(&pkt[..n], frame, &mut out);
        }
    }
}

#[test]
fn encoder_settings_and_signals() {
    let mut r = Rng(3);
    for simd in [false, true] {
        let arch = if simd { opus_rs::Arch::native() } else { opus_rs::Arch::Scalar };
        opus_rs::fuzzing::force_arch(Some(arch));
        for (fs, ch, app, ms) in [
            (48000, 2, Application::Audio, 20),
            (16000, 2, Application::Voip, 20),
            (48000, 1, Application::RestrictedLowDelay, 5),
            (8000, 1, Application::Voip, 60),
        ] {
            let mut e = OpusEncoder::new(fs, ch, app).unwrap();
            let frame = fs as usize * ms / 1000;
            for f in 0..2 * scale() {
                e.bitrate_bps = [6000, 32000, 128000, 1_500_000][f % 4];
                e.use_cbr = f % 3 == 0;
                e.complexity = (f % 11) as i32;
                let pcm: Vec<f32> = (0..frame * ch)
                    .map(|i| match f % 3 {
                        0 => 0.0,
                        1 => 0.5 + 0.1 * r.f(),
                        _ => (i as f32 * 0.01).sin() * 1.5,
                    })
                    .collect();
                let mut pkt = [0u8; 4000];
                let _ = e.encode(&pcm, frame, &mut pkt);
            }
        }
    }
}

#[test]
fn range_coder_api() {
    // Safe calls with bad arguments may panic; they must not be UB.
    let mut r = Rng(4);
    for _ in 0..8 * scale() {
        let size = r.byte() as u32 % 32;
        let mut enc = RangeCoder::new_encoder(size);
        let mut ops = Vec::new();
        for _ in 0..32 {
            ops.push((r.byte() % 6, r.next(), r.next()));
        }
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for &(op, a, b) in &ops {
                match op {
                    0 => enc.enc_uint(a % 1000, 1000),
                    1 => enc.enc_bits(a & 0xff, 8),
                    2 => enc.shrink(a % 4096),
                    3 => enc.encode_icdf((a % 8) as i32, &[200, 100, 0], 8),
                    4 => enc.done(),
                    _ => {
                        let mut d = RangeCoder::new_decoder(&b.to_le_bytes());
                        let _ = d.decode_icdf(&[200, 100, 0], 8);
                        let _ = d.dec_uint(a % 100 + 2);
                    }
                }
            }
        }));
    }
}

#[test]
fn kernels() {
    let mut r = Rng(5);
    let mode = opus_rs::modes::default_mode();
    for simd in [false, true] {
        let arch = if simd { opus_rs::Arch::native() } else { opus_rs::Arch::Scalar };
        opus_rs::fuzzing::force_arch(Some(arch));
        for n0 in [2usize, 16, 30, 64, 128] {
            let mut x: Vec<f32> = (0..n0 * 2).map(|_| r.f()).collect();
            opus_rs::bands::haar1(&mut x, n0, 1);
            opus_rs::bands::haar1(&mut x, n0 / 2, 2);
        }
        for (n, k) in [(2usize, 1i32), (8, 5), (16, 40), (37, 11), (176, 128)] {
            let x: Vec<f32> = (0..n).map(|_| r.f()).collect();
            let mut y = vec![0i32; n + 8];
            opus_rs::pvq::pvq_search(&x, &mut y, k, n, opus_rs::Arch::select());
        }
        for len in [0usize, 1, 7, 8, 15, 33, 240] {
            let x: Vec<i16> = (0..len).map(|_| r.next() as i16).collect();
            let (mut e, mut s) = (0, 0);
            opus_rs::silk::sigproc_fix::silk_sum_sqr_shift(&mut e, &mut s, &x, len);
        }
        for (n, ord) in [(1usize, 1usize), (7, 4), (64, 16), (240, 24)] {
            let x: Vec<f32> = (0..ord + n).map(|_| r.f()).collect();
            let num: Vec<f32> = (0..ord).map(|_| r.f() * 0.1).collect();
            let mut y = vec![0f32; n];
            opus_rs::celt_lpc::celt_fir(&x, &num, &mut y, n, ord, opus_rs::Arch::select());
        }
        for (t0, t1, n, overlap) in [(15usize, 15usize, 120usize, 120usize), (100, 400, 240, 120), (1024, 15, 960, 0)] {
            let hist = 1026;
            let x: Vec<f32> = (0..hist + n).map(|_| r.f()).collect();
            let mut y = vec![0f32; hist + n];
            opus_rs::fuzzing::comb_filter(&mut y, &x, hist, hist, t0, t1, n, 0.5, 0.3, 1, 2, mode.window, overlap);
        }
    }
}
