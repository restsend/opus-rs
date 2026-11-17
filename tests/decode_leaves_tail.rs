//! The decoder writes exactly the samples it returns: `output` past them is
//! left unchanged, as in libopus, for every mode, losses, FEC and the
//! neural concealment and enhancement when those features are on.
use opus_rs::{Application, OpusDecoder, OpusEncoder};

const MARK: f32 = 0.123;

fn signal(t: usize, fs: usize, c: usize) -> f32 {
    let t = t as f32 / fs as f32;
    let mut x = 0.0;
    for h in 1..8 {
        x += (2.0 * std::f32::consts::PI * (180.0 + 20.0 * c as f32) * h as f32 * t).sin()
            / h as f32;
    }
    0.3 * x * (0.6 + 0.4 * (3.0 * t).sin())
}

/// Decodes a stream of `app` packets with losses and FEC into a 120 ms buffer
/// and checks the samples past each call's result.
fn check(
    fs: usize,
    ch: usize,
    dec_ch: usize,
    app: Application,
    ms10: usize,
    kbps: i32,
    complexity: i32,
) {
    let n = fs * ms10 / 10000;
    let mut enc = OpusEncoder::new(fs as i32, ch, app).unwrap();
    enc.bitrate_bps = kbps * 1000;
    enc.use_inband_fec = true;
    enc.packet_loss_perc = 20;
    let mut dec = OpusDecoder::new(fs as i32, dec_ch).unwrap();
    dec.set_complexity(complexity).unwrap();
    let mut packets = Vec::new();
    for k in 0..40 {
        let pcm: Vec<f32> = (k * n..(k + 1) * n)
            .flat_map(|t| (0..ch).map(move |c| signal(t, fs, c)))
            .collect();
        let mut p = [0u8; 1500];
        let len = enc.encode(&pcm, n, &mut p).unwrap();
        packets.push(p[..len].to_vec());
    }
    let mut out = vec![MARK; fs * 120 / 1000 * dec_ch];
    for (k, p) in packets.iter().enumerate() {
        out.fill(MARK);
        let r = match k % 7 {
            3 => dec.decode(&[], n, &mut out),
            4 => dec.decode_fec(p, n, &mut out),
            // A ToC byte alone is a valid packet for frame count code 0 only.
            5 if p[0] & 3 == 0 => dec.decode(&p[..1], n, &mut out),
            _ => dec.decode(p, n, &mut out),
        };
        let r = r.unwrap();
        assert!(
            out[r * dec_ch..].iter().all(|&x| x == MARK),
            "{fs} Hz {ch}->{dec_ch} ch {app:?} {ms10}/10 ms {kbps} kb/s complexity {complexity}: \
             packet {k} wrote past its {r} samples"
        );
    }
}

#[test]
fn decoder_leaves_the_output_past_its_samples_unchanged() {
    for &(fs, app, kbps) in &[
        (8000, Application::Voip, 12),
        (16000, Application::Voip, 20),
        (48000, Application::Voip, 24),
        (48000, Application::Audio, 64),
        (48000, Application::RestrictedLowDelay, 96),
    ] {
        for &ms10 in &[25, 100, 200, 600] {
            for &(ch, dec_ch) in &[(1, 1), (2, 2), (2, 1), (1, 2)] {
                check(fs, ch, dec_ch, app, ms10, kbps * ch as i32, 0);
            }
        }
    }
}

/// Deep PLC (complexity 5 and up) and LACE/NoLACE (6, 7 and up) write their
/// output through other paths.
#[cfg(any(feature = "deep-plc", feature = "osce"))]
#[test]
fn neural_decoder_paths_leave_the_output_past_their_samples_unchanged() {
    for complexity in [5, 6, 10] {
        for &(fs, app, kbps) in &[
            (16000, Application::Voip, 20),
            (48000, Application::Voip, 24),
        ] {
            check(fs, 1, 1, app, 200, kbps, complexity);
            check(fs, 2, 2, app, 200, 2 * kbps, complexity);
        }
    }
}
