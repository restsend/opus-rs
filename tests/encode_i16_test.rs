//! Issue #28: `OpusEncoder::encode_i16()` must produce byte-for-byte
//! identical packets to the float entry point `encode()` fed with the same
//! audio converted via `sample as f32 / 32768.0`, across SILK-only,
//! Hybrid and CELT-only modes.

use opus_rs::{Application, OpusDecoder, OpusEncoder};

/// Deterministic test signal: speech-like sine + LCG noise, moderate level.
struct SignalGen {
    lcg: u32,
    t: usize,
    fs: usize,
}

impl SignalGen {
    fn new(fs: usize) -> Self {
        Self {
            lcg: 0x1234_5678,
            t: 0,
            fs,
        }
    }

    fn next_i16(&mut self) -> i16 {
        // LCG noise in [-1, 1]
        self.lcg = self.lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (self.lcg as i32 >> 16) as f32 / 32768.0;
        // 220 Hz fundamental + harmonics, slowly drifting amplitude
        let amp = 0.35 * (0.75 + 0.25 * (self.t as f32 / (self.fs as f32 * 0.7)).sin());
        let tone = (2.0 * core::f32::consts::PI * 220.0 * self.t as f32 / self.fs as f32).sin()
            + 0.5 * (2.0 * core::f32::consts::PI * 660.0 * self.t as f32 / self.fs as f32).sin();
        let v = amp * (0.6 * tone / 1.5 + 0.4 * noise);
        self.t += 1;
        (v * 32767.0 * 0.6).clamp(-32768.0, 32767.0) as i16
    }

    fn frame_i16(&mut self, frame_size: usize, channels: usize) -> Vec<i16> {
        (0..frame_size * channels)
            .map(|_| self.next_i16())
            .collect()
    }
}

fn to_f32(i16_pcm: &[i16]) -> Vec<f32> {
    i16_pcm.iter().map(|&s| s as f32 / 32768.0).collect()
}

#[derive(Debug)]
struct EncConfig {
    fs: i32,
    channels: usize,
    app: Application,
    bitrate: i32,
    complexity: i32,
    use_cbr: bool,
    fec: bool,
    loss: i32,
}

impl EncConfig {
    fn make_encoders(&self) -> (OpusEncoder, OpusEncoder) {
        let mut a = OpusEncoder::new(self.fs, self.channels, self.app).unwrap();
        let mut b = OpusEncoder::new(self.fs, self.channels, self.app).unwrap();
        for enc in [&mut a, &mut b] {
            enc.bitrate_bps = self.bitrate;
            enc.complexity = self.complexity;
            enc.use_cbr = self.use_cbr;
            enc.use_inband_fec = self.fec;
            enc.packet_loss_perc = self.loss;
        }
        (a, b)
    }
}

/// Encode the same PCM through the f32 path (converted) and the native i16
/// path; the packet bytes must match exactly on every frame.
fn assert_parity(cfg: &EncConfig, frame_size: usize, n_frames: usize) {
    let (mut enc_f32, mut enc_i16) = cfg.make_encoders();
    let mut signal = SignalGen::new(cfg.fs as usize);
    let mut out_f32 = vec![0u8; 4000];
    let mut out_i16 = vec![0u8; 4000];

    for frame_idx in 0..n_frames {
        let pcm_i16 = signal.frame_i16(frame_size, cfg.channels);
        let pcm_f32 = to_f32(&pcm_i16);

        let n_a = enc_f32
            .encode(&pcm_f32, frame_size, &mut out_f32)
            .unwrap_or_else(|e| panic!("f32 path failed (cfg {cfg:?} frame {frame_idx}): {e}"));
        let n_b = enc_i16
            .encode_i16(&pcm_i16, frame_size, &mut out_i16)
            .unwrap_or_else(|e| panic!("i16 path failed (cfg {cfg:?} frame {frame_idx}): {e}"));

        assert_eq!(
            n_a, n_b,
            "packet size mismatch (cfg {cfg:?} frame {frame_idx})"
        );
        assert_eq!(
            &out_f32[..n_a],
            &out_i16[..n_b],
            "packet bytes mismatch (cfg {cfg:?} frame {frame_idx})"
        );
    }
}

#[test]
fn silk_only_parity_matrix() {
    // 8/12/16 kHz always selects SILK-only in this port (CELT needs >= 24 kHz).
    for &fs in &[8000i32, 12000, 16000] {
        for &channels in &[1usize, 2] {
            for &app in &[Application::Voip, Application::Audio] {
                for &complexity in &[0i32, 5, 10] {
                    // 20 ms and 40 ms (multi-frame SILK packet), VBR and CBR.
                    let frame_size = fs as usize / 50; // 20 ms
                    let frame_size40 = fs as usize / 25; // 40 ms
                    for &use_cbr in &[false, true] {
                        let cfg = EncConfig {
                            fs,
                            channels,
                            app,
                            bitrate: 24000,
                            complexity,
                            use_cbr,
                            fec: false,
                            loss: 0,
                        };
                        assert_parity(&cfg, frame_size, 6);
                        assert_parity(&cfg, frame_size40, 6);
                    }
                }
            }
        }
    }
}

#[test]
fn silk_voip_fec_parity_stereo_and_mono() {
    // In-band FEC exercises the LBRR section, including the stereo headers
    // written before every LBRR payload (issue #27 fix must keep holding on
    // the new i16 path).
    for &fs in &[8000i32, 16000] {
        for &channels in &[1usize, 2] {
            let cfg = EncConfig {
                fs,
                channels,
                app: Application::Voip,
                bitrate: 32000,
                complexity: 7,
                use_cbr: false,
                fec: true,
                loss: 20,
            };
            assert_parity(&cfg, fs as usize / 50, 8);
        }
    }
}

#[test]
fn hybrid_parity_24k_and_48k() {
    // 24 kHz defaults to Superwideband -> forced Hybrid.
    let cfg24 = EncConfig {
        fs: 24000,
        channels: 2,
        app: Application::Voip,
        bitrate: 32000,
        complexity: 7,
        use_cbr: false,
        fec: false,
        loss: 0,
    };
    assert_parity(&cfg24, 24000_usize / 50, 6);

    // 48 kHz Fullband + low bitrate -> SILK-only -> Hybrid.
    let cfg48 = EncConfig {
        fs: 48000,
        channels: 1,
        app: Application::Voip,
        bitrate: 32000,
        complexity: 7,
        use_cbr: false,
        fec: false,
        loss: 0,
    };
    assert_parity(&cfg48, 48000_usize / 50, 6);

    // Hybrid with FEC.
    let cfg48_fec = EncConfig {
        fs: 48000,
        channels: 2,
        app: Application::Voip,
        bitrate: 64000,
        complexity: 8,
        use_cbr: false,
        fec: true,
        loss: 15,
    };
    assert_parity(&cfg48_fec, 48000_usize / 50, 6);
}

#[test]
fn celt_only_parity_48k() {
    // 48 kHz VoIP @ 64 kbps crosses the CELT-only threshold.
    for &app in &[Application::Voip, Application::Audio] {
        let cfg = EncConfig {
            fs: 48000,
            channels: 2,
            app,
            bitrate: 96000,
            complexity: 9,
            use_cbr: false,
            fec: false,
            loss: 0,
        };
        assert_parity(&cfg, 48000_usize / 100, 6); // 10 ms
        assert_parity(&cfg, 48000_usize / 50, 6); // 20 ms
    }

    // RestrictedLowDelay is always CELT-only with zero delay compensation
    // (exercises the raw-input CELT arm).
    let cfg_rld = EncConfig {
        fs: 48000,
        channels: 1,
        app: Application::RestrictedLowDelay,
        bitrate: 96000,
        complexity: 9,
        use_cbr: false,
        fec: false,
        loss: 0,
    };
    assert_parity(&cfg_rld, 48000_usize / 50, 6);
}

#[test]
fn encode_i16_stream_stays_in_sync() {
    // Long stream: filter/range-coder state evolution must stay identical.
    let cfg = EncConfig {
        fs: 16000,
        channels: 2,
        app: Application::Voip,
        bitrate: 28000,
        complexity: 6,
        use_cbr: false,
        fec: true,
        loss: 10,
    };
    assert_parity(&cfg, 16000_usize / 50, 60);
}

#[test]
fn encode_i16_roundtrip_decodes() {
    // Sanity: the i16 packet decodes to the same samples as the f32 packet.
    let fs = 16000;
    let channels = 1;
    let frame_size = 320; // 20 ms
    let mut enc_a = OpusEncoder::new(fs, channels, Application::Voip).unwrap();
    let mut enc_b = OpusEncoder::new(fs, channels, Application::Voip).unwrap();
    enc_a.bitrate_bps = 24000;
    enc_b.bitrate_bps = 24000;
    let mut dec = OpusDecoder::new(fs, channels).unwrap();

    let mut signal = SignalGen::new(fs as usize);
    let mut pkt_a = vec![0u8; 4000];
    let mut pkt_b = vec![0u8; 4000];
    let mut out_f32 = vec![0.0f32; frame_size];
    for _ in 0..10 {
        let pcm_i16 = signal.frame_i16(frame_size, channels);
        let n_a = enc_a
            .encode(&to_f32(&pcm_i16), frame_size, &mut pkt_a)
            .unwrap();
        let n_b = enc_b.encode_i16(&pcm_i16, frame_size, &mut pkt_b).unwrap();
        assert_eq!(n_a, n_b);
        assert_eq!(&pkt_a[..n_a], &pkt_b[..n_b]);

        let got = dec.decode(&pkt_b[..n_b], frame_size, &mut out_f32).unwrap();
        assert_eq!(got, frame_size);
        assert!(
            out_f32.iter().any(|&v| v.abs() > 1e-4),
            "decoded output is digital silence"
        );
    }
}

#[test]
fn encode_i16_validation_parity() {
    let mut enc = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
    let mut out = vec![0u8; 4000];

    // An empty output buffer is an invalid argument, as in libopus (a
    // 1-byte buffer takes a ToC-only packet).
    let tiny = &mut [0u8; 0][..];
    assert_eq!(
        enc.encode_i16(&[0i16; 320], 320, tiny),
        Err(opus_rs::error::BAD_ARG)
    );

    // Input shorter than frame_size * channels must be rejected, not panic.
    assert_eq!(
        enc.encode_i16(&[0i16; 319], 320, &mut out),
        Err("Input buffer too small for frame")
    );

    // Invalid frame size mirrors encode().
    assert!(enc.encode_i16(&[0i16; 333], 333, &mut out).is_err());
}
