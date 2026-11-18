//! The encoder API against libopus's: the controls of `opus_encoder_ctl`
//! and the `opus_encode*` entry points.
//!
//! Packets are compared as the differential fuzzer does: libopus decodes
//! every opus-rs packet with the encoder's final range and the ToC agrees;
//! RestrictedLowDelay (CELT-only) packets are byte-identical.
use core::ffi::c_void;
use opus_rs::{Application, OpusEncoder};
use opusref::*;

/// A voiced test signal, frame `f` of `frame` samples per channel.
fn signal(fs: i32, ch: usize, frame: usize, f: usize) -> Vec<f32> {
    let pitch = 150.0 + 40.0 * ((f as f32) * 0.3).sin();
    (0..frame * ch)
        .map(|i| {
            let t = (f * frame + i / ch) as f32 / fs as f32;
            (1..12)
                .map(|h| ((h as f32) * pitch * t * 6.2832).sin() / h as f32)
                .sum::<f32>()
                * if i % ch == 0 { 0.2 } else { 0.13 }
        })
        .collect()
}

/// An opus-rs encoder, a libopus encoder with the same settings, and a
/// libopus decoder for the opus-rs packets.
struct Pair {
    rs: OpusEncoder,
    c: *mut c_void,
    dec: *mut c_void,
    fs: i32,
    ch: usize,
    app: i32,
    /// The signal's gain, and the right channel's.
    scale: f32,
    right: f32,
    /// Encode through `encode_i16` / `opus_encode`.
    int16: bool,
}

impl Pair {
    fn new(fs: i32, ch: usize, app: Application) -> Self {
        let mut err = 0;
        let c = unsafe { opus_encoder_create(fs, ch as i32, app as i32, &mut err) };
        assert!(!c.is_null() && err == 0);
        let dec = unsafe { opus_decoder_create(fs, ch as i32, &mut err) };
        assert!(!dec.is_null() && err == 0);
        Pair {
            rs: OpusEncoder::new(fs, ch, app).unwrap(),
            c,
            dec,
            fs,
            ch,
            app: app as i32,
            scale: 1.0,
            right: 1.0,
            int16: false,
        }
    }

    fn ctl(&mut self, req: i32, value: i32) -> i32 {
        unsafe { opus_encoder_ctl(self.c, req, value) }
    }

    fn get(&mut self, req: i32) -> i32 {
        let mut v = 0i32;
        let r = unsafe { opus_encoder_ctl(self.c, req, &mut v as *mut i32) };
        assert_eq!(r, 0, "get {req}");
        v
    }

    /// Encodes `pcm` (`frame` samples per channel) on both and checks the
    /// packets, which code `coded` samples.
    fn encode_pcm(&mut self, pcm: &[f32], frame: usize, coded: usize, what: &str) {
        let (mut a, mut b) = ([0u8; 1500], [0u8; 1500]);
        let n = self.rs.encode(pcm, frame, &mut a);
        let m = unsafe { opus_encode_float(self.c, pcm.as_ptr(), frame as i32, b.as_mut_ptr(), 1500) };
        assert_eq!(n.is_ok(), m > 0, "{what}: {n:?} vs {m}");
        let Ok(n) = n else { return };
        self.check(&a[..n], &b[..m as usize], coded, what);
    }

    /// Encodes `pcm` on both and asserts the same result, packet bytes and
    /// final range.
    fn encode_exact(&mut self, pcm: &[f32], frame: usize, what: &str) {
        let (mut a, mut b) = ([0u8; 1500], [0u8; 1500]);
        let n = self.rs.encode(pcm, frame, &mut a);
        let m = unsafe { opus_encode_float(self.c, pcm.as_ptr(), frame as i32, b.as_mut_ptr(), 1500) };
        assert_eq!(n.map_or_else(|e| opus_rs::error::code(e).unwrap(), |n| n as i32), m, "{what}");
        if m > 0 {
            assert_eq!(a[..m as usize], b[..m as usize], "{what}: bytes");
        }
        let mut r = 0u32;
        unsafe { opus_encoder_ctl(self.c, GET_FINAL_RANGE, &mut r as *mut u32) };
        assert_eq!(self.rs.final_range(), r, "{what}: final range");
    }

    /// Encodes frame `f` on both and checks the packets.
    fn encode(&mut self, frame: usize, f: usize, what: &str) {
        let pcm: Vec<f32> = signal(self.fs, self.ch, frame, f)
            .iter()
            .enumerate()
            .map(|(i, x)| x * self.scale * if i % self.ch == 1 { self.right } else { 1.0 })
            .collect();
        let (mut a, mut b) = ([0u8; 1500], [0u8; 1500]);
        let (n, m) = if self.int16 {
            let pcm: Vec<i16> = pcm.iter().map(|&x| (x * 32768.0).round().clamp(-32768.0, 32767.0) as i16).collect();
            let n = self.rs.encode_i16(&pcm, frame, &mut a);
            let m = unsafe { opus_encode(self.c, pcm.as_ptr(), frame as i32, b.as_mut_ptr(), 1500) };
            (n, m)
        } else {
            let n = self.rs.encode(&pcm, frame, &mut a);
            let m = unsafe { opus_encode_float(self.c, pcm.as_ptr(), frame as i32, b.as_mut_ptr(), 1500) };
            (n, m)
        };
        assert_eq!(n.is_ok(), m > 0, "{what}: {n:?} vs {m}");
        let Ok(n) = n else { return };
        self.check(&a[..n], &b[..m as usize], frame, what);
    }

    /// The packet oracles: libopus decodes `pkt` with the encoder's final
    /// range, the ToC equals libopus's `reference`, and CELT-only streams
    /// are byte-identical.
    fn check(&mut self, pkt: &[u8], reference: &[u8], frame: usize, what: &str) {
        let mut out = vec![0f32; frame * self.ch];
        let d = unsafe {
            opus_decode_float(self.dec, pkt.as_ptr(), pkt.len() as i32, out.as_mut_ptr(), frame as i32, 0)
        };
        assert_eq!(d, frame as i32, "{what}: libopus decodes the opus-rs packet");
        let mut r = 0u32;
        unsafe { opus_decoder_ctl(self.dec, GET_FINAL_RANGE, &mut r as *mut u32) };
        if pkt.len() > 1 {
            assert_eq!(r, self.rs.final_range(), "{what}: final range");
        }
        assert_eq!(pkt[0] & 0xfc, reference[0] & 0xfc, "{what}: ToC");
        if self.app == Application::RestrictedLowDelay as i32 || self.app == Application::RestrictedCelt as i32 {
            assert_eq!(pkt, reference, "{what}: bytes");
        }
    }
}

impl Drop for Pair {
    fn drop(&mut self) {
        unsafe {
            opus_encoder_destroy(self.c);
            opus_decoder_destroy(self.dec);
        }
    }
}

const CONFIGS: [(i32, usize, Application, usize, i32); 6] = [
    (48000, 2, Application::RestrictedLowDelay, 960, 96000),
    (48000, 1, Application::RestrictedLowDelay, 240, 64000),
    (48000, 2, Application::Audio, 960, 64000),
    (16000, 1, Application::Voip, 320, 16000),
    (24000, 2, Application::Voip, 480, 32000),
    (8000, 1, Application::Audio, 160, 12000),
];

/// After a reset both encoders start over: the packets after it agree as
/// the first packets of a stream do, also for a second reset.
#[test]
fn reset_matches_opus_reset_state() {
    for (fs, ch, app, frame, rate) in CONFIGS {
        let mut p = Pair::new(fs, ch, app);
        p.rs.bitrate_bps = rate;
        p.ctl(SET_BITRATE, rate);
        for f in 0..30 {
            if f == 9 || f == 20 {
                p.rs.reset();
                p.ctl(RESET_STATE, 0);
            }
            p.encode(frame, f, &format!("{fs} {ch} {app:?} frame {f}"));
        }
        // A reset keeps the settings.
        assert_eq!(p.get(GET_BITRATE), rate);
    }
}

/// A reset encoder codes CELT-only streams like a new one.
#[test]
fn reset_codes_like_a_new_encoder() {
    let (fs, ch, frame) = (48000, 2, 960);
    let mut a = OpusEncoder::new(fs, ch, Application::RestrictedLowDelay).unwrap();
    let mut out = [0u8; 1500];
    for f in 0..7 {
        a.encode(&signal(fs, ch, frame, f), frame, &mut out).unwrap();
    }
    a.reset();
    let mut b = OpusEncoder::new(fs, ch, Application::RestrictedLowDelay).unwrap();
    for f in 0..10 {
        let pcm = signal(fs, ch, frame, f + 3);
        let mut out2 = [0u8; 1500];
        let n = a.encode(&pcm, frame, &mut out).unwrap();
        let m = b.encode(&pcm, frame, &mut out2).unwrap();
        assert_eq!(out[..n], out2[..m], "frame {f}");
    }
}

fn code(r: Result<(), &str>) -> i32 {
    r.map_or_else(|e| opus_rs::error::code(e).unwrap(), |()| 0)
}

/// `OPUS_SET_LSB_DEPTH`: quiet signals code alike at every depth, through
/// both entry points; out-of-range depths are rejected.
#[test]
fn lsb_depth_matches_libopus() {
    for (fs, ch, app, frame, rate) in CONFIGS {
        for depth in [8, 11, 16, 20, 24] {
            for int16 in [false, true] {
                let mut p = Pair::new(fs, ch, app);
                p.rs.bitrate_bps = rate;
                p.ctl(SET_BITRATE, rate);
                p.scale = 0.004;
                p.int16 = int16;
                for bad in [7, 25, -1] {
                    assert_eq!(code(p.rs.set_lsb_depth(bad)), p.ctl(SET_LSB_DEPTH, bad));
                }
                assert_eq!(code(p.rs.set_lsb_depth(depth)), p.ctl(SET_LSB_DEPTH, depth));
                assert_eq!(p.rs.lsb_depth(), p.get(GET_LSB_DEPTH));
                for f in 0..12 {
                    p.encode(frame, f, &format!("{fs} {ch} {app:?} depth {depth} i16 {int16} frame {f}"));
                }
            }
        }
    }
}

/// `OPUS_SET_FORCE_CHANNELS`: stereo input coded mono or stereo at any
/// rate; invalid counts are rejected.
#[test]
fn force_channels_matches_libopus() {
    for (fs, ch, app, frame, _) in CONFIGS {
        for rate in [12000, 64000] {
            for (force, value) in [(Some(1), 1), (Some(2), 2), (None, -1000)] {
                let mut p = Pair::new(fs, ch, app);
                p.rs.bitrate_bps = rate;
                p.ctl(SET_BITRATE, rate);
                for (bad, c) in [(Some(0), 0), (Some(3), 3)] {
                    assert_eq!(code(p.rs.set_force_channels(bad)), p.ctl(SET_FORCE_CHANNELS, c));
                }
                assert_eq!(code(p.rs.set_force_channels(force)), p.ctl(SET_FORCE_CHANNELS, value));
                assert_eq!(p.rs.force_channels().map_or(-1000, |v| v as i32), p.get(GET_FORCE_CHANNELS));
                for f in 0..12 {
                    p.encode(frame, f, &format!("{fs} {ch} {app:?} {rate} force {force:?} frame {f}"));
                }
            }
        }
    }
}

/// `OPUS_SET_PREDICTION_DISABLED`: streams without inter-frame prediction
/// in every mode.
#[test]
fn prediction_disabled_matches_libopus() {
    for (fs, ch, app, frame, rate) in CONFIGS {
        let mut p = Pair::new(fs, ch, app);
        p.rs.bitrate_bps = rate;
        p.ctl(SET_BITRATE, rate);
        p.rs.set_prediction_disabled(true);
        assert_eq!(p.ctl(SET_PREDICTION_DISABLED, 1), 0);
        assert_eq!(p.ctl(SET_PREDICTION_DISABLED, 2), -1);
        assert_eq!(p.rs.prediction_disabled() as i32, p.get(GET_PREDICTION_DISABLED));
        for f in 0..16 {
            if f == 10 {
                p.rs.set_prediction_disabled(false);
                p.ctl(SET_PREDICTION_DISABLED, 0);
            }
            p.encode(frame, f, &format!("{fs} {ch} {app:?} frame {f}"));
        }
    }
}

/// `OPUS_SET_PHASE_INVERSION_DISABLED`: stereo CELT streams with intensity
/// stereo code alike either way.
#[test]
fn phase_inversion_matches_libopus() {
    for disabled in [false, true] {
        for (fs, ch, app, frame, _) in CONFIGS {
            let mut p = Pair::new(fs, ch, app);
            // A low rate codes the upper bands as intensity stereo.
            p.rs.bitrate_bps = 24000;
            p.ctl(SET_BITRATE, 24000);
            // Out-of-phase channels.
            p.right = -1.5;
            assert_eq!(p.rs.phase_inversion_disabled() as i32, p.get(GET_PHASE_INVERSION_DISABLED));
            p.rs.set_phase_inversion_disabled(disabled);
            assert_eq!(p.ctl(SET_PHASE_INVERSION_DISABLED, disabled as i32), 0);
            assert_eq!(p.ctl(SET_PHASE_INVERSION_DISABLED, -1), -1);
            assert_eq!(p.rs.phase_inversion_disabled() as i32, p.get(GET_PHASE_INVERSION_DISABLED));
            for f in 0..16 {
                p.encode(frame, f, &format!("{fs} {ch} {app:?} disabled {disabled} frame {f}"));
            }
        }
    }
}

/// `OPUS_SET_EXPERT_FRAME_DURATION`: each call codes the set duration from
/// the start of a longer input, which the analysis reads ahead into.
#[test]
fn expert_frame_duration_matches_libopus() {
    use opus_rs::FrameDuration::*;
    for (fs, ch, app, _, rate) in CONFIGS {
        for (d, ms) in [(Ms2_5, 2.5), (Ms10, 10.0), (Ms20, 20.0), (Ms60, 60.0), (Ms100, 100.0)] {
            let mut p = Pair::new(fs, ch, app);
            p.rs.bitrate_bps = rate;
            p.ctl(SET_BITRATE, rate);
            p.rs.complexity = 10;
            p.ctl(SET_COMPLEXITY, 10);
            p.rs.set_expert_frame_duration(d);
            assert_eq!(p.ctl(SET_EXPERT_FRAME_DURATION, d as i32), 0);
            assert_eq!(p.ctl(SET_EXPERT_FRAME_DURATION, 5010), -1);
            assert_eq!(p.rs.expert_frame_duration() as i32, p.get(GET_EXPERT_FRAME_DURATION));
            let coded = (fs as f32 * ms / 1000.0) as usize;
            let long = signal(fs, ch, fs as usize * 2, 0);
            let mut pos = 0;
            for (k, api_ms) in [120, 100, 60, 40, 20, 120, 80].iter().cycle().take(14).enumerate() {
                let api = fs as usize * api_ms / 1000;
                let what = format!("{fs} {ch} {app:?} {ms} ms call {k} of {api_ms} ms");
                p.encode_pcm(&long[pos * ch..(pos + api) * ch], api, coded, &what);
                if coded <= api {
                    pos += coded;
                }
            }
        }
    }
}

/// The validating setters and the queries: libopus's results for valid
/// and invalid values, and the bitrate query after frames of every
/// duration with AUTO, MAX and fixed rates.
#[test]
fn setters_and_queries_match_libopus() {
    for (fs, ch, app, _, _) in CONFIGS {
        let mut p = Pair::new(fs, ch, app);
        assert_eq!(p.rs.sample_rate(), p.get(GET_SAMPLE_RATE));
        assert_eq!(p.rs.application() as i32, p.get(GET_APPLICATION));
        for v in [i32::MIN, -1000, -2, -1, 0, 1, 499, 500, 501, 64000, 750_000, 1_500_001, i32::MAX] {
            assert_eq!(code(p.rs.set_bitrate(v)), p.ctl(SET_BITRATE, v), "bitrate {v}");
            assert_eq!(p.rs.bitrate(), p.get(GET_BITRATE), "bitrate {v}");
        }
        for v in [-1, 0, 5, 10, 11] {
            assert_eq!(code(p.rs.set_complexity(v)), p.ctl(SET_COMPLEXITY, v));
            let v = p.rs.complexity;
            assert_eq!(v, p.get(GET_COMPLEXITY));
        }
        for v in [-1, 0, 37, 100, 101] {
            assert_eq!(code(p.rs.set_packet_loss_perc(v)), p.ctl(SET_PACKET_LOSS_PERC, v));
            let v = p.rs.packet_loss_perc;
            assert_eq!(v, p.get(GET_PACKET_LOSS_PERC));
        }
        for v in [-1, 0, 1, 2, 3] {
            assert_eq!(code(p.rs.set_inband_fec(v)), p.ctl(SET_INBAND_FEC, v));
            assert_eq!(p.rs.inband_fec(), p.get(GET_INBAND_FEC));
        }
        for v in [false, true] {
            p.rs.set_vbr(v);
            p.ctl(SET_VBR, v as i32);
            assert_eq!(p.rs.vbr() as i32, p.get(GET_VBR));
        }
        // The application changes before the first frame only.
        for a in [Application::Voip, Application::Audio, Application::RestrictedLowDelay, app] {
            assert_eq!(code(p.rs.set_application(a)), p.ctl(SET_APPLICATION, a as i32));
            assert_eq!(p.rs.application() as i32, p.get(GET_APPLICATION));
        }
        assert_eq!(p.rs.lookahead() as i32, p.get(GET_LOOKAHEAD));
        for (k, rate) in [-1000, -1, 32000].into_iter().enumerate() {
            p.rs.set_bitrate(rate).unwrap();
            p.ctl(SET_BITRATE, rate);
            for units in [1, 2, 4, 8, 16, 24, 48] {
                let frame = fs as usize / 400 * units;
                p.encode(frame, k * 10 + units, &format!("{fs} {ch} {app:?} rate {rate} {units}"));
                assert_eq!(p.rs.bitrate(), p.get(GET_BITRATE), "{fs} {app:?} rate {rate} frame {frame}");
            }
        }
        for a in [Application::Voip, Application::Audio, Application::RestrictedLowDelay] {
            assert_eq!(code(p.rs.set_application(a)), p.ctl(SET_APPLICATION, a as i32));
            assert_eq!(p.rs.application() as i32, p.get(GET_APPLICATION));
        }
        p.rs.reset();
        p.ctl(RESET_STATE, 0);
        assert_eq!(p.rs.bitrate(), p.get(GET_BITRATE));
    }
}

/// `OPUS_SET_INBAND_FEC(2)`: FEC without the switch to SILK for music.
#[test]
fn inband_fec_2_matches_libopus() {
    for (fs, ch, app, frame, _) in CONFIGS {
        for fec in [1, 2] {
            let mut p = Pair::new(fs, ch, app);
            p.rs.set_bitrate(48000).unwrap();
            p.ctl(SET_BITRATE, 48000);
            p.rs.set_signal(opus_rs::Signal::Music);
            p.ctl(SET_SIGNAL, 3002);
            p.rs.set_packet_loss_perc(30).unwrap();
            p.ctl(SET_PACKET_LOSS_PERC, 30);
            p.rs.set_inband_fec(fec).unwrap();
            p.ctl(SET_INBAND_FEC, fec);
            for f in 0..12 {
                p.encode(frame, f, &format!("{fs} {ch} {app:?} fec {fec} frame {f}"));
            }
        }
    }
}

/// `opus_encode24`: 24-bit input, also beyond full scale, whose signal
/// analysis reads the integers.
#[test]
fn encode24_matches_libopus() {
    for (fs, ch, app, frame, rate) in CONFIGS {
        for gain in [0.5f32, 3.0, 400.0] {
            let mut p = Pair::new(fs, ch, app);
            p.rs.set_bitrate(rate).unwrap();
            p.ctl(SET_BITRATE, rate);
            p.rs.complexity = 10;
            p.ctl(SET_COMPLEXITY, 10);
            for f in 0..12 {
                let pcm: Vec<i32> = signal(fs, ch, frame, f)
                    .iter()
                    .map(|&x| (x * gain * 8_388_608.0) as i32)
                    .collect();
                let (mut a, mut b) = ([0u8; 1500], [0u8; 1500]);
                let n = p.rs.encode24(&pcm, frame, &mut a);
                let m = unsafe { opus_encode24(p.c, pcm.as_ptr(), frame as i32, b.as_mut_ptr(), 1500) };
                let what = format!("{fs} {ch} {app:?} gain {gain} frame {f}");
                assert_eq!(n.is_ok(), m > 0, "{what}");
                p.check(&a[..n.unwrap()], &b[..m as usize], frame, &what);
            }
        }
    }
}

/// OPUS_APPLICATION_RESTRICTED_SILK and _CELT: SILK-only streams of 10 ms
/// frames and more up to wideband, CELT-only streams without delay
/// compensation; the application stays fixed.
#[test]
fn restricted_applications_match_libopus() {
    for app in [Application::RestrictedSilk, Application::RestrictedCelt] {
        for (fs, ch) in [(48000, 2), (16000, 1), (24000, 1), (8000, 2)] {
            for rate in [12000, 32000, 96000] {
                let mut p = Pair::new(fs, ch, app);
                assert_eq!(p.rs.lookahead() as i32, p.get(GET_LOOKAHEAD), "{app:?} {fs}");
                assert_eq!(p.rs.application() as i32, p.get(GET_APPLICATION), "{app:?}");
                p.rs.set_bitrate(rate).unwrap();
                p.ctl(SET_BITRATE, rate);
                for a in [Application::Voip, Application::RestrictedSilk, Application::RestrictedCelt] {
                    assert_eq!(code(p.rs.set_application(a)), p.ctl(SET_APPLICATION, a as i32));
                    assert_eq!(p.rs.application() as i32, p.get(GET_APPLICATION), "{app:?} after {a:?}");
                }
                for (f, units) in [8, 1, 4, 24, 48, 2, 16, 8, 40].into_iter().enumerate() {
                    let frame = fs as usize / 400 * units;
                    p.encode(frame, f, &format!("{app:?} {fs} {ch} {rate} {units}"));
                }
            }
        }
    }
}

/// Digital silence takes libopus's maximum and minimum (celt_maxabs16):
/// a NaN drops the samples before it, so loud samples followed by a NaN
/// and zeros are silence, and the frames go out as DTX.
#[test]
fn nan_input_silence_matches_libopus() {
    for app in [Application::Voip, Application::Audio, Application::RestrictedLowDelay] {
        let (fs, frame) = (48000, 960);
        let mut p = Pair::new(fs, 1, app);
        p.rs.set_dtx(true);
        p.ctl(SET_DTX, 1);
        for f in 0..40 {
            let mut pcm = signal(fs, 1, frame, f);
            if f >= 10 {
                pcm[frame / 2] = f32::NAN;
                pcm[frame / 2 + 1..].fill(0.0);
            }
            p.encode_exact(&pcm, frame, &format!("{app:?} frame {f}"));
        }
    }
}

/// A reset keeps the last FEC decision (silk_mode.LBRR_coded), which the
/// next packet's FEC hysteresis starts from.
#[test]
fn reset_keeps_the_fec_decision() {
    for loss in [1, 5, 10] {
        for rate in (16000..=24000).step_by(500) {
            let (fs, frame) = (48000, 960);
            let mut p = Pair::new(fs, 1, Application::Voip);
            p.rs.set_inband_fec(1).unwrap();
            p.ctl(SET_INBAND_FEC, 1);
            p.rs.set_packet_loss_perc(loss).unwrap();
            p.ctl(SET_PACKET_LOSS_PERC, loss);
            p.rs.set_max_bandwidth(opus_rs::Bandwidth::Wideband);
            p.ctl(SET_MAX_BANDWIDTH, 1103);
            for f in 0..12 {
                let r = if f < 5 { 32000 } else { rate };
                p.rs.set_bitrate(r).unwrap();
                p.ctl(SET_BITRATE, r);
                if f == 7 {
                    p.rs.reset();
                    p.ctl(RESET_STATE, 0);
                }
                let pcm = signal(fs, 1, frame, f);
                p.encode_exact(&pcm, frame, &format!("loss {loss} rate {rate} frame {f}"));
            }
        }
    }
}

/// The errors of `opus_encode`, `opus_encode24` and `opus_encode_float`:
/// OPUS_BAD_ARG for an invalid frame size and an empty buffer,
/// OPUS_BUFFER_TOO_SMALL for 100 ms in one byte.
#[test]
fn encode_errors_match_libopus() {
    for (fs, ch, app, frame, _) in CONFIGS {
        let mut p = Pair::new(fs, ch, app);
        for (frame_size, max) in [(frame, 1500), (100, 1500), (frame, 0), (fs as usize / 10, 1), (frame, 1500), (961, 1500)] {
            let mut out = [0u8; 1500];
            for entry in 0..3 {
                let pcm = signal(fs, ch, frame_size, 0);
                let (r, m) = match entry {
                    0 => (p.rs.encode(&pcm, frame_size, &mut out[..max]), unsafe {
                        opus_encode_float(p.c, pcm.as_ptr(), frame_size as i32, out.as_mut_ptr(), max as i32)
                    }),
                    1 => {
                        let x: Vec<i16> = pcm.iter().map(|&v| (v * 32768.0) as i16).collect();
                        (p.rs.encode_i16(&x, frame_size, &mut out[..max]), unsafe {
                            opus_encode(p.c, x.as_ptr(), frame_size as i32, out.as_mut_ptr(), max as i32)
                        })
                    }
                    _ => {
                        let x: Vec<i32> = pcm.iter().map(|&v| (v * 8_388_608.0) as i32).collect();
                        (p.rs.encode24(&x, frame_size, &mut out[..max]), unsafe {
                            opus_encode24(p.c, x.as_ptr(), frame_size as i32, out.as_mut_ptr(), max as i32)
                        })
                    }
                };
                let what = format!("{fs} {ch} {app:?} frame {frame_size} max {max} entry {entry}");
                assert_eq!(r.map_or_else(|e| opus_rs::error::code(e).unwrap_or(0), |n| n as i32), m, "{what}");
            }
        }
    }
}
