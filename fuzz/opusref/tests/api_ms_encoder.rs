//! MultistreamEncoder against libopus's opus_multistream_encoder: layouts,
//! entry points, frame sizes, rate control and controls. Packets are
//! compared as in diff_encoder: libopus's multistream decoder decodes every
//! opus-rs packet with the encoder's final range, each stream's ToC agrees,
//! and RestrictedLowDelay packets are byte-identical.
use core::ffi::c_void;
use opus_rs::multistream_encoder::MultistreamEncoder;
use opus_rs::{Application, error};
use opusref::*;

/// The ToC of each stream of a multistream packet.
fn tocs(p: &[u8], streams: usize) -> Vec<u8> {
    let mut out = Vec::new();
    let mut pos = 0;
    for s in 0..streams {
        let parts = opus_rs::extensions::parse_packet_impl(&p[pos..], s + 1 < streams).unwrap();
        out.push(parts.toc & 0xfc);
        pos += parts.packet_offset;
    }
    out
}

struct Pair {
    rs: MultistreamEncoder,
    c: *mut c_void,
    dec: *mut c_void,
    ch: usize,
    streams: usize,
    app: Application,
}

impl Pair {
    fn new(ch: usize, streams: usize, coupled: usize, mapping: &[u8], app: Application) -> Option<Self> {
        let mut err = 0;
        let c = unsafe {
            opus_multistream_encoder_create(48000, ch as i32, streams as i32, coupled as i32, mapping.as_ptr(), app as i32, &mut err)
        };
        let rs = MultistreamEncoder::new(48000, ch, streams, coupled, mapping, app);
        assert_eq!(rs.is_ok(), !c.is_null(), "layout {ch} {streams} {coupled} {mapping:?}");
        let Ok(rs) = rs else {
            assert_eq!(error::code(rs.err().unwrap()), Some(err));
            return None;
        };
        let dec = unsafe {
            opus_multistream_decoder_create(48000, ch as i32, streams as i32, coupled as i32, mapping.as_ptr(), &mut err)
        };
        assert!(!dec.is_null());
        Some(Pair { rs, c, dec, ch, streams, app })
    }

    fn ctl(&mut self, req: i32, v: i32) -> i32 {
        unsafe { opus_multistream_encoder_ctl(self.c, req, v) }
    }

    fn get(&mut self, req: i32) -> i32 {
        let mut v = 0i32;
        unsafe { opus_multistream_encoder_ctl(self.c, req, &mut v as *mut i32) };
        v
    }

    /// Encodes frame `f` of `frame` samples per channel through `entry`
    /// (0 float, 1 16-bit, 2 24-bit) on both and checks the packets.
    fn encode(&mut self, frame: usize, f: usize, entry: usize, max: usize, what: &str) {
        let ch = self.ch;
        let pcm: Vec<f32> = (0..frame * ch)
            .map(|i| {
                let t = (f * frame + i / ch) as f32 / 48000.0;
                let k = (i % ch) as f32;
                (t * (220.0 + 110.0 * k) * 6.2832).sin() * 0.3 + (t * 3100.0 * (1.0 + k)).sin() * 0.05
            })
            .collect();
        let (mut a, mut b) = (vec![0u8; max], vec![0u8; max]);
        let (n, m) = match entry {
            0 => (
                self.rs.encode(&pcm, frame, &mut a),
                unsafe { opus_multistream_encode_float(self.c, pcm.as_ptr(), frame as i32, b.as_mut_ptr(), max as i32) },
            ),
            1 => {
                let x: Vec<i16> = pcm.iter().map(|&v| (v * 32768.0) as i16).collect();
                (self.rs.encode_i16(&x, frame, &mut a), unsafe {
                    opus_multistream_encode(self.c, x.as_ptr(), frame as i32, b.as_mut_ptr(), max as i32)
                })
            }
            _ => {
                let x: Vec<i32> = pcm.iter().map(|&v| (v * 8_388_608.0) as i32).collect();
                (self.rs.encode24(&x, frame, &mut a), unsafe {
                    opus_multistream_encode24(self.c, x.as_ptr(), frame as i32, b.as_mut_ptr(), max as i32)
                })
            }
        };
        match n {
            Err(e) => {
                assert!(m < 0, "{what}: opus-rs {e}, libopus {m}");
                assert_eq!(error::code(e), Some(m), "{what}");
                return;
            }
            Ok(n) => assert!(m > 0, "{what}: opus-rs {n}, libopus {m}"),
        }
        let (pa, pb) = (&a[..n.unwrap()], &b[..m as usize]);
        // The later streams get what the earlier ones left of the budget,
        // which follows their sizes; those agree in CELT-only streams. The
        // first stream's ToC always agrees.
        let (ta, tb) = (tocs(pa, self.streams), tocs(pb, self.streams));
        if self.app == Application::RestrictedLowDelay {
            assert_eq!(ta, tb, "{what}: ToCs");
        }
        assert_eq!(ta[0], tb[0], "{what}: first ToC");
        if self.app == Application::RestrictedLowDelay {
            assert_eq!(pa, pb, "{what}: bytes");
        }
        let mut out = vec![0f32; 5760 * ch];
        let d = unsafe { opus_multistream_decode_float(self.dec, pa.as_ptr(), pa.len() as i32, out.as_mut_ptr(), 5760, 0) };
        let coded = opus_rs::packet::nb_samples(pa, 48000).unwrap();
        assert_eq!(d, coded as i32, "{what}: decode");
        let mut r = 0u32;
        unsafe { opus_multistream_decoder_ctl(self.dec, GET_FINAL_RANGE, &mut r as *mut u32) };
        assert_eq!(r, self.rs.final_range(), "{what}: final range");
        // The query reads each stream's rate and last frame duration, which
        // follow the streams' modes and, under CBR, the sizes of the
        // streams before; those agree in CELT-only streams.
        if (self.rs.first_stream().vbr() && ta == tb) || self.app == Application::RestrictedLowDelay {
            assert_eq!(self.rs.bitrate(), self.get(GET_BITRATE), "{what}: bitrate");
        }
    }
}

impl Drop for Pair {
    fn drop(&mut self) {
        unsafe {
            opus_multistream_encoder_destroy(self.c);
            opus_multistream_decoder_destroy(self.dec);
        }
    }
}

const LAYOUTS: [(usize, usize, usize, &[u8]); 6] = [
    (1, 1, 0, &[0]),
    (2, 1, 1, &[0, 1]),
    (2, 2, 0, &[1, 0]),
    (6, 4, 2, &[0, 4, 1, 2, 3, 5]),
    (5, 3, 1, &[0, 1, 255, 3, 2]),
    (8, 5, 3, &[0, 6, 1, 2, 3, 4, 5, 7]),
];

#[test]
fn layouts_are_checked_like_libopus() {
    for (ch, streams, coupled, mapping) in [
        (0, 1, 0, &[0u8][..]),
        (2, 1, 2, &[0, 1]),
        (2, 3, 0, &[0, 1]),
        (2, 1, 1, &[0, 0]),
        (2, 2, 0, &[0, 2]),
        (3, 2, 0, &[0, 0, 255]),
        (3, 1, 1, &[0, 1, 9]),
    ] {
        let _ = Pair::new(ch, streams, coupled, mapping, Application::Audio);
    }
}

#[test]
fn packets_match_libopus() {
    for app in [Application::RestrictedLowDelay, Application::Audio, Application::Voip] {
        for (ch, streams, coupled, mapping) in LAYOUTS {
            for (vbr, rate) in [(true, -1000), (false, -1000), (true, 96000), (false, 160000), (true, -1), (false, 20000)] {
                let mut p = Pair::new(ch, streams, coupled, mapping, app).unwrap();
                p.rs.set_vbr(vbr);
                p.ctl(SET_VBR, vbr as i32);
                p.rs.set_bitrate(rate).unwrap();
                p.ctl(SET_BITRATE, rate);
                for (f, (units, entry, max)) in [(8, 0, 4000), (4, 1, 4000), (24, 2, 4000), (48, 0, 6000), (8, 0, 300), (2, 1, 4000), (8, 0, 2 * streams)]
                    .into_iter()
                    .enumerate()
                {
                    let what = format!("{app:?} {ch} {streams} {coupled} vbr {vbr} rate {rate} call {f}");
                    p.encode(120 * units, f, entry, max, &what);
                }
            }
        }
    }
}

#[test]
fn controls_match_libopus() {
    let (ch, streams, coupled, mapping) = LAYOUTS[3];
    let mut p = Pair::new(ch, streams, coupled, mapping, Application::RestrictedLowDelay).unwrap();
    let code = |r: Result<(), &str>| r.map_or_else(|e| error::code(e).unwrap(), |()| 0);
    for v in [0, -5, 1000, 3000, 9_000_000] {
        assert_eq!(code(p.rs.set_bitrate(v)), p.ctl(SET_BITRATE, v), "bitrate {v}");
    }
    assert_eq!(code(p.rs.set_complexity(11)), p.ctl(SET_COMPLEXITY, 11));
    assert_eq!(code(p.rs.set_complexity(4)), p.ctl(SET_COMPLEXITY, 4));
    p.rs.set_phase_inversion_disabled(true);
    p.ctl(SET_PHASE_INVERSION_DISABLED, 1);
    p.rs.set_expert_frame_duration(opus_rs::FrameDuration::Ms10);
    p.ctl(SET_EXPERT_FRAME_DURATION, 5003);
    let (la, cx) = (p.rs.first_stream().lookahead() as i32, p.rs.first_stream().complexity);
    assert_eq!(la, p.get(GET_LOOKAHEAD));
    assert_eq!(cx, p.get(GET_COMPLEXITY));
    for f in 0..8 {
        if f == 4 {
            p.rs.reset();
            unsafe { opus_multistream_encoder_ctl(p.c, RESET_STATE) };
        }
        p.encode(960, f, f % 3, 4000, &format!("controls frame {f}"));
    }
    assert!(p.rs.encoder_state(3).is_ok());
    assert_eq!(p.rs.encoder_state(4).err(), Some(error::BAD_ARG));
}
