//! Issue #51: under CBR with in-band FEC, SILK's frame target must leave room
//! for the packet's LBRR section, as libopus's does (enc_API.c
//! `nBitsUsedLBRR`, `nBitsExceeded`, `bitsBalance`). A target that ignores it
//! busts the packet, and the encoder falls back to an empty frame that every
//! decoder conceals: a 20 ms dropout, and no LBRR for the next packet.
//!
//! Each case encodes the same voiced signal with opus-rs and with libopus (the
//! `opus` dev-dependency) at the same settings, and counts per stream:
//! - PLC frames: packets whose frame is at most 1 byte;
//! - LBRR packets: packets whose SILK header sets an LBRR flag.
//!
//! Before the fix, mono busted 8, 5 and 3 packets in 50 at 16, 20 and
//! 24 kb/s (libopus 0, 2, 2), and stereo, whose LBRR section carries the side
//! too, busted 15 at 20 and 24 kb/s and 10 at 32 kb/s. Stereo LBRR then
//! alternated with gaps: 5-20 LBRR packets where libopus codes 29-30.

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Encoder as CEnc,
    Signal as CSignal,
};
use opus_rs::{Application, OpusEncoder};
use std::f32::consts::PI;

const SR: usize = 16000;
const FRAME: usize = SR / 50;
const PACKETS: usize = 50;
const LOSS_PERC: i32 = 40;
const RATES: [i32; 5] = [16000, 20000, 24000, 32000, 48000];

/// The voiced-speech-like signal of `issue_27_libopus_oracle.rs`: a 140 Hz
/// harmonic series with slow pitch drift and a syllable-rate envelope; the
/// right channel is a scaled, phase-shifted copy.
fn voiced(ch: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * ch);
    let mut phase = 0.0f32;
    for i in 0..n {
        let t = i as f32 / SR as f32;
        let f0 = 140.0 + 15.0 * (2.0 * PI * 0.7 * t).sin();
        phase += 2.0 * PI * f0 / SR as f32;
        let env = 0.6 + 0.4 * (2.0 * PI * 3.0 * t).sin();
        let (mut l, mut r) = (0.0, 0.0);
        for h in 1..=8 {
            let a = 0.25 / h as f32;
            l += a * (h as f32 * phase).sin();
            r += a * (h as f32 * phase + 0.3 * h as f32).sin();
        }
        out.push(l * env * 0.5);
        if ch == 2 {
            out.push(r * env * 0.4);
        }
    }
    out
}

/// The only frame of a one-frame packet: code 0, or code 3 with a frame
/// count of 1 and optional padding (how CBR packets are padded).
fn single_frame(pkt: &[u8]) -> &[u8] {
    match pkt[0] & 0x03 {
        0 => &pkt[1..],
        3 => {
            assert_eq!(pkt[1] & 0x3f, 1, "expected a one-frame code-3 packet");
            let (mut start, mut padding) = (2, 0);
            if pkt[1] & 0x40 != 0 {
                loop {
                    let b = pkt[start] as usize;
                    start += 1;
                    if b == 255 {
                        padding += 254;
                    } else {
                        padding += b;
                        break;
                    }
                }
            }
            &pkt[start..pkt.len() - padding]
        }
        code => panic!("unexpected code-{code} packet"),
    }
}

/// Whether a 20 ms SILK frame's header sets an LBRR flag. The header's flags
/// are the top bits of its first byte: per channel, the VAD flag then the
/// LBRR flag (mid, then side).
fn carries_lbrr(frame: &[u8], ch: usize) -> bool {
    let lbrr_bits = if ch == 2 { 0x40 | 0x10 } else { 0x40 };
    frame.len() > 1 && frame[0] & lbrr_bits != 0
}

#[derive(Clone, Copy, Default)]
struct Counts {
    plc: usize,
    lbrr: usize,
}

impl Counts {
    fn add(&mut self, pkt: &[u8], ch: usize) {
        assert!(pkt[0] >> 3 <= 11, "expected a SILK-only TOC");
        let frame = single_frame(pkt);
        if frame.len() <= 1 {
            self.plc += 1;
        }
        if carries_lbrr(frame, ch) {
            self.lbrr += 1;
        }
    }
}

/// opus-rs's and libopus's counts for one second of CBR with in-band FEC.
fn measure(ch: usize, rate: i32) -> (Counts, Counts) {
    let input = voiced(ch, FRAME * PACKETS);

    let mut rs_enc = OpusEncoder::new(SR as i32, ch, Application::Voip).unwrap();
    rs_enc.bitrate_bps = rate;
    rs_enc.use_cbr = true;
    rs_enc.use_inband_fec = true;
    rs_enc.packet_loss_perc = LOSS_PERC;

    // libopus at opus-rs's settings, pinned to voice, wideband and (for two
    // channels) a stereo stream, which it would otherwise drop at low rates.
    let c_ch = if ch == 2 { CCh::Stereo } else { CCh::Mono };
    let mut c_enc = CEnc::new(SR as u32, c_ch, CApp::Voip).unwrap();
    c_enc.set_bitrate(CBitrate::Bits(rate)).unwrap();
    c_enc.set_vbr(false).unwrap();
    c_enc.set_complexity(rs_enc.complexity).unwrap();
    c_enc.set_bandwidth(CBw::Wideband).unwrap();
    c_enc.set_signal(CSignal::Voice).unwrap();
    if ch == 2 {
        c_enc.set_force_channels(Some(CCh::Stereo)).unwrap();
    }
    c_enc.set_inband_fec(true).unwrap();
    c_enc.set_packet_loss_perc(LOSS_PERC).unwrap();

    let (mut rs, mut c) = (Counts::default(), Counts::default());
    let mut pkt = vec![0u8; 1500];
    for p in 0..PACKETS {
        let pcm = &input[p * FRAME * ch..(p + 1) * FRAME * ch];
        let n = rs_enc
            .encode(pcm, FRAME, &mut pkt)
            .unwrap_or_else(|e| panic!("{ch}ch {rate}: packet {p}: opus-rs encode failed: {e}"));
        rs.add(&pkt[..n], ch);
        let n = c_enc.encode_float(pcm, &mut pkt).unwrap();
        c.add(&pkt[..n], ch);
    }
    (rs, c)
}

fn check(ch: usize) {
    let mut table = String::from("rate   | opus-rs PLC LBRR | libopus PLC LBRR\n");
    let mut failures = Vec::new();
    for rate in RATES {
        let (rs, c) = measure(ch, rate);
        table += &format!(
            "{:>2} k   | {:>11} {:>4} | {:>11} {:>4}\n",
            rate / 1000,
            rs.plc,
            rs.lbrr,
            c.plc,
            c.lbrr
        );
        // libopus busts its own packets now and then in this squeezed
        // regime; one more per second is noise.
        if rs.plc > c.plc + 1 {
            failures.push(format!("{rate}: {} PLC frames, libopus {}", rs.plc, c.plc));
        }
        // Where libopus codes LBRR, opus-rs must too, and not drop it for
        // want of room: at least 2/3 as many packets (the closest case,
        // stereo at 20 kb/s, measures 19-27 against libopus's ~30 across
        // input levels 0.90-1.10x). At 16 kb/s libopus's `decide_fec` turns
        // FEC off and opus-rs's doesn't, a separate gap.
        if c.lbrr > 0 && 3 * rs.lbrr < 2 * c.lbrr {
            failures.push(format!(
                "{rate}: {} LBRR packets, libopus {}",
                rs.lbrr, c.lbrr
            ));
        }
    }
    println!("{ch}ch, 20 ms, CBR, in-band FEC at {LOSS_PERC}% loss, {PACKETS} packets:\n{table}");
    assert!(
        failures.is_empty(),
        "{ch}ch: opus-rs busts packets or drops LBRR: {failures:?}\n{table}"
    );
}

#[test]
fn mono_cbr_fec_fits_lbrr_like_libopus() {
    check(1);
}

#[test]
fn stereo_cbr_fec_fits_lbrr_like_libopus() {
    check(2);
}
