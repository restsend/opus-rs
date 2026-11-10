//! The bandwidth follows libopus's decision (`opus_encode_native`).
//!
//! libopus chooses the bandwidth from the equivalent rate (with thresholds
//! between music's and speech's by its estimate of speech, and hysteresis),
//! caps it by `OPUS_SET_MAX_BANDWIDTH`, and keeps SILK and Hybrid at
//! wideband below 15 kb/s of packet. The tests code with opus-rs and libopus
//! at the same settings and compare the ToC configuration (mode, bandwidth,
//! duration) of every packet.

#[path = "common/vbr.rs"]
mod vbr;

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Encoder as CEnc,
};
use opus_rs::{Application, Bandwidth, OpusEncoder};

struct Case {
    sr: usize,
    voip: bool,
    bitrate: i32,
    cbr: bool,
    complexity: i32,
    max_bandwidth: Option<(Bandwidth, CBw)>,
}

/// The ToC configuration of every packet, for opus-rs and for libopus.
fn configs(case: &Case, input: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let frame = case.sr / 50;
    let mut rs = OpusEncoder::new(
        case.sr as i32,
        1,
        if case.voip {
            Application::Voip
        } else {
            Application::Audio
        },
    )
    .unwrap();
    rs.bitrate_bps = case.bitrate;
    rs.complexity = case.complexity;
    rs.use_cbr = case.cbr;
    let mut c = CEnc::new(
        case.sr as u32,
        CCh::Mono,
        if case.voip { CApp::Voip } else { CApp::Audio },
    )
    .unwrap();
    c.set_bitrate(CBitrate::Bits(case.bitrate)).unwrap();
    c.set_complexity(case.complexity).unwrap();
    c.set_vbr(!case.cbr).unwrap();
    if let Some((rs_max, c_max)) = case.max_bandwidth {
        rs.set_max_bandwidth(rs_max);
        c.set_max_bandwidth(c_max).unwrap();
    }
    let (mut ours, mut theirs) = (Vec::new(), Vec::new());
    let mut packet = [0u8; 1500];
    for x in input.chunks_exact(frame) {
        let n = rs.encode(x, frame, &mut packet).unwrap();
        assert!(n > 0);
        ours.push(packet[0] >> 3);
        let n = c.encode_float(x, &mut packet).unwrap();
        assert!(n > 0);
        theirs.push(packet[0] >> 3);
    }
    (ours, theirs)
}

fn check(label: &str, case: Case, input: &[f32]) {
    let (ours, theirs) = configs(&case, input);
    let first = (0..ours.len()).find(|&i| ours[i] != theirs[i]);
    assert_eq!(
        first,
        None,
        "{label}: first differing packet, configs opus-rs {:?}, libopus {:?}",
        &ours[first.unwrap_or(0)..(first.unwrap_or(0) + 12).min(ours.len())],
        &theirs[first.unwrap_or(0)..(first.unwrap_or(0) + 12).min(theirs.len())]
    );
}

#[test]
fn the_rate_chooses_the_bandwidth() {
    let speech = vbr::speech(48000, 1, 48000 * 6);
    for (bitrate, complexity) in [(9000, 5), (12000, 5), (16000, 5), (12000, 9), (24000, 9)] {
        check(
            &format!("speech 48 kHz VoIP {bitrate} b/s complexity {complexity}"),
            Case {
                sr: 48000,
                voip: true,
                bitrate,
                cbr: false,
                complexity,
                max_bandwidth: None,
            },
            &speech,
        );
    }
}

#[test]
fn hybrid_needs_15_kbps_of_packet() {
    let speech = vbr::speech(24000, 1, 24000 * 6);
    for bitrate in [13000, 14000, 16000] {
        check(
            &format!("speech 24 kHz VoIP {bitrate} b/s CBR"),
            Case {
                sr: 24000,
                voip: true,
                bitrate,
                cbr: true,
                complexity: 5,
                max_bandwidth: None,
            },
            &speech,
        );
    }
}

#[test]
fn the_maximum_bandwidth_caps_it() {
    let speech = vbr::speech(48000, 1, 48000 * 6);
    check(
        "speech 48 kHz Audio 64 kb/s, wideband at most",
        Case {
            sr: 48000,
            voip: false,
            bitrate: 64000,
            cbr: false,
            complexity: 5,
            max_bandwidth: Some((Bandwidth::Wideband, CBw::Wideband)),
        },
        &speech,
    );
}
