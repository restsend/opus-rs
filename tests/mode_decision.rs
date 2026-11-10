//! The mode follows libopus's decision (`opus_encode_native`) where it
//! depends on in-band FEC with expected loss, on the stereo width and on the
//! sampling rate.
//!
//! Both encoders code the fixture at complexity 5, where libopus runs no
//! signal analysis either, so both estimate speech from the application;
//! libopus codes the API rate's bandwidth, as opus-rs does. Every frame's
//! mode must be libopus's.

#[path = "common/vbr.rs"]
mod vbr;

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Encoder as CEnc,
};
use opus_rs::{Application, OpusEncoder};

struct Case {
    sr: usize,
    channels: usize,
    voip: bool,
    ms: usize,
    bitrate: i32,
    fec: bool,
    loss: i32,
}

/// The mode of every packet: S(ILK-only), H(ybrid) or C(ELT-only).
fn mode_letters(packets: &[Vec<u8>]) -> String {
    packets
        .iter()
        .map(|p| match p[0] >> 3 {
            0..=11 => 'S',
            12..=15 => 'H',
            _ => 'C',
        })
        .collect()
}

fn encode_both(case: &Case) -> (String, String) {
    let frame = case.sr * case.ms / 1000;
    let frames = 6000 / case.ms;
    let input = vbr::speech(case.sr, case.channels, frame * frames);
    let mut rs = OpusEncoder::new(
        case.sr as i32,
        case.channels,
        if case.voip {
            Application::Voip
        } else {
            Application::Audio
        },
    )
    .unwrap();
    rs.bitrate_bps = case.bitrate;
    rs.complexity = 5;
    rs.use_inband_fec = case.fec;
    rs.packet_loss_perc = case.loss;
    let channels = if case.channels == 2 {
        CCh::Stereo
    } else {
        CCh::Mono
    };
    let mut c = CEnc::new(
        case.sr as u32,
        channels,
        if case.voip { CApp::Voip } else { CApp::Audio },
    )
    .unwrap();
    c.set_bitrate(CBitrate::Bits(case.bitrate)).unwrap();
    c.set_complexity(5).unwrap();
    c.set_vbr(true).unwrap();
    c.set_inband_fec(case.fec).unwrap();
    c.set_packet_loss_perc(case.loss).unwrap();
    c.set_force_channels(Some(channels)).unwrap();
    c.set_bandwidth(match case.sr {
        16000 => CBw::Wideband,
        24000 => CBw::Superwideband,
        _ => CBw::Fullband,
    })
    .unwrap();
    let (mut rs_packets, mut c_packets) = (Vec::new(), Vec::new());
    let mut packet = [0u8; 1500];
    for f in 0..frames {
        let pcm = &input[f * frame * case.channels..(f + 1) * frame * case.channels];
        let n = rs.encode(pcm, frame, &mut packet).unwrap();
        rs_packets.push(packet[..n].to_vec());
        let n = c.encode_float(pcm, &mut packet).unwrap();
        c_packets.push(packet[..n].to_vec());
    }
    (mode_letters(&rs_packets), mode_letters(&c_packets))
}

fn check(case: Case, expected_libopus: char) {
    let (rs, c) = encode_both(&case);
    let label = format!(
        "{} Hz {} ch {} {} ms {} b/s FEC {} loss {}%",
        case.sr,
        case.channels,
        if case.voip { "VoIP" } else { "Audio" },
        case.ms,
        case.bitrate,
        case.fec,
        case.loss
    );
    // The case is the one meant: libopus codes this mode after its first
    // second.
    assert!(
        c[c.len() / 6..].chars().all(|m| m == expected_libopus),
        "{label}: libopus {c}"
    );
    assert_eq!(rs, c, "{label}: opus-rs (first line) against libopus");
}

#[test]
fn fec_with_expected_loss_codes_silk_or_hybrid() {
    // Without the loss these rates are CELT-only for both. With in-band FEC
    // and more than (128 - voice estimate) / 16 percent loss, libopus codes
    // SILK, here Hybrid.
    check(
        Case {
            sr: 48000,
            channels: 1,
            voip: true,
            ms: 20,
            bitrate: 80000,
            fec: false,
            loss: 10,
        },
        'C',
    );
    check(
        Case {
            sr: 48000,
            channels: 1,
            voip: true,
            ms: 20,
            bitrate: 80000,
            fec: true,
            loss: 10,
        },
        'H',
    );
    check(
        Case {
            sr: 48000,
            channels: 1,
            voip: false,
            ms: 20,
            bitrate: 64000,
            fec: true,
            loss: 10,
        },
        'H',
    );
    check(
        Case {
            sr: 24000,
            channels: 1,
            voip: true,
            ms: 10,
            bitrate: 96000,
            fec: true,
            loss: 15,
        },
        'H',
    );
}

#[test]
fn identical_stereo_channels_use_the_mono_threshold() {
    // The speech threshold goes from mono's 64 kb/s to stereo's 44 kb/s with
    // the stereo width: two identical channels keep mono's.
    check(
        Case {
            sr: 48000,
            channels: 2,
            voip: true,
            ms: 20,
            bitrate: 56000,
            fec: false,
            loss: 0,
        },
        'H',
    );
}

#[test]
fn wideband_input_codes_celt_at_high_rates() {
    // The threshold does not depend on the sampling rate: 16 kHz VoIP at
    // 96 kb/s is CELT-only (wideband) in libopus.
    check(
        Case {
            sr: 16000,
            channels: 1,
            voip: true,
            ms: 20,
            bitrate: 96000,
            fec: false,
            loss: 0,
        },
        'C',
    );
}
