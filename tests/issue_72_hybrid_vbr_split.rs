//! Real-speech hybrid VBR allocation regression for #72.

#[path = "common/vbr.rs"]
mod vbr;

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Decoder as CDec,
    Encoder as CEnc, Signal as CSignal,
};
use opus_rs::{Application, OpusEncoder};

#[test]
fn hybrid_speech_vbr_tracks_libopus_with_and_without_fec() {
    for sr in [24000, 48000] {
        for ms in [10, 20] {
            let frame = sr * ms / 1000;
            let packets = 2000 / ms;
            for channels in [1, 2] {
                let input = vbr::speech(sr, channels, frame * packets);
                let c_channels = if channels == 1 {
                    CCh::Mono
                } else {
                    CCh::Stereo
                };
                for rate in [12000, 16000, 20000, 24000, 28000, 32000] {
                    for fec in [false, true] {
                        let mut rs =
                            OpusEncoder::new(sr as i32, channels, Application::Voip).unwrap();
                        rs.bitrate_bps = rate;
                        rs.complexity = 9;
                        rs.use_cbr = false;
                        rs.use_inband_fec = fec;
                        rs.packet_loss_perc = if fec { 10 } else { 0 };
                        let mut c = CEnc::new(sr as u32, c_channels, CApp::Voip).unwrap();
                        c.set_bitrate(CBitrate::Bits(rate)).unwrap();
                        c.set_complexity(9).unwrap();
                        c.set_vbr(true).unwrap();
                        c.set_vbr_constraint(false).unwrap();
                        c.set_bandwidth(if sr == 24000 {
                            CBw::Superwideband
                        } else {
                            CBw::Fullband
                        })
                        .unwrap();
                        c.set_signal(CSignal::Voice).unwrap();
                        c.set_force_channels(Some(c_channels)).unwrap();
                        c.set_inband_fec(fec).unwrap();
                        c.set_packet_loss_perc(if fec { 10 } else { 0 }).unwrap();
                        let mut decoder = CDec::new(sr as u32, c_channels).unwrap();
                        let mut decoded = vec![0.0; frame * channels];
                        let mut packet = [0u8; 1500];
                        let mut bytes = [0usize; 2];
                        for p in 0..packets {
                            let pcm = &input[p * frame * channels..(p + 1) * frame * channels];
                            let n_rs = rs.encode(pcm, frame, &mut packet).unwrap();
                            assert_eq!(
                                decoder
                                    .decode_float(&packet[..n_rs], &mut decoded, false)
                                    .unwrap(),
                                frame,
                            );
                            assert!(decoded.iter().all(|sample| sample.is_finite()));
                            let n_c = c.encode_float(pcm, &mut packet).unwrap();
                            if p >= 10 {
                                bytes[0] += n_rs;
                                bytes[1] += n_c;
                            }
                        }
                        let ratio = bytes[0] as f64 / bytes[1] as f64;
                        println!(
                            "{sr} Hz {ms} ms {rate} b/s {channels}ch FEC={fec}: opus-rs/libopus {ratio:.4}"
                        );
                        assert!(
                            (0.85..=1.10).contains(&ratio),
                            "{sr} Hz {ms} ms {rate} b/s {channels}ch FEC={fec}: {ratio}",
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn hybrid_vbr_celt_share_below_501_bps_keeps_an_unlimited_rate() {
    // At 24 kHz, 20 ms and 2500 b/s the super-wideband SILK share leaves
    // CELT 450 b/s. libopus's OPUS_SET_BITRATE refuses rates of 500 b/s or
    // less, so CELT keeps the unlimited rate it was given for the frame and
    // fills the packet (1276 bytes in libopus 1.6.1), digital silence too,
    // as only a CELT rate makes CELT VBR. At 48 kHz there is no
    // super-wideband boost, CELT gets 750 b/s and packets stay small.
    for (sr, unlimited) in [(24000usize, true), (48000, false)] {
        let frame = sr / 50;
        let mut enc = OpusEncoder::new(sr as i32, 1, Application::Voip).unwrap();
        enc.bitrate_bps = 2500;
        enc.complexity = 9;
        enc.use_cbr = false;
        let mut decoder = CDec::new(sr as u32, CCh::Mono).unwrap();
        let mut decoded = vec![0.0; frame];
        let mut packet = [0u8; 1500];
        let mut input = vbr::speech(sr, 1, frame * 10);
        input.resize(frame * 13, 0.0);
        for p in 0..13 {
            let n = enc
                .encode(&input[p * frame..(p + 1) * frame], frame, &mut packet)
                .unwrap();
            assert!((12..16).contains(&(packet[0] >> 3)), "hybrid TOC");
            assert_eq!(n == 1276, unlimited, "{sr} Hz packet {p}: {n} bytes");
            assert_eq!(
                decoder
                    .decode_float(&packet[..n], &mut decoded, false)
                    .unwrap(),
                frame
            );
        }
    }
}
