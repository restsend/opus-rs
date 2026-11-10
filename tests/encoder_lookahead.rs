//! `OpusEncoder::lookahead` matches libopus `OPUS_GET_LOOKAHEAD` for every
//! rate and application.
use opus::{Application as CApp, Channels as CCh, Encoder as CEnc};
use opus_rs::{Application, OpusEncoder};

#[test]
fn lookahead_matches_libopus() {
    for rate in [8000, 12000, 16000, 24000, 48000] {
        for channels in [1, 2] {
            for (app, capp) in [
                (Application::Voip, CApp::Voip),
                (Application::Audio, CApp::Audio),
                (Application::RestrictedLowDelay, CApp::LowDelay),
            ] {
                let enc = OpusEncoder::new(rate, channels, app).unwrap();
                let cch = if channels == 1 {
                    CCh::Mono
                } else {
                    CCh::Stereo
                };
                let mut c = CEnc::new(rate as u32, cch, capp).unwrap();
                assert_eq!(
                    enc.lookahead(),
                    c.get_lookahead().unwrap() as usize,
                    "{rate} Hz, {channels} ch, {app:?}"
                );
            }
        }
    }
}
