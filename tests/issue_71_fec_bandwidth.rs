//! Issue #71: FEC can reduce the bandwidth below Hybrid's CELT start band.
//! Select SILK-only for those frames, as libopus does after `decide_fec`,
//! rather than passing an empty band range to CELT's allocation.

use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::f32::consts::PI;

const FRAMES: usize = 3;

#[derive(Clone, Copy, Debug)]
enum Signal {
    Silence,
    Sine,
    Noise,
}

fn input(signal: Signal, sample_rate: i32, channels: usize, frame_size: usize) -> Vec<i16> {
    let mut state = 0x1234_5678u32;
    (0..FRAMES * frame_size * channels)
        .map(|i| match signal {
            Signal::Silence => 0,
            Signal::Sine => {
                let phase = 2.0 * PI * 440.0 * (i / channels) as f32 / sample_rate as f32
                    + (i % channels) as f32 * 0.3;
                (phase.sin() * 16000.0) as i16
            }
            Signal::Noise => {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state as i32 >> 16) / 2) as i16
            }
        })
        .collect()
}

fn check(channels: usize) {
    let mut cases = 0;
    for sample_rate in [16000, 24000, 48000] {
        for frame_ms in [10, 20] {
            let frame_size = sample_rate as usize * frame_ms / 1000;
            for bitrate in [12000, 16000, 20000, 24000, 28000, 32000] {
                for signal in [Signal::Silence, Signal::Sine, Signal::Noise] {
                    let context = format!(
                        "{channels}ch {sample_rate} Hz {frame_ms} ms {bitrate} b/s {signal:?}"
                    );
                    let pcm = input(signal, sample_rate, channels, frame_size);
                    let mut encoder =
                        OpusEncoder::new(sample_rate, channels, Application::Voip).unwrap();
                    encoder.bitrate_bps = bitrate;
                    encoder.use_inband_fec = true;
                    encoder.packet_loss_perc = 10;
                    let mut decoder = OpusDecoder::new(sample_rate, channels).unwrap();
                    let mut packet = [0u8; 1500];
                    let mut output = vec![0.0f32; frame_size * channels];

                    // The first frame reproduced the panic; later frames also
                    // exercise the LBRR payload and FEC's bandwidth hysteresis.
                    for (frame, samples) in pcm.chunks_exact(frame_size * channels).enumerate() {
                        let len = encoder
                            .encode_i16(samples, frame_size, &mut packet)
                            .unwrap_or_else(|e| panic!("{context} frame {frame}: encode: {e}"));
                        assert!(len > 1, "{context} frame {frame}: empty packet");

                        // At 24 kb/s, FEC narrows the first frame to wideband
                        // or below. Its TOC must describe SILK-only, even when
                        // the encoder starts at 24 or 48 kHz.
                        if frame == 0 && sample_rate >= 24000 && bitrate == 24000 {
                            assert!(packet[0] >> 3 < 12, "{context}: expected SILK-only");
                        }
                        let decoded = decoder
                            .decode(&packet[..len], frame_size, &mut output)
                            .unwrap_or_else(|e| panic!("{context} frame {frame}: decode: {e}"));
                        assert_eq!(decoded, frame_size, "{context} frame {frame}");
                    }
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 108);
    println!(
        "{channels}ch: {cases} configurations, {} packets decoded",
        cases * FRAMES
    );
}

#[test]
fn mono_fec_bandwidth_matrix_roundtrips() {
    check(1);
}

#[test]
fn stereo_fec_bandwidth_matrix_roundtrips() {
    check(2);
}
