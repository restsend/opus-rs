//! CBR packets padded to two bytes must stay valid Opus packets.
//!
//! When a CBR frame has fewer than 3 bytes, the encoder emits a ToC-only
//! "PLC" frame and pads it to the CBR size as a code 3 packet, as libopus's
//! `opus_packet_pad` does. At exactly two bytes there is no room for a padding
//! length, and `opus_packet_pad` leaves the padding flag clear; the encoder set
//! it anyway, so every such packet (for example 6 kb/s with 2.5 ms frames) was
//! malformed (RFC 6716 section 3.2.5): libopus and opus-rs both reject it.

use opus::{Channels as CCh, Decoder as CDec};
use opus_rs::{Application, OpusDecoder, OpusEncoder};

#[test]
fn low_rate_cbr_packets_decode_in_libopus_and_opus_rs() {
    let mut two_byte_packets = 0;
    for rate in [8000, 48000] {
        for ch in [1, 2] {
            for app in [
                Application::Voip,
                Application::Audio,
                Application::RestrictedLowDelay,
            ] {
                // 2.5, 5, 10 and 20 ms frames.
                for units in [1usize, 2, 4, 8] {
                    let frame = rate / 400 * units;
                    for bitrate in (500..=10000).step_by(500) {
                        let mut enc = OpusEncoder::new(rate as i32, ch, app).unwrap();
                        enc.bitrate_bps = bitrate;
                        enc.use_cbr = true;
                        let channels = if ch == 2 { CCh::Stereo } else { CCh::Mono };
                        let mut c_dec = CDec::new(rate as u32, channels).unwrap();
                        let mut rs_dec = OpusDecoder::new(rate as i32, ch).unwrap();
                        let input: Vec<f32> = (0..frame * ch)
                            .map(|i| ((i % 37) as f32 - 18.0) / 64.0)
                            .collect();
                        let mut pkt = [0u8; 1500];
                        let mut pcm = vec![0f32; 5760 * ch];
                        for f in 0..3 {
                            // Frame sizes SILK or Hybrid cannot code are an
                            // encoder error, not a packet; skip them.
                            let Ok(n) = enc.encode(&input, frame, &mut pkt) else {
                                break;
                            };
                            let what = format!(
                                "{rate} Hz, {ch} ch, {units} x 2.5 ms, {bitrate} b/s, frame {f}: {:02x?}",
                                &pkt[..n.min(4)]
                            );
                            two_byte_packets += usize::from(n == 2);
                            assert_eq!(
                                c_dec.decode_float(&pkt[..n], &mut pcm, false).ok(),
                                Some(frame),
                                "libopus, {what}"
                            );
                            assert_eq!(
                                rs_dec.decode(&pkt[..n], 5760, &mut pcm),
                                Ok(frame),
                                "opus-rs, {what}"
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(two_byte_packets > 0, "no two-byte packet was produced");
}
