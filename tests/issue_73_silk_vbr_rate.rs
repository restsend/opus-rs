//! SILK conformance and the hybrid VBR rate of #73.
//!
//! The reference values come from libopus main at f1afc4222a with the
//! patches of fuzz/opusref/patches (scalar builds): fixed-point with
//! `silk-fixed`, float by default, the build whose SILK encoder opus-rs
//! follows: its symbols, the packets of whole encodes and the helper
//! functions below match it exactly.

#[path = "common/vbr.rs"]
mod vbr;

use opus::{
    Application as CApp, Bandwidth as CBw, Bitrate as CBitrate, Channels as CCh, Decoder as CDec,
    Encoder as CEnc, Signal as CSignal,
};
use opus_rs::silk::{
    control_codec::{silk_control_encoder, silk_setup_complexity, silk_setup_lbrr},
    control_snr::silk_control_snr,
    define::TYPE_VOICED,
    enc_api::{silk_encode_do_vad, silk_encode_frame},
    hp_variable_cutoff::silk_hp_variable_cutoff,
    init_encoder::silk_init_encoder,
    macros::silk_add_pos_sat32,
    nlsf::silk_nlsf_vq_weights_laroia,
    resampler::silk_resampler_down2_3,
    structs::SilkEncoderState,
};
use opus_rs::{Application, Bandwidth, OpusEncoder, Signal, range_coder::RangeCoder};

fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;

/// The value of the build's SILK encoder: fixed-point libopus main's with
/// `silk-fixed`, float libopus main's by default.
fn silk_ref<T>(fixed: T, float: T) -> T {
    if cfg!(feature = "silk-fixed") { fixed } else { float }
}

#[test]
fn positive_saturation_matches_libopus_ltp_rate_costs() {
    // LTP rate costs can be negative. libopus ADD_POS_SAT32 saturates a
    // negative wrapped sum, including the first cost in an accumulator.
    assert_eq!(silk_add_pos_sat32(0, -5000), i32::MAX);
    assert_eq!(silk_add_pos_sat32(i32::MAX, -5000), i32::MAX - 5000);
    assert_eq!(silk_add_pos_sat32(i32::MAX, 1), i32::MAX);
    assert_eq!(silk_add_pos_sat32(20, -10), 10);
}

#[test]
fn lbrr_gain_increase_matches_libopus() {
    for (loss, increase) in [(5, 7), (10, 6), (20, 4), (25, 3), (40, 3)] {
        let mut enc = SilkEncoderState::default();
        enc.s_cmn.packet_loss_perc = loss;
        silk_setup_lbrr(&mut enc, 1);
        assert_eq!(enc.s_cmn.lbrr_gain_increases, 7);
        silk_setup_lbrr(&mut enc, 1);
        assert_eq!(enc.s_cmn.lbrr_gain_increases, increase, "loss {loss}");
    }
}

#[test]
fn hp_cutoff_smoothing_rounds_like_libopus() {
    // Twelve voiced updates of the variable high-pass smoother from the
    // initial state, for three pitch lags, input qualities and activities.
    const EXPECTED: [i32; 18] = [
        209136, 199464, 209009, 199464, 197460, 195567, 203145, 198516, 200774, 197284, 195114,
        194344, 195114, 194344, 194732, 194142, 193811, 193644,
    ];
    let mut expected = EXPECTED.iter();
    for lag in [130, 200, 254] {
        for quality in [0, 16000, 30000] {
            for activity in [255, 97] {
                let mut enc = SilkEncoderState::default();
                silk_init_encoder(&mut enc, opus_rs::Arch::Scalar);
                let cmn = &mut enc.s_cmn;
                cmn.prev_signal_type = TYPE_VOICED;
                cmn.fs_khz = 16;
                cmn.prev_lag = lag;
                cmn.input_quality_bands_q15[0] = quality;
                cmn.speech_activity_q8 = activity;
                for _ in 0..12 {
                    silk_hp_variable_cutoff(cmn);
                }
                assert_eq!(
                    cmn.variable_hp_smth1_q15,
                    *expected.next().unwrap(),
                    "lag {lag} quality {quality} activity {activity}"
                );
            }
        }
    }
}

#[test]
fn nlsf_weights_measure_the_last_gap_from_pi() {
    let nlsf = [
        887, 1666, 3081, 4071, 4616, 6360, 9187, 11747, 13467, 16999, 19530, 21966, 24216, 26410,
        28528, 30653,
    ];
    let mut weights = [0i16; 16];
    silk_nlsf_vq_weights_laroia(&mut weights, &nlsf, 16);
    assert_eq!(
        weights,
        [
            315, 260, 224, 372, 315, 121, 97, 127, 113, 88, 104, 111, 117, 120, 122, 122
        ]
    );
}

#[test]
fn pitch_downsampler_2_3_matches_libopus() {
    let input: Vec<i16> = (0..48)
        .map(|i| ((i * 7919) % 20000 - 10000) as i16)
        .collect();
    let mut state = [0i32; 6];
    let mut output = [0i16; 32];
    silk_resampler_down2_3(&mut state, &mut output, &input, 48);
    assert_eq!(
        output,
        [
            0, -2867, -5799, 3096, -1504, 5973, 904, -2734, 4660, 2583, -3005, 1322, 4620, -2479,
            -1776, 5066, -836, -3597, 3299, 1435, -4200, 115, 3435, -3680, -2970, 3870, -2031,
            -4793, 2103, 239, -5395, -1081
        ]
    );
    assert_eq!(state, [-109176, 2333672, -1820891, -54570, 732492, 21672]);
}

#[test]
fn pitch_thresholds_round_to_nearest() {
    // SILK_FIX_CONST(0.8 / 0.76 / 0.74 / 0.72 / 0.7, 16), per complexity.
    let expected = [
        52429, 49807, 52429, 49807, 48497, 48497, 47186, 47186, 45875, 45875, 45875,
    ];
    for (complexity, &threshold) in expected.iter().enumerate() {
        let mut enc = SilkEncoderState::default();
        silk_setup_complexity(&mut enc, complexity as i32);
        assert_eq!(
            enc.s_cmn.pitch_estimation_threshold_q16, threshold,
            "complexity {complexity}"
        );
    }
}

#[test]
fn fixed_target_silk_symbols_and_excitation_match_libopus() {
    // All 902 16 kHz / 20 ms frames of answer_16k.wav, with a fixed
    // 20000 b/s SNR target and a nonbinding 10000-bit cap. Each block hash
    // covers 41 frames: SNR, VAD, every coded index, the entropy bit count
    // and all 320 excitation pulses, as i32 LE values.
    let input = vbr::speech_pcm();
    let mut enc = SilkEncoderState::default();
    silk_init_encoder(&mut enc, opus_rs::Arch::Scalar);
    silk_control_encoder(&mut enc, 16, 20, 20000, 9);
    for (block, &expected) in TRACE_BLOCKS.iter().enumerate() {
        let mut hash = FNV_OFFSET;
        for frame in block * 41..(block + 1) * 41 {
            let pcm = &input[frame * 320..(frame + 1) * 320];
            enc.s_cmn.n_frames_encoded = 0;
            silk_control_snr(&mut enc.s_cmn, 20000);
            silk_encode_do_vad(&mut enc, pcm, 1);
            let mut rc = RangeCoder::new_encoder(1275);
            let mut bytes = 0;
            silk_encode_frame(&mut enc, pcm, &mut rc, &mut bytes, 0, 10000, 0);
            let ix = &enc.s_cmn.indices;
            let mut values = vec![
                frame as i32,
                enc.s_cmn.snr_db_q7,
                enc.s_cmn.speech_activity_q8,
                ix.signal_type as i32,
                ix.quant_offset_type as i32,
            ];
            values.extend(ix.gains_indices.map(i32::from));
            values.extend([
                ix.lag_index as i32,
                ix.contour_index as i32,
                ix.per_index as i32,
            ]);
            values.extend(ix.ltp_index.map(i32::from));
            values.extend(ix.nlsf_indices.map(i32::from));
            values.extend([
                ix.nlsf_interp_coef_q2 as i32,
                ix.ltp_scale_index as i32,
                ix.seed as i32,
                rc.tell(),
            ]);
            values.extend(enc.pulses[..320].iter().map(|&p| p as i32));
            for v in values {
                hash = fnv(hash, &v.to_le_bytes());
            }
        }
        assert_eq!(
            hash,
            expected,
            "SILK frames {}..{}",
            block * 41,
            (block + 1) * 41
        );
    }
}

/// The fixture at `sr`, linearly interpolated in f32 and rounded to 16 bits.
fn speech_i16(sr: usize) -> Vec<i16> {
    let source: Vec<f32> = vbr::speech_pcm()
        .iter()
        .map(|&x| x as f32 / 32768.0)
        .collect();
    // In u64: the products overflow a 32-bit usize.
    let sr = sr as u64;
    (0..source.len() as u64 * sr / 16000)
        .map(|i| {
            let pos = i * 16000;
            let at = (pos / sr) as usize;
            let fraction = (pos % sr) as f32 / sr as f32;
            let x = source[at] + fraction * (source[(at + 1).min(source.len() - 1)] - source[at]);
            (x * 32768.0).clamp(-32768.0, 32767.0).round_ties_even() as i16
        })
        .collect()
}

/// Encode the whole fixture with 16-bit input and return the packets.
fn encode_speech(sr: usize, ms: usize, rate: i32, channels: usize, fec: bool) -> Vec<Vec<u8>> {
    let mono = speech_i16(sr);
    let frame = sr * ms / 1000;
    let mut enc = OpusEncoder::new(sr as i32, channels, Application::Voip).unwrap();
    enc.bitrate_bps = rate;
    enc.complexity = 9;
    enc.use_cbr = false;
    // As the libopus encodes of the hashes: speech, at the API rate's
    // bandwidth.
    enc.set_signal(Signal::Voice);
    enc.set_bandwidth(match sr {
        16000 => Bandwidth::Wideband,
        24000 => Bandwidth::Superwideband,
        _ => Bandwidth::Fullband,
    });
    enc.use_inband_fec = fec;
    enc.packet_loss_perc = if fec { 10 } else { 0 };
    let mut packet = [0u8; 1500];
    (0..mono.len() / frame)
        .map(|p| {
            let pcm: Vec<i16> = mono[p * frame..(p + 1) * frame]
                .iter()
                .flat_map(|&x| std::iter::repeat_n(x, channels))
                .collect();
            let n = enc.encode_i16(&pcm, frame, &mut packet).unwrap();
            packet[..n].to_vec()
        })
        .collect()
}

fn packet_hash(hash: u64, packet: &[u8]) -> u64 {
    fnv(fnv(hash, &(packet.len() as u32).to_le_bytes()), packet)
}

#[test]
fn silk_packets_match_libopus() {
    // Whole encodes of answer_16k.wav (VoIP, VBR, complexity 9). Each hash
    // covers 200 consecutive packets, length and bytes.
    for (sr, ms, rate, channels, expected) in PACKET_BLOCKS {
        let packets = encode_speech(sr, ms, rate, channels, false);
        let blocks: Vec<u64> = packets
            .chunks(200)
            .map(|chunk| chunk.iter().fold(FNV_OFFSET, |h, p| packet_hash(h, p)))
            .collect();
        assert_eq!(
            blocks, expected,
            "{sr} Hz {ms} ms {rate} b/s {channels} channel(s)"
        );
    }
}

#[test]
fn fec_bandwidth_switch_waits_like_libopus() {
    // 10% loss makes FEC narrow the bandwidth: libopus starts SILK at 8 kHz,
    // keeps it there until a switch is allowed, then moves to 12 kHz. The
    // two packets around the switch end with 27 bytes of 5 ms CELT
    // redundancy, which the fixed-point build codes in fixed point; all else
    // matches.
    let packets = encode_speech(16000, 20, 20000, 1, true);
    let configs: Vec<u8> = packets.iter().map(|p| p[0] >> 3).collect();
    assert!(
        configs[..25].iter().all(|&c| c == 1),
        "{:?}",
        &configs[..30]
    );
    assert!(
        configs[25..].iter().all(|&c| c == 5),
        "{:?}",
        &configs[..30]
    );
    assert_eq!(
        (packets[24].len(), packets[25].len()),
        (76, silk_ref(69, 67))
    );
    let hash = packets.iter().enumerate().fold(FNV_OFFSET, |h, (i, p)| {
        if (24..=25).contains(&i) {
            fnv(fnv(h, &(p.len() as u32).to_le_bytes()), &p[..p.len() - 27])
        } else {
            packet_hash(h, p)
        }
    });
    assert_eq!(hash, silk_ref(0x53eef2e810c802d9, 0xb00662c5c3b4cb08));
}

#[test]
fn silk_speech_vbr_tracks_libopus_after_warmup() {
    for ms in [10, 20] {
        let frame = 16000 * ms / 1000;
        let packets = 2000 / ms;
        let input = vbr::speech(16000, 1, frame * packets);
        for rate in [12000, 16000, 20000, 24000, 28000, 32000] {
            let mut rs = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
            rs.set_signal(Signal::Voice);
            rs.bitrate_bps = rate;
            rs.complexity = 9;
            rs.use_cbr = false;
            let mut c = CEnc::new(16000, CCh::Mono, CApp::Voip).unwrap();
            c.set_bitrate(CBitrate::Bits(rate)).unwrap();
            c.set_complexity(9).unwrap();
            c.set_vbr(true).unwrap();
            c.set_vbr_constraint(false).unwrap();
            c.set_bandwidth(CBw::Wideband).unwrap();
            c.set_signal(CSignal::Voice).unwrap();
            let mut bytes = [0usize; 2];
            let mut packet = [0u8; 1500];
            for p in 0..packets {
                let pcm = &input[p * frame..(p + 1) * frame];
                let n_rs = rs.encode(pcm, frame, &mut packet).unwrap();
                assert!(packet[0] >> 3 < 12);
                let n_c = c.encode_float(pcm, &mut packet).unwrap();
                assert!(packet[0] >> 3 < 12);
                if p >= 10 {
                    bytes[0] += n_rs;
                    bytes[1] += n_c;
                }
            }
            let ratio = bytes[0] as f64 / bytes[1] as f64;
            println!("16000 Hz {ms} ms {rate} b/s: opus-rs/libopus {ratio:.4}");
            assert!(
                (0.85..=1.10).contains(&ratio),
                "{ms} ms {rate} b/s: {ratio}"
            );
        }
    }
}

#[test]
fn fec_bandwidth_reduction_keeps_speech_decodable() {
    for sr in [16000, 24000, 48000] {
        for ms in [10, 20] {
            for channels in [1, 2] {
                let frame = sr * ms / 1000;
                let packets = 2000 / ms;
                let input = vbr::speech(sr, channels, frame * packets);
                let mut enc = OpusEncoder::new(sr as i32, channels, Application::Voip).unwrap();
                enc.bitrate_bps = 20000;
                enc.use_inband_fec = true;
                enc.packet_loss_perc = 10;
                enc.complexity = 9;
                let c_channels = if channels == 1 {
                    CCh::Mono
                } else {
                    CCh::Stereo
                };
                let mut dec = CDec::new(sr as u32, c_channels).unwrap();
                let mut decoded = Vec::new();
                let mut pcm = vec![0.0; frame * channels];
                let mut packet = [0u8; 1500];
                for p in 0..packets {
                    let data = &input[p * frame * channels..(p + 1) * frame * channels];
                    let n = enc.encode(data, frame, &mut packet).unwrap();
                    assert_eq!(
                        dec.decode_float(&packet[..n], &mut pcm, false).unwrap(),
                        frame
                    );
                    decoded.extend_from_slice(&pcm);
                }
                let delay = enc.lookahead();
                let best_snr = (delay - 3..=delay + 3)
                    .map(|lag| {
                        let mut signal = 0.0f64;
                        let mut noise = 0.0f64;
                        for i in sr / 10..frame * packets - lag {
                            for ch in 0..channels {
                                let x = f64::from(input[i * channels + ch]);
                                let y = f64::from(decoded[(i + lag) * channels + ch]);
                                signal += x * x;
                                noise += (x - y) * (x - y);
                            }
                        }
                        10.0 * (signal / noise.max(1e-30)).log10()
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                assert!(
                    best_snr > 0.0,
                    "{sr} Hz {ms} ms {channels}ch FEC SNR {best_snr}"
                );
            }
        }
    }
}

#[test]
fn silk_follows_a_change_of_complexity() {
    // libopus sets SILK's complexity up for every packet: complexity 9, 3
    // from 2 s and 6 from 4 s (VoIP, VBR, 20 kb/s). Hashes of every packet
    // of libopus main at the same settings.
    for (ms, channels, expected) in [
        (20, 1, silk_ref(0xc2b4123c72d2001d_u64, 0x4aa9d64c650bcdae)),
        (10, 2, silk_ref(0x13bab6f47995f635, 0x13c93e02f4146af1)),
    ] {
        let mono = speech_i16(16000);
        let frame = 16 * ms;
        let mut enc = OpusEncoder::new(16000, channels, Application::Voip).unwrap();
        enc.bitrate_bps = 20000;
        enc.set_signal(Signal::Voice);
        enc.set_bandwidth(Bandwidth::Wideband);
        let mut packet = [0u8; 1500];
        let hash = (0..mono.len() / frame).fold(FNV_OFFSET, |h, p| {
            enc.complexity = match p * ms {
                ..2000 => 9,
                ..4000 => 3,
                _ => 6,
            };
            let pcm: Vec<i16> = mono[p * frame..(p + 1) * frame]
                .iter()
                .flat_map(|&x| std::iter::repeat_n(x, channels))
                .collect();
            let n = enc.encode_i16(&pcm, frame, &mut packet).unwrap();
            packet_hash(h, &packet[..n])
        });
        assert_eq!(hash, expected, "{ms} ms {channels} channel(s)");
    }
}

#[test]
fn float_voip_input_rounds_like_libopus() {
    // libopus turns the VoIP input into 16-bit samples with FLOAT2INT16,
    // which rounds ties to even, the fixed-point build at its float entry
    // point, the float build after the high-pass: the fixture moved by half
    // a step, so every sample is a tie (16 kHz, 20 ms, 20 kb/s). Hash of
    // every packet of libopus main at the same settings.
    let pcm: Vec<f32> = speech_i16(16000)
        .iter()
        .map(|&x| (x as f32 + 0.5) / 32768.0)
        .collect();
    let mut enc = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
    enc.bitrate_bps = 20000;
    enc.complexity = 9;
    enc.set_signal(Signal::Voice);
    enc.set_bandwidth(Bandwidth::Wideband);
    let mut packet = [0u8; 1500];
    let hash = pcm.chunks_exact(320).fold(FNV_OFFSET, |h, frame| {
        let n = enc.encode(frame, 320, &mut packet).unwrap();
        packet_hash(h, &packet[..n])
    });
    assert_eq!(hash, silk_ref(0x5fdf12e5a3df2fea, 0x56404b345a8d4159));
}

#[test]
fn silk_follows_a_change_of_frame_duration() {
    // 40 ms SILK-only packets, then 40 ms hybrid packets, which are two 20 ms
    // frames: SILK must set its packet up for 20 ms frames again, as libopus
    // does for every packet (silk_setup_fs). FEC at 30 % loss narrows 20 kb/s
    // to SILK-only, no loss keeps it hybrid. Stereo SILK failed to encode,
    // mono SILK coded 20 ms of input as 40 ms.
    let (sr, frame) = (24000, 960);
    for channels in [1, 2] {
        let packets = 50;
        let input = vbr::speech(sr, channels, frame * packets);
        let mut enc = OpusEncoder::new(sr as i32, channels, Application::Voip).unwrap();
        enc.bitrate_bps = 20000;
        enc.use_inband_fec = true;
        enc.complexity = 9;
        // Super-wideband, which no loss codes as hybrid.
        enc.set_bandwidth(Bandwidth::Superwideband);
        let c_channels = if channels == 1 {
            CCh::Mono
        } else {
            CCh::Stereo
        };
        let mut dec = CDec::new(sr as u32, c_channels).unwrap();
        let (mut decoded, mut modes) = (Vec::new(), String::new());
        let mut pcm = vec![0.0; frame * channels];
        let mut packet = [0u8; 1500];
        for p in 0..packets {
            enc.packet_loss_perc = if (p / 10).is_multiple_of(2) { 30 } else { 0 };
            let data = &input[p * frame * channels..(p + 1) * frame * channels];
            let n = enc.encode(data, frame, &mut packet).unwrap();
            modes.push(if packet[0] >> 3 < 12 { 'S' } else { 'H' });
            assert_eq!(
                dec.decode_float(&packet[..n], &mut pcm, false).unwrap(),
                frame
            );
            decoded.extend_from_slice(&pcm);
        }
        assert!(modes.contains("SH") && modes.contains("HS"), "{modes}");
        let lag = enc.lookahead();
        let (mut signal, mut noise) = (0.0f64, 0.0f64);
        for i in sr / 10..frame * packets - lag {
            for ch in 0..channels {
                let x = f64::from(input[i * channels + ch]);
                let y = f64::from(decoded[(i + lag) * channels + ch]);
                signal += x * x;
                noise += (x - y) * (x - y);
            }
        }
        let snr = 10.0 * (signal / noise).log10();
        assert!(snr > 3.0, "{channels} channel(s): {snr:.2} dB");
    }
}

#[test]
fn hybrid_vbr_rate_tracks_libopus() {
    // The overshoot reported in #73 (24 kHz / 20 ms / 28 kb/s) came from the
    // CELT layer of hybrid packets, which spent the packet's capacity.
    for (sr, ms, rate, fec) in [
        (24000, 20, 28000, false),
        (24000, 10, 32000, false),
        (48000, 20, 28000, true),
    ] {
        let frame = sr * ms / 1000;
        let packets = 4000 / ms;
        let input = vbr::speech(sr, 1, frame * packets);
        let mut rs = OpusEncoder::new(sr as i32, 1, Application::Voip).unwrap();
        rs.bitrate_bps = rate;
        rs.complexity = 9;
        rs.use_cbr = false;
        rs.use_inband_fec = fec;
        rs.packet_loss_perc = if fec { 10 } else { 0 };
        let mut c = CEnc::new(sr as u32, CCh::Mono, CApp::Voip).unwrap();
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
        c.set_inband_fec(fec).unwrap();
        c.set_packet_loss_perc(if fec { 10 } else { 0 }).unwrap();
        let mut bytes = [0usize; 2];
        let mut packet = [0u8; 1500];
        for p in 0..packets {
            let pcm = &input[p * frame..(p + 1) * frame];
            let n_rs = rs.encode(pcm, frame, &mut packet).unwrap();
            let n_c = c.encode_float(pcm, &mut packet).unwrap();
            if p >= 10 {
                bytes[0] += n_rs;
                bytes[1] += n_c;
            }
        }
        let ratio = bytes[0] as f64 / bytes[1] as f64;
        println!("{sr} Hz {ms} ms {rate} b/s FEC={fec}: opus-rs/libopus {ratio:.4}");
        assert!(
            (0.95..=1.05).contains(&ratio),
            "{sr} Hz {ms} ms {rate} b/s FEC={fec}: {ratio}"
        );
    }
}

/// opus-rs, whose SILK packets equal the reference build's
/// (fuzz/opusref/tests/silk_fixed.rs, silk_float.rs), 41 frames per hash:
/// with `silk-fixed`.
#[cfg(feature = "silk-fixed")]
const TRACE_BLOCKS: [u64; 22] = [
    0xef0e22ef06b524cc,
    0x42840a8ea5bc5ebe,
    0xc2b7acacf2be3800,
    0x783a6b2da8953933,
    0x66026bdd4586a864,
    0x6eed661b974b1240,
    0x5fab1d089df61d7a,
    0x3724bb1f981437d5,
    0x21f73470eb9cc406,
    0xf63c550c92ee5144,
    0xe07bff6b1dfda685,
    0x65043fe5156f455c,
    0x98823529c211241d,
    0xfca1b521407b4c86,
    0xd27b6b985e932953,
    0xc8d2524918391d3b,
    0x96e43011966a383e,
    0x0ce7ef9cc3d0f846,
    0xd97120eed0ae2680,
    0x6286cbf46ed45af0,
    0xfc949fd2791bd800,
    0xad7941cc419cd65b,
];
/// The same with the float encoder.
#[cfg(not(feature = "silk-fixed"))]
const TRACE_BLOCKS: [u64; 22] = [
    0x7dde110e2489533d,
    0x74fb652497b66905,
    0xab43e1042c9529f6,
    0x3b588a7668b9af8a,
    0xf0adc0777c3edc07,
    0x7f9ecd1146958cba,
    0x85154fe6e5e05319,
    0x57bd46f9c4a3d320,
    0x2db3c7818c21f418,
    0x1039d9807888671a,
    0xd3d3449d27365cca,
    0x699bb4258fded34f,
    0x9b8b2fda8fa3cb4e,
    0x26fbbc367946b323,
    0x80698aeb2897c74c,
    0x0767edc075375312,
    0x789a10eb03ddaed4,
    0x05ffcdc4065abe6a,
    0xde59f9c46f888bca,
    0x822422e524bc6737,
    0x72f223a866f62340,
    0x5a985774be6c6c1a,
];

/// libopus main fixed point: sample rate, frame ms, bitrate, channels and
/// the hashes of each 200 packets. Stereo at 12 kb/s codes one SILK
/// channel, as libopus chooses.
#[cfg(feature = "silk-fixed")]
const PACKET_BLOCKS: [(usize, usize, i32, usize, &[u64]); 5] = [
    (
        16000,
        20,
        20000,
        1,
        &[
            0x336fafd05cbc1929,
            0x537e0e5f99fe3c43,
            0xf3a3ab6941299249,
            0xbd530a03e5c15180,
            0xb475dca8ac88a5a5,
        ],
    ),
    (
        16000,
        20,
        24000,
        1,
        &[
            0x678749bc9f8fc56d,
            0x8bde1075dab1641c,
            0xb6c5f7e0482a364f,
            0xe44b7de79b47a3a6,
            0x097d92bffee1146c,
        ],
    ),
    (
        16000,
        10,
        20000,
        1,
        &[
            0xdf718265f0079649,
            0x5ad18ed8d57cb0ee,
            0x634d3d6d9963360e,
            0x8ff602cfe1b58444,
            0x87d0a49d6d713d89,
            0xa0ae19a527ad0779,
            0x6e62c00546dafe94,
            0x96272563821572c0,
            0x91b1ca65f4bab004,
            0xf91a53a18012bce9,
        ],
    ),
    (
        16000,
        10,
        12000,
        2,
        &[
            0x7d75c694186bbec1,
            0x23b3e15982fd7da3,
            0xccd4d3cda7f58f25,
            0xaeac731e62b00198,
            0x19120f8c80530adb,
            0x63a3e75a6aa32dee,
            0x512cfa05d42cda7f,
            0xdd62b0959d2952a2,
            0x0de2cd31d596b40f,
            0x7d33cd19ec91c0d3,
        ],
    ),
    (
        12000,
        20,
        20000,
        1,
        &[
            0xffad26f941860f1c,
            0x4dd665c734da1d59,
            0xb7ace9d570144f7a,
            0xfa8b230444a5fb0f,
            0xebcb44488f0a4b27,
        ],
    ),
];
/// The same of float libopus main.
#[cfg(not(feature = "silk-fixed"))]
const PACKET_BLOCKS: [(usize, usize, i32, usize, &[u64]); 5] = [
    (
        16000,
        20,
        20000,
        1,
        &[
            0x5454015cc633660f,
            0x902bb712e50a707b,
            0x33d39440dfc80b91,
            0xb2649f512d74f6db,
            0x3e0f49f81a5e7c52,
        ],
    ),
    (
        16000,
        20,
        24000,
        1,
        &[
            0x874e52ff7cee0194,
            0x4a6b6a363922bb67,
            0xd5e5671eddc3572f,
            0x753ebb8b496e9a4c,
            0x7d5f598fbcb5e26d,
        ],
    ),
    (
        16000,
        10,
        20000,
        1,
        &[
            0xa89496972cb71d33,
            0x8f886403ff286c3a,
            0xa65d320cffe0d22a,
            0xf36887ed3895ffff,
            0x5fdc9b5cf53a70a9,
            0x93e28113a7d8d22b,
            0xc50449e476fd76f6,
            0x5f3fc2ff9a3edec5,
            0x76a76e2afd619352,
            0xabf5856271449320,
        ],
    ),
    (
        16000,
        10,
        12000,
        2,
        &[
            0xdb15d015ce2349c5,
            0xd448d8f35ff13915,
            0x2e63f63083b316e5,
            0xd63a181821fc0ebf,
            0xc083fa082df8e24e,
            0x1a6eed08627f854b,
            0x46011ce1dcb3d768,
            0xd679ca7e8bc4989f,
            0xd3390e1dda48e132,
            0xbc4952ab251da65c,
        ],
    ),
    (
        12000,
        20,
        20000,
        1,
        &[
            0x07f23cca2bf40720,
            0xa8c72702fa59cceb,
            0xc3e935c3fca4e03e,
            0x082318d72939e9bd,
            0x6113a7cbcf5e4db3,
        ],
    ),
];
