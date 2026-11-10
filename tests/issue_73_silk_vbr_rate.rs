//! SILK conformance and the hybrid VBR rate of #73.
//!
//! The reference values come from fixed-point libopus 1.6.1 (scalar build),
//! which the opus-rs SILK encoder follows: its symbols, the packets of whole
//! encodes and the helper functions below match it exactly.

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
                silk_init_encoder(&mut enc, 0);
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
    silk_init_encoder(&mut enc, 0);
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
    (0..source.len() * sr / 16000)
        .map(|i| {
            let pos = i * 16000;
            let at = pos / sr;
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
fn silk_packets_match_fixed_point_libopus() {
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
    // redundancy, which libopus codes in fixed point; all else matches.
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
    assert_eq!((packets[24].len(), packets[25].len()), (77, 69));
    let hash = packets.iter().enumerate().fold(FNV_OFFSET, |h, (i, p)| {
        if (24..=25).contains(&i) {
            fnv(fnv(h, &(p.len() as u32).to_le_bytes()), &p[..p.len() - 27])
        } else {
            packet_hash(h, p)
        }
    });
    assert_eq!(hash, 0x63147ec8ca3250f8);
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

/// libopus 1.6.1 fixed point, 41 frames per hash.
const TRACE_BLOCKS: [u64; 22] = [
    0x71e78e86d6f4cb72,
    0x4490e71ef6f48ad3,
    0x545d7888d0b398d5,
    0x5c9375677fff91a2,
    0x4a16c9c9983e1bf0,
    0xc7ec689d1a4ce0ca,
    0xb33b2a7497c721de,
    0xac67cd3216d9742f,
    0xfd5cc9a73154c6c4,
    0x275c1f101a4e9538,
    0x3e4bfa86928138e7,
    0xf90a9e1e9e082cad,
    0xad1179fefdbdc01e,
    0xec74cdb690ee3b3d,
    0x3539b2f89d83bb06,
    0xcebdc9d19d8f8bb5,
    0xf97fe77d306319fc,
    0xa39e0e40bda7b1c7,
    0x29fa8ab0410faa24,
    0x09c60d0112af7578,
    0x3921083ebc95c076,
    0xbcc87deb72449060,
];

/// libopus 1.6.1 fixed point: sample rate, frame ms, bitrate, channels and
/// the hashes of each 200 packets.
const PACKET_BLOCKS: [(usize, usize, i32, usize, &[u64]); 5] = [
    (
        16000,
        20,
        20000,
        1,
        &[
            0x1afdaa98f4693731,
            0x651e503410d59f7f,
            0x28e67b0ac6241f0a,
            0x0b58e27f109c4db7,
            0xef04355cea592650,
        ],
    ),
    (
        16000,
        20,
        24000,
        1,
        &[
            0x5fc8df4ea8f334ae,
            0x981baa4dd6c5540e,
            0x28a98fa198dfe05d,
            0x229f85f5c2c61e74,
            0xe9d300a6fce337cc,
        ],
    ),
    (
        16000,
        10,
        20000,
        1,
        &[
            0x060ffd423647f90c,
            0xe01be72e1c2ccdef,
            0x0e661541ce72e897,
            0x874dd96047c090c4,
            0x862c81c8164e5e77,
            0x23c91c6d670b35e3,
            0xc4c724a7216ee1dc,
            0xdc1eda980f821ff8,
            0xc2886f02e67bc478,
            0x91defcedfe78afdd,
        ],
    ),
    (
        16000,
        10,
        12000,
        2,
        &[
            0x9d7263ec3d7d71f1,
            0x144101e249010883,
            0x35a7162158f70e60,
            0xffe7c58676fce654,
            0xb7845f2bbcfb1caa,
            0x64cdeb84999b3b2a,
            0xe4ce0e257f931f39,
            0xf7fc0bae469a4d0f,
            0x8094e69178a01237,
            0x532d71fe70adf4f4,
        ],
    ),
    (
        12000,
        20,
        20000,
        1,
        &[
            0x25e3c6df073b5838,
            0x4f6b585b59c4427c,
            0x090171dfc4045826,
            0x275e24518653411b,
            0x9391901f034d475f,
        ],
    ),
];
