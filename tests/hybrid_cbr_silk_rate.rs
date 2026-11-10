//! CBR SILK rate control follows fixed-point libopus.
//!
//! In hybrid CBR, libopus codes SILK as VBR under a cap and leaves the rest
//! of the packet to CELT (opus_encoder.c:2163-2178), so the SILK encoder
//! states run with `useCBR` 0, which the noise shaping reads
//! (control_codec.c:76, noise_shape_analysis_FIX.c:171). In SILK-only CBR
//! with FEC, the LBRR section is coded however little room it leaves: the
//! rate control squeezes the frame into the rest, and a frame that still
//! busts the packet makes it a packet of one zero byte, padded, that decoders
//! conceal (enc_API.c:354-407, opus_encoder.c:2578-2589). In stereo, each
//! channel's rate state machine that is ready to switch the internal rate
//! makes room for the redundancy (enc_API.c:260-268,
//! control_audio_bandwidth.c:86-90).
//!
//! The tests encode all 18 s of the fixture with 16-bit input in CBR, VoIP,
//! complexity 9, and compare with hashes of fixed-point libopus 1.6.1's
//! packets at the same settings (voice signal, the bandwidth and the stream's
//! channel count forced; loss rate 10% with FEC).

// Only the fixture reader is used here.
#[allow(dead_code)]
#[path = "common/vbr.rs"]
mod vbr;

use opus_rs::range_coder::RangeCoder;
use opus_rs::silk::dec_api::SilkDecoder;
use opus_rs::{Application, OpusEncoder};

fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;

/// The length and bytes of every packet.
fn packets_hash(packets: &[Vec<u8>]) -> u64 {
    packets.iter().fold(FNV_OFFSET, |h, p| {
        fnv(fnv(h, &(p.len() as u32).to_le_bytes()), p)
    })
}

/// The frames of a packet without its padding (RFC 6716 section 3.2).
fn frames(p: &[u8]) -> Vec<&[u8]> {
    let size = |d: &[u8]| {
        if d[0] < 252 {
            (usize::from(d[0]), 1)
        } else {
            (usize::from(d[0]) + 4 * usize::from(d[1]), 2)
        }
    };
    match p[0] & 3 {
        0 => vec![&p[1..]],
        1 => p[1..].chunks((p.len() - 1) / 2).collect(),
        2 => {
            let (n, k) = size(&p[1..]);
            vec![&p[1 + k..1 + k + n], &p[1 + k + n..]]
        }
        _ => {
            let count = usize::from(p[1] & 63);
            let (mut at, mut padding) = (2, 0);
            if p[1] & 0x40 != 0 {
                loop {
                    let b = usize::from(p[at]);
                    at += 1;
                    padding += if b == 255 { 254 } else { b };
                    if b != 255 {
                        break;
                    }
                }
            }
            let end = p.len() - padding;
            let mut lens = Vec::new();
            if p[1] & 0x80 != 0 {
                for _ in 0..count - 1 {
                    let (n, k) = size(&p[at..]);
                    lens.push(n);
                    at += k;
                }
                lens.push(end - at - lens.iter().sum::<usize>());
            } else {
                lens = vec![(end - at) / count; count];
            }
            let mut out = Vec::new();
            for n in lens {
                out.push(&p[at..at + n]);
                at += n;
            }
            out
        }
    }
}

/// Decode the SILK layer of every SILK-only and hybrid frame, the SILK
/// decoder reset after CELT, and return a hash of the bits each takes and
/// the audio it decodes to, and the number of such frames. A frame of 0 or 1
/// byte, which decoders conceal, counts as one without decoding.
fn silk_layers(packets: &[Vec<u8>], sr: i32, channels: i32) -> (u64, usize) {
    let mut silk = SilkDecoder::new();
    silk.init(sr.min(16000), channels);
    let (mut hash, mut layers, mut after_celt) = (FNV_OFFSET, 0, false);
    let mut out = [0i16; 2 * 960];
    for p in packets {
        let config = usize::from(p[0] >> 3);
        if config >= 16 {
            after_celt = true;
            continue;
        }
        if after_celt {
            silk.reset();
            after_celt = false;
        }
        let (rate, ms) = if config >= 12 {
            (16000, [10, 20][config % 2])
        } else {
            (
                [8000, 12000, 16000][config / 4],
                [10, 20, 40, 60][config % 4],
            )
        };
        let coded = if p[0] & 4 != 0 { 2 } else { 1 };
        for frame in frames(p) {
            layers += 1;
            if frame.len() <= 1 {
                hash = fnv(hash, b"concealed");
                continue;
            }
            let mut rc = RangeCoder::new_decoder(frame);
            let (mut decoded, total) = (0, (ms * rate / 1000) as usize);
            while decoded < total {
                let n =
                    silk.decode_with_channels(&mut rc, &mut out, 0, decoded == 0, ms, rate, coded);
                assert!(n > 0);
                hash = fnv(hash, &rc.tell().to_le_bytes());
                for s in &out[..n as usize * coded.min(channels) as usize] {
                    hash = fnv(hash, &s.to_le_bytes());
                }
                decoded += n as usize;
            }
        }
    }
    (hash, layers)
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

/// Encode the whole fixture in CBR at `rate(ms)` b/s, both channels equal,
/// with FEC at 10% loss or without.
fn encode_cbr(
    sr: usize,
    ms: usize,
    channels: usize,
    fec: bool,
    rate: impl Fn(usize) -> i32,
) -> Vec<Vec<u8>> {
    let mono = speech_i16(sr);
    let frame = sr * ms / 1000;
    let mut enc = OpusEncoder::new(sr as i32, channels, Application::Voip).unwrap();
    enc.complexity = 9;
    enc.use_cbr = true;
    enc.use_inband_fec = fec;
    enc.packet_loss_perc = if fec { 10 } else { 0 };
    let mut packet = [0u8; 1500];
    (0..mono.len() / frame)
        .map(|p| {
            enc.bitrate_bps = rate(p * ms);
            let pcm: Vec<i16> = mono[p * frame..(p + 1) * frame]
                .iter()
                .flat_map(|&x| std::iter::repeat_n(x, channels))
                .collect();
            let n = enc.encode_i16(&pcm, frame, &mut packet).unwrap();
            packet[..n].to_vec()
        })
        .collect()
}

#[test]
fn hybrid_cbr_silk_layer_matches_fixed_point_libopus() {
    // Hybrid CBR at 16-32 kb/s, with and without FEC. Before this change the
    // SILK layer parted from libopus's soon after speech activity first fell
    // (at packets 21, 18 and 18 here): libopus then lowers the coding SNR,
    // and opus-rs, running SILK as CBR, did not.
    let cases: [(usize, usize, usize, i32, bool, u64); 3] = [
        (24000, 20, 1, 16000, false, 0x567928009ad3872d),
        (48000, 10, 2, 24000, false, 0xff39be8bf9e4a6e5),
        (48000, 20, 1, 32000, true, 0xdda780ca600b0a9c),
    ];
    for (sr, ms, channels, rate, fec, expected) in cases {
        let packets = encode_cbr(sr, ms, channels, fec, |_| rate);
        assert!(packets.iter().all(|p| (12..16).contains(&(p[0] >> 3))));
        assert_eq!(
            silk_layers(&packets, sr as i32, channels as i32),
            (expected, packets.len()),
            "{sr} Hz {ms} ms {channels} ch {rate} b/s FEC {fec}"
        );
    }
}

#[test]
fn hybrid_celt_cbr_switches_keep_silk_like_fixed_point_libopus() {
    // CBR switching between hybrid at 20 kb/s and CELT at 100 kb/s every
    // 400 ms: the SILK layer of every hybrid frame, through each restart.
    let cases: [(usize, usize, usize, u64, usize); 2] = [
        (48000, 20, 1, 0xce8d84b6ef3d6144, 483),
        (24000, 10, 2, 0xdc6674a2f2f6cedf, 943),
    ];
    for (sr, ms, channels, expected, layers) in cases {
        let packets = encode_cbr(sr, ms, channels, false, |at| {
            if (at / 400).is_multiple_of(2) {
                20000
            } else {
                100000
            }
        });
        assert_eq!(
            silk_layers(&packets, sr as i32, channels as i32),
            (expected, layers),
            "{sr} Hz {ms} ms {channels} ch"
        );
    }
}

#[test]
fn cbr_fec_packets_match_fixed_point_libopus() {
    // With FEC at 10% loss these streams are SILK-only, and every packet is
    // byte-identical to fixed-point libopus's, also the 15, 1 and 23 whose
    // frame the LBRR section leaves too little room and that decoders
    // conceal. Before this change opus-rs dropped the LBRR section from such
    // packets instead (mono at 24 kb/s parts from libopus at packet 24), and
    // in stereo cut the cap of the packet before an internal-rate switch
    // once where both channels' state machines cut it (from packet 24 at
    // 20 kb/s).
    let cases: [(usize, usize, usize, i32, u64, usize); 3] = [
        (24000, 10, 1, 24000, 0x00de474e3e2f5e7c, 15),
        (24000, 20, 2, 20000, 0x0a752d6e04b25a02, 1),
        (48000, 10, 2, 28000, 0xac0af6e75ea8cf97, 23),
    ];
    for (sr, ms, channels, rate, expected, concealed) in cases {
        let packets = encode_cbr(sr, ms, channels, true, |_| rate);
        let busted = packets
            .iter()
            .filter(|p| frames(p).iter().all(|f| f.len() <= 1))
            .count();
        assert_eq!(
            (packets_hash(&packets), busted),
            (expected, concealed),
            "{sr} Hz {ms} ms {channels} ch {rate} b/s"
        );
    }
}
