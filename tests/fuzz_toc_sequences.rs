//! Deterministic TOC-sequence fuzz harness for `OpusDecoder::decode`.
//!
//! Previous fuzz coverage only threw single packets at fresh decoders or
//! replayed encoder output; nothing exercised a *persistent* decoder across
//! arbitrary TOC sequences (bandwidth/mode switches, 1-byte PLC packets,
//! garbage payloads). That gap is exactly what let the cross-bandwidth
//! `lag_prev` panic (issue #27 deep scan) survive.
//!
//! This harness drives one decoder per (rate, channels) with a seeded LCG —
//! no external deps, fully reproducible — mixing:
//!   * random TOC byte + random-length garbage payload,
//!   * 1-byte payloads (PLC trigger),
//!   * truncated packets.
//! Decode errors are fine; panics are not.

use opus_rs::OpusDecoder;

/// xorshift64* — deterministic, dependency-free.
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u32() as usize) % n
        }
    }
}

/// RFC 6716 §3.1: samples per frame for a TOC config at a given API rate.
fn toc_frame_samples(toc: u8, rate: i32) -> usize {
    let config = (toc >> 3) as i32;
    let duration_ms: i32 = if config < 12 {
        // SILK: 10/20/40/60 ms
        [10, 20, 40, 60][(config % 4) as usize]
    } else if config < 16 {
        // Hybrid SWB: 10/20/40/60 ms
        [10, 20, 40, 60][(config % 4) as usize]
    } else if config < 20 {
        // Hybrid FB: 10/20/40/60 ms
        [10, 20, 40, 60][(config % 4) as usize]
    } else {
        // CELT: 2.5/5/10/20 ms
        [2, 5, 10, 20][(config % 4) as usize]
    };
    (duration_ms * rate / 1000) as usize
}

fn run_decoder(rate: i32, channels: usize, seed: u64, iterations: usize) {
    let mut dec = OpusDecoder::new(rate, channels).expect("decoder");
    let mut rng = Lcg(seed);
    let max_frame = toc_frame_samples(0x3F, rate).max(rate as usize / 25 * 3);
    let mut pcm = vec![0.0f32; max_frame * channels + 2];

    for _ in 0..iterations {
        // Packet shape: 40% garbage TOC + payload, 20% 1-byte (PLC),
        // 20% valid-ish SILK/CELT TOC + garbage, 20% truncated garbage.
        let mode = rng.below(5);
        let pkt: Vec<u8> = match mode {
            0 | 1 => {
                let toc = rng.next_u32() as u8;
                let len = rng.below(300);
                let mut p = Vec::with_capacity(len + 1);
                p.push(toc);
                for _ in 0..len {
                    p.push(rng.next_u32() as u8);
                }
                p
            }
            2 | 3 => {
                // 1-byte packet: TOC only (lost frame / DTC trigger).
                vec![rng.next_u32() as u8]
            }
            _ => {
                let len = rng.below(60);
                (0..len).map(|_| rng.next_u32() as u8).collect()
            }
        };
        if pkt.is_empty() {
            continue;
        }

        // Frame size derived from the packet TOC like a well-behaved caller;
        // occasionally also try an out-of-range size to abuse the API.
        let frame_size = if rng.below(10) == 0 {
            rng.below(max_frame + 1)
        } else {
            toc_frame_samples(pkt[0], rate)
        };

        // Errors are expected and fine; panics are the failure mode.
        let _ = dec.decode(&pkt, frame_size, &mut pcm);
    }
}

#[test]
fn fuzz_toc_sequences_mono_16k() {
    run_decoder(16000, 1, 0xDEAD_BEEF, 4000);
}

#[test]
fn fuzz_toc_sequences_stereo_48k() {
    run_decoder(48000, 2, 0x1234_5678, 3000);
}

#[test]
fn fuzz_toc_sequences_mono_8k() {
    run_decoder(8000, 1, 0x0BAD_C0DE, 3000);
}

#[test]
fn fuzz_toc_sequences_stereo_24k() {
    run_decoder(24000, 2, 0x5EED_5EED, 3000);
}

#[test]
fn fuzz_toc_sequences_bandwidth_flip_flop() {
    // Deterministic adversarial pattern aimed at bandwidth-switch +
    // packet-loss interactions: alternate WB voiced-like TOCs (config 8/9)
    // with 1-byte PLC packets and NB TOCs (config 0/1).
    let mut dec = OpusDecoder::new(16000, 1).expect("decoder");
    let mut rng = Lcg(0xABCD_1234);
    let mut pcm = vec![0.0f32; 1200];
    for i in 0..2000 {
        let pkt: Vec<u8> = match i % 4 {
            0 | 2 => {
                let toc = [0x40u8, 0x48, 0x44, 0x4C][rng.below(4)]; // WB SILK
                let len = 1 + rng.below(120);
                let mut p = vec![toc];
                for _ in 0..len {
                    p.push(rng.next_u32() as u8);
                }
                p
            }
            1 => vec![0x48u8], // 1-byte WB PLC trigger
            _ => {
                let toc = [0x00u8, 0x08, 0x04, 0x0C][rng.below(4)]; // NB/MB SILK
                let len = 1 + rng.below(60);
                let mut p = vec![toc];
                for _ in 0..len {
                    p.push(rng.next_u32() as u8);
                }
                p
            }
        };
        let frame_size = toc_frame_samples(pkt[0], 16000);
        let _ = dec.decode(&pkt, frame_size, &mut pcm);
    }
}
