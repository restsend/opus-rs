//! High-bitrate CELT interop reproduction for issue #11.
//!
//! Encodes stereo signals at 128–192 kbps with the crate encoder, then decodes
//! with (a) the crate decoder and (b) system libopus via ffmpeg, measuring SNR.
//! The issue reports the stream decodes to garbage in libopus above ~160 kbps
//! even though the encoder output looks fine in self-loopback.

use opus_rs::{Application, OpusDecoder, OpusEncoder};
use std::process::Command;

const FRAME_SIZE: usize = 960; // 20 ms @ 48 kHz

fn make_sine_stereo(freq_l: f32, freq_r: f32, n_frames: usize) -> Vec<f32> {
    let mut pcm = vec![0.0f32; n_frames * FRAME_SIZE * 2];
    for (i, x) in pcm.chunks_exact_mut(2).enumerate() {
        let t = i as f32 / 48000.0;
        x[0] = 0.5 * (2.0 * std::f32::consts::PI * freq_l * t).sin();
        x[1] = 0.5 * (2.0 * std::f32::consts::PI * freq_r * t).sin();
    }
    pcm
}

fn make_complex_stereo(n_frames: usize) -> Vec<f32> {
    // Three-tone chord per channel plus a little noise.
    let mut pcm = vec![0.0f32; n_frames * FRAME_SIZE * 2];
    let mut seed = 0x12345678u32;
    for (i, x) in pcm.chunks_exact_mut(2).enumerate() {
        let t = i as f32 / 48000.0;
        let mut l = 0.0f32;
        let mut r = 0.0f32;
        for &f in &[261.63f32, 329.63, 392.0] {
            l += 0.2 * (2.0 * std::f32::consts::PI * f * t).sin();
        }
        for &f in &[293.66f32, 349.23, 440.0] {
            r += 0.2 * (2.0 * std::f32::consts::PI * f * t).sin();
        }
        // xorshift noise
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let n = (seed & 0xffff) as f32 / 65535.0 - 0.5;
        x[0] = l + 0.02 * n;
        x[1] = r + 0.02 * n;
    }
    pcm
}

fn snr_db(input: &[f32], output: &[f32], delay_lo: i32, delay_hi: i32) -> f64 {
    // Measure over a late window (past the encoder+decoder algorithmic delay)
    // and search the delay band around the C-faithful 624-sample (13 ms) delay:
    // the CELT lookahead (240) plus the dc_reject/hp delay ring (192) and the
    // remaining opus-layer delay_compensation (~192).
    let n_active = input.len().min(24000);
    let active = (input.len() - n_active)..input.len();
    let mut best = f64::NEG_INFINITY;
    for delay in delay_lo..=delay_hi {
        let mut e_in = 0f64;
        let mut e_diff = 0f64;
        for i in active.clone() {
            let j = i as i64 + delay as i64;
            if j < 0 || j as usize >= output.len() {
                continue;
            }
            let a = input[i] as f64;
            let b = output[j as usize] as f64;
            e_in += a * a;
            let d = a - b;
            e_diff += d * d;
        }
        if e_diff < 1e-30 {
            return 200.0;
        }
        let v = 10.0 * (e_in / e_diff).log10();
        best = best.max(v);
    }
    best
}

fn encode_frames(enc: &mut OpusEncoder, pcm: &[f32]) -> Vec<Vec<u8>> {
    let n_frames = pcm.len() / (FRAME_SIZE * 2);
    let mut packets = Vec::with_capacity(n_frames);
    let mut buf = vec![0u8; 2048];
    for f in 0..n_frames {
        let frame = &pcm[f * FRAME_SIZE * 2..(f + 1) * FRAME_SIZE * 2];
        let len = enc
            .encode(frame, FRAME_SIZE, &mut buf)
            .expect("encode failed");
        packets.push(buf[..len].to_vec());
    }
    packets
}

fn decode_with_crate(packets: &[Vec<u8>]) -> Vec<f32> {
    let mut dec = OpusDecoder::new(48000, 2).unwrap();
    let mut out = Vec::with_capacity(packets.len() * FRAME_SIZE * 2);
    let mut buf = vec![0.0f32; FRAME_SIZE * 2];
    for p in packets {
        dec.decode(p, FRAME_SIZE, &mut buf).expect("decode failed");
        out.extend_from_slice(&buf);
    }
    out
}

// ── Minimal Ogg/Opus muxer ─────────────────────────────────────────────────

fn crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for (i, e) in table.iter_mut().enumerate() {
        let mut r = (i as u32) << 24;
        for _ in 0..8 {
            r = if r & 0x80000000 != 0 {
                (r << 1) ^ 0x04c11db7
            } else {
                r << 1
            };
        }
        *e = r;
    }
    table
}

fn ogg_crc(data: &[u8]) -> u32 {
    let table = crc_table();
    let mut crc = 0u32;
    for &b in data {
        crc = (crc << 8) ^ table[((crc >> 24) as u8 ^ b) as usize];
    }
    crc
}

fn ogg_page(serial: u32, seq: u32, granule: u64, header_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut segments = Vec::new();
    let mut remaining = payload.len();
    loop {
        if remaining == 0 {
            segments.push(0);
            break;
        }
        if remaining < 255 {
            segments.push(remaining as u8);
            break;
        }
        segments.push(255);
        remaining -= 255;
    }
    let mut page = Vec::with_capacity(27 + segments.len() + payload.len());
    page.extend_from_slice(b"OggS");
    page.push(0);
    page.push(header_type);
    page.extend_from_slice(&granule.to_le_bytes());
    page.extend_from_slice(&serial.to_le_bytes());
    page.extend_from_slice(&seq.to_le_bytes());
    page.extend_from_slice(&[0, 0, 0, 0]); // checksum placeholder
    page.push(segments.len() as u8);
    page.extend_from_slice(&segments);
    page.extend_from_slice(payload);
    let crc = ogg_crc(&page);
    page[22..26].copy_from_slice(&crc.to_le_bytes());
    page
}

fn mux_ogg(packets: &[Vec<u8>]) -> Vec<u8> {
    const SERIAL: u32 = 0x4f7075_73; // "Opus"
    let mut head = Vec::new();
    head.extend_from_slice(b"OpusHead");
    head.push(1); // version
    head.push(2); // channels
    head.extend_from_slice(&0u16.to_le_bytes()); // pre-skip
    head.extend_from_slice(&48000u32.to_le_bytes()); // input sample rate
    head.extend_from_slice(&0u16.to_le_bytes()); // output gain
    head.push(0); // channel mapping family

    let mut tags = Vec::new();
    tags.extend_from_slice(b"OpusTags");
    tags.extend_from_slice(&4u32.to_le_bytes());
    tags.extend_from_slice(b"test");
    tags.extend_from_slice(&0u32.to_le_bytes());

    let mut out = Vec::new();
    out.extend_from_slice(&ogg_page(SERIAL, 0, 0, 0x02, &head));
    out.extend_from_slice(&ogg_page(SERIAL, 1, 0, 0x00, &tags));
    let mut granule = 0u64;
    for (i, p) in packets.iter().enumerate() {
        granule += FRAME_SIZE as u64;
        let htype = if i + 1 == packets.len() { 0x04 } else { 0x00 };
        out.extend_from_slice(&ogg_page(SERIAL, (2 + i) as u32, granule, htype, p));
    }
    out
}

fn decode_with_ffmpeg(packets: &[Vec<u8>]) -> Option<Vec<f32>> {
    let ogg = mux_ogg(packets);
    let dir = std::env::temp_dir();
    let ogg_path = dir.join("opusrs_repro.ogg");
    let raw_path = dir.join("opusrs_repro.raw");
    std::fs::write(&ogg_path, &ogg).ok()?;
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(&ogg_path)
        .args(["-f", "f32le", "-ac", "2", "-ar", "48000"])
        .arg(&raw_path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let raw = std::fs::read(&raw_path).ok()?;
    let mut pcm = Vec::with_capacity(raw.len() / 4);
    for chunk in raw.chunks_exact(4) {
        pcm.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    Some(pcm)
}

#[test]
fn high_bitrate_interop() {
    let n_frames = 100; // 2 s
    let signals: [(&str, Vec<f32>); 2] = [
        ("sine", make_sine_stereo(440.0, 660.0, n_frames)),
        ("complex", make_complex_stereo(n_frames)),
    ];

    for (sig_name, pcm) in &signals {
        for &kbps in &[128, 160, 176, 192] {
            let mut enc = OpusEncoder::new(48000, 2, Application::Audio).unwrap();
            enc.bitrate_bps = kbps * 1000;
            let packets = encode_frames(&mut enc, pcm);

            let crate_out = decode_with_crate(&packets);
            let crate_snr = snr_db(pcm, &crate_out, 0, 700);

            let mut libopus_snr = f64::NEG_INFINITY;
            let ffmpeg_available;
            match decode_with_ffmpeg(&packets) {
                Some(ff_out) => {
                    ffmpeg_available = true;
                    libopus_snr = snr_db(pcm, &ff_out, 0, 700);
                }
                None => {
                    ffmpeg_available = false;
                }
            }

            let status = if libopus_snr > 15.0 || !ffmpeg_available {
                "ok"
            } else {
                "BROKEN"
            };
            println!(
                "{sig_name:8} {kbps:4}k crate={crate_snr:6.1}dB libopus={libopus_snr:6.1}dB [{status}]"
            );

            // Self-loopback must always be clean.
            assert!(
                crate_snr > 15.0,
                "{sig_name} @ {kbps}k: crate self-loopback degraded ({crate_snr:.1} dB)"
            );
            // Interop with libopus must be clean too (issue #11 regression).
            if ffmpeg_available {
                assert!(
                    libopus_snr > 15.0,
                    "{sig_name} @ {kbps}k: libopus decode of crate stream degraded ({libopus_snr:.1} dB)"
                );
            }
        }
    }
}
