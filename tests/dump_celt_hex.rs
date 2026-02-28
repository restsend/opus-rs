/// Dump CELT encoded packets as hex for cross-decoder testing
use opus_rs::celt::CeltEncoder;
use opus_rs::modes::default_mode;
use opus_rs::range_coder::RangeCoder;

#[test]
fn dump_celt_packets_hex() {
    let mode = default_mode();
    let channels = 1;
    let frame_size = 960;
    let n_bytes = 160;

    let mut encoder = CeltEncoder::new(mode, channels);

    let freq = 440.0;

    for f in 0..5 {
        let mut pcm_in = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            let sample = f * frame_size + i;
            let t = sample as f32 / 48000.0;
            pcm_in[i] = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.4;
        }

        let mut rc = RangeCoder::new_encoder(n_bytes as u32);
        encoder.encode(&pcm_in, frame_size, &mut rc);
        rc.done();

        // Print packet as hex (full buffer)
        let packet = &rc.buf[..n_bytes];
        let hex: String = packet.iter().map(|b| format!("{:02x}", b)).collect();
        println!("PACKET_FRAME_{}: {}", f, hex);
        println!("  offs={} end_offs={} tell_bits={}", rc.offs, rc.end_offs, rc.tell());

        // Also print with TOC byte prepended (for opus_decode)
        let toc = 0xf8u8; // Fullband CELT 20ms mono
        print!("OPUS_PACKET_{}: {:02x}", f, toc);
        for b in packet.iter() {
            print!("{:02x}", b);
        }
        println!();
    }
}
