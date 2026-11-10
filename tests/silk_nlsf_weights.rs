//! SILK weights the NLSFs as libopus does, before stabilizing them.
//!
//! libopus computes the Laroia weights of the NLSF quantizer from the
//! unquantized NLSFs as the LPC analysis left them (process_NLSFs.c:63), and
//! stabilizes them only inside `silk_NLSF_encode` (NLSF_encode.c:67). When
//! two NLSFs come out of the analysis in the wrong order, the weights of the
//! stabilized vector differ, and so do the coded NLSF indices. The test
//! encodes all 18 s of the fixture with 16-bit input at 16 kHz, 20 ms,
//! 12 kb/s, CBR, VoIP, complexity 9, where that happens at packet 526, and
//! compares every packet with a hash of fixed-point libopus 1.6.1's at the
//! same settings (voice signal, wideband, mono; loss rate 10% with FEC).

// Only the fixture reader is used here.
#[allow(dead_code)]
#[path = "common/vbr.rs"]
mod vbr;

use opus_rs::{Application, OpusEncoder};

fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &b| {
        (hash ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
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

#[test]
fn silk_nlsf_weights_match_fixed_point_libopus() {
    // Every packet's length and bytes. FEC changes nothing at this rate:
    // libopus, and opus-rs, code no LBRR data.
    for fec in [false, true] {
        let mono = speech_i16(16000);
        let frame = 320;
        let mut enc = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
        enc.bitrate_bps = 12000;
        enc.complexity = 9;
        enc.use_cbr = true;
        enc.use_inband_fec = fec;
        enc.packet_loss_perc = if fec { 10 } else { 0 };
        let mut packet = [0u8; 1500];
        let mut hash = 0xcbf29ce484222325;
        let mut packets = 0;
        for pcm in mono.chunks_exact(frame) {
            let n = enc.encode_i16(pcm, frame, &mut packet).unwrap();
            hash = fnv(fnv(hash, &(n as u32).to_le_bytes()), &packet[..n]);
            packets += 1;
        }
        assert_eq!(
            (hash, packets),
            (0x7f90f3deb1a4d284, 902),
            "16 kHz 20 ms 12 kb/s CBR, FEC {fec}"
        );
    }
}
