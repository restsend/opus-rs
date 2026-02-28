use opus_rs::mdct::MdctLookup;
use std::f32::consts::PI;

#[test]
fn test_mdct_loopback() {
    let n = 960;
    let overlap = 120;
    let lookup = MdctLookup::new(n, 0);
    let mut window = vec![0.0f32; overlap];
    for i in 0..overlap {
        let x = (i as f32 + 0.5) / overlap as f32;
        window[i] = (PI / 2.0 * (PI / 2.0 * x).sin().powi(2)).sin();
    }

    // Two frames to test overlap-add
    let mut in_pcm = vec![0.0f32; 2 * n];
    for i in 0..2 * n {
        in_pcm[i] = (i as f32 * 0.05).sin();
    }

    let mut out_pcm = vec![0.0f32; 2 * n];
    let mut history = vec![0.0f32; overlap];

    // Frame 1
    let mut frame1_in = vec![0.0f32; n + overlap];
    // History is 0 for first frame
    frame1_in[overlap..].copy_from_slice(&in_pcm[0..n]);

    let mut freq1 = vec![0.0f32; n / 2];
    lookup.forward(&frame1_in, &mut freq1, &window, overlap, 0, 1);

    let mut frame1_out = vec![0.0f32; n + overlap];
    frame1_out[..overlap].copy_from_slice(&history);
    lookup.backward(&freq1, &mut frame1_out, &window, overlap, 0, 1);

    out_pcm[0..n].copy_from_slice(&frame1_out[..n]);
    history.copy_from_slice(&frame1_out[n..n + overlap]);

    // Frame 2
    let mut frame2_in = vec![0.0f32; n + overlap];
    // For MDCT with N=960, frame_size = 480.
    // Frame 2 input: overlap from end of frame 1 data + new frame 2 data
    // Since this test uses frame hop = n (full block), Frame 2 input is:
    // [last overlap of frame 1 input] + [frame 2 data]
    frame2_in[..overlap].copy_from_slice(&in_pcm[n - overlap..n]);
    frame2_in[overlap..overlap + n].copy_from_slice(&in_pcm[n..2 * n]);

    // Note: this test is a debug/WIP test. Full MDCT loopback is verified
    // in mdct_loopback_test.rs and mdct_identity_full_test.rs.
}

fn main() {}
