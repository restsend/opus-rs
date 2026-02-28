use opus_rs::mdct::MdctLookup;
use std::f32::consts::PI;

#[test]
fn test_mdct_tdac() {
    let n = 960;
    let overlap = 120;
    let lookup = MdctLookup::new(n, 0);
    let mut window = vec![0.0f32; overlap];
    for i in 0..overlap {
        let x = (i as f32 + 0.5) / overlap as f32;
        window[i] = (PI / 2.0 * (PI / 2.0 * x).sin().powi(2)).sin();
    }

    let mut in_buf = vec![0.0f32; n + overlap];
    for i in 0..n {
        in_buf[overlap / 2 + i] = (i as f32 * 0.1).sin();
    }

    let mut freq = vec![0.0f32; n / 2];
    lookup.forward(&in_buf[overlap / 2..], &mut freq, &window, overlap, 0, 1);

    let mut out_buf = vec![0.0f32; n + overlap];
    lookup.backward(&freq, &mut out_buf, &window, overlap, 0, 1);
}
