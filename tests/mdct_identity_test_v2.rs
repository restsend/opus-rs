
use opus_rs::mdct::MdctLookup;

#[test]
fn test_mdct_roundtrip() {
    let n = 256;
    let overlap = 64;
    let mdct = MdctLookup::new(n, 0);

    let mut window = vec![0.0f32; overlap];
    for i in 0..overlap {
        let x = (i as f32 + 0.5) / overlap as f32;
        window[i] = (std::f32::consts::PI * 0.5 * ( (std::f32::consts::PI * 0.5 * x).sin() ).powi(2)).sin();
    }

    // MDCT forward needs n + overlap samples
    let input_size = n + overlap;
    let mut freq1 = vec![0.0; n/2];
    let mut freq2 = vec![0.0; n/2];

    let input1 = vec![1.0f32; input_size];
    let input2 = vec![1.0f32; input_size];

    mdct.forward(&input1, &mut freq1, &window, overlap, 0, 1);
    mdct.forward(&input2, &mut freq2, &window, overlap, 0, 1);

    // MDCT backward outputs n + overlap samples
    let mut out = vec![0.0; n * 2];
    let mut history = vec![0.0; overlap / 2];

    // Frame 1
    let mut out_f1 = vec![0.0; n + overlap];
    out_f1[..overlap/2].copy_from_slice(&history);
    mdct.backward(&freq1, &mut out_f1, &window, overlap, 0, 1);
    history.copy_from_slice(&out_f1[n..n + overlap/2]);
    out[..n/2].copy_from_slice(&out_f1[..n/2]);

    // Frame 2
    let mut out_f2 = vec![0.0; n + overlap];
    out_f2[..overlap/2].copy_from_slice(&history);
    mdct.backward(&freq2, &mut out_f2, &window, overlap, 0, 1);
    out[n/2..n].copy_from_slice(&out_f2[..n/2]);

    println!("Out[n/2-10..n/2+10]: {:?}", &out[n/2-10..n/2+10]);
}
