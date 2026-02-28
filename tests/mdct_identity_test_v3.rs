use opus_rs::mdct::MdctLookup;

#[test]
fn test_mdct_scaling() {
    let n = 256;
    let overlap = 64;
    let mdct = MdctLookup::new(n, 0);
    let window = vec![1.0f32; overlap];
    let mut freq = vec![0.0; n / 2];
    let input = vec![1.0f32; n / 2 + overlap];
    mdct.forward(&input, &mut freq, &window, overlap, 0, 1);
    let mut out = vec![0.0; n / 2 + overlap];
    mdct.backward(&freq, &mut out, &window, overlap, 0, 1);
    println!("Value: {}", out[n / 4 + overlap / 2]);
}
