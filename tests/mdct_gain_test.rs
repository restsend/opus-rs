use opus_rs::mdct::MdctLookup;

#[test]
fn test_mdct_gain() {
    let n = 1920;
    let overlap = 120;
    let mdct = MdctLookup::new(n, 0);
    
    let mut window = vec![0.0f32; overlap];
    for i in 0..overlap {
        window[i] = (std::f32::consts::PI * (i as f32 + 0.5) / overlap as f32).sin();
    }
    
    let mut input = vec![0.0f32; n/2 + overlap];
    for i in 0..input.len() {
        input[i] = 1.0; // DC signal
    }
    
    let mut freq = vec![0.0f32; n / 2];
    mdct.forward(&input, &mut freq, &window, overlap, 0, 1);
    
    println!("Freq[0]: {}", freq[0]);
    
    let mut output = vec![0.0f32; n/2 + overlap];
    // Warm up decoder history with some zeros
    mdct.backward(&freq, &mut output, &window, overlap, 0, 1);
    
    println!("Output[core]: {:?}", &output[overlap..overlap+10]);
    
    let gain = output[overlap] / input[overlap];
    println!("Total Loopback Gain: {}", gain);
}
