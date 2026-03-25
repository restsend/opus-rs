use opus_rs::pvq::{cwrsi, icwrs};

#[test]
fn test_pvq_sign() {
    let n = 1;
    let k = 2;

    // Test Positive
    let y_pos = vec![2];
    let i_pos = icwrs(n, k, &y_pos);
    println!("Positive y=[2] -> i={}", i_pos);
    let mut y_out = vec![0; 1];
    cwrsi(n, k, i_pos, &mut y_out);
    println!("i={} -> y_out={:?}", i_pos, y_out);

    // Test Negative
    let y_neg = vec![-2];
    let i_neg = icwrs(n, k, &y_neg);
    println!("Negative y=[-2] -> i={}", i_neg);
    let mut y_out2 = vec![0; 1];
    cwrsi(n, k, i_neg, &mut y_out2);
    println!("i={} -> y_out={:?}", i_neg, y_out2);
}
