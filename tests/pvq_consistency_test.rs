use opus_rs::pvq::{cwrsi, icwrs};

#[test]
fn test_pvq_consistency() {
    let n = 8u32;
    let k = 4u32;
    let y_in = vec![1, -1, 0, 2, 0, 0, 0, 0];

    let i = icwrs(n, k, &y_in);
    println!("i = {}", i);

    let mut y_out = vec![0i32; n as usize];
    cwrsi(n, k, i, &mut y_out);

    println!("y_in  = {:?}", y_in);
    println!("y_out = {:?}", y_out);

    assert_eq!(y_in, y_out);
}
