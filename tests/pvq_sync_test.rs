use opus_rs::pvq::{icwrs, cwrsi};

#[test]
fn test_pvq_sync() {
    let n = 20u32;
    let k = 10u32; 
    let mut y = vec![0i32; n as usize];
    y[0] = 5;
    y[5] = -3;
    y[19] = 2;

    let index = icwrs(n, k, &y);
    println!("Vector: {:?}", y);
    println!("N={}, K={}, Index={}", n, k, index);

    let mut y2 = vec![0i32; n as usize];
    cwrsi(n, k, index, &mut y2);
    println!("Decoded: {:?}", y2);

    assert_eq!(y, y2, "PVQ Sync Failure!");
    println!("PVQ Sync Success for N=20, K=10");

    let k = 10u32;
    let n = 10u32;
    let mut y = vec![0i32; n as usize];
    y[0] = 5;
    y[5] = -3;
    y[9] = 2;
    let index = icwrs(n, k, &y);
    println!("N={}, K={}, Index (big K)={}", n, k, index);
    let mut y2 = vec![0i32; n as usize];
    cwrsi(n, k, index, &mut y2);
    assert_eq!(y, y2, "PVQ Sync Failure (big K)!");
    println!("PVQ Sync Success for N=10, K=10");
}

fn main() {
}
