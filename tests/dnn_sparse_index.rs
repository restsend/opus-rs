//! libopus's `find_idx_check` accepts sparse index tables with negative
//! block counts or positions, which its sparse kernels then use as
//! offsets; opus-rs rejects such tables when it loads a layer.
#![cfg(feature = "dnn")]

use opus_rs::dnn::weights::{WEIGHT_BLOCK_SIZE, WEIGHT_TYPE_INT, WeightError, WeightList};

/// A blob holding one int32 array "idx" with `values`.
fn blob(values: &[i32]) -> Vec<u8> {
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let block = data.len().div_ceil(WEIGHT_BLOCK_SIZE) * WEIGHT_BLOCK_SIZE;
    let mut b = vec![0u8; WEIGHT_BLOCK_SIZE + block];
    b[..4].copy_from_slice(b"DNNw");
    b[8..12].copy_from_slice(&WEIGHT_TYPE_INT.to_le_bytes());
    b[12..16].copy_from_slice(&(data.len() as i32).to_le_bytes());
    b[16..20].copy_from_slice(&(block as i32).to_le_bytes());
    b[20..23].copy_from_slice(b"idx");
    b[WEIGHT_BLOCK_SIZE..WEIGHT_BLOCK_SIZE + data.len()].copy_from_slice(&data);
    b
}

/// A sparse layer of 16 inputs and 8 outputs with index table `values`.
fn load(values: &[i32]) -> Result<(), WeightError> {
    let b = blob(values);
    let list = WeightList::parse(&b).unwrap();
    list.linear(None, None, None, None, Some("idx"), None, None, 16, 8)
        .map(|_| ())
}

#[test]
fn rejects_negative_sparse_block_counts_and_positions() {
    assert_eq!(load(&[1, 4]), Ok(()));
    assert_eq!(load(&[-1]), Err(WeightError::Missing("idx")));
    assert_eq!(load(&[1, -4]), Err(WeightError::Missing("idx")));
}
