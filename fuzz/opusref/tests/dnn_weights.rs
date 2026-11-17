//! The embedded weight blobs of opus-rs (`dnn-weights/`) equal, byte for
//! byte, what libopus's record writer makes of the arrays compiled into the
//! reference. `OPUSREF_WRITE_DNN_WEIGHTS=1` writes them instead.
#![cfg(feature = "dnn")]
use opus_rs::dnn::builtin;
use opusref::dnn::*;

#[test]
fn embedded_blobs_equal_the_reference_arrays() {
    let models: [(&str, *const WeightArray, &[u8]); 8] = unsafe {
        [
            ("pitchdnn", pitchdnn_arrays.as_ptr(), builtin::PITCHDNN),
            ("fargan", fargan_arrays.as_ptr(), builtin::FARGAN),
            ("plc", plcmodel_arrays.as_ptr(), builtin::PLC),
            (
                "dred_rdovae_enc",
                rdovaeenc_arrays.as_ptr(),
                builtin::DRED_RDOVAE_ENC,
            ),
            (
                "dred_rdovae_dec",
                rdovaedec_arrays.as_ptr(),
                builtin::DRED_RDOVAE_DEC,
            ),
            ("lace", lacelayers_arrays.as_ptr(), builtin::LACE),
            ("nolace", nolacelayers_arrays.as_ptr(), builtin::NOLACE),
            ("bbwenet", bbwenetlayers_arrays.as_ptr(), builtin::BBWENET),
        ]
    };
    let write = std::env::var_os("OPUSREF_WRITE_DNN_WEIGHTS").is_some();
    for (name, list, embedded) in models {
        let blob = blob_of(list);
        if write {
            let p = format!(
                "{}/../../dnn-weights/{name}.bin",
                env!("CARGO_MANIFEST_DIR")
            );
            std::fs::write(p, &blob).unwrap();
        } else {
            assert!(
                blob == embedded,
                "{name}: embedded blob differs from the reference arrays"
            );
        }
    }
}

/// libopus's linear_init and opus-rs's WeightList::linear accept and reject
/// the same sparse index tables (8 inputs, 8 outputs: at most 2 blocks).
#[test]
fn sparse_index_checks_match() {
    use core::ffi::{c_char, c_void};
    use opus_rs::dnn::weights::WeightList;
    let null = core::ptr::null::<c_char>();
    for (idx, ok) in [
        (&[2, 0, 4][..], true),
        (&[2, 4, 4], true),
        (&[3, 0, 4, 4], false),
        (&[1, 6], false),
        (&[-1], false),
        (&[1, 0, 1, 4], false),
    ] {
        let blocks = idx[0].max(0) as usize;
        let idx_bytes: Vec<u8> = idx.iter().flat_map(|x: &i32| x.to_le_bytes()).collect();
        let w = vec![1u8; 32 * blocks.max(1)];
        let scale: Vec<u8> = [0.5f32; 8].iter().flat_map(|x| x.to_le_bytes()).collect();
        let arrays = [
            ("w\0", 3, &w[..]),
            ("idx\0", 1, &idx_bytes[..]),
            ("s\0", 0, &scale[..]),
        ]
        .map(|(name, kind, data)| WeightArray {
            name: name.as_ptr().cast(),
            kind,
            size: data.len() as i32,
            data: data.as_ptr().cast(),
        });
        let list = [
            arrays[0],
            arrays[1],
            arrays[2],
            WeightArray {
                name: core::ptr::null(),
                kind: 0,
                size: 0,
                data: core::ptr::null(),
            },
        ];
        let blob = blob_of(list.as_ptr());
        let mut parsed: *mut WeightArray = core::ptr::null_mut();
        let mut layer = LinearLayer::default();
        let c_ret = unsafe {
            assert_eq!(
                parse_weights(
                    &mut parsed,
                    blob.as_ptr().cast::<c_void>(),
                    blob.len() as i32
                ),
                3
            );
            linear_init(
                &mut layer,
                parsed,
                null,
                null,
                c"w".as_ptr(),
                null,
                c"idx".as_ptr(),
                null,
                c"s".as_ptr(),
                8,
                8,
            )
        };
        let rs_ret = WeightList::parse(&blob).unwrap().linear(
            None,
            None,
            Some("w"),
            None,
            Some("idx"),
            None,
            Some("s"),
            8,
            8,
        );
        assert_eq!((c_ret == 0, rs_ret.is_ok()), (ok, ok), "{idx:?}");
    }
}
