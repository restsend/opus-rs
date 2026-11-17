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
            ("dred_rdovae_enc", rdovaeenc_arrays.as_ptr(), builtin::DRED_RDOVAE_ENC),
            ("dred_rdovae_dec", rdovaedec_arrays.as_ptr(), builtin::DRED_RDOVAE_DEC),
            ("lace", lacelayers_arrays.as_ptr(), builtin::LACE),
            ("nolace", nolacelayers_arrays.as_ptr(), builtin::NOLACE),
            ("bbwenet", bbwenetlayers_arrays.as_ptr(), builtin::BBWENET),
        ]
    };
    let write = std::env::var_os("OPUSREF_WRITE_DNN_WEIGHTS").is_some();
    for (name, list, embedded) in models {
        let blob = blob_of(list);
        if write {
            let p = format!("{}/../../dnn-weights/{name}.bin", env!("CARGO_MANIFEST_DIR"));
            std::fs::write(p, &blob).unwrap();
        } else {
            assert!(blob == embedded, "{name}: embedded blob differs from the reference arrays");
        }
    }
}
