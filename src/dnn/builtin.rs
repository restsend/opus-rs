//! The weights of libopus 1.6.1, embedded in the crate.
//!
//! Each file in `dnn-weights/` is the blob that libopus's record writer
//! (`write_weights` of `dnn/write_lpcnet_weights.c`) produces for one
//! model's array list, compiled as libopus builds it by default
//! (DISABLE_DEBUG_FLOAT: int8 layers carry only their int8 weights). See
//! `dnn-weights/README.md` for the provenance and the SHA-256 of each file;
//! `fuzz/opusref/tests/dnn_weights.rs` checks them byte for byte against
//! the arrays compiled into the reference libopus.

use super::models;
use super::weights::{WeightError, WeightList};

/// A blob at a 64-byte boundary: its records then start at cache line
/// boundaries (`WEIGHT_BLOCK_SIZE`), as libopus's compiled arrays do, and each
/// 32-byte load of an int8 weight block stays within one cache line.
#[repr(C, align(64))]
struct Aligned<T: ?Sized>(T);

/// `include_bytes!` of `$path` at a 64-byte boundary.
macro_rules! blob {
    ($path:literal) => {{
        static A: &Aligned<[u8]> = &Aligned(*include_bytes!($path));
        &A.0
    }};
}

pub static PITCHDNN: &[u8] = blob!("../../dnn-weights/pitchdnn.bin");
pub static FARGAN: &[u8] = blob!("../../dnn-weights/fargan.bin");
pub static PLC: &[u8] = blob!("../../dnn-weights/plc.bin");
#[cfg(feature = "dred")]
pub static DRED_RDOVAE_ENC: &[u8] = blob!("../../dnn-weights/dred_rdovae_enc.bin");
#[cfg(feature = "dred")]
pub static DRED_RDOVAE_DEC: &[u8] = blob!("../../dnn-weights/dred_rdovae_dec.bin");
#[cfg(feature = "osce")]
pub static LACE: &[u8] = blob!("../../dnn-weights/lace.bin");
#[cfg(feature = "osce")]
pub static NOLACE: &[u8] = blob!("../../dnn-weights/nolace.bin");
#[cfg(feature = "osce")]
pub static BBWENET: &[u8] = blob!("../../dnn-weights/bbwenet.bin");

fn list(blob: &'static [u8]) -> Result<WeightList<'static>, WeightError> {
    WeightList::parse(blob)
}

/// The built-in PitchDNN model.
pub fn pitchdnn() -> Result<models::pitchdnn::PitchDnn<'static>, WeightError> {
    models::pitchdnn::PitchDnn::init(&list(PITCHDNN)?)
}
/// The built-in FARGAN model.
pub fn fargan() -> Result<models::fargan::Fargan<'static>, WeightError> {
    models::fargan::Fargan::init(&list(FARGAN)?)
}
/// The built-in PLC model.
pub fn plc() -> Result<models::plc::PlcModel<'static>, WeightError> {
    models::plc::PlcModel::init(&list(PLC)?)
}
/// The built-in RDOVAE encoder.
#[cfg(feature = "dred")]
pub fn dred_rdovae_enc() -> Result<models::dred_rdovae_enc::RdovaeEnc<'static>, WeightError> {
    models::dred_rdovae_enc::RdovaeEnc::init(&list(DRED_RDOVAE_ENC)?)
}
/// The built-in RDOVAE decoder.
#[cfg(feature = "dred")]
pub fn dred_rdovae_dec() -> Result<models::dred_rdovae_dec::RdovaeDec<'static>, WeightError> {
    models::dred_rdovae_dec::RdovaeDec::init(&list(DRED_RDOVAE_DEC)?)
}
/// The built-in LACE model.
#[cfg(feature = "osce")]
pub fn lace() -> Result<models::lace::LaceLayers<'static>, WeightError> {
    models::lace::LaceLayers::init(&list(LACE)?)
}
/// The built-in NoLACE model.
#[cfg(feature = "osce")]
pub fn nolace() -> Result<models::nolace::NolaceLayers<'static>, WeightError> {
    models::nolace::NolaceLayers::init(&list(NOLACE)?)
}
/// The built-in BBWENet model.
#[cfg(feature = "osce")]
pub fn bbwenet() -> Result<models::bbwenet::BbwenetLayers<'static>, WeightError> {
    models::bbwenet::BbwenetLayers::init(&list(BBWENET)?)
}

#[cfg(test)]
mod tests {
    /// SHA-256 (FIPS 180-4), for the integrity check of the blobs.
    fn sha256(data: &[u8]) -> [u8; 32] {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let bits = (data.len() as u64) * 8;
        let mut tail = [0u8; 128];
        let rest = &data[data.len() / 64 * 64..];
        tail[..rest.len()].copy_from_slice(rest);
        tail[rest.len()] = 0x80;
        let tail_len = if rest.len() < 56 { 64 } else { 128 };
        tail[tail_len - 8..tail_len].copy_from_slice(&bits.to_be_bytes());
        for block in data[..data.len() / 64 * 64]
            .chunks(64)
            .chain(tail[..tail_len].chunks(64))
        {
            let mut w = [0u32; 64];
            for i in 0..16 {
                w[i] = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap());
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let mut v = h;
            for i in 0..64 {
                let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
                let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
                let t1 = v[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
                let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
                let t2 = s0.wrapping_add(maj);
                v = [
                    t1.wrapping_add(t2),
                    v[0],
                    v[1],
                    v[2],
                    v[3].wrapping_add(t1),
                    v[4],
                    v[5],
                    v[6],
                ];
            }
            for (a, b) in h.iter_mut().zip(v) {
                *a = a.wrapping_add(b);
            }
        }
        let mut out = [0u8; 32];
        for (i, x) in h.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes());
        }
        out
    }

    #[test]
    fn sha256_known_answers() {
        let hex = |d: [u8; 32]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(
            hex(sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let m = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        assert_eq!(
            hex(sha256(m)),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn blobs_match_sha256sums() {
        let sums = include_str!("../../dnn-weights/SHA256SUMS");
        let mut blobs = vec![
            ("pitchdnn.bin", super::PITCHDNN),
            ("fargan.bin", super::FARGAN),
            ("plc.bin", super::PLC),
        ];
        #[cfg(feature = "dred")]
        blobs.extend([
            ("dred_rdovae_enc.bin", super::DRED_RDOVAE_ENC),
            ("dred_rdovae_dec.bin", super::DRED_RDOVAE_DEC),
        ]);
        #[cfg(feature = "osce")]
        blobs.extend([
            ("lace.bin", super::LACE),
            ("nolace.bin", super::NOLACE),
            ("bbwenet.bin", super::BBWENET),
        ]);
        assert_eq!(sums.lines().count(), 8);
        for (name, blob) in blobs {
            let line = sums.lines().find(|l| l.ends_with(name)).unwrap();
            let want = line.split_whitespace().next().unwrap();
            let got: String = sha256(blob).iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(got, want, "{name}");
        }
    }

    #[test]
    fn every_builtin_model_loads() {
        super::pitchdnn().unwrap();
        super::fargan().unwrap();
        super::plc().unwrap();
        #[cfg(feature = "dred")]
        {
            super::dred_rdovae_enc().unwrap();
            super::dred_rdovae_dec().unwrap();
        }
        #[cfg(feature = "osce")]
        {
            super::lace().unwrap();
            super::nolace().unwrap();
            super::bbwenet().unwrap();
        }
    }
}
