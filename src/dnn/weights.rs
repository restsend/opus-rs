//! Weight blobs and layer initialization (libopus `dnn/parse_lpcnet_weights.c`).
//!
//! A blob is a sequence of records: a 64-byte header (`WeightHead`: "DNNw",
//! version, type, size, block size, a NUL-terminated name of up to 43 bytes)
//! followed by `block_size` bytes, of which the first `size` are the array.
//! The layers borrow their arrays from the blob and read them with
//! little-endian loads, so a blob needs no alignment and no copy.

use super::nnet::{Conv2dLayer, LinearLayer};

/// Size of a record header and the alignment of the record data
/// (`WEIGHT_BLOCK_SIZE`).
pub const WEIGHT_BLOCK_SIZE: usize = 64;
/// Bytes of the record name, NUL terminator included.
const NAME_LEN: usize = 44;
/// Values of one block of a sparse layer (8 rows by 4 columns).
const SPARSE_BLOCK_SIZE: usize = 32;

/// `WEIGHT_TYPE_float`.
pub const WEIGHT_TYPE_FLOAT: i32 = 0;
/// `WEIGHT_TYPE_int`.
pub const WEIGHT_TYPE_INT: i32 = 1;
/// `WEIGHT_TYPE_qweight`.
pub const WEIGHT_TYPE_QWEIGHT: i32 = 2;
/// `WEIGHT_TYPE_int8`.
pub const WEIGHT_TYPE_INT8: i32 = 3;

/// A blob that `WeightList::parse` rejects, or a model whose arrays are
/// missing from a blob or have the wrong size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightError {
    /// A record header is malformed (libopus `parse_weights` returns -1).
    InvalidBlob,
    /// An array a layer needs is missing or has the wrong size (libopus
    /// `linear_init`/`conv2d_init` return 1). Holds the array name.
    Missing(&'static str),
}

/// Little-endian `f32` values borrowed from a blob.
#[derive(Clone, Copy, Debug, Default)]
pub struct F32s<'a>(&'a [[u8; 4]]);

impl<'a> F32s<'a> {
    /// The values of `bytes`, which must hold a whole number of `f32`s.
    pub fn new(bytes: &'a [u8]) -> Self {
        let (v, rest) = bytes.as_chunks::<4>();
        assert!(rest.is_empty(), "f32 array length is not a multiple of 4");
        F32s(v)
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Value `i`.
    #[inline(always)]
    pub fn at(&self, i: usize) -> f32 {
        f32::from_le_bytes(self.0[i])
    }
    /// Values `start..`.
    #[inline(always)]
    pub fn from(&self, start: usize) -> F32s<'a> {
        F32s(&self.0[start..])
    }
    /// Values `start..start + N`.
    #[inline(always)]
    pub fn array<const N: usize>(&self, start: usize) -> [f32; N] {
        let v: &[[u8; 4]; N] = self.0[start..start + N].try_into().unwrap();
        v.map(f32::from_le_bytes)
    }
}

/// Little-endian `i32` values borrowed from a blob.
#[derive(Clone, Copy, Debug, Default)]
pub struct I32s<'a>(&'a [[u8; 4]]);

impl<'a> I32s<'a> {
    /// The values of `bytes`, which must hold a whole number of `i32`s.
    pub fn new(bytes: &'a [u8]) -> Self {
        let (v, rest) = bytes.as_chunks::<4>();
        assert!(rest.is_empty(), "i32 array length is not a multiple of 4");
        I32s(v)
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    #[inline(always)]
    pub fn at(&self, i: usize) -> i32 {
        i32::from_le_bytes(self.0[i])
    }
}

/// One array of a blob (libopus `WeightArray`).
#[derive(Clone, Copy, Debug)]
pub struct WeightArray<'a> {
    /// The name, without its NUL terminator.
    pub name: &'a [u8],
    /// One of the `WEIGHT_TYPE_*` values.
    pub kind: i32,
    /// The `size` bytes of the array.
    pub data: &'a [u8],
}

/// A blob whose records all passed libopus's `parse_record` checks.
#[derive(Clone, Copy, Debug)]
pub struct WeightList<'a> {
    blob: &'a [u8],
}

fn header_int(h: &[u8], at: usize) -> i32 {
    i32::from_le_bytes([h[at], h[at + 1], h[at + 2], h[at + 3]])
}

/// libopus `parse_record`: the array at the front of `data` and the rest of
/// `data`, or `None` where libopus returns -1. libopus's `parse_weights`
/// also rejects a record with size 0 (it accepts only `ret > 0`).
fn parse_record(data: &[u8]) -> Option<(WeightArray<'_>, &[u8])> {
    if data.len() < WEIGHT_BLOCK_SIZE {
        return None;
    }
    let h = &data[..WEIGHT_BLOCK_SIZE];
    let kind = header_int(h, 8);
    let size = header_int(h, 12);
    let block_size = header_int(h, 16);
    let name = &h[20..20 + NAME_LEN];
    if block_size < size
        || block_size as i64 > (data.len() - WEIGHT_BLOCK_SIZE) as i64
        || name[NAME_LEN - 1] != 0
        || size <= 0
    {
        return None;
    }
    let name_len = name.iter().position(|&c| c == 0).unwrap_or(NAME_LEN);
    let body = &data[WEIGHT_BLOCK_SIZE..];
    Some((
        WeightArray {
            name: &name[..name_len],
            kind,
            data: &body[..size as usize],
        },
        &body[block_size as usize..],
    ))
}

impl<'a> WeightList<'a> {
    /// Checks every record of `blob` (libopus `parse_weights`).
    pub fn parse(blob: &'a [u8]) -> Result<Self, WeightError> {
        let mut rest = blob;
        while !rest.is_empty() {
            match parse_record(rest) {
                Some((_, r)) => rest = r,
                None => return Err(WeightError::InvalidBlob),
            }
        }
        Ok(WeightList { blob })
    }

    /// The arrays in blob order.
    pub fn iter(&self) -> impl Iterator<Item = WeightArray<'a>> + 'a {
        let mut rest = self.blob;
        core::iter::from_fn(move || {
            let (a, r) = parse_record(rest)?;
            rest = r;
            Some(a)
        })
    }

    /// The first array called `name` (libopus `find_array_entry`).
    pub fn find(&self, name: &str) -> Option<WeightArray<'a>> {
        self.iter().find(|a| a.name == name.as_bytes())
    }

    /// The data of array `name` if it has `size` bytes (`find_array_check`).
    fn find_check(&self, name: &'static str, size: usize) -> Result<&'a [u8], WeightError> {
        match self.find(name) {
            Some(a) if a.data.len() == size => Ok(a.data),
            _ => Err(WeightError::Missing(name)),
        }
    }

    /// `opt_array_check`: no array is fine, an array of the wrong size is
    /// an error.
    fn opt_check(&self, name: &'static str, size: usize) -> Result<Option<&'a [u8]>, WeightError> {
        match self.find(name) {
            None => Ok(None),
            Some(a) if a.data.len() == size => Ok(Some(a.data)),
            Some(_) => Err(WeightError::Missing(name)),
        }
    }

    /// `find_idx_check`: the index table of a sparse layer and its block
    /// count.
    fn find_idx_check(
        &self,
        name: &'static str,
        nb_in: usize,
        nb_out: usize,
    ) -> Result<(I32s<'a>, usize), WeightError> {
        let err = WeightError::Missing(name);
        let a = self.find(name).ok_or(err)?;
        let idx = I32s(a.data.as_chunks::<4>().0);
        let mut remain = idx.len();
        let mut p = 0;
        let mut nb_out = nb_out as i64;
        let mut total = 0usize;
        while remain > 0 {
            let nb_blocks = idx.at(p);
            p += 1;
            // Each group of 8 outputs lists at most one block per 4 inputs.
            if nb_blocks < 0 || remain < nb_blocks as usize + 1 || nb_blocks as usize > nb_in / 4 {
                return Err(err);
            }
            for _ in 0..nb_blocks {
                let pos = idx.at(p);
                p += 1;
                if pos < 0 || pos as i64 + 3 >= nb_in as i64 || pos & 3 != 0 {
                    return Err(err);
                }
            }
            nb_out -= 8;
            remain -= nb_blocks as usize + 1;
            total += nb_blocks as usize;
        }
        if nb_out != 0 {
            return Err(err);
        }
        Ok((idx, total))
    }

    /// libopus `linear_init`. The names are those of the model's
    /// `init_*` function; `None` stands for a NULL name.
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &self,
        bias: Option<&'static str>,
        subias: Option<&'static str>,
        weights: Option<&'static str>,
        float_weights: Option<&'static str>,
        weights_idx: Option<&'static str>,
        diag: Option<&'static str>,
        scale: Option<&'static str>,
        nb_inputs: usize,
        nb_outputs: usize,
    ) -> Result<LinearLayer<'a>, WeightError> {
        let mut l = LinearLayer {
            nb_inputs,
            nb_outputs,
            ..LinearLayer::default()
        };
        if let Some(n) = bias {
            l.bias = Some(F32s::new(self.find_check(n, 4 * nb_outputs)?));
        }
        if let Some(n) = subias {
            l.subias = Some(F32s::new(self.find_check(n, 4 * nb_outputs)?));
        }
        if let Some(n) = weights_idx {
            let (idx, total) = self.find_idx_check(n, nb_inputs, nb_outputs)?;
            l.weights_idx = Some(idx);
            if let Some(n) = weights {
                l.weights = Some(self.find_check(n, SPARSE_BLOCK_SIZE * total)?);
            }
            if let Some(n) = float_weights {
                l.float_weights = self
                    .opt_check(n, 4 * SPARSE_BLOCK_SIZE * total)?
                    .map(F32s::new);
            }
        } else {
            if let Some(n) = weights {
                l.weights = Some(self.find_check(n, nb_inputs * nb_outputs)?);
            }
            if let Some(n) = float_weights {
                l.float_weights = self
                    .opt_check(n, 4 * nb_inputs * nb_outputs)?
                    .map(F32s::new);
            }
        }
        if let Some(n) = diag {
            l.diag = Some(F32s::new(self.find_check(n, 4 * nb_outputs)?));
        }
        if weights.is_some() {
            // libopus looks up `scale` whenever `weights` is named; a NULL
            // scale name would crash its strcmp. The generated models
            // always name both.
            let n = scale.ok_or(WeightError::Missing("scale"))?;
            l.scale = Some(F32s::new(self.find_check(n, 4 * nb_outputs)?));
        }
        Ok(l)
    }

    /// libopus `conv2d_init`.
    pub fn conv2d(
        &self,
        bias: Option<&'static str>,
        float_weights: Option<&'static str>,
        in_channels: usize,
        out_channels: usize,
        ktime: usize,
        kheight: usize,
    ) -> Result<Conv2dLayer<'a>, WeightError> {
        let mut c = Conv2dLayer {
            in_channels,
            out_channels,
            ktime,
            kheight,
            ..Conv2dLayer::default()
        };
        if let Some(n) = bias {
            c.bias = Some(F32s::new(self.find_check(n, 4 * out_channels)?));
        }
        if let Some(n) = float_weights {
            let size = 4 * in_channels * out_channels * ktime * kheight;
            c.float_weights = self.opt_check(n, size)?.map(F32s::new);
        }
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(name: &str, kind: i32, data: &[u8]) -> [u8; WEIGHT_BLOCK_SIZE] {
        let mut h = [0u8; WEIGHT_BLOCK_SIZE];
        h[..4].copy_from_slice(b"DNNw");
        h[8..12].copy_from_slice(&kind.to_le_bytes());
        h[12..16].copy_from_slice(&(data.len() as i32).to_le_bytes());
        let block = data.len().div_ceil(WEIGHT_BLOCK_SIZE) * WEIGHT_BLOCK_SIZE;
        h[16..20].copy_from_slice(&(block as i32).to_le_bytes());
        h[20..20 + name.len()].copy_from_slice(name.as_bytes());
        h
    }

    #[test]
    fn parses_and_finds_records() {
        let mut blob = [0u8; 4 * WEIGHT_BLOCK_SIZE];
        let a = [1u8, 0, 128, 63, 0, 0, 0, 64];
        blob[..64].copy_from_slice(&record("a", WEIGHT_TYPE_FLOAT, &a));
        blob[64..72].copy_from_slice(&a);
        blob[128..192].copy_from_slice(&record("b", WEIGHT_TYPE_INT8, &[1, 2]));
        blob[192] = 7;
        let list = WeightList::parse(&blob).unwrap();
        assert_eq!(list.iter().count(), 2);
        let f = F32s::new(list.find("a").unwrap().data);
        assert_eq!(f.at(1), 2.0);
        assert_eq!(list.find("b").unwrap().data, &[7, 0]);
        assert!(list.find("c").is_none());
        assert_eq!(
            WeightList::parse(&blob[..100]).unwrap_err(),
            WeightError::InvalidBlob
        );
    }
}
