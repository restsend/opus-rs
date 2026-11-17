//! The weight arrays the layers read: little-endian values borrowed from a
//! weight blob, so a blob needs no alignment and no copy.

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
