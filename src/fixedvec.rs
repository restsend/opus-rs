//! Minimal `Vec`-like growable buffer backed by **inline fixed-size storage**.
//!
//! `FixedVec<T, N>` stores its elements in a `[T; N]` with a tracked length — no
//! global allocator is required. Every slot always holds a valid `T`; slots past
//! the length hold [`ConstDefault::DEFAULT`] or a value left over from before a
//! `truncate`. It mirrors the subset of the `Vec` API used throughout opus-rs
//! (`resize`, `push`, `truncate`, `len`, indexing, and `Deref<Target = [T]>` over
//! the first `len` elements), so it can be dropped in as a heap-free replacement
//! field-for-field.
//!
//! This is what lets opus-rs be `#![no_std]` with **no `alloc`** at all.

use core::ops::{Deref, DerefMut};

/// A value of `Self` that is available in `const` contexts, used to fill the
/// unused slots of a [`FixedVec`] so that it can be built by a `const fn`.
pub trait ConstDefault: Sized {
    const DEFAULT: Self;
}

macro_rules! const_default_zero {
    ($($t:ty),*) => { $(impl ConstDefault for $t { const DEFAULT: Self = 0 as $t; })* };
}
const_default_zero!(u8, i16, i32, u32, f32);

impl<T: 'static> ConstDefault for &[T] {
    const DEFAULT: Self = &[];
}

impl<T> ConstDefault for Option<T> {
    const DEFAULT: Self = None;
}

#[derive(Debug, Clone)]
pub struct FixedVec<T, const N: usize> {
    len: usize,
    buf: [T; N],
}

impl<T: ConstDefault, const N: usize> FixedVec<T, N> {
    /// Create an empty buffer. `const`-constructible so it can initialize `static`s
    /// and struct fields in `const` contexts.
    pub const fn new() -> Self {
        Self {
            len: 0,
            buf: [const { T::DEFAULT }; N],
        }
    }

    /// Create a buffer of length `len` where every element is a clone of `value`.
    pub fn from_value(value: T, len: usize) -> Self
    where
        T: Clone,
    {
        let mut s = Self::new();
        s.resize(len, value);
        s
    }

    /// Copy a slice into a fresh buffer.
    pub fn from_slice(src: &[T]) -> Self
    where
        T: Clone,
    {
        let mut s = Self::new();
        s.extend_from_slice(src);
        s
    }
}

impl<T, const N: usize> FixedVec<T, N> {
    /// The compile-time capacity (number of elements that can be stored).
    pub const CAPACITY: usize = N;

    /// Wrap a full array of which the first `len` elements are in use.
    pub const fn from_array(buf: [T; N], len: usize) -> Self {
        assert!(len <= N, "FixedVec capacity exceeded");
        Self { len, buf }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn capacity(&self) -> usize {
        N
    }

    /// View the elements in use as a slice.
    pub fn as_slice(&self) -> &[T] {
        &self.buf[..self.len]
    }

    /// View the elements in use as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.buf[..self.len]
    }

    pub fn push(&mut self, value: T) {
        assert!(self.len < N, "FixedVec push past capacity");
        self.buf[self.len] = value;
        self.len += 1;
    }

    /// Resize the logical length. When growing, new slots are set to clones of
    /// `value`.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        assert!(new_len <= N, "FixedVec capacity exceeded");
        if new_len > self.len {
            self.buf[self.len..new_len].fill(value);
        }
        self.len = new_len;
    }

    /// Set the length to `len` with every element equal to `value`.
    pub fn fill_to(&mut self, len: usize, value: T)
    where
        T: Clone,
    {
        assert!(len <= N, "FixedVec capacity exceeded");
        self.buf[..len].fill(value);
        self.len = len;
    }

    pub fn truncate(&mut self, new_len: usize) {
        if new_len < self.len {
            self.len = new_len;
        }
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Append all elements from `src` (cloning).
    pub fn extend_from_slice(&mut self, src: &[T])
    where
        T: Clone,
    {
        let new_len = self.len + src.len();
        assert!(new_len <= N, "FixedVec capacity exceeded");
        self.buf[self.len..new_len].clone_from_slice(src);
        self.len = new_len;
    }
}

impl<T: ConstDefault, const N: usize> Default for FixedVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Deref for FixedVec<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, const N: usize> DerefMut for FixedVec<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}
