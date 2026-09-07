//! Minimal `Vec`-like growable buffer backed by **inline fixed-size storage**.
//!
//! `FixedVec<T, N>` stores its elements in a `[core::mem::MaybeUninit<T>; N]` with a
//! tracked length — no global allocator is required. It mirrors the subset of the
//! `Vec` API used throughout opus-rs (`resize`, `push`, `truncate`, `len`,
//! indexing, and `Deref<Target = [T]>` over the *initialized* range), so it can be
//! dropped in as a heap-free replacement field-for-field.
//!
//! This is what lets opus-rs be `#![no_std]` with **no `alloc`** at all.

use core::mem::MaybeUninit;
use core::ops::{Deref, DerefMut};

#[derive(Debug)]
pub struct FixedVec<T, const N: usize> {
    len: usize,
    buf: [MaybeUninit<T>; N],
}

impl<T, const N: usize> FixedVec<T, N> {
    /// The compile-time capacity (number of elements that can be stored).
    pub const CAPACITY: usize = N;

    /// Create an empty buffer. `const`-constructible so it can initialize `static`s
    /// and struct fields in `const` contexts.
    pub const fn new() -> Self {
        Self {
            len: 0,
            buf: [const { MaybeUninit::uninit() }; N],
        }
    }

    /// Create a buffer of length `len` where every element is a clone of `value`.
    pub fn from_value(value: T, len: usize) -> Self
    where
        T: Clone,
    {
        assert!(len <= N, "FixedVec capacity exceeded");
        let mut s = Self::new();
        for i in 0..len {
            s.buf[i].write(value.clone());
        }
        s.len = len;
        s
    }

    /// Copy a slice into a fresh buffer.
    pub fn from_slice(src: &[T]) -> Self
    where
        T: Clone,
    {
        assert!(src.len() <= N, "FixedVec capacity exceeded");
        let mut s = Self::new();
        for (i, v) in src.iter().cloned().enumerate() {
            s.buf[i].write(v);
        }
        s.len = src.len();
        s
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

    /// View the initialized elements as a slice.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `buf[..len]` holds initialized `T`s; `MaybeUninit<T>` has the
        // same layout as `T`, so the cast is sound for the initialized prefix.
        unsafe { core::slice::from_raw_parts(self.buf.as_ptr() as *const T, self.len) }
    }

    /// View the initialized elements as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.buf.as_mut_ptr() as *mut T, self.len) }
    }

    /// Raw pointer to the start of the (full) backing storage. Useful for code that
    /// writes into the buffer with explicit bounds tracking (e.g. the range coder).
    pub fn as_storage_ptr(&self) -> *const T {
        self.buf.as_ptr() as *const T
    }

    pub fn as_storage_mut_ptr(&mut self) -> *mut T {
        self.buf.as_mut_ptr() as *mut T
    }

    pub fn push(&mut self, value: T) {
        // `assert!` (not `debug_assert!`): in release builds a silent overflow
        // here would be out-of-bounds UB, e.g. `KissFftState::new(nfft)` with
        // nfft > KISS_MAX_N (issue #27 deep-scan).
        assert!(self.len < N, "FixedVec push past capacity");
        self.buf[self.len].write(value);
        self.len += 1;
    }

    /// Resize the logical length. When growing, new slots are filled with clones of
    /// `value`; when shrinking, dropped elements are properly dropped.
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        assert!(new_len <= N, "FixedVec capacity exceeded");
        if new_len > self.len {
            for i in self.len..new_len {
                self.buf[i].write(value.clone());
            }
        } else {
            for i in new_len..self.len {
                unsafe { self.buf[i].assume_init_drop() };
            }
        }
        self.len = new_len;
    }

    pub fn truncate(&mut self, new_len: usize) {
        if new_len < self.len {
            for i in new_len..self.len {
                unsafe { self.buf[i].assume_init_drop() };
            }
            self.len = new_len;
        }
    }

    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Set the logical length without initializing/deinitializing elements.
    ///
    /// Mirrors `Vec::set_len`. The caller is responsible for ensuring elements in
    /// `..new_len` are initialized (or treated as raw bytes). Used by the range
    /// coder which manages its own byte-level writes.
    ///
    /// # Safety
    /// After this call, `as_slice()` will expose `new_len` elements. The caller
    /// must guarantee that the `..new_len` range holds valid `T`s (or that they are
    /// only ever read as raw bytes / overwritten before being read as `T`).
    pub unsafe fn set_len(&mut self, new_len: usize) {
        assert!(new_len <= N, "FixedVec capacity exceeded");
        self.len = new_len;
    }

    /// Append all elements from `src` (cloning).
    pub fn extend_from_slice(&mut self, src: &[T])
    where
        T: Clone,
    {
        let new_len = self.len + src.len();
        assert!(new_len <= N, "FixedVec capacity exceeded");
        for v in src {
            self.buf[self.len].write(v.clone());
            self.len += 1;
        }
    }
}

impl<T, const N: usize> Default for FixedVec<T, N> {
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

impl<T: Clone, const N: usize> Clone for FixedVec<T, N> {
    fn clone(&self) -> Self {
        let mut s = Self::new();
        for i in 0..self.len {
            // SAFETY: `buf[..len]` is initialized.
            let v = unsafe { self.buf[i].assume_init_ref() }.clone();
            s.buf[i].write(v);
        }
        s.len = self.len;
        s
    }
}

impl<T, const N: usize> Drop for FixedVec<T, N> {
    fn drop(&mut self) {
        for i in 0..self.len {
            unsafe { self.buf[i].assume_init_drop() };
        }
    }
}

// `FixedVec` owns its storage inline and (once fully initialized) is safe to share.
unsafe impl<T: Sync, const N: usize> Sync for FixedVec<T, N> {}
unsafe impl<T: Send, const N: usize> Send for FixedVec<T, N> {}
