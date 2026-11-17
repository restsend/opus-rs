//! FFI to the neural network core of the reference (feature `dnn`): the
//! layers and the generic layer functions (`dnn_shim.c`).
use core::ffi::c_int;

/// `LinearLayer`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LinearLayer {
    pub bias: *const f32,
    pub subias: *const f32,
    pub weights: *const i8,
    pub float_weights: *const f32,
    pub weights_idx: *const c_int,
    pub diag: *const f32,
    pub scale: *const f32,
    pub nb_inputs: c_int,
    pub nb_outputs: c_int,
}

/// `Conv2dLayer`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Conv2dLayer {
    pub bias: *const f32,
    pub float_weights: *const f32,
    pub in_channels: c_int,
    pub out_channels: c_int,
    pub ktime: c_int,
    pub kheight: c_int,
}

impl Default for LinearLayer {
    fn default() -> Self {
        LinearLayer {
            bias: core::ptr::null(),
            subias: core::ptr::null(),
            weights: core::ptr::null(),
            float_weights: core::ptr::null(),
            weights_idx: core::ptr::null(),
            diag: core::ptr::null(),
            scale: core::ptr::null(),
            nb_inputs: 0,
            nb_outputs: 0,
        }
    }
}

unsafe extern "C" {
    pub fn ref_compute_linear(l: *const LinearLayer, out: *mut f32, input: *const f32);
    pub fn ref_compute_activation(out: *mut f32, input: *const f32, n: c_int, act: c_int);
    pub fn ref_compute_generic_dense(l: *const LinearLayer, out: *mut f32, input: *const f32, act: c_int);
    pub fn ref_compute_generic_gru(i: *const LinearLayer, r: *const LinearLayer, state: *mut f32, input: *const f32);
    pub fn ref_compute_glu(l: *const LinearLayer, out: *mut f32, input: *const f32);
    pub fn ref_compute_generic_conv1d(
        l: *const LinearLayer,
        out: *mut f32,
        mem: *mut f32,
        input: *const f32,
        n: c_int,
        act: c_int,
    );
    pub fn ref_compute_generic_conv1d_dilation(
        l: *const LinearLayer,
        out: *mut f32,
        mem: *mut f32,
        input: *const f32,
        n: c_int,
        dil: c_int,
        act: c_int,
    );
    pub fn ref_compute_conv2d(
        c: *const Conv2dLayer,
        out: *mut f32,
        mem: *mut f32,
        input: *const f32,
        h: c_int,
        hs: c_int,
        act: c_int,
    );
}

/// A small deterministic generator for test inputs (xorshift64*).
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
    }
    /// Uniform in [-1, 1).
    pub fn uni(&mut self) -> f32 {
        (self.next_u32() as f64 / 2147483648.0 - 1.0) as f32
    }
    pub fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
}

/// Bitwise float comparison: NaNs compare equal to NaNs.
pub fn same_bits(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()))
}
