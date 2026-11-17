//! FFI to the reference's plc code (feature `dnn`).
use core::ffi::c_void;

unsafe extern "C" {
    pub fn ref_fargan_create() -> *mut c_void;
    pub fn ref_fargan_destroy(st: *mut c_void);
    pub fn ref_fargan_cont(st: *mut c_void, pcm0: *const f32, features0: *const f32);
    pub fn ref_fargan_synthesize(st: *mut c_void, pcm: *mut f32, features: *const f32);
    pub fn ref_fargan_synthesize_int(st: *mut c_void, pcm: *mut i16, features: *const f32);

    pub fn ref_lpcnet_plc_create() -> *mut c_void;
    pub fn ref_lpcnet_plc_destroy(st: *mut c_void);
    pub fn ref_lpcnet_plc_reset(st: *mut c_void);
    pub fn ref_lpcnet_plc_update(st: *mut c_void, pcm: *mut i16);
    pub fn ref_lpcnet_plc_conceal(st: *mut c_void, pcm: *mut i16);
    /// `features` may be null (a frame DRED does not cover).
    pub fn ref_lpcnet_plc_fec_add(st: *mut c_void, features: *const f32);
    pub fn ref_lpcnet_plc_fec_clear(st: *mut c_void);
}
