//! FFI to the reference's OSCE code (feature `dnn`): `dnn_shim_osce.c` and
//! the wrappers of the static functions of `osce.c` (`wrap/dnn/osce.c`).
//! The C states are opaque pointers that the shims allocate and
//! initialize; free them with [`ref_osce_free`].
use crate::dnn::LinearLayer;
use core::ffi::{c_char, c_int, c_void};

pub type OsceModel = c_void;

unsafe extern "C" {
    /// The built-in models (`osce_load_models(m, NULL, 0)`, `loaded` set).
    pub fn ref_osce_model_new() -> *mut OsceModel;
    pub fn ref_osce_model_set_loaded(m: *mut OsceModel, loaded: c_int);
    pub fn ref_osce_free(p: *mut c_void);
    /// A layer of the model by its field name, or null.
    pub fn ref_osce_layer(m: *const OsceModel, name: *const c_char) -> *const LinearLayer;
    /// The window of LACE (0), NoLACE (1) or BBWENet's AF1/AF2/AF3 (2/3/4).
    pub fn ref_osce_window(m: *const OsceModel, which: c_int) -> *const f32;

    pub fn ref_compute_overlap_window(w: *mut f32, n: c_int);
    pub fn ref_adaconv_state_new() -> *mut c_void;
    pub fn ref_adacomb_state_new() -> *mut c_void;
    pub fn ref_adashape_state_new() -> *mut c_void;
    pub fn ref_adaconv_process_frame(
        st: *mut c_void,
        out: *mut f32,
        input: *const f32,
        features: *const f32,
        kernel: *const LinearLayer,
        gain: *const LinearLayer,
        feature_dim: c_int,
        frame_size: c_int,
        overlap_size: c_int,
        in_channels: c_int,
        out_channels: c_int,
        kernel_size: c_int,
        left_padding: c_int,
        gain_a: f32,
        gain_b: f32,
        shape_gain: f32,
        window: *const f32,
    );
    pub fn ref_adacomb_process_frame(
        st: *mut c_void,
        out: *mut f32,
        input: *const f32,
        features: *const f32,
        kernel: *const LinearLayer,
        gain: *const LinearLayer,
        global_gain: *const LinearLayer,
        pitch_lag: c_int,
        feature_dim: c_int,
        frame_size: c_int,
        overlap_size: c_int,
        kernel_size: c_int,
        left_padding: c_int,
        gain_a: f32,
        gain_b: f32,
        log_gain_limit: f32,
        window: *const f32,
    );
    pub fn ref_adashape_process_frame(
        st: *mut c_void,
        out: *mut f32,
        input: *const f32,
        features: *const f32,
        alpha1f: *const LinearLayer,
        alpha1t: *const LinearLayer,
        alpha2: *const LinearLayer,
        feature_dim: c_int,
        frame_size: c_int,
        avg_pool_k: c_int,
        interpolate_k: c_int,
    );

    /// A `silk_decoder_state` after `silk_init_decoder`, with a
    /// `silk_decoder_control`.
    pub fn ref_osce_dec_new() -> *mut c_void;
    /// Sets the fields `osce_enhance_frame` reads. `pred_coef_q12` holds
    /// 2 x 16 values, `ltp_coef_q14` 20, `pitch_l` and `gains_q16` 4.
    pub fn ref_osce_dec_set(
        d: *mut c_void,
        fs_khz: c_int,
        nb_subfr: c_int,
        lpc_order: c_int,
        signal_type: c_int,
        pitch_l: *const c_int,
        gains_q16: *const i32,
        pred_coef_q12: *const i16,
        ltp_coef_q14: *const i16,
    );
    pub fn ref_osce_dec_features(
        d: *mut c_void,
        xq: *const i16,
        num_bits: c_int,
        features: *mut f32,
        numbits: *mut f32,
        periods: *mut c_int,
    );
    pub fn ref_osce_enhance_frame(m: *mut OsceModel, d: *mut c_void, xq: *mut i16, num_bits: c_int);
    pub fn ref_osce_reset(d: *mut c_void, method: c_int);
    pub fn ref_osce_method(d: *const c_void) -> c_int;
    pub fn ref_osce_reset_count(d: *const c_void) -> c_int;
    pub fn ref_osce_numbits_smooth(d: *const c_void) -> f32;

    pub fn ref_osce_bwe_new() -> *mut c_void;
    pub fn ref_osce_bwe_reset(st: *mut c_void);
    pub fn ref_osce_bwe(
        m: *mut OsceModel,
        st: *mut c_void,
        xq48: *mut i16,
        xq16: *mut i16,
        len: c_int,
    );
    pub fn ref_osce_bwe_features(st: *mut c_void, features: *mut f32, xq16: *const i16, len: c_int);
    pub fn ref_osce_cross_fade_10ms(x_enhanced: *mut f32, x_in: *mut f32, len: c_int);
    pub fn ref_osce_bwe_cross_fade_10ms(x_fadein: *mut i16, x_fadeout: *mut i16, len: c_int);

    // wrap/dnn/osce.c
    pub fn ref_lace_numbits_embedding(emb: *mut f32, numbits: f32, min_val: f32, max_val: f32);
    pub fn ref_lace_state_new() -> *mut c_void;
    pub fn ref_nolace_state_new() -> *mut c_void;
    pub fn ref_bbwenet_state_new() -> *mut c_void;
    pub fn ref_lace_process_20ms_frame(
        m: *mut OsceModel,
        st: *mut c_void,
        out: *mut f32,
        input: *const f32,
        features: *const f32,
        numbits: *const f32,
        periods: *const c_int,
    );
    pub fn ref_nolace_process_20ms_frame(
        m: *mut OsceModel,
        st: *mut c_void,
        out: *mut f32,
        input: *const f32,
        features: *const f32,
        numbits: *const f32,
        periods: *const c_int,
    );
    pub fn ref_bbwenet_process_frames(
        m: *mut OsceModel,
        st: *mut c_void,
        out: *mut f32,
        input: *const f32,
        features: *const f32,
        num_frames: c_int,
    );
    pub fn ref_resamp_state_new() -> *mut c_void;
    pub fn ref_upsamp_2x(st: *mut c_void, out: *mut f32, input: *const f32, n: c_int);
    pub fn ref_interpol_3_2(st: *mut c_void, out: *mut f32, input: *const f32, n: c_int);
    pub fn ref_apply_valin_activation(x: *mut f32, n: c_int);
}

/// [`ref_osce_layer`] with a Rust name; panics for an unknown name.
pub fn layer(m: *const OsceModel, name: &str) -> *const LinearLayer {
    let c = std::ffi::CString::new(name).unwrap();
    let p = unsafe { ref_osce_layer(m, c.as_ptr()) };
    assert!(!p.is_null(), "no layer {name}");
    p
}
