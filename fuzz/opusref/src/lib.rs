//! FFI to the reference libopus build (float, scalar C).
#![allow(non_camel_case_types, clippy::missing_safety_doc)]
use core::ffi::{c_int, c_uchar, c_void};

pub type OpusEncoder = c_void;
pub type OpusDecoder = c_void;
pub type CELTMode = c_void;

unsafe extern "C" {
    pub fn opus_encoder_create(fs: i32, ch: c_int, app: c_int, err: *mut c_int)
    -> *mut OpusEncoder;
    pub fn opus_encoder_destroy(st: *mut OpusEncoder);
    pub fn opus_encoder_ctl(st: *mut OpusEncoder, req: c_int, ...) -> c_int;
    pub fn opus_multistream_decoder_create(
        fs: i32,
        channels: c_int,
        streams: c_int,
        coupled: c_int,
        mapping: *const c_uchar,
        err: *mut c_int,
    ) -> *mut c_void;
    pub fn opus_multistream_decoder_destroy(st: *mut c_void);
    pub fn opus_multistream_decoder_ctl(st: *mut c_void, req: c_int, ...) -> c_int;
    pub fn opus_multistream_decode_float(
        st: *mut c_void,
        data: *const c_uchar,
        len: i32,
        pcm: *mut f32,
        frame_size: c_int,
        fec: c_int,
    ) -> c_int;
    pub fn opus_multistream_encoder_create(
        fs: i32,
        channels: c_int,
        streams: c_int,
        coupled: c_int,
        mapping: *const c_uchar,
        application: c_int,
        err: *mut c_int,
    ) -> *mut c_void;
    pub fn opus_multistream_encoder_destroy(st: *mut c_void);
    pub fn opus_multistream_encoder_ctl(st: *mut c_void, req: c_int, ...) -> c_int;
    pub fn opus_multistream_encode_float(
        st: *mut c_void,
        pcm: *const f32,
        frame_size: c_int,
        data: *mut c_uchar,
        max: i32,
    ) -> i32;
    pub fn opus_encode(
        st: *mut OpusEncoder,
        pcm: *const i16,
        fs: c_int,
        data: *mut c_uchar,
        max: i32,
    ) -> i32;
    pub fn opus_encode_float(
        st: *mut OpusEncoder,
        pcm: *const f32,
        fs: c_int,
        data: *mut c_uchar,
        max: i32,
    ) -> i32;
    pub fn opus_decoder_create(fs: i32, ch: c_int, err: *mut c_int) -> *mut OpusDecoder;
    pub fn opus_decoder_destroy(st: *mut OpusDecoder);
    pub fn opus_decoder_ctl(st: *mut OpusDecoder, req: c_int, ...) -> c_int;
    pub fn opus_decode_float(
        st: *mut OpusDecoder,
        data: *const c_uchar,
        len: i32,
        pcm: *mut f32,
        fs: c_int,
        fec: c_int,
    ) -> c_int;
    pub fn opus_decode(
        st: *mut OpusDecoder,
        data: *const c_uchar,
        len: i32,
        pcm: *mut i16,
        fs: c_int,
        fec: c_int,
    ) -> c_int;

    pub fn opus_strerror(error: c_int) -> *const core::ffi::c_char;
    pub fn opus_get_version_string() -> *const core::ffi::c_char;
    pub fn opus_packet_parse(
        data: *const c_uchar,
        len: i32,
        toc: *mut c_uchar,
        frames: *mut *const c_uchar,
        size: *mut i16,
        payload_offset: *mut c_int,
    ) -> c_int;

    // Internal CELT/SILK functions (non-static in libopus).
    pub fn ref_celt_mode48() -> *const CELTMode;
    pub fn ref_celt_encoder_create(fs: i32, channels: c_int) -> *mut c_void;
    pub fn ref_celt_encoder_destroy(st: *mut c_void);
    pub fn ref_celt_encoder_ctl(st: *mut c_void, request: c_int, value: c_int) -> c_int;
    pub fn ref_celt_encode(
        st: *mut c_void,
        pcm: *const f32,
        frame_size: c_int,
        out: *mut c_uchar,
        bytes: c_int,
    ) -> c_int;
    pub fn ref_celt_encode_after(
        st: *mut c_void,
        pcm: *const f32,
        frame_size: c_int,
        out: *mut c_uchar,
        bytes: c_int,
        prefix: c_int,
    ) -> c_int;
    pub fn comb_filter(
        y: *mut f32,
        x: *mut f32,
        t0: c_int,
        t1: c_int,
        n: c_int,
        g0: f32,
        g1: f32,
        tap0: c_int,
        tap1: c_int,
        window: *const f32,
        overlap: c_int,
        arch: c_int,
    );
    pub fn haar1(x: *mut f32, n0: c_int, stride: c_int);
    pub fn compute_band_energies(
        m: *const CELTMode,
        x: *const f32,
        band_e: *mut f32,
        end: c_int,
        c: c_int,
        lm: c_int,
        arch: c_int,
    );
    pub fn pitch_search(
        x_lp: *const f32,
        y: *mut f32,
        len: c_int,
        max_pitch: c_int,
        pitch: *mut c_int,
        arch: c_int,
    );
    pub fn remove_doubling(
        x: *mut f32,
        maxperiod: c_int,
        minperiod: c_int,
        n: c_int,
        t0: *mut c_int,
        prev_period: c_int,
        prev_gain: f32,
        arch: c_int,
    ) -> f32;
    pub fn exp_rotation(
        x: *mut f32,
        len: c_int,
        dir: c_int,
        stride: c_int,
        k: c_int,
        spread: c_int,
    );
    pub fn op_pvq_search_c(x: *mut f32, iy: *mut c_int, k: c_int, n: c_int, arch: c_int) -> f32;
    pub fn celt_fir_c(
        x: *const f32,
        num: *const f32,
        y: *mut f32,
        n: c_int,
        ord: c_int,
        arch: c_int,
    );
    pub fn celt_iir(
        x: *const f32,
        den: *const f32,
        y: *mut f32,
        n: c_int,
        ord: c_int,
        mem: *mut f32,
        arch: c_int,
    );
    pub fn _celt_lpc(lpc: *mut f32, ac: *const f32, p: c_int);
    pub fn renormalise_vector(x: *mut f32, n: c_int, gain: f32, arch: c_int);
    pub fn stereo_itheta(
        x: *const f32,
        y: *const f32,
        stereo: c_int,
        n: c_int,
        arch: c_int,
    ) -> c_int;
    pub fn amp2Log2(
        m: *const CELTMode,
        eff_end: c_int,
        end: c_int,
        band_e: *const f32,
        band_log_e: *mut f32,
        c: c_int,
    );
    pub fn _celt_autocorr(
        x: *const f32,
        ac: *mut f32,
        window: *const f32,
        overlap: c_int,
        lag: c_int,
        n: c_int,
        arch: c_int,
    ) -> c_int;
    pub fn silk_sum_sqr_shift(energy: *mut i32, shift: *mut c_int, x: *const i16, len: c_int);
    pub fn silk_inner_prod_aligned_scale(
        a: *const i16,
        b: *const i16,
        scale: c_int,
        len: c_int,
    ) -> i32;
    pub fn silk_lin2log(x: i32) -> i32;
    pub fn silk_log2lin(x: i32) -> i32;
}

unsafe extern "C" {
    fn ofz_trace_start();
    fn ofz_trace_stop() -> c_int;
    fn ofz_trace_get(
        i: c_int,
        file: *mut *const core::ffi::c_char,
        line: *mut c_int,
        val: *mut c_int,
    );
}

/// One range-decoder call made by the reference decoder: the libopus source
/// location of the call and the decoded value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    pub file: &'static str,
    pub line: i32,
    pub val: i32,
}

unsafe extern "C" {
    fn ofz_take_silk_input(p: *mut *const i16) -> c_int;
}

/// The 16-bit samples the reference encoder handed to SILK in its last coded
/// SILK call, interleaved; empty if SILK did not run since the last call.
pub fn take_silk_input() -> Vec<i16> {
    let mut p = core::ptr::null();
    let n = unsafe { ofz_take_silk_input(&mut p) };
    unsafe { core::slice::from_raw_parts(p, n as usize) }.to_vec()
}

/// Runs `f` and returns the symbols the reference decoder read meanwhile.
pub fn trace<T>(f: impl FnOnce() -> T) -> (T, Vec<Symbol>) {
    unsafe {
        ofz_trace_start();
        let r = f();
        let n = ofz_trace_stop();
        let syms = (0..n)
            .map(|i| {
                let (mut file, mut line, mut val) = (core::ptr::null(), 0, 0);
                ofz_trace_get(i, &mut file, &mut line, &mut val);
                let file = core::ffi::CStr::from_ptr(file).to_str().unwrap_or("?");
                let file = file.rsplit('/').next().unwrap_or(file);
                Symbol { file, line, val }
            })
            .collect();
        (r, syms)
    }
}

macro_rules! w2 { ($($n:ident)*) => { unsafe extern "C" { $(pub fn $n(a: i32, b: i32) -> i32;)* } } }
macro_rules! w3 { ($($n:ident)*) => { unsafe extern "C" { $(pub fn $n(a: i32, b: i32, c: i32) -> i32;)* } } }
macro_rules! w1 { ($($n:ident)*) => { unsafe extern "C" { $(pub fn $n(a: i32) -> i32;)* } } }
w2!(ref_silk_lshift_sat32 ref_silk_add_sat32 ref_silk_sub_sat32 ref_silk_add_pos_sat32 ref_silk_rshift_round
    ref_silk_smulwb ref_silk_smulwt ref_silk_smulww ref_silk_smulbb ref_silk_smmul ref_silk_div32_16
    ref_silk_inverse32_varq);
w3!(ref_silk_smlawb ref_silk_smlawt ref_silk_smlaww ref_silk_smlabb ref_silk_limit_32 ref_silk_add_lshift32
    ref_silk_add_rshift32 ref_silk_sub_lshift32 ref_silk_div32_varq);
w1!(ref_silk_sat16 ref_silk_clz32 ref_silk_sqrt_approx ref_silk_rand);

// CTL request numbers (opus_defines.h, opus_private.h).
pub const SET_APPLICATION: c_int = 4000;
pub const GET_APPLICATION: c_int = 4001;
pub const SET_BITRATE: c_int = 4002;
pub const GET_BITRATE: c_int = 4003;
pub const SET_MAX_BANDWIDTH: c_int = 4004;
pub const GET_MAX_BANDWIDTH: c_int = 4005;
pub const SET_VBR: c_int = 4006;
pub const GET_VBR: c_int = 4007;
pub const SET_BANDWIDTH: c_int = 4008;
pub const GET_BANDWIDTH: c_int = 4009;
pub const SET_COMPLEXITY: c_int = 4010;
pub const GET_COMPLEXITY: c_int = 4011;
pub const SET_INBAND_FEC: c_int = 4012;
pub const GET_INBAND_FEC: c_int = 4013;
pub const SET_PACKET_LOSS_PERC: c_int = 4014;
pub const GET_PACKET_LOSS_PERC: c_int = 4015;
pub const SET_DTX: c_int = 4016;
pub const GET_DTX: c_int = 4017;
pub const SET_VBR_CONSTRAINT: c_int = 4020;
pub const GET_VBR_CONSTRAINT: c_int = 4021;
pub const SET_FORCE_CHANNELS: c_int = 4022;
pub const GET_FORCE_CHANNELS: c_int = 4023;
pub const SET_SIGNAL: c_int = 4024;
pub const GET_SIGNAL: c_int = 4025;
pub const GET_LOOKAHEAD: c_int = 4027;
pub const GET_SAMPLE_RATE: c_int = 4029;
pub const GET_FINAL_RANGE: c_int = 4031;
pub const GET_PITCH: c_int = 4033;
pub const SET_GAIN: c_int = 4034;
pub const GET_GAIN: c_int = 4045;
pub const SET_LSB_DEPTH: c_int = 4036;
pub const GET_LSB_DEPTH: c_int = 4037;
pub const GET_LAST_PACKET_DURATION: c_int = 4039;
pub const SET_EXPERT_FRAME_DURATION: c_int = 4040;
pub const GET_EXPERT_FRAME_DURATION: c_int = 4041;
pub const SET_PREDICTION_DISABLED: c_int = 4042;
pub const GET_PREDICTION_DISABLED: c_int = 4043;
pub const SET_PHASE_INVERSION_DISABLED: c_int = 4046;
pub const GET_PHASE_INVERSION_DISABLED: c_int = 4047;
pub const GET_IN_DTX: c_int = 4049;
pub const SET_DRED_DURATION: c_int = 4050;
pub const GET_DRED_DURATION: c_int = 4051;
pub const SET_DNN_BLOB: c_int = 4052;
pub const GET_DNN_BLOB: c_int = 4053;
pub const SET_OSCE_BWE: c_int = 4054;
pub const GET_OSCE_BWE: c_int = 4055;
pub const SET_QEXT: c_int = 4056;
pub const GET_QEXT: c_int = 4057;
pub const SET_IGNORE_EXTENSIONS: c_int = 4058;
pub const GET_IGNORE_EXTENSIONS: c_int = 4059;
pub const RESET_STATE: c_int = 4028;
pub const SET_FORCE_MODE: c_int = 11002;

unsafe extern "C" {
    /// `clt_mdct_forward_c`/`clt_mdct_backward_c` of the static 48 kHz mode,
    /// with its window and overlap.
    pub fn ref_mdct_forward(input: *mut f32, out: *mut f32, shift: c_int, stride: c_int);
    pub fn ref_mdct_backward(input: *mut f32, out: *mut f32, shift: c_int, stride: c_int);
    /// `opus_fft_impl` of the static mode's FFT at `shift` on interleaved
    /// (re, im) pairs, already in bit-reversed order.
    pub fn ref_fft_impl(x: *mut f32, shift: c_int);
    /// The static mode's MDCT twiddles (all shifts, 1800 values) and its
    /// 120-sample window.
    pub fn ref_mdct_trig() -> *const f32;
    pub fn ref_window() -> *const f32;
}

/// The neural network functions of the reference (feature `dnn`).
#[cfg(feature = "dnn")]
pub mod dnn;
/// DRED of the reference.
#[cfg(feature = "dnn")]
pub mod dnn_dred;
/// The DRED decoder API, DRED encoder control and packet extensions of
/// the reference.
#[cfg(feature = "dnn")]
pub mod dnn_dred_api;
/// OSCE of the reference.
#[cfg(feature = "dnn")]
pub mod dnn_osce;
/// Deep PLC (FARGAN, LPCNet PLC) of the reference.
#[cfg(feature = "dnn")]
pub mod dnn_plc;
