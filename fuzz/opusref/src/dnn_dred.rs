//! FFI to the reference's dred code (feature `dnn`): `dnn_shim_dred.c`.
//! The C states stay opaque; accessor shims copy their fields out.
use core::ffi::{c_int, c_void};

unsafe extern "C" {
    pub static dred_latent_quant_scales_q8: [u8; 400];
    pub static dred_latent_dead_zone_q8: [u8; 400];
    pub static dred_latent_r_q8: [u8; 400];
    pub static dred_latent_p0_q8: [u8; 400];
    pub static dred_state_quant_scales_q8: [u8; 800];
    pub static dred_state_dead_zone_q8: [u8; 800];
    pub static dred_state_r_q8: [u8; 800];
    pub static dred_state_p0_q8: [u8; 800];

    pub fn ref_compute_quantizer(q0: c_int, dq: c_int, qmax: c_int, i: c_int) -> c_int;
    pub fn ref_laplace_p0_encode(
        vals: *const c_int,
        p0: *const u16,
        decay: *const u16,
        n: c_int,
        buf: *mut u8,
        cap: c_int,
    ) -> c_int;
    pub fn ref_laplace_p0_decode(
        vals: *mut c_int,
        p0: *const u16,
        decay: *const u16,
        n: c_int,
        buf: *const u8,
        len: c_int,
    ) -> c_int;

    pub fn ref_rdovae_enc_create() -> *mut c_void;
    pub fn ref_rdovae_enc_destroy(e: *mut c_void);
    pub fn ref_rdovae_encode_dframe(
        e: *mut c_void,
        latents: *mut f32,
        initial_state: *mut f32,
        input: *const f32,
    );
    pub fn ref_rdovae_enc_state(e: *const c_void, out: *mut f32, n: *mut c_int) -> c_int;

    pub fn ref_rdovae_dec_create() -> *mut c_void;
    pub fn ref_rdovae_dec_destroy(d: *mut c_void);
    pub fn ref_rdovae_dec_init_states(d: *mut c_void, initial_state: *const f32);
    pub fn ref_rdovae_decode_qframe(d: *mut c_void, qframe: *mut f32, z: *const f32);
    pub fn ref_rdovae_decode_all(
        d: *mut c_void,
        features: *mut f32,
        state: *const f32,
        latents: *const f32,
        nb: c_int,
    );
    pub fn ref_rdovae_dec_state(d: *const c_void, out: *mut f32, n: *mut c_int) -> c_int;

    pub fn ref_dred_encoder_create(fs: c_int, channels: c_int) -> *mut c_void;
    pub fn ref_dred_encoder_destroy(e: *mut c_void);
    pub fn ref_dred_encoder_reset(e: *mut c_void);
    pub fn ref_dred_compute_latents(
        e: *mut c_void,
        pcm: *const f32,
        frame_size: c_int,
        extra_delay: c_int,
    );
    pub fn ref_dred_encode_silk_frame(
        e: *mut c_void,
        buf: *mut u8,
        max_chunks: c_int,
        max_bytes: c_int,
        q0: c_int,
        dq: c_int,
        qmax: c_int,
        activity_mem: *mut u8,
    ) -> c_int;
    pub fn ref_dred_encoder_ints(e: *const c_void, out: *mut c_int);
    pub fn ref_dred_encoder_floats(e: *const c_void, which: c_int, n: *mut c_int) -> *const f32;
    pub fn ref_dred_encoder_rdovae_state(e: *const c_void, out: *mut f32, n: *mut c_int) -> c_int;
    pub fn ref_filter_df2t(
        input: *const f32,
        out: *mut f32,
        len: c_int,
        b0: f32,
        b: *const f32,
        a: *const f32,
        order: c_int,
        mem: *mut f32,
    );

    pub fn ref_dred_create() -> *mut c_void;
    pub fn ref_dred_destroy(d: *mut c_void);
    pub fn ref_dred_ec_decode(
        d: *mut c_void,
        bytes: *const u8,
        num_bytes: c_int,
        max_dred_features: c_int,
        dred_frame_offset: c_int,
    ) -> c_int;
    pub fn ref_dred_ints(d: *const c_void, out: *mut c_int);
    pub fn ref_dred_floats(d: *mut c_void, which: c_int, n: *mut c_int) -> *mut f32;
    pub fn ref_dred_process(d: *mut c_void, model: *const c_void);
}

/// The GRU and conv states of a C RDOVAE state (from `get`), in field
/// order, and its `initialized` flag.
fn state_of(get: impl FnOnce(*mut f32, *mut c_int) -> c_int) -> (Vec<f32>, bool) {
    let mut v = vec![0f32; 2048];
    let mut n = 0;
    let init = get(v.as_mut_ptr(), &mut n);
    v.truncate(n as usize);
    (v, init != 0)
}

/// A reference RDOVAE encoder (built-in model, zeroed state).
pub struct RdovaeEnc(*mut c_void);
impl RdovaeEnc {
    pub fn new() -> Self {
        let p = unsafe { ref_rdovae_enc_create() };
        assert!(!p.is_null());
        RdovaeEnc(p)
    }
    pub fn encode_dframe(
        &mut self,
        latents: &mut [f32; 25],
        state: &mut [f32; 50],
        input: &[f32; 40],
    ) {
        unsafe {
            ref_rdovae_encode_dframe(
                self.0,
                latents.as_mut_ptr(),
                state.as_mut_ptr(),
                input.as_ptr(),
            )
        }
    }
    pub fn state(&self) -> (Vec<f32>, bool) {
        state_of(|o, n| unsafe { ref_rdovae_enc_state(self.0, o, n) })
    }
}
impl Default for RdovaeEnc {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for RdovaeEnc {
    fn drop(&mut self) {
        unsafe { ref_rdovae_enc_destroy(self.0) }
    }
}

/// A reference RDOVAE decoder (built-in model, zeroed state).
pub struct RdovaeDec(*mut c_void);
impl RdovaeDec {
    pub fn new() -> Self {
        let p = unsafe { ref_rdovae_dec_create() };
        assert!(!p.is_null());
        RdovaeDec(p)
    }
    pub fn init_states(&mut self, initial_state: &[f32; 50]) {
        unsafe { ref_rdovae_dec_init_states(self.0, initial_state.as_ptr()) }
    }
    pub fn decode_qframe(&mut self, qframe: &mut [f32; 80], z: &[f32; 26]) {
        unsafe { ref_rdovae_decode_qframe(self.0, qframe.as_mut_ptr(), z.as_ptr()) }
    }
    /// `DRED_rdovae_decode_all`: `4*nb` feature frames.
    pub fn decode_all(&self, state: &[f32; 50], latents: &[f32], nb: usize) -> Vec<f32> {
        assert!(latents.len() >= 26 * nb);
        let mut f = vec![0f32; 80 * nb];
        unsafe {
            ref_rdovae_decode_all(
                self.0,
                f.as_mut_ptr(),
                state.as_ptr(),
                latents.as_ptr(),
                nb as c_int,
            )
        };
        f
    }
    pub fn state(&self) -> (Vec<f32>, bool) {
        state_of(|o, n| unsafe { ref_rdovae_dec_state(self.0, o, n) })
    }
}
impl Default for RdovaeDec {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for RdovaeDec {
    fn drop(&mut self) {
        unsafe { ref_rdovae_dec_destroy(self.0) }
    }
}

/// The fields of a C `DREDEnc`.
#[derive(Clone, Debug, PartialEq)]
pub struct DredEncFields {
    /// loaded, Fs, channels, input_buffer_fill, dred_offset, latent_offset,
    /// last_extra_dred_offset, latents_buffer_fill.
    pub ints: [i32; 8],
    pub input_buffer: Vec<f32>,
    pub latents_buffer: Vec<f32>,
    pub state_buffer: Vec<f32>,
    pub resample_mem: Vec<f32>,
    /// The RDOVAE encoder state as `RdovaeEnc::state`.
    pub rdovae: (Vec<f32>, bool),
}

/// A reference `DREDEnc` (built-in models).
pub struct DredEnc(*mut c_void);
impl DredEnc {
    pub fn new(fs: i32, channels: i32) -> Self {
        DredEnc(unsafe { ref_dred_encoder_create(fs, channels) })
    }
    pub fn reset(&mut self) {
        unsafe { ref_dred_encoder_reset(self.0) }
    }
    /// `pcm` must hold `2*frame_size` values for stereo (see
    /// `dred_compute_latents`).
    pub fn compute_latents(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        channels: usize,
        extra_delay: i32,
    ) {
        assert!(pcm.len() >= frame_size * channels);
        unsafe { ref_dred_compute_latents(self.0, pcm.as_ptr(), frame_size as c_int, extra_delay) }
    }
    /// `dred_encode_silk_frame`; `activity_mem` must cover every byte the C
    /// code reads (424).
    #[allow(clippy::too_many_arguments)]
    pub fn encode_silk_frame(
        &mut self,
        max_chunks: i32,
        max_bytes: i32,
        q0: i32,
        dq: i32,
        qmax: i32,
        activity_mem: &mut [u8],
    ) -> Vec<u8> {
        assert!(activity_mem.len() >= 424);
        let mut buf = vec![0u8; max_bytes.max(1) as usize];
        let n = unsafe {
            ref_dred_encode_silk_frame(
                self.0,
                buf.as_mut_ptr(),
                max_chunks,
                max_bytes,
                q0,
                dq,
                qmax,
                activity_mem.as_mut_ptr(),
            )
        };
        buf.truncate(n as usize);
        buf
    }
    pub fn fields(&self) -> DredEncFields {
        let mut ints = [0i32; 8];
        unsafe { ref_dred_encoder_ints(self.0, ints.as_mut_ptr()) };
        let f = |w| unsafe {
            let mut n = 0;
            let p = ref_dred_encoder_floats(self.0, w, &mut n);
            core::slice::from_raw_parts(p, n as usize).to_vec()
        };
        DredEncFields {
            ints,
            input_buffer: f(0),
            latents_buffer: f(1),
            state_buffer: f(2),
            resample_mem: f(3),
            rdovae: state_of(|o, n| unsafe { ref_dred_encoder_rdovae_state(self.0, o, n) }),
        }
    }
}
impl Drop for DredEnc {
    fn drop(&mut self) {
        unsafe { ref_dred_encoder_destroy(self.0) }
    }
}

/// The fields of a C `OpusDRED`.
#[derive(Clone, Debug, PartialEq)]
pub struct DredFields {
    pub fec_features: Vec<f32>,
    pub state: Vec<f32>,
    pub latents: Vec<f32>,
    pub nb_latents: i32,
    pub process_stage: i32,
    pub dred_offset: i32,
}

/// A reference `OpusDRED`, zeroed.
pub struct Dred(*mut c_void);
impl Dred {
    pub fn new() -> Self {
        Dred(unsafe { ref_dred_create() })
    }
    fn floats(&mut self, which: c_int) -> &mut [f32] {
        unsafe {
            let mut n = 0;
            let p = ref_dred_floats(self.0, which, &mut n);
            core::slice::from_raw_parts_mut(p, n as usize)
        }
    }
    pub fn latents_mut(&mut self) -> &mut [f32] {
        self.floats(2)
    }
    pub fn ec_decode(
        &mut self,
        bytes: &[u8],
        max_dred_features: i32,
        dred_frame_offset: i32,
    ) -> i32 {
        unsafe {
            ref_dred_ec_decode(
                self.0,
                bytes.as_ptr(),
                bytes.len() as c_int,
                max_dred_features,
                dred_frame_offset,
            )
        }
    }
    pub fn process(&mut self, model: &RdovaeDec) {
        unsafe { ref_dred_process(self.0, model.0) }
    }
    pub fn fields(&mut self) -> DredFields {
        let mut ints = [0i32; 3];
        unsafe { ref_dred_ints(self.0, ints.as_mut_ptr()) };
        DredFields {
            fec_features: self.floats(0).to_vec(),
            state: self.floats(1).to_vec(),
            latents: self.floats(2).to_vec(),
            nb_latents: ints[0],
            process_stage: ints[1],
            dred_offset: ints[2],
        }
    }
}
impl Default for Dred {
    fn default() -> Self {
        Self::new()
    }
}
impl Drop for Dred {
    fn drop(&mut self) {
        unsafe { ref_dred_destroy(self.0) }
    }
}

/// `ec_laplace_encode_p0` over `vals` (each with its `p0` and `decay`)
/// into `cap` bytes: the buffer after `ec_enc_done` and `ec_tell` before
/// it, or `None` on a coder error.
pub fn laplace_p0_encode(
    vals: &[i32],
    p0: &[u16],
    decay: &[u16],
    cap: usize,
) -> Option<(Vec<u8>, i32)> {
    let mut buf = vec![0u8; cap];
    let tell = unsafe {
        ref_laplace_p0_encode(
            vals.as_ptr(),
            p0.as_ptr(),
            decay.as_ptr(),
            vals.len() as c_int,
            buf.as_mut_ptr(),
            cap as c_int,
        )
    };
    (tell >= 0).then_some((buf, tell))
}

/// `ec_laplace_decode_p0` `p0.len()` times: the values and the final
/// `ec_tell`.
pub fn laplace_p0_decode(p0: &[u16], decay: &[u16], buf: &[u8]) -> (Vec<i32>, i32) {
    let mut vals = vec![0i32; p0.len()];
    let tell = unsafe {
        ref_laplace_p0_decode(
            vals.as_mut_ptr(),
            p0.as_ptr(),
            decay.as_ptr(),
            p0.len() as c_int,
            buf.as_ptr(),
            buf.len() as c_int,
        )
    };
    (vals, tell)
}
