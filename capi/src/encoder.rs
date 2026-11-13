//! `opus_encoder_*` / `opus_encode*`, after libopus's opus_encoder.c.

use core::ffi::c_int;
use core::mem::size_of;
use core::ptr;
use core::slice;

use opus_rs::{Application, Bandwidth, Signal};

use crate::decoder::valid_fs;
use crate::{
    OPUS_ALLOC_FAIL, OPUS_APPLICATION_AUDIO, OPUS_APPLICATION_RESTRICTED_CELT,
    OPUS_APPLICATION_RESTRICTED_LOWDELAY, OPUS_APPLICATION_RESTRICTED_SILK, OPUS_APPLICATION_VOIP,
    OPUS_AUTO, OPUS_BAD_ARG, OPUS_BANDWIDTH_FULLBAND, OPUS_BANDWIDTH_NARROWBAND, OPUS_BITRATE_MAX,
    OPUS_BUFFER_TOO_SMALL, OPUS_FRAMESIZE_2_5_MS, OPUS_FRAMESIZE_40_MS, OPUS_FRAMESIZE_120_MS,
    OPUS_FRAMESIZE_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, OPUS_SIGNAL_MUSIC, OPUS_SIGNAL_VOICE,
    OPUS_UNIMPLEMENTED, CtlArgs, guard, req, set_out,
};

/// libopus's `OpusEncoder`: the C API's settings, applied to the opus-rs
/// encoder (which holds all the codec state inline) before each frame.
///
/// Settings opus-rs has no counterpart for are kept, validated and reported
/// back like libopus's, but do not change the coding: force_channels,
/// lsb_depth, prediction_disabled, phase_inversion_disabled, voice_ratio,
/// force_mode, lfe.
#[repr(C)]
pub struct OpusEncoder {
    fs: i32,
    channels: i32,
    application: i32,
    user_bitrate: i32,
    use_vbr: i32,
    vbr_constraint: i32,
    complexity: i32,
    fec_config: i32,
    packet_loss_perc: i32,
    use_dtx: i32,
    signal: i32,
    user_bandwidth: i32,
    max_bandwidth: i32,
    force_channels: i32,
    lsb_depth: i32,
    variable_duration: i32,
    prediction_disabled: i32,
    phase_inversion_disabled: i32,
    voice_ratio: i32,
    force_mode: i32,
    lfe: i32,
    prev_framesize: i32,
    /// No frame coded since init or reset (libopus `first`).
    first: i32,
    inner: opus_rs::OpusEncoder,
}

fn rs_application(application: c_int) -> Option<Application> {
    match application {
        OPUS_APPLICATION_VOIP => Some(Application::Voip),
        OPUS_APPLICATION_AUDIO => Some(Application::Audio),
        // RESTRICTED_CELT is CELT-only with the low-delay lookahead, as
        // RESTRICTED_LOWDELAY codes.
        OPUS_APPLICATION_RESTRICTED_LOWDELAY | OPUS_APPLICATION_RESTRICTED_CELT => {
            Some(Application::RestrictedLowDelay)
        }
        _ => None,
    }
}

fn rs_bandwidth(bw: i32) -> Bandwidth {
    match bw {
        1101 => Bandwidth::Narrowband,
        1102 => Bandwidth::Mediumband,
        1103 => Bandwidth::Wideband,
        1104 => Bandwidth::Superwideband,
        1105 => Bandwidth::Fullband,
        _ => Bandwidth::Auto,
    }
}

fn bits_to_bitrate(bits: i32, fs: i32, frame_size: i32) -> i32 {
    bits * (6 * fs / frame_size) / 6
}

/// libopus `frame_size_select`: the frame `OPUS_SET_EXPERT_FRAME_DURATION`
/// picks out of the `frame_size` samples given, or -1.
fn frame_size_select(application: i32, frame_size: i32, variable_duration: i32, fs: i32) -> i32 {
    if frame_size < fs / 400 {
        return -1;
    }
    let new_size = if variable_duration == OPUS_FRAMESIZE_ARG {
        frame_size
    } else if (OPUS_FRAMESIZE_2_5_MS..=OPUS_FRAMESIZE_120_MS).contains(&variable_duration) {
        if variable_duration <= OPUS_FRAMESIZE_40_MS {
            (fs / 400) << (variable_duration - OPUS_FRAMESIZE_2_5_MS)
        } else {
            (variable_duration - OPUS_FRAMESIZE_2_5_MS - 2) * fs / 50
        }
    } else {
        return -1;
    };
    if new_size > frame_size {
        return -1;
    }
    let n = new_size;
    if 400 * n != fs
        && 200 * n != fs
        && 100 * n != fs
        && 50 * n != fs
        && 25 * n != fs
        && 50 * n != 3 * fs
        && 50 * n != 4 * fs
        && 50 * n != 5 * fs
        && 50 * n != 6 * fs
    {
        return -1;
    }
    if application == OPUS_APPLICATION_RESTRICTED_SILK && n < fs / 100 {
        return -1;
    }
    n
}

enum Pcm<'a> {
    I16(&'a [i16]),
    F32(&'a [f32]),
}

impl OpusEncoder {
    unsafe fn init(st: *mut Self, fs: i32, channels: i32, application: c_int) -> c_int {
        if !valid_fs(fs) || !(1..=2).contains(&channels) {
            return OPUS_BAD_ARG;
        }
        if application == OPUS_APPLICATION_RESTRICTED_SILK {
            return OPUS_UNIMPLEMENTED;
        }
        let Some(app) = rs_application(application) else {
            return OPUS_BAD_ARG;
        };
        let inner = match opus_rs::OpusEncoder::new(fs, channels as usize, app) {
            Ok(e) => e,
            Err(_) => return OPUS_INTERNAL_ERROR,
        };
        unsafe {
            st.write(Self {
                fs,
                channels,
                application,
                user_bitrate: OPUS_AUTO,
                use_vbr: 1,
                vbr_constraint: 1,
                complexity: 9,
                fec_config: 0,
                packet_loss_perc: 0,
                use_dtx: 0,
                signal: OPUS_AUTO,
                user_bandwidth: OPUS_AUTO,
                max_bandwidth: OPUS_BANDWIDTH_FULLBAND,
                force_channels: OPUS_AUTO,
                lsb_depth: 24,
                variable_duration: OPUS_FRAMESIZE_ARG,
                prediction_disabled: 0,
                phase_inversion_disabled: 0,
                voice_ratio: -1,
                force_mode: OPUS_AUTO,
                lfe: 0,
                prev_framesize: 0,
                first: 1,
                inner,
            });
        }
        OPUS_OK
    }

    /// Restarts the codec state, keeping the settings (`OPUS_RESET_STATE`).
    fn reset(&mut self) {
        let app = rs_application(self.application).unwrap();
        if let Ok(inner) = opus_rs::OpusEncoder::new(self.fs, self.channels as usize, app) {
            self.inner = inner;
        }
        self.first = 1;
    }

    /// libopus `user_bitrate_to_bitrate`.
    fn bitrate(&self, frame_size: i32, max_data_bytes: i32) -> i32 {
        let frame_size = if frame_size == 0 { self.fs / 400 } else { frame_size };
        let max_bitrate = bits_to_bitrate(max_data_bytes * 8, self.fs, frame_size);
        let user = match self.user_bitrate {
            OPUS_AUTO => 60 * self.fs / frame_size + self.fs * self.channels,
            OPUS_BITRATE_MAX => 1_500_000,
            b => b,
        };
        user.min(max_bitrate)
    }

    unsafe fn encode(&mut self, pcm: Pcm<'_>, frame_size: i32, data: *mut u8, out_data_bytes: i32) -> i32 {
        let max_data_bytes = out_data_bytes.min(1276 * 6);
        if frame_size <= 0 || max_data_bytes <= 0 {
            return OPUS_BAD_ARG;
        }
        // 100 ms does not fit in one byte.
        if max_data_bytes == 1 && self.fs == frame_size * 10 {
            return OPUS_BUFFER_TOO_SMALL;
        }
        let bitrate = self.bitrate(frame_size, max_data_bytes);
        let e = &mut self.inner;
        e.bitrate_bps = bitrate;
        e.complexity = self.complexity;
        e.use_cbr = self.use_vbr == 0;
        e.set_vbr_constraint(self.vbr_constraint != 0);
        e.use_inband_fec = self.fec_config != 0;
        e.packet_loss_perc = self.packet_loss_perc;
        e.set_dtx(self.use_dtx != 0);
        e.set_signal(match self.signal {
            OPUS_SIGNAL_VOICE => Signal::Voice,
            OPUS_SIGNAL_MUSIC => Signal::Music,
            _ => Signal::Auto,
        });
        e.set_bandwidth(rs_bandwidth(self.user_bandwidth));
        e.set_max_bandwidth(rs_bandwidth(self.max_bandwidth));

        let out = unsafe { slice::from_raw_parts_mut(data, max_data_bytes as usize) };
        let ret = match pcm {
            Pcm::I16(pcm) => e.encode_i16(pcm, frame_size as usize, out),
            Pcm::F32(pcm) => e.encode(pcm, frame_size as usize, out),
        };
        match ret {
            Ok(n) => {
                self.first = 0;
                self.prev_framesize = frame_size;
                n as i32
            }
            Err(e) if e.contains("too small") => OPUS_BUFFER_TOO_SMALL,
            Err(e) if e.contains("frame size") || e.contains("frame_size") => OPUS_BAD_ARG,
            Err(_) => OPUS_INTERNAL_ERROR,
        }
    }

    pub(crate) unsafe fn ctl(&mut self, request: c_int, mut ap: impl CtlArgs) -> c_int {
        unsafe {
            macro_rules! get {
                ($ty:ty, $v:expr) => {{
                    let p = ap.ptr::<$ty>();
                    if p.is_null() {
                        return OPUS_BAD_ARG;
                    }
                    *p = $v;
                }};
            }
            // A setting checked against `valid`, stored in `field`.
            macro_rules! set {
                ($field:ident, $valid:expr) => {{
                    let v = ap.int();
                    #[allow(clippy::redundant_closure_call)]
                    if !($valid)(v) {
                        return OPUS_BAD_ARG;
                    }
                    self.$field = v;
                }};
            }
            let bool_value = |v: i32| (0..=1).contains(&v);
            let bandwidth_value = |v: i32| (OPUS_BANDWIDTH_NARROWBAND..=OPUS_BANDWIDTH_FULLBAND).contains(&v);
            match request {
                req::SET_APPLICATION => {
                    let v = ap.int();
                    if self.application == OPUS_APPLICATION_RESTRICTED_SILK
                        || self.application == OPUS_APPLICATION_RESTRICTED_CELT
                        || !matches!(v, OPUS_APPLICATION_VOIP | OPUS_APPLICATION_AUDIO | OPUS_APPLICATION_RESTRICTED_LOWDELAY)
                        || (self.first == 0 && self.application != v)
                    {
                        return OPUS_BAD_ARG;
                    }
                    if v != self.application {
                        // opus-rs fixes the application at construction; no
                        // frame has been coded, so start over with the new one.
                        self.application = v;
                        self.reset();
                    }
                }
                req::GET_APPLICATION => get!(i32, self.application),
                req::SET_BITRATE => {
                    let mut v = ap.int();
                    if v != OPUS_AUTO && v != OPUS_BITRATE_MAX {
                        if v <= 0 {
                            return OPUS_BAD_ARG;
                        }
                        v = v.clamp(500, 750_000 * self.channels);
                    }
                    self.user_bitrate = v;
                }
                req::GET_BITRATE => get!(i32, self.bitrate(self.prev_framesize, 1276)),
                req::SET_FORCE_CHANNELS => {
                    set!(force_channels, |v: i32| (1..=self.channels).contains(&v) || v == OPUS_AUTO)
                }
                req::GET_FORCE_CHANNELS => get!(i32, self.force_channels),
                req::SET_MAX_BANDWIDTH => set!(max_bandwidth, bandwidth_value),
                req::GET_MAX_BANDWIDTH => get!(i32, self.max_bandwidth),
                req::SET_BANDWIDTH => set!(user_bandwidth, |v| bandwidth_value(v) || v == OPUS_AUTO),
                req::GET_BANDWIDTH => get!(i32, self.inner.bandwidth() as i32),
                req::SET_DTX => set!(use_dtx, bool_value),
                req::GET_DTX => get!(i32, self.use_dtx),
                req::SET_COMPLEXITY => set!(complexity, |v: i32| (0..=10).contains(&v)),
                req::GET_COMPLEXITY => get!(i32, self.complexity),
                req::SET_INBAND_FEC => set!(fec_config, |v: i32| (0..=2).contains(&v)),
                req::GET_INBAND_FEC => get!(i32, self.fec_config),
                req::SET_PACKET_LOSS_PERC => set!(packet_loss_perc, |v: i32| (0..=100).contains(&v)),
                req::GET_PACKET_LOSS_PERC => get!(i32, self.packet_loss_perc),
                req::SET_VBR => set!(use_vbr, bool_value),
                req::GET_VBR => get!(i32, self.use_vbr),
                req::SET_VOICE_RATIO => set!(voice_ratio, |v: i32| (-1..=100).contains(&v)),
                req::GET_VOICE_RATIO => get!(i32, self.voice_ratio),
                req::SET_VBR_CONSTRAINT => set!(vbr_constraint, bool_value),
                req::GET_VBR_CONSTRAINT => get!(i32, self.vbr_constraint),
                req::SET_SIGNAL => {
                    set!(signal, |v| matches!(v, OPUS_AUTO | OPUS_SIGNAL_VOICE | OPUS_SIGNAL_MUSIC))
                }
                req::GET_SIGNAL => get!(i32, self.signal),
                req::GET_LOOKAHEAD => get!(i32, self.inner.lookahead() as i32),
                req::GET_SAMPLE_RATE => get!(i32, self.fs),
                req::GET_FINAL_RANGE => get!(u32, self.inner.final_range()),
                req::SET_LSB_DEPTH => set!(lsb_depth, |v: i32| (8..=24).contains(&v)),
                req::GET_LSB_DEPTH => get!(i32, self.lsb_depth),
                req::SET_EXPERT_FRAME_DURATION => {
                    set!(variable_duration, |v: i32| (OPUS_FRAMESIZE_ARG..=OPUS_FRAMESIZE_120_MS).contains(&v))
                }
                req::GET_EXPERT_FRAME_DURATION => get!(i32, self.variable_duration),
                req::SET_PREDICTION_DISABLED => set!(prediction_disabled, bool_value),
                req::GET_PREDICTION_DISABLED => get!(i32, self.prediction_disabled),
                req::SET_PHASE_INVERSION_DISABLED => set!(phase_inversion_disabled, bool_value),
                req::GET_PHASE_INVERSION_DISABLED => get!(i32, self.phase_inversion_disabled),
                req::RESET_STATE => self.reset(),
                // MODE_SILK_ONLY..=MODE_CELT_ONLY
                req::SET_FORCE_MODE => set!(force_mode, |v: i32| (1000..=1002).contains(&v) || v == OPUS_AUTO),
                req::SET_LFE => self.lfe = ap.int(),
                req::SET_ENERGY_MASK => {
                    let _ = ap.ptr::<f32>();
                }
                req::GET_IN_DTX => get!(i32, self.inner.in_dtx() as i32),
                _ => return OPUS_UNIMPLEMENTED,
            }
            OPUS_OK
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_encoder_get_size(channels: c_int) -> c_int {
    if !(1..=2).contains(&channels) {
        return 0;
    }
    crate::align_size(size_of::<OpusEncoder>()) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_init(st: *mut OpusEncoder, fs: i32, channels: c_int, application: c_int) -> c_int {
    if st.is_null() {
        return OPUS_BAD_ARG;
    }
    guard(OPUS_INTERNAL_ERROR, || unsafe { OpusEncoder::init(st, fs, channels, application) })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_create(
    fs: i32,
    channels: c_int,
    application: c_int,
    error: *mut c_int,
) -> *mut OpusEncoder {
    unsafe {
        if !valid_fs(fs)
            || !(1..=2).contains(&channels)
            || !(rs_application(application).is_some() || application == OPUS_APPLICATION_RESTRICTED_SILK)
        {
            set_out(error, OPUS_BAD_ARG);
            return ptr::null_mut();
        }
        let st = crate::alloc_state(opus_encoder_get_size(channels) as usize) as *mut OpusEncoder;
        if st.is_null() {
            set_out(error, OPUS_ALLOC_FAIL);
            return st;
        }
        let ret = opus_encoder_init(st, fs, channels, application);
        set_out(error, ret);
        if ret != OPUS_OK {
            crate::free_state(st);
            return ptr::null_mut();
        }
        st
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_destroy(st: *mut OpusEncoder) {
    unsafe { crate::free_state(st) }
}

/// The frame to code out of `analysis_frame_size` samples, and its input.
unsafe fn frame<'a, T>(st: &OpusEncoder, pcm: *const T, analysis_frame_size: c_int) -> Option<(i32, &'a [T])> {
    let frame_size = frame_size_select(st.application, analysis_frame_size, st.variable_duration, st.fs);
    if frame_size <= 0 || pcm.is_null() {
        return None;
    }
    Some((frame_size, unsafe { slice::from_raw_parts(pcm, (frame_size * st.channels) as usize) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encode(
    st: *mut OpusEncoder,
    pcm: *const i16,
    analysis_frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
) -> i32 {
    guard(OPUS_INTERNAL_ERROR, || unsafe {
        let st = &mut *st;
        match frame(st, pcm, analysis_frame_size) {
            Some((n, pcm)) => st.encode(Pcm::I16(pcm), n, data, max_data_bytes),
            None => OPUS_BAD_ARG,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encode24(
    st: *mut OpusEncoder,
    pcm: *const i32,
    analysis_frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
) -> i32 {
    guard(OPUS_INTERNAL_ERROR, || unsafe {
        let st = &mut *st;
        match frame(st, pcm, analysis_frame_size) {
            Some((n, pcm)) => {
                let pcm: Vec<f32> = pcm.iter().map(|&s| s as f32 / 8388608.0).collect();
                st.encode(Pcm::F32(&pcm), n, data, max_data_bytes)
            }
            None => OPUS_BAD_ARG,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encode_float(
    st: *mut OpusEncoder,
    pcm: *const f32,
    analysis_frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
) -> i32 {
    guard(OPUS_INTERNAL_ERROR, || unsafe {
        let st = &mut *st;
        match frame(st, pcm, analysis_frame_size) {
            Some((n, pcm)) => st.encode(Pcm::F32(pcm), n, data, max_data_bytes),
            None => OPUS_BAD_ARG,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_ctl(st: *mut OpusEncoder, request: c_int, args: ...) -> c_int {
    guard(OPUS_INTERNAL_ERROR, || unsafe { (*st).ctl(request, args) })
}
