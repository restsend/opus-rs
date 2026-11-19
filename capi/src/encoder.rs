//! `opus_encoder_*` / `opus_encode*` over [`opus_rs::OpusEncoder`].

use core::ffi::c_int;
use core::mem::{MaybeUninit, size_of};
use core::slice;

use opus_rs::{Application, Bandwidth, FrameDuration, OpusMode, Signal};

use crate::{
    CtlArgs, OPUS_AUTO, OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, OPUS_UNIMPLEMENTED, code,
    get, guard, ok, req, set_bool,
};

/// libopus's `OpusEncoder` is the opus-rs encoder: one flat block (opus-rs
/// is built without `heap`), so C callers may place it with
/// `opus_encoder_init` and copy it with `memcpy`.
pub type OpusEncoder = opus_rs::OpusEncoder;

const _: () = assert!(align_of::<OpusEncoder>() <= 16, "malloc alignment");

/// An `OPUS_APPLICATION_*` value.
pub(crate) fn application(value: c_int) -> Option<Application> {
    Some(match value {
        2048 => Application::Voip,
        2049 => Application::Audio,
        2051 => Application::RestrictedLowDelay,
        2052 => Application::RestrictedSilk,
        2053 => Application::RestrictedCelt,
        _ => return None,
    })
}

/// An `OPUS_BANDWIDTH_*` value.
fn bandwidth(value: i32) -> Option<Bandwidth> {
    Some(match value {
        1101 => Bandwidth::Narrowband,
        1102 => Bandwidth::Mediumband,
        1103 => Bandwidth::Wideband,
        1104 => Bandwidth::Superwideband,
        1105 => Bandwidth::Fullband,
        _ => return None,
    })
}

/// An `OPUS_FRAMESIZE_*` value.
pub(crate) fn frame_duration(value: i32) -> Option<FrameDuration> {
    use FrameDuration::*;
    let all = [Argument, Ms2_5, Ms5, Ms10, Ms20, Ms40, Ms60, Ms80, Ms100, Ms120];
    all.into_iter().find(|&d| d as i32 == value)
}

/// Applies `set` to an accepted value, `OPUS_BAD_ARG` for `None`.
fn set<T>(value: Option<T>, set: impl FnOnce(T)) -> c_int {
    value.map_or(OPUS_BAD_ARG, |v| {
        set(v);
        OPUS_OK
    })
}

/// `opus_encoder_ctl` on `st` with the arguments in `ap`.
pub(crate) unsafe fn ctl(st: &mut OpusEncoder, request: c_int, mut ap: impl CtlArgs) -> c_int {
    let ap = &mut ap;
    // SAFETY: `ap` is the argument list of a ctl `request`, whose next
    // argument has the type opus_defines.h gives for `request`.
    unsafe {
        match request {
            req::SET_APPLICATION => match application(ap.int()) {
                Some(a) => ok(st.set_application(a)),
                None => OPUS_BAD_ARG,
            },
            req::GET_APPLICATION => get(ap, st.application() as i32),
            req::SET_BITRATE => ok(st.set_bitrate(ap.int())),
            req::GET_BITRATE => get(ap, st.bitrate()),
            req::SET_FORCE_CHANNELS => match ap.int() {
                OPUS_AUTO => ok(st.set_force_channels(None)),
                v => ok(st.set_force_channels(Some(usize::try_from(v).unwrap_or(0)))),
            },
            req::GET_FORCE_CHANNELS => get(ap, st.force_channels().map_or(OPUS_AUTO, |c| c as i32)),
            req::SET_MAX_BANDWIDTH => set(bandwidth(ap.int()), |b| st.set_max_bandwidth(b)),
            req::GET_MAX_BANDWIDTH => get(ap, st.max_bandwidth() as i32),
            req::SET_BANDWIDTH => match ap.int() {
                OPUS_AUTO => set(Some(Bandwidth::Auto), |b| st.set_bandwidth(b)),
                v => set(bandwidth(v), |b| st.set_bandwidth(b)),
            },
            req::GET_BANDWIDTH => get(ap, st.bandwidth() as i32),
            req::SET_DTX => set_bool(ap, |v| st.set_dtx(v)),
            req::GET_DTX => get(ap, st.dtx() as i32),
            req::SET_COMPLEXITY => ok(st.set_complexity(ap.int())),
            req::GET_COMPLEXITY => get(ap, st.complexity),
            req::SET_INBAND_FEC => ok(st.set_inband_fec(ap.int())),
            req::GET_INBAND_FEC => get(ap, st.inband_fec()),
            req::SET_PACKET_LOSS_PERC => ok(st.set_packet_loss_perc(ap.int())),
            req::GET_PACKET_LOSS_PERC => get(ap, st.packet_loss_perc),
            req::SET_VBR => set_bool(ap, |v| st.set_vbr(v)),
            req::GET_VBR => get(ap, st.vbr() as i32),
            req::SET_VBR_CONSTRAINT => set_bool(ap, |v| st.set_vbr_constraint(v)),
            req::GET_VBR_CONSTRAINT => get(ap, st.vbr_constraint() as i32),
            req::SET_SIGNAL => {
                let signal = match ap.int() {
                    OPUS_AUTO => Some(Signal::Auto),
                    3001 => Some(Signal::Voice),
                    3002 => Some(Signal::Music),
                    _ => None,
                };
                set(signal, |s| st.set_signal(s))
            }
            req::GET_SIGNAL => get(ap, st.signal() as i32),
            req::GET_LOOKAHEAD => get(ap, st.lookahead() as i32),
            req::GET_SAMPLE_RATE => get(ap, st.sample_rate()),
            req::GET_FINAL_RANGE => get(ap, st.final_range()),
            req::SET_LSB_DEPTH => ok(st.set_lsb_depth(ap.int())),
            req::GET_LSB_DEPTH => get(ap, st.lsb_depth()),
            req::SET_EXPERT_FRAME_DURATION => set(frame_duration(ap.int()), |d| st.set_expert_frame_duration(d)),
            req::GET_EXPERT_FRAME_DURATION => get(ap, st.expert_frame_duration() as i32),
            req::SET_PREDICTION_DISABLED => set_bool(ap, |v| st.set_prediction_disabled(v)),
            req::GET_PREDICTION_DISABLED => get(ap, st.prediction_disabled() as i32),
            req::SET_PHASE_INVERSION_DISABLED => set_bool(ap, |v| st.set_phase_inversion_disabled(v)),
            req::GET_PHASE_INVERSION_DISABLED => get(ap, st.phase_inversion_disabled() as i32),
            req::RESET_STATE => {
                st.reset();
                OPUS_OK
            }
            req::SET_FORCE_MODE => {
                let mode = match ap.int() {
                    OPUS_AUTO => Some(None),
                    1000 => Some(Some(OpusMode::SilkOnly)),
                    1001 => Some(Some(OpusMode::Hybrid)),
                    1002 => Some(Some(OpusMode::CeltOnly)),
                    _ => None,
                };
                set(mode, |m| st.set_force_mode(m))
            }
            req::GET_IN_DTX => get(ap, st.in_dtx() as i32),
            #[cfg(feature = "dred")]
            req::SET_DRED_DURATION => ok(st.set_dred_duration(ap.int())),
            #[cfg(feature = "dred")]
            req::GET_DRED_DURATION => get(ap, st.dred_duration()),
            _ => OPUS_UNIMPLEMENTED,
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
pub unsafe extern "C" fn opus_encoder_init(st: *mut OpusEncoder, fs: i32, channels: c_int, app: c_int) -> c_int {
    let Some(app) = application(app).filter(|_| !st.is_null()) else {
        return OPUS_BAD_ARG;
    };
    // SAFETY: `st` points to opus_encoder_get_size() writable bytes with
    // malloc's alignment (opus.h).
    let slot = unsafe { &mut *st.cast::<MaybeUninit<OpusEncoder>>() };
    guard(OPUS_INTERNAL_ERROR, || match OpusEncoder::init(slot, fs, channels as usize, app) {
        Ok(_) => OPUS_OK,
        Err(_) => OPUS_BAD_ARG,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_create(fs: i32, channels: c_int, app: c_int, error: *mut c_int) -> *mut OpusEncoder {
    // SAFETY: `create` hands `init` a block of the size it asks for; the
    // arguments are the caller's, as the header requires.
    unsafe { crate::create(opus_encoder_get_size(channels), error, |st| opus_encoder_init(st, fs, channels, app)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_destroy(st: *mut OpusEncoder) {
    // SAFETY: `st` is NULL or came from opus_encoder_create (opus.h).
    unsafe { crate::free_state(st) }
}

/// The checks every encode call shares, then `encode(pcm, frame_size, out)`
/// with the C input of `frame_size` samples of `channels` and the output
/// of `max_data_bytes`. `select` is the encoder's `select_frame_size` of
/// `frame_size`: `None` rejects it before the input is read, as libopus's
/// `frame_size_select` does.
pub(crate) unsafe fn encode_with<T>(
    pcm: *const T,
    frame_size: c_int,
    data: *mut u8,
    max_data_bytes: i32,
    channels: usize,
    select: Option<usize>,
    encode: impl FnOnce(&[T], usize, &mut [u8]) -> Result<usize, &'static str>,
) -> c_int {
    let frame_size = usize::try_from(frame_size).unwrap_or(0);
    let len = frame_size.checked_mul(channels).filter(|_| select.is_some());
    let (Some(len), false, false, 1..) = (len, pcm.is_null(), data.is_null(), max_data_bytes) else {
        return OPUS_BAD_ARG;
    };
    // SAFETY: `pcm` holds `frame_size` samples per channel and `data`
    // `max_data_bytes` bytes (opus.h).
    let (pcm, out) =
        unsafe { (slice::from_raw_parts(pcm, len), slice::from_raw_parts_mut(data, max_data_bytes as usize)) };
    guard(OPUS_INTERNAL_ERROR, || encode(pcm, frame_size, out).map_or_else(code, |n| n as c_int))
}

/// The encode calls: `st` is an encoder from opus_encoder_init or
/// opus_encoder_create (opus.h), the rest is checked by `encode_with`.
macro_rules! encode_fn {
    ($name:ident, $t:ty, $encode:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            st: *mut OpusEncoder,
            pcm: *const $t,
            analysis_frame_size: c_int,
            data: *mut u8,
            max_data_bytes: i32,
        ) -> i32 {
            // SAFETY: see the macro.
            let st = unsafe { &mut *st };
            let select = usize::try_from(analysis_frame_size).ok().and_then(|n| st.select_frame_size(n));
            // SAFETY: the C buffers as `encode_with` requires (opus.h).
            unsafe {
                encode_with(pcm, analysis_frame_size, data, max_data_bytes, st.channels(), select, |p, n, out| {
                    st.$encode(p, n, out)
                })
            }
        }
    };
}

encode_fn!(opus_encode, i16, encode_i16);
encode_fn!(opus_encode24, i32, encode24);
encode_fn!(opus_encode_float, f32, encode);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_encoder_ctl(st: *mut OpusEncoder, request: c_int, args: ...) -> c_int {
    // SAFETY: `st` is an initialized encoder and `args` the ctl's arguments
    // (opus.h).
    guard(OPUS_INTERNAL_ERROR, || unsafe { ctl(&mut *st, request, args) })
}
