//! `opus_multistream_*` over opus-rs's multistream views. Each state is
//! one flat block, as in libopus: a header (the layout, or the encoder's
//! header) followed by one opus-rs decoder or encoder per stream.

use core::ffi::c_int;
use core::mem::MaybeUninit;
use core::slice;

use opus_rs::multistream::{MultistreamDecoderView, MultistreamLayout};
use opus_rs::multistream_encoder::{MultistreamEncoderHeader, MultistreamEncoderView};

use crate::decoder::{self, OpusDecoder};
use crate::encoder::{self, OpusEncoder};
use crate::{
    Arg, CtlArgs, OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, OPUS_UNIMPLEMENTED, block_size, code, decode_with, get,
    guard, ok, req, set_out, tail,
};

/// libopus's `OpusMSDecoder`: the layout, then the stream decoders.
pub type OpusMSDecoder = MultistreamLayout;

/// libopus's `OpusMSEncoder`: the header, then the stream encoders.
pub type OpusMSEncoder = MultistreamEncoderHeader;

/// libopus `opus_multistream_decoder_ctl` on `decoders`: getters ask the
/// first stream, setters go to every stream.
pub(crate) unsafe fn decoder_ctl(decoders: &mut [OpusDecoder], request: c_int, mut ap: impl CtlArgs) -> c_int {
    // SAFETY: `ap` is the argument list of a ctl `request` (opus.h); each
    // forwarded call passes on the argument read here.
    unsafe {
        match request {
            req::GET_BANDWIDTH
            | req::GET_SAMPLE_RATE
            | req::GET_GAIN
            | req::GET_LAST_PACKET_DURATION
            | req::GET_PHASE_INVERSION_DISABLED
            | req::GET_COMPLEXITY => decoder::ctl(&mut decoders[0], request, Arg::Ptr(ap.ptr())),
            req::GET_FINAL_RANGE => get(&mut ap, decoders.iter().fold(0, |r, d| r ^ d.final_range())),
            req::RESET_STATE => {
                decoders.iter_mut().for_each(OpusDecoder::reset);
                OPUS_OK
            }
            req::MULTISTREAM_GET_DECODER_STATE => match usize::try_from(ap.int()).ok().and_then(|s| decoders.get_mut(s)) {
                Some(dec) => get(&mut ap, dec as *mut OpusDecoder),
                None => OPUS_BAD_ARG,
            },
            req::SET_GAIN | req::SET_COMPLEXITY | req::SET_PHASE_INVERSION_DISABLED => {
                let arg = Arg::Int(ap.int());
                decoders.iter_mut().map(|d| decoder::ctl(d, request, arg)).find(|&r| r != OPUS_OK).unwrap_or(OPUS_OK)
            }
            _ => OPUS_UNIMPLEMENTED,
        }
    }
}

/// Writes new stream decoders for `layout` after the header at `st`.
///
/// # Safety
///
/// `st` points to `block_size::<H, OpusDecoder>(layout.streams())`
/// writable bytes with malloc's alignment.
pub(crate) unsafe fn init_decoders<H>(st: *mut H, fs: i32, layout: &MultistreamLayout) -> c_int {
    // SAFETY: see the function.
    let slots: &mut [MaybeUninit<OpusDecoder>] = unsafe { tail(st, layout.streams()) };
    let all = slots.iter_mut().enumerate().all(|(s, slot)| OpusDecoder::init(slot, fs, layout.stream_channels(s)).is_ok());
    if all { OPUS_OK } else { OPUS_BAD_ARG }
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_multistream_decoder_get_size(nb_streams: c_int, nb_coupled_streams: c_int) -> i32 {
    if nb_streams < 1 || nb_coupled_streams > nb_streams || nb_coupled_streams < 0 {
        return 0;
    }
    block_size::<OpusMSDecoder, OpusDecoder>(nb_streams as usize) as i32
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decoder_init(
    st: *mut OpusMSDecoder,
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    mapping: *const u8,
) -> c_int {
    let (Ok(channels), Ok(streams), Ok(coupled), false, false) = (
        usize::try_from(channels),
        usize::try_from(streams),
        usize::try_from(coupled_streams),
        st.is_null(),
        mapping.is_null(),
    ) else {
        return OPUS_BAD_ARG;
    };
    // SAFETY: `mapping` holds `channels` bytes (opus_multistream.h).
    let Ok(layout) = MultistreamLayout::new(channels, streams, coupled, unsafe { slice::from_raw_parts(mapping, channels) })
    else {
        return OPUS_BAD_ARG;
    };
    // SAFETY: `st` points to opus_multistream_decoder_get_size() writable
    // bytes with malloc's alignment.
    guard(OPUS_INTERNAL_ERROR, || unsafe {
        st.write(layout);
        init_decoders(st, fs, &layout)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decoder_create(
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    mapping: *const u8,
    error: *mut c_int,
) -> *mut OpusMSDecoder {
    let size = opus_multistream_decoder_get_size(streams, coupled_streams);
    unsafe { crate::create(size, error, |st| opus_multistream_decoder_init(st, fs, channels, streams, coupled_streams, mapping)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decoder_destroy(st: *mut OpusMSDecoder) {
    // SAFETY: `st` is NULL or came from opus_multistream_decoder_create.
    unsafe { crate::free_state(st) }
}

/// The decode calls: `st` is an initialized multistream decoder
/// (opus_multistream.h), the rest is checked by `decode_with`.
macro_rules! decode_fn {
    ($name:ident, $t:ty, $decode:ident, $decode_fec:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            st: *mut OpusMSDecoder,
            data: *const u8,
            len: i32,
            pcm: *mut $t,
            frame_size: c_int,
            decode_fec: c_int,
        ) -> c_int {
            // SAFETY: see the macro; the layout and the decoders do not
            // overlap.
            let (layout, decoders): (_, &mut [OpusDecoder]) = unsafe { (&*st, tail(st, (*st).streams())) };
            // libopus decodes at most 120 ms.
            let (channels, frame_size) = (layout.channels(), frame_size.min(decoders[0].sample_rate() / 25 * 3));
            let Ok(mut ms) = MultistreamDecoderView::new(layout, decoders) else {
                return OPUS_INTERNAL_ERROR;
            };
            guard(OPUS_INTERNAL_ERROR, || unsafe {
                decode_with(data, len, pcm, frame_size, decode_fec, channels, |p, n, out, fec| {
                    if fec { ms.$decode_fec(p, n, out) } else { ms.$decode(p, n, out) }
                })
            })
        }
    };
}

decode_fn!(opus_multistream_decode, i16, decode_i16, decode_fec_i16);
decode_fn!(opus_multistream_decode24, i32, decode24, decode_fec24);
decode_fn!(opus_multistream_decode_float, f32, decode, decode_fec);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decoder_ctl(st: *mut OpusMSDecoder, request: c_int, args: ...) -> c_int {
    // SAFETY: `st` is an initialized multistream decoder and `args` the
    // ctl's arguments (opus_multistream.h).
    guard(OPUS_INTERNAL_ERROR, || unsafe { decoder_ctl(tail(st, (*st).streams()), request, args) })
}

/// The view of the multistream (or projection) encoder `st`.
///
/// # Safety
///
/// `st` is an encoder from `init_encoders`, used by nothing else while the
/// view lives.
pub(crate) unsafe fn encoder_view<'a>(st: *mut OpusMSEncoder) -> Option<MultistreamEncoderView<'a>> {
    // SAFETY: see the function; the header and the encoders do not overlap.
    unsafe { MultistreamEncoderView::new(&mut *st, tail(st, (*st).streams())).ok() }
}

/// Writes `head` and new stream encoders at `st`.
///
/// # Safety
///
/// `st` points to `block_size::<OpusMSEncoder, OpusEncoder>(head.streams())`
/// writable bytes with malloc's alignment.
pub(crate) unsafe fn init_encoders(st: *mut OpusMSEncoder, head: Result<OpusMSEncoder, &str>) -> c_int {
    let head = match head {
        Ok(h) => h,
        Err(e) => return code(e),
    };
    guard(OPUS_INTERNAL_ERROR, || {
        // SAFETY: see the function.
        let slots: &mut [MaybeUninit<OpusEncoder>] = unsafe { tail(st, head.streams()) };
        for (s, slot) in slots.iter_mut().enumerate() {
            match head.stream_encoder(s) {
                Ok(enc) => _ = slot.write(enc),
                Err(e) => return code(e),
            }
        }
        unsafe { st.write(head) };
        OPUS_OK
    })
}

/// libopus `opus_multistream_encoder_ctl` on the encoder `st`: getters
/// ask the first stream, setters go to every stream.
///
/// # Safety
///
/// `st` is an encoder from `init_encoders` and `ap` the argument list of a
/// ctl `request` (opus.h).
pub(crate) unsafe fn encoder_ctl(st: *mut OpusMSEncoder, request: c_int, mut ap: impl CtlArgs) -> c_int {
    // SAFETY: see the function; each forwarded call passes on the argument
    // read here, and the header and the encoders do not overlap.
    unsafe {
        let (head, encoders): (_, &mut [OpusEncoder]) = (&mut *st, tail(st, (*st).streams()));
        match request {
            req::SET_BITRATE => ok(head.set_bitrate(ap.int())),
            req::GET_BITRATE => get(&mut ap, encoders.iter().map(OpusEncoder::bitrate).sum::<i32>()),
            req::GET_LSB_DEPTH
            | req::GET_VBR
            | req::GET_APPLICATION
            | req::GET_BANDWIDTH
            | req::GET_COMPLEXITY
            | req::GET_PACKET_LOSS_PERC
            | req::GET_DTX
            | req::GET_VBR_CONSTRAINT
            | req::GET_SIGNAL
            | req::GET_LOOKAHEAD
            | req::GET_SAMPLE_RATE
            | req::GET_INBAND_FEC
            | req::GET_FORCE_CHANNELS
            | req::GET_PREDICTION_DISABLED
            | req::GET_PHASE_INVERSION_DISABLED => encoder::ctl(&mut encoders[0], request, Arg::Ptr(ap.ptr())),
            req::GET_FINAL_RANGE => get(&mut ap, encoders.iter().fold(0, |r, e| r ^ e.final_range())),
            req::SET_LSB_DEPTH
            | req::SET_COMPLEXITY
            | req::SET_VBR
            | req::SET_VBR_CONSTRAINT
            | req::SET_MAX_BANDWIDTH
            | req::SET_BANDWIDTH
            | req::SET_SIGNAL
            | req::SET_APPLICATION
            | req::SET_INBAND_FEC
            | req::SET_PACKET_LOSS_PERC
            | req::SET_DTX
            | req::SET_FORCE_MODE
            | req::SET_FORCE_CHANNELS
            | req::SET_PREDICTION_DISABLED
            | req::SET_PHASE_INVERSION_DISABLED => {
                let arg = Arg::Int(ap.int());
                encoders.iter_mut().map(|e| encoder::ctl(e, request, arg)).find(|&r| r != OPUS_OK).unwrap_or(OPUS_OK)
            }
            req::MULTISTREAM_GET_ENCODER_STATE => match usize::try_from(ap.int()).ok().and_then(|s| encoders.get_mut(s)) {
                Some(enc) => get(&mut ap, enc as *mut OpusEncoder),
                None => OPUS_BAD_ARG,
            },
            req::SET_EXPERT_FRAME_DURATION => match encoder::frame_duration(ap.int()) {
                Some(d) => {
                    head.set_expert_frame_duration(d);
                    OPUS_OK
                }
                None => OPUS_BAD_ARG,
            },
            req::GET_EXPERT_FRAME_DURATION => get(&mut ap, head.expert_frame_duration() as i32),
            req::RESET_STATE => match encoder_view(st) {
                Some(mut ms) => {
                    ms.reset();
                    OPUS_OK
                }
                None => OPUS_INTERNAL_ERROR,
            },
            _ => OPUS_UNIMPLEMENTED,
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_multistream_encoder_get_size(nb_streams: c_int, nb_coupled_streams: c_int) -> i32 {
    if nb_streams < 1 || nb_coupled_streams > nb_streams || nb_coupled_streams < 0 {
        return 0;
    }
    block_size::<OpusMSEncoder, OpusEncoder>(nb_streams as usize) as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_multistream_surround_encoder_get_size(channels: c_int, mapping_family: c_int) -> i32 {
    let head = usize::try_from(channels)
        .map_err(drop)
        .and_then(|c| MultistreamEncoderHeader::new_surround(48000, c, mapping_family, opus_rs::Application::Audio).map_err(drop));
    head.map_or(0, |h| block_size::<OpusMSEncoder, OpusEncoder>(h.streams()) as i32)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encoder_init(
    st: *mut OpusMSEncoder,
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    mapping: *const u8,
    application: c_int,
) -> c_int {
    let (Ok(channels), Ok(streams), Ok(coupled), Some(app), false, false) = (
        usize::try_from(channels),
        usize::try_from(streams),
        usize::try_from(coupled_streams),
        encoder::application(application),
        st.is_null(),
        mapping.is_null(),
    ) else {
        return OPUS_BAD_ARG;
    };
    // SAFETY: `mapping` holds `channels` bytes and `st` has
    // opus_multistream_encoder_get_size() bytes (opus_multistream.h).
    unsafe {
        let mapping = slice::from_raw_parts(mapping, channels);
        init_encoders(st, MultistreamEncoderHeader::new(fs, channels, streams, coupled, mapping, app))
    }
}

/// Writes the layout of the encoder `st` to the out-parameters of the
/// surround and projection init calls.
///
/// # Safety
///
/// `st` is initialized; each pointer is NULL or valid for its writes,
/// `mapping` for the channel count.
pub(crate) unsafe fn write_layout(st: *const OpusMSEncoder, streams: *mut c_int, coupled: *mut c_int, mapping: *mut u8) {
    // SAFETY: see the function.
    unsafe {
        set_out(streams, (*st).streams() as c_int);
        set_out(coupled, (*st).coupled_streams() as c_int);
        if !mapping.is_null() {
            slice::from_raw_parts_mut(mapping, (*st).channels()).copy_from_slice((*st).mapping());
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_surround_encoder_init(
    st: *mut OpusMSEncoder,
    fs: i32,
    channels: c_int,
    mapping_family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    mapping: *mut u8,
    application: c_int,
) -> c_int {
    let (Ok(channels), Some(app), false) = (usize::try_from(channels), encoder::application(application), st.is_null())
    else {
        return OPUS_BAD_ARG;
    };
    // SAFETY: `st` has opus_multistream_surround_encoder_get_size() bytes
    // and the out-parameters are valid (opus_multistream.h).
    unsafe {
        let ret = init_encoders(st, MultistreamEncoderHeader::new_surround(fs, channels, mapping_family, app));
        if ret == OPUS_OK {
            write_layout(st, streams, coupled_streams, mapping);
        }
        ret
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encoder_create(
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    mapping: *const u8,
    application: c_int,
    error: *mut c_int,
) -> *mut OpusMSEncoder {
    let size = opus_multistream_encoder_get_size(streams, coupled_streams);
    unsafe {
        crate::create(size, error, |st| {
            opus_multistream_encoder_init(st, fs, channels, streams, coupled_streams, mapping, application)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_surround_encoder_create(
    fs: i32,
    channels: c_int,
    mapping_family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    mapping: *mut u8,
    application: c_int,
    error: *mut c_int,
) -> *mut OpusMSEncoder {
    let size = opus_multistream_surround_encoder_get_size(channels, mapping_family);
    unsafe {
        crate::create(size, error, |st| {
            opus_multistream_surround_encoder_init(st, fs, channels, mapping_family, streams, coupled_streams, mapping, application)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encoder_destroy(st: *mut OpusMSEncoder) {
    // SAFETY: `st` is NULL or came from opus_multistream_encoder_create.
    unsafe { crate::free_state(st) }
}

/// The encode calls of a multistream or projection encoder `st`
/// (opus_multistream.h, opus_projection.h); the rest is checked by
/// `encode_with`.
macro_rules! encode_fn {
    ($name:ident, $t:ty, $encode:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            st: *mut $crate::multistream::OpusMSEncoder,
            pcm: *const $t,
            frame_size: core::ffi::c_int,
            data: *mut u8,
            max_data_bytes: i32,
        ) -> core::ffi::c_int {
            // SAFETY: see the macro.
            let head = unsafe { &*st };
            let select = usize::try_from(frame_size).ok().and_then(|n| head.select_frame_size(n));
            let channels = head.channels();
            let Some(mut ms) = (unsafe { $crate::multistream::encoder_view(st) }) else {
                return $crate::OPUS_INTERNAL_ERROR;
            };
            unsafe {
                $crate::encoder::encode_with(pcm, frame_size, data, max_data_bytes, channels, select, |p, n, out| {
                    ms.$encode(p, n, out)
                })
            }
        }
    };
}

encode_fn!(opus_multistream_encode, i16, encode_i16);
encode_fn!(opus_multistream_encode24, i32, encode24);
encode_fn!(opus_multistream_encode_float, f32, encode);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_encoder_ctl(st: *mut OpusMSEncoder, request: c_int, args: ...) -> c_int {
    // SAFETY: `st` is an initialized multistream encoder and `args` the
    // ctl's arguments (opus_multistream.h).
    guard(OPUS_INTERNAL_ERROR, || unsafe { encoder_ctl(st, request, args) })
}
