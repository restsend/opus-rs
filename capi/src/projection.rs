//! `opus_projection_*` over opus-rs's projection encoder header and
//! decoder view. A projection encoder is a multistream encoder whose
//! header holds the matrices; a projection decoder is a multistream
//! decoder followed by its demixing matrix, in one flat block.

use core::ffi::c_int;
use core::slice;

use opus_rs::multistream::{MultistreamDecoderView, MultistreamLayout};
use opus_rs::multistream_encoder::MultistreamEncoderHeader;
use opus_rs::projection::ProjectionDecoderView;

use crate::decoder::OpusDecoder;
use crate::encoder::{self, OpusEncoder};
use crate::multistream::{self, OpusMSEncoder, encode_fn, init_decoders, init_encoders, write_layout};
use crate::{CtlArgs, OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, block_size, decode_with, get, guard, ok, tail};

/// libopus's `OpusProjectionEncoder`: a multistream encoder with the
/// projection matrices in its header.
pub type OpusProjectionEncoder = OpusMSEncoder;

/// libopus's `OpusProjectionDecoder`: the layout, the stream decoders,
/// then the demixing matrix.
pub type OpusProjectionDecoder = MultistreamLayout;

const GET_DEMIXING_MATRIX_GAIN: c_int = 6001;
const GET_DEMIXING_MATRIX_SIZE: c_int = 6003;
const GET_DEMIXING_MATRIX: c_int = 6005;

/// The projection header of `channels` channels in `mapping_family`.
fn header(fs: i32, channels: c_int, mapping_family: c_int, app: opus_rs::Application) -> Result<OpusMSEncoder, &'static str> {
    let channels = usize::try_from(channels).map_err(|_| opus_rs::error::BAD_ARG)?;
    MultistreamEncoderHeader::new_projection(fs, channels, mapping_family, app)
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_projection_ambisonics_encoder_get_size(channels: c_int, mapping_family: c_int) -> i32 {
    header(48000, channels, mapping_family, opus_rs::Application::Audio)
        .map_or(0, |h| block_size::<OpusMSEncoder, OpusEncoder>(h.streams()) as i32)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_ambisonics_encoder_init(
    st: *mut OpusProjectionEncoder,
    fs: i32,
    channels: c_int,
    mapping_family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    application: c_int,
) -> c_int {
    let (Some(app), false) = (encoder::application(application), st.is_null()) else {
        return OPUS_BAD_ARG;
    };
    // SAFETY: `st` has opus_projection_ambisonics_encoder_get_size() bytes
    // and the out-parameters are NULL or valid (opus_projection.h).
    unsafe {
        let ret = init_encoders(st, header(fs, channels, mapping_family, app));
        if ret == OPUS_OK {
            write_layout(st, streams, coupled_streams, core::ptr::null_mut());
        }
        ret
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_ambisonics_encoder_create(
    fs: i32,
    channels: c_int,
    mapping_family: c_int,
    streams: *mut c_int,
    coupled_streams: *mut c_int,
    application: c_int,
    error: *mut c_int,
) -> *mut OpusProjectionEncoder {
    let size = opus_projection_ambisonics_encoder_get_size(channels, mapping_family);
    unsafe {
        crate::create(size, error, |st| {
            opus_projection_ambisonics_encoder_init(st, fs, channels, mapping_family, streams, coupled_streams, application)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_encoder_destroy(st: *mut OpusProjectionEncoder) {
    // SAFETY: `st` is NULL or came from opus_projection_ambisonics_encoder_create.
    unsafe { crate::free_state(st) }
}

encode_fn!(opus_projection_encode, i16, encode_i16);
encode_fn!(opus_projection_encode24, i32, encode24);
encode_fn!(opus_projection_encode_float, f32, encode);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_encoder_ctl(st: *mut OpusProjectionEncoder, request: c_int, args: ...) -> c_int {
    let mut ap = args;
    // SAFETY: `st` is an initialized projection encoder and `ap` the ctl's
    // arguments (opus_projection.h).
    guard(OPUS_INTERNAL_ERROR, || unsafe {
        let head = &*st;
        match request {
            GET_DEMIXING_MATRIX_GAIN => get(&mut ap, head.demixing_matrix_gain()),
            GET_DEMIXING_MATRIX_SIZE => get(&mut ap, head.demixing_matrix_size() as i32),
            GET_DEMIXING_MATRIX => {
                let (matrix, size) = (ap.ptr::<u8>(), ap.int());
                if matrix.is_null() || size as usize != head.demixing_matrix_size() {
                    return OPUS_BAD_ARG;
                }
                ok(head.demixing_matrix(slice::from_raw_parts_mut(matrix, size as usize)))
            }
            _ => multistream::encoder_ctl(st, request, ap),
        }
    })
}

/// The demixing matrix cells of `channels` by `streams + coupled`, or
/// `None` for sizes libopus has no decoder for.
fn matrix_cells(channels: c_int, streams: c_int, coupled: c_int) -> Option<usize> {
    let inputs = streams.checked_add(coupled)?;
    let valid = (1..=255).contains(&channels)
        && (1..=255).contains(&streams)
        && (0..=streams).contains(&coupled)
        && inputs <= 255
        && inputs * channels * 2 <= 65004;
    valid.then_some((inputs * channels) as usize)
}

/// The decoders and the demixing matrix of a projection decoder.
///
/// # Safety
///
/// `st` points to `opus_projection_decoder_get_size()` bytes of the
/// decoder's layout, used by nothing else while the slices live.
unsafe fn decoder_parts<'a>(st: *mut OpusProjectionDecoder, layout: &MultistreamLayout) -> (&'a mut [OpusDecoder], &'a mut [i16]) {
    let n = layout.streams();
    let cells = layout.channels() * (n + layout.coupled_streams());
    // SAFETY: see the function: the matrix follows the decoders, at an even
    // offset.
    unsafe {
        let matrix = st.cast::<u8>().add(block_size::<OpusProjectionDecoder, OpusDecoder>(n)).cast::<i16>();
        (tail(st, n), slice::from_raw_parts_mut(matrix, cells))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_projection_decoder_get_size(channels: c_int, streams: c_int, coupled_streams: c_int) -> i32 {
    matrix_cells(channels, streams, coupled_streams).map_or(0, |cells| {
        (block_size::<OpusProjectionDecoder, OpusDecoder>(streams as usize) + 2 * cells) as i32
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decoder_init(
    st: *mut OpusProjectionDecoder,
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    demixing_matrix: *const u8,
    demixing_matrix_size: i32,
) -> c_int {
    let Some(cells) = matrix_cells(channels, streams, coupled_streams) else {
        return OPUS_BAD_ARG;
    };
    if st.is_null() || demixing_matrix.is_null() || demixing_matrix_size as usize != 2 * cells {
        return OPUS_BAD_ARG;
    }
    let identity: [u8; 255] = core::array::from_fn(|i| i as u8);
    let (channels, streams, coupled) = (channels as usize, streams as usize, coupled_streams as usize);
    let Ok(layout) = MultistreamLayout::new(channels, streams, coupled, &identity[..channels]) else {
        return OPUS_BAD_ARG;
    };
    // SAFETY: `st` has opus_projection_decoder_get_size() bytes with
    // malloc's alignment and `demixing_matrix` holds
    // `demixing_matrix_size` bytes (opus_projection.h).
    guard(OPUS_INTERNAL_ERROR, || unsafe {
        st.write(layout);
        let (_, matrix) = decoder_parts(st, &layout);
        let bytes = slice::from_raw_parts(demixing_matrix, 2 * cells);
        for (cell, b) in matrix.iter_mut().zip(bytes.chunks_exact(2)) {
            *cell = i16::from_le_bytes([b[0], b[1]]);
        }
        init_decoders(st, fs, &layout)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decoder_create(
    fs: i32,
    channels: c_int,
    streams: c_int,
    coupled_streams: c_int,
    demixing_matrix: *const u8,
    demixing_matrix_size: i32,
    error: *mut c_int,
) -> *mut OpusProjectionDecoder {
    let size = opus_projection_decoder_get_size(channels, streams, coupled_streams);
    unsafe {
        crate::create(size, error, |st| {
            opus_projection_decoder_init(st, fs, channels, streams, coupled_streams, demixing_matrix, demixing_matrix_size)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decoder_destroy(st: *mut OpusProjectionDecoder) {
    // SAFETY: `st` is NULL or came from opus_projection_decoder_create.
    unsafe { crate::free_state(st) }
}

/// The decode calls: `st` is an initialized projection decoder
/// (opus_projection.h), the rest is checked by `decode_with`.
macro_rules! decode_fn {
    ($name:ident, $t:ty, $decode:ident, $decode_fec:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            st: *mut OpusProjectionDecoder,
            data: *const u8,
            len: i32,
            pcm: *mut $t,
            frame_size: c_int,
            decode_fec: c_int,
        ) -> c_int {
            // SAFETY: see the macro.
            let layout = unsafe { &*st };
            let (decoders, matrix) = unsafe { decoder_parts(st, layout) };
            // libopus decodes at most 120 ms.
            let (channels, frame_size) = (layout.channels(), frame_size.min(decoders[0].sample_rate() / 25 * 3));
            let view = MultistreamDecoderView::new(layout, decoders).and_then(|ms| ProjectionDecoderView::new(ms, matrix));
            let Ok(mut dec) = view else {
                return OPUS_INTERNAL_ERROR;
            };
            guard(OPUS_INTERNAL_ERROR, || unsafe {
                decode_with(data, len, pcm, frame_size, decode_fec, channels, |p, n, out, fec| {
                    if fec { dec.$decode_fec(p, n, out) } else { dec.$decode(p, n, out) }
                })
            })
        }
    };
}

decode_fn!(opus_projection_decode, i16, decode_i16, decode_fec_i16);
decode_fn!(opus_projection_decode24, i32, decode24, decode_fec24);
decode_fn!(opus_projection_decode_float, f32, decode, decode_fec);

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_projection_decoder_ctl(st: *mut OpusProjectionDecoder, request: c_int, args: ...) -> c_int {
    // SAFETY: `st` is an initialized projection decoder and `args` the
    // ctl's arguments (opus_projection.h); libopus forwards every ctl to
    // the multistream decoder.
    guard(OPUS_INTERNAL_ERROR, || unsafe { multistream::decoder_ctl(decoder_parts(st, &*st).0, request, args) })
}
