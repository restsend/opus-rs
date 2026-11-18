//! `opus_multistream_decoder_*` over opus-rs's `MultistreamDecoder`.

use core::ffi::c_int;
use core::mem::size_of;
use core::ptr;
use core::slice;

use opus_rs::multistream::{ChannelMappingTable, MultistreamDecoder};

use crate::decoder::{self, OpusDecoder};
use crate::{
    Arg, CtlArgs, OPUS_ALLOC_FAIL, OPUS_BAD_ARG, OPUS_INTERNAL_ERROR, OPUS_OK, OPUS_UNIMPLEMENTED, align_size,
    decode_with, guard, req, set_out,
};

/// libopus's `OpusMSDecoder`.
pub struct OpusMSDecoder {
    channels: usize,
    ms: MultistreamDecoder,
}

impl OpusMSDecoder {
    /// libopus `opus_multistream_decoder_ctl`: getters ask the first
    /// stream, setters go to every stream.
    unsafe fn ctl(&mut self, request: c_int, mut ap: impl CtlArgs) -> c_int {
        let ms = &mut self.ms;
        // SAFETY: `ap` is the argument list of a ctl `request` (opus.h);
        // each forwarded call passes on the argument read here.
        unsafe {
            match request {
                req::GET_BANDWIDTH
                | req::GET_SAMPLE_RATE
                | req::GET_GAIN
                | req::GET_LAST_PACKET_DURATION
                | req::GET_PHASE_INVERSION_DISABLED
                | req::GET_COMPLEXITY => decoder::ctl(ms.decoder_state(0).unwrap(), request, Arg::Ptr(ap.ptr())),
                req::GET_FINAL_RANGE => crate::get(&mut ap, ms.final_range()),
                req::RESET_STATE => {
                    ms.reset();
                    OPUS_OK
                }
                req::MULTISTREAM_GET_DECODER_STATE => {
                    let Ok(dec) = usize::try_from(ap.int()).map_err(drop).and_then(|s| ms.decoder_state(s).map_err(drop))
                    else {
                        return OPUS_BAD_ARG;
                    };
                    crate::get(&mut ap, dec as *mut OpusDecoder)
                }
                req::SET_GAIN | req::SET_COMPLEXITY | req::SET_PHASE_INVERSION_DISABLED => {
                    let arg = Arg::Int(ap.int());
                    let mut s = 0;
                    while let Ok(dec) = ms.decoder_state(s) {
                        let ret = decoder::ctl(dec, request, arg);
                        if ret != OPUS_OK {
                            return ret;
                        }
                        s += 1;
                    }
                    OPUS_OK
                }
                _ => OPUS_UNIMPLEMENTED,
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_multistream_decoder_get_size(nb_streams: c_int, nb_coupled_streams: c_int) -> i32 {
    if nb_streams < 1 || nb_coupled_streams > nb_streams || nb_coupled_streams < 0 {
        return 0;
    }
    // The size of the flat layout: the header, then one decoder per stream.
    (align_size(size_of::<OpusMSDecoder>()) + nb_streams as usize * align_size(size_of::<OpusDecoder>())) as i32
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
    if st.is_null() || mapping.is_null() || !(1..=255).contains(&channels) || !(0..=255).contains(&streams) {
        return OPUS_BAD_ARG;
    }
    let Ok(coupled_count) = u8::try_from(coupled_streams) else {
        return OPUS_BAD_ARG;
    };
    let table = ChannelMappingTable {
        channels: channels as u8,
        stream_count: streams as u8,
        coupled_count,
        // SAFETY: `mapping` holds `channels` bytes (opus_multistream.h).
        mapping: unsafe { slice::from_raw_parts(mapping, channels as usize) }.to_vec(),
    };
    guard(OPUS_INTERNAL_ERROR, || match MultistreamDecoder::new(fs, table) {
        Ok(ms) => {
            // SAFETY: `st` points to opus_multistream_decoder_get_size()
            // writable bytes with malloc's alignment.
            unsafe { st.write(OpusMSDecoder { channels: channels as usize, ms }) };
            OPUS_OK
        }
        Err(_) => OPUS_BAD_ARG,
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
    // SAFETY: `error` is NULL or valid for a write; `mapping` holds
    // `channels` bytes (opus_multistream.h).
    unsafe {
        let size = opus_multistream_decoder_get_size(streams, coupled_streams);
        if size == 0 {
            set_out(error, OPUS_BAD_ARG);
            return ptr::null_mut();
        }
        let st = crate::alloc_state(size as usize).cast::<OpusMSDecoder>();
        if st.is_null() {
            set_out(error, OPUS_ALLOC_FAIL);
            return st;
        }
        let ret = opus_multistream_decoder_init(st, fs, channels, streams, coupled_streams, mapping);
        set_out(error, ret);
        if ret != OPUS_OK {
            crate::free_state(st);
            return ptr::null_mut();
        }
        st
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decoder_destroy(st: *mut OpusMSDecoder) {
    if !st.is_null() {
        // SAFETY: `st` came from opus_multistream_decoder_create.
        unsafe {
            ptr::drop_in_place(st);
            crate::free_state(st);
        }
    }
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
            // SAFETY: see the macro.
            let st = unsafe { &mut *st };
            // libopus decodes at most 120 ms.
            let frame_size = frame_size.min(st.ms.sample_rate() / 25 * 3);
            let ms = &mut st.ms;
            guard(OPUS_INTERNAL_ERROR, || unsafe {
                decode_with(data, len, pcm, frame_size, decode_fec, st.channels, |p, n, out, fec| {
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
    guard(OPUS_INTERNAL_ERROR, || unsafe { (*st).ctl(request, args) })
}
