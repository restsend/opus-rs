//! `opus_multistream_decoder_*`, after libopus's opus_multistream_decoder.c:
//! one `OpusDecoder` per stream, laid out after the channel layout in one
//! block.

use core::ffi::{VaList, c_int, c_void};
use core::mem::size_of;
use core::ptr;
use core::slice;

use crate::decoder::{OpusDecoder, opus_decoder_get_size, opus_decoder_init};
use crate::packet::{self, parse};
use crate::{
    Arg, CtlArgs, OPUS_ALLOC_FAIL, OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_INTERNAL_ERROR,
    OPUS_INVALID_PACKET, OPUS_OK, OPUS_UNIMPLEMENTED, align_size, guard, req, set_out,
};

/// libopus's `OpusMSDecoder` (its `ChannelLayout`); the stream decoders
/// follow it.
#[repr(C)]
pub struct OpusMSDecoder {
    nb_channels: c_int,
    nb_streams: c_int,
    nb_coupled_streams: c_int,
    mapping: [u8; 256],
}

impl OpusMSDecoder {
    fn mapping(&self) -> &[u8] {
        &self.mapping[..self.nb_channels as usize]
    }

    /// libopus `validate_layout`.
    fn valid(&self) -> bool {
        let max_channel = self.nb_streams + self.nb_coupled_streams;
        max_channel <= 255 && self.mapping().iter().all(|&m| (m as c_int) < max_channel || m == 255)
    }

    /// The decoder of stream `s`.
    unsafe fn stream(&mut self, s: c_int) -> *mut OpusDecoder {
        let coupled = align_size(opus_decoder_get_size(2) as usize);
        let mono = align_size(opus_decoder_get_size(1) as usize);
        let s = s as usize;
        let c = (self.nb_coupled_streams as usize).min(s);
        let offset = align_size(size_of::<Self>()) + c * coupled + (s - c) * mono;
        unsafe { (self as *mut Self as *mut u8).add(offset) as *mut OpusDecoder }
    }

    /// The output channels stream `s` (its `side`: 0 left / mono, 1 right)
    /// plays on.
    fn channels_of(&self, s: c_int, side: c_int) -> impl Iterator<Item = usize> + '_ {
        let target = if s < self.nb_coupled_streams { s * 2 + side } else { s + self.nb_coupled_streams };
        self.mapping().iter().enumerate().filter(move |&(_, &m)| m as c_int == target).map(|(c, _)| c)
    }

    /// libopus `opus_multistream_packet_validate`: the samples per stream,
    /// which every stream must agree on.
    fn validate(&self, mut data: &[u8], fs: i32) -> c_int {
        let mut samples = 0;
        for s in 0..self.nb_streams {
            if data.is_empty() {
                return OPUS_INVALID_PACKET;
            }
            let p = match parse(data, s != self.nb_streams - 1) {
                Ok(p) => p,
                Err(e) => return e,
            };
            let n = packet::nb_samples(&data[..p.packet_offset], fs);
            if s != 0 && samples != n {
                return OPUS_INVALID_PACKET;
            }
            samples = n;
            data = &data[p.packet_offset..];
        }
        samples
    }

    /// libopus `opus_multistream_decode_native`. `out(channel, src, n)`
    /// writes `n` samples of a channel from `src` (buffer, stride, offset),
    /// or silence for a muted one.
    unsafe fn decode_native(
        &mut self,
        mut data: *const u8,
        mut len: i32,
        mut frame_size: c_int,
        decode_fec: c_int,
        soft_clip: bool,
        mut out: impl FnMut(usize, Option<(&[f32], usize, usize)>, usize),
    ) -> c_int {
        if frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        let fs = unsafe { (*self.stream(0)).fs() };
        frame_size = frame_size.min(fs / 25 * 3);
        let mut buf = vec![0.0f32; 2 * frame_size as usize];
        let do_plc = len == 0;
        if len < 0 {
            return OPUS_BAD_ARG;
        }
        if !do_plc && len < 2 * self.nb_streams - 1 {
            return OPUS_INVALID_PACKET;
        }
        if !do_plc {
            let ret = self.validate(unsafe { slice::from_raw_parts(data, len as usize) }, fs);
            if ret < 0 {
                return ret;
            } else if ret > frame_size {
                return OPUS_BUFFER_TOO_SMALL;
            }
        }
        for s in 0..self.nb_streams {
            let dec = unsafe { &mut *self.stream(s) };
            if !do_plc && len <= 0 {
                return OPUS_INTERNAL_ERROR;
            }
            let mut packet_offset = 0;
            let ret = unsafe {
                dec.decode_native(
                    data,
                    len,
                    &mut buf,
                    frame_size,
                    decode_fec,
                    s != self.nb_streams - 1,
                    &mut packet_offset,
                    soft_clip,
                )
            };
            if !do_plc {
                data = unsafe { data.add(packet_offset) };
                len -= packet_offset as i32;
            }
            if ret <= 0 {
                return ret;
            }
            frame_size = ret;
            let n = ret as usize;
            if s < self.nb_coupled_streams {
                for c in self.channels_of(s, 0).collect::<Vec<_>>() {
                    out(c, Some((&buf, 2, 0)), n);
                }
                for c in self.channels_of(s, 1).collect::<Vec<_>>() {
                    out(c, Some((&buf, 2, 1)), n);
                }
            } else {
                for c in self.channels_of(s, 0).collect::<Vec<_>>() {
                    out(c, Some((&buf, 1, 0)), n);
                }
            }
        }
        for c in 0..self.nb_channels as usize {
            if self.mapping[c] == 255 {
                out(c, None, frame_size as usize);
            }
        }
        frame_size
    }

    /// Decodes into the interleaved `pcm`, converting each sample.
    unsafe fn decode<T: Copy + Default>(
        &mut self,
        data: *const u8,
        len: i32,
        pcm: *mut T,
        frame_size: c_int,
        decode_fec: c_int,
        soft_clip: bool,
        convert: impl Fn(f32) -> T,
    ) -> c_int {
        let channels = self.nb_channels as usize;
        unsafe {
            self.decode_native(data, len, frame_size, decode_fec, soft_clip, |c, src, n| {
                let dst = slice::from_raw_parts_mut(pcm, n * channels);
                match src {
                    Some((buf, stride, offset)) => {
                        for i in 0..n {
                            dst[i * channels + c] = convert(buf[i * stride + offset]);
                        }
                    }
                    None => {
                        for i in 0..n {
                            dst[i * channels + c] = T::default();
                        }
                    }
                }
            })
        }
    }

    unsafe fn ctl(&mut self, request: c_int, mut ap: VaList) -> c_int {
        unsafe {
            match request {
                // int32 getters ask the first stream.
                req::GET_BANDWIDTH
                | req::GET_SAMPLE_RATE
                | req::GET_GAIN
                | req::GET_LAST_PACKET_DURATION
                | req::GET_PHASE_INVERSION_DISABLED
                | req::GET_COMPLEXITY => {
                    let arg = Arg::Ptr(ap.ptr::<c_void>());
                    (*self.stream(0)).ctl(request, arg)
                }
                req::GET_FINAL_RANGE => {
                    let value = ap.ptr::<u32>();
                    if value.is_null() {
                        return OPUS_BAD_ARG;
                    }
                    *value = 0;
                    for s in 0..self.nb_streams {
                        let mut tmp = 0u32;
                        let ret = (*self.stream(s)).ctl(request, Arg::Ptr(&raw mut tmp as *mut c_void));
                        if ret != OPUS_OK {
                            return ret;
                        }
                        *value ^= tmp;
                    }
                    OPUS_OK
                }
                req::RESET_STATE => {
                    for s in 0..self.nb_streams {
                        let ret = (*self.stream(s)).ctl(request, Arg::Int(0));
                        if ret != OPUS_OK {
                            return ret;
                        }
                    }
                    OPUS_OK
                }
                req::MULTISTREAM_GET_DECODER_STATE => {
                    let stream_id = ap.int();
                    if stream_id < 0 || stream_id >= self.nb_streams {
                        return OPUS_BAD_ARG;
                    }
                    let value = ap.ptr::<*mut OpusDecoder>();
                    if value.is_null() {
                        return OPUS_BAD_ARG;
                    }
                    *value = self.stream(stream_id);
                    OPUS_OK
                }
                req::SET_GAIN | req::SET_COMPLEXITY | req::SET_PHASE_INVERSION_DISABLED => {
                    let arg = Arg::Int(ap.int());
                    for s in 0..self.nb_streams {
                        let ret = (*self.stream(s)).ctl(request, arg);
                        if ret != OPUS_OK {
                            return ret;
                        }
                    }
                    OPUS_OK
                }
                _ => OPUS_UNIMPLEMENTED,
            }
        }
    }
}

fn valid_counts(channels: c_int, streams: c_int, coupled_streams: c_int) -> bool {
    (1..=255).contains(&channels)
        && coupled_streams <= streams
        && streams >= 1
        && coupled_streams >= 0
        && streams <= 255 - coupled_streams
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_multistream_decoder_get_size(nb_streams: c_int, nb_coupled_streams: c_int) -> i32 {
    if nb_streams < 1 || nb_coupled_streams > nb_streams || nb_coupled_streams < 0 {
        return 0;
    }
    let coupled = align_size(opus_decoder_get_size(2) as usize);
    let mono = align_size(opus_decoder_get_size(1) as usize);
    (align_size(size_of::<OpusMSDecoder>())
        + nb_coupled_streams as usize * coupled
        + (nb_streams - nb_coupled_streams) as usize * mono) as i32
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
    if !valid_counts(channels, streams, coupled_streams) {
        return OPUS_BAD_ARG;
    }
    unsafe {
        let st = &mut *st;
        st.nb_channels = channels;
        st.nb_streams = streams;
        st.nb_coupled_streams = coupled_streams;
        ptr::copy_nonoverlapping(mapping, st.mapping.as_mut_ptr(), channels as usize);
        if !st.valid() {
            return OPUS_BAD_ARG;
        }
        for s in 0..streams {
            let ret = opus_decoder_init(st.stream(s), fs, if s < coupled_streams { 2 } else { 1 });
            if ret != OPUS_OK {
                return ret;
            }
        }
    }
    OPUS_OK
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
    unsafe {
        if !valid_counts(channels, streams, coupled_streams) {
            set_out(error, OPUS_BAD_ARG);
            return ptr::null_mut();
        }
        let size = opus_multistream_decoder_get_size(streams, coupled_streams) as usize;
        let st = crate::alloc_state(size) as *mut OpusMSDecoder;
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
    unsafe { crate::free_state(st) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decode(
    st: *mut OpusMSDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut i16,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    guard(OPUS_INTERNAL_ERROR, || unsafe {
        (*st).decode(data, len, pcm, frame_size, decode_fec, true, |x| {
            (x * 32768.0).clamp(-32768.0, 32767.0).round_ties_even() as i16
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decode24(
    st: *mut OpusMSDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut i32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    guard(OPUS_INTERNAL_ERROR, || unsafe {
        (*st).decode(data, len, pcm, frame_size, decode_fec, false, |x| {
            (x * 8388608.0).clamp(-2147483648.0, 2147483520.0).round_ties_even() as i32
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decode_float(
    st: *mut OpusMSDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut f32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    guard(OPUS_INTERNAL_ERROR, || unsafe { (*st).decode(data, len, pcm, frame_size, decode_fec, false, |x| x) })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_multistream_decoder_ctl(st: *mut OpusMSDecoder, request: c_int, args: ...) -> c_int {
    guard(OPUS_INTERNAL_ERROR, || unsafe { (*st).ctl(request, args) })
}
