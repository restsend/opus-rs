//! `opus_decoder_*` / `opus_decode*`, after libopus's opus_decoder.c.

use core::ffi::c_int;
use core::mem::{MaybeUninit, size_of};
use core::ptr;
use core::slice;

use crate::packet::{self, parse};
use crate::repacketizer::undelimit;
use crate::{
    OPUS_ALLOC_FAIL, OPUS_BAD_ARG, OPUS_BUFFER_TOO_SMALL, OPUS_INTERNAL_ERROR, OPUS_INVALID_PACKET,
    OPUS_OK, OPUS_UNIMPLEMENTED, CtlArgs, guard, req, set_out,
};

/// libopus's `OpusDecoder`: the settings the C API keeps beside the opus-rs
/// decoder, which holds all the codec state inline.
#[repr(C)]
pub struct OpusDecoder {
    fs: i32,
    channels: i32,
    decode_gain: i32,
    complexity: i32,
    ignore_extensions: i32,
    phase_inversion_disabled: i32,
    /// The last packet's bandwidth, 0 before any (libopus `bandwidth`).
    bandwidth: i32,
    last_packet_duration: i32,
    softclip_mem: [f32; 2],
    inner: opus_rs::OpusDecoder,
}

pub(crate) fn valid_fs(fs: i32) -> bool {
    matches!(fs, 8000 | 12000 | 16000 | 24000 | 48000)
}

/// Maps an opus-rs decoder error to libopus's code.
fn decode_error(e: &str) -> c_int {
    if e.contains("too small") {
        OPUS_BUFFER_TOO_SMALL
    } else if e.contains("multiple of 2.5 ms") {
        OPUS_BAD_ARG
    } else if e.starts_with("Code") || e.starts_with("Invalid") || e.contains("frame") {
        OPUS_INVALID_PACKET
    } else {
        OPUS_INTERNAL_ERROR
    }
}

impl OpusDecoder {
    /// Writes a fresh decoder (libopus `opus_decoder_init`) to `st`.
    unsafe fn init(st: *mut Self, fs: i32, channels: i32) -> c_int {
        if !valid_fs(fs) || !(1..=2).contains(&channels) {
            return OPUS_BAD_ARG;
        }
        unsafe {
            // Plain fields first, then the codec state straight into place:
            // at ~175 KB it should not go through the stack.
            ptr::write_bytes(st as *mut u8, 0, size_of::<Self>());
            (*st).fs = fs;
            (*st).channels = channels;
            let slot = &mut *(&raw mut (*st).inner as *mut MaybeUninit<opus_rs::OpusDecoder>);
            match opus_rs::OpusDecoder::init(slot, fs, channels as usize) {
                Ok(_) => OPUS_OK,
                Err(_) => OPUS_INTERNAL_ERROR,
            }
        }
    }

    /// libopus `opus_decode_native`, writing up to `frame_size` samples per
    /// channel into `pcm`.
    pub(crate) unsafe fn decode_native(
        &mut self,
        data: *const u8,
        len: i32,
        pcm: &mut [f32],
        frame_size: i32,
        decode_fec: c_int,
        self_delimited: bool,
        packet_offset: &mut usize,
        soft_clip: bool,
    ) -> c_int {
        if !(0..=1).contains(&decode_fec) {
            return OPUS_BAD_ARG;
        }
        // FEC and PLC take whole multiples of 2.5 ms.
        let plc = len == 0 || data.is_null();
        if (decode_fec != 0 || plc) && frame_size % (self.fs / 400) != 0 {
            return OPUS_BAD_ARG;
        }
        let pcm = &mut pcm[..frame_size as usize * self.channels as usize];
        let ret = if plc {
            self.inner.decode(&[], frame_size as usize, pcm)
        } else {
            let Some(packet) = (unsafe { packet::packet(data, len) }) else {
                return OPUS_BAD_ARG;
            };
            let parsed = match parse(packet, self_delimited) {
                Ok(p) => p,
                Err(e) => return e,
            };
            *packet_offset = parsed.packet_offset;
            // opus-rs takes standard packets: rewrite a multistream
            // packet's self-delimited ones.
            let standard;
            let packet = if self_delimited {
                let mut buf = vec![0u8; parsed.packet_offset + 2 * parsed.count + 2];
                match undelimit(&packet[..parsed.packet_offset], &mut buf) {
                    Ok(n) => buf.truncate(n),
                    Err(e) => return e,
                }
                standard = buf;
                &standard[..]
            } else {
                packet
            };
            let packet_frame_size = packet::samples_per_frame(parsed.toc, self.fs);
            if decode_fec != 0 {
                let ret = self.inner.decode_fec(packet, frame_size as usize, pcm);
                if ret.is_ok() && frame_size >= packet_frame_size {
                    self.bandwidth = packet::bandwidth(parsed.toc);
                }
                ret
            } else {
                if parsed.count as i32 * packet_frame_size > frame_size {
                    return OPUS_BUFFER_TOO_SMALL;
                }
                let ret = self.inner.decode(packet, frame_size as usize, pcm);
                if ret.is_ok() {
                    self.bandwidth = packet::bandwidth(parsed.toc);
                }
                ret
            }
        };
        let n = match ret {
            Ok(n) => n,
            Err(e) => return decode_error(e),
        };
        let ch = self.channels as usize;
        if self.decode_gain != 0 {
            // celt_exp2 of the float build, with libopus's constant.
            #[allow(clippy::excessive_precision)]
            let gain = (core::f64::consts::LN_2 * (6.48814081e-4f32 * self.decode_gain as f32) as f64).exp() as f32;
            for s in pcm[..n * ch].iter_mut() {
                *s *= gain;
            }
        }
        // Only decoded packets are soft-clipped; PLC and FEC output is not.
        if soft_clip && !plc && decode_fec == 0 {
            packet::soft_clip(pcm, n, ch, &mut self.softclip_mem);
        } else if !plc && decode_fec == 0 {
            self.softclip_mem = [0.0; 2];
        }
        self.last_packet_duration = n as i32;
        n as c_int
    }

    /// The samples a non-FEC decode of `data` produces at most, as the
    /// int16 and int24 entry points cap `frame_size` with.
    unsafe fn cap_frame_size(&self, data: *const u8, len: i32, frame_size: i32, decode_fec: c_int) -> Result<i32, c_int> {
        if !data.is_null() && len > 0 && decode_fec == 0 {
            let n = packet::nb_samples(unsafe { slice::from_raw_parts(data, len as usize) }, self.fs);
            if n > 0 { Ok(frame_size.min(n)) } else { Err(OPUS_INVALID_PACKET) }
        } else {
            Ok(frame_size)
        }
    }

    /// Decodes into a temporary float buffer and converts with `convert`.
    unsafe fn decode_converted<T>(
        &mut self,
        data: *const u8,
        len: i32,
        pcm: *mut T,
        frame_size: i32,
        decode_fec: c_int,
        soft_clip: bool,
        convert: impl Fn(f32) -> T,
    ) -> c_int {
        if frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        let frame_size = match unsafe { self.cap_frame_size(data, len, frame_size, decode_fec) } {
            Ok(n) => n,
            Err(e) => return e,
        };
        let ch = self.channels as usize;
        let mut out = vec![0.0f32; frame_size as usize * ch];
        let ret = unsafe { self.decode_native(data, len, &mut out, frame_size, decode_fec, false, &mut 0, soft_clip) };
        if ret > 0 {
            let pcm = unsafe { slice::from_raw_parts_mut(pcm, ret as usize * ch) };
            for (d, s) in pcm.iter_mut().zip(&out) {
                *d = convert(*s);
            }
        }
        ret
    }

    fn reset(&mut self) {
        let slot = unsafe { &mut *(&raw mut self.inner as *mut MaybeUninit<opus_rs::OpusDecoder>) };
        let _ = opus_rs::OpusDecoder::init(slot, self.fs, self.channels as usize);
        self.bandwidth = 0;
        self.last_packet_duration = 0;
        self.softclip_mem = [0.0; 2];
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
            match request {
                req::GET_BANDWIDTH => get!(i32, self.bandwidth),
                req::SET_COMPLEXITY => {
                    let v = ap.int();
                    if !(0..=10).contains(&v) {
                        return OPUS_BAD_ARG;
                    }
                    self.complexity = v;
                }
                req::GET_COMPLEXITY => get!(i32, self.complexity),
                req::GET_FINAL_RANGE => get!(u32, self.inner.final_range()),
                req::RESET_STATE => self.reset(),
                req::GET_SAMPLE_RATE => get!(i32, self.fs),
                // opus-rs does not expose the pitch.
                req::GET_PITCH => get!(i32, 0),
                req::GET_GAIN => get!(i32, self.decode_gain),
                req::SET_GAIN => {
                    let v = ap.int();
                    if !(-32768..=32767).contains(&v) {
                        return OPUS_BAD_ARG;
                    }
                    self.decode_gain = v;
                }
                req::GET_LAST_PACKET_DURATION => get!(i32, self.last_packet_duration),
                req::SET_PHASE_INVERSION_DISABLED => {
                    let v = ap.int();
                    if !(0..=1).contains(&v) {
                        return OPUS_BAD_ARG;
                    }
                    self.phase_inversion_disabled = v;
                }
                req::GET_PHASE_INVERSION_DISABLED => get!(i32, self.phase_inversion_disabled),
                req::SET_IGNORE_EXTENSIONS => {
                    let v = ap.int();
                    if !(0..=1).contains(&v) {
                        return OPUS_BAD_ARG;
                    }
                    self.ignore_extensions = v;
                }
                req::GET_IGNORE_EXTENSIONS => get!(i32, self.ignore_extensions),
                _ => return OPUS_UNIMPLEMENTED,
            }
            OPUS_OK
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn opus_decoder_get_size(channels: c_int) -> c_int {
    if !(1..=2).contains(&channels) {
        return 0;
    }
    crate::align_size(size_of::<OpusDecoder>()) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_init(st: *mut OpusDecoder, fs: i32, channels: c_int) -> c_int {
    if st.is_null() {
        return OPUS_BAD_ARG;
    }
    guard(OPUS_INTERNAL_ERROR, || unsafe { OpusDecoder::init(st, fs, channels) })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_create(fs: i32, channels: c_int, error: *mut c_int) -> *mut OpusDecoder {
    unsafe {
        if !valid_fs(fs) || !(1..=2).contains(&channels) {
            set_out(error, OPUS_BAD_ARG);
            return ptr::null_mut();
        }
        let st = crate::alloc_state(opus_decoder_get_size(channels) as usize) as *mut OpusDecoder;
        if st.is_null() {
            set_out(error, OPUS_ALLOC_FAIL);
            return st;
        }
        let ret = opus_decoder_init(st, fs, channels);
        set_out(error, ret);
        if ret != OPUS_OK {
            crate::free_state(st);
            return ptr::null_mut();
        }
        st
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_destroy(st: *mut OpusDecoder) {
    unsafe { crate::free_state(st) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decode(
    st: *mut OpusDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut i16,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    guard(OPUS_INTERNAL_ERROR, || unsafe {
        (*st).decode_converted(data, len, pcm, frame_size, decode_fec, true, |x| {
            (x * 32768.0).clamp(-32768.0, 32767.0).round_ties_even() as i16
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decode24(
    st: *mut OpusDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut i32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    guard(OPUS_INTERNAL_ERROR, || unsafe {
        (*st).decode_converted(data, len, pcm, frame_size, decode_fec, false, |x| {
            (x * 8388608.0).clamp(-2147483648.0, 2147483520.0).round_ties_even() as i32
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decode_float(
    st: *mut OpusDecoder,
    data: *const u8,
    len: i32,
    pcm: *mut f32,
    frame_size: c_int,
    decode_fec: c_int,
) -> c_int {
    guard(OPUS_INTERNAL_ERROR, || unsafe {
        if frame_size <= 0 {
            return OPUS_BAD_ARG;
        }
        let st = &mut *st;
        let out = slice::from_raw_parts_mut(pcm, frame_size as usize * st.channels as usize);
        st.decode_native(data, len, out, frame_size, decode_fec, false, &mut 0, false)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_ctl(st: *mut OpusDecoder, request: c_int, args: ...) -> c_int {
    guard(OPUS_INTERNAL_ERROR, || unsafe { (*st).ctl(request, args) })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn opus_decoder_get_nb_samples(dec: *const OpusDecoder, packet: *const u8, len: i32) -> c_int {
    unsafe { crate::packet::opus_packet_get_nb_samples(packet, len, (*dec).fs) }
}

impl OpusDecoder {
    pub(crate) fn fs(&self) -> i32 {
        self.fs
    }
}
