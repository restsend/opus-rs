#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

mod compat;
mod fixedvec;

pub mod bands;
pub mod celt;
pub mod celt_lpc;
pub mod hp_cutoff;
pub mod kiss_fft;
pub mod mdct;
pub mod modes;
/// Multistream decode uses `Vec`-based plumbing, so it is only available
/// with the `std` (default) feature set; the core codec stays `no_std`
/// + no-`alloc` capable (see `split_self_delimited` docs).
#[cfg(feature = "std")]
pub mod multistream;
pub mod pitch;
pub mod pvq;
pub mod quant_bands;
pub mod range_coder;
pub mod rate;
pub mod silk;

pub use silk::{SilkResampler, SilkResamplerDown1_3, SilkResamplerDown1_6};

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::fixedvec::FixedVec;
pub use celt::{CeltDecoder, CeltEncoder};
use core::mem::MaybeUninit;
use hp_cutoff::{dc_reject_float, hp_cutoff, hp_cutoff_float, hp_cutoff_i16};
use range_coder::RangeCoder;
use silk::control_codec::{
    SilkRateLimits, silk_control_audio_bandwidth, silk_control_encoder, silk_setup_lbrr,
};
use silk::enc_api::{
    silk_encode, silk_encode_prefill, silk_encode_prefill_stereo, silk_encode_stereo_packet,
    silk_reset_for_rate_switch,
};
use silk::init_encoder::silk_init_encoder;
use silk::lin2log::silk_lin2log;
use silk::log2lin::silk_log2lin;
use silk::macros::*;
use silk::structs::SilkEncoderState;

// --- Heap-free buffer capacity constants (worst case: 2 channels). ---
const OPUS_MAX_CHANNELS: usize = 2;
/// Largest API frame in samples/channel (120 ms @ 48 kHz = 5760). Used by the
/// encoder's per-frame input buffers (sized `frame_size * channels`).
const OPUS_MAX_FRAME: usize = 5760;
/// Largest SILK packet at the internal rate: 60 ms at 16 kHz.
const SILK_MAX_PACKET_SAMPLES: usize = 3 * silk::define::MAX_FRAME_LENGTH;
/// Largest *single-frame* samples/channel (60 ms @ 48 kHz = 2880). The decoder's
/// staging buffers hold one sub-frame at a time, so they're sized to this — not
/// the full packet. (Halves the decoder footprint vs. a naive 5760/channel.)
const OPUS_MAX_SUBFRAME: usize = 2880;
/// Decoder per-sub-frame staging cap: `OPUS_MAX_SUBFRAME * max_channels`.
const OPUS_SUBFRAME_SCRATCH: usize = OPUS_MAX_SUBFRAME * OPUS_MAX_CHANNELS;
/// High-pass filter state memory (`channels * 2`).
const OPUS_HP_MEM: usize = OPUS_MAX_CHANNELS * 2;
/// Decoder `w_pcm_i16` cap (`960 * max_channels`).
const OPUS_PCM_I16: usize = 960 * OPUS_MAX_CHANNELS;
/// Decoder `prev_pcm_tail` cap (`240 * max_channels`).
const OPUS_PCM_TAIL: usize = 240 * OPUS_MAX_CHANNELS;
/// Max number of frames encoded in one Opus packet (RFC 6716 caps at 48 for
/// 2.5 ms codes in a 120 ms packet).
const OPUS_MAX_PACKET_FRAMES: usize = 48;
/// RFC 6716 §3.1: a single Opus packet carries at most 1276 bytes of data.
const OPUS_MAX_PACKET_BYTES: usize = 1276;
/// C `OpusEncoder.delay_buffer[MAX_ENCODER_BUFFER*2]` (480 samples * 2 ch).
/// Holds the delay-compensation ring feeding the CELT encoder.
const OPUS_DELAY_BUF: usize = 480 * OPUS_MAX_CHANNELS;
/// CELT/hybrid input staging cap: `(delay_compensation + frame)*channels`.
/// Worst case 60 ms stereo @ 48 kHz: (192 + 2880) * 2 = 6144.
const OPUS_PCM_BUF: usize = (192 + OPUS_MAX_SUBFRAME) * OPUS_MAX_CHANNELS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Application {
    Voip = 2048,
    Audio = 2049,
    RestrictedLowDelay = 2051,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bandwidth {
    Auto = -1000,
    Narrowband = 1101,
    Mediumband = 1102,
    Wideband = 1103,
    Superwideband = 1104,
    Fullband = 1105,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpusMode {
    SilkOnly,
    Hybrid,
    CeltOnly,
}

pub struct OpusEncoder {
    #[cfg(not(feature = "heap"))]
    celt_enc: CeltEncoder,
    #[cfg(feature = "heap")]
    celt_enc: Box<CeltEncoder>,
    /// SILK encoder for mono input, or for the mid of stereo input (it also
    /// holds the stereo state).
    #[cfg(not(feature = "heap"))]
    silk_enc: SilkEncoderState,
    #[cfg(feature = "heap")]
    silk_enc: Box<SilkEncoderState>,
    /// SILK encoder for the side of stereo input (libopus `state_Fxx[1]`).
    #[cfg(not(feature = "heap"))]
    silk_enc_side: SilkEncoderState,
    #[cfg(feature = "heap")]
    silk_enc_side: Box<SilkEncoderState>,
    application: Application,
    sampling_rate: i32,
    channels: usize,
    bandwidth: Bandwidth,
    pub bitrate_bps: i32,
    pub complexity: i32,
    pub use_cbr: bool,

    pub use_inband_fec: bool,

    pub packet_loss_perc: i32,
    silk_initialized: bool,
    mode: OpusMode,
    prev_enc_mode: Option<OpusMode>,
    /// When set, `encode_impl` uses this mode instead of running the
    /// mode-selection heuristic. Used to keep all sub-frames of a multiframe
    /// packet in the same mode (libopus `user_forced_mode`).
    forced_mode: Option<OpusMode>,
    /// The mode-switch decisions for the next sub-frame of a multiframe
    /// packet, which libopus makes once for the whole packet.
    frame_transition: Option<FrameTransition>,

    variable_hp_smth2_q15: i32,
    hp_mem: FixedVec<i32, OPUS_HP_MEM>,

    #[cfg(not(feature = "heap"))]
    buf_filtered: FixedVec<i16, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_filtered: Box<FixedVec<i16, OPUS_MAX_FRAME>>,
    #[cfg(not(feature = "heap"))]
    buf_silk_input: FixedVec<i16, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_silk_input: Box<FixedVec<i16, OPUS_MAX_FRAME>>,
    /// Right channel of a stereo frame at the SILK internal rate, beside the
    /// left in `buf_silk_input`.
    #[cfg(not(feature = "heap"))]
    buf_silk_input_right: FixedVec<i16, SILK_MAX_PACKET_SAMPLES>,
    #[cfg(feature = "heap")]
    buf_silk_input_right: Box<FixedVec<i16, SILK_MAX_PACKET_SAMPLES>>,
    /// One channel of a stereo frame at the API rate, de-interleaved for its
    /// SILK resampler.
    #[cfg(not(feature = "heap"))]
    buf_silk_api_channel: FixedVec<i16, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_silk_api_channel: Box<FixedVec<i16, OPUS_MAX_FRAME>>,
    #[cfg(not(feature = "heap"))]
    buf_celt_input: FixedVec<f32, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_celt_input: Box<FixedVec<f32, OPUS_MAX_FRAME>>,
    /// C `delay_buffer[MAX_ENCODER_BUFFER*2]`: delay-compensation ring feeding
    /// the CELT encoder (see `encoder_buffer` / `delay_compensation`).
    #[cfg(not(feature = "heap"))]
    delay_buffer: FixedVec<f32, OPUS_DELAY_BUF>,
    #[cfg(feature = "heap")]
    delay_buffer: Box<FixedVec<f32, OPUS_DELAY_BUF>>,
    /// Interleaved `pcm_buf` staging: `delay prefix + filtered frame` exactly
    /// like C `opus_encode_frame_native()` builds it before CELT encoding.
    #[cfg(not(feature = "heap"))]
    buf_celt_pcm: FixedVec<f32, OPUS_PCM_BUF>,
    #[cfg(feature = "heap")]
    buf_celt_pcm: Box<FixedVec<f32, OPUS_PCM_BUF>>,
    /// f32 staging of the current frame for the CELT/Hybrid float pipeline
    /// when the caller feeds `encode_i16()` (issue #28).
    #[cfg(not(feature = "heap"))]
    buf_f32_frame: FixedVec<f32, OPUS_MAX_FRAME>,
    #[cfg(feature = "heap")]
    buf_f32_frame: Box<FixedVec<f32, OPUS_MAX_FRAME>>,
    /// C `encoder_buffer` (= Fs/100): samples of ring history kept between
    /// frames (0 for restricted-latency applications).
    encoder_buffer: usize,
    /// C `delay_compensation` (= Fs/250): 4 ms lookahead prefix prepended to
    /// each CELT frame (0 for restricted-latency applications).
    delay_compensation: usize,
    /// Float filter state for `dc_reject_float` / `hp_cutoff_float`
    /// (C `st->hp_mem[4]`, opus_val32 in the float build).
    hp_mem_float: [f32; 4],
    prev_hb_gain: f32,
    /// libopus `silk_mode.opusCanSwitch` and `silk_bw_switch`: SILK asked to
    /// change its internal rate, which the next packet does.
    silk_opus_can_switch: bool,
    silk_bw_switch: bool,
    /// The 16-bit SILK input of the last 10 ms (libopus `delay_buffer` in the
    /// fixed-point build), which a SILK prefill encodes.
    #[cfg(not(feature = "heap"))]
    silk_delay_buffer: FixedVec<i16, OPUS_DELAY_BUF>,
    #[cfg(feature = "heap")]
    silk_delay_buffer: Box<FixedVec<i16, OPUS_DELAY_BUF>>,
    /// Encoder-direction SILK input resampler (C `resampler_state` with
    /// `forEnc = 1`): API rate -> SILK internal rate, carrying the
    /// `delay_matrix_enc` input delay and the private down-FIR chain. It
    /// resamples mono input, or the left channel of stereo input.
    silk_resampler_enc: silk::resampler::SilkResampler,
    /// The right channel's SILK input resampler: libopus resamples each
    /// stereo channel separately and splits mid/side at the internal rate.
    silk_resampler_enc_right: silk::resampler::SilkResampler,

    rc: RangeCoder,
    /// Range encoder state after the last packet (libopus `rangeFinal`), 0
    /// when no frame was coded.
    range_final: u32,
}

/// Format-agnostic PCM input for `encode_impl` (issue #28): `F32` is the
/// classic float path; `I16` reaches the SILK integer pipeline directly
/// and is converted to f32 only where the CELT pipeline requires it.
#[derive(Clone, Copy)]
enum PcmInput<'a> {
    F32(&'a [f32]),
    I16(&'a [i16]),
}

impl PcmInput<'_> {
    fn len(&self) -> usize {
        match self {
            PcmInput::F32(s) => s.len(),
            PcmInput::I16(s) => s.len(),
        }
    }
}

/// How a frame takes part in a switch into or out of CELT-only mode
/// (opus_encoder.c:1540-1581). `redundancy` adds a 5 ms CELT frame: at the
/// start of the frame when `celt_to_silk`, else at its end. `to_celt` keeps
/// the previous mode for this frame and makes the next one CELT-only.
/// `prefill` primes a SILK encoder restarted after CELT.
#[derive(Clone, Copy, Default)]
struct FrameTransition {
    redundancy: bool,
    celt_to_silk: bool,
    to_celt: bool,
    prefill: bool,
}

fn compute_equiv_rate(
    bitrate: i32,
    channels: usize,
    frame_rate: i32,
    vbr: bool,
    mode: Option<OpusMode>,
    complexity: i32,
    loss: i32,
) -> i32 {
    let mut equiv = bitrate;
    if frame_rate > 50 {
        equiv -= (40 * channels as i32 + 20) * (frame_rate - 50);
    }
    if !vbr {
        equiv -= equiv / 12;
    }
    equiv = equiv * (90 + complexity) / 100;
    if matches!(mode, Some(OpusMode::SilkOnly | OpusMode::Hybrid)) {
        if complexity < 2 {
            equiv = equiv * 4 / 5;
        }
        equiv -= equiv * loss / (6 * loss + 10);
    } else if mode == Some(OpusMode::CeltOnly) {
        if complexity < 5 {
            equiv = equiv * 9 / 10;
        }
    } else {
        equiv -= equiv * loss / (12 * loss + 20);
    }
    equiv
}

#[cfg(test)]
mod fec_rate_tests {
    use super::{OpusMode, compute_equiv_rate};

    #[test]
    fn fec_uses_the_selected_modes_loss_cost() {
        assert_eq!(compute_equiv_rate(20000, 1, 50, true, None, 9, 10), 18386);
        for mode in [OpusMode::SilkOnly, OpusMode::Hybrid] {
            assert_eq!(
                compute_equiv_rate(20000, 1, 50, true, Some(mode), 9, 10),
                16972
            );
            assert_eq!(
                compute_equiv_rate(16000, 2, 50, false, Some(mode), 9, 40),
                12197
            );
        }
        assert_eq!(
            compute_equiv_rate(20000, 1, 50, true, Some(OpusMode::CeltOnly), 9, 10),
            19800,
        );
    }
}

/// `fec_thresholds` (opus_encoder.c:160): LBRR rate threshold and hysteresis
/// per bandwidth, starting at narrowband.
const FEC_THRESHOLDS: [i32; 10] = [
    12000, 1000, // NB
    14000, 1000, // MB
    16000, 1000, // WB
    20000, 1000, // SWB
    22000, 1000, // FB
];

fn bandwidth_fec_index(bw: Bandwidth) -> usize {
    match bw {
        Bandwidth::Narrowband => 0,
        Bandwidth::Mediumband => 1,
        Bandwidth::Wideband => 2,
        Bandwidth::Superwideband => 3,
        Bandwidth::Fullband | Bandwidth::Auto => 4,
    }
}

fn bandwidth_from_fec_index(idx: usize) -> Bandwidth {
    match idx {
        0 => Bandwidth::Narrowband,
        1 => Bandwidth::Mediumband,
        2 => Bandwidth::Wideband,
        3 => Bandwidth::Superwideband,
        _ => Bandwidth::Fullband,
    }
}

/// Port of opus_encoder.c `decide_fec()`: gate in-band FEC on the packet loss
/// setting and on whether the equivalent rate can afford LBRR at the current
/// bandwidth. With loss > 5%, the bandwidth is reduced until FEC fits; the
/// caller's per-frame bandwidth is updated accordingly.
fn decide_fec(
    use_in_band_fec: i32,
    packet_loss_perc: i32,
    last_fec: i32,
    mode: OpusMode,
    bandwidth: &mut Bandwidth,
    rate: i32,
) -> i32 {
    if use_in_band_fec == 0 || packet_loss_perc == 0 || mode == OpusMode::CeltOnly {
        return 0;
    }
    let orig_bandwidth = *bandwidth;
    loop {
        let idx = bandwidth_fec_index(*bandwidth);
        let mut lbrr_rate_thres_bps = FEC_THRESHOLDS[2 * idx];
        let hysteresis = FEC_THRESHOLDS[2 * idx + 1];
        if last_fec == 1 {
            lbrr_rate_thres_bps -= hysteresis;
        }
        if last_fec == 0 {
            lbrr_rate_thres_bps += hysteresis;
        }
        // silk_SMULWB(threshold * (125 - min(loss, 25)), 0.01 in Q16)
        lbrr_rate_thres_bps = silk_smulwb(
            lbrr_rate_thres_bps * (125 - packet_loss_perc.min(25)),
            655,
        );
        // If loss <= 5%, we look at whether we have enough rate to enable FEC.
        // If loss > 5%, we decrease the bandwidth until we can enable FEC.
        if rate > lbrr_rate_thres_bps {
            return 1;
        } else if packet_loss_perc <= 5 {
            return 0;
        } else if bandwidth_fec_index(*bandwidth) > 0 {
            let lower = bandwidth_fec_index(*bandwidth) - 1;
            *bandwidth = bandwidth_from_fec_index(lower);
        } else {
            break;
        }
    }
    // Couldn't find any bandwidth to enable FEC, keep original bandwidth.
    *bandwidth = orig_bandwidth;
    0
}

fn compute_mode_threshold(
    application: Application,
    channels: usize,
    prev_was_celt: bool,
    has_prev_mode: bool,
    voice_est: i32,
) -> i32 {
    let mode_voice = if channels == 1 { 64000 } else { 44000 };
    let mode_music = 10000;

    let diff = mode_voice - mode_music;
    let offset = (voice_est * voice_est * diff) >> 14;
    let mut threshold = mode_music + offset;

    if application == Application::Voip {
        threshold += 8000;
    }

    if has_prev_mode {
        if prev_was_celt {
            threshold -= 4000;
        } else {
            threshold += 4000;
        }
    }

    if application == Application::RestrictedLowDelay {
        threshold = 0;
    }

    threshold
}

/// C `bits_to_bitrate` (opus_private.h, libopus 1.6): bits per frame to b/s,
/// through the 6x frame rate so 60 ms frames (16.67 Hz) round like libopus.
fn bits_to_bitrate(bits: i32, fs: i32, frame_size: usize) -> i32 {
    (bits as i64 * (6 * fs as i64 / frame_size as i64) / 6) as i32
}

/// C `bitrate_to_bits` (opus_private.h, libopus 1.6): b/s to bits per frame.
fn bitrate_to_bits(bitrate: i32, fs: i32, frame_size: usize) -> i32 {
    (bitrate as i64 * 6 / (6 * fs as i64 / frame_size as i64)) as i32
}

/// SILK's share of a hybrid packet, following libopus's per-channel table.
fn compute_silk_rate_for_hybrid(
    rate_bps: i32,
    bandwidth: Bandwidth,
    frame20ms: bool,
    vbr: bool,
    fec: bool,
    channels: usize,
) -> i32 {
    const RATE_TABLE: [[i32; 5]; 7] = [
        [0, 0, 0, 0, 0],
        [12000, 10000, 10000, 11000, 11000],
        [16000, 13500, 13500, 15000, 15000],
        [20000, 16000, 16000, 18000, 18000],
        [24000, 18000, 18000, 21000, 21000],
        [32000, 22000, 22000, 28000, 28000],
        [64000, 38000, 38000, 50000, 50000],
    ];
    let channels = channels as i32;
    let rate_bps = rate_bps / channels;
    let entry = 1 + usize::from(frame20ms) + 2 * usize::from(fec);
    let n = RATE_TABLE.len();
    let mut i = 1;
    while i < n && RATE_TABLE[i][0] <= rate_bps {
        i += 1;
    }
    let mut silk_rate = if i == n {
        RATE_TABLE[n - 1][entry] + (rate_bps - RATE_TABLE[n - 1][0]) / 2
    } else {
        let [x0, ..] = RATE_TABLE[i - 1];
        let [x1, ..] = RATE_TABLE[i];
        let lo = RATE_TABLE[i - 1][entry];
        let hi = RATE_TABLE[i][entry];
        (lo * (x1 - rate_bps) + hi * (rate_bps - x0)) / (x1 - x0)
    };
    if !vbr {
        silk_rate += 100;
    }
    if bandwidth == Bandwidth::Superwideband {
        silk_rate += 300;
    }
    silk_rate *= channels;
    if channels == 2 && rate_bps >= 12000 {
        silk_rate -= 1000;
    }
    silk_rate
}

/// The CELT encoder's bitrate for a frame. libopus sets it to
/// `OPUS_BITRATE_MAX` for every frame and, under VBR only, to the requested
/// rate, which in hybrid mode is what the SILK share leaves
/// (opus_encoder.c:2286, 2448-2464). CELT's `OPUS_SET_BITRATE` refuses rates
/// of 500 b/s or less, so CELT then keeps the unlimited rate, and caps the
/// rate at 750 kb/s per channel (celt_encoder.c:3001-3009).
fn celt_bitrate(mode: OpusMode, vbr: bool, bitrate: i32, silk_rate: i32, channels: usize) -> i32 {
    let requested = if mode == OpusMode::Hybrid {
        bitrate - silk_rate
    } else {
        bitrate
    };
    if !vbr || requested <= 500 {
        celt::OPUS_BITRATE_MAX
    } else {
        requested.min(750000 * channels as i32)
    }
}

/// The size of a mode switch's 5 ms CELT redundancy frame, or 0 when too few
/// bytes are left for it to beat the decoder's concealment (libopus
/// `compute_redundancy_bytes`, opus_encoder.c:1142-1168).
fn compute_redundancy_bytes(
    max_data_bytes: usize,
    bitrate_bps: i32,
    frame_rate: i32,
    channels: usize,
) -> usize {
    let base_bits = 40 * channels as i32 + 20;
    // The equivalent rate of 5 ms frames, raised by half: the frame is short.
    let rate = 3 * (bitrate_bps + base_bits * (200 - frame_rate)) / 2;
    let available_bits = max_data_bytes as i32 * 8 - 2 * base_bits;
    let cap = (available_bits * 240 / (240 + 48000 / frame_rate) + base_bits) / 8;
    let bytes = (rate / 1600).min(cap);
    if bytes > 4 + 8 * channels as i32 {
        bytes.min(257) as usize
    } else {
        0
    }
}

/// CELT-codes `n` samples per channel of interleaved `pcm`, from band
/// `start_band`, as a frame of `out.len()` bytes: the redundancy frames and
/// prefills of a mode switch, which libopus codes with a range coder of
/// their own (`celt_encode_with_ec` without one). `vbr` says whether the
/// encoder's rate settings make the frame VBR. Returns the frame's final
/// range.
fn encode_celt_side_frame(
    celt: &mut CeltEncoder,
    pcm: &[f32],
    n: usize,
    ch: usize,
    start_band: usize,
    vbr: bool,
    out: &mut [u8],
) -> u32 {
    // At most 5 ms of stereo at 48 kHz.
    let mut planar = [0.0f32; 480];
    for i in 0..n {
        for c in 0..ch {
            planar[c * n + i] = pcm[i * ch + c];
        }
    }
    let bytes = out.len();
    let mut rc = RangeCoder::new_encoder(bytes as u32);
    celt.encode_with_budget_vbr(
        &planar[..n * ch],
        n,
        &mut rc,
        start_band,
        bytes as i32 * 8,
        vbr,
    );
    rc.done();
    out.copy_from_slice(&rc.buf[..bytes]);
    rc.rng
}

fn hybrid_hb_gain(celt_rate: i32) -> f32 {
    1.0 - 0.5 * (-(celt_rate as f32) / 1024.0).exp2()
}

/// libopus `pcm_buf` and `delay_buffer` (opus_encoder.c:1966-2028,
/// 2297-2312): `pcm` receives the delayed history, then the high-passed
/// frame, and the delay line moves on by the frame.
#[allow(clippy::too_many_arguments)]
fn push_delay_history(
    delay_buffer: &mut [f32],
    pcm: &mut [f32],
    hp_mem: &mut [f32; 4],
    input: &[f32],
    voip: bool,
    cutoff_hz: i32,
    frame_size: usize,
    ch: usize,
    fs: i32,
) {
    let ebuf = delay_buffer.len() / ch;
    let delay = pcm.len() / ch - frame_size;
    // 1. the delayed history
    let prefix = delay * ch;
    pcm[..prefix].copy_from_slice(&delay_buffer[(ebuf - delay) * ch..ebuf * ch]);
    // 2. the high-passed frame
    let out = &mut pcm[prefix..];
    if voip {
        hp_cutoff_float(input, cutoff_hz, out, hp_mem, frame_size, ch, fs);
    } else {
        dc_reject_float(input, 3, out, hp_mem, frame_size, ch, fs);
    }
    // 3. NaN guard, in libopus's shape (`!(sum < 1e9f) || celt_isnan(sum)`).
    let sum: f32 = pcm[prefix..].iter().map(|v| v * v).sum();
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum < 1e9) || sum.is_nan() {
        pcm[prefix..].fill(0.0);
        *hp_mem = [0.0; 4];
    }
    // 4. the delay line
    let n = (frame_size + delay) * ch;
    if ebuf > frame_size + delay {
        let keep = ebuf - frame_size - delay;
        delay_buffer.copy_within(ch * frame_size..ch * (frame_size + keep), 0);
        delay_buffer[ch * keep..ch * keep + n].copy_from_slice(&pcm[..n]);
    } else {
        delay_buffer.copy_from_slice(&pcm[n - ebuf * ch..n]);
    }
}

/// Sample `k` of the 2.5 ms fade-in that starts libopus's SILK prefill:
/// gain_fade from 0 to Q15ONE with the squared CELT window
/// (opus_encoder.c:2202-2203), in the fixed-point arithmetic of the 16-bit
/// SILK input.
fn prefill_fade(x: i16, k: usize, fs: i32) -> i16 {
    let window = modes::default_mode().window;
    let inc = (48000 / fs).max(1) as usize;
    match window.get(k * inc) {
        Some(&w) => {
            // The Q15 window of the fixed-point build.
            let w = ((f64::from(w) * 32768.0 + 0.5) as i32).min(32767);
            let w = (w * w) >> 15;
            let g = (w * 32767) >> 15;
            ((g * i32::from(x)) >> 15) as i16
        }
        None => x,
    }
}

/// libopus `silk_setup_resamplers` for a running channel encoder whose
/// internal rate changes to `new_khz` (control_codec.c:146-187): its
/// buffered input is resampled to the API rate, and back through the input
/// resampler, set up afresh for the new rate, which leaves both the buffer
/// and the resampler state at the new rate.
fn silk_move_history(
    enc: &mut SilkEncoderState,
    resampler: &mut silk::resampler::SilkResampler,
    api_hz: i32,
    new_khz: i32,
) -> Result<(), &'static str> {
    let buf_ms = (enc.s_cmn.nb_subfr * 10) as usize + silk::define::LA_SHAPE_MS;
    let old_samples = buf_ms * enc.s_cmn.fs_khz as usize;
    let api_samples = buf_ms * (api_hz / 1000) as usize;
    let new_samples = buf_ms * new_khz as usize;
    // At most 45 ms at 48 kHz.
    let mut api = [0i16; 45 * 48];
    let api = &mut api[..api_samples];
    let mut temp = silk::resampler::SilkResampler::default();
    if temp.init(enc.s_cmn.fs_khz * 1000, api_hz) != 0
        || temp.process(api, &enc.s_cmn.x_buf[..old_samples], old_samples as i32) != 0
        || resampler.init_for_enc(api_hz, new_khz * 1000) != 0
        || resampler.process(&mut enc.s_cmn.x_buf[..new_samples], api, api_samples as i32) != 0
    {
        return Err("Unsupported SILK resampling ratio");
    }
    Ok(())
}

/// Move the history of the 16-bit SILK input on by `frame`: it keeps the
/// latest `history.len()` samples.
fn push_silk_history(history: &mut [i16], frame: &[i16]) {
    let n = history.len();
    if frame.len() >= n {
        history.copy_from_slice(&frame[frame.len() - n..]);
    } else {
        history.copy_within(frame.len().., 0);
        history[n - frame.len()..].copy_from_slice(frame);
    }
}

fn gain_fade(pcm: &mut [f32], previous: f32, gain: f32, channels: usize, fs: i32) {
    let window = modes::default_mode().window;
    let inc = 48000 / fs as usize;
    let overlap = window.len() / inc;
    for (i, samples) in pcm.chunks_exact_mut(channels).enumerate() {
        let g = if i < overlap {
            let w = window[i * inc] * window[i * inc];
            w * gain + (1.0 - w) * previous
        } else {
            gain
        };
        for sample in samples {
            *sample *= g;
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod silk_rate_tests {
    use super::{Bandwidth, compute_silk_rate_for_hybrid, gain_fade, hybrid_hb_gain};

    #[test]
    fn test_reference_table_exact_entries() {
        assert_eq!(
            compute_silk_rate_for_hybrid(12000, Bandwidth::Wideband, true, true, false, 1),
            10000
        );
        assert_eq!(
            compute_silk_rate_for_hybrid(16000, Bandwidth::Wideband, true, true, false, 1),
            13500
        );
        assert_eq!(
            compute_silk_rate_for_hybrid(20000, Bandwidth::Wideband, true, true, false, 1),
            16000
        );
        assert_eq!(
            compute_silk_rate_for_hybrid(24000, Bandwidth::Wideband, true, true, false, 1),
            18000
        );
        assert_eq!(
            compute_silk_rate_for_hybrid(32000, Bandwidth::Wideband, true, true, false, 1),
            22000
        );
        assert_eq!(
            compute_silk_rate_for_hybrid(64000, Bandwidth::Wideband, true, true, false, 1),
            38000
        );
    }

    #[test]
    fn test_32kbps_gives_22kbps_silk() {
        assert_eq!(
            compute_silk_rate_for_hybrid(32000, Bandwidth::Wideband, true, true, false, 1),
            22000
        );
    }

    #[test]
    fn test_interpolation_between_table_entries() {
        let r = compute_silk_rate_for_hybrid(18000, Bandwidth::Wideband, true, true, false, 1);
        assert_eq!(r, 14750);
    }

    #[test]
    fn test_above_table_max_gives_half_extra() {
        let r = compute_silk_rate_for_hybrid(72000, Bandwidth::Wideband, true, true, false, 1);
        assert_eq!(r, 38000 + (72000 - 64000) / 2);
    }

    /// Stereo looks the rate up per channel, doubles it and, from 12 kb/s per
    /// channel, takes 1 kb/s off (libopus opus_encoder.c).
    #[test]
    fn test_stereo_allocates_per_channel() {
        let stereo =
            |rate| compute_silk_rate_for_hybrid(rate, Bandwidth::Wideband, true, true, false, 2);
        assert_eq!(stereo(32000), 2 * 13500 - 1000);
        assert_eq!(stereo(24000), 2 * 10000 - 1000);
        assert_eq!(stereo(20000), 2 * 8333);
        assert_eq!(stereo(64000), 2 * 22000 - 1000);
    }

    #[test]
    fn fec_columns_and_rate_boosts_match_libopus() {
        for frame20ms in [false, true] {
            for (rate, expected) in [
                (12000, 11000),
                (16000, 15000),
                (18000, 16500),
                (20000, 18000),
                (24000, 21000),
                (32000, 28000),
                (64000, 50000),
                (72000, 54000),
            ] {
                for vbr in [false, true] {
                    for bandwidth in [Bandwidth::Fullband, Bandwidth::Superwideband] {
                        let boost = i32::from(!vbr) * 100
                            + i32::from(bandwidth == Bandwidth::Superwideband) * 300;
                        assert_eq!(
                            compute_silk_rate_for_hybrid(rate, bandwidth, frame20ms, vbr, true, 1),
                            expected + boost,
                        );
                        assert_eq!(
                            compute_silk_rate_for_hybrid(
                                rate * 2,
                                bandwidth,
                                frame20ms,
                                vbr,
                                true,
                                2
                            ),
                            (expected + boost) * 2 - 1000,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn celt_rate_follows_libopus_bitrate_ctl() {
        use super::{OpusMode, celt_bitrate};
        let max = crate::celt::OPUS_BITRATE_MAX;
        assert_eq!(celt_bitrate(OpusMode::Hybrid, true, 12000, 9966, 1), 2034);
        // At or below 500 b/s CELT keeps the unlimited rate.
        assert_eq!(celt_bitrate(OpusMode::Hybrid, true, 2500, 2050, 1), max);
        assert_eq!(celt_bitrate(OpusMode::Hybrid, true, 2500, 2000, 1), max);
        assert_eq!(celt_bitrate(OpusMode::Hybrid, true, 2501, 2000, 1), 501);
        // CBR never sets a CELT rate.
        assert_eq!(celt_bitrate(OpusMode::Hybrid, false, 32000, 22000, 1), max);
        assert_eq!(celt_bitrate(OpusMode::CeltOnly, false, 64000, 0, 1), max);
        assert_eq!(celt_bitrate(OpusMode::CeltOnly, true, 64000, 0, 2), 64000);
        assert_eq!(
            celt_bitrate(OpusMode::CeltOnly, true, 2_000_000, 0, 2),
            1_500_000
        );
        assert_eq!(celt_bitrate(OpusMode::CeltOnly, true, 500, 0, 1), max);
    }

    #[test]
    fn redundancy_size_follows_libopus() {
        use super::compute_redundancy_bytes;
        // (packet bytes, b/s, frames/s, channels, libopus's size)
        for (bytes, rate, frame_rate, channels, expected) in [
            (1276, 20000, 50, 1, 27),
            (50, 20000, 50, 1, 14),
            (15, 6000, 50, 1, 0),
            (1276, 100000, 100, 2, 103),
            (1276, 500000, 50, 1, 257),
            (1276, 32000, 25, 1, 39),
            (125, 100000, 50, 1, 29),
            (25, 20000, 100, 1, 0),
            (60, 24000, 50, 2, 0),
            (1276, 8000, 50, 1, 15),
        ] {
            assert_eq!(
                compute_redundancy_bytes(bytes, rate, frame_rate, channels),
                expected,
                "{bytes} bytes, {rate} b/s, {frame_rate} frames/s, {channels} channel(s)"
            );
        }
    }

    #[test]
    fn high_band_gain_follows_the_celt_share() {
        assert_eq!(hybrid_hb_gain(0), 0.5);
        assert_eq!(hybrid_hb_gain(1024), 0.75);
        assert_eq!(hybrid_hb_gain(2048), 0.875);
        assert!(hybrid_hb_gain(8000) > 0.997);
    }

    #[test]
    fn high_band_gain_fades_over_the_overlap_at_each_api_rate() {
        for fs in [24000, 48000] {
            let overlap = fs as usize / 400;
            for channels in [1, 2] {
                let mut pcm = vec![1.0; fs as usize / 100 * channels];
                gain_fade(&mut pcm, 1.0, 0.75, channels, fs);
                assert!(pcm[0] > 0.999);
                assert_eq!(pcm[overlap * channels], 0.75);
                for samples in pcm.chunks_exact(channels) {
                    assert!(samples.iter().all(|&sample| sample == samples[0]));
                }
                assert!(pcm.windows(channels + 1).all(|w| w[0] >= w[channels]));
                gain_fade(&mut pcm, 0.75, 1.0, channels, fs);
                assert_eq!(pcm[overlap * channels], 0.75);
            }
        }
    }
}

/// Uniformly borrow a codec-state field that is a `Box<T>` under the `heap`
/// feature and a plain `T` otherwise, so call sites don't need `#[cfg]` on
/// every `&self.field` / `&mut self.field`.
///
/// Under `heap` the field is a `Box<...>`, so `T: Deref(DerefMut)` resolves to
/// the box itself and the `Target` is the inner type; without `heap` the field
/// is the inner type directly (`?Sized` lets `&mut [i16]`-style coercion choose
/// the slice target).
#[cfg(feature = "heap")]
#[inline(always)]
fn state_ref<T: core::ops::Deref>(b: &T) -> &T::Target {
    b
}
#[cfg(not(feature = "heap"))]
#[inline(always)]
fn state_ref<T: ?Sized>(b: &T) -> &T {
    b
}

#[cfg(feature = "heap")]
#[inline(always)]
fn state_mut<T: core::ops::DerefMut>(b: &mut T) -> &mut T::Target {
    b
}
#[cfg(not(feature = "heap"))]
#[inline(always)]
fn state_mut<T: ?Sized>(b: &mut T) -> &mut T {
    b
}

impl OpusEncoder {
    pub fn new(
        sampling_rate: i32,
        channels: usize,
        application: Application,
    ) -> Result<Self, &'static str> {
        if ![8000, 12000, 16000, 24000, 48000].contains(&sampling_rate) {
            return Err("Invalid sampling rate");
        }
        if ![1, 2].contains(&channels) {
            return Err("Invalid number of channels");
        }

        let mode = modes::default_mode();
        #[cfg(feature = "heap")]
        let celt_enc = Box::new(CeltEncoder::with_sampling_rate(mode, channels, sampling_rate));
        #[cfg(not(feature = "heap"))]
        let celt_enc = CeltEncoder::with_sampling_rate(mode, channels, sampling_rate);

        #[cfg(feature = "heap")]
        let mut silk_enc = Box::new(SilkEncoderState::default());
        #[cfg(not(feature = "heap"))]
        let mut silk_enc = SilkEncoderState::default();
        if silk_init_encoder(state_mut(&mut silk_enc), 0) != 0 {
            return Err("SILK encoder initialization failed");
        }
        #[cfg(feature = "heap")]
        let mut silk_enc_side = Box::new(SilkEncoderState::default());
        #[cfg(not(feature = "heap"))]
        let mut silk_enc_side = SilkEncoderState::default();
        if silk_init_encoder(state_mut(&mut silk_enc_side), 0) != 0 {
            return Err("SILK encoder initialization failed");
        }

        let (opus_mode, bw) = match application {
            Application::Voip => {
                let bw = match sampling_rate {
                    8000 => Bandwidth::Narrowband,
                    12000 => Bandwidth::Mediumband,
                    16000 => Bandwidth::Wideband,
                    24000 => Bandwidth::Superwideband,
                    48000 => Bandwidth::Fullband,
                    _ => Bandwidth::Narrowband,
                };

                let mode = if sampling_rate > 16000 {
                    OpusMode::Hybrid
                } else {
                    OpusMode::SilkOnly
                };
                (mode, bw)
            }
            Application::RestrictedLowDelay => {
                let bw = match sampling_rate {
                    8000 => Bandwidth::Narrowband,
                    12000 => Bandwidth::Mediumband,
                    16000 => Bandwidth::Wideband,
                    24000 => Bandwidth::Superwideband,
                    _ => Bandwidth::Fullband,
                };
                (OpusMode::CeltOnly, bw)
            }
            Application::Audio => {
                if sampling_rate <= 16000 {
                    let bw = match sampling_rate {
                        8000 => Bandwidth::Narrowband,
                        12000 => Bandwidth::Mediumband,
                        _ => Bandwidth::Wideband,
                    };
                    (OpusMode::SilkOnly, bw)
                } else {
                    let bw = match sampling_rate {
                        24000 => Bandwidth::Superwideband,
                        _ => Bandwidth::Fullband,
                    };
                    (OpusMode::Hybrid, bw)
                }
            }
        };

        use silk::lin2log::silk_lin2log;
        let variable_hp_smth2_q15 = silk_lin2log(60) << 8;

        Ok(Self {
            celt_enc,
            silk_enc,
            silk_enc_side,
            application,
            sampling_rate,
            channels,
            bandwidth: bw,
            bitrate_bps: 64000,
            complexity: 9,
            use_cbr: false,
            use_inband_fec: false,
            packet_loss_perc: 0,
            silk_initialized: false,
            prev_enc_mode: None,
            forced_mode: None,
            frame_transition: None,
            mode: opus_mode,
            variable_hp_smth2_q15,
            hp_mem: FixedVec::from_value(0, channels * 2),

            #[cfg(not(feature = "heap"))]
            buf_filtered: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_filtered: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_silk_input: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_silk_input: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_silk_input_right: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_silk_input_right: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_silk_api_channel: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_silk_api_channel: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_celt_input: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_celt_input: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            delay_buffer: FixedVec::from_value(0.0, OPUS_DELAY_BUF),
            #[cfg(feature = "heap")]
            delay_buffer: Box::new(FixedVec::from_value(0.0, OPUS_DELAY_BUF)),
            #[cfg(not(feature = "heap"))]
            buf_celt_pcm: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_celt_pcm: Box::new(FixedVec::new()),
            #[cfg(not(feature = "heap"))]
            buf_f32_frame: FixedVec::new(),
            #[cfg(feature = "heap")]
            buf_f32_frame: Box::new(FixedVec::new()),
            encoder_buffer: if matches!(application, Application::RestrictedLowDelay) {
                0
            } else {
                (sampling_rate / 100) as usize
            },
            delay_compensation: if matches!(application, Application::RestrictedLowDelay) {
                0
            } else {
                (sampling_rate / 250) as usize
            },
            hp_mem_float: [0.0; 4],
            prev_hb_gain: 1.0,
            silk_opus_can_switch: false,
            silk_bw_switch: false,
            #[cfg(not(feature = "heap"))]
            silk_delay_buffer: FixedVec::from_value(0, OPUS_DELAY_BUF),
            #[cfg(feature = "heap")]
            silk_delay_buffer: Box::new(FixedVec::from_value(0, OPUS_DELAY_BUF)),
            silk_resampler_enc: silk::resampler::SilkResampler::default(),
            silk_resampler_enc_right: silk::resampler::SilkResampler::default(),
            rc: RangeCoder::new_encoder(1),
            range_final: 0,
        })
    }

    /// The encoder's lookahead in samples at its sampling rate, like libopus
    /// `OPUS_GET_LOOKAHEAD`: the pre-skip an Ogg Opus `OpusHead` needs.
    /// `Fs/400 + Fs/250` (312 samples at 48 kHz), only `Fs/400` for
    /// `RestrictedLowDelay`.
    pub fn lookahead(&self) -> usize {
        self.sampling_rate as usize / 400 + self.delay_compensation
    }

    /// The range encoder's final state after the last packet, like libopus
    /// `OPUS_GET_FINAL_RANGE`. A decoder reports the same value after
    /// decoding that packet, so the two can be compared to check that they
    /// agree on every symbol. It is 0 for a packet that codes no frame (a
    /// ToC-only packet, or the empty frame sent when a budget is exceeded).
    pub fn final_range(&self) -> u32 {
        self.range_final
    }

    pub fn enable_hybrid_mode(&mut self) -> Result<(), &'static str> {
        if self.sampling_rate != 24000 && self.sampling_rate != 48000 {
            return Err("Hybrid mode requires 24kHz or 48kHz sampling rate");
        }
        let bw = if self.sampling_rate == 48000 {
            Bandwidth::Fullband
        } else {
            Bandwidth::Superwideband
        };
        self.mode = OpusMode::Hybrid;
        self.bandwidth = bw;
        self.silk_initialized = false;
        Ok(())
    }

    /// Encode a frame of interleaved f32 PCM (scale ±1.0; C `opus_encode_float`
    /// convention where 1.0 maps to PCM16 32768). Samples are consumed exactly
    /// up to `frame_size * channels`.
    pub fn encode(
        &mut self,
        input: &[f32],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        self.encode_impl(PcmInput::F32(input), frame_size, output)
    }

    /// Encode a frame of interleaved PCM16 (`opus_int16` C `opus_encode()`
    /// convention). Produces byte-for-byte identical packets to
    /// `encode()` called with the same audio converted via
    /// `sample as f32 / 32768.0`, while skipping the f32 round-trip in the
    /// SILK/VoIP path: the integer high-pass biquad and SILK pipeline are fed
    /// natively, with no temporary conversion buffer (issue #28).
    pub fn encode_i16(
        &mut self,
        input: &[i16],
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        self.encode_impl(PcmInput::I16(input), frame_size, output)
    }

    fn encode_impl(
        &mut self,
        input: PcmInput<'_>,
        frame_size: usize,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        if output.len() < 2 {
            return Err("Output buffer too small");
        }
        self.range_final = 0;

        let frame_rate = frame_rate_from_params(self.sampling_rate, frame_size)
            .ok_or("Invalid frame size for sampling rate")?;

        // Clamp caller-provided control fields the way libopus's ctl
        // interface does (OPUS_SET_BITRATE clamps to [500, 3000000];
        // complexity to [0, 10]). Unclamped extremes overflow the bitrate
        // arithmetic downstream (bitrate*6 in celt, `equiv*(90+complexity)`
        // in the mode selection) — issue #27 deep scan.
        self.bitrate_bps = self.bitrate_bps.clamp(500, 3_000_000);
        self.complexity = self.complexity.clamp(0, 10);
        self.packet_loss_perc = self.packet_loss_perc.clamp(0, 100);

        // The encoder consumes exactly frame_size * channels samples per call;
        // reject short input up front instead of panicking inside the HP filter
        // or the float→int conversion loops (issue #27 deep-scan).
        if input.len() < frame_size * self.channels {
            return Err("Input buffer too small for frame");
        }

        // Mode selection: match C's opus_encode_native() behavior.
        // C reference auto-selects between SILK_ONLY and CELT_ONLY; Hybrid is
        // produced afterwards by bandwidth overrides (SILK-only + FB/SWB → Hybrid).
        let mut mode = if let Some(forced) = self.forced_mode {
            forced
        } else if self.application == Application::RestrictedLowDelay {
            OpusMode::CeltOnly
        } else {
            let equiv = compute_equiv_rate(
                self.bitrate_bps,
                self.channels,
                frame_rate,
                !self.use_cbr,
                None,
                self.complexity,
                self.packet_loss_perc,
            );
            let prev_was_celt = self.prev_enc_mode == Some(OpusMode::CeltOnly);
            let has_prev_mode = self.prev_enc_mode.is_some();
            let voice_est = match self.application {
                Application::Voip => 115,
                Application::Audio => 48,
                Application::RestrictedLowDelay => 0,
            };
            let threshold = compute_mode_threshold(
                self.application,
                self.channels,
                prev_was_celt,
                has_prev_mode,
                voice_est,
            );
            if equiv >= threshold && self.sampling_rate >= 24000 {
                OpusMode::CeltOnly
            } else {
                OpusMode::SilkOnly
            }
        };

        // Per-frame packet budget (C computes this before the mode-dependent
        // encode; the starved-budget rules below need it too).
        let target_bits =
            (self.bitrate_bps as i64 * frame_size as i64 / self.sampling_rate as i64) as i32;
        let cbr_bytes = ((target_bits + 4) / 8) as usize;
        let max_data_bytes = output.len();

        // C parity (opus_encoder.c): the nominal-size cap applies in CBR mode
        // only. In VBR mode the CELT-internal bound (vbr_rate/reservoir) decides
        // the per-frame size, allowing overshoot (borrowing) and undershoot.
        // Capping VBR at nominal defeats the reservoir and starves complex frames.
        let mut n_bytes = if self.use_cbr {
            cbr_bytes
                .min(max_data_bytes)
                .clamp(1, OPUS_MAX_PACKET_BYTES)
        } else {
            max_data_bytes.clamp(1, OPUS_MAX_PACKET_BYTES)
        };

        // C opus_encoder.c:1525-1527: a budget representing less than 6 kb/s
        // (9 kb/s for frames longer than 20 ms) cannot carry SILK — libopus
        // switches to CELT-only mode, which codes it as a small valid packet
        // instead of a PLC placeholder (issue #46).
        let tiny_budget_rate: i64 = if frame_rate > 50 { 9000 } else { 6000 };
        if ((n_bytes as i64) * 8 * (self.sampling_rate as i64))
            < (tiny_budget_rate * (frame_size as i64))
        {
            mode = OpusMode::CeltOnly;
        }

        // Mode switches (opus_encoder.c:1540-1556): entering or leaving
        // CELT-only adds a 5 ms CELT frame of redundancy. Going to CELT, this
        // frame keeps the previous mode and the redundancy ends it.
        let transition = match self.frame_transition.take() {
            Some(transition) => transition,
            None => {
                let mut transition = FrameTransition::default();
                let prev = self.prev_enc_mode.unwrap_or(mode);
                if (prev == OpusMode::CeltOnly) != (mode == OpusMode::CeltOnly) {
                    transition.redundancy = true;
                    transition.celt_to_silk = mode != OpusMode::CeltOnly;
                    if !transition.celt_to_silk {
                        if frame_size >= self.sampling_rate as usize / 100 {
                            mode = prev;
                            transition.to_celt = true;
                        } else {
                            transition.redundancy = false;
                        }
                    }
                }
                // Leaving CELT, SILK starts afresh and is prefilled
                // (opus_encoder.c:1576-1581).
                if mode != OpusMode::CeltOnly && self.prev_enc_mode == Some(OpusMode::CeltOnly) {
                    for enc in [&mut self.silk_enc, &mut self.silk_enc_side] {
                        *state_mut(enc) = SilkEncoderState::default();
                        silk_init_encoder(state_mut(enc), 0);
                    }
                    self.silk_initialized = false;
                    self.silk_opus_can_switch = false;
                    transition.prefill = true;
                }
                transition
            }
        };
        let FrameTransition {
            mut redundancy,
            mut celt_to_silk,
            to_celt,
            prefill: prefill_after_celt,
        } = transition;

        // C opus_encoder.c: with the base mode selected, decide whether
        // this packet codes in-band FEC (decide_fec()); it can lower the
        // frame's bandwidth when a high loss rate demands FEC at any cost.
        // The reduction is per-frame: `self.bandwidth` (the user/auto state)
        // is left untouched, matching C where the bandwidth is re-derived
        // every frame.
        let mut frame_bandwidth = self.bandwidth;
        let equiv_rate = compute_equiv_rate(
            self.bitrate_bps,
            self.channels,
            frame_rate,
            !self.use_cbr,
            Some(mode),
            self.complexity,
            self.packet_loss_perc,
        );
        let lbrr_coded = decide_fec(
            if self.use_inband_fec { 1 } else { 0 },
            self.packet_loss_perc,
            self.silk_enc.s_cmn.lbrr_enabled,
            mode,
            &mut frame_bandwidth,
            equiv_rate,
        );

        // FEC can lower the bandwidth. Choose SILK-only or Hybrid after
        // that decision, as libopus does: Hybrid needs bands above wideband.
        if mode == OpusMode::SilkOnly
            && (frame_bandwidth == Bandwidth::Superwideband
                || frame_bandwidth == Bandwidth::Fullband)
        {
            mode = OpusMode::Hybrid;
        }
        if mode == OpusMode::Hybrid
            && (frame_bandwidth == Bandwidth::Narrowband
                || frame_bandwidth == Bandwidth::Mediumband
                || frame_bandwidth == Bandwidth::Wideband)
        {
            mode = OpusMode::SilkOnly;
        }

        // 40/60 ms frames are only legal as a single Opus frame in SILK-only
        // mode. CELT and Hybrid cap a frame at 20 ms, so libopus encodes 20 ms
        // sub-frames and repacketizes them into one code 1/2/3 packet
        // (opus_encoder.c `encode_multiframe_packet`). (issue #35)
        if self.forced_mode.is_none()
            && mode != OpusMode::SilkOnly
            && frame_size > self.sampling_rate as usize / 50
        {
            return self.encode_multiframe(input, frame_size, mode, transition, output);
        }

        if mode == OpusMode::CeltOnly {
            // silk_mode.LBRR_coded persists across frames even while SILK is
            // idle; it feeds decide_fec's hysteresis on the next SILK frame.
            self.silk_enc.s_cmn.lbrr_enabled = 0;
            self.silk_enc_side.s_cmn.lbrr_enabled = 0;
        }

        if mode == OpusMode::CeltOnly {
            match frame_rate {
                400 | 200 | 100 | 50 => {}
                _ => return Err("Unsupported frame size for CELT-only mode"),
            }
        }

        if mode == OpusMode::Hybrid {
            match frame_rate {
                100 | 50 => {}
                _ => return Err("Unsupported frame size for Hybrid mode"),
            }
        }

        if mode == OpusMode::SilkOnly {
            match frame_rate {
                100 | 50 | 25 | 16 => {}
                // 2.5/5 ms are CELT-only durations: SILK frames are 10 ms or
                // longer (libopus), and sub-10 ms frames break the SILK
                // analysis geometry (e.g. the VAD band split at 12 kHz).
                _ => return Err("Unsupported frame size for SILK-only mode"),
            }
        }

        let mut toc = gen_toc(mode, frame_rate, frame_bandwidth, self.channels);
        output[0] = toc;

        // C opus_encoder.c:1226-1243: a budget too small to code anything
        // useful produces a TOC-only "PLC" packet — decoders conceal it as a
        // loss. Encoding into 0-2 bytes would otherwise overflow the range
        // coder (issue #45). Under CBR the packet is padded to the nominal
        // size, exactly like libopus's opus_packet_pad.
        let sub_3_bytes_per_frame =
            (self.bitrate_bps as i64) < 3 * frame_rate as i64 * 8;
        let tiny_long_frame =
            frame_rate < 50
                && (n_bytes * (frame_rate as usize) < 300 || self.bitrate_bps < 2400);
        if n_bytes < 3 || sub_3_bytes_per_frame || tiny_long_frame {
            let mut toc_mode = mode;
            let mut packet_code = 0u8;
            let mut num_multiframes = 0u8;
            let mut frame_rate_adj = frame_rate;
            if frame_rate > 100 {
                toc_mode = OpusMode::CeltOnly;
            }
            // 40 ms -> 2 x 20 ms in CELT/hybrid mode.
            if frame_rate == 25 && toc_mode != OpusMode::SilkOnly {
                frame_rate_adj = 50;
                packet_code = 1;
            }
            // >= 60 ms frames: 1 x 60 ms, or 2 x 40 ms / 2 x 60 ms multiframes.
            if frame_rate <= 16 {
                if n_bytes == 1 || (toc_mode == OpusMode::SilkOnly && frame_rate != 10) {
                    toc_mode = OpusMode::SilkOnly;
                    packet_code = if frame_rate <= 12 { 1 } else { 0 };
                    frame_rate_adj = if frame_rate == 12 { 25 } else { 16 };
                } else {
                    num_multiframes = (50 / frame_rate) as u8;
                    frame_rate_adj = 50;
                    packet_code = 3;
                }
            }
            let mut plc_bw = frame_bandwidth;
            if toc_mode == OpusMode::SilkOnly && bandwidth_fec_index(plc_bw) > 2 {
                // SilkOnly is wideband at most (C: bw > OPUS_BANDWIDTH_WIDEBAND).
                plc_bw = Bandwidth::Wideband;
            }
            output[0] = gen_toc(toc_mode, frame_rate_adj, plc_bw, self.channels) | packet_code;
            let mut ret = if packet_code <= 1 { 1 } else { 2 };
            if packet_code == 3 {
                output[1] = num_multiframes;
            }
            if self.use_cbr {
                let target_total = n_bytes.min(output.len());
                if target_total >= 2 {
                    // opus_packet_pad of the TOC-only packet: code 3 with
                    // `m` empty frames and zero padding.
                    let m: usize = if packet_code == 3 {
                        num_multiframes as usize
                    } else if packet_code == 1 {
                        2
                    } else {
                        1
                    };
                    output[0] = (output[0] & 0xFC) | 0x03;
                    output[1] = 0x40 | m as u8; // padding present, CBR, m frames
                    if target_total > 2 {
                        let pad_amount = target_total - 2;
                        let nb_255s = (pad_amount - 1) / 255;
                        let mut ptr = 2;
                        for _ in 0..nb_255s {
                            output[ptr] = 255;
                            ptr += 1;
                        }
                        output[ptr] = (pad_amount - 255 * nb_255s - 1) as u8;
                        ptr += 1;
                        for b in &mut output[ptr..target_total] {
                            *b = 0;
                        }
                    }
                    ret = target_total;
                }
            }
            // C leaves st->prev_mode untouched on this path.
            return Ok(ret);
        }

        let init_rc_size = n_bytes - 1;
        self.rc.reset_for_encode(init_rc_size as u32);

        // C opus_encode_frame_native: high-pass cutoff state is updated for ALL
        // modes (celt_encoder.c:1969-1977); the filtered signal is what feeds
        // both SILK (hybrid) and CELT. Compute it here so CELT-only frames also
        // keep `variable_hp_smth2_q15` in sync with libopus.
        let hp_freq_smth1 = if mode == OpusMode::CeltOnly {
            silk_lin2log(60) << 8
        } else {
            self.silk_enc.s_cmn.variable_hp_smth1_q15
        };

        // The first frame at a new SILK rate starts with redundancy too
        // (opus_encoder.c:1933-1956). CELT-only frames code none.
        if self.silk_bw_switch && mode != OpusMode::CeltOnly {
            redundancy = true;
            celt_to_silk = true;
        }
        if mode == OpusMode::CeltOnly {
            redundancy = false;
        }
        let mut redundancy_bytes = 0usize;
        if redundancy {
            redundancy_bytes =
                compute_redundancy_bytes(n_bytes, self.bitrate_bps, frame_rate, self.channels);
            redundancy = redundancy_bytes != 0;
        }

        // SILK_FIX_CONST(VARIABLE_HP_SMTH_COEF2, 16): 0.015 in Q16, rounded to nearest.
        const VARIABLE_HP_SMTH_COEF2_Q16: i32 = 983;
        self.variable_hp_smth2_q15 = silk_smlawb(
            self.variable_hp_smth2_q15,
            hp_freq_smth1 - self.variable_hp_smth2_q15,
            VARIABLE_HP_SMTH_COEF2_Q16,
        );

        let cutoff_hz = silk_log2lin(silk_rshift(self.variable_hp_smth2_q15, 8));

        // The 16-bit high-passed input of SILK. libopus filters every frame,
        // so the filter state and the history a SILK prefill codes stay
        // current through CELT-only frames (opus_encoder.c:2004-2012).
        if self.application != Application::RestrictedLowDelay {
            let required_size = frame_size * self.channels;
            self.buf_filtered.resize(required_size, 0);
            let voip = self.application == Application::Voip;
            match input {
                PcmInput::I16(samples) => {
                    if voip {
                        // Native i16 path: feed the integer HP biquad directly,
                        // skipping the f32 round-trip and its temporary
                        // conversion buffer (issue #28).
                        hp_cutoff_i16(
                            samples,
                            cutoff_hz,
                            state_mut(&mut self.buf_filtered),
                            &mut self.hp_mem,
                            frame_size,
                            self.channels,
                            self.sampling_rate,
                        );
                    } else {
                        let filtered = state_mut(&mut self.buf_filtered);
                        filtered[..required_size].copy_from_slice(&samples[..required_size]);
                    }
                }
                PcmInput::F32(samples) => {
                    if voip {
                        hp_cutoff(
                            samples,
                            cutoff_hz,
                            state_mut(&mut self.buf_filtered),
                            &mut self.hp_mem,
                            frame_size,
                            self.channels,
                            self.sampling_rate,
                        );
                    } else {
                        let filtered = state_mut(&mut self.buf_filtered);
                        for (i, &x) in samples.iter().enumerate().take(required_size) {
                            filtered[i] = (x * 32768.0).clamp(-32768.0, 32767.0) as i16;
                        }
                    }
                }
            }
        }

        let mut silk_bitrate = 0;
        let mut hb_gain = 1.0;
        let mut silk_switch_ready = false;
        let mut silk_prefilled = false;
        if mode == OpusMode::SilkOnly || mode == OpusMode::Hybrid {
            let frame_ms = (frame_size as i32 * 1000) / self.sampling_rate;

            // C opus_encoder.c `bits_target` / `total_bitRate` (1.6): SILK
            // codes at the requested bitrate, capped by the packet budget,
            // less the TOC byte. Under VBR the budget is the caller's output
            // buffer, which must cap the rate, not set it (issue #43). CBR
            // first re-quantizes the bitrate to the whole bytes it codes, as
            // opus_encode_native does, so the cap is the packet itself.
            let rate_bps = if self.use_cbr {
                bits_to_bitrate(8 * n_bytes as i32, self.sampling_rate, frame_size)
            } else {
                self.bitrate_bps
            };
            let rate_bits = bitrate_to_bits(rate_bps, self.sampling_rate, frame_size);
            let bits_target = (8 * (n_bytes - redundancy_bytes) as i32).min(rate_bits) - 8;
            let total_bitrate = bits_to_bitrate(bits_target, self.sampling_rate, frame_size);

            // The internal sample rate (opus_encoder.c:2112-2147, enc_API.c,
            // control_audio_bandwidth.c): the frame's bandwidth asks for a
            // rate, which SILK reaches only by its switching state machine.
            let mut desired_fs_hz = match frame_bandwidth {
                Bandwidth::Narrowband => 8000,
                Bandwidth::Mediumband => 12000,
                _ => 16000,
            };
            let mut max_fs_hz = 16000;
            if mode == OpusMode::SilkOnly {
                let mut effective_max_rate =
                    bits_to_bitrate(n_bytes as i32 * 8, self.sampling_rate, frame_size);
                if frame_rate > 50 {
                    effective_max_rate = effective_max_rate * 2 / 3;
                }
                if effective_max_rate < 8000 {
                    max_fs_hz = 12000;
                    desired_fs_hz = desired_fs_hz.min(12000);
                }
                if effective_max_rate < 7000 {
                    max_fs_hz = 8000;
                    desired_fs_hz = desired_fs_hz.min(8000);
                }
            }
            let rate_limits = SilkRateLimits {
                api_fs_hz: self.sampling_rate,
                desired_fs_hz,
                max_fs_hz,
                // Hybrid doesn't reduce the SILK bandwidth.
                min_fs_hz: if mode == OpusMode::Hybrid {
                    16000
                } else {
                    8000
                },
            };
            // A switch prepared by the previous packet restarts the SILK
            // encoders, keeping their low-pass state, and prefills them
            // (opus_encoder.c:1933-1940, prefill 2 in enc_API.c:215-246).
            let silk_switch_prefill = core::mem::take(&mut self.silk_bw_switch);
            if silk_switch_prefill {
                for enc in [&mut self.silk_enc, &mut self.silk_enc_side]
                    .into_iter()
                    .take(self.channels)
                {
                    silk_reset_for_rate_switch(state_mut(enc));
                }
                self.silk_initialized = false;
            }
            let allow_switch = self.silk_enc.allow_bandwidth_switch != 0;
            let rate_control = |enc: &mut SilkEncoderState, can_switch: bool| {
                let (mut ready, mut unused_max_bits) = (false, 0);
                let fs = silk_control_audio_bandwidth(
                    &mut enc.s_cmn,
                    rate_limits,
                    allow_switch,
                    can_switch,
                    frame_ms,
                    &mut unused_max_bits,
                    &mut ready,
                );
                (fs, ready)
            };
            let silk_fs_khz = if !self.silk_initialized && !silk_switch_prefill {
                // A fresh encoder starts at the desired rate.
                rate_limits.desired_fs_hz.min(rate_limits.api_fs_hz) / 1000
            } else {
                let (fs, ready) =
                    rate_control(state_mut(&mut self.silk_enc), self.silk_opus_can_switch);
                if self.channels == 2 {
                    // The side follows the mid's rate (force_fs_kHz) but
                    // keeps its own low-pass state machine.
                    rate_control(
                        state_mut(&mut self.silk_enc_side),
                        self.silk_opus_can_switch,
                    );
                }
                if !silk_switch_prefill {
                    silk_switch_ready = ready;
                }
                fs
            };
            let rate_change = !self.silk_initialized || self.silk_enc.s_cmn.fs_khz != silk_fs_khz;
            // A new frame duration alone sets the packet up again: libopus
            // runs silk_setup_fs for every packet (control_codec.c:208-238),
            // as a 40 ms SILK-only packet followed by 20 ms hybrid frames
            // needs, and drops the FEC flags of the old layout
            // (enc_API.c:207, 268-272).
            let duration_change = self.silk_enc.s_cmn.packet_size_ms != frame_ms;
            if rate_change || duration_change {
                // A running encoder keeps its buffered input across a rate
                // change: libopus moves it to the API rate and back through
                // the new input resampler (silk_setup_resamplers,
                // control_codec.c:134-197), as when SILK-only frames at 8 or
                // 12 kHz turn hybrid.
                let old_khz = self.silk_enc.s_cmn.fs_khz;
                let moved = rate_change && old_khz != 0 && old_khz != silk_fs_khz;
                if moved {
                    silk_move_history(
                        state_mut(&mut self.silk_enc),
                        &mut self.silk_resampler_enc,
                        self.sampling_rate,
                        silk_fs_khz,
                    )?;
                    if self.channels == 2 {
                        let side = state_mut(&mut self.silk_enc_side);
                        if side.s_cmn.fs_khz != 0 {
                            silk_move_history(
                                side,
                                &mut self.silk_resampler_enc_right,
                                self.sampling_rate,
                                silk_fs_khz,
                            )?;
                        } else if self
                            .silk_resampler_enc_right
                            .init_for_enc(self.sampling_rate, silk_fs_khz * 1000)
                            != 0
                        {
                            return Err("Unsupported SILK resampling ratio");
                        }
                    }
                }
                silk_control_encoder(
                    state_mut(&mut self.silk_enc),
                    silk_fs_khz,
                    frame_ms,
                    total_bitrate,
                    self.complexity,
                );
                self.silk_enc.s_cmn.use_cbr = if self.use_cbr { 1 } else { 0 };

                self.silk_enc.s_cmn.n_channels = self.channels as i32;
                self.silk_initialized = true;
                if duration_change {
                    self.silk_enc.s_cmn.lbrr_flags.fill(0);
                    self.silk_enc_side.s_cmn.lbrr_flags.fill(0);
                }
                // C silk_setup_resamplers: re-init the encoder-direction
                // resampler whenever the SILK rate changes; the input delay
                // comes from delay_matrix_enc.
                if rate_change && !moved {
                    let ret = self
                        .silk_resampler_enc
                        .init_for_enc(self.sampling_rate as i32, silk_fs_khz * 1000);
                    if ret != 0 {
                        return Err("Unsupported SILK resampling ratio");
                    }
                }
                if self.channels == 2 {
                    if rate_change && !moved {
                        self.silk_resampler_enc_right = self.silk_resampler_enc.clone();
                    }
                    // The side runs at the mid's rate and frame size (C
                    // silk_control_encoder with force_fs_kHz).
                    let side = state_mut(&mut self.silk_enc_side);
                    silk_control_encoder(
                        side,
                        silk_fs_khz,
                        frame_ms,
                        total_bitrate,
                        self.complexity,
                    );
                    side.s_cmn.use_cbr = self.silk_enc.s_cmn.use_cbr;
                    side.s_cmn.n_channels = 2;
                }
            }

            // C silk_setup_LBRR (control_codec.c): per-packet FEC state and
            // LBRR gain increase (7 on the first FEC packet, then
            // max(7 - 0.2*loss, 3)), for each coded channel.
            for enc in [&mut self.silk_enc, &mut self.silk_enc_side]
                .into_iter()
                .take(self.channels)
            {
                enc.s_cmn.use_in_band_fec = if self.use_inband_fec { 1 } else { 0 };
                enc.s_cmn.packet_loss_perc = self.packet_loss_perc.clamp(0, 100);
                silk_setup_lbrr(state_mut(enc), lbrr_coded);
            }

            let silk_frame_len_samples =
                frame_size * silk_fs_khz as usize / (self.sampling_rate.max(1000) / 1000) as usize;
            silk_bitrate = if mode == OpusMode::Hybrid {
                let frame_duration_ms = frame_size as i32 * 1000 / self.sampling_rate;
                let frame20ms = frame_duration_ms >= 20;
                compute_silk_rate_for_hybrid(
                    total_bitrate,
                    frame_bandwidth,
                    frame20ms,
                    !self.use_cbr,
                    lbrr_coded != 0,
                    self.channels,
                )
            } else {
                // C: SILK gets all bits, at the requested rate (issue #43).
                total_bitrate
            };
            if mode == OpusMode::Hybrid {
                hb_gain = hybrid_hb_gain(total_bitrate - silk_bitrate);
            }

            // C opus_encoder.c:1806-1826: prefill the fresh SILK encoder with
            // the tail of the delay buffer, zeroed before a 2.5 ms fade-in,
            // resampled through the fresh encoder resampler so its delay
            // state also starts consistent.
            silk_prefilled = (prefill_after_celt || silk_switch_prefill)
                && self.encoder_buffer > 0
                && self.delay_compensation > 0;
            if silk_prefilled {
                let ch = self.channels;
                let f2_5 = self.sampling_rate / 400;
                let eb = self.encoder_buffer;
                let offset = eb - self.delay_compensation.min(eb) - f2_5 as usize;
                let pre_resampled_len =
                    eb * silk_fs_khz as usize * 1000 / self.sampling_rate as usize;
                // Stack-fixed: encoder_buffer = Fs/100 <= 480, and the
                // resampled prefill is at most the same length (Copy at equal
                // rates). No allocation: this crate is heap-free without the
                // `heap` feature.
                if ch == 2 {
                    // Each channel through its own resampler, then mid/side
                    // at the internal rate, as C silk_Encode prefills two
                    // internal channels.
                    let (mut pre_l, mut pre_r) = ([0i16; 480], [0i16; 480]);
                    for i in offset.min(eb)..eb {
                        let k = i - offset;
                        pre_l[i] =
                            prefill_fade(self.silk_delay_buffer[2 * i], k, self.sampling_rate);
                        pre_r[i] =
                            prefill_fade(self.silk_delay_buffer[2 * i + 1], k, self.sampling_rate);
                    }
                    let (mut int_l, mut int_r) = ([0i16; 480], [0i16; 480]);
                    let int_l = &mut int_l[..pre_resampled_len];
                    let int_r = &mut int_r[..pre_resampled_len];
                    self.silk_resampler_enc
                        .process(int_l, &pre_l[..eb], eb as i32);
                    self.silk_resampler_enc_right
                        .process(int_r, &pre_r[..eb], eb as i32);
                    silk_encode_prefill_stereo(
                        state_mut(&mut self.silk_enc),
                        state_mut(&mut self.silk_enc_side),
                        int_l,
                        int_r,
                        silk_bitrate,
                        1,
                    );
                } else {
                    let mut pre = [0i16; 480];
                    let pre = &mut pre[..eb];
                    for i in offset.min(eb)..eb {
                        pre[i] =
                            prefill_fade(self.silk_delay_buffer[i], i - offset, self.sampling_rate);
                    }
                    let mut pre_internal = [0i16; 480];
                    let pre_internal = &mut pre_internal[..pre_resampled_len];
                    self.silk_resampler_enc
                        .process(pre_internal, pre, eb as i32);
                    silk_encode_prefill(state_mut(&mut self.silk_enc), pre_internal, 1);
                }
            }
            if silk_prefilled {
                // The coding call controls again, now unable to switch, and
                // repeats the LBRR setup of the prefill call, which counts
                // as a packet with LBRR.
                let (_, ready) = rate_control(state_mut(&mut self.silk_enc), false);
                if self.channels == 2 {
                    rate_control(state_mut(&mut self.silk_enc_side), false);
                }
                silk_switch_ready = ready;
                for enc in [&mut self.silk_enc, &mut self.silk_enc_side]
                    .into_iter()
                    .take(self.channels)
                {
                    silk_setup_lbrr(state_mut(enc), lbrr_coded);
                }
            }

            let input_i16 = state_ref(&self.buf_filtered);

            // SILK input resampling (C silk_Encode always runs silk_resampler
            // on the API-rate input; the Copy path still applies its
            // delay_matrix_enc input delay at equal rates). Stereo resamples
            // each channel on its own and splits mid/side at the internal
            // rate (enc_API.c, nChannelsInternal 2), in SILK-only and Hybrid
            // alike.
            self.buf_silk_input.resize(silk_frame_len_samples, 0);
            if self.channels == 2 {
                self.buf_silk_input_right.resize(silk_frame_len_samples, 0);
                for c in 0..2 {
                    self.buf_silk_api_channel.resize(frame_size, 0);
                    for (s, lr) in self
                        .buf_silk_api_channel
                        .iter_mut()
                        .zip(input_i16.chunks_exact(2))
                    {
                        *s = lr[c];
                    }
                    let (resampler, out) = if c == 0 {
                        (
                            &mut self.silk_resampler_enc,
                            state_mut(&mut self.buf_silk_input).as_mut_slice(),
                        )
                    } else {
                        (
                            &mut self.silk_resampler_enc_right,
                            state_mut(&mut self.buf_silk_input_right).as_mut_slice(),
                        )
                    };
                    if resampler.process(out, &self.buf_silk_api_channel, frame_size as i32) != 0 {
                        return Err("SILK input resampling failed");
                    }
                }
            } else {
                let got = self.silk_resampler_enc.process(
                    state_mut(&mut self.buf_silk_input),
                    input_i16,
                    frame_size as i32,
                );
                if got != 0 {
                    return Err("SILK input resampling failed");
                }
            }
            let silk_input: &[i16] = state_ref(&self.buf_silk_input);

            let mut pn_bytes = 0;

            // Max bits for SILK, counting the TOC, the redundancy bytes, and
            // 1 bit for the redundancy position (20 bits more for its flag and
            // size in hybrid) (opus_encoder.c:2154-2161).
            let mut packet_max_bits = ((n_bytes - 1) * 8) as i32;
            if redundancy && redundancy_bytes >= 2 {
                packet_max_bits -= redundancy_bytes as i32 * 8 + 1;
                if mode == OpusMode::Hybrid {
                    packet_max_bits -= 20;
                }
            }
            let mut silk_max_bits = if mode == OpusMode::Hybrid {
                let total_max_bits = packet_max_bits;
                if self.use_cbr {
                    let silk_bits = (silk_bitrate as i64 * frame_size as i64
                        / self.sampling_rate as i64) as i32;
                    let other_bits = 0i32.max(total_max_bits - silk_bits);
                    0i32.max(total_max_bits - other_bits * 3 / 4)
                } else {
                    let frame_duration_ms = frame_size as i32 * 1000 / self.sampling_rate;
                    let frame20ms = frame_duration_ms >= 20;
                    let max_bit_rate = compute_silk_rate_for_hybrid(
                        total_max_bits * self.sampling_rate / frame_size as i32,
                        frame_bandwidth,
                        frame20ms,
                        true,
                        lbrr_coded != 0,
                        self.channels,
                    );
                    max_bit_rate * frame_size as i32 / self.sampling_rate
                }
            } else {
                packet_max_bits
            };
            if silk_switch_ready {
                // Room for the redundant frame of the coming switch.
                silk_max_bits -= silk_max_bits * 5 / (frame_ms + 5);
            }
            let silk_use_cbr = if mode == OpusMode::Hybrid && self.use_cbr {
                0
            } else if self.use_cbr {
                1
            } else {
                0
            };
            let ret = if self.channels == 2 {
                silk_encode_stereo_packet(
                    state_mut(&mut self.silk_enc),
                    state_mut(&mut self.silk_enc_side),
                    silk_input,
                    state_ref(&self.buf_silk_input_right),
                    silk_input.len(),
                    &mut self.rc,
                    &mut pn_bytes,
                    silk_bitrate,
                    silk_max_bits,
                    silk_use_cbr,
                    1,
                )
            } else {
                silk_encode(
                    state_mut(&mut self.silk_enc),
                    silk_input,
                    silk_input.len(),
                    &mut self.rc,
                    &mut pn_bytes,
                    silk_bitrate,
                    silk_max_bits,
                    silk_use_cbr,
                    1,
                )
            };
            if ret != 0 {
                return Err("SILK encoding failed");
            }
            // The TOC signals the rate SILK coded at (opus_encoder.c:2221-2232).
            if mode == OpusMode::SilkOnly {
                frame_bandwidth = match silk_fs_khz {
                    8 => Bandwidth::Narrowband,
                    12 => Bandwidth::Mediumband,
                    _ => Bandwidth::Wideband,
                };
                toc = gen_toc(mode, frame_rate, frame_bandwidth, self.channels);
                output[0] = toc;
            }
            // A ready switch happens at the next packet, which prefills SILK
            // again.
            self.silk_opus_can_switch = silk_switch_ready;
            self.silk_bw_switch = silk_switch_ready;
            if silk_switch_ready {
                // The packet before the switch ends with redundancy
                // (opus_encoder.c:2250-2260).
                redundancy_bytes =
                    compute_redundancy_bytes(n_bytes, self.bitrate_bps, frame_rate, self.channels);
                redundancy = redundancy_bytes != 0;
                celt_to_silk = false;
            }
        } else {
            self.silk_bw_switch = false;
        }
        // The last 10 ms of SILK input, which a prefill reuses (libopus
        // `delay_buffer` in the fixed-point build), in every mode.
        if self.application != Application::RestrictedLowDelay {
            let n = self.encoder_buffer * self.channels;
            push_silk_history(
                &mut self.silk_delay_buffer[..n],
                &self.buf_filtered[..frame_size * self.channels],
            );
        }

        // Signal the redundancy, if there is room (opus_encoder.c:2350-2381).
        // SILK-only packets imply it by their length.
        let hybrid = mode == OpusMode::Hybrid;
        if mode != OpusMode::CeltOnly
            && self.rc.tell() + 17 + 20 * i32::from(hybrid) <= 8 * (n_bytes as i32 - 1)
        {
            if hybrid {
                self.rc.encode_bit_logp(redundancy, 12);
            }
            if redundancy {
                self.rc.encode_bit_logp(celt_to_silk, 1);
                let tell = self.rc.tell();
                // Keep the 8 bits of the size, and a few bits for CELT.
                let used = if hybrid {
                    (tell + 8 + 3 + 7) >> 3
                } else {
                    (tell + 7) >> 3
                };
                let max_redundancy = (n_bytes as i32 - 1) - used;
                redundancy_bytes =
                    (redundancy_bytes as i32).min(max_redundancy).clamp(2, 257) as usize;
                if hybrid {
                    self.rc.enc_uint(redundancy_bytes as u32 - 2, 256);
                }
            }
        } else {
            redundancy = false;
        }
        if !redundancy {
            self.silk_bw_switch = false;
            redundancy_bytes = 0;
        }

        if mode == OpusMode::Hybrid {
            let nb_compr_bytes = (n_bytes - 1 - redundancy_bytes) as u32;
            self.rc.shrink(nb_compr_bytes);
        }

        let silk_ret_bytes = if mode == OpusMode::SilkOnly {
            ((self.rc.tell() + 7) >> 3) as usize
        } else {
            0
        };

        // libopus parity: adjust nbCompressedBytes for CELT/Hybrid based on tell (celt_encoder.c:1913-1921)
        // tmp = bitrate*frame_size + tell*Fs; nbCompressed = (tmp+4*Fs)/(8*Fs)
        // This is the CBR branch (vbr==0) in celt_encoder.c; for VBR the encoder
        // does *not* do this adjustment - it uses the vbr_bound logic instead.
        // Doing it for VBR would double-shrink and corrupt the budget.
        if mode != OpusMode::SilkOnly && self.use_cbr {
            let tell = self.rc.tell();
            if tell > 1 {
                let tmp = self.bitrate_bps as i64 * frame_size as i64
                    + tell as i64 * self.sampling_rate as i64;
                let adjusted = ((tmp + 4 * self.sampling_rate as i64)
                    / (8 * self.sampling_rate as i64)) as usize;
                let new_n = adjusted.clamp(1, OPUS_MAX_PACKET_BYTES).min(max_data_bytes);
                if new_n < n_bytes {
                    n_bytes = new_n;
                    // shrink range coder to new size (keep SILK bytes, trim tail)
                    let new_payload = n_bytes - 1;
                    self.rc.shrink(new_payload as u32);
                }
            }
        }
        let previous_hb_gain = self.prev_hb_gain;
        self.prev_hb_gain = hb_gain;
        let ch = self.channels;
        let n2 = self.sampling_rate as usize / 200;
        let n4 = self.sampling_rate as usize / 400;
        let has_delay = self.delay_compensation > 0;

        // Entering CELT or hybrid mode from another mode, CELT restarts and
        // a 2.5 ms frame of the audio before this one prefills it
        // (opus_encoder.c:2297-2302, 2478-2486). A SILK prefill in this frame
        // has faded that audio in (opus_encoder.c:2201-2204).
        let celt_prefill = mode != OpusMode::SilkOnly
            && self.prev_enc_mode.is_some_and(|prev| prev != mode)
            && has_delay;
        let mut tmp_prefill = [0.0f32; 240];
        let tmp_prefill = &mut tmp_prefill[..n4 * ch];
        if celt_prefill {
            let end = self.encoder_buffer - self.delay_compensation;
            tmp_prefill.copy_from_slice(&self.delay_buffer[(end - n4) * ch..end * ch]);
            if silk_prefilled {
                gain_fade(tmp_prefill, 0.0, 1.0, ch, self.sampling_rate);
            }
        }

        // Float view of the current frame for the CELT pipeline. For the
        // i16 entry point this converts once into a scratch buffer with
        // the exact per-sample scaling a caller of `encode()` performs
        // (`s as f32 / 32768.0`), so packets stay byte-for-byte identical
        // across the two entry points (issue #28).
        let input_f32: &[f32] = match input {
            PcmInput::F32(samples) => samples,
            PcmInput::I16(samples) => {
                let n = frame_size * ch;
                self.buf_f32_frame.resize(n, 0.0);
                let staging = state_mut(&mut self.buf_f32_frame);
                for (dst, &src) in staging[..n].iter_mut().zip(samples[..n].iter()) {
                    *dst = src as f32 / 32768.0;
                }
                state_ref(&self.buf_f32_frame)
            }
        };

        // Build the CELT input exactly like C `opus_encode_frame_native`:
        //   pcm_buf = [delay-compensation prefix from ring]
        //             [dc_reject / hp_cutoff'd current frame]
        // CELT then reads pcm_buf[0..frame_size*channels] (opus_encoder.c:
        // 1966-2010, 2493). This delay + filter step is what libopus feeds
        // its CELT encoder; passing the raw input diverges from 1.6. The
        // delay line moves on in every mode, so the CELT input after a
        // SILK-only stretch starts from recent history, and the redundancy
        // frames of a SILK-only packet code it.
        if has_delay {
            let ebuf = self.encoder_buffer;
            let total = (self.delay_compensation + frame_size) * ch;
            self.buf_celt_pcm.resize(total, 0.0);
            push_delay_history(
                &mut self.delay_buffer[..ebuf * ch],
                &mut self.buf_celt_pcm[..total],
                &mut self.hp_mem_float,
                input_f32,
                self.application == Application::Voip,
                cutoff_hz,
                frame_size,
                ch,
                self.sampling_rate,
            );
            // Fade after saving unscaled history, so SILK is unaffected.
            if previous_hb_gain < 1.0 || hb_gain < 1.0 {
                gain_fade(
                    &mut self.buf_celt_pcm[..frame_size * ch],
                    previous_hb_gain,
                    hb_gain,
                    ch,
                    self.sampling_rate,
                );
            }
        }

        // The CELT settings of every frame (opus_encoder.c:2262-2295):
        // libopus `CELT_SET_END_BAND` matches the TOC bandwidth so the
        // decoder reads the same number of coded bands (issue #37).
        if mode != OpusMode::SilkOnly || redundancy {
            self.celt_enc.complexity = self.complexity;
            self.celt_enc.set_end_band(match frame_bandwidth {
                Bandwidth::Narrowband => 13,
                Bandwidth::Mediumband | Bandwidth::Wideband => 17,
                Bandwidth::Superwideband => 19,
                Bandwidth::Fullband | Bandwidth::Auto => 21,
            });
            // libopus forwards the loss rate to CELT (OPUS_SET_PACKET_LOSS_PERC).
            self.celt_enc.set_loss_rate(self.packet_loss_perc);
            self.celt_enc.set_bitrate(celt::OPUS_BITRATE_MAX);
        }
        if mode != OpusMode::SilkOnly {
            self.celt_enc.set_prediction(2);
        }

        // Leaving CELT, the packet starts with a 5 ms CELT frame of all
        // bands, coded with the CELT state of the frames before
        // (opus_encoder.c:2427-2440).
        let mut redundant = [0u8; 257];
        let redundant = &mut redundant[..redundancy_bytes];
        // The packet's final range includes the redundant frame's
        // (opus_encoder.c `rangeFinal = enc.rng ^ redundant_rng`).
        let mut redundant_rng = 0u32;
        if redundancy && celt_to_silk {
            self.celt_enc.set_vbr(false);
            redundant_rng = encode_celt_side_frame(
                &mut self.celt_enc,
                &self.buf_celt_pcm[..n2 * ch],
                n2,
                ch,
                0,
                false,
                redundant,
            );
            self.celt_enc.reset_state();
        }

        if mode != OpusMode::SilkOnly {
            let start_band = if hybrid { 17 } else { 0 };
            // In hybrid mode, libopus forwards the SILK layer's signal type
            // and offset to CELT (CELT_SET_SILK_INFO, opus_encoder.c:2420-2424).
            if hybrid {
                let ix = &self.silk_enc.s_cmn.indices;
                let offset = silk::tables::SILK_QUANTIZATION_OFFSETS_Q10
                    [(ix.signal_type >> 1) as usize][ix.quant_offset_type as usize];
                self.celt_enc
                    .set_silk_info(i32::from(ix.signal_type), i32::from(offset));
            }
            let celt_rate = celt_bitrate(mode, !self.use_cbr, self.bitrate_bps, silk_bitrate, ch);
            // CELT codes VBR only with a rate (celt_encoder.c `vbr_rate`).
            let celt_vbr = !self.use_cbr && celt_rate != celt::OPUS_BITRATE_MAX;
            self.celt_enc.set_bitrate(celt_rate);
            self.celt_enc.set_vbr(!self.use_cbr);
            // libopus: Hybrid VBR is unconstrained (can steal from SILK), CELT-only constrained
            self.celt_enc
                .set_constrained_vbr(mode == OpusMode::CeltOnly);
            if celt_prefill {
                self.celt_enc.reset_state();
                let mut dummy = [0u8; 2];
                encode_celt_side_frame(
                    &mut self.celt_enc,
                    tmp_prefill,
                    n4,
                    ch,
                    start_band,
                    celt_vbr,
                    &mut dummy,
                );
                self.celt_enc.set_prediction(0);
            }

            // 5. deinterleave pcm_buf[0..frame_size*ch] (interleaved) to
            //    channel-major for the Rust CELT encoder.
            let celt_input: &[f32] = if has_delay {
                let n = frame_size * ch;
                self.buf_celt_input.resize(n, 0.0);
                for i in 0..frame_size {
                    for c in 0..ch {
                        self.buf_celt_input[c * frame_size + i] = self.buf_celt_pcm[i * ch + c];
                    }
                }
                state_ref(&self.buf_celt_input)
            } else if ch == 1 {
                input_f32
            } else {
                let n = frame_size * ch;
                self.buf_celt_input.resize(n, 0.0);
                for i in 0..frame_size {
                    for c in 0..ch {
                        self.buf_celt_input[c * frame_size + i] = input_f32[i * ch + c];
                    }
                }
                state_ref(&self.buf_celt_input)
            };

            let total_packet_bits = ((n_bytes - 1 - redundancy_bytes) * 8) as i32;
            if self.rc.tell() <= total_packet_bits {
                self.celt_enc.encode_with_budget_vbr(
                    celt_input,
                    frame_size,
                    &mut self.rc,
                    start_band,
                    total_packet_bits,
                    celt_vbr,
                );
            }
        }

        // Going to CELT, the packet ends with a 5 ms CELT frame of all bands,
        // coded by a restarted CELT encoder that the 2.5 ms before primes;
        // the next frame continues from it (opus_encoder.c:2514-2546).
        if redundancy && !celt_to_silk {
            self.celt_enc.reset_state();
            self.celt_enc.set_prediction(0);
            self.celt_enc.set_vbr(false);
            self.celt_enc.set_bitrate(celt::OPUS_BITRATE_MAX);
            let tail = &self.buf_celt_pcm[(frame_size - n2 - n4) * ch..frame_size * ch];
            let mut dummy = [0u8; 2];
            encode_celt_side_frame(
                &mut self.celt_enc,
                &tail[..n4 * ch],
                n4,
                ch,
                0,
                false,
                &mut dummy,
            );
            redundant_rng = encode_celt_side_frame(
                &mut self.celt_enc,
                &tail[n4 * ch..],
                n2,
                ch,
                0,
                false,
                redundant,
            );
        }
        let next_prev_mode = if to_celt { OpusMode::CeltOnly } else { mode };

        self.rc.done();

        // C opus_encoder.c:2170-2179: "In the unlikely case that the SILK
        // encoder busted its target, tell the decoder to call the PLC" — emit
        // a minimal packet with an EMPTY frame instead of failing hard. The
        // zero-length frame makes the decoder conceal (libopus pads the same
        // 2-byte packet with opus_packet_pad under CBR). With FEC the LBRR
        // section can squeeze the main frame below what even the
        // damage-control re-encode fits, which is exactly the case this
        // fallback exists for.
        if self.rc.error != 0 || self.rc.tell() > (n_bytes as i32 - 1) * 8 {
            if output.len() < 2 {
                return Err("Output buffer too small");
            }
            let mut ret = 2usize;
            if self.use_cbr && n_bytes >= 2 {
                // CBR: pad to the nominal size as a code-3 packet holding a
                // single EMPTY frame (opus_packet_pad of the 2-byte packet).
                let target_total = n_bytes.min(output.len());
                if target_total > 2 {
                    output[0] = toc | 0x03;
                    output[1] = 0x41; // 1 frame, padding present, CBR
                    let pad_amount = target_total - 2;
                    let nb_255s = (pad_amount - 1) / 255;
                    let mut ptr = 2;
                    for _ in 0..nb_255s {
                        output[ptr] = 255;
                        ptr += 1;
                    }
                    output[ptr] = (pad_amount - 255 * nb_255s - 1) as u8;
                    ptr += 1;
                    for b in &mut output[ptr..target_total] {
                        *b = 0;
                    }
                    self.prev_enc_mode = Some(next_prev_mode);
                    return Ok(target_total);
                }
                ret = target_total.max(1);
            }
            // VBR: a 2-byte code-0 packet whose frame length byte is 0.
            output[0] = toc;
            output[1] = 0;
            self.prev_enc_mode = Some(next_prev_mode);
            return Ok(ret);
        }
        self.range_final = self.rc.rng ^ redundant_rng;

        self.prev_enc_mode = Some(next_prev_mode);
        if mode == OpusMode::SilkOnly {
            let mut ret = silk_ret_bytes.min(self.rc.storage as usize);
            if redundancy_bytes > 0 {
                // The redundancy follows the SILK bytes, and the decoder finds
                // it from the frame length, so no zero byte may go.
                let rb = redundancy_bytes.min(self.rc.storage as usize - ret);
                self.rc.buf[ret..ret + rb].copy_from_slice(&redundant[..rb]);
                ret += rb;
            } else {
                // Trailing zero bytes of a SILK-only frame are implied.
                while ret > 2 && self.rc.buf[ret - 1] == 0 {
                    ret -= 1;
                }
            }

            let target_total = if self.use_cbr {
                n_bytes.min(output.len())
            } else {
                (ret + 1).min(output.len())
            };

            let silk_len = ret;

            if !self.use_cbr || silk_len + 1 >= target_total {
                // VBR or payload fills the target: simple code 0 packet
                output[0] = toc;
                let copy_len = silk_len.min(target_total - 1);
                output[1..1 + copy_len].copy_from_slice(&self.rc.buf[..copy_len]);
                return Ok((copy_len + 1).min(output.len()));
            }

            output[0] = toc | 0x03;

            if silk_len + 2 >= target_total {
                output[1] = 0x01;
                let copy_len = (target_total - 2).min(silk_len);
                output[2..2 + copy_len].copy_from_slice(&self.rc.buf[..copy_len]);
                return Ok(target_total.min(output.len()));
            }

            let pad_amount = target_total - silk_len - 2;
            output[1] = 0x41;

            let nb_255s = (pad_amount - 1) / 255;
            let mut ptr = 2;
            for _ in 0..nb_255s {
                output[ptr] = 255;
                ptr += 1;
            }
            output[ptr] = (pad_amount - 255 * nb_255s - 1) as u8;
            ptr += 1;

            output[ptr..ptr + silk_len].copy_from_slice(&self.rc.buf[..silk_len]);
            ptr += silk_len;

            let fill_end = target_total.min(output.len());
            for byte in output[ptr..fill_end].iter_mut() {
                *byte = 0;
            }

            return Ok(target_total.min(output.len()));
        }

        // For CELT/Hybrid, respect possible VBR shrink performed by CeltEncoder
        // (e.g. silence); the redundancy follows what CELT used
        // (opus_encoder.c:2502-2507).
        let payload_len = (self.rc.storage as usize).min(output.len() - 1);
        output[1..1 + payload_len].copy_from_slice(&self.rc.buf[..payload_len]);
        let end = 1 + payload_len + redundancy_bytes;
        if end > output.len() {
            return Err("Output buffer too small");
        }
        output[1 + payload_len..end].copy_from_slice(redundant);
        Ok(end)
    }

    /// Repacketize 40/60 ms CELT/Hybrid input as 20 ms sub-frames in one code
    /// 1/2/3 Opus packet (libopus `encode_multiframe_packet`). Every sub-frame
    /// is forced to `mode` so the packet carries a single config. A mode
    /// switch's redundancy goes to the first sub-frame, or going to CELT, to
    /// the last (opus_encoder.c:1781-1783).
    fn encode_multiframe(
        &mut self,
        input: PcmInput<'_>,
        frame_size: usize,
        mode: OpusMode,
        transition: FrameTransition,
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        const MAX_SUB: usize = 3; // up to 60 ms of 20 ms frames
        let ch = self.channels;
        let enc_frame = (self.sampling_rate / 50) as usize; // 20 ms
        if enc_frame == 0 || !frame_size.is_multiple_of(enc_frame) {
            return Err("Invalid multi-frame size");
        }
        let nb = frame_size / enc_frame;
        if !(2..=MAX_SUB).contains(&nb) {
            return Err("Unsupported multi-frame size");
        }

        // RFC 6716 size field: one byte below 252, else two.
        fn put_size(out: &mut [u8], pos: &mut usize, len: usize) -> Result<(), &'static str> {
            if len < 252 {
                if *pos >= out.len() {
                    return Err("Output buffer too small");
                }
                out[*pos] = len as u8;
                *pos += 1;
            } else {
                if *pos + 1 >= out.len() {
                    return Err("Output buffer too small");
                }
                let first = 252 + (len & 3);
                out[*pos] = first as u8;
                out[*pos + 1] = ((len - first) >> 2) as u8;
                *pos += 2;
            }
            Ok(())
        }

        let mut sub = [[0u8; 1276]; MAX_SUB];
        let mut lens = [0usize; MAX_SUB];
        self.forced_mode = Some(mode);
        let enc_res = (|| -> Result<(), &'static str> {
            for i in 0..nb {
                let start = i * enc_frame * ch;
                let end = start + enc_frame * ch;
                let sub_input = match input {
                    PcmInput::F32(s) => PcmInput::F32(&s[start..end]),
                    PcmInput::I16(s) => PcmInput::I16(&s[start..end]),
                };
                let to_celt = transition.to_celt && i == nb - 1;
                self.frame_transition = Some(FrameTransition {
                    redundancy: transition.redundancy
                        && (to_celt || (!transition.to_celt && i == 0)),
                    celt_to_silk: transition.celt_to_silk,
                    to_celt,
                    prefill: transition.prefill && i == 0,
                });
                let n = self.encode_impl(sub_input, enc_frame, &mut sub[i])?;
                if n == 0 {
                    return Err("Empty sub-frame");
                }
                lens[i] = n - 1;
            }
            Ok(())
        })();
        self.forced_mode = None;
        self.frame_transition = None;
        enc_res?;

        let toc = sub[0][0] & 0xFC;
        let mut pos = 0usize;
        let push =
            |out: &mut [u8], pos: &mut usize, byte: u8| -> Result<(), &'static str> {
                if *pos >= out.len() {
                    return Err("Output buffer too small");
                }
                out[*pos] = byte;
                *pos += 1;
                Ok(())
            };
        // Use explicit per-frame lengths (code 2 / code 3 VBR) for every
        // sub-frame packet: it is always valid, and avoids the code-3 CBR
        // assumption that all frames are byte-identical in size.
        if nb == 2 {
            push(output, &mut pos, toc | 0x02)?;
            put_size(output, &mut pos, lens[0])?;
        } else {
            push(output, &mut pos, toc | 0x03)?;
            push(output, &mut pos, nb as u8 | 0x80)?;
            for i in 0..nb - 1 {
                put_size(output, &mut pos, lens[i])?;
            }
        }

        for i in 0..nb {
            let payload = &sub[i][1..1 + lens[i]];
            if pos + payload.len() > output.len() {
                return Err("Output buffer too small");
            }
            output[pos..pos + payload.len()].copy_from_slice(payload);
            pos += payload.len();
        }
        Ok(pos)
    }
}

pub struct OpusDecoder {
    #[cfg(not(feature = "heap"))]
    celt_dec: CeltDecoder,
    #[cfg(feature = "heap")]
    celt_dec: Box<CeltDecoder>,
    #[cfg(not(feature = "heap"))]
    silk_dec: silk::dec_api::SilkDecoder,
    #[cfg(feature = "heap")]
    silk_dec: Box<silk::dec_api::SilkDecoder>,
    sampling_rate: i32,
    channels: usize,

    prev_mode: Option<OpusMode>,

    /// Whether the previous frame had redundancy (mode transition marker).
    prev_redundancy: bool,
    frame_size: usize,

    bandwidth: Bandwidth,

    stream_channels: usize,

    silk_resampler: silk::resampler::SilkResampler,

    /// Second resampler instance for stereo channel 1.
    silk_resampler_2: silk::resampler::SilkResampler,

    prev_internal_rate: i32,

    /// Range decoder state after the last decoded frame (libopus
    /// `OPUS_GET_FINAL_RANGE`), 0 after a concealed one.
    range_final: u32,

    pub hybrid_skip_celt: bool,

    #[cfg(not(feature = "heap"))]
    w_pcm_i16: FixedVec<i16, OPUS_PCM_I16>,
    #[cfg(feature = "heap")]
    w_pcm_i16: Box<FixedVec<i16, OPUS_PCM_I16>>,
    #[cfg(not(feature = "heap"))]
    w_silk_out: FixedVec<f32, OPUS_SUBFRAME_SCRATCH>,
    #[cfg(feature = "heap")]
    w_silk_out: Box<FixedVec<f32, OPUS_SUBFRAME_SCRATCH>>,
    #[cfg(not(feature = "heap"))]
    w_pcm_resampled: FixedVec<i16, OPUS_SUBFRAME_SCRATCH>,
    #[cfg(feature = "heap")]
    w_pcm_resampled: Box<FixedVec<i16, OPUS_SUBFRAME_SCRATCH>>,
    #[cfg(not(feature = "heap"))]
    w_celt_planar: FixedVec<f32, OPUS_SUBFRAME_SCRATCH>,
    #[cfg(feature = "heap")]
    w_celt_planar: Box<FixedVec<f32, OPUS_SUBFRAME_SCRATCH>>,
    #[cfg(not(feature = "heap"))]
    w_celt_out: FixedVec<f32, OPUS_SUBFRAME_SCRATCH>,
    #[cfg(feature = "heap")]
    w_celt_out: Box<FixedVec<f32, OPUS_SUBFRAME_SCRATCH>>,

    /// Tail of the previous frame's output, used for smooth_fade at mode
    /// transitions (libopus pcm_transition + smooth_fade).
    #[cfg(not(feature = "heap"))]
    prev_pcm_tail: FixedVec<f32, OPUS_PCM_TAIL>,
    #[cfg(feature = "heap")]
    prev_pcm_tail: Box<FixedVec<f32, OPUS_PCM_TAIL>>,
}

impl OpusDecoder {
    pub fn new(sampling_rate: i32, channels: usize) -> Result<Self, &'static str> {
        let mut decoder = MaybeUninit::uninit();
        Self::init(&mut decoder, sampling_rate, channels)?;
        Ok(unsafe { decoder.assume_init() })
    }

    /// Builds a decoder in `slot` instead of returning it.
    ///
    /// Without the `heap` feature the decoder is about 175 KB, and `new` returns it
    /// through the stack. `init` writes it straight into memory you provide, such as
    /// a `static`, so it never has to fit on the stack.
    pub fn init(
        slot: &mut MaybeUninit<Self>,
        sampling_rate: i32,
        channels: usize,
    ) -> Result<&mut Self, &'static str> {
        if ![8000, 12000, 16000, 24000, 48000].contains(&sampling_rate) {
            return Err("Invalid sampling rate");
        }
        if ![1, 2].contains(&channels) {
            return Err("Invalid number of channels");
        }

        let mode = modes::default_mode();
        let this = slot.as_mut_ptr();
        unsafe {
            #[cfg(feature = "heap")]
            {
                (&raw mut (*this).celt_dec).write(Box::new(CeltDecoder::new(mode, channels, sampling_rate)));
                (&raw mut (*this).silk_dec).write(Box::new(silk::dec_api::SilkDecoder::new()));
            }
            #[cfg(not(feature = "heap"))]
            {
                CeltDecoder::write_new(&raw mut (*this).celt_dec, mode, channels, sampling_rate);
                silk::dec_api::SilkDecoder::write_new(&raw mut (*this).silk_dec);
            }
            (&raw mut (*this).sampling_rate).write(sampling_rate);
            (&raw mut (*this).channels).write(channels);
            (&raw mut (*this).prev_mode).write(None);
            (&raw mut (*this).prev_redundancy).write(false);
            (&raw mut (*this).frame_size).write(0);
            (&raw mut (*this).bandwidth).write(Bandwidth::Auto);
            (&raw mut (*this).stream_channels).write(channels);
            (&raw mut (*this).silk_resampler).write(silk::resampler::SilkResampler::default());
            (&raw mut (*this).silk_resampler_2).write(silk::resampler::SilkResampler::default());
            (&raw mut (*this).prev_internal_rate).write(0);
            (&raw mut (*this).range_final).write(0);
            (&raw mut (*this).hybrid_skip_celt).write(false);
            #[cfg(feature = "heap")]
            {
                (&raw mut (*this).w_pcm_i16).write(Box::new(FixedVec::from_value(0i16, 960 * channels)));
                (&raw mut (*this).w_silk_out).write(Box::new(FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels)));
                (&raw mut (*this).w_pcm_resampled).write(Box::new(FixedVec::from_value(0i16, OPUS_MAX_SUBFRAME * channels)));
                (&raw mut (*this).w_celt_planar).write(Box::new(FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels)));
                (&raw mut (*this).w_celt_out).write(Box::new(FixedVec::from_value(0.0f32, OPUS_MAX_SUBFRAME * channels)));
                (&raw mut (*this).prev_pcm_tail).write(Box::new(FixedVec::from_value(0.0f32, 240 * channels)));
            }
            #[cfg(not(feature = "heap"))]
            {
                FixedVec::write_from_value(&raw mut (*this).w_pcm_i16, 0i16, 960 * channels);
                FixedVec::write_from_value(&raw mut (*this).w_silk_out, 0.0f32, OPUS_MAX_SUBFRAME * channels);
                FixedVec::write_from_value(&raw mut (*this).w_pcm_resampled, 0i16, OPUS_MAX_SUBFRAME * channels);
                FixedVec::write_from_value(&raw mut (*this).w_celt_planar, 0.0f32, OPUS_MAX_SUBFRAME * channels);
                FixedVec::write_from_value(&raw mut (*this).w_celt_out, 0.0f32, OPUS_MAX_SUBFRAME * channels);
                FixedVec::write_from_value(&raw mut (*this).prev_pcm_tail, 0.0f32, 240 * channels);
            }
        }

        let decoder = unsafe { slot.assume_init_mut() };
        decoder.silk_dec.init(sampling_rate.min(16000), channels as i32);
        decoder.silk_dec.channel_state[0].fs_api_hz = sampling_rate;
        Ok(decoder)
    }

    /// The range decoder's final state after the last frame decoded, like
    /// libopus `OPUS_GET_FINAL_RANGE`: it equals the encoder's final range
    /// for that frame (the value an Opus test vector stores with each
    /// packet), XORed with the redundant frame's when the frame has one. It
    /// is 0 after a frame that was concealed (lost, empty or a single byte).
    pub fn final_range(&self) -> u32 {
        self.range_final
    }

    /// Decode one Opus packet into interleaved `f32` PCM, returning the
    /// number of samples per channel written.
    ///
    /// Packet loss follows libopus: pass an empty packet (`&[]`) to conceal
    /// `frame_size` samples (a multiple of 2.5 ms), or a ToC-only 1-byte
    /// packet to conceal the duration that ToC describes. Either way the
    /// frame is concealed in the previous packet's mode; the placeholder's
    /// mode, bandwidth and channel bits are ignored.
    ///
    /// A packet's channel count need not match the decoder's, as in libopus:
    /// a stereo decoder plays a mono packet on both channels, and a mono
    /// decoder downmixes a stereo packet (SILK plays the mid). libopus
    /// encoders send mono packets in stereo streams at low bitrates.
    pub fn decode(
        &mut self,
        input: &[u8],
        frame_size: usize,
        output: &mut [f32],
    ) -> Result<usize, &'static str> {
        // A packet of 0 or 1 bytes (ToC only) is a lost/DTX frame: conceal it
        // in the previous mode without touching the mode state (issue #15).
        if input.len() <= 1 {
            return self.decode_lost(input.first().copied(), frame_size, output);
        }

        let toc = input[0];
        let mode = mode_from_toc(toc);
        let packet_channels = channels_from_toc(toc);
        let bandwidth = bandwidth_from_toc(toc);
        let frame_duration_ms = frame_duration_ms_from_toc(toc);

        let frame_payloads = frame_payloads(input)?;
        let frame_count = frame_payloads.len();

        self.frame_size = frame_size;
        self.bandwidth = bandwidth;

        // Derive the actual per-frame sample count from the TOC, not from the
        // caller's frame_size. This prevents panics in bands.rs/celt.rs when
        // the caller passes a mismatched frame_size (issue #7 sub-item 1):
        // the internal decoders always get the correct geometry.
        let toc_frame_size = frame_samples_from_toc(toc, self.sampling_rate)
            .ok_or("Invalid TOC for sampling rate")?;
        let decoded_total = toc_frame_size * frame_count;
        if frame_size < decoded_total {
            return Err("frame_size too small for packet");
        }
        if output.len() < decoded_total * self.channels {
            return Err("Output buffer too small for packet");
        }
        // The packet's channel count steers both layers from here on.
        self.stream_channels = packet_channels;
        self.celt_dec.set_stream_channels(packet_channels);
        // Zero-fill any extra space the caller provided beyond what the packet
        // actually produces, so stale data is never left in the buffer.
        if output.len() > decoded_total * self.channels {
            for v in &mut output[decoded_total * self.channels..] {
                *v = 0.0;
            }
        }
        let sub_frame_size = toc_frame_size;
        let sub_output_len = sub_frame_size * self.channels;

        // Detect mode transition and reset CELT decoder state to prevent
        // cross-mode artifacts (libopus opus_decoder.c:602-604).
        // This is the primary fix for issue #8/#9 alignment divergence:
        // stale CELT MDCT/prefilter state at SILK↔CELT boundaries causes
        // discontinuities that accumulate across transitions.
        let mode_transition = matches!(
            self.prev_mode,
            Some(prev) if prev != mode && !self.prev_redundancy
        );
        // The transitions libopus's opus_decode_frame bridges with 5 ms of the old mode concealed
        // forward: into CELT from SILK or Hybrid (unless that frame's redundancy did it), and out of CELT. A Hybrid
        // frame carrying redundancy clears it below (libopus: `if (redundancy) transition = 0`).
        let mut transition = match self.prev_mode {
            Some(prev) => {
                (mode == OpusMode::CeltOnly && prev != OpusMode::CeltOnly && !self.prev_redundancy)
                    || (mode != OpusMode::CeltOnly && prev == OpusMode::CeltOnly)
            }
            None => false,
        };
        let from_celt = self.prev_mode == Some(OpusMode::CeltOnly) && mode != OpusMode::CeltOnly;
        if from_celt {
            // SILK starts over, as libopus's silk_ResetDecoder on a switch from CELT (its resampler too: the reset
            // clears the SILK rate, so the next set_fs initialises it again).
            self.silk_dec.reset();
            self.prev_internal_rate = 0;
        }
        // A SILK-only frame decides its out-of-CELT bridge only after its SILK
        // layer runs: when the frame carries a redundancy tail there is nothing
        // to bridge (libopus: `if (redundancy) transition = 0`), and without one
        // the old CELT state conceals forward from inside the branch below.
        // Only a switch into CELT resets CELT here (libopus's reset at
        // opus_decode_frame 551-553 sits inside `mode != MODE_SILK_ONLY`): a
        // Hybrid frame resets it after its CELT-to-SILK redundant frame is
        // decoded from the old state, and a SILK-only frame resets it right
        // before its own redundant frame, if it carries one.
        if mode_transition && mode == OpusMode::CeltOnly {
            self.celt_dec.reset_state();
        }

        // Generate SILK PLC audio for the mode-transition bridge. libopus
        // synthesizes 5ms (F5) of pitch-extrapolated audio in the OLD mode
        // (opus_decoder.c:387-391) and crossfades it with the new frame. We
        // reuse the F5-sized prev_pcm_tail buffer for this bridge.
        let f5_bridge = self.sampling_rate as usize / 200; // F5 = Fs/200
        if transition
            && mode == OpusMode::CeltOnly
            && f5_bridge > 0
            && matches!(
                self.prev_mode,
                Some(OpusMode::SilkOnly) | Some(OpusMode::Hybrid)
            )
            && self.prev_internal_rate > 0
        {
            let internal_rate = self.prev_internal_rate;
            let plc_internal_len = (10 * internal_rate / 1000) as usize;
            let mut plc_rc = RangeCoder::new_decoder(&[]);
            let mut plc_i16: FixedVec<i16, OPUS_PCM_I16> =
                FixedVec::from_value(0i16, plc_internal_len * self.channels);
            let n = self.silk_dec.decode(
                &mut plc_rc,
                &mut plc_i16,
                silk::decode_frame::FLAG_PACKET_LOST,
                true,
                10,
                internal_rate,
            );
            if n > 0 {
                let bridge_ch = f5_bridge * self.channels;
                let bridge_len = bridge_ch.min(self.prev_pcm_tail.len());
                if internal_rate == self.sampling_rate {
                    // No resampling: copy PLC samples directly (planar). A
                    // mono stream fills both channels of a stereo decoder.
                    let n_us = n as usize;
                    let silk_channels = self
                        .channels
                        .min(self.silk_dec.n_channels_internal as usize);
                    for ch in 0..self.channels {
                        let src_base = ch.min(silk_channels - 1) * n_us;
                        for i in 0..(bridge_len / self.channels).min(n_us) {
                            let dst = i * self.channels + ch;
                            if dst < bridge_len {
                                self.prev_pcm_tail[dst] = plc_i16[src_base + i] as f32 / 32768.0;
                            }
                        }
                    }
                } else if self.silk_resampler.is_initialized() {
                    // Resample channel 0 to the API rate for the bridge.
                    // The resampler must receive an output buffer sized for the
                    // FULL resampled length (issue #15): truncating the buffer to
                    // `f5_bridge` caused out-of-bounds writes in the Up2HQ/Copy
                    // paths. We size it fully, then copy only the bridge window.
                    let ratio = self.sampling_rate as f64 / internal_rate as f64;
                    let full_len = ((n as f64 * ratio) as usize).max(1);
                    let out_len = full_len.min(f5_bridge);
                    let n_us = n as usize;
                    let mut resampled: FixedVec<i16, OPUS_MAX_FRAME> =
                        FixedVec::from_value(0i16, full_len);
                    self.silk_resampler
                        .process(&mut resampled, &plc_i16[..n_us], n);
                    for i in 0..out_len {
                        if i < bridge_len / self.channels {
                            for ch in 0..self.channels {
                                self.prev_pcm_tail[i * self.channels + ch] =
                                    resampled[i] as f32 / 32768.0;
                            }
                        }
                    }
                }
            }
        }

        // Track whether this packet uses Hybrid redundancy.
        let mut has_redundancy = false;

        match mode {
            OpusMode::SilkOnly => {
                let internal_sample_rate = match bandwidth {
                    Bandwidth::Narrowband => 8000,
                    Bandwidth::Mediumband => 12000,
                    Bandwidth::Wideband => 16000,
                    _ => 16000,
                };
                if internal_sample_rate != self.prev_internal_rate
                    || !self.silk_resampler.is_initialized()
                {
                    self.silk_resampler
                        .init(internal_sample_rate, self.sampling_rate);
                    self.silk_resampler_2
                        .init(internal_sample_rate, self.sampling_rate);
                }
                // Always track the SILK internal rate so the mode-transition
                // PLC bridge can be generated (even when no resampling is
                // needed, e.g. 16kHz decoder + SILK WB).
                self.prev_internal_rate = internal_sample_rate;

                // 5 ms redundant frames and their crossfade window, shared by
                // every frame of the packet. The redundant frame ends where
                // the packet's bandwidth does (libopus 499-524): narrowband
                // 13, mediumband and wideband 17.
                let f5 = self.sampling_rate as usize / 200;
                let f2_5 = f5 / 2;
                let redundant_end_band = match bandwidth {
                    Bandwidth::Narrowband => 13,
                    Bandwidth::Mediumband | Bandwidth::Wideband => 17,
                    _ => 21,
                };
                let window = modes::default_mode().window;
                let inc = (48000 / self.sampling_rate) as usize;

                for (fi, payload) in frame_payloads.iter().enumerate() {
                    let out_start = fi * sub_output_len;
                    // C opus_decode_frame: a zero-length frame (e.g. the
                    // encoder's busted-budget fallback packet) is concealed,
                    // not decoded.
                    let flag = if payload.is_empty() {
                        silk::decode_frame::FLAG_PACKET_LOST
                    } else {
                        silk::decode_frame::FLAG_DECODE_NORMAL
                    };
                    let mut rc = RangeCoder::new_decoder(payload);
                    self.decode_silk_frames(
                        &mut rc,
                        flag,
                        frame_duration_ms,
                        internal_sample_rate,
                        sub_frame_size,
                    )?;
                    output[out_start..out_start + sub_output_len]
                        .copy_from_slice(&self.w_silk_out[..sub_output_len]);

                    // A SILK-only frame's redundancy is implicit (libopus
                    // opus_decode_frame 452-479): no flag is coded — when bits
                    // are left for it, redundancy is on, `celt_to_silk` reads
                    // as one bit, and the frame's tail past the SILK layer is
                    // a 5 ms redundant CELT frame. Going to CELT it decodes
                    // from a reset CELT state — which the next CELT frame
                    // carries on — and its second half fades into the frame's
                    // last 2.5 ms (595-603). Coming from CELT it decodes from
                    // the old state, opens the frame and fades into its first
                    // 5 ms (608-617). Only the SILK-to-CELT direction counts
                    // as `prev_redundancy` (657), which spares the next frame
                    // its transition bridge and CELT reset (346-352, 551-553).
                    let len = payload.len();
                    let mut redundancy_bytes = 0usize;
                    let mut celt_to_silk = false;
                    let mut redundant = [0.0f32; REDUNDANT_MAX];
                    let mut redundant_rng = 0u32;
                    let has_tail = !payload.is_empty() && rc.tell() + 17 <= (8 * len) as i32;
                    if has_tail {
                        celt_to_silk = rc.decode_bit_logp(1);
                        // The tail is everything the SILK layer left:
                        // `redundancy_bytes = len - ceil(ec_tell/8)`.
                        redundancy_bytes = len - ((rc.tell() as usize + 7) >> 3);
                        // Keep the main range coder in sync, as libopus's
                        // `dec.storage -= redundancy_bytes`.
                        rc.storage -= redundancy_bytes as u32;
                        if celt_to_silk {
                            // From the old CELT state, before any reset — a
                            // CELT-to-SILK switch's bridge (532-541).
                            redundant_rng = self.decode_redundant(
                                &payload[len - redundancy_bytes..],
                                redundant_end_band,
                                f5,
                                &mut redundant,
                            );
                        }
                    }
                    if fi == 0 {
                        if has_tail {
                            // libopus: `if (redundancy) transition = 0`.
                            transition = false;
                        } else if from_celt {
                            // Without a tail there is nothing to fade into:
                            // bridge out of CELT with 5 ms of the old CELT
                            // state concealed forward (libopus's
                            // `pcm_transition`), not the previous frame's last
                            // 5 ms, which already played.
                            self.conceal_celt_bridge()?;
                        }
                    }
                    let ch = self.channels;
                    if has_tail && !celt_to_silk && f5 > 0 && sub_output_len >= f5 * ch {
                        // SILK to CELT: reset CELT, decode the redundant frame
                        // with all bands from that fresh state — the next CELT
                        // frame carries it on — and fade its second half into
                        // the frame's last 2.5 ms (595-603).
                        self.celt_dec.reset_state();
                        redundant_rng = self.decode_redundant(
                            &payload[len - redundancy_bytes..],
                            redundant_end_band,
                            f5,
                            &mut redundant,
                        );
                        let at = out_start + (sub_frame_size - f2_5) * ch;
                        let mut tail = [0.0f32; REDUNDANT_MAX];
                        tail[..f2_5 * ch].copy_from_slice(&output[at..at + f2_5 * ch]);
                        smooth_fade(
                            &tail[..f2_5 * ch],
                            &redundant[f2_5 * ch..f5 * ch],
                            &mut output[at..at + f2_5 * ch],
                            f2_5,
                            ch,
                            window,
                            inc,
                        );
                    } else if has_tail
                        && celt_to_silk
                        && (self.prev_mode != Some(OpusMode::SilkOnly) || self.prev_redundancy)
                        && f5 > 0
                        && sub_output_len >= f5 * ch
                    {
                        // CELT to SILK: the redundant frame opens this one and
                        // fades into its first 5 ms (608-617). It is ignored
                        // when the CELT state it came from is stale (606-608).
                        output[out_start..out_start + f2_5 * ch]
                            .copy_from_slice(&redundant[..f2_5 * ch]);
                        let at = out_start + f2_5 * ch;
                        let mut next = [0.0f32; REDUNDANT_MAX];
                        next[..f2_5 * ch].copy_from_slice(&output[at..at + f2_5 * ch]);
                        smooth_fade(
                            &redundant[f2_5 * ch..f5 * ch],
                            &next[..f2_5 * ch],
                            &mut output[at..at + f2_5 * ch],
                            f2_5,
                            ch,
                            window,
                            inc,
                        );
                    }
                    has_redundancy = has_tail && !celt_to_silk;
                    self.range_final = if len <= 1 { 0 } else { rc.rng ^ redundant_rng };
                }
                decoded_total
            }

            OpusMode::CeltOnly => {
                let celt_end_band = self.celt_end_band_from_toc(toc);

                for (fi, payload) in frame_payloads.iter().enumerate() {
                    let mut rc = RangeCoder::new_decoder(payload);
                    let total_bits = (payload.len() * 8) as i32;
                    let needed = sub_frame_size * self.channels;
                    let out_start = fi * needed;
                    let out_end = (out_start + needed).min(output.len());

                    if output.len() < out_end {
                        return Err("Output buffer too small");
                    }

                    if self.channels == 1 {
                        self.celt_dec.decode_from_range_coder_with_band_range(
                            &mut rc,
                            total_bits,
                            sub_frame_size,
                            &mut output[out_start..out_end],
                            0,
                            celt_end_band,
                        );
                        for sample in &mut output[out_start..out_end] {
                            *sample = sample.clamp(-1.0, 1.0);
                        }
                    } else {
                        self.celt_dec.decode_from_range_coder_with_band_range(
                            &mut rc,
                            total_bits,
                            sub_frame_size,
                            &mut self.w_celt_planar[..needed],
                            0,
                            celt_end_band,
                        );
                        for i in 0..sub_frame_size {
                            for ch in 0..self.channels {
                                let idx = out_start + i * self.channels + ch;
                                output[idx] =
                                    self.w_celt_planar[ch * sub_frame_size + i].clamp(-1.0, 1.0);
                            }
                        }
                    }
                    self.range_final = if payload.len() <= 1 { 0 } else { rc.rng };
                }
                decoded_total
            }

            OpusMode::Hybrid => {
                let internal_sample_rate = 16000;
                let celt_end_band = self.celt_end_band_from_toc(toc);

                if internal_sample_rate != self.prev_internal_rate
                    || !self.silk_resampler.is_initialized()
                {
                    self.silk_resampler
                        .init(internal_sample_rate, self.sampling_rate);
                    self.silk_resampler_2
                        .init(internal_sample_rate, self.sampling_rate);
                }
                self.prev_internal_rate = internal_sample_rate;

                for (fi, payload) in frame_payloads.iter().enumerate() {
                    let mut rc = RangeCoder::new_decoder(payload);
                    // SILK layer of this frame -> w_silk_out (interleaved, API rate).
                    // A zero-length frame is concealed, not decoded (C
                    // opus_decode_frame).
                    let flag = if payload.is_empty() {
                        silk::decode_frame::FLAG_PACKET_LOST
                    } else {
                        silk::decode_frame::FLAG_DECODE_NORMAL
                    };
                    self.decode_silk_frames(
                        &mut rc,
                        flag,
                        frame_duration_ms,
                        internal_sample_rate,
                        sub_frame_size,
                    )?;
                    let silk_out_len = sub_frame_size * self.channels;

                    // A mode switch's redundancy (RFC 6716 §4.5.1; libopus
                    // opus_decode_frame): after the SILK layer, when enough bits
                    // are left, a flag; if set, which way the switch goes and the
                    // size of a 5 ms CELT frame carried at the packet's end. The
                    // main CELT layer reads only the bytes before it; the
                    // redundant frame is decoded on its own and crossfaded in
                    // below.
                    let mut len = payload.len();
                    let mut redundancy = false;
                    let mut celt_to_silk = false;
                    let mut redundancy_bytes = 0usize;
                    if !payload.is_empty() && rc.tell() + 17 + 20 <= (8 * len) as i32 {
                        redundancy = rc.decode_bit_logp(12);
                        if redundancy {
                            celt_to_silk = rc.decode_bit_logp(1);
                            redundancy_bytes = rc.dec_uint(256) as usize + 2;
                            if redundancy_bytes > len
                                || ((len - redundancy_bytes) * 8) < rc.tell() as usize
                            {
                                len = 0;
                                redundancy_bytes = 0;
                                redundancy = false;
                            } else {
                                len -= redundancy_bytes;
                                rc.storage -= redundancy_bytes as u32;
                            }
                        }
                    }
                    // libopus remembers only a SILK-to-CELT frame's redundancy (`prev_redundancy = redundancy &&
                    // !celt_to_silk`).
                    has_redundancy = redundancy && !celt_to_silk;
                    let f5 = self.sampling_rate as usize / 200;
                    let f2_5 = f5 / 2;
                    let mut redundant = [0.0f32; REDUNDANT_MAX];
                    let mut redundant_rng = 0u32;
                    if fi == 0 {
                        if redundancy {
                            transition = false;
                        } else if from_celt {
                            self.conceal_celt_bridge()?;
                        }
                    }
                    if redundancy && celt_to_silk {
                        redundant_rng = self.decode_redundant(
                            &payload[len..len + redundancy_bytes],
                            celt_end_band,
                            f5,
                            &mut redundant,
                        );
                    }
                    if fi == 0 && mode_transition {
                        self.celt_dec.reset_state();
                    }

                    let total_bits = (len * 8) as i32;
                    if len <= 1 {
                        self.w_celt_out[..silk_out_len].fill(0.0);
                    } else {
                        let (celt_dec, celt_planar) = (
                            state_mut(&mut self.celt_dec),
                            state_mut(&mut self.w_celt_planar),
                        );
                        celt_dec.decode_from_range_coder_with_band_range(
                            &mut rc,
                            total_bits,
                            sub_frame_size,
                            &mut celt_planar[..silk_out_len],
                            17,
                            celt_end_band,
                        );

                        if self.channels == 1 {
                            self.w_celt_out[..silk_out_len]
                                .copy_from_slice(&self.w_celt_planar[..silk_out_len]);
                        } else {
                            for i in 0..sub_frame_size {
                                for ch in 0..self.channels {
                                    self.w_celt_out[i * self.channels + ch] =
                                        self.w_celt_planar[ch * sub_frame_size + i];
                                }
                            }
                        }
                    }

                    let out_start = fi * silk_out_len;
                    let total = silk_out_len.min(output.len() - out_start);
                    for j in 0..total {
                        output[out_start + j] =
                            (self.w_silk_out[j] + self.w_celt_out[j]).clamp(-1.0, 1.0);
                    }

                    // The redundant frame crossfaded in (libopus
                    // opus_decode_frame): going SILK to CELT, the frame's last
                    // 2.5 ms fade into its second half (decoded from a fresh CELT
                    // state, which the next CELT frame then carries on from);
                    // going CELT to SILK, its first half opens the frame and its
                    // second half fades into the frame's next 2.5 ms.
                    let ch = self.channels;
                    let use_redundant = !celt_to_silk
                        || self.prev_mode != Some(OpusMode::SilkOnly)
                        || self.prev_redundancy;
                    if redundancy && use_redundant && f5 > 0 && total >= f5 * ch {
                        let window = modes::default_mode().window;
                        let inc = (48000 / self.sampling_rate) as usize;
                        if !celt_to_silk {
                            self.celt_dec.reset_state();
                            redundant_rng = self.decode_redundant(
                                &payload[len..len + redundancy_bytes],
                                celt_end_band,
                                f5,
                                &mut redundant,
                            );
                            let at = out_start + (sub_frame_size - f2_5) * ch;
                            let mut tail = [0.0f32; REDUNDANT_MAX];
                            tail[..f2_5 * ch].copy_from_slice(&output[at..at + f2_5 * ch]);
                            smooth_fade(
                                &tail[..f2_5 * ch],
                                &redundant[f2_5 * ch..f5 * ch],
                                &mut output[at..at + f2_5 * ch],
                                f2_5,
                                ch,
                                window,
                                inc,
                            );
                        } else {
                            output[out_start..out_start + f2_5 * ch]
                                .copy_from_slice(&redundant[..f2_5 * ch]);
                            let at = out_start + f2_5 * ch;
                            let mut next = [0.0f32; REDUNDANT_MAX];
                            next[..f2_5 * ch].copy_from_slice(&output[at..at + f2_5 * ch]);
                            smooth_fade(
                                &redundant[f2_5 * ch..f5 * ch],
                                &next[..f2_5 * ch],
                                &mut output[at..at + f2_5 * ch],
                                f2_5,
                                ch,
                                window,
                                inc,
                            );
                        }
                    }
                    self.range_final = if len <= 1 { 0 } else { rc.rng ^ redundant_rng };
                }
                decoded_total
            }
        };

        // Apply PLC-style bridging at mode transitions (libopus
        // opus_decoder.c:660-679). The first F2_5 of the output is replaced
        // with the previous frame's tail (PLC bridge), and the next F2_5 is
        // crossfaded between the bridge and the new frame's CELT output.
        // F5 = Fs/200, F2_5 = Fs/400.
        let f2_5 = self.sampling_rate as usize / 400;
        let f5 = f2_5 * 2;
        if transition && f5 > 0 && decoded_total >= f5 {
            let window = modes::default_mode().window;
            let inc = (48000 / self.sampling_rate) as usize;
            let f2_5_ch = f2_5 * self.channels;
            let f5_ch = f5 * self.channels;
            // First F2_5: pure bridging audio from previous frame's tail.
            output[..f2_5_ch].copy_from_slice(&self.prev_pcm_tail[..f2_5_ch]);
            // Next F2_5: crossfade bridge → new CELT output.
            let new_mid: FixedVec<f32, OPUS_PCM_TAIL> =
                FixedVec::from_slice(&output[f2_5_ch..f5_ch]);
            smooth_fade(
                &self.prev_pcm_tail[f2_5_ch..f5_ch],
                &new_mid,
                &mut output[f2_5_ch..f5_ch],
                f2_5,
                self.channels,
                window,
                inc,
            );
        }

        self.save_pcm_tail(output, decoded_total);
        self.prev_mode = Some(mode);
        self.prev_redundancy = has_redundancy;
        Ok(decoded_total)
    }

    /// Rebuild a lost packet from the in-band FEC of the packet after it,
    /// into interleaved `f32` PCM, returning the number of samples per
    /// channel written (always `frame_size`). This is libopus's
    /// `opus_decode(..., decode_fec = 1)`.
    ///
    /// `input` is the packet that followed the loss. When it is SILK-only or
    /// Hybrid and its first frame carries LBRR (the encoder's
    /// `use_inband_fec`), the last frame's worth of `frame_size` (one frame
    /// of `input`, 10 to 60 ms) is rebuilt from that LBRR, and anything
    /// before it is concealed. A SILK frame without LBRR is concealed too,
    /// and all of `frame_size` is, as for `decode(&[], frame_size, ..)`, when
    /// `input` is CELT-only, follows a CELT packet, or has frames longer than
    /// `frame_size`. Hybrid uses only the SILK layer's LBRR, with the CELT
    /// layer concealed. `frame_size` is a multiple of 2.5 ms.
    ///
    /// `input` itself is not consumed: decode it with [`Self::decode`] next.
    pub fn decode_fec(
        &mut self,
        input: &[u8],
        frame_size: usize,
        output: &mut [f32],
    ) -> Result<usize, &'static str> {
        if input.len() <= 1 {
            return self.decode_lost(None, frame_size, output);
        }
        let toc = input[0];
        let mode = mode_from_toc(toc);
        let packet_frame = frame_samples_from_toc(toc, self.sampling_rate)
            .ok_or("Invalid TOC for sampling rate")?;
        let payloads = frame_payloads(input)?;
        // No FEC can be present: conceal it all (opus_decode_native).
        if frame_size < packet_frame
            || payloads[0].len() <= 1
            || mode == OpusMode::CeltOnly
            || self.prev_mode == Some(OpusMode::CeltOnly)
        {
            return self.decode_lost(None, frame_size, output);
        }
        let channels = self.channels;
        if output.len() < frame_size * channels {
            return Err("Output buffer too small for packet");
        }
        // Conceal everything but the duration the packet's FEC covers.
        let head = frame_size - packet_frame;
        if head > 0 {
            self.decode_lost(None, head, output)?;
        }

        let bandwidth = bandwidth_from_toc(toc);
        let packet_channels = channels_from_toc(toc);
        self.frame_size = frame_size;
        self.bandwidth = bandwidth;
        self.stream_channels = packet_channels;
        self.celt_dec.set_stream_channels(packet_channels);
        let internal_rate = match (mode, bandwidth) {
            (OpusMode::SilkOnly, Bandwidth::Narrowband) => 8000,
            (OpusMode::SilkOnly, Bandwidth::Mediumband) => 12000,
            _ => 16000,
        };
        if internal_rate != self.prev_internal_rate || !self.silk_resampler.is_initialized() {
            self.silk_resampler.init(internal_rate, self.sampling_rate);
            self.silk_resampler_2
                .init(internal_rate, self.sampling_rate);
        }
        self.prev_internal_rate = internal_rate;

        // The first frame's LBRR, frame by frame; a SILK frame without LBRR
        // is concealed (silk_decode_frame).
        let mut rc = RangeCoder::new_decoder(payloads[0]);
        self.decode_silk_frames(
            &mut rc,
            silk::decode_frame::FLAG_DECODE_LBRR,
            frame_duration_ms_from_toc(toc),
            internal_rate,
            packet_frame,
        )?;
        self.range_final = rc.rng;
        let len = packet_frame * channels;
        if mode == OpusMode::Hybrid {
            // FEC never reads the CELT layer: conceal it above SILK's band,
            // from a reset state unless the last packet was Hybrid too
            // (opus_decode_frame 551-553).
            if self.prev_mode.is_some_and(|m| m != OpusMode::Hybrid) && !self.prev_redundancy {
                self.celt_dec.reset_state();
            }
            self.conceal_celt(packet_frame, 17)?;
        } else {
            self.w_celt_out[..len].fill(0.0);
        }
        let tail = &mut output[head * channels..frame_size * channels];
        for (i, d) in tail.iter_mut().enumerate() {
            *d = (self.w_silk_out[i] + self.w_celt_out[i]).clamp(-1.0, 1.0);
        }
        output[frame_size * channels..].fill(0.0);

        self.save_pcm_tail(output, frame_size);
        self.prev_mode = Some(mode);
        self.prev_redundancy = false;
        Ok(frame_size)
    }

    /// Conceal a lost frame (libopus `opus_decode_frame` with `data == NULL`).
    ///
    /// Only the duration comes from the caller: the placeholder ToC of a
    /// 1-byte packet, or `frame_size` for an empty one. Mode, bandwidth, SILK
    /// rate and resamplers stay the previous packet's, so a placeholder such
    /// as `[0]` (SILK NB 10 ms) can no longer switch a CELT or wideband stream
    /// into another mode and fake two mode transitions (issue #15).
    fn decode_lost(
        &mut self,
        toc: Option<u8>,
        frame_size: usize,
        output: &mut [f32],
    ) -> Result<usize, &'static str> {
        let f2_5 = self.sampling_rate as usize / 400;
        let (f5, f10, f20) = (2 * f2_5, 4 * f2_5, 8 * f2_5);

        let total = match toc {
            None => {
                if frame_size == 0 || !frame_size.is_multiple_of(f2_5) {
                    return Err("Lost frame: frame_size must be a multiple of 2.5 ms");
                }
                frame_size
            }
            Some(toc) => {
                // A ToC-only packet: code 0 is one empty frame, codes 1 and 2
                // are two; code 3 lacks its frame-count byte (libopus rejects).
                let count = match toc & 0x03 {
                    0 => 1,
                    1 | 2 => 2,
                    _ => return Err("Code 3 packet too short"),
                };
                let per_frame = frame_samples_from_toc(toc, self.sampling_rate)
                    .ok_or("Invalid TOC for sampling rate")?;
                let total = per_frame * count;
                if frame_size < total {
                    return Err("frame_size too small for packet");
                }
                total
            }
        };
        let channels = self.channels;
        if output.len() < total * channels {
            return Err("Output buffer too small for packet");
        }
        output[total * channels..].fill(0.0);

        let mut pos = 0usize;
        while pos < total {
            let remaining = total - pos;
            let dst = &mut output[pos * channels..total * channels];
            let produced = match self.prev_mode {
                None => {
                    dst.fill(0.0);
                    remaining
                }
                Some(OpusMode::CeltOnly) => {
                    // Largest CELT frame that fits; `remaining` is a multiple
                    // of 2.5 ms, so one always does.
                    let n = [f20, f10, f5, f2_5]
                        .into_iter()
                        .find(|&c| c <= remaining)
                        .unwrap_or(f2_5);
                    self.conceal_celt(n, 0)?;
                    for (d, s) in dst.iter_mut().zip(&self.w_celt_out[..n * channels]) {
                        *d = s.clamp(-1.0, 1.0);
                    }
                    n
                }
                Some(prev @ (OpusMode::SilkOnly | OpusMode::Hybrid)) => {
                    // SILK conceals in 10/20 ms frames; a shorter request
                    // takes the head of a 10 ms frame, as libopus does.
                    let (ms, n) = if remaining >= f20 {
                        (20, f20)
                    } else {
                        (10, f10)
                    };
                    let hybrid = prev == OpusMode::Hybrid;
                    let rate = if hybrid {
                        16000
                    } else {
                        self.prev_internal_rate
                    };
                    let mut rc = RangeCoder::new_decoder(&[]);
                    self.decode_silk_frames(
                        &mut rc,
                        silk::decode_frame::FLAG_PACKET_LOST,
                        ms,
                        rate,
                        n,
                    )?;
                    if hybrid {
                        self.conceal_celt(n, 17)?;
                    } else {
                        self.w_celt_out[..n * channels].fill(0.0);
                    }
                    let take = n.min(remaining);
                    for (i, d) in dst[..take * channels].iter_mut().enumerate() {
                        *d = (self.w_silk_out[i] + self.w_celt_out[i]).clamp(-1.0, 1.0);
                    }
                    take
                }
            };
            pos += produced;
        }

        self.save_pcm_tail(output, total);
        self.range_final = 0;
        Ok(total)
    }

    /// 5 ms of the old CELT state concealed forward, the bridge out of CELT (libopus
    /// opus_decode_frame's `pcm_transition`), into `prev_pcm_tail`.
    fn conceal_celt_bridge(&mut self) -> Result<(), &'static str> {
        let f5 = self.sampling_rate as usize / 200;
        if f5 == 0 {
            return Ok(());
        }
        self.conceal_celt(f5, 0)?;
        let n = (f5 * self.channels).min(self.prev_pcm_tail.len());
        for i in 0..n {
            self.prev_pcm_tail[i] = self.w_celt_out[i].clamp(-1.0, 1.0);
        }
        Ok(())
    }

    /// CELT concealment for `n` samples from band `start_band` (libopus
    /// `celt_decode_lost`): pitch repetition or noise, decaying. Output lands
    /// interleaved in `w_celt_out[..n * channels]`.
    fn conceal_celt(&mut self, n: usize, start_band: usize) -> Result<(), &'static str> {
        let len = n * self.channels;
        let decoded = {
            let (celt_dec, planar) = (
                state_mut(&mut self.celt_dec),
                state_mut(&mut self.w_celt_planar),
            );
            celt_dec.decode_lost(n, &mut planar[..len], start_band)
        };
        if decoded != n {
            return Err("CELT concealment failed");
        }
        for i in 0..n {
            for ch in 0..self.channels {
                self.w_celt_out[i * self.channels + ch] = self.w_celt_planar[ch * n + i];
            }
        }
        Ok(())
    }

    /// Keep the last F5 of output for the next mode-transition bridge.
    fn save_pcm_tail(&mut self, output: &[f32], decoded: usize) {
        let tail_len = (self.sampling_rate as usize / 200) * self.channels;
        let out_total = decoded * self.channels;
        if out_total >= tail_len && tail_len <= self.prev_pcm_tail.len() {
            self.prev_pcm_tail[..tail_len]
                .copy_from_slice(&output[out_total - tail_len..out_total]);
        }
    }
}

impl OpusDecoder {
    /// Decode one Opus frame of SILK audio — `duration_ms` at `internal_rate`
    /// — into `w_silk_out[..api_len * channels]`, interleaved at the API rate.
    ///
    /// A 40/60 ms payload carries 2/3 internal SILK frames in one range-coded
    /// stream; libopus loops silk_Decode until the full duration is produced,
    /// with new_packet set only on the first call (issue #27). With
    /// `FLAG_PACKET_LOST` the same loop drives SILK packet-loss concealment.
    fn decode_silk_frames(
        &mut self,
        rc: &mut RangeCoder,
        lost_flag: i32,
        duration_ms: i32,
        internal_rate: i32,
        api_len: usize,
    ) -> Result<(), &'static str> {
        let internal_frame_size = (duration_ms * internal_rate / 1000) as usize;
        let pcm_i16_len = internal_frame_size * self.channels;
        debug_assert!(pcm_i16_len <= self.w_pcm_i16.len());
        let out_len = api_len * self.channels;
        self.w_silk_out[..out_len].fill(0.0);

        let mut frame_pos = 0usize; // internal-rate samples decoded so far
        let mut new_packet = true;
        while frame_pos < internal_frame_size {
            // libopus takes the channel count from the packet only when there
            // is data; concealment keeps the last frame's.
            let prev_channels = self.silk_dec.n_channels_internal as usize;
            let stream_channels = if lost_flag == silk::decode_frame::FLAG_PACKET_LOST {
                prev_channels
            } else {
                self.stream_channels
            };
            // libopus keeps a resampler per SILK channel (dec_API.c). When a
            // stereo decoder's stream turns stereo, the right one starts from
            // the left's. The first mono frame after stereo, at an unchanged
            // rate, still resamples the right channel with its own; libopus
            // resets SILK after CELT, so that frame never follows CELT there.
            if self.channels == 2 && stream_channels == 2 && prev_channels == 1 {
                self.silk_resampler_2 = self.silk_resampler.clone();
            }
            let stereo_to_mono = stream_channels == 1
                && prev_channels == 2
                && internal_rate == self.silk_dec.sample_rate()
                && matches!(
                    self.prev_mode,
                    Some(OpusMode::SilkOnly) | Some(OpusMode::Hybrid)
                );
            let ret = {
                let (silk_dec, pcm_i16) = (
                    state_mut(&mut self.silk_dec),
                    state_mut(&mut self.w_pcm_i16),
                );
                silk_dec.decode_with_channels(
                    rc,
                    &mut pcm_i16[..pcm_i16_len],
                    lost_flag,
                    new_packet,
                    duration_ms,
                    internal_rate,
                    stream_channels as i32,
                )
            };
            new_packet = false;

            if ret < 0 {
                return Err("SILK decoding failed");
            }
            let decoded_samples = ret as usize;
            if decoded_samples == 0 {
                break;
            }

            // SILK decoder outputs planar for THIS frame:
            // ch0 at [0..fl], ch1 at [fl..2*fl].
            //
            // libopus always runs `silk_resampler`, even when the API and SILK
            // internal rates are equal: the Copy path still carries the
            // resampler's `inputDelay` (12 samples at 16k, 4 at 8k). Bypassing
            // it made equal-rate SILK output early versus libopus (and versus
            // opus-rs's own resampled paths).
            let resampled_len =
                decoded_samples * self.sampling_rate as usize / internal_rate as usize;
            let api_pos = frame_pos * self.sampling_rate as usize / internal_rate as usize;
            let copy_len = resampled_len.min(api_len - api_pos);
            debug_assert!(resampled_len * self.channels <= self.w_pcm_resampled.len());
            // Resample channel 0.
            {
                let (res, inp, out) = (
                    &mut self.silk_resampler,
                    state_ref(&self.w_pcm_i16),
                    state_mut(&mut self.w_pcm_resampled),
                );
                res.process(
                    &mut out[..resampled_len],
                    &inp[..decoded_samples],
                    decoded_samples as i32,
                );
            }
            // Channel 1 (stereo output only). A mono stream is copied from
            // channel 0, except on the frame that collapses stereo to mono.
            if self.channels == 2 {
                let (res, inp, out) = (
                    &mut self.silk_resampler_2,
                    state_ref(&self.w_pcm_i16),
                    state_mut(&mut self.w_pcm_resampled),
                );
                if stream_channels == 2 || stereo_to_mono {
                    let src = if stream_channels == 2 {
                        &inp[decoded_samples..2 * decoded_samples]
                    } else {
                        &inp[..decoded_samples]
                    };
                    res.process(
                        &mut out[resampled_len..2 * resampled_len],
                        src,
                        decoded_samples as i32,
                    );
                } else {
                    out.copy_within(..resampled_len, resampled_len);
                }
            }
            for i in 0..copy_len {
                for ch in 0..self.channels {
                    let v = self.w_pcm_resampled[ch * resampled_len + i] as f32 / 32768.0;
                    let idx = (api_pos + i) * self.channels + ch;
                    if idx < out_len {
                        self.w_silk_out[idx] = v;
                    }
                }
            }
            frame_pos += decoded_samples;
        }
        Ok(())
    }

    #[inline(always)]
    fn celt_end_band_from_toc(&self, toc: u8) -> usize {
        let mode = modes::default_mode();
        let top = mode.eff_ebands;
        if mode_from_toc(toc) == OpusMode::CeltOnly && toc >= 0x80 {
            const FROM_OPUS_TABLE: [u8; 16] = [
                0x80, 0x88, 0x90, 0x98, 0x40, 0x48, 0x50, 0x58, 0x20, 0x28, 0x30, 0x38, 0x00, 0x08,
                0x10, 0x18,
            ];
            let idx = ((toc >> 3) - 16) as usize;
            let data0 = FROM_OPUS_TABLE[idx] | (toc & 0x7);
            let trim = (data0 >> 5) as usize;
            return top.saturating_sub(2 * trim).max(1);
        }
        // A Hybrid frame's CELT layer ends where its bandwidth does (libopus opus_decode_frame:
        // super-wideband 19 bands, fullband 21); it decoded to 21 at every bandwidth.
        if mode_from_toc(toc) == OpusMode::Hybrid && (toc >> 3) < 14 {
            return top.saturating_sub(2).max(1);
        }
        top
    }
}

fn frame_rate_from_params(sampling_rate: i32, frame_size: usize) -> Option<i32> {
    let frame_size = frame_size as i32;
    if frame_size <= 0 {
        return None;
    }
    // libopus `frame_size_select`: a frame is 2.5, 5, 10, 20, 40 or 60 ms.
    // Validating by duration (not by `sampling_rate % frame_size == 0`) is what
    // lets 60 ms through: its rate is 16.67 frames/s, so the old divisibility
    // test rejected it before encoding could start (issue #35). Longer frames
    // (>60 ms) are handled by the caller's multiframe path.
    let fs = sampling_rate;
    let valid = frame_size == fs / 400
        || frame_size == fs / 200
        || frame_size == fs / 100
        || frame_size == fs / 50
        || frame_size == fs / 25
        || 50 * frame_size == 3 * fs;
    if !valid {
        return None;
    }
    Some(fs / frame_size)
}

fn gen_toc(mode: OpusMode, frame_rate: i32, bandwidth: Bandwidth, channels: usize) -> u8 {
    let mut rate = frame_rate;
    let mut period = 0;
    while rate < 400 {
        rate <<= 1;
        period += 1;
    }

    let mut toc = match mode {
        OpusMode::SilkOnly => {
            let bw = (bandwidth as i32 - Bandwidth::Narrowband as i32) << 5;
            let per = (period - 2) << 3;
            (bw | per) as u8
        }
        OpusMode::CeltOnly => {
            let mut tmp = bandwidth as i32 - Bandwidth::Mediumband as i32;
            if tmp < 0 {
                tmp = 0;
            }
            let per = period << 3;
            (0x80 | (tmp << 5) | per) as u8
        }
        OpusMode::Hybrid => {
            let base_config = if bandwidth == Bandwidth::Superwideband {
                12
            } else {
                14
            };
            let period_offset = if frame_rate >= 100 { 0 } else { 1 };
            ((base_config + period_offset) << 3) as u8
        }
    };

    if channels == 2 {
        toc |= 0x04;
    }
    toc
}

fn mode_from_toc(toc: u8) -> OpusMode {
    if toc & 0x80 != 0 {
        OpusMode::CeltOnly
    } else if toc & 0x60 == 0x60 {
        OpusMode::Hybrid
    } else {
        OpusMode::SilkOnly
    }
}

fn bandwidth_from_toc(toc: u8) -> Bandwidth {
    let mode = mode_from_toc(toc);
    match mode {
        OpusMode::SilkOnly => {
            let bw_bits = (toc >> 5) & 0x03;
            match bw_bits {
                0 => Bandwidth::Narrowband,
                1 => Bandwidth::Mediumband,
                2 => Bandwidth::Wideband,
                _ => Bandwidth::Wideband,
            }
        }
        OpusMode::Hybrid => {
            let bw_bit = (toc >> 4) & 0x01;
            if bw_bit == 0 {
                Bandwidth::Superwideband
            } else {
                Bandwidth::Fullband
            }
        }
        OpusMode::CeltOnly => {
            let bw_bits = (toc >> 5) & 0x03;
            match bw_bits {
                0 => Bandwidth::Mediumband,
                1 => Bandwidth::Wideband,
                2 => Bandwidth::Superwideband,
                3 => Bandwidth::Fullband,
                _ => Bandwidth::Fullband,
            }
        }
    }
}

fn frame_duration_ms_from_toc(toc: u8) -> i32 {
    let mode = mode_from_toc(toc);
    match mode {
        OpusMode::SilkOnly => {
            let config = (toc >> 3) & 0x03;
            match config {
                0 => 10,
                1 => 20,
                2 => 40,
                3 => 60,
                _ => 20,
            }
        }
        OpusMode::Hybrid => {
            let config = (toc >> 3) & 0x01;
            if config == 0 { 10 } else { 20 }
        }
        OpusMode::CeltOnly => {
            let config = (toc >> 3) & 0x03;
            match config {
                0 => 2,
                1 => 5,
                2 => 10,
                3 => 20,
                _ => 20,
            }
        }
    }
}

/// Compute the per-frame sample count implied by the TOC byte at a given
/// sampling rate. For CELT this uses the frame-rate derivation (which handles
/// the 2.5 ms case correctly, unlike integer millisecond arithmetic).
fn frame_samples_from_toc(toc: u8, sampling_rate: i32) -> Option<usize> {
    let mode = mode_from_toc(toc);
    match mode {
        OpusMode::CeltOnly => {
            let period = ((toc >> 3) & 0x03) as i32;
            let frame_rate = 400 >> period;
            if frame_rate == 0 || sampling_rate % frame_rate != 0 {
                return None;
            }
            Some((sampling_rate / frame_rate) as usize)
        }
        OpusMode::SilkOnly | OpusMode::Hybrid => {
            let duration_ms = frame_duration_ms_from_toc(toc);
            Some((sampling_rate as i64 * duration_ms as i64 / 1000) as usize)
        }
    }
}

fn channels_from_toc(toc: u8) -> usize {
    if toc & 0x04 != 0 { 2 } else { 1 }
}

/// The frames of a packet of at least 2 bytes, split as its ToC's code says
/// (RFC 6716 §3.2; libopus `opus_packet_parse_impl`).
fn frame_payloads(input: &[u8]) -> Result<FixedVec<&[u8], OPUS_MAX_PACKET_FRAMES>, &'static str> {
    Ok(match input[0] & 0x03 {
        0 => FixedVec::from_slice(&[&input[1..]]),
        1 => {
            let data_len = input.len() - 1;
            // RFC 6716 §3.2.1: code 1 carries two equal-size (CBR) frames,
            // so the payload length must be even. libopus rejects odd lengths.
            if !data_len.is_multiple_of(2) {
                return Err("Code 1: payload length must be even");
            }
            let half = data_len / 2;
            if half == 0 {
                return Err("Code 1: empty frame");
            }
            FixedVec::from_slice(&[&input[1..1 + half], &input[1 + half..]])
        }
        2 => {
            let data = &input[1..];
            if data.is_empty() {
                return Err("Code 2 packet has no data");
            }
            let (first_len, header_size) = parse_frame_size(data)?;
            if header_size + first_len > data.len() {
                return Err("Code 2: first frame size exceeds packet");
            }
            FixedVec::from_slice(&[
                &data[header_size..header_size + first_len],
                &data[header_size + first_len..],
            ])
        }
        3 => {
            if input.len() < 2 {
                return Err("Code 3 packet too short");
            }
            let count_byte = input[1];
            let n_frames = (count_byte & 0x3F) as usize;
            if !(1..=48).contains(&n_frames) {
                return Err("Code 3: invalid frame count");
            }
            let frame_count = n_frames;
            // Bit 6 = padding flag, bit 7 = VBR flag (RFC 6716 §3.2.1).
            let padding_flag = (count_byte & 0x40) != 0;
            let vbr = (count_byte & 0x80) != 0;

            // Parse the optional padding length bytes that follow the count
            // byte. The padding *content* (pad_len bytes) lives at the end of
            // the packet and is not part of any frame.
            let mut ptr = 2usize;
            let mut pad_len = 0usize;
            if padding_flag {
                loop {
                    if ptr >= input.len() {
                        return Err("Code 3: padding overflow");
                    }
                    let p = input[ptr] as usize;
                    ptr += 1;
                    if p == 255 {
                        pad_len += 254;
                    } else {
                        pad_len += p;
                        break;
                    }
                }
            }
            if ptr + pad_len > input.len() {
                return Err("Code 3: padding exceeds packet");
            }
            let payload_end = input.len() - pad_len;
            let payload = &input[ptr..payload_end];

            let mut payloads: FixedVec<&[u8], OPUS_MAX_PACKET_FRAMES> = FixedVec::new();
            if frame_count == 1 {
                // Single frame: the entire payload region is the frame, both
                // for VBR and CBR (no length prefix is present).
                payloads.push(payload);
            } else if vbr {
                // VBR (V=1): the per-frame length fields for all frames
                // except the last come first, then all frame payloads
                // (RFC 6716 §3.2.1; libopus `opus_packet_parse_impl`).
                // Interleaving length and data — as this used to — parsed
                // any libopus code-3 VBR packet wrong.
                let mut cursor = 0usize;
                let mut frame_bytes = 0usize;
                let mut sizes = [0usize; OPUS_MAX_PACKET_FRAMES];
                for i in 0..frame_count - 1 {
                    if cursor >= payload.len() {
                        return Err("Code 3: unexpected end in VBR header");
                    }
                    let (frame_len, header_bytes) = parse_frame_size(&payload[cursor..])?;
                    cursor += header_bytes;
                    sizes[i] = frame_len;
                    frame_bytes += frame_len;
                }
                if frame_bytes > payload.len() - cursor {
                    return Err("Code 3: frame length exceeds packet");
                }
                sizes[frame_count - 1] = payload.len() - cursor - frame_bytes;
                let mut data_pos = cursor;
                for i in 0..frame_count {
                    payloads.push(&payload[data_pos..data_pos + sizes[i]]);
                    data_pos += sizes[i];
                }
            } else {
                // CBR (V=0): remaining bytes are split equally into M frames
                // (RFC 6716 §3.2.1: "the remaining bytes are split into M
                // equal chunks").
                if !payload.len().is_multiple_of(frame_count) {
                    return Err("Code 3 CBR: payload not divisible by frame count");
                }
                let frame_len = payload.len() / frame_count;
                for i in 0..frame_count {
                    payloads.push(&payload[i * frame_len..(i + 1) * frame_len]);
                }
            }
            payloads
        }
        _ => unreachable!(),
    })
}

/// A 5 ms redundant CELT frame at 48 kHz in stereo, the most it holds.
const REDUNDANT_MAX: usize = 480;

impl OpusDecoder {
    /// A mode switch's 5 ms redundant CELT frame (libopus opus_decode_frame:
    /// all bands from 0, its own range coder over its bytes), interleaved into
    /// `out`, returning its final range.
    fn decode_redundant(
        &mut self,
        data: &[u8],
        end_band: usize,
        f5: usize,
        out: &mut [f32],
    ) -> u32 {
        let n = f5 * self.channels;
        if data.is_empty() || n > REDUNDANT_MAX {
            return 0;
        }
        let mut planar = [0.0f32; REDUNDANT_MAX];
        let mut rrc = RangeCoder::new_decoder(data);
        self.celt_dec.decode_from_range_coder_with_band_range(
            &mut rrc,
            (data.len() * 8) as i32,
            f5,
            &mut planar[..n],
            0,
            end_band,
        );
        if self.channels == 1 {
            out[..n].copy_from_slice(&planar[..n]);
        } else {
            for i in 0..f5 {
                for c in 0..self.channels {
                    out[i * self.channels + c] = planar[c * f5 + i];
                }
            }
        }
        rrc.rng
    }
}

/// Crossfade two signals using a squared-sine window (libopus smooth_fade).
/// `window` is the 120-sample CELT window at 48 kHz; `inc` = 48000/Fs strides it.
fn smooth_fade(
    in1: &[f32],
    in2: &[f32],
    out: &mut [f32],
    overlap: usize,
    channels: usize,
    window: &[f32],
    inc: usize,
) {
    for c in 0..channels {
        for i in 0..overlap {
            let wi = i * inc;
            if wi >= window.len() {
                break;
            }
            let w = window[wi] * window[wi];
            out[i * channels + c] = w * in2[i * channels + c] + (1.0 - w) * in1[i * channels + c];
        }
    }
}

/// Parse an Opus frame length per RFC 6716 §3.2.1, identical to libopus
/// `parse_size()`:
///   - `0`: no frame (DTX / lost packet)
///   - `1..=251`: length of the frame in bytes (one byte consumed)
///   - `252..=255`: a second byte is read; length = `second*4 + first`
///
/// Returns `(length, bytes_consumed)`.
pub(crate) fn parse_frame_size(data: &[u8]) -> Result<(usize, usize), &'static str> {
    let first = *data.first().ok_or("truncated frame length")? as usize;
    if first < 252 {
        Ok((first, 1))
    } else {
        let second = *data.get(1).ok_or("truncated frame length")? as usize;
        Ok((second * 4 + first, 2))
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn frame_size_from_toc(toc: u8, sampling_rate: i32) -> Option<usize> {
        let mode = mode_from_toc(toc);
        match mode {
            OpusMode::CeltOnly => {
                let period = ((toc >> 3) & 0x03) as i32;
                let frame_rate = 400 >> period;
                if frame_rate == 0 || sampling_rate % frame_rate != 0 {
                    return None;
                }
                Some((sampling_rate / frame_rate) as usize)
            }
            OpusMode::SilkOnly => {
                let duration_ms = frame_duration_ms_from_toc(toc);
                Some((sampling_rate as i64 * duration_ms as i64 / 1000) as usize)
            }
            OpusMode::Hybrid => {
                let duration_ms = frame_duration_ms_from_toc(toc);
                Some((sampling_rate as i64 * duration_ms as i64 / 1000) as usize)
            }
        }
    }

    #[test]
    fn gen_toc_matches_celt_reference_values() {
        let sampling_rate = 48_000;
        let cases = [
            (120usize, 0xE0u8),
            (240usize, 0xE8u8),
            (480usize, 0xF0u8),
            (960usize, 0xF8u8),
        ];

        for (frame_size, expected_toc) in cases {
            let frame_rate = frame_rate_from_params(sampling_rate, frame_size).unwrap();
            let toc = gen_toc(OpusMode::CeltOnly, frame_rate, Bandwidth::Fullband, 1);
            assert_eq!(
                toc, expected_toc,
                "frame_size {} expected TOC {:02X} got {:02X}",
                frame_size, expected_toc, toc
            );
            let decoded_size = frame_size_from_toc(toc, sampling_rate).unwrap();
            assert_eq!(decoded_size, frame_size);
        }

        let stereo_toc = gen_toc(
            OpusMode::CeltOnly,
            frame_rate_from_params(sampling_rate, 960).unwrap(),
            Bandwidth::Fullband,
            2,
        );
        assert_eq!(channels_from_toc(stereo_toc), 2);
    }

    #[test]
    fn test_celt_decoder_large_frame_sizes() {
        let sampling_rate = 48000;
        let channels = 1;

        let mut decoder = OpusDecoder::new(sampling_rate, channels).unwrap();

        let frame_sizes = [120, 240, 480, 960];

        for frame_size in frame_sizes {
            let toc = gen_toc(
                OpusMode::CeltOnly,
                frame_rate_from_params(sampling_rate, frame_size).unwrap(),
                Bandwidth::Fullband,
                channels,
            );
            let packet = [toc, 0, 0, 0, 0];

            let mut output = vec![0.0f32; frame_size * channels];

            let _ = decoder.decode(&packet, frame_size, &mut output);
        }

        let channels = 2;
        let mut decoder = OpusDecoder::new(sampling_rate, channels).unwrap();

        for frame_size in frame_sizes {
            let toc = gen_toc(
                OpusMode::CeltOnly,
                frame_rate_from_params(sampling_rate, frame_size).unwrap(),
                Bandwidth::Fullband,
                channels,
            );
            let packet = [toc, 0, 0, 0, 0];

            let mut output = vec![0.0f32; frame_size * channels];
            let _ = decoder.decode(&packet, frame_size, &mut output);
        }
    }

    #[test]
    fn test_celt_decoder_edge_case_frame_sizes() {
        let sampling_rate = 48000;
        let channels = 1;
        let mut decoder = OpusDecoder::new(sampling_rate, channels).unwrap();

        let edge_sizes = [2048, 2167, 2168, 2169, 2880, 3072];

        for frame_size in edge_sizes {
            let mut output = vec![0.0f32; frame_size * channels];

            let _ = decoder.decode(&[0x80, 0, 0, 0], frame_size, &mut output);
        }
    }

    // Regression test for: "index out of bounds: the len is 48 but the index is 119"
    // Root cause: frame_size=48 at 48kHz gives frame_rate=1000, which is not a valid
    // Hybrid-mode frame rate but was not validated.  CELT's lm-search then silently
    // fell back to lm=0, computed n2=120, and wrote output[119] into a 48-element
    // slice.  Triggered via G.729-decoded PCM (8kHz) passed to a 48kHz Opus encoder
    // without proper resampling, so the encoder received 48 samples instead of 480.
    #[test]
    fn test_invalid_small_frame_size_returns_error_not_panic() {
        let mut enc = OpusEncoder::new(48000, 2, Application::Voip).unwrap();
        enc.bitrate_bps = 64000;
        enc.complexity = 5;
        enc.use_cbr = true;

        // 48 samples at 48kHz = 1ms → frame_rate=1000, invalid for Hybrid mode.
        let input = vec![0.0f32; 48 * 2]; // stereo interleaved
        let mut output = vec![0u8; 256];

        let result = enc.encode(&input, 48, &mut output);
        assert!(
            result.is_err(),
            "encode with invalid frame_size=48 should return Err, not panic"
        );
    }

    // Also verify that the Audio application path (always Hybrid at 48 kHz) rejects
    // the same bad frame size.
    #[test]
    fn test_invalid_small_frame_size_audio_application_returns_error() {
        let mut enc = OpusEncoder::new(48000, 1, Application::Audio).unwrap();
        let input = vec![0.0f32; 48];
        let mut output = vec![0u8; 256];

        let result = enc.encode(&input, 48, &mut output);
        assert!(
            result.is_err(),
            "Audio/48kHz encoder with frame_size=48 should return Err"
        );
    }
}
