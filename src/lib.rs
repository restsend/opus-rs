#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]

/// Intermediate-value dumps for the differential tools (`OFZ_DUMP`).
#[cfg(feature = "fuzzing")]
macro_rules! dumpf {
    ($t:expr, $x:expr) => {
        crate::fuzzing::dumpf($t, $x)
    };
}
#[cfg(not(feature = "fuzzing"))]
macro_rules! dumpf {
    ($t:expr, $x:expr) => {};
}
#[cfg(feature = "fuzzing")]
macro_rules! dumpi {
    ($t:expr, $x:expr) => {
        crate::fuzzing::dumpi($t, $x)
    };
}
#[cfg(not(feature = "fuzzing"))]
macro_rules! dumpi {
    ($t:expr, $x:expr) => {};
}

mod compat;
mod fixedvec;
mod mlp;
mod mlp_data;
#[rustfmt::skip]
mod static_modes;

pub mod analysis;
pub mod bands;
pub mod celt;
pub mod celt_lpc;
/// The neural network features of libopus 1.6.1 (cargo features `dnn`,
/// `deep-plc`, `dred`, `osce`).
#[cfg(feature = "dnn")]
pub mod dnn;
/// Deep redundancy in the decoder (cargo feature `dred`).
#[cfg(feature = "dred")]
pub mod dred;
/// Opus packet extensions, which carry DRED (cargo feature `dred`).
#[cfg(feature = "dred")]
pub mod extensions;
pub mod hp_cutoff;
pub mod kiss_fft;
mod mathops;
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

/// Internals for the differential fuzz harnesses in `fuzz/`. Not a stable API.
#[cfg(feature = "fuzzing")]
#[doc(hidden)]
pub mod fuzzing {
    pub use crate::celt::fuzz_comb_filter as comb_filter;
    pub use crate::compat::SIMD_ENABLED;
    use std::cell::RefCell;
    use std::vec::Vec;

    std::thread_local!(static SILK_INPUT: RefCell<Vec<i16>> = const { RefCell::new(Vec::new()) });

    /// Records the 16-bit API-rate samples handed to SILK this frame.
    pub(crate) fn record_silk_input(x: &[i16]) {
        SILK_INPUT.with(|b| b.replace(x.to_vec()));
    }

    /// The samples SILK received in the last frame that ran SILK.
    pub fn take_silk_input() -> Vec<i16> {
        SILK_INPUT.with(|b| b.take())
    }

    fn dump_on() -> bool {
        std::thread_local!(static ON: bool = std::env::var_os("OFZ_DUMP").is_some());
        ON.with(|v| *v)
    }

    /// With `OFZ_DUMP` set, prints an intermediate array to stderr as
    /// `R <tag> f <n> <hex bits>...`; the reference prints its dumps with `D`.
    pub(crate) fn dumpf(tag: &str, x: &[f32]) {
        if dump_on() {
            let v: Vec<String> = x.iter().map(|v| format!("{:08x}", v.to_bits())).collect();
            std::eprintln!("R {tag} f {} {}", x.len(), v.join(" "));
        }
    }

    pub(crate) fn dumpi(tag: &str, x: &[i32]) {
        if dump_on() {
            let v: Vec<String> = x.iter().map(|v| v.to_string()).collect();
            std::eprintln!("R {tag} i {} {}", x.len(), v.join(" "));
        }
    }
}
pub mod silk;

pub use silk::{SilkResampler, SilkResamplerDown1_3, SilkResamplerDown1_6};

#[cfg(not(feature = "std"))]
use crate::compat::Math;
use crate::fixedvec::FixedVec;
pub use celt::{CeltDecoder, CeltEncoder};
use celt::AnalysisInfo;
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
/// High-pass filter state memory (`channels * 2`).
const OPUS_HP_MEM: usize = OPUS_MAX_CHANNELS * 2;
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

/// The kind of signal the encoder codes, as libopus `OPUS_SET_SIGNAL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// Estimated by the encoder: by the signal analysis where it runs, else
    /// from the application.
    Auto = -1000,
    /// Speech (`OPUS_SIGNAL_VOICE`).
    Voice = 3001,
    /// Music (`OPUS_SIGNAL_MUSIC`).
    Music = 3002,
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
    /// Constrained VBR (see [`OpusEncoder::set_vbr_constraint`]).
    vbr_constraint: bool,
    pub use_cbr: bool,

    pub use_inband_fec: bool,

    pub packet_loss_perc: i32,
    /// Discontinuous transmission (see [`OpusEncoder::set_dtx`]).
    use_dtx: bool,
    /// Whether SILK ran its own DTX on the last frame (libopus
    /// `silk_mode.useDTX`).
    silk_use_dtx: bool,
    /// Milliseconds (Q1) of digital silence in a row, for DTX (libopus
    /// `nb_no_activity_ms_Q1`).
    nb_no_activity_ms_q1: i32,
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
    /// The bandwidth and in-band FEC decision for the next sub-frame of a
    /// multiframe packet: its frames share one ToC, and libopus decides both
    /// once for the whole packet (opus_encode_native, before
    /// encode_multiframe_packet).
    frame_fec: Option<(Bandwidth, i32)>,
    /// The frame is not the last of a multiframe packet (libopus
    /// `nonfinal_frame`): SILK may then not switch its rate.
    nonfinal_frame: bool,

    variable_hp_smth2_q15: i32,
    /// libopus `width_mem`, the state of `compute_stereo_width`.
    width_mem: StereoWidthState,
    /// libopus `signal_type` (`OPUS_SET_SIGNAL`).
    signal: Signal,
    /// libopus's signal analysis of the input (`TonalityAnalysisState`).
    #[cfg(not(feature = "heap"))]
    analysis: analysis::TonalityAnalysisState,
    #[cfg(feature = "heap")]
    analysis: Box<analysis::TonalityAnalysisState>,
    /// The analysis read position before the last packet's analysis
    /// (libopus `analysis_read_pos_bak`), when it ran.
    analysis_read_bak: Option<(usize, i32)>,
    /// The analysis and digital-silence flag of the next frame of a
    /// multiframe packet, which reads its share of the packet's analysis.
    frame_analysis: Option<(AnalysisInfo, bool)>,
    /// libopus `peak_signal_energy`, for the activity of quiet frames.
    peak_signal_energy: f32,
    /// libopus `voice_ratio`: the percentage of speech the analysis found,
    /// -1 when it gave none.
    voice_ratio: i32,
    /// libopus `detected_bandwidth`: the input's bandwidth by the analysis.
    detected_bandwidth: Option<Bandwidth>,
    /// libopus `user_bandwidth` (`OPUS_SET_BANDWIDTH`), `Auto` by default.
    user_bandwidth: Bandwidth,
    /// libopus `max_bandwidth` (`OPUS_SET_MAX_BANDWIDTH`).
    max_bandwidth: Bandwidth,
    /// libopus `auto_bandwidth`: the last bandwidth the rate chose.
    auto_bandwidth: Bandwidth,
    /// libopus `first`: no frame coded yet.
    first: bool,
    /// libopus `silk_mode.allowBandwidthSwitch` and
    /// `inWBmodeWithoutVariableLP`, as the last SILK call left them.
    silk_allow_bandwidth_switch: bool,
    silk_in_wb_without_variable_lp: bool,

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
    /// State of the integer high-pass of SILK's input for VoIP at 8 to
    /// 24 kHz.
    hp_mem: FixedVec<i32, OPUS_HP_MEM>,
    /// Float filter state for `dc_reject_float` / `hp_cutoff_float`
    /// (C `st->hp_mem[4]`, opus_val32 in the float build).
    hp_mem_float: [f32; 4],
    prev_hb_gain: f32,
    /// libopus `hybrid_stereo_width_Q14` and `silk_mode.stereoWidth_Q14`:
    /// the stereo width the CELT input faded to last and the one it fades to.
    hybrid_stereo_width_q14: i32,
    stereo_width_q14: i32,
    /// libopus `stream_channels`, `prev_channels` and `silk_mode.toMono`:
    /// the channels the rate allows coding, those of the last packet, and
    /// the delay of a SILK stereo-to-mono switch.
    stream_channels: usize,
    prev_channels: usize,
    silk_to_mono: bool,
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

/// libopus `is_digital_silence`: no sample above the input's resolution
/// (24 bits for float input, 16 for 16-bit input).
fn is_digital_silence(input: PcmInput<'_>, len: usize) -> bool {
    match input {
        PcmInput::F32(s) => s[..len].iter().all(|x| x.abs() <= 1.0 / (1 << 24) as f32),
        PcmInput::I16(s) => s[..len].iter().all(|&x| x == 0),
    }
}

/// libopus `decide_dtx_mode`: whether a frame is sent as DTX, counting the
/// inactive time in milliseconds (Q1); an active frame resets the count.
fn decide_dtx_mode(activity: bool, nb_no_activity_ms_q1: &mut i32, frame_size_ms_q1: i32) -> bool {
    use silk::define::{MAX_CONSECUTIVE_DTX, NB_SPEECH_FRAMES_BEFORE_DTX};
    if activity {
        *nb_no_activity_ms_q1 = 0;
        return false;
    }
    *nb_no_activity_ms_q1 += frame_size_ms_q1;
    if *nb_no_activity_ms_q1 > NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2 {
        if *nb_no_activity_ms_q1 <= (NB_SPEECH_FRAMES_BEFORE_DTX + MAX_CONSECUTIVE_DTX) * 20 * 2 {
            return true;
        }
        *nb_no_activity_ms_q1 = NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2;
    }
    false
}

/// libopus `DTX_ACTIVITY_THRESHOLD`: the activity probability of the signal
/// analysis below which a frame counts as inactive.
const DTX_ACTIVITY_THRESHOLD: f32 = 0.1;
/// libopus `PSEUDO_SNR_THRESHOLD` (25 dB): an inactive frame this close to
/// the loudest active one still counts as active.
const PSEUDO_SNR_THRESHOLD: f32 = 316.23;
/// libopus `VAD_NO_DECISION`: the analysis is off and the mode is not
/// CELT-only, so SILK's own voice activity decides (opus_encoder.c:2235).
const VAD_NO_DECISION: i32 = -1;

/// libopus `compute_frame_energy` (float build): the mean square of the
/// input, at the float API's scale.
fn compute_frame_energy(input: PcmInput<'_>, len: usize) -> f32 {
    let sum = match input {
        PcmInput::F32(s) => s[..len].iter().fold(0.0f32, |acc, &x| acc + x * x),
        PcmInput::I16(s) => s[..len].iter().fold(0.0f32, |acc, &x| {
            let x = f32::from(x) / 32768.0;
            acc + x * x
        }),
    };
    sum / len as f32
}

/// libopus `{mono,stereo}_{voice,music}_bandwidth_thresholds` (the mono and
/// stereo tables are the same): the rate and hysteresis for NB<->MB,
/// MB<->WB, WB<->SWB and SWB<->FB.
const VOICE_BANDWIDTH_THRESHOLDS: [i32; 8] = [9000, 700, 9000, 700, 13500, 1000, 14000, 2000];
const MUSIC_BANDWIDTH_THRESHOLDS: [i32; 8] = [9000, 700, 9000, 700, 11000, 1000, 12000, 2000];

/// libopus `StereoWidthState`.
#[derive(Clone, Copy, Default)]
struct StereoWidthState {
    xx: f32,
    xy: f32,
    yy: f32,
    smoothed_width: f32,
    max_follower: f32,
}

/// libopus `compute_stereo_width` (float build): how far the two channels
/// of the input are from mono, 0 to 1, smoothed over about a second.
fn compute_stereo_width(
    input: PcmInput<'_>,
    frame_size: usize,
    fs: i32,
    mem: &mut StereoWidthState,
) -> f32 {
    let sample = |i: usize| match input {
        PcmInput::F32(s) => s[i],
        PcmInput::I16(s) => f32::from(s[i]) / 32768.0,
    };
    let frame_rate = fs / frame_size as i32;
    let short_alpha = 25.0 / 50.max(frame_rate) as f32;
    let (mut xx, mut xy, mut yy) = (0.0f32, 0.0f32, 0.0f32);
    // Four samples at a time: the frame size is a multiple of 4 except for
    // 2.5 ms frames at 12 kHz, which lose their last two samples.
    let mut i = 0;
    while i + 3 < frame_size {
        let (mut pxx, mut pxy, mut pyy) = (0.0f32, 0.0f32, 0.0f32);
        for k in 0..4 {
            let x = sample(2 * (i + k));
            let y = sample(2 * (i + k) + 1);
            pxx += x * x;
            pxy += x * y;
            pyy += y * y;
        }
        xx += pxx;
        xy += pxy;
        yy += pyy;
        i += 4;
    }
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(xx < 1e9) || xx.is_nan() || !(yy < 1e9) || yy.is_nan() {
        xx = 0.0;
        xy = 0.0;
        yy = 0.0;
    }
    mem.xx += short_alpha * (xx - mem.xx);
    // Written to avoid overflows on an abrupt change of sign.
    mem.xy = (1.0 - short_alpha) * mem.xy + short_alpha * xy;
    mem.yy += short_alpha * (yy - mem.yy);
    mem.xx = mem.xx.max(0.0);
    mem.xy = mem.xy.max(0.0);
    mem.yy = mem.yy.max(0.0);
    if mem.xx.max(mem.yy) > 8e-4 {
        let sqrt_xx = mem.xx.sqrt();
        let sqrt_yy = mem.yy.sqrt();
        let qrrt_xx = sqrt_xx.sqrt();
        let qrrt_yy = sqrt_yy.sqrt();
        // Inter-channel correlation.
        mem.xy = mem.xy.min(sqrt_xx * sqrt_yy);
        let corr = mem.xy / (1e-15 + sqrt_xx * sqrt_yy);
        // Approximate loudness difference.
        let ldiff = (qrrt_xx - qrrt_yy).abs() / (1e-15 + qrrt_xx + qrrt_yy);
        let width = (1.0 - corr * corr).sqrt().min(1.0) * ldiff;
        // Smoothing over one second, and a peak follower.
        mem.smoothed_width += (width - mem.smoothed_width) / frame_rate as f32;
        mem.max_follower = (mem.max_follower - 0.02 / frame_rate as f32).max(mem.smoothed_width);
    }
    (20.0 * mem.max_follower).min(1.0)
}

/// The bitrate (equivalent 20 ms rate) from which libopus codes CELT-only
/// rather than SILK or Hybrid (opus_encode_native): between music's and
/// speech's threshold by the voice estimate (Q7), speech's between mono's
/// and stereo's by the stereo width, raised for VoIP, with hysteresis.
fn compute_mode_threshold(
    application: Application,
    stereo_width: f32,
    prev_mode: Option<OpusMode>,
    voice_est: i32,
) -> i32 {
    // libopus `mode_thresholds`: speech 64000 (mono) and 44000 (stereo),
    // music 10000 (whose stereo entry libopus uses for both).
    let mode_voice = ((1.0 - stereo_width) * 64000.0 + stereo_width * 44000.0) as i32;
    let mode_music = ((1.0 - stereo_width) * 10000.0 + stereo_width * 10000.0) as i32;
    let mut threshold = mode_music + ((voice_est * voice_est * (mode_voice - mode_music)) >> 14);
    // A bias towards SILK for VoIP, for its useful features.
    if application == Application::Voip {
        threshold += 8000;
    }
    match prev_mode {
        Some(OpusMode::CeltOnly) => threshold -= 4000,
        Some(_) => threshold += 4000,
        None => {}
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

/// libopus's Hybrid high-band gain, Q15ONE - SHR32(celt_exp2(-rate/1024), 1)
/// (opus_encoder.c:2061), in the float build, where SHR32 shifts nothing.
fn hybrid_hb_gain(celt_rate: i32) -> f32 {
    1.0 - mathops::celt_exp2(-celt_rate as f32 * (1.0 / 1024.0))
}

/// libopus `pcm_buf` and `delay_buffer` (opus_encoder.c:1966-2028,
/// 2297-2312): `pcm` receives the delayed history, then the high-passed
/// frame, and the delay line moves on by the frame.
#[allow(clippy::too_many_arguments)]
/// libopus FLOAT2INT16 (float_cast.h): scale, saturate, round to nearest
/// with ties to even (`lrintf`). Adding and removing 1.5 * 2^23 rounds an
/// f32 below 2^22 in magnitude to an integer in the current (nearest-even)
/// rounding mode.
fn float2int16(x: f32) -> i16 {
    let x = (x * 32768.0).clamp(-32768.0, 32767.0);
    ((x + 12_582_912.0) - 12_582_912.0) as i16
}

/// The frame part of libopus's pcm_buf (opus_encoder.c:1982-2012): the
/// input high-passed (VoIP) or DC-rejected, zeroed with the filter state if
/// it is not finite.
fn filter_frame(
    out: &mut [f32],
    hp_mem: &mut [f32; 4],
    input: &[f32],
    voip: bool,
    cutoff_hz: i32,
    frame_size: usize,
    ch: usize,
    fs: i32,
) {
    if voip {
        hp_cutoff_float(input, cutoff_hz, out, hp_mem, frame_size, ch, fs);
    } else {
        dc_reject_float(input, 3, out, hp_mem, frame_size, ch, fs);
    }
    // NaN guard, in libopus's shape (`!(sum < 1e9f) || celt_isnan(sum)`).
    let sum: f32 = out.iter().map(|v| v * v).sum();
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum < 1e9) || sum.is_nan() {
        out.fill(0.0);
        *hp_mem = [0.0; 4];
    }
}

/// Puts the delayed history in front of the filtered frame in `pcm` and
/// moves the delay line on by the frame.
fn push_delay_history(delay_buffer: &mut [f32], pcm: &mut [f32], frame_size: usize, ch: usize) {
    let ebuf = delay_buffer.len() / ch;
    let delay = pcm.len() / ch - frame_size;
    let prefix = delay * ch;
    pcm[..prefix].copy_from_slice(&delay_buffer[(ebuf - delay) * ch..ebuf * ch]);
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

/// libopus `opus_packet_pad` of a code 0 packet `output[..len]` to `total`
/// bytes: a code 3 packet of the one frame and zero padding.
fn pad_packet(output: &mut [u8], len: usize, total: usize) -> usize {
    if total <= len {
        return len;
    }
    let frame = len - 1;
    let pad_amount = total - frame - 2;
    let nb_255s = pad_amount.saturating_sub(1) / 255;
    let header = if pad_amount == 0 { 2 } else { 3 + nb_255s };
    output.copy_within(1..len, header);
    output[0] |= 0x03;
    if pad_amount == 0 {
        output[1] = 0x01;
    } else {
        output[1] = 0x41;
        output[2..2 + nb_255s].fill(255);
        output[2 + nb_255s] = (pad_amount - 255 * nb_255s - 1) as u8;
    }
    output[header + frame..total].fill(0);
    total
}

/// libopus `stereo_fade` (opus_encoder.c:548-575): moves the two channels
/// towards their mid by 1 - width, cross-fading over the overlap.
fn stereo_fade(pcm: &mut [f32], g1: f32, g2: f32, fs: i32) {
    let window = modes::default_mode().window;
    let inc = (48000 / fs as usize).max(1);
    let overlap = window.len() / inc;
    let (g1, g2) = (1.0 - g1, 1.0 - g2);
    for (i, s) in pcm.chunks_exact_mut(2).enumerate() {
        let g = if i < overlap {
            let w = window[i * inc] * window[i * inc];
            w * g2 + (1.0 - w) * g1
        } else {
            g2
        };
        let diff = g * (0.5 * (s[0] - s[1]));
        s[0] -= diff;
        s[1] += diff;
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
        // The float build: 1 - 2^(-rate/1024).
        assert_eq!(hybrid_hb_gain(0), 0.0);
        assert_eq!(hybrid_hb_gain(1024), 0.5);
        assert_eq!(hybrid_hb_gain(2048), 0.75);
        assert!(hybrid_hb_gain(8000) > 0.995);
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
            vbr_constraint: true,
            use_cbr: false,
            use_inband_fec: false,
            packet_loss_perc: 0,
            use_dtx: false,
            silk_use_dtx: false,
            nb_no_activity_ms_q1: 0,
            silk_initialized: false,
            prev_enc_mode: None,
            forced_mode: None,
            frame_transition: None,
            frame_fec: None,
            nonfinal_frame: false,
            mode: opus_mode,
            variable_hp_smth2_q15,
            width_mem: StereoWidthState::default(),
            signal: Signal::Auto,
            #[cfg(not(feature = "heap"))]
            analysis: analysis::TonalityAnalysisState::new(sampling_rate),
            #[cfg(feature = "heap")]
            analysis: Box::new(analysis::TonalityAnalysisState::new(sampling_rate)),
            analysis_read_bak: None,
            frame_analysis: None,
            peak_signal_energy: 0.0,
            voice_ratio: -1,
            detected_bandwidth: None,
            user_bandwidth: Bandwidth::Auto,
            max_bandwidth: Bandwidth::Fullband,
            auto_bandwidth: Bandwidth::Fullband,
            first: true,
            silk_allow_bandwidth_switch: false,
            silk_in_wb_without_variable_lp: false,

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
            hp_mem: FixedVec::from_value(0, channels * 2),
            hp_mem_float: [0.0; 4],
            prev_hb_gain: 1.0,
            hybrid_stereo_width_q14: 1 << 14,
            stereo_width_q14: 0,
            stream_channels: channels,
            prev_channels: 0,
            silk_to_mono: false,
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

    /// Discontinuous transmission, as libopus `OPUS_SET_DTX` (off by
    /// default). After 200 ms without speech the encoder sends packets of
    /// the ToC byte alone, which decoders conceal with comfort noise, and a
    /// coded frame every 400 ms to refresh the background. As in libopus,
    /// the signal analysis decides where it runs (complexity 7 and above,
    /// 16 kHz and above), in every mode; elsewhere SILK's voice activity
    /// detection does, and VoIP streams with DTX code in SILK, whose
    /// detector needs it. Digital silence always counts as inactive where
    /// the analysis would run.
    pub fn set_dtx(&mut self, enabled: bool) {
        self.use_dtx = enabled;
    }

    /// Whether DTX is on (libopus `OPUS_GET_DTX`).
    pub fn dtx(&self) -> bool {
        self.use_dtx
    }

    /// Whether the encoder is in DTX, so that the last packet was or would
    /// have been a ToC-only packet (libopus `OPUS_GET_IN_DTX`).
    pub fn in_dtx(&self) -> bool {
        let silk_mode = matches!(
            self.prev_enc_mode,
            Some(OpusMode::SilkOnly | OpusMode::Hybrid)
        );
        if self.silk_use_dtx && silk_mode {
            let quiet = |enc: &SilkEncoderState| {
                enc.s_cmn.no_speech_counter >= silk::define::NB_SPEECH_FRAMES_BEFORE_DTX
            };
            quiet(state_ref(&self.silk_enc))
                && (self.channels == 1
                    || self.silk_enc.stereo.prev_decode_only_middle != 0
                    || quiet(state_ref(&self.silk_enc_side)))
        } else if self.use_dtx {
            self.nb_no_activity_ms_q1 >= silk::define::NB_SPEECH_FRAMES_BEFORE_DTX * 20 * 2
        } else {
            false
        }
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
        self.user_bandwidth = bw;
        self.silk_initialized = false;
        Ok(())
    }

    /// The kind of signal to code, as libopus `OPUS_SET_SIGNAL` (`Auto` by
    /// default). `Voice` and `Music` bias the choice between SILK or Hybrid
    /// and CELT-only like a certain estimate of speech; `Auto` estimates it
    /// from the signal analysis where it runs (complexity 7 and up, 16 kHz
    /// and up), else from the application.
    pub fn set_signal(&mut self, signal: Signal) {
        self.signal = signal;
    }

    /// The kind of signal set (libopus `OPUS_GET_SIGNAL`).
    pub fn signal(&self) -> Signal {
        self.signal
    }

    /// The coded bandwidth, as libopus `OPUS_SET_BANDWIDTH`: `Auto` (the
    /// default) chooses it from the bitrate, the maximum bandwidth and the
    /// bandwidth the signal analysis detects; any other value is used at
    /// any rate, within the sampling rate and the packet size.
    pub fn set_bandwidth(&mut self, bandwidth: Bandwidth) {
        self.user_bandwidth = bandwidth;
    }

    /// The bandwidth of the last frame, as libopus `OPUS_GET_BANDWIDTH`.
    pub fn bandwidth(&self) -> Bandwidth {
        self.bandwidth
    }

    /// The highest bandwidth `Auto` may choose, as libopus
    /// `OPUS_SET_MAX_BANDWIDTH` (fullband by default; `Auto` means
    /// fullband).
    pub fn set_max_bandwidth(&mut self, bandwidth: Bandwidth) {
        self.max_bandwidth = if bandwidth == Bandwidth::Auto {
            Bandwidth::Fullband
        } else {
            bandwidth
        };
    }

    /// The maximum bandwidth (libopus `OPUS_GET_MAX_BANDWIDTH`).
    pub fn max_bandwidth(&self) -> Bandwidth {
        self.max_bandwidth
    }

    /// Constrained VBR, as libopus `OPUS_SET_VBR_CONSTRAINT` (on by
    /// default). It applies to VBR CELT-only frames: constrained, their
    /// sizes stay within what a one-frame buffer at the target bitrate
    /// allows; unconstrained, they vary freely around the target. SILK and
    /// Hybrid frames are unconstrained either way, and CBR (`use_cbr`) is
    /// unaffected.
    pub fn set_vbr_constraint(&mut self, constrained: bool) {
        self.vbr_constraint = constrained;
    }

    /// Whether VBR is constrained (libopus `OPUS_GET_VBR_CONSTRAINT`).
    pub fn vbr_constraint(&self) -> bool {
        self.vbr_constraint
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

    /// libopus's signal analysis of a packet (run_analysis in
    /// opus_encode_native, float build): the analysis, and whether the
    /// packet is digital silence. It runs for complexity 7 and up at 16 kHz
    /// and up, where libopus also detects digital silence and tracks the
    /// energy of active frames, and is reset when it stops running. The
    /// analysis sees the input's actual depth: 16 for `encode_i16`'s PCM16
    /// and 24 for `encode`'s float, like libopus's two entry points
    /// (opus_encoder.c passes 16 and MAX_ENCODING_DEPTH).
    fn analyse(&mut self, input: PcmInput<'_>, frame_size: usize) -> (AnalysisInfo, bool) {
        let (info, is_silence) = self.run_analysis(input, frame_size);
        // The speech estimate of the mode decision: from the analysis's
        // music probability, the one for staying in the previous mode; the
        // last one is kept through digital silence.
        if !is_silence {
            self.voice_ratio = -1;
        }
        if info.valid && self.signal == Signal::Auto {
            let prob = match self.prev_enc_mode {
                None => info.music_prob,
                Some(OpusMode::CeltOnly) => info.music_prob_max,
                Some(_) => info.music_prob_min,
            };
            self.voice_ratio = (0.5 + f64::from(100.0 * (1.0 - prob))).floor() as i32;
        }
        // The bandwidth of the input, by the highest analysis band with
        // signal.
        self.detected_bandwidth = info.valid.then_some(match info.bandwidth {
            ..=12 => Bandwidth::Narrowband,
            13..=14 => Bandwidth::Mediumband,
            15..=16 => Bandwidth::Wideband,
            17..=18 => Bandwidth::Superwideband,
            _ => Bandwidth::Fullband,
        });
        (info, is_silence)
    }

    /// libopus `voice_est`, the probability of speech in Q7: from the signal
    /// type, else from the analysis (never above 90% for Audio), else from
    /// the application.
    fn voice_estimate(&self) -> i32 {
        match (self.signal, self.application) {
            (Signal::Voice, _) => 127,
            (Signal::Music, _) => 0,
            (Signal::Auto, Application::Audio) if self.voice_ratio >= 0 => {
                ((self.voice_ratio * 327) >> 8).min(115)
            }
            (Signal::Auto, _) if self.voice_ratio >= 0 => (self.voice_ratio * 327) >> 8,
            (Signal::Auto, Application::Voip) => 115,
            (Signal::Auto, _) => 48,
        }
    }

    /// libopus's bandwidth decision before FEC (opus_encode_native): from
    /// the rate where SILK can switch (or in CELT, or at the start), with
    /// thresholds between music's and speech's by the voice estimate and
    /// hysteresis; then the maximum, the user's bandwidth, wideband for
    /// SILK and Hybrid below 15 kb/s of packet, the sampling rate, and the
    /// bandwidth the analysis detected, unless the user set one.
    fn decide_bandwidth(
        &mut self,
        mode: OpusMode,
        equiv_rate: i32,
        voice_est: i32,
        max_rate: i32,
    ) -> Bandwidth {
        use Bandwidth::*;
        let rank = |bw: Bandwidth| bandwidth_fec_index(bw) as i32;
        let mut bandwidth = self.bandwidth;
        if mode == OpusMode::CeltOnly || self.first || self.silk_allow_bandwidth_switch {
            let mut thresholds = [0i32; 8];
            for (i, t) in thresholds.iter_mut().enumerate() {
                let (voice, music) = (VOICE_BANDWIDTH_THRESHOLDS[i], MUSIC_BANDWIDTH_THRESHOLDS[i]);
                *t = music + ((voice_est * voice_est * (voice - music)) >> 14);
            }
            // From fullband (4) down to narrowband (0).
            let mut bw = 4;
            loop {
                let mut threshold = thresholds[2 * (bw - 1) as usize];
                let hysteresis = thresholds[2 * (bw - 1) as usize + 1];
                if !self.first {
                    if rank(self.auto_bandwidth) >= bw {
                        threshold -= hysteresis;
                    } else {
                        threshold += hysteresis;
                    }
                }
                if equiv_rate >= threshold {
                    break;
                }
                bw -= 1;
                if bw == 0 {
                    break;
                }
            }
            // Mediumband only when asked for or in mode transitions.
            if bw == 1 {
                bw = 2;
            }
            bandwidth = bandwidth_from_fec_index(bw as usize);
            self.auto_bandwidth = bandwidth;
            // No switch to SWB/FB until SILK has fully switched to wideband
            // and its variable low-pass is off.
            if !self.first
                && mode != OpusMode::CeltOnly
                && !self.silk_in_wb_without_variable_lp
                && rank(bandwidth) > rank(Wideband)
            {
                bandwidth = Wideband;
            }
        }
        if rank(bandwidth) > rank(self.max_bandwidth) {
            bandwidth = self.max_bandwidth;
        }
        if self.user_bandwidth != Auto {
            bandwidth = self.user_bandwidth;
        }
        // No Hybrid at unsafe CBR or maximum rates.
        if mode != OpusMode::CeltOnly && max_rate < 15000 && rank(bandwidth) > rank(Wideband) {
            bandwidth = Wideband;
        }
        // Nothing above the input's Nyquist rate.
        let top = match self.sampling_rate {
            8000 => Narrowband,
            12000 => Mediumband,
            16000 => Wideband,
            24000 => Superwideband,
            _ => Fullband,
        };
        if rank(bandwidth) > rank(top) {
            bandwidth = top;
        }
        // Down to the bandwidth the analysis detected, conservatively: never
        // below wideband in SILK and Hybrid, whose switches would need
        // redundancy.
        if let (Some(detected), Auto) = (self.detected_bandwidth, self.user_bandwidth) {
            let ch = self.stream_channels as i32;
            let min_detected = if equiv_rate <= 18000 * ch && mode == OpusMode::CeltOnly {
                Narrowband
            } else if equiv_rate <= 24000 * ch && mode == OpusMode::CeltOnly {
                Mediumband
            } else if equiv_rate <= 30000 * ch {
                Wideband
            } else if equiv_rate <= 44000 * ch {
                Superwideband
            } else {
                Fullband
            };
            let detected = if rank(detected) > rank(min_detected) {
                detected
            } else {
                min_detected
            };
            if rank(detected) < rank(bandwidth) {
                bandwidth = detected;
            }
        }
        bandwidth
    }

    /// The analysis of a packet and its digital-silence flag (see
    /// `analyse`).
    fn run_analysis(&mut self, input: PcmInput<'_>, frame_size: usize) -> (AnalysisInfo, bool) {
        self.analysis_read_bak = None;
        let n = frame_size * self.channels;
        // Digital silence is detected for every frame, at every complexity
        // and rate (opus_encoder.c:1246); the gate below only runs the
        // analysis itself (opus_encoder.c:1252).
        let is_silence = is_digital_silence(input, n);
        let info = if self.complexity >= 7 && self.sampling_rate >= 16000 {
            self.analysis_read_bak = Some(self.analysis.read_position());
            let pcm = match input {
                PcmInput::F32(s) => analysis::AnalysisPcm::F32(&s[..n]),
                PcmInput::I16(s) => analysis::AnalysisPcm::I16(&s[..n]),
            };
            self.analysis.run(
                modes::default_mode().mdct.fft(0),
                pcm,
                frame_size,
                self.channels,
                match input {
                    PcmInput::F32(_) => 24,
                    PcmInput::I16(_) => 16,
                },
            )
        } else {
            if self.analysis.initialized() {
                self.analysis.reset();
            }
            AnalysisInfo::zeroed()
        };
        // Track the peak signal energy of active frames, wherever the
        // analysis ran or not (opus_encoder.c:1311-1321).
        if (!info.valid || info.activity_probability > DTX_ACTIVITY_THRESHOLD) && !is_silence {
            let energy = compute_frame_energy(input, n);
            let decayed = 0.999 * self.peak_signal_energy;
            self.peak_signal_energy = if decayed > energy { decayed } else { energy };
        }
        (info, is_silence)
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
        // The frames of a multiframe packet share its decisions.
        let packet_level = self.frame_fec.is_none();

        let frame_rate = frame_rate_from_params(self.sampling_rate, frame_size)
            .ok_or("Invalid frame size for sampling rate")?;

        // Clamp caller-provided control fields the way libopus's ctl
        // interface does (OPUS_SET_BITRATE clamps to [500, 750000 per
        // channel], opus_encoder.c:2820-2828; complexity to [0, 10]).
        // Unclamped extremes overflow the bitrate arithmetic downstream
        // (bitrate*6 in celt, `equiv*(90+complexity)` in the mode
        // selection) — issue #27 deep scan.
        self.bitrate_bps = self.bitrate_bps.clamp(500, 750_000 * self.channels as i32);
        self.complexity = self.complexity.clamp(0, 10);
        self.packet_loss_perc = self.packet_loss_perc.clamp(0, 100);

        // The encoder consumes exactly frame_size * channels samples per call;
        // reject short input up front instead of panicking inside the HP filter
        // or the float→int conversion loops (issue #27 deep-scan).
        if input.len() < frame_size * self.channels {
            return Err("Input buffer too small for frame");
        }

        // libopus's signal analysis of the packet; the frames of a multiframe
        // packet read their part of it.
        let (analysis_info, is_silence) = match self.frame_analysis.take() {
            Some(frame) => frame,
            None => self.analyse(input, frame_size),
        };
        // libopus's own DTX decides where its analysis ran or the frame is
        // digital silence; SILK's DTX on the other frames.
        let silk_use_dtx = self.use_dtx && !(analysis_info.valid || is_silence);
        self.silk_use_dtx = silk_use_dtx;

        // The rate-dependent mono/stereo decision, once per packet
        // (opus_encoder.c:1409-1453).
        if packet_level {
            self.stream_channels = if self.channels == 2 {
                let equiv = compute_equiv_rate(
                    self.bitrate_bps,
                    self.channels,
                    frame_rate,
                    !self.use_cbr,
                    None,
                    self.complexity,
                    self.packet_loss_perc,
                );
                let voice_est = self.voice_estimate();
                let mut threshold = 17000 + ((voice_est * voice_est * (19000 - 17000)) >> 14);
                if self.stream_channels == 2 {
                    threshold -= 1000;
                } else {
                    threshold += 1000;
                }
                if equiv > threshold { 2 } else { 1 }
            } else {
                1
            };
        }

        // How far stereo input is from mono (libopus compute_stereo_width),
        // once per packet.
        let stereo_width = if self.channels == 2 && self.forced_mode.is_none() {
            compute_stereo_width(input, frame_size, self.sampling_rate, &mut self.width_mem)
        } else {
            0.0
        };

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
                self.stream_channels,
                frame_rate,
                !self.use_cbr,
                None,
                self.complexity,
                self.packet_loss_perc,
            );
            let voice_est = self.voice_estimate();
            let threshold = compute_mode_threshold(
                self.application,
                stereo_width,
                self.prev_enc_mode,
                voice_est,
            );
            let mut mode = if equiv >= threshold {
                OpusMode::CeltOnly
            } else {
                OpusMode::SilkOnly
            };
            // With in-band FEC and enough expected loss, SILK, whose LBRR
            // frames recover losses: above (128 - voice_est) / 16 percent.
            if self.use_inband_fec && self.packet_loss_perc > (128 - voice_est) >> 4 {
                mode = OpusMode::SilkOnly;
            }
            // libopus: voice with DTX is coded in SILK, whose voice activity
            // detection drives DTX.
            if silk_use_dtx && voice_est > 100 {
                mode = OpusMode::SilkOnly;
            }
            mode
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
        if self.forced_mode.is_none()
            && ((n_bytes as i64) * 8 * (self.sampling_rate as i64))
                < (tiny_budget_rate * (frame_size as i64))
        {
            mode = OpusMode::CeltOnly;
        }

        // SILK and Hybrid frames are 10 ms or longer: shorter frames are
        // coded in CELT, as libopus overrides the mode to meet the requested
        // frame size (opus_encode_native).
        if mode != OpusMode::CeltOnly && frame_size < self.sampling_rate as usize / 100 {
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

        // The frame's voice activity (opus_encode_frame_native:1916-1931).
        // None (0) in digital silence; from the analysis where it ran, with
        // an inactive frame counting as active when it is loud enough
        // against the loudest active frame; in CELT-only without an
        // analysis, the same comparison at half the noise energy; SILK and
        // Hybrid without an analysis leave the decision to SILK's own VAD
        // (`VAD_NO_DECISION`, resolved from `signalType` after the SILK
        // encoding, opus_encoder.c:2235-2241).
        let mut activity = VAD_NO_DECISION;
        if is_silence {
            activity = 0;
        } else if analysis_info.valid {
            activity = i32::from(analysis_info.activity_probability >= DTX_ACTIVITY_THRESHOLD);
            if activity == 0 {
                let noise_energy = compute_frame_energy(input, frame_size * self.channels);
                activity =
                    i32::from(self.peak_signal_energy < PSEUDO_SNR_THRESHOLD * noise_energy);
            }
        } else if mode == OpusMode::CeltOnly {
            // Boosting peak energy a bit because we didn't just average the
            // active frames (C's HALF32).
            let noise_energy = compute_frame_energy(input, frame_size * self.channels);
            activity = i32::from(
                f64::from(self.peak_signal_energy)
                    < f64::from(PSEUDO_SNR_THRESHOLD) * f64::from(0.5f32 * noise_energy),
            );
        }

        // The bandwidth and in-band FEC of the packet (opus_encode_native):
        // the bandwidth from the rate, the user's settings, the packet size,
        // the sampling rate and the analysis; then FEC, which can lower it
        // when the loss rate demands FEC at any cost. Like libopus's
        // `st->bandwidth` it carries over to the next packet.
        let (mut frame_bandwidth, lbrr_coded) = match self.frame_fec.take() {
            Some(decided) => decided,
            None => {
                let equiv_rate = compute_equiv_rate(
                    self.bitrate_bps,
                    self.stream_channels,
                    frame_rate,
                    !self.use_cbr,
                    Some(mode),
                    self.complexity,
                    self.packet_loss_perc,
                );
                let max_data_bytes = if self.use_cbr {
                    let bits =
                        self.bitrate_bps as i64 * frame_size as i64 / self.sampling_rate as i64;
                    (((bits + 4) / 8) as usize).min(output.len()).max(1)
                } else {
                    output.len().min(6 * OPUS_MAX_PACKET_BYTES)
                };
                let max_rate =
                    bits_to_bitrate(max_data_bytes as i32 * 8, self.sampling_rate, frame_size);
                let voice_est = self.voice_estimate();
                let mut frame_bandwidth =
                    self.decide_bandwidth(mode, equiv_rate, voice_est, max_rate);
                let lbrr_coded = decide_fec(
                    if self.use_inband_fec { 1 } else { 0 },
                    self.packet_loss_perc,
                    self.silk_enc.s_cmn.lbrr_enabled,
                    mode,
                    &mut frame_bandwidth,
                    equiv_rate,
                );
                // CELT has no mediumband: libopus codes it as wideband.
                if mode == OpusMode::CeltOnly && frame_bandwidth == Bandwidth::Mediumband {
                    frame_bandwidth = Bandwidth::Wideband;
                }
                self.bandwidth = frame_bandwidth;
                (frame_bandwidth, lbrr_coded)
            }
        };

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
        // Frames over 60 ms are split in every mode (80, 100 and 120 ms).
        if self.forced_mode.is_none()
            && ((mode != OpusMode::SilkOnly && frame_size > self.sampling_rate as usize / 50)
                || frame_size > 3 * self.sampling_rate as usize / 50)
        {
            return self.encode_multiframe(
                input,
                frame_size,
                mode,
                transition,
                (frame_bandwidth, lbrr_coded),
                output,
            );
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

        // A SILK stereo-to-mono switch waits two frames for SILK's downmix
        // (opus_encoder.c:1560-1570).
        if packet_level {
            if self.stream_channels == 1
                && self.prev_channels == 2
                && !self.silk_to_mono
                && mode != OpusMode::CeltOnly
                && self.prev_enc_mode != Some(OpusMode::CeltOnly)
            {
                self.silk_to_mono = true;
                self.stream_channels = 2;
            } else {
                self.silk_to_mono = false;
            }
        }
        // The channels coded: SILK and Hybrid frames code the input's, as
        // opus-rs's SILK encoder has no mono coding of stereo input.
        let coded_channels = if mode == OpusMode::CeltOnly {
            self.stream_channels
        } else {
            self.channels
        };
        let mut toc = gen_toc(mode, frame_rate, frame_bandwidth, coded_channels);
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
            output[0] = gen_toc(toc_mode, frame_rate_adj, plc_bw, coded_channels) | packet_code;
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
                    // CBR, m frames, and the padding flag if there is any
                    // padding (two bytes have none).
                    output[1] = m as u8;
                    if target_total > 2 {
                        output[1] |= 0x40;
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
                compute_redundancy_bytes(n_bytes, self.bitrate_bps, frame_rate, coded_channels);
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

        // The frame part of libopus's pcm_buf (opus_encoder.c:1982-2012):
        // this frame high-passed (VoIP) or DC-rejected, in float. CELT reads
        // it later behind the delayed history; SILK codes it rounded to 16
        // bits (enc_API.c, RES2INT16), except for VoIP (below). libopus
        // filters every frame, so the filter state stays current through
        // CELT-only frames, and RestrictedLowDelay's CELT input is
        // DC-rejected too (without the delay prefix).
        //
        // For the i16 entry point the frame is converted once into a scratch
        // buffer with the exact per-sample scaling a caller of `encode()`
        // performs (`s as f32 / 32768.0`), so packets stay byte-for-byte
        // identical across the two entry points (issue #28).
        if let PcmInput::I16(samples) = input {
            let n = frame_size * self.channels;
            self.buf_f32_frame.resize(n, 0.0);
            let staging = state_mut(&mut self.buf_f32_frame);
            for (dst, &src) in staging[..n].iter_mut().zip(samples[..n].iter()) {
                *dst = src as f32 / 32768.0;
            }
        }
        {
            let input_f32: &[f32] = match input {
                PcmInput::F32(samples) => samples,
                PcmInput::I16(_) => state_ref(&self.buf_f32_frame),
            };
            let ch = self.channels;
            let prefix = self.delay_compensation * ch;
            let total = prefix + frame_size * ch;
            self.buf_celt_pcm.resize(total, 0.0);
            filter_frame(
                &mut self.buf_celt_pcm[prefix..total],
                &mut self.hp_mem_float,
                input_f32,
                self.application == Application::Voip,
                cutoff_hz,
                frame_size,
                ch,
                self.sampling_rate,
            );
            // SILK's input. VoIP follows the fixed-point build, whose
            // `hp_cutoff` is an integer filter on the 16-bit input, so its
            // SILK layer matches that build byte for byte
            // (tests/silk_fixed_point_compare.rs, hybrid_cbr_silk_rate.rs).
            // The other applications follow the float build: SILK reads the
            // DC-rejected frame of pcm_buf rounded to 16 bits, so an input's
            // DC offset is gone in SILK packets as it is in CELT ones.
            let integer_hp = self.application == Application::Voip;
            self.buf_filtered.resize(frame_size * ch, 0);
            if integer_hp {
                let filtered = state_mut(&mut self.buf_filtered);
                match input {
                    // Native i16 path: the integer biquad takes it directly
                    // (issue #28).
                    PcmInput::I16(samples) => hp_cutoff_i16(
                        samples,
                        cutoff_hz,
                        filtered,
                        &mut self.hp_mem,
                        frame_size,
                        ch,
                        self.sampling_rate,
                    ),
                    PcmInput::F32(samples) => hp_cutoff(
                        samples,
                        cutoff_hz,
                        filtered,
                        &mut self.hp_mem,
                        frame_size,
                        ch,
                        self.sampling_rate,
                    ),
                }
            } else if self.application != Application::RestrictedLowDelay {
                let filtered = state_mut(&mut self.buf_filtered);
                for (d, &x) in filtered.iter_mut().zip(&self.buf_celt_pcm[prefix..total]) {
                    *d = float2int16(x);
                }
            }
        }

        let mut silk_bitrate = 0;
        let mut hb_gain = 1.0;
        let mut silk_switch_ready = false;
        // How often a channel's rate state machine found the switch ready in
        // this packet's control calls: each makes room for the redundancy in
        // the control struct the channels share (enc_API.c:260-268,
        // control_audio_bandwidth.c:86-90, 114-118).
        let mut silk_switch_room = 0;
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
                // The side follows the mid's rate (force_fs_kHz) but keeps
                // its own low-pass state machine, which can be ready too.
                let side_ready = self.channels == 2
                    && rate_control(
                        state_mut(&mut self.silk_enc_side),
                        self.silk_opus_can_switch,
                    )
                    .1;
                silk_switch_room += u32::from(ready) + u32::from(side_ready);
                if !silk_switch_prefill {
                    silk_switch_ready = ready || side_ready;
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
                        .init_for_enc(self.sampling_rate, silk_fs_khz * 1000);
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
                    side.s_cmn.n_channels = 2;
                }
            }

            // SILK codes CBR only in SILK-only frames: in hybrid CBR it runs
            // as VBR under a cap, and CELT takes up the rest of the packet
            // (opus_encoder.c:2150, 2163-2178). Like libopus's
            // silk_control_encoder (control_codec.c:76), the encoder states
            // take the flag for every packet, since the noise shaping reads
            // it.
            let silk_use_cbr = self.use_cbr && mode != OpusMode::Hybrid;

            // C silk_setup_LBRR (control_codec.c): per-packet FEC state and
            // LBRR gain increase (7 on the first FEC packet, then
            // max(7 - 0.2*loss, 3)), for each coded channel.
            for enc in [&mut self.silk_enc, &mut self.silk_enc_side]
                .into_iter()
                .take(self.channels)
            {
                enc.s_cmn.use_cbr = silk_use_cbr as i32;
                enc.s_cmn.use_in_band_fec = if self.use_inband_fec { 1 } else { 0 };
                enc.s_cmn.use_dtx = silk_use_dtx as i32;
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
                        activity,
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
                    silk_encode_prefill(state_mut(&mut self.silk_enc), pre_internal, activity);
                }
            }
            if silk_prefilled {
                // The coding call controls again, now unable to switch, and
                // repeats the LBRR setup of the prefill call, which counts
                // as a packet with LBRR.
                let (_, ready) = rate_control(state_mut(&mut self.silk_enc), false);
                let side_ready =
                    self.channels == 2 && rate_control(state_mut(&mut self.silk_enc_side), false).1;
                silk_switch_room += u32::from(ready) + u32::from(side_ready);
                silk_switch_ready = ready || side_ready;
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
            // Room for the redundant frame of the coming switch.
            for _ in 0..silk_switch_room {
                silk_max_bits -= silk_max_bits * 5 / (frame_ms + 5);
            }
            let silk_use_cbr = silk_use_cbr as i32;
            #[cfg(feature = "fuzzing")]
            fuzzing::record_silk_input(&state_ref(&self.buf_filtered)[..frame_size * self.channels]);
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
                    activity,
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
                    activity,
                )
            };
            if ret != 0 {
                return Err("SILK encoding failed");
            }
            // SILK's own VAD decides when Opus left no activity decision
            // (opus_encoder.c:2235-2241, from `silk_mode.signalType`).
            if activity == VAD_NO_DECISION {
                activity = i32::from(
                    i32::from(self.silk_enc.s_cmn.indices.signal_type)
                        != silk::define::TYPE_NO_VOICE_ACTIVITY,
                );
            }
            // What the next bandwidth decision may do (libopus
            // `silk_mode.allowBandwidthSwitch`, `inWBmodeWithoutVariableLP`).
            self.silk_allow_bandwidth_switch = self.silk_enc.allow_bandwidth_switch != 0;
            self.silk_in_wb_without_variable_lp =
                self.silk_enc.s_cmn.fs_khz == 16 && self.silk_enc.s_cmn.s_lp.mode == 0;
            // The TOC signals the rate SILK coded at (opus_encoder.c:2221-2232).
            if mode == OpusMode::SilkOnly {
                frame_bandwidth = match silk_fs_khz {
                    8 => Bandwidth::Narrowband,
                    12 => Bandwidth::Mediumband,
                    _ => Bandwidth::Wideband,
                };
                toc = gen_toc(mode, frame_rate, frame_bandwidth, coded_channels);
                output[0] = toc;
            }
            // A ready switch happens at the next packet, which prefills SILK
            // again; within a multiframe packet only after its last frame
            // (opus_encoder.c:2065).
            let silk_can_switch = silk_switch_ready && !self.nonfinal_frame;
            self.silk_opus_can_switch = silk_can_switch;
            // SILK DTX: the packet is the ToC alone, and the frame does not
            // count as coded (opus_encode_frame_native).
            if pn_bytes == 0 {
                return Ok(1);
            }
            self.silk_bw_switch = silk_can_switch;
            if silk_can_switch {
                // The packet before the switch ends with redundancy
                // (opus_encoder.c:2250-2260).
                redundancy_bytes =
                    compute_redundancy_bytes(n_bytes, self.bitrate_bps, frame_rate, coded_channels);
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

        // Build the CELT input exactly like C `opus_encode_frame_native`:
        //   pcm_buf = [delay-compensation prefix from ring]
        //             [dc_reject / hp_cutoff'd current frame]
        // CELT then reads pcm_buf[0..frame_size*channels] (opus_encoder.c:
        // 1966-2010, 2493). This delay + filter step is what libopus feeds
        // its CELT encoder; passing the raw input diverges from 1.6. The
        // delay line moves on in every mode, so the CELT input after a
        // SILK-only stretch starts from recent history, and the redundancy
        // frames of a SILK-only packet code it.
        {
            // The filtered frame is in place since before SILK ran; the
            // history and the delay line move on here, after the prefills
            // read the delay buffer.
            let ebuf = self.encoder_buffer;
            let total = (self.delay_compensation + frame_size) * ch;
            push_delay_history(
                &mut self.delay_buffer[..ebuf * ch],
                &mut self.buf_celt_pcm[..total],
                frame_size,
                ch,
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

        // The stereo width (opus_encoder.c:2320-2348): from the rate, or in
        // Hybrid stereo SILK's; below full width the CELT input fades
        // towards mono.
        if mode != OpusMode::Hybrid || self.stream_channels == 1 {
            let equiv_rate = compute_equiv_rate(
                self.bitrate_bps,
                self.stream_channels,
                frame_rate,
                !self.use_cbr,
                Some(mode),
                self.complexity,
                self.packet_loss_perc,
            );
            self.stereo_width_q14 = if equiv_rate > 32000 {
                16384
            } else if equiv_rate < 16000 {
                0
            } else {
                16384 - 2048 * (32000 - equiv_rate) / (equiv_rate - 14000)
            };
        } else {
            self.stereo_width_q14 = i32::from(self.silk_enc.stereo.smth_width_q14);
        }
        if ch == 2 && (self.hybrid_stereo_width_q14 < 1 << 14 || self.stereo_width_q14 < 1 << 14) {
            stereo_fade(
                &mut self.buf_celt_pcm[..frame_size * 2],
                self.hybrid_stereo_width_q14 as f32 * (1.0 / 16384.0),
                self.stereo_width_q14 as f32 * (1.0 / 16384.0),
                self.sampling_rate,
            );
            self.hybrid_stereo_width_q14 = self.stereo_width_q14;
        }

        // The CELT settings of every frame (opus_encoder.c:2262-2295):
        // libopus `CELT_SET_END_BAND` matches the TOC bandwidth so the
        // decoder reads the same number of coded bands (issue #37).
        if mode != OpusMode::SilkOnly || redundancy {
            // libopus `CELT_SET_ANALYSIS`: the analysis steers CELT's
            // allocation, trim, VBR target and pitch pre-filter.
            self.celt_enc.set_analysis(&analysis_info);
            self.celt_enc.set_stream_channels(coded_channels);
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
            // libopus: Hybrid VBR is unconstrained (can steal from SILK);
            // CELT-only VBR follows OPUS_SET_VBR_CONSTRAINT.
            self.celt_enc
                .set_constrained_vbr(mode == OpusMode::CeltOnly && self.vbr_constraint);
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
            let celt_input: &[f32] = {
                let n = frame_size * ch;
                self.buf_celt_input.resize(n, 0.0);
                for i in 0..frame_size {
                    for c in 0..ch {
                        self.buf_celt_input[c * frame_size + i] = self.buf_celt_pcm[i * ch + c];
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

        self.first = false;
        // libopus's own DTX (decide_dtx_mode), where its analysis ran or the
        // frame is digital silence: after 200 ms without activity the frame
        // is sent as its ToC alone, with one coded frame every 400 ms.
        if self.use_dtx && (analysis_info.valid || is_silence) {
            let frame_ms_q1 = (2000 * frame_size as i64 / self.sampling_rate as i64) as i32;
            if decide_dtx_mode(activity != 0, &mut self.nb_no_activity_ms_q1, frame_ms_q1) {
                self.prev_enc_mode = Some(next_prev_mode);
                self.prev_channels = self.stream_channels;
                output[0] = toc;
                return Ok(1);
            }
        } else {
            self.nb_no_activity_ms_q1 = 0;
        }

        // C opus_encoder.c:2578-2589 (1.6.1): "In the unlikely case that the
        // SILK encoder busted its target, tell the decoder to call the PLC":
        // the packet is the ToC and a zero byte, a 1-byte frame that decoders
        // conceal. With FEC the LBRR section can squeeze the main frame below
        // what even the damage-control re-encode fits, which is exactly the
        // case this fallback exists for.
        if self.rc.error != 0 || self.rc.tell() > (n_bytes as i32 - 1) * 8 {
            if output.len() < 2 {
                return Err("Output buffer too small");
            }
            let mut ret = 2usize;
            if self.use_cbr && n_bytes >= 2 {
                // CBR: libopus pads that 2-byte packet, the ToC and a zero
                // byte, to the nominal size (opus_packet_pad): a code-3
                // packet of the 1-byte frame.
                let target_total = n_bytes.min(output.len());
                if target_total > 2 {
                    let pad_amount = target_total - 3;
                    output[0] = toc | 0x03;
                    // 1 frame, CBR, and the padding flag if there is any.
                    output[1] = if pad_amount > 0 { 0x41 } else { 0x01 };
                    let mut ptr = 2;
                    if pad_amount > 0 {
                        let nb_255s = (pad_amount - 1) / 255;
                        for _ in 0..nb_255s {
                            output[ptr] = 255;
                            ptr += 1;
                        }
                        output[ptr] = (pad_amount - 255 * nb_255s - 1) as u8;
                        ptr += 1;
                    }
                    // The frame's zero byte, then the zero padding.
                    for b in &mut output[ptr..target_total] {
                        *b = 0;
                    }
                    self.prev_enc_mode = Some(next_prev_mode);
                self.prev_channels = self.stream_channels;
                    return Ok(target_total);
                }
                ret = target_total.max(1);
            }
            // VBR: the 2-byte code-0 packet.
            output[0] = toc;
            output[1] = 0;
            self.prev_enc_mode = Some(next_prev_mode);
                self.prev_channels = self.stream_channels;
            return Ok(ret);
        }
        self.range_final = self.rc.rng ^ redundant_rng;

        self.prev_enc_mode = Some(next_prev_mode);
                self.prev_channels = self.stream_channels;
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
                cbr_bytes.min(max_data_bytes).max(1)
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
        // CBR pads the packet to its size, which can exceed the 1276 bytes
        // of one frame (opus_encoder.c:2602, 2646-2653).
        if self.use_cbr {
            return Ok(pad_packet(output, end, cbr_bytes.min(max_data_bytes)));
        }
        Ok(end)
    }

    /// Repacketize 40/60 ms CELT/Hybrid input as 20 ms sub-frames, and
    /// 80-120 ms input as SILK's 40, 20 or 60 ms frames or CELT/Hybrid's
    /// 20 ms ones, in one code 1/2/3 Opus packet (libopus
    /// `encode_multiframe_packet`). Every sub-frame
    /// is forced to `mode` and to the packet's bandwidth and FEC decision
    /// `fec`, so the packet carries a single config. A mode switch's
    /// redundancy goes to the first sub-frame, or going to CELT, to the last
    /// (opus_encoder.c:1781-1783).
    fn encode_multiframe(
        &mut self,
        input: PcmInput<'_>,
        frame_size: usize,
        mode: OpusMode,
        transition: FrameTransition,
        fec: (Bandwidth, i32),
        output: &mut [u8],
    ) -> Result<usize, &'static str> {
        const MAX_SUB: usize = 6; // up to 120 ms of 20 ms frames
        let ch = self.channels;
        // libopus encode_multiframe_packet: SILK codes 80 ms as 2 x 40 ms,
        // 120 ms as 2 x 60 ms and 100 ms as 5 x 20 ms; CELT and Hybrid use
        // 20 ms frames.
        let fs = self.sampling_rate as usize;
        let enc_frame = if mode == OpusMode::SilkOnly && frame_size == 2 * fs / 25 {
            fs / 25
        } else if mode == OpusMode::SilkOnly && frame_size == 3 * fs / 25 {
            3 * fs / 50
        } else {
            fs / 50
        };
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
        // Where each sub-packet's frame starts in `sub`.
        let mut offs = [0usize; MAX_SUB];
        // Under CBR the packet is the CBR size, to which it is padded
        // (encode_multiframe_packet).
        let cbr_total = if self.use_cbr {
            let bits = self.bitrate_bps as i64 * frame_size as i64 / self.sampling_rate as i64;
            // max_data_bytes is at most six frames of 1275 bytes
            // (opus_encoder.c:1221).
            Some((((bits + 4) / 8) as usize).clamp(1, 6 * 1275).min(output.len()))
        } else {
            None
        };
        // Each frame gets at most its share of the packet (the CBR size, or
        // the output buffer in VBR) less the worst-case header, and no more
        // than the bitrate's bytes per frame, as in libopus.
        let packet_room = cbr_total.unwrap_or(output.len());
        let max_header = if nb == 2 { 3 } else { 2 + (nb - 1) * 2 };
        let max_len_sum = (nb + packet_room).saturating_sub(max_header);
        let frame_bytes = (3 * self.bitrate_bps as i64
            / (3 * 8 * self.sampling_rate as i64 / enc_frame as i64))
            as usize;
        let sub_budget = frame_bytes.min(max_len_sum / nb).clamp(2, 1276);
        let mut used = 0usize;
        // The frames read the packet's analysis one at a time, from where it
        // started (libopus `analysis_read_pos_bak`).
        let analysed = self.analysis_read_bak;
        if let Some(read_pos) = analysed {
            self.analysis.set_read_position(read_pos);
        }
        self.forced_mode = Some(mode);
        let enc_res = (|| -> Result<(), &'static str> {
            for i in 0..nb {
                let start = i * enc_frame * ch;
                let end = start + enc_frame * ch;
                let sub_input = match input {
                    PcmInput::F32(s) => PcmInput::F32(&s[start..end]),
                    PcmInput::I16(s) => PcmInput::I16(&s[start..end]),
                };
                // Each frame's analysis, where it ran, and digital silence,
                // which libopus detects for every frame (opus_encoder.c:1802).
                let info = if analysed.is_some() {
                    self.analysis.get_info(enc_frame)
                } else {
                    AnalysisInfo::zeroed()
                };
                self.frame_analysis = Some((info, is_digital_silence(sub_input, enc_frame * ch)));
                let to_celt = transition.to_celt && i == nb - 1;
                self.frame_transition = Some(FrameTransition {
                    redundancy: transition.redundancy
                        && (to_celt || (!transition.to_celt && i == 0)),
                    celt_to_silk: transition.celt_to_silk,
                    to_celt,
                    prefill: transition.prefill && i == 0,
                });
                self.frame_fec = Some(fec);
                self.nonfinal_frame = i < nb - 1;
                let budget = sub_budget.min(max_len_sum.saturating_sub(used)).max(2);
                let n = self.encode_impl(sub_input, enc_frame, &mut sub[i][..budget])?;
                if n == 0 {
                    return Err("Empty sub-frame");
                }
                used += n;
                // A sub-packet is not always a ToC and a frame: CBR pads it
                // as a code 3 packet, and a frame over its budget or below 3
                // bytes is sent empty, padded the same way (a two-byte code 3
                // packet has no room for padding). Take its one frame, as
                // libopus's repacketizer does.
                if n == 2 && sub[i][0] & 0x03 == 0x03 {
                    (offs[i], lens[i]) = (2, 0);
                    continue;
                }
                let frames = frame_payloads(&sub[i][..n])?;
                if frames.len() != 1 {
                    return Err("Unexpected sub-frame packet");
                }
                offs[i] = frames[0].as_ptr() as usize - sub[i].as_ptr() as usize;
                lens[i] = frames[0].len();
            }
            Ok(())
        })();
        self.forced_mode = None;
        self.frame_transition = None;
        self.frame_fec = None;
        self.frame_analysis = None;
        self.nonfinal_frame = false;
        enc_res?;

        // Every 20 ms frame in DTX: libopus's repacketizer sends the empty
        // frames as code 1 (two frames) or code 3 CBR, unpadded.
        if lens[..nb].iter().all(|&len| len == 0) && sub[..nb].iter().all(|s| s[0] & 0x03 == 0) {
            let toc = sub[0][0] & 0xFC;
            if nb == 2 {
                output[0] = toc | 0x01;
                return Ok(1);
            }
            if output.len() < 2 {
                return Err("Output buffer too small");
            }
            output[0] = toc | 0x03;
            output[1] = nb as u8;
            return Ok(2);
        }

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
        // The packet layout is libopus's repacketizer
        // (opus_repacketizer_out_range_impl, repacketizer.c:194-318): code 1
        // for two equal frames, code 2 for two different ones, code 3 for
        // more — and code 3 carries the VBR flag and per-frame size bytes
        // only when the frames differ in size. Under CBR the packet pads to
        // its nominal size, which turns any layout into padded code 3
        // (opus_encoder.c:1831).
        let vbr = lens[..nb].iter().any(|&len| len != lens[0]);
        let length_bytes: usize = if vbr {
            lens[..nb - 1]
                .iter()
                .map(|&len| if len < 252 { 1 } else { 2 })
                .sum()
        } else {
            0
        };
        let sum = lens[..nb].iter().sum::<usize>();
        let code3_tot = if vbr { 2 + length_bytes + sum } else { 2 + nb * lens[0] };
        let maxlen = cbr_total.unwrap_or(0);
        let pad = cbr_total.is_some();
        // Code 3 runs for more than two frames, and under CBR whenever the
        // frames leave room to pad up to the nominal size. The total for
        // code 1/2 matches C (repacketizer.c:194-224).
        let tot_size = match nb {
            2 if vbr => lens[0] + lens[1] + 2 + usize::from(lens[0] >= 252),
            2 => 2 * lens[0] + 1,
            _ => code3_tot,
        };
        let code3 = nb > 2 || (pad && tot_size < maxlen);
        let mut pad_amount = 0;
        if code3 {
            pad_amount = maxlen.saturating_sub(code3_tot);
            let pad_flag = if pad_amount > 0 { 0x40 } else { 0 };
            push(output, &mut pos, toc | 0x03)?;
            push(output, &mut pos, nb as u8 | if vbr { 0x80 } else { 0 } | pad_flag)?;
            if pad_amount > 0 {
                let nb_255s = (pad_amount - 1) / 255;
                for _ in 0..nb_255s {
                    push(output, &mut pos, 255)?;
                }
                push(output, &mut pos, (pad_amount - 255 * nb_255s - 1) as u8)?;
                // The length bytes are part of the padding.
                pad_amount -= nb_255s + 1;
            }
            if vbr {
                for i in 0..nb - 1 {
                    put_size(output, &mut pos, lens[i])?;
                }
            }
        } else if vbr {
            push(output, &mut pos, toc | 0x02)?;
            put_size(output, &mut pos, lens[0])?;
        } else {
            push(output, &mut pos, toc | 0x01)?;
        }

        for i in 0..nb {
            let payload = &sub[i][offs[i]..offs[i] + lens[i]];
            if pos + payload.len() > output.len() {
                return Err("Output buffer too small");
            }
            output[pos..pos + payload.len()].copy_from_slice(payload);
            pos += payload.len();
        }
        if pos + pad_amount > output.len() {
            return Err("Output buffer too small");
        }
        output[pos..pos + pad_amount].fill(0);
        pos += pad_amount;
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
    /// The decoder complexity, 0 to 10 (libopus `OPUS_SET_COMPLEXITY` on a
    /// decoder). The neural network features use it: deep PLC from 5, LACE
    /// at 6 and NoLACE from 7.
    complexity: i32,
    /// What the Opus layer tells SILK (libopus `DecControl`).
    dec_control: silk::dec_api::SilkDecControl,

    // libopus clears the fields below on a reset.
    /// Channels of the last packet (libopus `stream_channels`).
    stream_channels: usize,
    /// Bandwidth of the last packet.
    bandwidth: Bandwidth,
    /// Mode of the last packet (libopus `st->mode`); decode_fec checks it.
    mode: Option<OpusMode>,
    /// Mode of the last decoded or concealed frame (libopus `prev_mode`).
    prev_mode: Option<OpusMode>,
    /// Samples per frame of the last packet (libopus `st->frame_size`).
    frame_size: usize,
    /// Whether the last frame ended with a SILK-to-CELT redundant frame.
    prev_redundancy: bool,
    last_packet_duration: usize,

    /// Range decoder state after the last decoded frame (libopus
    /// `OPUS_GET_FINAL_RANGE`), 0 after a concealed one.
    range_final: u32,

    pub hybrid_skip_celt: bool,
}

impl OpusDecoder {
    pub fn new(sampling_rate: i32, channels: usize) -> Result<Self, &'static str> {
        Self::check_settings(sampling_rate, channels)?;
        let mut decoder = Self::empty();
        decoder.setup(sampling_rate, channels);
        Ok(decoder)
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
        Self::check_settings(sampling_rate, channels)?;
        let decoder = slot.write(Self::empty());
        decoder.setup(sampling_rate, channels);
        Ok(decoder)
    }

    fn check_settings(sampling_rate: i32, channels: usize) -> Result<(), &'static str> {
        if ![8000, 12000, 16000, 24000, 48000].contains(&sampling_rate) {
            return Err("Invalid sampling rate");
        }
        if ![1, 2].contains(&channels) {
            return Err("Invalid number of channels");
        }
        Ok(())
    }

    /// A decoder with empty buffers for [`Self::setup`]. Without the `heap`
    /// feature it is a constant, which the compiler copies straight into the
    /// destination, so `init` needs no stack for it.
    #[cfg(not(feature = "heap"))]
    #[inline(always)]
    fn empty() -> Self {
        Self::EMPTY
    }

    #[cfg(not(feature = "heap"))]
    const EMPTY: Self = Self {
        celt_dec: CeltDecoder::EMPTY,
        silk_dec: silk::dec_api::SilkDecoder::EMPTY,
        sampling_rate: 0,
        channels: 0,
        complexity: 0,
        dec_control: silk::dec_api::SilkDecControl {
            n_channels_api: 0,
            n_channels_internal: 0,
            api_sample_rate: 0,
            internal_sample_rate: 0,
            payload_size_ms: 0,
            #[cfg(feature = "deep-plc")]
            enable_deep_plc: false,
            #[cfg(feature = "osce")]
            osce_method: crate::dnn::osce::OsceMethod::None,
        },
        stream_channels: 0,
        bandwidth: Bandwidth::Auto,
        mode: None,
        prev_mode: None,
        frame_size: 0,
        prev_redundancy: false,
        last_packet_duration: 0,
        range_final: 0,
        hybrid_skip_celt: false,
    };

    #[cfg(feature = "heap")]
    fn empty() -> Self {
        Self {
            celt_dec: Box::new(CeltDecoder::EMPTY),
            silk_dec: Box::new(silk::dec_api::SilkDecoder::EMPTY),
            sampling_rate: 0,
            channels: 0,
            complexity: 0,
            dec_control: silk::dec_api::SilkDecControl::default(),
            stream_channels: 0,
            bandwidth: Bandwidth::Auto,
            mode: None,
            prev_mode: None,
            frame_size: 0,
            prev_redundancy: false,
            last_packet_duration: 0,
            range_final: 0,
            hybrid_skip_celt: false,
        }
    }

    /// Sets an [`Self::empty`] decoder up for valid settings, in place
    /// (libopus `opus_decoder_init`).
    fn setup(&mut self, sampling_rate: i32, channels: usize) {
        self.sampling_rate = sampling_rate;
        self.channels = channels;
        self.stream_channels = channels;
        self.dec_control.api_sample_rate = sampling_rate;
        self.dec_control.n_channels_api = channels as i32;
        self.silk_dec.init_states();
        self.celt_dec
            .init(modes::default_mode(), channels, sampling_rate);
        self.prev_mode = None;
        self.frame_size = sampling_rate as usize / 400;
    }

    /// The range decoder's final state after the last frame decoded, like
    /// libopus `OPUS_GET_FINAL_RANGE`: it equals the encoder's final range
    /// for that frame (the value an Opus test vector stores with each
    /// packet), XORed with the redundant frame's when the frame has one. It
    /// is 0 after a frame that was concealed (lost, empty or a single byte).
    pub fn final_range(&self) -> u32 {
        self.range_final
    }

    /// Sets the decoder complexity (libopus `OPUS_SET_COMPLEXITY`), 0 to 10;
    /// a new decoder has 0. Without the neural network features it changes
    /// nothing.
    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), &'static str> {
        if !(0..=10).contains(&complexity) {
            return Err("complexity must be 0 to 10");
        }
        self.complexity = complexity;
        // libopus passes it on to CELT (OPUS_SET_COMPLEXITY).
        #[cfg(feature = "deep-plc")]
        {
            self.celt_dec.deep_plc.complexity = complexity;
        }
        Ok(())
    }

    /// The decoder complexity (libopus `OPUS_GET_COMPLEXITY`).
    pub fn complexity(&self) -> i32 {
        self.complexity
    }

    /// Decode one Opus packet into interleaved `f32` PCM, returning the
    /// number of samples per channel written.
    ///
    /// Packet loss follows libopus: pass an empty packet (`&[]`) to conceal
    /// `frame_size` samples (a multiple of 2.5 ms), or a ToC-only 1-byte
    /// packet to conceal the duration that ToC describes. Either way the
    /// frame is concealed in the previous packet's mode.
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
        let n = self.decode_native(input, frame_size, output, false)?;
        // Leave no stale samples past the decoded ones.
        if let Some(rest) = output.get_mut(n * self.channels..) {
            rest.fill(0.0);
        }
        Ok(n)
    }

    /// Rebuild a lost packet from the in-band FEC of the packet after it,
    /// into interleaved `f32` PCM, returning the number of samples per
    /// channel written (always `frame_size`). This is libopus's
    /// `opus_decode(..., decode_fec = 1)`.
    ///
    /// `input` is the packet that followed the loss. When it is SILK-only or
    /// Hybrid, the last frame's worth of `frame_size` (one frame of `input`)
    /// is rebuilt from the LBRR of its first frame (concealed where a SILK
    /// frame has none), and anything before it is concealed. All of
    /// `frame_size` is concealed, as for `decode(&[], frame_size, ..)`, when
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
        let n = self.decode_native(input, frame_size, output, true)?;
        if let Some(rest) = output.get_mut(n * self.channels..) {
            rest.fill(0.0);
        }
        Ok(n)
    }

    /// libopus `opus_decode_native` (not self-delimited, no soft clipping).
    fn decode_native(
        &mut self,
        data: &[u8],
        frame_size: usize,
        pcm: &mut [f32],
        decode_fec: bool,
    ) -> Result<usize, &'static str> {
        let fs = self.sampling_rate as usize;
        let ch = self.channels;
        // For FEC/PLC, frame_size has to be a multiple of 2.5 ms.
        if (decode_fec || data.is_empty()) && !frame_size.is_multiple_of(fs / 400) {
            return Err("frame_size must be a multiple of 2.5 ms");
        }
        if data.is_empty() {
            // checked_mul: frame_size is caller-chosen and may be absurd.
            if frame_size.checked_mul(ch).is_none_or(|n| pcm.len() < n) {
                return Err("Output buffer too small for packet");
            }
            let mut pcm_count = 0;
            loop {
                let ret =
                    self.decode_frame(None, &mut pcm[pcm_count * ch..], frame_size - pcm_count, false)?;
                pcm_count += ret;
                if pcm_count >= frame_size {
                    break;
                }
            }
            self.last_packet_duration = pcm_count;
            return Ok(pcm_count);
        }

        let toc = data[0];
        let packet_mode = mode_from_toc(toc);
        let packet_bandwidth = bandwidth_from_toc(toc);
        let packet_frame_size = frame_samples_from_toc(toc, self.sampling_rate)
            .ok_or("Invalid TOC for sampling rate")?;
        let packet_stream_channels = channels_from_toc(toc);

        let frames = opus_packet_parse(data)?;

        if decode_fec {
            // If no FEC can be present, run the PLC.
            if frame_size < packet_frame_size
                || packet_mode == OpusMode::CeltOnly
                || self.mode == Some(OpusMode::CeltOnly)
            {
                return self.decode_native(&[], frame_size, pcm, false);
            }
            if frame_size.checked_mul(ch).is_none_or(|n| pcm.len() < n) {
                return Err("Output buffer too small for packet");
            }
            // Otherwise, run the PLC on everything except the size for which
            // there might be FEC.
            let duration_copy = self.last_packet_duration;
            if frame_size - packet_frame_size != 0 {
                if let Err(e) =
                    self.decode_native(&[], frame_size - packet_frame_size, pcm, false)
                {
                    self.last_packet_duration = duration_copy;
                    return Err(e);
                }
            }
            // Complete with FEC.
            self.mode = Some(packet_mode);
            self.bandwidth = packet_bandwidth;
            self.frame_size = packet_frame_size;
            self.stream_channels = packet_stream_channels;
            let at = ch * (frame_size - packet_frame_size);
            self.decode_frame(Some(frames[0]), &mut pcm[at..], packet_frame_size, true)?;
            self.last_packet_duration = frame_size;
            return Ok(frame_size);
        }

        let count = frames.len();
        if count * packet_frame_size > frame_size {
            return Err("frame_size too small for packet");
        }
        if pcm.len() < count * packet_frame_size * ch {
            return Err("Output buffer too small for packet");
        }

        // Update the state as the last step to avoid updating it on an
        // invalid packet.
        self.mode = Some(packet_mode);
        self.bandwidth = packet_bandwidth;
        self.frame_size = packet_frame_size;
        self.stream_channels = packet_stream_channels;

        let mut nb_samples = 0;
        for frame in frames.iter() {
            let ret = self.decode_frame(
                Some(*frame),
                &mut pcm[nb_samples * ch..],
                frame_size - nb_samples,
                false,
            )?;
            nb_samples += ret;
        }
        self.last_packet_duration = nb_samples;
        Ok(nb_samples)
    }

    /// libopus `opus_decode_frame`: decodes one frame, or conceals one when
    /// `data` is `None` or at most one byte, into interleaved `pcm`.
    fn decode_frame(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: usize,
        decode_fec: bool,
    ) -> Result<usize, &'static str> {
        let fs = self.sampling_rate as usize;
        let ch = self.channels;
        let f20 = fs / 50;
        let f10 = f20 >> 1;
        let f5 = f10 >> 1;
        let f2_5 = f5 >> 1;
        if frame_size < f2_5 {
            return Err("frame_size too small");
        }
        // Limit frame_size to avoid excessive stack allocations.
        let mut frame_size = frame_size.min(fs / 25 * 3);
        // Payloads of 1 (2 including ToC) or 0 trigger the PLC/DTX.
        let mut len = data.map_or(0, |d| d.len()) as i32;
        let data = data.filter(|_| len > 1);
        if data.is_none() {
            // In that case, don't conceal more than what the ToC says.
            frame_size = frame_size.min(self.frame_size);
        }
        let mut audiosize;
        let mode;
        let bandwidth;
        let mut dec;
        if let Some(d) = data {
            audiosize = self.frame_size;
            mode = self.mode.ok_or("no packet mode")?;
            bandwidth = Some(self.bandwidth);
            dec = RangeCoder::new_decoder(d);
        } else {
            audiosize = frame_size;
            // Run PLC using last used mode (CELT if we ended with CELT
            // redundancy).
            let plc_mode = if self.prev_redundancy {
                Some(OpusMode::CeltOnly)
            } else {
                self.prev_mode
            };
            bandwidth = None;
            let Some(m) = plc_mode else {
                // If we haven't got any packet yet, all we can do is return
                // zeros.
                pcm[..audiosize * ch].fill(0.0);
                return Ok(audiosize);
            };
            mode = m;
            // Avoids trying to run the PLC on sizes other than 2.5 (CELT),
            // 5 (CELT), 10, or 20 (e.g. 12.5 or 30 ms).
            if audiosize > f20 {
                let mut at = 0;
                loop {
                    let ret = self.decode_frame(None, &mut pcm[at * ch..], audiosize.min(f20), false)?;
                    at += ret;
                    audiosize -= ret;
                    if audiosize == 0 {
                        break;
                    }
                }
                return Ok(frame_size);
            } else if audiosize < f20 {
                if audiosize > f10 {
                    audiosize = f10;
                } else if mode != OpusMode::SilkOnly && audiosize > f5 && audiosize < f10 {
                    audiosize = f5;
                }
            }
            dec = RangeCoder::new_decoder(&[]);
        }

        // CELT accumulates on top of the SILK output.
        let celt_accum = mode != OpusMode::CeltOnly;

        let mut transition = false;
        let mut pcm_transition = [0.0f32; F5_MAX];
        if data.is_some()
            && self.prev_mode.is_some_and(|prev| {
                (mode == OpusMode::CeltOnly && prev != OpusMode::CeltOnly && !self.prev_redundancy)
                    || (mode != OpusMode::CeltOnly && prev == OpusMode::CeltOnly)
            })
        {
            transition = true;
        }
        let transition_size = f5.min(audiosize);
        if transition && mode == OpusMode::CeltOnly {
            let _ = self.decode_frame(None, &mut pcm_transition[..transition_size * ch], transition_size, false);
        }
        if audiosize > frame_size {
            return Err("frame_size too small for packet");
        }
        frame_size = audiosize;

        // SILK processing.
        if mode != OpusMode::CeltOnly {
            let pcm_too_small = frame_size < f10;
            let mut pcm_silk = [0.0f32; F10_MAX];

            if self.prev_mode == Some(OpusMode::CeltOnly) {
                self.silk_dec.reset();
            }

            // The SILK PLC cannot produce frames of less than 10 ms.
            self.dec_control.payload_size_ms = 10.max(1000 * audiosize as i32 / fs as i32);

            if data.is_some() {
                self.dec_control.n_channels_internal = self.stream_channels as i32;
                self.dec_control.internal_sample_rate = if mode == OpusMode::SilkOnly {
                    match bandwidth {
                        Some(Bandwidth::Narrowband) => 8000,
                        Some(Bandwidth::Mediumband) => 12000,
                        _ => 16000,
                    }
                } else {
                    16000
                };
            }
            #[cfg(feature = "deep-plc")]
            {
                self.dec_control.enable_deep_plc = self.complexity >= 5;
            }
            #[cfg(feature = "osce")]
            {
                use crate::dnn::osce::OsceMethod;
                self.dec_control.osce_method = match self.complexity {
                    7.. => OsceMethod::Nolace,
                    6 => OsceMethod::Lace,
                    _ => OsceMethod::None,
                };
            }

            let lost_flag = if data.is_none() {
                silk::decode_frame::FLAG_PACKET_LOST
            } else if decode_fec {
                silk::decode_frame::FLAG_DECODE_LBRR
            } else {
                silk::decode_frame::FLAG_DECODE_NORMAL
            };
            let ctrl = self.dec_control;
            let pcm_ptr: &mut [f32] = if pcm_too_small { &mut pcm_silk } else { pcm };
            let mut decoded_samples = 0;
            loop {
                // Call SILK decoder.
                let first_frame = decoded_samples == 0;
                let (silk_ret, mut silk_frame_size) = self.silk_dec.silk_decode(
                    &ctrl,
                    lost_flag,
                    first_frame,
                    &mut dec,
                    &mut pcm_ptr[decoded_samples * ch..],
                );
                if silk_ret != 0 {
                    if lost_flag != silk::decode_frame::FLAG_DECODE_NORMAL {
                        // PLC failure should not be fatal.
                        silk_frame_size = frame_size;
                        let at = decoded_samples * ch;
                        let end = (at + frame_size * ch).min(pcm_ptr.len());
                        pcm_ptr[at..end].fill(0.0);
                    } else {
                        return Err("SILK decoding failed");
                    }
                }
                decoded_samples += silk_frame_size;
                if decoded_samples >= frame_size {
                    break;
                }
            }
            if pcm_too_small {
                pcm[..frame_size * ch].copy_from_slice(&pcm_silk[..frame_size * ch]);
            }
        }

        let mut redundancy = false;
        let mut redundancy_bytes = 0i32;
        let mut celt_to_silk = false;
        if !decode_fec
            && mode != OpusMode::CeltOnly
            && data.is_some()
            && dec.tell() + 17 + 20 * (mode == OpusMode::Hybrid) as i32 <= 8 * len
        {
            // Check if we have a redundant 0-8 kHz band.
            redundancy = mode != OpusMode::Hybrid || dec.decode_bit_logp(12);
            if redundancy {
                celt_to_silk = dec.decode_bit_logp(1);
                // redundancy_bytes will be at least two, in the non-hybrid
                // case due to the ec_tell() check above.
                redundancy_bytes = if mode == OpusMode::Hybrid {
                    dec.dec_uint(256) as i32 + 2
                } else {
                    len - ((dec.tell() + 7) >> 3)
                };
                len -= redundancy_bytes;
                // This is a sanity check. It should never happen for a valid
                // packet, so the exact behaviour is not normative.
                if len * 8 < dec.tell() {
                    len = 0;
                    redundancy_bytes = 0;
                    redundancy = false;
                }
                // Shrink decoder because of raw bits.
                dec.storage = dec.storage.wrapping_sub(redundancy_bytes as u32);
            }
        }
        let start_band = if mode != OpusMode::CeltOnly { 17 } else { 0 };

        if redundancy {
            transition = false;
        }

        if transition && mode != OpusMode::CeltOnly {
            let _ = self.decode_frame(None, &mut pcm_transition[..transition_size * ch], transition_size, false);
        }

        if let Some(bw) = bandwidth {
            let endband = match bw {
                Bandwidth::Narrowband => 13,
                Bandwidth::Mediumband | Bandwidth::Wideband => 17,
                Bandwidth::Superwideband => 19,
                _ => 21,
            };
            self.celt_dec.set_end_band(endband);
        }
        self.celt_dec.set_stream_channels(self.stream_channels);

        let (len_u, rb) = (len.max(0) as usize, redundancy_bytes as usize);
        let redundant_data = data.map(|d| &d[len_u.min(d.len())..(len_u + rb).min(d.len())]);
        let mut redundant_audio = [0.0f32; F5_MAX];
        let mut redundant_rng = 0u32;

        // 5 ms redundant frame for CELT->SILK.
        if redundancy && celt_to_silk {
            // If the previous frame did not use CELT (the first redundancy
            // frame in a transition from SILK may have been lost) then the
            // CELT decoder is stale at this point and the redundancy audio is
            // not useful, however the final range is still needed (for
            // testing), so the redundancy is always decoded but the decoded
            // audio may not be used.
            self.celt_dec.set_start_band(0);
            let _ = self.celt_dec.decode_with_ec(
                redundant_data,
                None,
                &mut redundant_audio[..f5 * ch],
                f5,
                false,
            );
            redundant_rng = self.celt_dec.rng();
        }

        // MUST be after PLC.
        self.celt_dec.set_start_band(start_band);

        let mut celt_ok = true;
        if mode != OpusMode::SilkOnly {
            let celt_frame_size = f20.min(frame_size);
            // Make sure to discard any previous CELT state.
            if self.prev_mode.is_some_and(|prev| prev != mode) && !self.prev_redundancy {
                self.celt_dec.reset_state();
            }
            // Decode CELT.
            let celt_data = if decode_fec {
                None
            } else {
                data.map(|d| &d[..len_u])
            };
            #[cfg(feature = "deep-plc")]
            let lpcnet = self.silk_dec.lpcnet.as_mut();
            #[cfg(not(feature = "deep-plc"))]
            let lpcnet = None;
            celt_ok = self
                .celt_dec
                .decode_with_ec_dred(celt_data, Some(&mut dec), pcm, celt_frame_size, celt_accum, lpcnet)
                .is_some();
            self.range_final = self.celt_dec.rng();
        } else {
            // For hybrid -> SILK transitions, we let the CELT MDCT do a
            // fade-out by decoding a silence frame.
            if self.prev_mode == Some(OpusMode::Hybrid)
                && !(redundancy && celt_to_silk && self.prev_redundancy)
            {
                self.celt_dec.set_start_band(0);
                let silence = [0xFF, 0xFF];
                let _ = self
                    .celt_dec
                    .decode_with_ec(Some(&silence), None, pcm, f2_5, celt_accum);
            }
            self.range_final = dec.rng;
        }

        let window = modes::default_mode().window;
        let fs_i = self.sampling_rate;

        // 5 ms redundant frame for SILK->CELT.
        if redundancy && !celt_to_silk {
            self.celt_dec.reset_state();
            self.celt_dec.set_start_band(0);
            let _ = self.celt_dec.decode_with_ec(
                redundant_data,
                None,
                &mut redundant_audio[..f5 * ch],
                f5,
                false,
            );
            redundant_rng = self.celt_dec.rng();
            let at = ch * (frame_size - f2_5);
            smooth_fade_into_in1(
                &redundant_audio[ch * f2_5..ch * f5],
                &mut pcm[at..at + ch * f2_5],
                f2_5,
                ch,
                window,
                fs_i,
            );
        }
        // 5 ms redundant frame for CELT->SILK; ignore if the previous frame
        // did not use CELT (the first redundancy frame in a transition from
        // SILK may have been lost).
        if redundancy
            && celt_to_silk
            && (self.prev_mode != Some(OpusMode::SilkOnly) || self.prev_redundancy)
        {
            pcm[..ch * f2_5].copy_from_slice(&redundant_audio[..ch * f2_5]);
            smooth_fade_into_in2(
                &redundant_audio[ch * f2_5..ch * f5],
                &mut pcm[ch * f2_5..ch * f5],
                f2_5,
                ch,
                window,
                fs_i,
            );
        }
        if transition {
            if audiosize >= f5 {
                pcm[..ch * f2_5].copy_from_slice(&pcm_transition[..ch * f2_5]);
                smooth_fade_into_in2(
                    &pcm_transition[ch * f2_5..ch * f5],
                    &mut pcm[ch * f2_5..ch * f5],
                    f2_5,
                    ch,
                    window,
                    fs_i,
                );
            } else {
                // Not enough time to do a clean transition, but we do it
                // anyway. This will not preserve amplitude perfectly and may
                // introduce a bit of temporal aliasing.
                smooth_fade_into_in2(
                    &pcm_transition[..ch * f2_5],
                    &mut pcm[..ch * f2_5],
                    f2_5,
                    ch,
                    window,
                    fs_i,
                );
            }
        }

        if len <= 1 {
            self.range_final = 0;
        } else {
            self.range_final ^= redundant_rng;
        }

        self.prev_mode = Some(mode);
        self.prev_redundancy = redundancy && !celt_to_silk;

        if celt_ok {
            Ok(audiosize)
        } else {
            Err("CELT decoding failed")
        }
    }
}

/// One 5 ms frame at 48 kHz in stereo, the most a transition or redundant
/// frame holds.
const F5_MAX: usize = 240 * OPUS_MAX_CHANNELS;
/// One 10 ms frame at 48 kHz in stereo.
const F10_MAX: usize = 480 * OPUS_MAX_CHANNELS;

/// The frames of a packet (libopus `opus_packet_parse_impl`, not
/// self-delimited).
fn opus_packet_parse(
    data: &[u8],
) -> Result<FixedVec<&[u8], OPUS_MAX_PACKET_FRAMES>, &'static str> {
    const INVALID: &str = "Invalid packet";
    let size = |d: &[u8]| -> Result<(usize, usize), &'static str> {
        parse_frame_size(d).map_err(|_| INVALID)
    };
    let Some((&toc, mut rest)) = data.split_first() else {
        return Err(INVALID);
    };
    let framesize = frame_samples_from_toc(toc, 48000).ok_or(INVALID)?;
    let mut sizes = [0usize; OPUS_MAX_PACKET_FRAMES];
    let count;
    let last_size;
    match toc & 0x3 {
        // One frame.
        0 => {
            count = 1;
            last_size = rest.len();
        }
        // Two CBR frames.
        1 => {
            count = 2;
            if rest.len() & 1 != 0 {
                return Err(INVALID);
            }
            last_size = rest.len() / 2;
            sizes[0] = last_size;
        }
        // Two VBR frames.
        2 => {
            count = 2;
            let (s, bytes) = size(rest)?;
            rest = &rest[bytes..];
            if s > rest.len() {
                return Err(INVALID);
            }
            sizes[0] = s;
            last_size = rest.len() - s;
        }
        // Multiple CBR/VBR frames (from 0 to 120 ms).
        _ => {
            let Some((&ch, r)) = rest.split_first() else {
                return Err(INVALID);
            };
            rest = r;
            count = (ch & 0x3F) as usize;
            if count == 0 || framesize * count > 5760 {
                return Err(INVALID);
            }
            // Padding flag is bit 6.
            let mut len = rest.len() as i64;
            if ch & 0x40 != 0 {
                loop {
                    let Some((&p, r)) = rest.split_first() else {
                        return Err(INVALID);
                    };
                    rest = r;
                    len -= 1;
                    let tmp = if p == 255 { 254 } else { p as i64 };
                    len -= tmp;
                    if p != 255 {
                        break;
                    }
                }
            }
            if len < 0 {
                return Err(INVALID);
            }
            // The padding sits at the end of the packet.
            rest = &rest[..len as usize];
            // VBR flag is bit 7.
            if ch & 0x80 != 0 {
                let mut last = rest.len() as i64;
                for s in sizes.iter_mut().take(count - 1) {
                    let (n, bytes) = size(rest)?;
                    rest = &rest[bytes..];
                    if n > rest.len() {
                        return Err(INVALID);
                    }
                    *s = n;
                    last -= (bytes + n) as i64;
                }
                if last < 0 {
                    return Err(INVALID);
                }
                last_size = last as usize;
            } else {
                last_size = rest.len() / count;
                if last_size * count != rest.len() {
                    return Err(INVALID);
                }
                sizes[..count - 1].fill(last_size);
            }
        }
    }
    // The last frame's size (all frames' in CBR) is not coded and may exceed
    // 1275 bytes.
    if last_size > 1275 {
        return Err(INVALID);
    }
    sizes[count - 1] = last_size;
    let mut frames = FixedVec::new();
    for &s in &sizes[..count] {
        let (frame, r) = rest.split_at(s);
        frames.push(frame);
        rest = r;
    }
    Ok(frames)
}

/// libopus `smooth_fade` into `in1`: `in1 = w*in2 + (1-w)*in1` with the
/// squared CELT window `w` rising over `overlap` samples.
fn smooth_fade_into_in1(
    in2: &[f32],
    in1: &mut [f32],
    overlap: usize,
    channels: usize,
    window: &[f32],
    fs: i32,
) {
    let inc = (48000 / fs) as usize;
    for c in 0..channels {
        for i in 0..overlap {
            let w = window[i * inc] * window[i * inc];
            let k = i * channels + c;
            in1[k] = w * in2[k] + (1.0 - w) * in1[k];
        }
    }
}

/// libopus `smooth_fade` into `in2`: `in2 = w*in2 + (1-w)*in1`.
fn smooth_fade_into_in2(
    in1: &[f32],
    in2: &mut [f32],
    overlap: usize,
    channels: usize,
    window: &[f32],
    fs: i32,
) {
    let inc = (48000 / fs) as usize;
    for c in 0..channels {
        for i in 0..overlap {
            let w = window[i * inc] * window[i * inc];
            let k = i * channels + c;
            in2[k] = w * in2[k] + (1.0 - w) * in1[k];
        }
    }
}

fn frame_rate_from_params(sampling_rate: i32, frame_size: usize) -> Option<i32> {
    let frame_size = frame_size as i32;
    if frame_size <= 0 {
        return None;
    }
    // libopus `frame_size_select`: a frame is 2.5, 5, 10, 20, 40, 60, 80, 100
    // or 120 ms. Validating by duration (not by `sampling_rate % frame_size
    // == 0`) is what lets 60 ms through: its rate is 16.67 frames/s, so the
    // old divisibility test rejected it before encoding could start (issue
    // #35). Frames over 60 ms are coded as several shorter frames
    // (`encode_multiframe`).
    let fs = sampling_rate;
    let valid = frame_size == fs / 400
        || frame_size == fs / 200
        || frame_size == fs / 100
        || frame_size == fs / 50
        || frame_size == fs / 25
        || 50 * frame_size == 3 * fs
        || 50 * frame_size == 4 * fs
        || 50 * frame_size == 5 * fs
        || 50 * frame_size == 6 * fs;
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
                // libopus opus_packet_get_bandwidth: CELT has no mediumband.
                0 => Bandwidth::Narrowband,
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

    #[test]
    fn decoder_complexity_is_0_to_10() {
        let mut dec = OpusDecoder::new(48000, 1).unwrap();
        assert_eq!(dec.complexity(), 0);
        dec.set_complexity(10).unwrap();
        assert_eq!(dec.complexity(), 10);
        assert!(dec.set_complexity(11).is_err());
        assert!(dec.set_complexity(-1).is_err());
        assert_eq!(dec.complexity(), 10);
    }

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
