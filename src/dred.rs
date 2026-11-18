//! DRED in the Opus decoder and encoder (libopus `src/opus_decoder.c` and
//! `src/opus_encoder.c` under ENABLE_DRED): [`OpusDredDecoder`] finds and
//! reads the DRED extension of a packet into an [`OpusDred`],
//! [`OpusDecoder::dred_decode_float`] conceals a loss with the features it
//! carries, and [`OpusEncoder::set_dred_duration`] makes the encoder send
//! DRED.

pub use crate::dnn::dred::OpusDred;
use crate::dnn::dred::coding::compute_quantizer;
use crate::dnn::dred::{
    DRED_EXPERIMENTAL_BYTES, DRED_EXPERIMENTAL_VERSION, DRED_EXTENSION_ID, DRED_MAX_DATA_SIZE,
    DRED_MAX_FRAMES, DRED_MIN_BYTES, DRED_NUM_REDUNDANCY_FRAMES, DredEnc,
};
use crate::dnn::models::dred_rdovae_constants::DRED_NUM_FEATURES;
use crate::dnn::models::dred_rdovae_dec::RdovaeDec;
use crate::dnn::vec::c_floor;
use crate::dnn::weights::WeightError;
use crate::extensions::{
    ExtensionData, ExtensionIterator, parse_packet, samples_per_frame_48k, write_packet,
};
use crate::{FrameArgs, OpusDecoder, OpusEncoder, PcmInput, bitrate_to_bits, bits_to_bitrate};

/// libopus `OpusDREDDecoder`: the RDOVAE decoder model that turns the
/// latents of a DRED payload into features.
#[derive(Clone, Copy)]
pub struct OpusDredDecoder {
    model: RdovaeDec<'static>,
}

/// `dred_find_payload`: the DRED payload of `data` (without the
/// experimental header) and the position of its extension in the packet in
/// 2.5 ms units, or `None` when the packet has none.
fn dred_find_payload(data: &[u8]) -> Result<Option<(&[u8], i32)>, &'static str> {
    let parts = parse_packet(data)?;
    let frame_size = samples_per_frame_48k(parts.toc) as i32;
    let mut iter = ExtensionIterator::new(parts.padding, parts.count);
    while let Some(ext) = iter.find(DRED_EXTENSION_ID)? {
        let dred_frame_offset = ext.frame as i32 * frame_size / 120;
        // The experimental extension starts with 'D' and its version.
        if ext.data.len() > DRED_EXPERIMENTAL_BYTES
            && ext.data[0] == b'D'
            && ext.data[1] == DRED_EXPERIMENTAL_VERSION
        {
            return Ok(Some((
                &ext.data[DRED_EXPERIMENTAL_BYTES..],
                dred_frame_offset,
            )));
        }
    }
    Ok(None)
}

impl OpusDredDecoder {
    /// `opus_dred_decoder_create`: a decoder with the built-in model.
    pub fn new() -> Result<Self, WeightError> {
        Ok(OpusDredDecoder {
            model: crate::dnn::builtin::dred_rdovae_dec()?,
        })
    }

    /// A decoder with the given model (libopus `OPUS_SET_DNN_BLOB`).
    pub fn with_model(model: RdovaeDec<'static>) -> Self {
        OpusDredDecoder { model }
    }

    /// `opus_dred_parse`: reads the DRED extension of `data` into `dred`,
    /// enough of it for `max_dred_samples` samples at `sampling_rate`
    /// before the packet, and unless `defer_processing` computes the
    /// features ([`Self::process`]). Returns the number of samples DRED
    /// covers before the packet's end (0 without DRED) and `dred_end`, the
    /// samples at the end of that range that hold no DRED data.
    ///
    /// `dred.process_stage` is -1 when the packet carries no DRED data or
    /// is invalid.
    pub fn parse(
        &self,
        dred: &mut OpusDred,
        data: &[u8],
        max_dred_samples: i32,
        sampling_rate: i32,
        defer_processing: bool,
    ) -> Result<(i32, i32), &'static str> {
        dred.process_stage = -1;
        let Some((payload, dred_frame_offset)) = dred_find_payload(data)? else {
            return Ok((0, 0));
        };
        // libopus divides by the rate only here.
        if sampling_rate <= 0 {
            return Err("invalid sampling rate");
        }
        // Rounded up.
        let fs = sampling_rate as i64;
        let max_dred_features = ((100 * max_dred_samples as i64 + fs - 1) / fs)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        dred.ec_decode(payload, max_dred_features, dred_frame_offset);
        if !defer_processing {
            dred.process(&self.model);
        }
        let dred_end = (-(dred.dred_offset as i64) * fs / 400).max(0) as i32;
        let samples =
            (dred.nb_latents as i64 * fs / 25 - dred.dred_offset as i64 * fs / 400).max(0) as i32;
        Ok((samples, dred_end))
    }

    /// `opus_dred_process` in place: the features of the latents a
    /// deferred [`Self::parse`] read. An error unless `dred` holds parsed
    /// DRED data.
    pub fn process(&self, dred: &mut OpusDred) -> Result<(), &'static str> {
        if dred.process(&self.model) {
            Ok(())
        } else {
            Err(crate::error::BAD_ARG)
        }
    }
}

impl OpusDecoder {
    /// `opus_decoder_dred_decode_float`: conceals `frame_size` samples (a
    /// multiple of 2.5 ms) of a loss with the DRED data of `dred`, which a
    /// later packet carried; `dred_offset` is the position of the
    /// concealed audio's end in samples before that packet's start. An
    /// error unless `dred` holds processed DRED data. Returns the samples
    /// per channel written.
    pub fn dred_decode_float(
        &mut self,
        dred: &OpusDred,
        dred_offset: i32,
        output: &mut [f32],
        frame_size: usize,
    ) -> Result<usize, &'static str> {
        let f2_5 = self.sampling_rate as usize / 400;
        if frame_size == 0 || !frame_size.is_multiple_of(f2_5) {
            return Err(crate::error::BAD_ARG);
        }
        if dred.process_stage != 2 {
            return Err(crate::error::BAD_ARG);
        }
        self.queue_dred_features(dred, dred_offset, frame_size);
        self.decode(&[], frame_size, output)
    }

    /// `opus_decoder_dred_decode`: [`Self::dred_decode_float`] with the
    /// output rounded to 16 bits.
    pub fn dred_decode(
        &mut self,
        dred: &OpusDred,
        dred_offset: i32,
        output: &mut [i16],
        frame_size: usize,
    ) -> Result<usize, &'static str> {
        self.dred_decode_int(dred, dred_offset, output, frame_size)
    }

    /// `opus_decoder_dred_decode24`: [`Self::dred_decode_float`] with the
    /// output in 24-bit integers.
    pub fn dred_decode24(
        &mut self,
        dred: &OpusDred,
        dred_offset: i32,
        output: &mut [i32],
        frame_size: usize,
    ) -> Result<usize, &'static str> {
        self.dred_decode_int(dred, dred_offset, output, frame_size)
    }

    fn dred_decode_int<T: crate::decode_int::IntSample>(
        &mut self,
        dred: &OpusDred,
        dred_offset: i32,
        output: &mut [T],
        frame_size: usize,
    ) -> Result<usize, &'static str> {
        let f2_5 = self.sampling_rate as usize / 400;
        if frame_size == 0 || !frame_size.is_multiple_of(f2_5) {
            return Err(crate::error::BAD_ARG);
        }
        if dred.process_stage != 2 {
            return Err(crate::error::BAD_ARG);
        }
        let channels = self.channels;
        if frame_size
            .checked_mul(channels)
            .is_none_or(|n| output.len() < n)
        {
            return Err("Output buffer too small for packet");
        }
        self.queue_dred_features(dred, dred_offset, frame_size);
        // The concealment in pieces of 20 ms through a float buffer; a
        // remainder under 25 ms goes in one piece, so that the last piece
        // holds the 5 ms tail the next mode switch reads. The pieces
        // conceal what one call would.
        const MAX: usize = 1200 * 2;
        let mut buf = [0.0f32; MAX];
        let f20 = 8 * f2_5;
        let mut pos = 0;
        while pos < frame_size {
            let left = frame_size - pos;
            let n = if left < f20 + 2 * f2_5 { left } else { f20 };
            let out = &mut buf[..n * channels];
            self.decode(&[], n, out)?;
            for (o, &x) in output[pos * channels..(pos + n) * channels]
                .iter_mut()
                .zip(out.iter())
            {
                *o = T::from_float(x);
            }
            pos += n;
        }
        // One call of libopus conceals all of it.
        self.last_packet_duration = frame_size;
        Ok(frame_size)
    }

    /// The DRED block of libopus `opus_decode_native`: queues the DRED
    /// features that cover the next `frame_size` samples in the deep PLC.
    fn queue_dred_features(&mut self, dred: &OpusDred, dred_offset: i32, frame_size: usize) {
        let Some(lpcnet) = self.silk_dec.lpcnet.as_mut() else {
            return;
        };
        lpcnet.fec_clear();
        let f10 = self.sampling_rate / 100;
        // After an update (blend == 0) the PLC needs two more 10 ms frames.
        let init_frames = if lpcnet.blend == 0 { 2 } else { 0 };
        let features_per_frame = (frame_size as i32 / f10).max(1);
        let needed_feature_frames = init_frames + features_per_frame;
        // Floored: the 5 ms overlap makes up for the missing 0.5.
        let base = c_floor(
            ((dred_offset as f32 + (dred.dred_offset.wrapping_mul(f10) / 4) as f32) / f10 as f32)
                as f64,
        );
        let last = (4 * dred
            .nb_latents
            .clamp(0, DRED_NUM_REDUNDANCY_FRAMES as i32 / 2)
            - 1) as i64;
        for i in 0..needed_feature_frames {
            let feature_offset = (init_frames - i - 2) as i64 + base as i64;
            if feature_offset <= last && feature_offset >= 0 {
                let k = feature_offset as usize * DRED_NUM_FEATURES;
                lpcnet.fec_add(Some(&dred.fec_features[k..k + DRED_NUM_FEATURES]));
            } else if feature_offset >= 0 {
                lpcnet.fec_add(None);
            }
        }
    }
}

/// `dred_bits_table`: the approximate bits of a latent chunk per quantizer
/// level.
const DRED_BITS_TABLE: [f32; 16] = [
    73.2, 68.1, 62.5, 57.0, 51.5, 45.7, 39.9, 32.4, 26.4, 20.4, 16.3, 13.0, 9.3, 8.2, 7.2, 6.4,
];

/// `estimate_dred_bitrate`: the approximate bits of a DRED payload of
/// `duration` 10 ms frames, and the number of chunks that fit in
/// `target_bits`.
fn estimate_dred_bitrate(
    q0: i32,
    dq: i32,
    qmax: i32,
    duration: i32,
    target_bits: i32,
) -> (i32, i32) {
    // Signaling DRED costs 3 bytes, the experimental header 2 more.
    let mut bits = (8 * (3 + DRED_EXPERIMENTAL_BYTES)) as f32;
    // Approximation for the size of the initial state.
    bits += 50.0 + DRED_BITS_TABLE[q0 as usize];
    let dred_chunks = ((duration + 5) / 4).min(DRED_NUM_REDUNDANCY_FRAMES as i32 / 2);
    let mut target_chunks = 0;
    for i in 0..dred_chunks {
        let q = compute_quantizer(q0, dq, qmax, i);
        bits += DRED_BITS_TABLE[q as usize];
        if bits < target_bits as f32 {
            target_chunks = i + 1;
        }
    }
    (c_floor((0.5f32 + bits) as f64), target_chunks)
}

/// The DRED fields of libopus's `OpusEncoder`: the DRED encoder, the
/// duration, the quantizers and chunk target the bitrate sets, and the
/// voice activity of the last frames.
#[derive(Clone)]
pub(crate) struct EncoderDred {
    pub enc: DredEnc<'static>,
    /// `dred_duration`, in 10 ms frames.
    pub duration: i32,
    q0: i32,
    dq: i32,
    qmax: i32,
    target_chunks: i32,
    /// The voice activity at 2.5 ms resolution, newest first.
    activity_mem: [u8; 4 * DRED_MAX_FRAMES],
    /// `dred_bitrate_bps` of the current packet.
    pub bitrate_bps: i32,
    /// The frame reached the point where libopus adds DRED.
    pub reached: bool,
}

impl EncoderDred {
    /// An encoder without models, for [`Self::init`].
    pub const EMPTY: Self = EncoderDred {
        enc: DredEnc::EMPTY,
        duration: 0,
        q0: 0,
        dq: 0,
        qmax: 0,
        target_chunks: 0,
        activity_mem: [0; 4 * DRED_MAX_FRAMES],
        bitrate_bps: 0,
        reached: false,
    };

    /// `dred_encoder_init` with the built-in models, and no DRED, on an
    /// [`Self::EMPTY`] encoder.
    pub fn init(&mut self, fs: i32, channels: usize) {
        let model = crate::dnn::builtin::dred_rdovae_enc().expect("built-in DRED encoder models");
        let pitchdnn = crate::dnn::builtin::pitchdnn().expect("built-in DRED encoder models");
        self.enc.init(model, pitchdnn, fs, channels as i32);
    }

    /// `dred_encoder_reset` and the DRED fields an encoder reset clears,
    /// the duration among them.
    pub fn reset(&mut self) {
        self.enc.reset();
        self.duration = 0;
        self.q0 = 0;
        self.dq = 0;
        self.qmax = 0;
        self.target_chunks = 0;
        self.activity_mem = [0; 4 * DRED_MAX_FRAMES];
    }

    /// `compute_dred_bitrate`: the share of `bitrate_bps` that goes to
    /// DRED in a packet of `frame_size` samples, 0 when DRED cannot have at
    /// least two chunks. Sets the quantizers and the chunk target.
    pub fn compute_bitrate(
        &mut self,
        bitrate_bps: i32,
        frame_size: usize,
        fs: i32,
        inband_fec: bool,
        loss: i32,
    ) -> i32 {
        let (mut dred_frac, bitrate_offset) = if inband_fec {
            ((0.7f32).min(3.0 * loss as f32 / 100.0), 20000)
        } else if loss > 5 {
            ((0.8f32).min(0.55 + loss as f32 / 100.0), 12000)
        } else {
            (12.0 * loss as f32 / 100.0, 12000)
        };
        // Longer packets need less redundancy.
        dred_frac =
            dred_frac / (dred_frac + (1.0 - dred_frac) * (frame_size as f32 * 50.0) / fs as f32);
        // An approximate fit from a few experiments.
        let ilog = 32 - (bitrate_bps - bitrate_offset).max(1).leading_zeros() as i32;
        let q0 = (51 - 3 * ilog).clamp(4, 15);
        let dq = if bitrate_bps - bitrate_offset > 36000 {
            3
        } else {
            5
        };
        let qmax = 15;
        let target_dred_bitrate = (dred_frac * (bitrate_bps - bitrate_offset) as f32) as i32;
        let target_dred_bitrate = target_dred_bitrate.max(0);
        let (max_dred_bits, target_chunks) = if self.duration > 0 {
            let target_bits = bitrate_to_bits(target_dred_bitrate, fs, frame_size);
            estimate_dred_bitrate(q0, dq, qmax, self.duration, target_bits)
        } else {
            (0, 0)
        };
        let mut dred_bitrate =
            target_dred_bitrate.min(bits_to_bitrate(max_dred_bits, fs, frame_size));
        // Too few bits for DRED to be worth it.
        if target_chunks < 2 {
            dred_bitrate = 0;
        }
        self.q0 = q0;
        self.dq = dq;
        self.qmax = qmax;
        self.target_chunks = target_chunks;
        dred_bitrate
    }

    /// The DRED part of `opus_encode_frame_native` before SILK: the
    /// latents of the frame's filtered input `pcm` (interleaved,
    /// `frame_size` samples per channel), which the encoder delays by
    /// `extra_delay` samples, and its voice `activity` (-1 when SILK
    /// decides it later). Without DRED, the latents and the activity are
    /// cleared.
    pub fn frame(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        extra_delay: usize,
        fs: i32,
        activity: i32,
    ) {
        if self.duration > 0 {
            self.enc
                .compute_latents(pcm, frame_size, extra_delay as i32);
            let n = frame_size * 400 / fs as usize;
            self.activity_mem.copy_within(..4 * DRED_MAX_FRAMES - n, n);
            self.set_activity(n, activity);
        } else {
            self.enc.latents_buffer_fill = 0;
            self.activity_mem = [0; 4 * DRED_MAX_FRAMES];
        }
    }

    /// Sets the newest `n` activity flags (2.5 ms each).
    pub fn set_activity(&mut self, n: usize, activity: i32) {
        self.activity_mem[..n].fill(activity as u8);
    }

    /// The end of `opus_encode_frame_native`: `packet` (`len` bytes, one
    /// frame) gets the DRED extension when `with_dred` (the frame is the
    /// first of its packet that is not DTX), and under CBR is padded to
    /// `max_bytes` (libopus `orig_max_data_bytes`). Returns the new
    /// length.
    pub fn finish_packet(
        &mut self,
        packet: &mut [u8],
        len: usize,
        max_bytes: usize,
        use_vbr: bool,
        with_dred: bool,
    ) -> Result<usize, &'static str> {
        let parts = parse_packet(&packet[..len])?;
        if parts.count != 1 {
            return Err("DRED needs a packet of one frame");
        }
        let mut frame = [0u8; 1275];
        let flen = parts.frames[0].len();
        frame[..flen].copy_from_slice(parts.frames[0]);
        let toc = parts.toc;
        // The packet before padding: the ToC and the frame.
        let ret = 1 + flen as i32;
        let mut buf = [0u8; DRED_MAX_DATA_SIZE];
        let mut dred_bytes = 0;
        if self.duration > 0 && with_dred {
            let mut dred_chunks =
                ((self.duration + 5) / 4).min(DRED_NUM_REDUNDANCY_FRAMES as i32 / 2);
            if use_vbr {
                dred_chunks = dred_chunks.min(self.target_chunks);
            }
            // The room left, less 3 bytes for code 3, the padding length
            // and the extension ID, and the longer padding length.
            let mut left = (DRED_MAX_DATA_SIZE as i32).min(max_bytes as i32 - ret - 3);
            left -= (left + 1 + DRED_EXPERIMENTAL_BYTES as i32) / 255;
            if dred_chunks >= 1 && left >= (DRED_MIN_BYTES + DRED_EXPERIMENTAL_BYTES) as i32 {
                // The experimental extension type and version.
                buf[0] = b'D';
                buf[1] = DRED_EXPERIMENTAL_VERSION;
                let n = self.enc.encode_silk_frame(
                    &mut buf[DRED_EXPERIMENTAL_BYTES..],
                    dred_chunks,
                    left - DRED_EXPERIMENTAL_BYTES as i32,
                    self.q0,
                    self.dq,
                    self.qmax,
                    &self.activity_mem,
                );
                if n > 0 {
                    dred_bytes = n as usize + DRED_EXPERIMENTAL_BYTES;
                }
            }
        }
        if dred_bytes > 0 {
            let ext = ExtensionData {
                id: DRED_EXTENSION_ID,
                frame: 0,
                data: &buf[..dred_bytes],
            };
            write_packet(packet, toc, &[&frame[..flen]], &[ext], max_bytes, !use_vbr)
        } else if !use_vbr && len < max_bytes {
            write_packet(packet, toc, &[&frame[..flen]], &[], max_bytes, true)
        } else {
            Ok(len)
        }
    }
}

impl OpusEncoder {
    /// Sets the DRED duration in 10 ms frames, 0 (no DRED, the default) to
    /// 104 (libopus `OPUS_SET_DRED_DURATION`). With DRED, every packet
    /// carries the encoder's RDOVAE latents of up to that much audio before
    /// it in a packet extension, taking a share of the bitrate that grows
    /// with the packet loss percentage; libopus's decoder (or
    /// [`OpusDredDecoder`]) rebuilds lost audio from it.
    pub fn set_dred_duration(&mut self, frames: i32) -> Result<(), &'static str> {
        if !(0..=DRED_MAX_FRAMES as i32).contains(&frames) {
            return Err(crate::error::BAD_ARG);
        }
        self.dred.duration = frames;
        Ok(())
    }

    /// The DRED duration in 10 ms frames (libopus `OPUS_GET_DRED_DURATION`).
    pub fn dred_duration(&self) -> i32 {
        self.dred.duration
    }

    /// One frame (`opus_encode_frame_native`) with its DRED extension and
    /// CBR padding.
    pub(crate) fn encode_subframe(
        &mut self,
        input: PcmInput<'_>,
        frame_size: usize,
        output: &mut [u8],
        args: FrameArgs,
    ) -> Result<usize, &'static str> {
        self.dred.reached = false;
        let len = self.encode_frame(input, frame_size, output, args)?;
        if !core::mem::take(&mut self.dred.reached) {
            return Ok(len);
        }
        // `output` is libopus's `orig_max_data_bytes`: a frame of a
        // multiframe packet gets its share, a packet its CBR size or the
        // buffer.
        let max_bytes = output.len();
        self.dred
            .finish_packet(output, len, max_bytes, !self.use_cbr, args.first_frame)
    }

    /// The DRED latents of a frame from its filtered input (libopus's
    /// `pcm_buf` after the delay), or clearing them without DRED.
    pub(crate) fn dred_frame(&mut self, frame_size: usize, activity: i32) {
        let (ch, fs) = (self.channels, self.sampling_rate);
        if self.dred.duration == 0 {
            self.dred.frame(&[], frame_size, 0, fs, activity);
        } else {
            let prefix = self.delay_compensation * ch;
            let pcm = &self.buf_celt_pcm[prefix..prefix + frame_size * ch];
            self.dred
                .frame(pcm, frame_size, self.delay_compensation, fs, activity);
        }
    }
}

/// The repacketizing of libopus `encode_multiframe_packet` when its frames
/// carry extensions (DRED): the frames `frames` of the sub-packets `subs`
/// in one packet of at most `maxlen` bytes, padded to it with `pad`, with
/// the sub-packets' extensions moved to their frames. `None` when there
/// are no extensions.
pub(crate) fn repacketize_extensions(
    output: &mut [u8],
    subs: &[&[u8]],
    frames: &[&[u8]],
    maxlen: usize,
    pad: bool,
) -> Option<Result<usize, &'static str>> {
    let mut exts = [ExtensionData {
        id: 0,
        frame: 0,
        data: &[],
    }; 6];
    let mut n = 0;
    for (i, sub) in subs.iter().enumerate() {
        let Ok(parts) = parse_packet(sub) else {
            continue;
        };
        let mut iter = ExtensionIterator::new(parts.padding, parts.count);
        while let Ok(Some(ext)) = iter.next() {
            if n < exts.len() {
                exts[n] = ExtensionData {
                    frame: ext.frame + i,
                    ..ext
                };
                n += 1;
            }
        }
    }
    if n == 0 {
        return None;
    }
    Some(write_packet(
        output,
        subs[0][0],
        frames,
        &exts[..n],
        maxlen,
        pad,
    ))
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::Application;

    #[test]
    fn dred_duration_is_0_to_104_frames() {
        let mut enc = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
        assert_eq!(enc.dred_duration(), 0);
        enc.set_dred_duration(104).unwrap();
        assert!(enc.set_dred_duration(105).is_err() && enc.set_dred_duration(-1).is_err());
        assert_eq!(enc.dred_duration(), 104);
    }

    /// Packets with DRED parse, and DRED conceals a loss in their place.
    #[test]
    fn encode_parse_and_conceal() {
        let mut enc = OpusEncoder::new(16000, 1, Application::Voip).unwrap();
        enc.bitrate_bps = 32000;
        enc.packet_loss_perc = 20;
        enc.set_dred_duration(100).unwrap();
        let mut dec = OpusDecoder::new(16000, 1).unwrap();
        let dred_dec = OpusDredDecoder::new().unwrap();
        let mut dred = std::boxed::Box::<OpusDred>::default();
        let mut with_dred = 0;
        for k in 0..100 {
            let pcm: std::vec::Vec<f32> = (0..320)
                .map(|i| {
                    (((k * 320 + i) as f32) * 0.07).sin()
                        * 0.3
                        * (((k * 320 + i) as f32) * 0.0021).sin()
                })
                .collect();
            let mut pkt = [0u8; 1500];
            let n = enc.encode(&pcm, 320, &mut pkt).unwrap();
            let (samples, _) = dred_dec
                .parse(&mut dred, &pkt[..n], 16000, 16000, false)
                .unwrap();
            let mut out = [0f32; 320];
            if samples > 0 && k % 10 == 9 {
                with_dred += 1;
                assert_eq!(dred.process_stage, 2);
                assert_eq!(dec.dred_decode_float(&dred, 320, &mut out, 320), Ok(320));
            }
            assert_eq!(dec.decode(&pkt[..n], 320, &mut out), Ok(320));
        }
        assert!(with_dred >= 8, "{with_dred}");
    }
}
