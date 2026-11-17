//! DRED in the Opus decoder (libopus `src/opus_decoder.c` under
//! ENABLE_DRED): [`OpusDredDecoder`] finds and reads the DRED extension of
//! a packet into an [`OpusDred`], and [`OpusDecoder::dred_decode_float`]
//! conceals a loss with the features it carries.

pub use crate::dnn::dred::OpusDred;
use crate::dnn::dred::{DRED_EXPERIMENTAL_BYTES, DRED_EXPERIMENTAL_VERSION, DRED_EXTENSION_ID, DRED_NUM_REDUNDANCY_FRAMES};
use crate::dnn::models::dred_rdovae_constants::DRED_NUM_FEATURES;
use crate::dnn::models::dred_rdovae_dec::RdovaeDec;
use crate::dnn::vec::c_floor;
use crate::dnn::weights::WeightError;
use crate::extensions::{ExtensionIterator, parse_packet, samples_per_frame_48k};
use crate::OpusDecoder;

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
        if ext.data.len() > DRED_EXPERIMENTAL_BYTES && ext.data[0] == b'D' && ext.data[1] == DRED_EXPERIMENTAL_VERSION {
            return Ok(Some((&ext.data[DRED_EXPERIMENTAL_BYTES..], dred_frame_offset)));
        }
    }
    Ok(None)
}

impl OpusDredDecoder {
    /// `opus_dred_decoder_create`: a decoder with the built-in model.
    pub fn new() -> Result<Self, WeightError> {
        Ok(OpusDredDecoder { model: crate::dnn::builtin::dred_rdovae_dec()? })
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
        if sampling_rate <= 0 {
            return Err("invalid sampling rate");
        }
        dred.process_stage = -1;
        let Some((payload, dred_frame_offset)) = dred_find_payload(data)? else {
            return Ok((0, 0));
        };
        let offset = (100 * max_dred_samples as i64 / sampling_rate as i64) as i32;
        let min_feature_frames = (2i32.saturating_add(offset)).min(2 * DRED_NUM_REDUNDANCY_FRAMES as i32);
        dred.ec_decode(payload, min_feature_frames, dred_frame_offset);
        if !defer_processing {
            dred.process(&self.model);
        }
        let fs = sampling_rate as i64;
        let dred_end = (-(dred.dred_offset as i64) * fs / 400).max(0) as i32;
        let samples = (dred.nb_latents as i64 * fs / 25 - dred.dred_offset as i64 * fs / 400).max(0) as i32;
        Ok((samples, dred_end))
    }

    /// `opus_dred_process` in place: the features of the latents a
    /// deferred [`Self::parse`] read. An error unless `dred` holds parsed
    /// DRED data.
    pub fn process(&self, dred: &mut OpusDred) -> Result<(), &'static str> {
        if dred.process(&self.model) { Ok(()) } else { Err("no parsed DRED data") }
    }
}

impl OpusDecoder {
    /// `opus_decoder_dred_decode_float`: conceals `frame_size` samples (a
    /// multiple of 2.5 ms) of a loss with the DRED data of `dred`, which a
    /// later packet carried; `dred_offset` is the position of the
    /// concealed audio's end in samples before that packet's start. With
    /// unprocessed `dred` this is ordinary concealment. Returns the samples
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
            return Err("Lost frame: frame_size must be a multiple of 2.5 ms");
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
        let f2_5 = self.sampling_rate as usize / 400;
        if frame_size == 0 || !frame_size.is_multiple_of(f2_5) {
            return Err("Lost frame: frame_size must be a multiple of 2.5 ms");
        }
        let channels = self.channels;
        if frame_size.checked_mul(channels).is_none_or(|n| output.len() < n) {
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
            for (o, &x) in output[pos * channels..(pos + n) * channels].iter_mut().zip(out.iter()) {
                *o = crate::float2int16(x);
            }
            pos += n;
        }
        Ok(frame_size)
    }

    /// The DRED block of libopus `opus_decode_native`: queues the DRED
    /// features that cover the next `frame_size` samples in the deep PLC.
    fn queue_dred_features(&mut self, dred: &OpusDred, dred_offset: i32, frame_size: usize) {
        if dred.process_stage != 2 {
            return;
        }
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
        let base = c_floor(((dred_offset as f32 + (dred.dred_offset.wrapping_mul(f10) / 4) as f32) / f10 as f32) as f64);
        let last = (4 * dred.nb_latents.clamp(0, DRED_NUM_REDUNDANCY_FRAMES as i32 / 2) - 1) as i64;
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
