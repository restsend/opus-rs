//! The DRED encoder (libopus `dnn/dred_encoder.c`): resamples the input to
//! 16 kHz, computes LPCNet features and RDOVAE latents every 20 ms, and
//! codes the newest initial state and latents into a DRED payload.

#[cfg(not(feature = "std"))]
use crate::compat::Math;

use super::super::builtin;
use super::super::lpcnet_enc::{
    LpcnetEncState, NB_TOTAL_FEATURES, lpcnet_compute_single_frame_features_float,
};
use super::super::models::dred_rdovae_constants::{
    DRED_LATENT_DIM, DRED_NUM_FEATURES, DRED_STATE_DIM,
};
use super::super::models::dred_rdovae_enc::RdovaeEnc;
use super::super::models::pitchdnn::PitchDnn;
use super::super::nnet::{Activation, compute_activation_inplace};
use super::super::vec::c_int;
use super::super::weights::WeightError;
use super::coding::compute_quantizer;
use super::laplace::ec_laplace_encode_p0;
use super::rdovae_enc::{RdovaeEncState, dred_rdovae_encode_dframe};
use super::stats::*;
use super::{
    DRED_DFRAME_SIZE, DRED_FRAME_SIZE, DRED_MAX_FRAMES, DRED_NUM_REDUNDANCY_FRAMES,
    DRED_SILK_ENCODER_DELAY,
};
use crate::range_coder::RangeCoder;

/// `RESAMPLING_ORDER`: the order of the resampling filters.
pub const RESAMPLING_ORDER: usize = 8;

/// `MAX_DOWNMIX_BUFFER` (without ENABLE_QEXT).
const MAX_DOWNMIX_BUFFER: usize = 960 * 2;

/// libopus `DREDEnc`. The fields from `input_buffer` on are those that
/// `dred_encoder_reset` clears. The model is always loaded.
#[derive(Clone)]
pub struct DredEnc<'a> {
    pub model: RdovaeEnc<'a>,
    pub lpcnet_enc_state: LpcnetEncState<'a>,
    pub rdovae_enc: RdovaeEncState,
    pub fs: i32,
    pub channels: i32,
    pub input_buffer: [f32; 2 * DRED_DFRAME_SIZE],
    pub input_buffer_fill: i32,
    pub dred_offset: i32,
    pub latent_offset: i32,
    pub last_extra_dred_offset: i32,
    pub latents_buffer: [f32; DRED_MAX_FRAMES * DRED_LATENT_DIM],
    pub latents_buffer_fill: i32,
    pub state_buffer: [f32; DRED_MAX_FRAMES * DRED_STATE_DIM],
    pub resample_mem: [f32; RESAMPLING_ORDER + 1],
}

impl<'a> DredEnc<'a> {
    /// `dred_encoder_init` with the given models: an encoder for `fs` Hz
    /// (8000, 12000, 16000, 24000 or 48000) and 1 or 2 channels.
    pub fn new(model: RdovaeEnc<'a>, pitchdnn: PitchDnn<'a>, fs: i32, channels: i32) -> Self {
        assert!(
            matches!(fs, 8000 | 12000 | 16000 | 24000 | 48000),
            "unsupported rate {fs}"
        );
        assert!(channels == 1 || channels == 2);
        let mut enc = DredEnc {
            model,
            lpcnet_enc_state: LpcnetEncState::new(pitchdnn),
            rdovae_enc: RdovaeEncState::default(),
            fs,
            channels,
            input_buffer: [0.0; 2 * DRED_DFRAME_SIZE],
            input_buffer_fill: 0,
            dred_offset: 0,
            latent_offset: 0,
            last_extra_dred_offset: 0,
            latents_buffer: [0.0; DRED_MAX_FRAMES * DRED_LATENT_DIM],
            latents_buffer_fill: 0,
            state_buffer: [0.0; DRED_MAX_FRAMES * DRED_STATE_DIM],
            resample_mem: [0.0; RESAMPLING_ORDER + 1],
        };
        enc.reset();
        enc
    }

    /// `dred_encoder_reset`.
    pub fn reset(&mut self) {
        self.input_buffer = [0.0; 2 * DRED_DFRAME_SIZE];
        self.dred_offset = 0;
        self.latent_offset = 0;
        self.last_extra_dred_offset = 0;
        self.latents_buffer = [0.0; DRED_MAX_FRAMES * DRED_LATENT_DIM];
        self.latents_buffer_fill = 0;
        self.state_buffer = [0.0; DRED_MAX_FRAMES * DRED_STATE_DIM];
        self.resample_mem = [0.0; RESAMPLING_ORDER + 1];
        self.input_buffer_fill = DRED_SILK_ENCODER_DELAY as i32;
        self.lpcnet_enc_state.reset();
        self.rdovae_enc = RdovaeEncState::default();
    }

    /// `dred_process_frame`: the latents and initial state of the 20 ms in
    /// `input_buffer`, pushed to the front of `latents_buffer` and
    /// `state_buffer`.
    fn process_frame(&mut self) {
        let mut feature_buffer = [0.0f32; 2 * NB_TOTAL_FEATURES];
        let mut input_buffer = [0.0f32; 2 * DRED_NUM_FEATURES];

        self.latents_buffer
            .copy_within(..(DRED_MAX_FRAMES - 1) * DRED_LATENT_DIM, DRED_LATENT_DIM);
        self.state_buffer
            .copy_within(..(DRED_MAX_FRAMES - 1) * DRED_STATE_DIM, DRED_STATE_DIM);

        let (f0, f1) = feature_buffer.split_at_mut(NB_TOTAL_FEATURES);
        let f0: &mut [f32; NB_TOTAL_FEATURES] = f0.try_into().unwrap();
        let f1: &mut [f32; NB_TOTAL_FEATURES] = f1.try_into().unwrap();
        lpcnet_compute_single_frame_features_float(
            &mut self.lpcnet_enc_state,
            &self.input_buffer,
            f0,
        );
        lpcnet_compute_single_frame_features_float(
            &mut self.lpcnet_enc_state,
            &self.input_buffer[DRED_FRAME_SIZE..],
            f1,
        );

        input_buffer[..DRED_NUM_FEATURES].copy_from_slice(&f0[..DRED_NUM_FEATURES]);
        input_buffer[DRED_NUM_FEATURES..].copy_from_slice(&f1[..DRED_NUM_FEATURES]);

        dred_rdovae_encode_dframe(
            &mut self.rdovae_enc,
            &self.model,
            &mut self.latents_buffer,
            &mut self.state_buffer,
            &input_buffer,
        );
        self.latents_buffer_fill =
            (self.latents_buffer_fill + 1).min(DRED_NUM_REDUNDANCY_FRAMES as i32);
    }

    /// `dred_convert_to_16k`: downmixes `in_len` samples per channel of `pcm`
    /// to mono int16 range and resamples them to `out_len` samples at
    /// 16 kHz.
    fn convert_to_16k(&mut self, pcm: &[f32], in_len: usize, out: &mut [f32], out_len: usize) {
        let mut downmix = [0.0f32; MAX_DOWNMIX_BUFFER];
        assert!(self.channels as usize * in_len <= MAX_DOWNMIX_BUFFER);
        debug_assert_eq!(in_len as i32 * 16000, out_len as i32 * self.fs);
        let up = match self.fs {
            8000 => 2,
            12000 => 4,
            16000 => 1,
            24000 => 2,
            _ => 1,
        };
        assert!(up * in_len <= MAX_DOWNMIX_BUFFER);
        if self.channels == 1 {
            for i in 0..in_len {
                downmix[up * i] = float2int16(up as f32 * pcm[i]) as f32 + 1e-30;
            }
        } else {
            for i in 0..in_len {
                let x = (0.5 * up as f64 * (pcm[2 * i] + pcm[2 * i + 1]) as f64) as f32;
                downmix[up * i] = float2int16(x) as f32 + 1e-30;
            }
        }
        let n = up * in_len;
        match self.fs {
            16000 => out[..out_len].copy_from_slice(&downmix[..out_len]),
            48000 | 24000 => {
                // ellip(7, .2, 70, 7750/24000)
                const B: [f32; 8] = [
                    0.005873358047,
                    0.012980854831,
                    0.014531340042,
                    0.014531340042,
                    0.012980854831,
                    0.005873358047,
                    0.004523418224,
                    0.0,
                ];
                const A: [f32; 8] = [
                    -3.878718597768,
                    7.748834257468,
                    -9.653651699533,
                    8.007342726666,
                    -4.379450178552,
                    1.463182111810,
                    -0.231720677804,
                    0.0,
                ];
                filter_df2t_inplace(
                    &mut downmix[..n],
                    0.004523418224,
                    &B,
                    &A,
                    &mut self.resample_mem,
                );
                for i in 0..out_len {
                    out[i] = downmix[3 * i];
                }
            }
            12000 => {
                // ellip(7, .2, 70, 5800/24000)
                const B: [f32; 8] = [
                    -0.001017101081,
                    0.003673127243,
                    0.001009165267,
                    0.001009165267,
                    0.003673127243,
                    -0.001017101081,
                    0.002033596776,
                    0.0,
                ];
                const A: [f32; 8] = [
                    -4.930414411612,
                    11.291643096504,
                    -15.322037343815,
                    13.216403930898,
                    -7.220409219553,
                    2.310550142771,
                    -0.334338618782,
                    0.0,
                ];
                filter_df2t_inplace(
                    &mut downmix[..n],
                    0.002033596776,
                    &B,
                    &A,
                    &mut self.resample_mem,
                );
                for i in 0..out_len {
                    out[i] = downmix[3 * i];
                }
            }
            _ => {
                // 8000 Hz: ellip(7, .2, 70, 3900/8000)
                const B: [f32; 8] = [
                    0.081670120929,
                    0.180401598565,
                    0.259391051971,
                    0.259391051971,
                    0.180401598565,
                    0.081670120929,
                    0.020109185709,
                    0.0,
                ];
                const A: [f32; 8] = [
                    -1.393651933659,
                    2.609789872676,
                    -2.403541968806,
                    2.056814957331,
                    -1.148908574570,
                    0.473001413788,
                    -0.110359852412,
                    0.0,
                ];
                filter_df2t(
                    &downmix[..out_len],
                    &mut out[..out_len],
                    0.020109185709,
                    &B,
                    &A,
                    &mut self.resample_mem,
                );
            }
        }
    }

    /// `dred_compute_latents`: feeds `frame_size` samples per channel of
    /// interleaved `pcm` (in the ±1 range) into the encoder. `extra_delay`
    /// is the encoder's delay in samples at `fs`; it sets `dred_offset`.
    ///
    /// Between 20 ms chunks, libopus advances `pcm` by the chunk's sample
    /// count per channel, also for stereo; this port does the same. `pcm`
    /// needs `frame_size * channels` values.
    pub fn compute_latents(&mut self, pcm: &[f32], frame_size: usize, extra_delay: i32) {
        let mut frame_size16k = frame_size as i32 * 16000 / self.fs;
        let curr_offset16k = 40 + extra_delay * 16000 / self.fs - self.input_buffer_fill;
        self.dred_offset = c_int((((curr_offset16k as f32 + 20.0) / 40.0) as f64).floor());
        self.latent_offset = 0;
        let mut pos = 0usize;
        while frame_size16k > 0 {
            let process_size16k = (2 * DRED_FRAME_SIZE as i32).min(frame_size16k);
            let process_size = process_size16k * self.fs / 16000;
            let fill = self.input_buffer_fill as usize;
            let mut chunk = [0.0f32; 2 * DRED_FRAME_SIZE];
            self.convert_to_16k(
                &pcm[pos..],
                process_size as usize,
                &mut chunk,
                process_size16k as usize,
            );
            self.input_buffer[fill..fill + process_size16k as usize]
                .copy_from_slice(&chunk[..process_size16k as usize]);
            self.input_buffer_fill += process_size16k;
            if self.input_buffer_fill >= 2 * DRED_FRAME_SIZE as i32 {
                self.process_frame();
                self.input_buffer_fill -= 2 * DRED_FRAME_SIZE as i32;
                let fill = self.input_buffer_fill as usize;
                self.input_buffer
                    .copy_within(2 * DRED_FRAME_SIZE..2 * DRED_FRAME_SIZE + fill, 0);
                // 15 ms (6*2.5 ms) is the ideal offset: the vocoder look-ahead.
                if self.dred_offset < 6 {
                    self.dred_offset += 8;
                } else {
                    self.latent_offset += 1;
                }
            }
            pos += process_size as usize;
            frame_size16k -= process_size16k;
        }
    }

    /// `dred_encode_silk_frame`: codes the newest initial state and up to
    /// `max_chunks` latent pairs into `buf` (at most `max_bytes` bytes) with
    /// quantizer levels from `q0`, `dq` and `qmax`. `activity_mem` holds
    /// the voice activity at 2.5 ms resolution, newest first; bytes past
    /// its end read as inactive. Returns the payload length, or 0 when
    /// there is nothing worth sending.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_silk_frame(
        &mut self,
        buf: &mut [u8],
        max_chunks: i32,
        max_bytes: i32,
        q0: i32,
        dq: i32,
        qmax: i32,
        activity_mem: &[u8],
    ) -> i32 {
        let mut prev_active = false;
        let mut extra_dred_offset = 0;
        let mut dred_encoded = 0;
        let mut delayed_dred = false;

        let mut latent_offset = self.latent_offset;
        // Delay new DRED data just out of silence: the main payload covers
        // that frame.
        if activity_mem.first().is_some_and(|&a| a != 0) && self.last_extra_dred_offset > 0 {
            latent_offset = self.last_extra_dred_offset;
            delayed_dred = true;
            self.last_extra_dred_offset = 0;
        }
        while latent_offset < self.latents_buffer_fill
            && !dred_voice_active(activity_mem, latent_offset)
        {
            latent_offset += 1;
            extra_dred_offset += 1;
        }
        if !delayed_dred {
            self.last_extra_dred_offset = extra_dred_offset;
        }

        let mut ec = RangeCoder::new_encoder(max_bytes.max(0) as u32);
        ec.enc_uint(q0 as u32, 16);
        ec.enc_uint(dq as u32, 8);
        let total_offset = 16 - (self.dred_offset - extra_dred_offset * 8);
        debug_assert!(total_offset >= 0);
        if total_offset > 31 {
            ec.enc_uint(1, 2);
            ec.enc_uint((total_offset >> 5) as u32, 256);
            ec.enc_uint((total_offset & 31) as u32, 32);
        } else {
            ec.enc_uint(0, 2);
            ec.enc_uint(total_offset as u32, 32);
        }
        debug_assert!(qmax >= q0);
        if q0 < 14 && dq > 0 {
            let nvals = 15 - (q0 + 1);
            let (fl, fh) = if qmax >= 15 {
                (0, nvals)
            } else {
                (nvals + qmax - (q0 + 1), nvals + qmax - q0)
            };
            ec.encode(fl as u32, fh as u32, 2 * nvals as u32);
        }
        let so = q0 as usize * DRED_STATE_DIM;
        let lo = latent_offset as usize * DRED_STATE_DIM;
        dred_encode_latents(
            &mut ec,
            &self.state_buffer[lo..lo + DRED_STATE_DIM],
            &DRED_STATE_QUANT_SCALES_Q8[so..],
            &DRED_STATE_DEAD_ZONE_Q8[so..],
            &DRED_STATE_R_Q8[so..],
            &DRED_STATE_P0_Q8[so..],
        );
        if ec.tell() > 8 * max_bytes {
            return 0;
        }
        let mut ec_bak = ec.clone();
        let n = (2 * max_chunks).min(self.latents_buffer_fill - latent_offset - 1);
        let mut i = 0;
        while i < n {
            let q_level = compute_quantizer(q0, dq, qmax, i / 2);
            let off = q_level as usize * DRED_LATENT_DIM;
            let lo = (i + latent_offset) as usize * DRED_LATENT_DIM;
            dred_encode_latents(
                &mut ec,
                &self.latents_buffer[lo..lo + DRED_LATENT_DIM],
                &DRED_LATENT_QUANT_SCALES_Q8[off..],
                &DRED_LATENT_DEAD_ZONE_Q8[off..],
                &DRED_LATENT_R_Q8[off..],
                &DRED_LATENT_P0_Q8[off..],
            );
            if ec.tell() > 8 * max_bytes {
                // Without one complete chunk, no DRED at all.
                if i == 0 {
                    return 0;
                }
                break;
            }
            let active = dred_voice_active(activity_mem, i + latent_offset);
            if active || prev_active {
                ec_bak = ec.clone();
                dred_encoded = i + 2;
            }
            prev_active = active;
            i += 2;
        }
        // No empty DRED payloads.
        if dred_encoded == 0 || (dred_encoded <= 2 && extra_dred_offset != 0) {
            return 0;
        }
        let mut ec = ec_bak;
        let fill = (ec.tell() + 7) / 8;
        ec.shrink(fill as u32);
        ec.done();
        buf[..fill as usize].copy_from_slice(&ec.buf[..fill as usize]);
        fill
    }
}

impl DredEnc<'static> {
    /// `dred_encoder_init` with the built-in RDOVAE and PitchDNN models.
    pub fn builtin(fs: i32, channels: i32) -> Result<Self, WeightError> {
        Ok(DredEnc::new(
            builtin::dred_rdovae_enc()?,
            builtin::pitchdnn()?,
            fs,
            channels,
        ))
    }
}

/// libopus `FLOAT2INT16` (float_cast.h): scaled to the int16 range,
/// saturated (NaN gives -32768) and rounded to nearest, ties to even.
fn float2int16(x: f32) -> i16 {
    let x = x * 32768.0;
    let x = if x > -32768.0 { x } else { -32768.0 };
    let x = if x < 32767.0 { x } else { 32767.0 };
    // Adding and removing 1.5 * 2^23 rounds |x| < 2^22 to an integer,
    // ties to even.
    ((x + 12_582_912.0) - 12_582_912.0) as i16
}

/// `filter_df2t`: a direct form II transposed IIR filter of order
/// `b.len()` with leading coefficient `b0` and memory `mem`
/// (`b.len() + 1` values).
pub fn filter_df2t(input: &[f32], out: &mut [f32], b0: f32, b: &[f32], a: &[f32], mem: &mut [f32]) {
    for (o, &xi) in out.iter_mut().zip(input) {
        *o = df2t_step(xi, b0, b, a, mem);
    }
}

/// `filter_df2t` with the output written over the input.
fn filter_df2t_inplace(x: &mut [f32], b0: f32, b: &[f32], a: &[f32], mem: &mut [f32]) {
    for v in x.iter_mut() {
        *v = df2t_step(*v, b0, b, a, mem);
    }
}

#[inline(always)]
fn df2t_step(xi: f32, b0: f32, b: &[f32], a: &[f32], mem: &mut [f32]) -> f32 {
    let yi = xi * b0 + mem[0];
    let nyi = -yi;
    for j in 0..b.len() {
        mem[j] = mem[j + 1] + b[j] * xi + a[j] * nyi;
    }
    yi
}

/// `dred_encode_latents`: quantizes `x` with a dead zone and codes it with
/// the Laplace coder.
fn dred_encode_latents(
    ec: &mut RangeCoder,
    x: &[f32],
    scale: &[u8],
    dzone: &[u8],
    r: &[u8],
    p0: &[u8],
) {
    const MAX_DIM: usize = if DRED_LATENT_DIM > DRED_STATE_DIM {
        DRED_LATENT_DIM
    } else {
        DRED_STATE_DIM
    };
    let dim = x.len();
    let mut xq = [0.0f32; MAX_DIM];
    let mut delta = [0.0f32; MAX_DIM];
    let mut deadzone = [0.0f32; MAX_DIM];
    let eps = 0.1f32;
    for i in 0..dim {
        delta[i] = dzone[i] as f32 * (1.0 / 256.0);
        xq[i] = x[i] * scale[i] as f32 * (1.0 / 256.0);
        deadzone[i] = xq[i] / (delta[i] + eps);
    }
    compute_activation_inplace(&mut deadzone[..dim], Activation::Tanh);
    for i in 0..dim {
        xq[i] -= delta[i] * deadzone[i];
        let q = c_int(((0.5f32 + xq[i]) as f64).floor());
        // Values the tables make impossible are not coded.
        if r[i] != 0 && p0[i] != 255 {
            ec_laplace_encode_p0(ec, q, (p0[i] as u16) << 7, (r[i] as u16) << 7);
        }
    }
}

/// `dred_voice_active`: whether any of the 16 activity flags (40 ms) from
/// latent `offset` is 1. Bytes past the end of `activity_mem` read as 0.
fn dred_voice_active(activity_mem: &[u8], offset: i32) -> bool {
    let start = 8 * offset as usize;
    (start..start + 16).any(|i| activity_mem.get(i) == Some(&1))
}
