//! The DRED payload reader (libopus `dnn/dred_decoder.c`).

use super::super::models::dred_rdovae_constants::{
    DRED_LATENT_DIM, DRED_NUM_FEATURES, DRED_STATE_DIM,
};
use super::super::models::dred_rdovae_dec::RdovaeDec;
use super::DRED_NUM_REDUNDANCY_FRAMES;
use super::coding::compute_quantizer;
use super::laplace::ec_laplace_decode_p0;
use super::rdovae_dec::dred_rdovae_decode_all;
use super::stats::*;
use crate::range_coder::RangeCoder;

/// The values of `OpusDRED::fec_features`.
pub const DRED_FEC_FEATURES_SIZE: usize = 2 * DRED_NUM_REDUNDANCY_FRAMES * DRED_NUM_FEATURES;
/// The values of `OpusDRED::latents`: per latent vector `DRED_LATENT_DIM`
/// values and its quantizer level.
pub const DRED_LATENTS_SIZE: usize = (DRED_NUM_REDUNDANCY_FRAMES / 2) * (DRED_LATENT_DIM + 1);

/// libopus `OpusDRED`: the decoded DRED data of one packet.
///
/// `process_stage` is -1 before parsing, 1 after [`OpusDred::ec_decode`]
/// (latents and state read) and 2 after [`OpusDred::process`] (features
/// computed).
#[derive(Clone, Debug)]
pub struct OpusDred {
    pub fec_features: [f32; DRED_FEC_FEATURES_SIZE],
    pub state: [f32; DRED_STATE_DIM],
    pub latents: [f32; DRED_LATENTS_SIZE],
    pub nb_latents: i32,
    pub process_stage: i32,
    pub dred_offset: i32,
}

impl Default for OpusDred {
    /// All zero.
    fn default() -> Self {
        OpusDred {
            fec_features: [0.0; DRED_FEC_FEATURES_SIZE],
            state: [0.0; DRED_STATE_DIM],
            latents: [0.0; DRED_LATENTS_SIZE],
            nb_latents: 0,
            process_stage: 0,
            dred_offset: 0,
        }
    }
}

/// `dred_decode_latents`: reads `x.len()` quantized values, bounds them to
/// +/-1024 (far beyond what an encoder produces) and scales them back.
pub fn dred_decode_latents(dec: &mut RangeCoder, x: &mut [f32], scale: &[u8], r: &[u8], p0: &[u8]) {
    for (i, v) in x.iter_mut().enumerate() {
        let q = if r[i] == 0 || p0[i] == 255 {
            0
        } else {
            ec_laplace_decode_p0(dec, (p0[i] as u16) << 7, (r[i] as u16) << 7)
        }
        .clamp(-1024, 1024);
        let s = if scale[i] == 0 { 1 } else { scale[i] };
        *v = q as f32 * 256.0 / s as f32;
    }
}

impl OpusDred {
    /// `dred_ec_decode`: reads the initial state and the latents of the
    /// DRED payload `bytes`, enough for `max_dred_features` 10 ms frames
    /// before the DRED offset. `dred_frame_offset` is the position of the
    /// DRED extension in the packet, in 2.5 ms units. Latent slots past
    /// the decoded ones keep their values. Returns the number of latent
    /// vectors read.
    pub fn ec_decode(
        &mut self,
        bytes: &[u8],
        max_dred_features: i32,
        dred_frame_offset: i32,
    ) -> i32 {
        let num_bytes = bytes.len() as i32;
        let mut ec = RangeCoder::new_decoder(bytes);
        let q0 = ec.dec_uint(16) as i32;
        let dq = ec.dec_uint(8) as i32;
        let extra_offset = if ec.dec_uint(2) != 0 {
            32 * ec.dec_uint(256) as i32
        } else {
            0
        };
        // The total offset, including the DRED position in the packet.
        self.dred_offset = 16 - ec.dec_uint(32) as i32 - extra_offset + dred_frame_offset;
        // The 10 ms feature frames that cover the requested duration plus
        // the offset (in 2.5 ms units).
        let min_feature_frames =
            ((4 * max_dred_features as i64 + self.dred_offset as i64 + 3) / 4).max(0);
        let mut qmax = 15;
        if q0 < 14 && dq > 0 {
            // Half the probability says qmax == 15, the other half is
            // uniform over the levels from q0 + 1 to 14.
            let nvals = 15 - (q0 + 1);
            let ft = (2 * nvals) as u32;
            let s = ec.decode(ft) as i32;
            if s >= nvals {
                qmax = q0 + (s - nvals) + 1;
                ec.update(s as u32, s as u32 + 1, ft);
            } else {
                ec.update(0, nvals as u32, ft);
            }
        }
        let so = q0 as usize * DRED_STATE_DIM;
        dred_decode_latents(
            &mut ec,
            &mut self.state,
            &DRED_STATE_QUANT_SCALES_Q8[so..],
            &DRED_STATE_R_Q8[so..],
            &DRED_STATE_P0_Q8[so..],
        );

        // Newest to oldest in the payload and in `latents`.
        let n = (DRED_NUM_REDUNDANCY_FRAMES as i64).min((min_feature_frames + 1) / 2) as i32;
        let mut i = 0;
        while i < n {
            // libopus FIXME: a last frame of fewer than 8 bits is missed.
            if 8 * num_bytes - ec.tell() <= 7 {
                break;
            }
            let q_level = compute_quantizer(q0, dq, qmax, i / 2);
            let off = q_level as usize * DRED_LATENT_DIM;
            let lo = (i / 2) as usize * (DRED_LATENT_DIM + 1);
            dred_decode_latents(
                &mut ec,
                &mut self.latents[lo..lo + DRED_LATENT_DIM],
                &DRED_LATENT_QUANT_SCALES_Q8[off..],
                &DRED_LATENT_R_Q8[off..],
                &DRED_LATENT_P0_Q8[off..],
            );
            self.latents[lo + DRED_LATENT_DIM] = (q_level as f64 * 0.125 - 1.0) as f32;
            i += 2;
        }
        self.process_stage = 1;
        self.nb_latents = i / 2;
        i / 2
    }

    /// The decoding step of libopus `opus_dred_process` on `self`: turns
    /// the latents into features with the RDOVAE decoder (stage 1 to 2).
    /// Returns false (and does nothing) unless `process_stage` is 1 or 2.
    pub fn process(&mut self, model: &RdovaeDec<'_>) -> bool {
        match self.process_stage {
            1 => {
                dred_rdovae_decode_all(
                    model,
                    &mut self.fec_features,
                    &self.state,
                    &self.latents,
                    self.nb_latents as usize,
                );
                self.process_stage = 2;
                true
            }
            2 => true,
            _ => false,
        }
    }
}
