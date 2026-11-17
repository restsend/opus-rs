//! Deep packet loss concealment (libopus `dnn/lpcnet_plc.c`): the PLC
//! model predicts the features of lost 10 ms frames from the features of
//! the decoded signal (or from DRED), and FARGAN synthesizes them.

use super::builtin;
use super::fargan::{FARGAN_CONT_SAMPLES, FarganState, fargan_cont, fargan_synthesize_int};
use super::freq::{FRAME_SIZE, NB_BANDS, burg_cepstral_analysis};
use super::lpcnet_enc::{
    LpcnetEncState, NB_FEATURES, NB_TOTAL_FEATURES, lpcnet_compute_single_frame_features_float,
};
use super::models::fargan::Fargan;
use super::models::pitchdnn::PitchDnn;
use super::models::plc::{
    PLC_DENSE_IN_OUT_SIZE, PLC_GRU1_STATE_SIZE, PLC_GRU2_STATE_SIZE, PlcModel,
};
use super::nnet::{Activation, compute_generic_dense, compute_generic_gru};
use super::weights::{WeightError, WeightList};

pub const PLC_MAX_FEC: usize = 104;
pub const CONT_VECTORS: usize = 5;
pub const PLC_BUF_SIZE: usize = (CONT_VECTORS + 10) * FRAME_SIZE;
const PLC_IN_SIZE: usize = 2 * NB_BANDS + NB_FEATURES + 1;

/// libopus `PLCNetState`.
#[derive(Clone, Copy)]
pub struct PlcNetState {
    pub gru1_state: [f32; PLC_GRU1_STATE_SIZE],
    pub gru2_state: [f32; PLC_GRU2_STATE_SIZE],
}

impl PlcNetState {
    const ZERO: PlcNetState = PlcNetState {
        gru1_state: [0.0; PLC_GRU1_STATE_SIZE],
        gru2_state: [0.0; PLC_GRU2_STATE_SIZE],
    };
}

/// The three models of deep PLC.
#[derive(Clone, Copy, Debug)]
pub struct PlcModels<'a> {
    pub plc: PlcModel<'a>,
    pub fargan: Fargan<'a>,
    pub pitchdnn: PitchDnn<'a>,
}

impl<'a> PlcModels<'a> {
    /// The models of a blob (libopus `lpcnet_plc_load_model`: the PLC
    /// model, PitchDNN and FARGAN from the same blob).
    pub fn from_blob(list: &WeightList<'a>) -> Result<Self, WeightError> {
        Ok(PlcModels {
            plc: PlcModel::init(list)?,
            pitchdnn: PitchDnn::init(list)?,
            fargan: Fargan::init(list)?,
        })
    }
}

impl PlcModels<'static> {
    /// The built-in models.
    pub fn builtin() -> Result<Self, WeightError> {
        Ok(PlcModels {
            plc: builtin::plc()?,
            fargan: builtin::fargan()?,
            pitchdnn: builtin::pitchdnn()?,
        })
    }
}

/// libopus `LPCNetPLCState`.
#[derive(Clone)]
pub struct LpcnetPlcState<'a> {
    pub model: PlcModel<'a>,
    pub fargan: FarganState<'a>,
    pub enc: LpcnetEncState<'a>,
    pub fec: [[f32; NB_FEATURES]; PLC_MAX_FEC],
    pub analysis_gap: i32,
    pub fec_read_pos: usize,
    pub fec_fill_pos: usize,
    pub fec_skip: i32,
    pub analysis_pos: i32,
    pub predict_pos: i32,
    pub pcm: [f32; PLC_BUF_SIZE],
    pub blend: i32,
    pub features: [f32; NB_TOTAL_FEATURES],
    pub cont_features: [f32; CONT_VECTORS * NB_FEATURES],
    pub loss_count: i32,
    pub plc_net: PlcNetState,
    pub plc_bak: [PlcNetState; 2],
}

impl<'a> LpcnetPlcState<'a> {
    /// `lpcnet_plc_init` with the given models.
    pub fn new(models: PlcModels<'a>) -> Self {
        let mut st = LpcnetPlcState {
            model: models.plc,
            fargan: FarganState::new(models.fargan),
            enc: LpcnetEncState::new(models.pitchdnn),
            fec: [[0.0; NB_FEATURES]; PLC_MAX_FEC],
            analysis_gap: 0,
            fec_read_pos: 0,
            fec_fill_pos: 0,
            fec_skip: 0,
            analysis_pos: 0,
            predict_pos: 0,
            pcm: [0.0; PLC_BUF_SIZE],
            blend: 0,
            features: [0.0; NB_TOTAL_FEATURES],
            cont_features: [0.0; CONT_VECTORS * NB_FEATURES],
            loss_count: 0,
            plc_net: PlcNetState::ZERO,
            plc_bak: [PlcNetState::ZERO; 2],
        };
        st.reset();
        st
    }

    /// `lpcnet_plc_reset`: clears everything from `fec` on and the feature
    /// analysis. The FARGAN state is kept, as in libopus.
    pub fn reset(&mut self) {
        self.fec = [[0.0; NB_FEATURES]; PLC_MAX_FEC];
        self.fec_read_pos = 0;
        self.fec_fill_pos = 0;
        self.fec_skip = 0;
        self.features = [0.0; NB_TOTAL_FEATURES];
        self.cont_features = [0.0; CONT_VECTORS * NB_FEATURES];
        self.plc_net = PlcNetState::ZERO;
        self.plc_bak = [PlcNetState::ZERO; 2];
        self.enc.reset();
        self.pcm = [0.0; PLC_BUF_SIZE];
        self.blend = 0;
        self.loss_count = 0;
        self.analysis_gap = 1;
        self.analysis_pos = PLC_BUF_SIZE as i32;
        self.predict_pos = PLC_BUF_SIZE as i32;
    }

    /// `lpcnet_plc_fec_add`: queues the features of one 10 ms frame from
    /// DRED, or (with `None`) a frame DRED does not cover.
    pub fn fec_add(&mut self, features: Option<&[f32]>) {
        match features {
            None => self.fec_skip += 1,
            Some(f) => {
                assert!(self.fec_fill_pos < PLC_MAX_FEC);
                self.fec[self.fec_fill_pos].copy_from_slice(&f[..NB_FEATURES]);
                self.fec_fill_pos += 1;
            }
        }
    }

    /// `lpcnet_plc_fec_clear`.
    pub fn fec_clear(&mut self) {
        self.fec_read_pos = 0;
        self.fec_fill_pos = 0;
        self.fec_skip = 0;
    }

    /// `compute_plc_pred`.
    fn compute_plc_pred(&mut self, out: &mut [f32], input: &[f32]) {
        let m = &self.model;
        let net = &mut self.plc_net;
        let mut tmp = [0.0f32; PLC_DENSE_IN_OUT_SIZE];
        compute_generic_dense(&m.plc_dense_in, &mut tmp, input, Activation::Tanh);
        compute_generic_gru(
            &m.plc_gru1_input,
            &m.plc_gru1_recurrent,
            &mut net.gru1_state,
            &tmp,
        );
        let g1 = net.gru1_state;
        compute_generic_gru(
            &m.plc_gru2_input,
            &m.plc_gru2_recurrent,
            &mut net.gru2_state,
            &g1,
        );
        compute_generic_dense(&m.plc_dense_out, out, &net.gru2_state, Activation::Linear);
    }

    /// `get_fec_or_pred`: the next DRED features, or a prediction. Returns
    /// true for DRED features.
    fn get_fec_or_pred(&mut self, out: &mut [f32]) -> bool {
        if self.fec_read_pos != self.fec_fill_pos && self.fec_skip == 0 {
            let mut plc_features = [0.0f32; PLC_IN_SIZE];
            let mut discard = [0.0f32; NB_FEATURES];
            out[..NB_FEATURES].copy_from_slice(&self.fec[self.fec_read_pos]);
            self.fec_read_pos += 1;
            plc_features[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES]
                .copy_from_slice(&out[..NB_FEATURES]);
            plc_features[2 * NB_BANDS + NB_FEATURES] = -1.0;
            self.compute_plc_pred(&mut discard, &plc_features);
            true
        } else {
            let zeros = [0.0f32; PLC_IN_SIZE];
            self.compute_plc_pred(out, &zeros);
            if self.fec_skip > 0 {
                self.fec_skip -= 1;
            }
            false
        }
    }

    /// `queue_features`.
    fn queue_features(&mut self, features: &[f32]) {
        self.cont_features.copy_within(NB_FEATURES.., 0);
        self.cont_features[(CONT_VECTORS - 1) * NB_FEATURES..]
            .copy_from_slice(&features[..NB_FEATURES]);
    }

    fn push_pcm(&mut self, pcm: &[i16]) {
        self.pcm.copy_within(FRAME_SIZE.., 0);
        for (o, &v) in self.pcm[PLC_BUF_SIZE - FRAME_SIZE..]
            .iter_mut()
            .zip(&pcm[..FRAME_SIZE])
        {
            *o = (1.0f32 / 32768.0) * v as f32;
        }
    }

    /// `lpcnet_plc_update`: records one received 10 ms frame (16 kHz).
    pub fn update(&mut self, pcm: &[i16]) {
        if self.analysis_pos - FRAME_SIZE as i32 >= 0 {
            self.analysis_pos -= FRAME_SIZE as i32;
        } else {
            self.analysis_gap = 1;
        }
        if self.predict_pos - FRAME_SIZE as i32 >= 0 {
            self.predict_pos -= FRAME_SIZE as i32;
        }
        self.push_pcm(pcm);
        self.loss_count = 0;
        self.blend = 0;
    }

    /// `lpcnet_plc_conceal`: synthesizes one lost 10 ms frame (16 kHz).
    pub fn conceal(&mut self, pcm: &mut [i16]) {
        const ATT_TABLE: [f32; 10] = [0.0, 0.0, -0.2, -0.2, -0.4, -0.4, -0.8, -0.8, -1.6, -1.6];
        if self.blend == 0 {
            let mut count = 0;
            self.plc_net = self.plc_bak[0];
            while self.analysis_pos + FRAME_SIZE as i32 <= PLC_BUF_SIZE as i32 {
                let mut x = [0.0f32; FRAME_SIZE];
                let mut plc_features = [0.0f32; PLC_IN_SIZE];
                let ap = self.analysis_pos as usize;
                for (o, &v) in x.iter_mut().zip(&self.pcm[ap..ap + FRAME_SIZE]) {
                    *o = 32768.0f32 * v;
                }
                burg_cepstral_analysis(&mut plc_features, &x);
                lpcnet_compute_single_frame_features_float(&mut self.enc, &x, &mut self.features);
                if (self.analysis_gap == 0 || count > 0) && self.analysis_pos >= self.predict_pos {
                    let mut features = self.features;
                    self.queue_features(&features);
                    plc_features[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES]
                        .copy_from_slice(&features[..NB_FEATURES]);
                    plc_features[2 * NB_BANDS + NB_FEATURES] = 1.0;
                    self.plc_bak[0] = self.plc_bak[1];
                    self.plc_bak[1] = self.plc_net;
                    self.compute_plc_pred(&mut features, &plc_features);
                    self.features[..NB_FEATURES].copy_from_slice(&features[..NB_FEATURES]);
                }
                self.analysis_pos += FRAME_SIZE as i32;
                count += 1;
            }
            for _ in 0..2 {
                self.plc_bak[0] = self.plc_bak[1];
                self.plc_bak[1] = self.plc_net;
                let mut features = self.features;
                self.get_fec_or_pred(&mut features);
                self.features = features;
                self.queue_features(&features);
            }
            let cont = self.cont_features;
            fargan_cont(
                &mut self.fargan,
                &self.pcm[PLC_BUF_SIZE - FARGAN_CONT_SAMPLES..],
                &cont,
            );
            self.analysis_gap = 0;
        }
        self.plc_bak[0] = self.plc_bak[1];
        self.plc_bak[1] = self.plc_net;
        let mut features = self.features;
        if self.get_fec_or_pred(&mut features) {
            self.loss_count = 0;
        } else {
            self.loss_count += 1;
        }
        let f0 = if self.loss_count >= 10 {
            features[0] + ATT_TABLE[9] - (2 * (self.loss_count - 9)) as f32
        } else {
            features[0] + ATT_TABLE[self.loss_count as usize]
        };
        features[0] = if -15.0 > f0 { -15.0 } else { f0 };
        self.features = features;
        fargan_synthesize_int(&mut self.fargan, pcm, &features);
        self.queue_features(&features);
        if self.analysis_pos - FRAME_SIZE as i32 >= 0 {
            self.analysis_pos -= FRAME_SIZE as i32;
        } else {
            self.analysis_gap = 1;
        }
        self.predict_pos = PLC_BUF_SIZE as i32;
        self.push_pcm(pcm);
        self.blend = 1;
    }
}
