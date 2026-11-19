//! The RDOVAE decoder (libopus `dnn/dred_rdovae_dec.c`).

use super::super::models::dred_rdovae_constants::{
    DRED_LATENT_DIM, DRED_MAX_CONV_INPUTS, DRED_NUM_FEATURES,
};
use super::super::models::dred_rdovae_dec::*;
use super::super::nnet::{
    Activation, compute_generic_conv1d, compute_generic_dense, compute_generic_gru, compute_glu,
};
use super::rdovae_enc::conv1_cond_init;
use crate::Arch;

/// libopus `RDOVAEDecState`.
#[derive(Clone, Debug)]
pub struct RdovaeDecState {
    pub initialized: bool,
    pub gru1_state: [f32; DEC_GRU1_STATE_SIZE],
    pub gru2_state: [f32; DEC_GRU2_STATE_SIZE],
    pub gru3_state: [f32; DEC_GRU3_STATE_SIZE],
    pub gru4_state: [f32; DEC_GRU4_STATE_SIZE],
    pub gru5_state: [f32; DEC_GRU5_STATE_SIZE],
    pub conv1_state: [f32; DEC_CONV1_STATE_SIZE],
    pub conv2_state: [f32; DEC_CONV2_STATE_SIZE],
    pub conv3_state: [f32; DEC_CONV3_STATE_SIZE],
    pub conv4_state: [f32; DEC_CONV4_STATE_SIZE],
    pub conv5_state: [f32; DEC_CONV5_STATE_SIZE],
}

impl Default for RdovaeDecState {
    /// A zeroed state.
    fn default() -> Self {
        RdovaeDecState {
            initialized: false,
            gru1_state: [0.0; DEC_GRU1_STATE_SIZE],
            gru2_state: [0.0; DEC_GRU2_STATE_SIZE],
            gru3_state: [0.0; DEC_GRU3_STATE_SIZE],
            gru4_state: [0.0; DEC_GRU4_STATE_SIZE],
            gru5_state: [0.0; DEC_GRU5_STATE_SIZE],
            conv1_state: [0.0; DEC_CONV1_STATE_SIZE],
            conv2_state: [0.0; DEC_CONV2_STATE_SIZE],
            conv3_state: [0.0; DEC_CONV3_STATE_SIZE],
            conv4_state: [0.0; DEC_CONV4_STATE_SIZE],
            conv5_state: [0.0; DEC_CONV5_STATE_SIZE],
        }
    }
}

/// The values of one `dred_rdovae_decode_qframe` output: four feature
/// frames, newest first.
pub const DRED_QFRAME_SIZE: usize = 4 * DRED_NUM_FEATURES;

/// `dred_rdovae_dec_init_states`: sets the GRU states from the
/// `DRED_STATE_DIM` initial state.
pub fn dred_rdovae_dec_init_states(
    h: &mut RdovaeDecState,
    model: &RdovaeDec<'_>,
    initial_state: &[f32],
    arch: Arch,
) {
    let mut hidden = [0.0f32; DEC_HIDDEN_INIT_OUT_SIZE];
    let mut state_init = [0.0f32;
        DEC_GRU1_STATE_SIZE
            + DEC_GRU2_STATE_SIZE
            + DEC_GRU3_STATE_SIZE
            + DEC_GRU4_STATE_SIZE
            + DEC_GRU5_STATE_SIZE];
    compute_generic_dense(
        &model.dec_hidden_init,
        &mut hidden,
        initial_state,
        Activation::Tanh,
        arch,
    );
    compute_generic_dense(
        &model.dec_gru_init,
        &mut state_init,
        &hidden,
        Activation::Tanh,
        arch,
    );
    let mut c = 0;
    for gru in [
        &mut h.gru1_state[..],
        &mut h.gru2_state[..],
        &mut h.gru3_state[..],
        &mut h.gru4_state[..],
        &mut h.gru5_state[..],
    ] {
        let n = gru.len();
        gru.copy_from_slice(&state_init[c..c + n]);
        c += n;
    }
    h.initialized = false;
}

const BUFFER_SIZE: usize = DEC_DENSE1_OUT_SIZE
    + DEC_GRU1_OUT_SIZE
    + DEC_GRU2_OUT_SIZE
    + DEC_GRU3_OUT_SIZE
    + DEC_GRU4_OUT_SIZE
    + DEC_GRU5_OUT_SIZE
    + DEC_CONV1_OUT_SIZE
    + DEC_CONV2_OUT_SIZE
    + DEC_CONV3_OUT_SIZE
    + DEC_CONV4_OUT_SIZE
    + DEC_CONV5_OUT_SIZE;

/// `dred_rdovae_decode_qframe`: decodes one latent vector
/// (`DRED_LATENT_DIM + 1` values, the last one the quantizer level) into
/// [`DRED_QFRAME_SIZE`] feature values.
pub fn dred_rdovae_decode_qframe(
    st: &mut RdovaeDecState,
    model: &RdovaeDec<'_>,
    qframe: &mut [f32],
    input: &[f32],
    arch: Arch,
) {
    let mut buffer = [0.0f32; BUFFER_SIZE];
    let mut conv_tmp = [0.0f32; DRED_MAX_CONV_INPUTS];
    let mut oi = 0;

    compute_generic_dense(
        &model.dec_dense1,
        &mut buffer[oi..],
        input,
        Activation::Tanh,
        arch,
    );
    oi += DEC_DENSE1_OUT_SIZE;

    // Five times: a GRU, a GLU of its state, a dense layer and a convolution.
    macro_rules! stage {
        ($gi:ident, $gr:ident, $gs:ident, $glu:ident, $gn:expr, $cd:ident, $cv:ident, $cs:ident, $cin:expr, $cn:expr) => {
            compute_generic_gru(&model.$gi, &model.$gr, &mut st.$gs, &buffer, arch);
            compute_glu(&model.$glu, &mut buffer[oi..], &st.$gs, arch);
            oi += $gn;
            conv1_cond_init(&mut st.$cs, $cin, 1, st.initialized);
            compute_generic_dense(&model.$cd, &mut conv_tmp, &buffer, Activation::Tanh, arch);
            compute_generic_conv1d(
                &model.$cv,
                &mut buffer[oi..],
                &mut st.$cs,
                &conv_tmp,
                $cn,
                Activation::Tanh,
                arch,
            );
            oi += $cn;
        };
    }
    stage!(
        dec_gru1_input,
        dec_gru1_recurrent,
        gru1_state,
        dec_glu1,
        DEC_GRU1_OUT_SIZE,
        dec_conv_dense1,
        dec_conv1,
        conv1_state,
        DEC_CONV1_IN_SIZE,
        DEC_CONV1_OUT_SIZE
    );
    stage!(
        dec_gru2_input,
        dec_gru2_recurrent,
        gru2_state,
        dec_glu2,
        DEC_GRU2_OUT_SIZE,
        dec_conv_dense2,
        dec_conv2,
        conv2_state,
        DEC_CONV2_IN_SIZE,
        DEC_CONV2_OUT_SIZE
    );
    stage!(
        dec_gru3_input,
        dec_gru3_recurrent,
        gru3_state,
        dec_glu3,
        DEC_GRU3_OUT_SIZE,
        dec_conv_dense3,
        dec_conv3,
        conv3_state,
        DEC_CONV3_IN_SIZE,
        DEC_CONV3_OUT_SIZE
    );
    stage!(
        dec_gru4_input,
        dec_gru4_recurrent,
        gru4_state,
        dec_glu4,
        DEC_GRU4_OUT_SIZE,
        dec_conv_dense4,
        dec_conv4,
        conv4_state,
        DEC_CONV4_IN_SIZE,
        DEC_CONV4_OUT_SIZE
    );
    stage!(
        dec_gru5_input,
        dec_gru5_recurrent,
        gru5_state,
        dec_glu5,
        DEC_GRU5_OUT_SIZE,
        dec_conv_dense5,
        dec_conv5,
        conv5_state,
        DEC_CONV5_IN_SIZE,
        DEC_CONV5_OUT_SIZE
    );
    debug_assert_eq!(oi, BUFFER_SIZE);

    compute_generic_dense(&model.dec_output, qframe, &buffer, Activation::Linear, arch);
    st.initialized = true;
}

/// `DRED_rdovae_decode_all`: decodes `nb_latents` latent vectors (each
/// `DRED_LATENT_DIM + 1` values) from a fresh state initialized with
/// `state` into `4 * nb_latents` feature frames of `DRED_NUM_FEATURES`.
pub fn dred_rdovae_decode_all(
    model: &RdovaeDec<'_>,
    features: &mut [f32],
    state: &[f32],
    latents: &[f32],
    nb_latents: usize,
    arch: Arch,
) {
    let mut dec = RdovaeDecState::default();
    dred_rdovae_dec_init_states(&mut dec, model, state, arch);
    for k in 0..nb_latents {
        dred_rdovae_decode_qframe(
            &mut dec,
            model,
            &mut features[k * DRED_QFRAME_SIZE..],
            &latents[k * (DRED_LATENT_DIM + 1)..],
            arch,
        );
    }
}
