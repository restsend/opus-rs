//! The RDOVAE encoder (libopus `dnn/dred_rdovae_enc.c`).

use super::super::models::dred_rdovae_constants::{
    DRED_LATENT_DIM, DRED_MAX_CONV_INPUTS, DRED_PADDED_LATENT_DIM, DRED_PADDED_STATE_DIM, DRED_STATE_DIM,
};
use super::super::models::dred_rdovae_enc::*;
use super::super::nnet::{
    Activation, compute_generic_conv1d, compute_generic_conv1d_dilation, compute_generic_dense, compute_generic_gru,
};

/// libopus `RDOVAEEncState`.
#[derive(Clone, Debug)]
pub struct RdovaeEncState {
    pub initialized: bool,
    pub gru1_state: [f32; ENC_GRU1_STATE_SIZE],
    pub gru2_state: [f32; ENC_GRU2_STATE_SIZE],
    pub gru3_state: [f32; ENC_GRU3_STATE_SIZE],
    pub gru4_state: [f32; ENC_GRU4_STATE_SIZE],
    pub gru5_state: [f32; ENC_GRU5_STATE_SIZE],
    pub conv1_state: [f32; ENC_CONV1_STATE_SIZE],
    pub conv2_state: [f32; 2 * ENC_CONV2_STATE_SIZE],
    pub conv3_state: [f32; 2 * ENC_CONV3_STATE_SIZE],
    pub conv4_state: [f32; 2 * ENC_CONV4_STATE_SIZE],
    pub conv5_state: [f32; 2 * ENC_CONV5_STATE_SIZE],
}

impl Default for RdovaeEncState {
    /// The zeroed state of `DRED_rdovae_init_encoder`.
    fn default() -> Self {
        RdovaeEncState {
            initialized: false,
            gru1_state: [0.0; ENC_GRU1_STATE_SIZE],
            gru2_state: [0.0; ENC_GRU2_STATE_SIZE],
            gru3_state: [0.0; ENC_GRU3_STATE_SIZE],
            gru4_state: [0.0; ENC_GRU4_STATE_SIZE],
            gru5_state: [0.0; ENC_GRU5_STATE_SIZE],
            conv1_state: [0.0; ENC_CONV1_STATE_SIZE],
            conv2_state: [0.0; 2 * ENC_CONV2_STATE_SIZE],
            conv3_state: [0.0; 2 * ENC_CONV3_STATE_SIZE],
            conv4_state: [0.0; 2 * ENC_CONV4_STATE_SIZE],
            conv5_state: [0.0; 2 * ENC_CONV5_STATE_SIZE],
        }
    }
}

/// `conv1_cond_init`: clears `dilation` blocks of `len` values of `mem`
/// on the first call after a reset. The flag is shared by all five
/// convolutions, so only the first one is cleared, as in libopus.
pub(super) fn conv1_cond_init(mem: &mut [f32], len: usize, dilation: usize, init: &mut bool) {
    if !*init {
        mem[..dilation * len].fill(0.0);
    }
    *init = true;
}

const BUFFER_SIZE: usize = ENC_DENSE1_OUT_SIZE
    + ENC_GRU1_OUT_SIZE
    + ENC_GRU2_OUT_SIZE
    + ENC_GRU3_OUT_SIZE
    + ENC_GRU4_OUT_SIZE
    + ENC_GRU5_OUT_SIZE
    + ENC_CONV1_OUT_SIZE
    + ENC_CONV2_OUT_SIZE
    + ENC_CONV3_OUT_SIZE
    + ENC_CONV4_OUT_SIZE
    + ENC_CONV5_OUT_SIZE;

/// `dred_rdovae_encode_dframe`: encodes one double feature frame
/// (`2*DRED_NUM_FEATURES` values) into `DRED_LATENT_DIM` latents and a
/// `DRED_STATE_DIM` initial state.
pub fn dred_rdovae_encode_dframe(
    st: &mut RdovaeEncState,
    model: &RdovaeEnc<'_>,
    latents: &mut [f32],
    initial_state: &mut [f32],
    input: &[f32],
) {
    let mut padded_latents = [0.0f32; DRED_PADDED_LATENT_DIM];
    let mut padded_state = [0.0f32; DRED_PADDED_STATE_DIM];
    let mut buffer = [0.0f32; BUFFER_SIZE];
    let mut state_hidden = [0.0f32; GDENSE1_OUT_SIZE];
    let mut conv_tmp = [0.0f32; DRED_MAX_CONV_INPUTS];
    let mut oi = 0;

    compute_generic_dense(&model.enc_dense1, &mut buffer[oi..], input, Activation::Tanh);
    oi += ENC_DENSE1_OUT_SIZE;

    compute_generic_gru(&model.enc_gru1_input, &model.enc_gru1_recurrent, &mut st.gru1_state, &buffer);
    buffer[oi..oi + ENC_GRU1_OUT_SIZE].copy_from_slice(&st.gru1_state[..ENC_GRU1_OUT_SIZE]);
    oi += ENC_GRU1_OUT_SIZE;
    conv1_cond_init(&mut st.conv1_state, ENC_CONV1_IN_SIZE, 1, &mut st.initialized);
    compute_generic_dense(&model.enc_conv_dense1, &mut conv_tmp, &buffer, Activation::Tanh);
    compute_generic_conv1d(
        &model.enc_conv1,
        &mut buffer[oi..],
        &mut st.conv1_state,
        &conv_tmp,
        ENC_CONV1_OUT_SIZE,
        Activation::Tanh,
    );
    oi += ENC_CONV1_OUT_SIZE;

    // GRU 2 to 5, each followed by a dilated convolution.
    macro_rules! stage {
        ($gi:ident, $gr:ident, $gs:ident, $gn:expr, $cd:ident, $cv:ident, $cs:ident, $cin:expr, $cn:expr) => {
            compute_generic_gru(&model.$gi, &model.$gr, &mut st.$gs, &buffer);
            buffer[oi..oi + $gn].copy_from_slice(&st.$gs[..$gn]);
            oi += $gn;
            conv1_cond_init(&mut st.$cs, $cin, 2, &mut st.initialized);
            compute_generic_dense(&model.$cd, &mut conv_tmp, &buffer, Activation::Tanh);
            compute_generic_conv1d_dilation(
                &model.$cv,
                &mut buffer[oi..],
                &mut st.$cs,
                &conv_tmp,
                $cn,
                2,
                Activation::Tanh,
            );
            oi += $cn;
        };
    }
    stage!(
        enc_gru2_input,
        enc_gru2_recurrent,
        gru2_state,
        ENC_GRU2_OUT_SIZE,
        enc_conv_dense2,
        enc_conv2,
        conv2_state,
        ENC_CONV2_IN_SIZE,
        ENC_CONV2_OUT_SIZE
    );
    stage!(
        enc_gru3_input,
        enc_gru3_recurrent,
        gru3_state,
        ENC_GRU3_OUT_SIZE,
        enc_conv_dense3,
        enc_conv3,
        conv3_state,
        ENC_CONV3_IN_SIZE,
        ENC_CONV3_OUT_SIZE
    );
    stage!(
        enc_gru4_input,
        enc_gru4_recurrent,
        gru4_state,
        ENC_GRU4_OUT_SIZE,
        enc_conv_dense4,
        enc_conv4,
        conv4_state,
        ENC_CONV4_IN_SIZE,
        ENC_CONV4_OUT_SIZE
    );
    stage!(
        enc_gru5_input,
        enc_gru5_recurrent,
        gru5_state,
        ENC_GRU5_OUT_SIZE,
        enc_conv_dense5,
        enc_conv5,
        conv5_state,
        ENC_CONV5_IN_SIZE,
        ENC_CONV5_OUT_SIZE
    );
    debug_assert_eq!(oi, BUFFER_SIZE);

    compute_generic_dense(&model.enc_zdense, &mut padded_latents, &buffer, Activation::Linear);
    latents[..DRED_LATENT_DIM].copy_from_slice(&padded_latents[..DRED_LATENT_DIM]);

    compute_generic_dense(&model.gdense1, &mut state_hidden, &buffer, Activation::Tanh);
    compute_generic_dense(&model.gdense2, &mut padded_state, &state_hidden, Activation::Linear);
    initial_state[..DRED_STATE_DIM].copy_from_slice(&padded_state[..DRED_STATE_DIM]);
}
