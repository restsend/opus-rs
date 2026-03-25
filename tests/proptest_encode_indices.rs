use opus_rs::range_coder::RangeCoder;
use opus_rs::silk::define::*;
use opus_rs::silk::encode_indices::silk_encode_indices;
use opus_rs::silk::structs::SilkEncoderState;
use opus_rs::silk::tables_nlsf::{SILK_NLSF_CB_NB_MB, SILK_NLSF_CB_WB};

use proptest::prelude::*;

fn max_lag_index(fs_khz: i32) -> i16 {
    ((PE_MAX_LAG_MS - PE_MIN_LAG_MS) as i32 * fs_khz) as i16
}

fn make_enc(fs_khz: i32, nb_subfr: i32, lag_index: i16, prev_lag_index: i16) -> SilkEncoderState {
    let mut enc = SilkEncoderState::default();
    enc.s_cmn.fs_khz = fs_khz;
    enc.s_cmn.nb_subfr = nb_subfr;
    enc.s_cmn.ec_prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
    enc.s_cmn.ec_prev_lag_index = prev_lag_index;
    enc.ps_nlsf_cb = Some(if fs_khz == 16 {
        &SILK_NLSF_CB_WB
    } else {
        &SILK_NLSF_CB_NB_MB
    });
    enc.s_cmn.indices.signal_type = TYPE_VOICED as i8;
    enc.s_cmn.indices.quant_offset_type = 0;
    enc.s_cmn.indices.lag_index = lag_index;
    enc.s_cmn.indices.contour_index = 0;
    enc.s_cmn.indices.per_index = 0;
    enc.s_cmn.indices.nlsf_interp_coef_q2 = 4;
    enc.s_cmn.indices.gains_indices[0] = 0;
    enc.s_cmn.indices.seed = 0;
    enc
}

fn fs_khz_strategy() -> impl Strategy<Value = i32> {
    prop_oneof![Just(8), Just(12), Just(16)]
}

fn nb_subfr_strategy() -> impl Strategy<Value = i32> {
    prop_oneof![Just(2), Just(4)]
}

proptest! {
    #[test]
    fn prop_lag_index_full_range_no_panic(
        fs_khz in fs_khz_strategy(),
        nb_subfr in nb_subfr_strategy(),
        lag_index in 0i16..=288i16,
    ) {
        let lag_index = lag_index.min(max_lag_index(fs_khz));
        let mut enc = make_enc(fs_khz, nb_subfr, lag_index, 0);
        let mut rc = RangeCoder::new_encoder(1024);
        silk_encode_indices(&mut enc, &mut rc, 0, false, CODE_INDEPENDENTLY);
        prop_assert!(
            rc.error == 0,
            "error set for fs_khz={} lag_index={}", fs_khz, lag_index
        );
    }

    #[test]
    fn prop_lag_index_conditional_coding(
        fs_khz in fs_khz_strategy(),
        nb_subfr in nb_subfr_strategy(),
        lag_index in 0i16..=288i16,
        prev_lag_index in 0i16..=288i16,
    ) {
        let max_idx = max_lag_index(fs_khz);
        let lag_index = lag_index.min(max_idx);
        let prev_lag_index = prev_lag_index.min(max_idx);
        let mut enc = make_enc(fs_khz, nb_subfr, lag_index, prev_lag_index);
        enc.s_cmn.ec_prev_signal_type = TYPE_VOICED;
        let mut rc = RangeCoder::new_encoder(1024);
        silk_encode_indices(&mut enc, &mut rc, 0, false, CODE_CONDITIONALLY);
        prop_assert!(
            rc.error == 0,
            "error set for fs_khz={} lag_index={} prev={}", fs_khz, lag_index, prev_lag_index
        );
    }

    #[test]
    fn prop_lag_index_beyond_max_is_clamped(
        fs_khz in fs_khz_strategy(),
        nb_subfr in nb_subfr_strategy(),
        lag_index in 0i16..=i16::MAX,
    ) {
        let mut enc = make_enc(fs_khz, nb_subfr, lag_index, 0);
        let mut rc = RangeCoder::new_encoder(1024);
        silk_encode_indices(&mut enc, &mut rc, 0, false, CODE_INDEPENDENTLY);
        prop_assert!(
            rc.error == 0,
            "error set for fs_khz={} lag_index={}", fs_khz, lag_index
        );
    }
}

#[test]
fn exhaustive_lag_index_all_sample_rates() {
    for fs_khz in [8i32, 12, 16] {
        let max_idx = max_lag_index(fs_khz);
        for lag_index in 0i16..=max_idx {
            for nb_subfr in [2i32, 4] {
                let mut enc = make_enc(fs_khz, nb_subfr, lag_index, 0);
                let mut rc = RangeCoder::new_encoder(1024);
                silk_encode_indices(&mut enc, &mut rc, 0, false, CODE_INDEPENDENTLY);
                assert_eq!(
                    rc.error, 0,
                    "fs_khz={fs_khz} nb_subfr={nb_subfr} lag_index={lag_index}"
                );
            }
        }
    }
}

#[test]
fn exhaustive_lag_index_conditional_all_sample_rates() {
    for fs_khz in [8i32, 12, 16] {
        let max_idx = max_lag_index(fs_khz);
        for lag_index in (0i16..=max_idx).step_by(4) {
            for prev_lag_index in (0i16..=max_idx).step_by(8) {
                let mut enc = make_enc(fs_khz, 4, lag_index, prev_lag_index);
                enc.s_cmn.ec_prev_signal_type = TYPE_VOICED;
                let mut rc = RangeCoder::new_encoder(1024);
                silk_encode_indices(&mut enc, &mut rc, 0, false, CODE_CONDITIONALLY);
                assert_eq!(
                    rc.error, 0,
                    "fs_khz={fs_khz} lag_index={lag_index} prev={prev_lag_index}"
                );
            }
        }
    }
}
