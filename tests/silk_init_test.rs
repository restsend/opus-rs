use opus_rs::silk::init_encoder::silk_init_encoder;
use opus_rs::silk::structs::SilkEncoderState;

#[test]
fn test_silk_init() {
    let mut enc_state = Box::new(SilkEncoderState::default());
    let ret = silk_init_encoder(&mut *enc_state, 0);
    assert_eq!(ret, 0, "silk_init_encoder failed");
    
    // Check some initialized values if possible.
    // e.g., first_frame_after_reset should be 1
    assert_eq!(enc_state.s_cmn.first_frame_after_reset, 1);
    
    // Check VAD state initialization (indirectly)
    // s_vad.ana_state should be zeroed (default) or set.
}
