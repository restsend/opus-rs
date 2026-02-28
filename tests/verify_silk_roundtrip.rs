/// Test SILK encoder-decoder roundtrip consistency
use opus_rs::silk::dec_api::SilkDecoder;

/// Test decoder initialization at different sample rates
#[test]
fn test_silk_decoder_initialization() {
    let mut decoder = SilkDecoder::new();

    // Test 16kHz
    let ret = decoder.init(16000, 1);
    assert_eq!(ret, 0);
    assert_eq!(decoder.frame_length(), 320); // 4 * 80 = 320

    // Reset and test 12kHz
    decoder.reset();
    let ret = decoder.init(12000, 1);
    assert_eq!(ret, 0);
    assert_eq!(decoder.frame_length(), 240); // 4 * 60 = 240

    // Reset and test 8kHz
    decoder.reset();
    let ret = decoder.init(8000, 1);
    assert_eq!(ret, 0);
    // 8kHz with nb_subfr=4: frame_length = 40 * 4 = 160 samples (20ms)
    assert_eq!(decoder.frame_length(), 160);
}

/// Test that decoder state is correctly initialized
#[test]
fn test_silk_decoder_state() {
    let mut decoder = SilkDecoder::new();
    let ret = decoder.init(16000, 1);
    assert_eq!(ret, 0, "Init failed");

    // Check that channel state is initialized
    assert_eq!(decoder.channel_state[0].fs_khz, 16);
    assert_eq!(decoder.channel_state[0].nb_subfr, 4);
    assert_eq!(decoder.channel_state[0].subfr_length, 80);
    assert_eq!(decoder.channel_state[0].lpc_order, 16);
}
