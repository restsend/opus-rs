//! A multistream layout must satisfy RFC 7845 §5.1.1 before decoding, as
//! libopus checks in `opus_multistream_decoder_init`/`validate_layout`: at
//! least one stream, at most 255 decoded channels, and each mapping entry a
//! decoded channel or 255. An out-of-range entry used to be accepted and then
//! indexed past the decoded buffer while routing channels.

use opus_rs::multistream::{ChannelMappingTable, MultistreamDecoder};

fn table(channels: u8, stream_count: u8, coupled_count: u8, mapping: &[u8]) -> ChannelMappingTable {
    ChannelMappingTable {
        channels,
        stream_count,
        coupled_count,
        mapping: mapping.to_vec(),
    }
}

#[test]
fn a_mapping_entry_past_the_decoded_channels_is_rejected() {
    // One mono stream decodes one channel; entry 1 is neither it nor 255.
    let decoded = MultistreamDecoder::new(48000, table(2, 1, 0, &[0, 1])).and_then(|mut ms| {
        let mut out = vec![0.0; 960 * 2];
        // A ToC-only CELT 20 ms packet: 960 concealed samples.
        ms.decode(&[0xf8], 960, &mut out)
    });
    assert!(decoded.is_err(), "invalid layout accepted: {decoded:?}");
}

#[test]
fn a_layout_without_streams_is_rejected() {
    assert!(MultistreamDecoder::new(48000, table(1, 0, 0, &[255])).is_err());
}

#[test]
fn more_than_255_decoded_channels_are_rejected() {
    assert!(MultistreamDecoder::new(48000, table(1, 255, 1, &[0])).is_err());
}

#[test]
fn valid_layouts_with_silent_channels_still_decode() {
    // A stereo stream plus a mono stream: decoded channels 0..=2, and one
    // silent output channel.
    let mut ms = MultistreamDecoder::new(48000, table(4, 2, 1, &[0, 1, 2, 255])).unwrap();
    let mut out = vec![1.0; 960 * 4];
    // Self-delimited ToC-only frame for the first stream, then a plain one.
    assert_eq!(ms.decode(&[0xfc, 0, 0xf8], 960, &mut out), Ok(960));
    assert!(out.iter().skip(3).step_by(4).all(|&s| s == 0.0));
}
