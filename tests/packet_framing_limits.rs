use opus_rs::OpusDecoder;

fn reject(packet: &[u8], capacity: usize) {
    let mut decoder = OpusDecoder::new(48000, 1).unwrap();
    let mut output = vec![0.0; capacity];
    assert!(
        decoder.decode(packet, capacity, &mut output).is_err(),
        "malformed packet accepted: TOC={:02x}, bytes={}",
        packet[0],
        packet.len()
    );
}

#[test]
fn code_zero_rejects_a_frame_over_1275_bytes() {
    let mut packet = vec![0xff; 1277];
    packet[0] = 0x80;
    reject(&packet, 5760);
}

#[test]
fn an_oversized_silk_frame_is_rejected_before_its_redundancy_is_parsed() {
    // Longer than the range decoder's 2048-byte buffer: the implicit SILK
    // redundancy length then exceeded the decoder's storage and underflowed.
    let mut packet = vec![0; 2101];
    packet[0] = 0x08;
    reject(&packet, 5760);
}

#[test]
fn code_one_rejects_frames_over_1275_bytes() {
    let mut packet = vec![0xff; 1 + 2 * 1276];
    packet[0] = 0x81;
    reject(&packet, 5760);
}

#[test]
fn code_two_rejects_an_oversized_implicit_last_frame() {
    let mut packet = vec![0xff; 2 + 1276];
    packet[0] = 0x82;
    packet[1] = 0;
    reject(&packet, 5760);
}

#[test]
fn code_three_rejects_an_oversized_cbr_frame() {
    let mut packet = vec![0xff; 2 + 1276];
    packet[0] = 0x83;
    packet[1] = 1;
    reject(&packet, 5760);
}

#[test]
fn code_three_rejects_an_oversized_implicit_vbr_frame() {
    let mut packet = vec![0xff; 3 + 1276];
    packet[0] = 0x83;
    packet[1] = 0x82;
    packet[2] = 0;
    reject(&packet, 5760);
}

#[test]
fn code_three_rejects_more_than_120_ms_even_with_a_large_buffer() {
    reject(&[0x1b, 3], 8640);
    reject(&[0x9b, 7], 6720);
}

#[test]
fn code_two_requires_the_first_length_byte() {
    reject(&[0x82], 5760);
}

#[test]
fn fec_rejects_malformed_packets_before_concealing() {
    for packet in [&[0x82][..], &[0x83][..], &[0x9b, 7][..]] {
        let mut decoder = OpusDecoder::new(48000, 1).unwrap();
        let mut output = vec![0.0; 5760];
        assert!(decoder.decode_fec(packet, 5760, &mut output).is_err());
    }
}

#[test]
fn self_delimited_packets_obey_the_duration_limit() {
    use opus_rs::multistream::split_self_delimited;

    let mut scratch = Vec::new();
    assert!(split_self_delimited(&[0x9b, 7, 0], &mut scratch).is_err());
    assert!(split_self_delimited(&[0x83, 49, 0], &mut scratch).is_err());
    assert_eq!(
        split_self_delimited(&[0x83, 48, 0], &mut scratch),
        Ok((&[0x83, 48][..], 3))
    );
}

#[test]
fn zero_length_frames_and_120_ms_packets_remain_valid() {
    for (packet, expected) in [
        (&[0x80][..], 120),
        (&[0x81][..], 240),
        (&[0x82, 0][..], 240),
        (&[0x83, 48][..], 5760),
        (&[0x83, 0xc2, 1, 0, 0][..], 240),
        (&[0x19][..], 5760),
    ] {
        let mut decoder = OpusDecoder::new(48000, 1).unwrap();
        let mut output = vec![0.0; 5760];
        assert_eq!(decoder.decode(packet, 5760, &mut output), Ok(expected));
    }
}

#[test]
fn malformed_code_three_headers_are_rejected() {
    for packet in [
        &[0x83][..],
        &[0x83, 0][..],
        &[0x83, 49][..],
        &[0x83, 0x41][..],
        &[0x83, 0x41, 255][..],
        &[0x83, 0x41, 2, 0][..],
        &[0x83, 0x82][..],
        &[0x83, 0x82, 252][..],
        &[0x83, 0x82, 1][..],
        &[0x83, 2, 0][..],
    ] {
        reject(packet, 5760);
    }
}
