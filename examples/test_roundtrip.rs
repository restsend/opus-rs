// Test roundtrip for key SILK parameters
use opus_rs::range_coder::RangeCoder;

fn test_single_encode_decode() {
    println!("\n=== Testing Single Value Roundtrip ===");

    // Simple table: just test value 0
    let icdf = [255u8; 256];

    // Encode 0
    let mut rc_enc = RangeCoder::new_encoder(1024);
    rc_enc.encode_icdf(0, &icdf, 8);
    rc_enc.done();
    let encoded = rc_enc.finish();

    println!(
        "Encoded {} bytes: {:02x?}",
        encoded.len(),
        &encoded[..encoded.len().min(20)]
    );

    // Decode
    let mut rc_dec = RangeCoder::new_decoder(encoded);
    let decoded = rc_dec.decode_icdf(&icdf, 8);
    println!("Decoded: {} (expected 0)", decoded);
}

fn test_encode_10_values() {
    println!("\n=== Testing 10 Values Roundtrip ===");

    // Simple table
    let icdf = [255u8; 256];

    for val in 0..10 {
        // Encode
        let mut rc_enc = RangeCoder::new_encoder(1024);
        rc_enc.encode_icdf(val, &icdf, 8);
        rc_enc.done();
        let encoded = rc_enc.finish();

        // Decode
        let mut rc_dec = RangeCoder::new_decoder(encoded.clone());
        let decoded = rc_dec.decode_icdf(&icdf, 8);

        if val == decoded {
            println!("OK: {} -> {} bytes -> {}", val, encoded.len(), decoded);
        } else {
            println!(
                "FAIL: {} -> {} bytes -> {} (expected {})",
                val,
                encoded.len(),
                decoded,
                val
            );
            return;
        }
    }
    println!("All tests passed!");
}

fn main() {
    test_single_encode_decode();
    test_encode_10_values();
}
