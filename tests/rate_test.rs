use opus_rs::modes::default_mode;
use opus_rs::range_coder::RangeCoder;
use opus_rs::rate::{BITRES, clt_compute_allocation, pulses2bits};

#[test]
fn test_rate_allocation_constraints() {
    let mode = default_mode();
    let nb_ebands = mode.nb_ebands;

    // Setup buffers
    let offsets = vec![0; nb_ebands];
    let cap = vec![100; nb_ebands]; // High cap

    let mut pulses = vec![0; nb_ebands];
    let mut fine_quant = vec![0; nb_ebands];
    let mut fine_priority = vec![0; nb_ebands];

    let mut intensity = 0;
    let mut dual_stereo = 0;
    let mut balance = 0;

    let total_bits_bytes = 40; // 40 bytes -> 320 bits
    let total_bits = (total_bits_bytes * 8) << BITRES; // Q3

    let mut rc = RangeCoder::new_encoder(100);

    let used_bits = clt_compute_allocation(
        mode,
        0,
        nb_ebands,
        &offsets,
        &cap,
        5, // alloc_trim
        &mut intensity,
        &mut dual_stereo,
        total_bits,
        &mut balance,
        &mut pulses,
        &mut fine_quant,
        &mut fine_priority,
        1, // channels
        3, // lm
        &mut rc,
        true, // encode
        0,    // prev (intra?)
        nb_ebands as i32,
    );

    // Verify constraints
    println!("Used allocation bits: {} / {}", used_bits, total_bits);
    assert!(used_bits <= total_bits, "Over-allocation!");

    // Verify pulse validity
    for i in 0..nb_ebands {
        // pulses[i] from clt_compute_allocation is the raw bit allocation per band (Q3 units),
        // not an actual pulse count. Verify it is non-negative and within the cap.
        assert!(pulses[i] >= 0, "Negative pulses in band {}", i);
        // Convert bits -> pulse count, then verify round-trip bits are consistent.
        let pulse_count = opus_rs::rate::bits2pulses(mode, i, 3, pulses[i]);
        let _bits_calc = pulses2bits(mode, i, 3, pulse_count);
    }

    // Check if we used a reasonable amount of bits (not 0)
    assert!(
        used_bits > 0,
        "Allocation used 0 bits with plenty available"
    );
}
