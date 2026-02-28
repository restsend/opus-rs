use opus_rs::modes::default_mode;
use opus_rs::quant_bands::{quant_coarse_energy, quant_fine_energy};
use opus_rs::range_coder::RangeCoder;

#[test]
fn test_quant_energy_match() {
    let mode = default_mode();
    let nb_ebands = mode.nb_ebands;
    
    // Inputs matching verify_quant.c
    let mut e_bands = vec![0.0f32; nb_ebands];
    for i in 0..nb_ebands {
        e_bands[i] = 5.0 + (i as f32 * 0.5).sin() * 2.0;
    }
    
    let mut old_e_bands = vec![0.0f32; nb_ebands]; // All zeros
    let mut error = vec![0.0f32; nb_ebands];
    
    // Encoder
    let mut enc = RangeCoder::new_encoder(1000);
    
    let start = 0;
    let end = nb_ebands;
    let budget = 10000;
    let channels = 1;
    let lm = 3;
    let intra = false; // Based on verify_quant.c logic (force_intra=0, delayedIntra=0 => intra=0)
    
    // 1. Coarse Energy
    quant_coarse_energy(
        mode,
        start,
        end,
        &e_bands,
        &mut old_e_bands,
        budget,
        &mut error,
        &mut enc,
        channels,
        lm,
        intra
    );
    
    // 2. Fine Energy
    let mut fine_quant = vec![0i32; nb_ebands];
    for i in 0..nb_ebands {
        fine_quant[i] = (i % 3) as i32;
    }
    
    quant_fine_energy(
        mode,
        start,
        end,
        &mut old_e_bands,
        &mut error,
        &fine_quant,
        &mut enc,
        channels
    );
    
    enc.done();
    
    // Verification
    // Reference output from verify_quant.c (Opus 1.6)
    let ref_old_e_bands = [
        5.000000, 5.749939, 6.724915, 7.399902, 6.949890, 6.074890, 5.399902, 4.349915, 
        3.424927, 2.999939, 2.949951, 3.574951, 4.199951, 5.249939, 6.424927, 6.399902, 
        7.149902, 6.574890, 5.399902, 4.849915, 3.924927
    ];
    
    let ref_bytes: [u8; 7] = [0xDB, 0x43, 0xC7, 0x87, 0x70, 0x68, 0xC4];
    let ref_last_bytes: [u8; 3] = [0x17, 0x1C, 0x14]; // [997], [998], [999]
    
    for i in 0..7 {
        assert_eq!(enc.buf[i], ref_bytes[i], "Byte {} mismatch", i);
    }
    
    // Check end bytes. RangeCoder done() flushed bits to the end of buffer.
    for i in 0..3 {
        assert_eq!(enc.buf[enc.storage as usize - 3 + i], ref_last_bytes[i], "End byte {} mismatch", i);
    }
    
    for i in 0..nb_ebands {
        let diff = (old_e_bands[i] - ref_old_e_bands[i]).abs();
        assert!(diff < 1e-4, "Band {} mismatch: expected {}, got {}", i, ref_old_e_bands[i], old_e_bands[i]);
    }
    
    println!("Everything matches bit-exactly (stream) and with low precision error (values)!");
}
