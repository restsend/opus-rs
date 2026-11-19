#![no_main]

libfuzzer_sys::fuzz_target!(|data: &[u8]| opus_rs_fuzz::diff_multistream::fuzz(data));
