#![no_main]

libfuzzer_sys::fuzz_target!(|inp: opus_rs_fuzz::api_args::Input| opus_rs_fuzz::api_args::run(inp));
