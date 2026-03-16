# Fuzz Testing for opus-rs

This directory contains fuzz testing targets for the opus-rs Opus codec implementation.
Fuzz testing helps identify potential issues like:
- Integer overflows/underflows
- Buffer overflows
- Panic conditions
- Edge cases in encoding/decoding
- Invalid input handling

## Prerequisites

Install cargo-fuzz:
```bash
cargo install cargo-fuzz
```

Note: Fuzz testing requires the nightly Rust toolchain.
```bash
rustup toolchain install nightly
```

## Available Targets

| Target | Description |
|--------|-------------|
| `fuzz_encoder` | Tests OpusEncoder with various input patterns |
| `fuzz_decoder` | Tests OpusDecoder with arbitrary byte patterns |
| `fuzz_roundtrip` | Tests encode-decode roundtrip consistency |
| `fuzz_silk_encoder` | Focuses on SILK encoder edge cases |
| `fuzz_celt_encoder` | Focuses on CELT encoder edge cases |
| `fuzz_overflow` | Tests extreme values and boundary conditions |

## Running Fuzz Tests

### Quick Run (60 seconds)
```bash
./fuzz.sh fuzz_encoder
```

### Extended Run (1 hour)
```bash
./fuzz.sh fuzz_encoder 3600
```

### Manual Run
```bash
cargo +nightly fuzz run fuzz_encoder --release -- -max_total_time=60
```

### Run All Targets
```bash
for target in fuzz_encoder fuzz_decoder fuzz_roundtrip fuzz_silk_encoder fuzz_celt_encoder fuzz_overflow; do
    echo "Running $target..."
    cargo +nightly fuzz run $target --release -- -max_total_time=300
done
```

### Reproduce a Crash
```bash
cargo +nightly fuzz run fuzz_encoder --release -- crash-<hash>
```

### Minimize a Crash
```bash
cargo +nightly fuzz minimize fuzz_encoder --release -- crash-<hash>
```

## Debugging Failures

When a fuzz test fails, cargo-fuzz will create a crash file. You can:

1. Run with the crash file to reproduce:
   ```bash
   cargo +nightly fuzz run fuzz_encoder --release -- crash-<hash>
   ```

2. Build a debug version for better stack traces:
   ```bash
   cargo +nightly fuzz run fuzz_encoder --dev -- crash-<hash>
   ```

3. Use rust-gdb or rust-lldb for detailed debugging.

## Tips for Finding Overflows

1. **Use the overflow fuzz target**: It specifically tests extreme values
   ```bash
   ./fuzz.sh fuzz_overflow 300
   ```

2. **Run with debug assertions**: The dev profile enables overflow checks
   ```bash
   cargo +nightly fuzz run fuzz_encoder --dev -- -max_total_time=60
   ```

3. **Check for specific issues**:
   - Integer overflow in sample rate calculations
   - Buffer size miscalculations
   - Edge cases in frame size handling

## Adding New Targets

To add a new fuzz target:

1. Create a new file in `fuzz_targets/`:
   ```rust
   #![no_main]
   use libfuzzer_sys::fuzz_target;

   fuzz_target!(|data: &[u8]| {
       // Your fuzz test code here
   });
   ```

2. Add the target to `Cargo.toml`:
   ```toml
   [[bin]]
   name = "your_new_target"
   path = "fuzz_targets/your_new_target.rs"
   test = false
   doc = false
   bench = false
   ```
