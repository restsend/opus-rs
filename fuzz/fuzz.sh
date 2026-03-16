#!/bin/bash
# Fuzz testing runner script
# Usage: ./fuzz.sh [target] [duration_seconds] [additional args]

set -e

TARGET=${1:-fuzz_encoder}
DURATION=${2:-60}
shift 2 2>/dev/null || true

echo "Running fuzz target: $TARGET"
echo "Duration: ${DURATION}s"
echo "Additional args: $@"

# Check if cargo-fuzz is installed
if ! command -v cargo fuzz &> /dev/null; then
    echo "Installing cargo-fuzz..."
    cargo install cargo-fuzz
fi

# Check for nightly toolchain
if ! rustup toolchain list | grep -q nightly; then
    echo "Installing nightly toolchain..."
    rustup toolchain install nightly
fi

cd "$(dirname "$0")/.."
cargo +nightly fuzz run "$TARGET" --release -- -max_total_time="$DURATION" "$@"
