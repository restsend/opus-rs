#!/usr/bin/env bash
# Runs the opus-rs tests on wasm32-wasip1 under wasmtime and prints a
# summary; exits non-zero if one fails. Every step runs twice: the codec
# built without and with the simd128 target feature (whose SILK kernel gives
# the scalar results).
#
# Needs the flake's wasm dev shell: nix develop .#wasm -c scripts/test-wasm.sh
#
# Steps:
#   suite        cargo test --release --lib --tests (the doctests need a
#                rustdoc for the target)
#   suite-dnn    the same with --features deep-plc,dred,osce
#   opusref      the differential tests against libopus built from source
#                for wasm32-wasip1 (float, --features dnn, --features fixed
#                --test silk_fixed)
#
# WASI has no threads or processes: the tests that need them are ignored
# there, and the replay of the fuzz reproducers (a process per input) runs on
# native targets only (scripts/test-all.sh).
set -euo pipefail

cd "$(dirname "$0")/.."
root=$PWD
export CARGO_BUILD_TARGET=wasm32-wasip1
# The tests read files by absolute (CARGO_MANIFEST_DIR) and relative paths.
export CARGO_TARGET_WASM32_WASIP1_RUNNER="wasmtime run --dir=$root --dir=. --"
target=${CARGO_TARGET_DIR:-$root/target}
results=()
failed=0

step() {
  local name=$1; shift
  echo "==> $name: $*"
  if "$@"; then
    results+=("ok      $name")
  else
    results+=("FAILED  $name")
    failed=1
  fi
}

in_fuzz() { (cd fuzz && "$@"); }

for simd in "" simd128; do
  s=${simd:+-$simd}
  export CARGO_TARGET_WASM32_WASIP1_RUSTFLAGS=${simd:+-C target-feature=+$simd}
  export CARGO_TARGET_DIR=$target/wasm$s
  step suite$s cargo test --release --no-fail-fast --lib --tests
  step suite-dnn$s cargo test --release --no-fail-fast --lib --tests --features deep-plc,dred,osce
  step opusref$s in_fuzz cargo test --release --no-fail-fast -p opusref
  step opusref-dnn$s in_fuzz cargo test --release --no-fail-fast -p opusref --features dnn
  step opusref-fixed$s in_fuzz cargo test --release -p opusref --features fixed --test silk_fixed
done

echo
echo "Summary:"
printf '  %s\n' "${results[@]}"
exit $failed
