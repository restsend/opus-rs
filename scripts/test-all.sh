#!/usr/bin/env bash
# Runs every opus-rs check and prints a summary; exits non-zero if one fails.
#
#   scripts/test-all.sh          everything below except the libFuzzer build
#   scripts/test-all.sh --fuzz   also builds the fuzz targets with cargo-fuzz
#                                (needs a nightly cargo and cargo-fuzz on PATH)
#                                and runs fuzz/regressions/check.sh
#
# Needs: stable cargo, a C/C++ compiler, pkg-config, cmake, libopus (the
# interop dev-dependencies), and the libopus 1.6.1 sources for fuzz/opusref:
# OPUSREF_SRC=<dir> or opus-1.6.1/ in the repository root. The flake's dev
# shell provides all of them: nix develop -c scripts/test-all.sh
#
# Steps:
#   suite            cargo test --release
#   suite-dnn        cargo test --release --features deep-plc,dred,osce
#   no_std           builds without std (libm), with and without the neural
#                    features, then its library and integration tests
#   opusref          the differential tests against libopus built from source:
#                    float, --features dnn, --features fixed --test silk_fixed
#   parity           opusref's parity example: every 48 kHz CELT configuration
#                    must give byte-identical packets
#   replay           the fuzz reproducers in fuzz/regressions, in cargo test,
#                    for the default, dnn and fixed builds
#
# Known failure, skipped: oracle_celt_sine_48k_mono_64k (tests/oracle_celt.rs)
# compares with packet bytes recorded from a libopus build with SSE2/AVX2
# kernels; opus-rs follows the scalar C code, and its packet for that input
# is byte-identical with scalar libopus 1.6.1 (fuzz/opusref).
set -euo pipefail

cd "$(dirname "$0")/.."
fuzz=0
for a in "$@"; do
  case $a in
    --fuzz) fuzz=1 ;;
    *) echo "usage: $0 [--fuzz]" >&2; exit 2 ;;
  esac
done

known=(--skip oracle_celt_sine_48k_mono_64k)
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

parity() {
  local out
  out=$(in_fuzz cargo run --release -p opusref --example parity) || return 1
  if grep -v 'identical 50/50' <<<"$out"; then
    echo "parity: the configurations above differ from libopus"
    return 1
  fi
}

step suite cargo test --release --no-fail-fast -- "${known[@]}"
step suite-dnn cargo test --release --no-fail-fast --features deep-plc,dred,osce -- "${known[@]}"
step no_std-build cargo build --release --no-default-features --features libm
step no_std-build-dnn cargo build --release --no-default-features --features libm,dred,osce
# Without `heap` the codec structs are inline arrays; the tests need a big stack.
export RUST_MIN_STACK=${RUST_MIN_STACK:-33554432}
step no_std-lib-tests cargo test --release --no-default-features --features libm --lib
step no_std-tests cargo test --release --no-fail-fast --no-default-features --features libm --test '*' -- "${known[@]}"
step opusref in_fuzz cargo test --release --no-fail-fast -p opusref
step opusref-dnn in_fuzz cargo test --release --no-fail-fast -p opusref --features dnn
step opusref-fixed in_fuzz cargo test --release -p opusref --features fixed --test silk_fixed
step parity parity
step replay in_fuzz cargo test --test regressions
step replay-dnn in_fuzz cargo test --features dnn --test regressions
step replay-fixed in_fuzz cargo test --features fixed --test regressions
if [ $fuzz = 1 ]; then
  step fuzz-build in_fuzz cargo fuzz build
  step check.sh fuzz/regressions/check.sh
fi

echo
echo "Summary (skipped known failure: oracle_celt_sine_48k_mono_64k):"
printf '  %s\n' "${results[@]}"
exit $failed
