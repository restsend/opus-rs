#!/usr/bin/env bash
# Runs every opus-rs check and prints a summary; exits non-zero if one fails.
#
#   scripts/test-all.sh          everything below except the libFuzzer build
#   scripts/test-all.sh --fuzz   also builds the fuzz targets with cargo-fuzz
#                                (needs a nightly cargo and cargo-fuzz on PATH)
#                                and runs fuzz/regressions/check.sh, then
#                                the same with --features fixed
#
# Needs: stable cargo, a C/C++ compiler, pkg-config, cmake, libopus (the
# interop dev-dependencies), python3, the pinned libopus sources for
# fuzz/opusref and capi: OPUSREF_SRC=<dir> or libopus/ in the repository
# root (scripts/fetch-libopus.sh), and the RFC 8251 test vectors in
# OPUS_TESTVECTORS. The flake's dev shell provides all of them:
# nix develop -c scripts/test-all.sh
#
# Steps:
#   suite            cargo test --release
#   suite-dnn        cargo test --release --features deep-plc,dred,osce
#   suite-silk-fixed cargo test --release --features silk-fixed (the
#                    fixed-point SILK encoder)
#   no_std           builds without std (libm), with and without the neural
#                    features and silk-fixed, then its library and
#                    integration tests
#   opusref          the differential tests against libopus built from source:
#                    float (with the float SILK grid, silk_float), --features
#                    dnn, --features fixed --test silk_fixed (fixed-point
#                    libopus against silk-fixed); opus-rs runs at the scalar
#                    level (OPUS_RS_ARCH unset)
#   opusref-simd     on x86-64, the same tests against libopus built as its
#                    configure builds it (--features simd), opus-rs at the
#                    CPU's level and at sse2 (OPUS_RS_ARCH, OPUSREF_ARCH)
#   opusref-dnn-simd on x86-64, the dnn tests (--features simd,dnn) against
#                    the simd reference with the neural features, opus-rs at
#                    the CPU's level and at sse2 and sse4_1
#   parity           opusref's parity example: every 48 kHz CELT configuration
#                    must give byte-identical packets (also for the simd pair)
#   replay           the fuzz reproducers in fuzz/regressions, in cargo test,
#                    for the default, dnn and fixed builds
#   capi             capi/test.sh on capi's libopus.so.0: libopus's test
#                    programs, the RFC 8251 vectors (OPUS_TESTVECTORS) and
#                    the cross-check with the reference libopus
#   capi-dnn         the same with the deep-plc, dred and osce features,
#                    and test_opus_dred
#
# Known failure off x86-64, skipped there: oracle_celt_sine_48k_mono_64k
# (tests/oracle_celt.rs) compares with packet bytes of libopus built with
# NEON kernels and contraction, which opus-rs does not reproduce yet.
set -euo pipefail

cd "$(dirname "$0")/.."
fuzz=0
for a in "$@"; do
  case $a in
    --fuzz) fuzz=1 ;;
    *) echo "usage: $0 [--fuzz]" >&2; exit 2 ;;
  esac
done

x86=0
[ "$(uname -m)" = x86_64 ] && x86=1
known=(--skip oracle_celt_sine_48k_mono_64k)
[ $x86 = 1 ] && known=()
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
  out=$(in_fuzz cargo run --release -p opusref "$@" --example parity) || return 1
  if grep -v 'identical 50/50' <<<"$out"; then
    echo "parity: the configurations above differ from libopus"
    return 1
  fi
}

# The opusref tests with opus-rs at level $1 against the simd reference at
# libopus level $2 (empty: the CPU's).
simd_pair() {
  (cd fuzz && OPUS_RS_ARCH=$1 OPUSREF_ARCH=$2 cargo test --release --no-fail-fast -p opusref --features simd)
}
# The same with the neural features (the dnn tests and the rest).
simd_pair_dnn() {
  (cd fuzz && OPUS_RS_ARCH=$1 OPUSREF_ARCH=$2 cargo test --release --no-fail-fast -p opusref --features simd,dnn)
}
simd_parity() { OPUS_RS_ARCH=native parity --features simd; }

# capi_test <test.sh option> <name> <cargo arguments>: builds capi's
# libopus.so.0 into <target>/capi-<name> and runs capi/test.sh on it.
capi_test() {
  local option=$1 target=${CARGO_TARGET_DIR:-$PWD/target}/capi-$2
  shift 2
  (cd capi && CARGO_TARGET_DIR=$target cargo build --release "$@") &&
    ln -sf libopus.so "$target/release/libopus.so.0" &&
    capi/test.sh "$target/release" "$target/tests" ${option:+"$option"}
}

step suite cargo test --release --no-fail-fast -- "${known[@]}"
step suite-dnn cargo test --release --no-fail-fast --features deep-plc,dred,osce -- "${known[@]}"
step suite-silk-fixed cargo test --release --no-fail-fast --features silk-fixed -- "${known[@]}"
step no_std-build cargo build --release --no-default-features --features libm
step no_std-build-dnn cargo build --release --no-default-features --features libm,dred,osce
step no_std-build-silk-fixed cargo build --release --no-default-features --features libm,silk-fixed
# Without `heap` the codec structs are inline arrays; the tests need a big stack.
export RUST_MIN_STACK=${RUST_MIN_STACK:-33554432}
step no_std-lib-tests cargo test --release --no-default-features --features libm --lib
step no_std-tests cargo test --release --no-fail-fast --no-default-features --features libm --test '*' -- "${known[@]}"
step opusref in_fuzz cargo test --release --no-fail-fast -p opusref
step opusref-dnn in_fuzz cargo test --release --no-fail-fast -p opusref --features dnn
step opusref-fixed in_fuzz cargo test --release -p opusref --features fixed --test silk_fixed
step parity parity
if [ $x86 = 1 ]; then
  step opusref-simd simd_pair native ""
  step opusref-simd-sse2 simd_pair sse2 2
  step opusref-dnn-simd simd_pair_dnn native ""
  step opusref-dnn-simd-sse2 simd_pair_dnn sse2 2
  step opusref-dnn-simd-sse4_1 simd_pair_dnn sse4_1 3
  step parity-simd simd_parity
fi
step replay in_fuzz cargo test --test regressions
step replay-dnn in_fuzz cargo test --features dnn --test regressions
step replay-fixed in_fuzz cargo test --features fixed --test regressions
step capi capi_test "" default
step capi-dnn capi_test --dred dnn --features deep-plc,dred,osce
if [ $fuzz = 1 ]; then
  step fuzz-build in_fuzz cargo fuzz build
  step check.sh fuzz/regressions/check.sh
  step fuzz-build-fixed in_fuzz cargo fuzz build --features fixed
  step check.sh-fixed env OFZ_BUILD=fixed fuzz/regressions/check.sh
fi

echo
echo "Summary${known[*]:+ (skipped known failure: oracle_celt_sine_48k_mono_64k)}:"
printf '  %s\n' "${results[@]}"
exit $failed
