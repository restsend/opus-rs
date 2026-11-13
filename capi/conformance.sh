#!/usr/bin/env bash
# Decoder conformance of the opus-rs C API against the RFC 8251 test vectors,
# case by case.
#
# libopus's tests/run_vectors.sh stops at the first failure of each rate.
# This runs every case (12 vectors x mono/stereo x 5 rates), in parallel,
# the same way: `opus_demo -d <rate> <channels> -ignore_extensions` (which
# also checks every packet's range coder state against the encoder's), then
# `opus_compare` against both references (testvectorNN.dec and
# testvectorNNm.dec), passing if either matches. It lists the failing cases
# with opus_compare's error.
#
# Usage: capi/conformance.sh [--ref] <vector dir>
#   --ref   also decode with the reference libopus (pkg-config `opus`), as a
#           check of the harness: it should pass everything.
#   JOBS    parallel jobs (default: nproc).
#
# Vectors: https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz
# (unpacks to opus_newvectors/).

set -u

here=$(cd "$(dirname "$0")" && pwd)
build=$here/target/libopus-tests
tmp=$build/conformance

# opus_compare's "Internal weighted error", or its last line.
compare() {
    local out
    out=$("$build/opus_compare" "$@" 2>&1) && { echo pass; return; }
    echo "$out" | sed -n 's/.*weighted error is \(.*\)/\1/p' | grep . || echo "$out" | tail -1
}

# One case: prints "<decoder> <rate> <vector> <channels> PASS|FAIL <detail>".
run_case() {
    local vectors=$1 dec=$2 rate=$3 f=$4 ch=$5
    local stereo=() out=$tmp/$dec-$f-$rate-$ch.out e1 e2
    [ "$ch" = 2 ] && stereo=(-s)
    local id
    id="$dec $(printf '%5s' "$rate") testvector$f $ch"
    if ! "$build/$dec" -d "$rate" "$ch" -ignore_extensions \
        "$vectors/testvector$f.bit" "$out" > "$out.log" 2>&1; then
        echo "$id FAIL decode failed: $(grep -m1 -i error "$out.log")"
        return
    fi
    e1=$(compare "${stereo[@]}" -r "$rate" "$vectors/testvector$f.dec" "$out")
    e2=$(compare "${stereo[@]}" -r "$rate" "$vectors/testvector${f}m.dec" "$out")
    rm -f "$out" "$out.log"
    if [ "$e1" = pass ] || [ "$e2" = pass ]; then
        echo "$id PASS"
    else
        echo "$id FAIL error $e1 (.dec) / $e2 (m.dec)"
    fi
}

if [ "${1:-}" = --case ]; then
    shift
    run_case "$@"
    exit 0
fi

ref=0
if [ "${1:-}" = --ref ]; then
    ref=1
    shift
fi
if [ $# != 1 ] || [ ! -f "$1/testvector01.bit" ]; then
    echo "usage: $0 [--ref] <vector dir (with testvector01.bit)>" >&2
    exit 2
fi
vectors=$(cd "$1" && pwd)

"$here/run_libopus_tests.sh" build > /dev/null || exit 2
decoders=(opus_demo)
if [ "$ref" = 1 ]; then
    if [ ! -x "$build/opus_demo_ref" ]; then
        echo "no reference libopus (pkg-config opus)" >&2
        exit 2
    fi
    decoders=(opus_demo_ref opus_demo)
fi
mkdir -p "$tmp"

results=$(
    for dec in "${decoders[@]}"; do
        for rate in 48000 24000 16000 12000 8000; do
            for f in 01 02 03 04 05 06 07 08 09 10 11 12; do
                for ch in 1 2; do
                    echo "$vectors $dec $rate $f $ch"
                done
            done
        done
    done | xargs -P "${JOBS:-$(nproc)}" -L 1 "$0" --case | sort -k1,1 -k2,2nr -k3,3 -k4,4
)

status=0
for dec in "${decoders[@]}"; do
    name=opus-rs
    [ "$dec" = opus_demo_ref ] && name="reference libopus"
    echo "== $name"
    mine=$(echo "$results" | awk -v d="$dec" '$1 == d')
    # Fields: decoder, rate, vector, channels, PASS|FAIL, detail...
    echo "$mine" | awk '$5 == "FAIL" {
        d = ""; for (i = 6; i <= NF; i++) d = d " " $i;
        printf "  FAIL %s %5s %-6s%s\n", $3, $2, ($4 == 2 ? "stereo" : "mono"), d
    }'
    pass=$(echo "$mine" | awk '$5 == "PASS"' | wc -l)
    echo "  $pass/120 pass"
    [ "$pass" = 120 ] || status=1
done
exit $status
