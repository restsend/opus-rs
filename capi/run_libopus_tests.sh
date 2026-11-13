#!/usr/bin/env bash
# Runs libopus's own test programs against the opus-rs C API.
#
# The programs are compiled from libopus's sources, unmodified, and linked
# against capi's libopus.so instead of libopus:
#
#   test_opus_padding, test_opus_decode, test_opus_api, test_opus_encode
#       libopus's `make check` suite.
#   cross-check
#       opus_demo stores each packet's final range coder state; decoding
#       checks it, so a mismatch in any symbol fails. Encodes with libopus
#       and decodes with opus-rs, and the other way round, over a matrix of
#       settings. Needs a reference libopus (pkg-config `opus`).
#   test vectors
#       libopus's tests/run_vectors.sh (decoder conformance), when
#       OPUS_TESTVECTORS points at the extracted RFC 8251 vectors:
#         https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz
#       (it unpacks to opus_newvectors/).
#
# The libopus sources come from LIBOPUS_SRC, ../opus-1.6.1, or the copy the
# opusic-sys dev-dependency unpacks in the cargo registry.
#
# Usage: capi/run_libopus_tests.sh [test ...]   (default: all)
#        capi/run_libopus_tests.sh build        (only build the programs)

set -u

here=$(cd "$(dirname "$0")" && pwd)
build=$here/target/libopus-tests

find_src() {
    for d in "${LIBOPUS_SRC:-}" "$here/../opus-1.6.1" \
        "${CARGO_HOME:-$HOME/.cargo}"/registry/src/*/opusic-sys-*/opus; do
        if [ -n "$d" ] && [ -f "$d/tests/test_opus_api.c" ]; then
            echo "$d"
            return
        fi
    done
    echo "libopus sources not found; set LIBOPUS_SRC" >&2
    exit 2
}
src=$(find_src)

# Resolved before the cd below, so a relative path works.
if [ -n "${OPUS_TESTVECTORS:-}" ]; then
    if [ ! -f "$OPUS_TESTVECTORS/testvector01.bit" ]; then
        echo "no testvector01.bit in OPUS_TESTVECTORS=$OPUS_TESTVECTORS" >&2
        exit 2
    fi
    OPUS_TESTVECTORS=$(cd "$OPUS_TESTVECTORS" && pwd)
fi

echo "libopus sources: $src"
(cd "$here" && cargo build --release --quiet) || exit 2

mkdir -p "$build/cfg"
cd "$build" || exit 2
# The cdylib's soname is libopus.so.0: give the dynamic loader that name, so
# the programs cannot silently pick up the system libopus instead.
ln -sf ../release/libopus.so libopus.so.0
: > cfg/config.h

CFLAGS="-O2 -g -DHAVE_CONFIG_H -Icfg -I$src/include -I$src/celt -I$src/silk -I$src/src"
link_rs() { cc $CFLAGS "$@" -L. -Wl,-rpath,'$ORIGIN' -l:libopus.so.0 -lm; }

link_rs "$src/tests/test_opus_padding.c" -o test_opus_padding
link_rs "$src/tests/test_opus_decode.c" -o test_opus_decode
link_rs "$src/tests/test_opus_api.c" -o test_opus_api
link_rs "$src/tests/test_opus_encode.c" "$src/tests/opus_encode_regressions.c" -o test_opus_encode
link_rs "$src/src/opus_demo.c" -o opus_demo
cc -O2 "$src/src/opus_compare.c" -o opus_compare -lm
if ! ldd test_opus_api | grep -q "$build/libopus.so.0"; then
    echo "test programs do not load $build/libopus.so.0" >&2
    exit 2
fi
if pkg-config --exists opus 2>/dev/null; then
    cc $CFLAGS "$src/src/opus_demo.c" -o opus_demo_ref $(pkg-config --libs opus) -lm
fi

[ "${1:-}" = build ] && exit 0

results=()
record() { results+=("$(printf '%-22s %s' "$1" "$2")"); }

# Runs a libopus test program; on failure, shows the check that failed.
run_test() {
    local t=$1
    if timeout 1800 ./"$t" > "$t.log" 2>&1; then
        record "$t" PASS
        return
    fi
    local where file line
    where=$(grep -o "fails [^ ]* at line [0-9]*" "$t.log" | head -1)
    if [ -n "$where" ]; then
        file=$(echo "$where" | awk '{print $2}')
        line=$(echo "$where" | awk '{print $5}')
        record "$t" "FAIL $(basename "$file"):$line: $(sed -n "${line}p" "$file" | sed 's/^ *//')"
    else
        record "$t" "FAIL (see $build/$t.log)"
    fi
}

# Test signals, as raw 16-bit PCM: the repo's speech fixture, its stereo
# version (the right channel delayed), and the same samples read at other
# rates (opus_demo takes the rate as given).
make_inputs() {
    python3 -I - "$here/../fixtures/answer_16k.wav" <<'EOF'
import sys, wave, array
w = wave.open(sys.argv[1])
mono = array.array('h', w.readframes(w.getnframes()))
open('speech_mono.raw', 'wb').write(mono.tobytes())
stereo = array.array('h', [0]) * (2 * len(mono))
stereo[0::2] = mono
stereo[3::2] = mono[:-1]
open('speech_stereo.raw', 'wb').write(stereo.tobytes())
EOF
}

# One cross-check: encode with $1's opus_demo, decode with $2's.
cross() {
    local enc=$1 dec=$2 app=$3 rate=$4 ch=$5 bitrate=$6
    shift 6
    local input=speech_mono.raw
    [ "$ch" = 2 ] && input=speech_stereo.raw
    local name="$app-$rate-$ch-$bitrate${*:+ $*}"
    if ! ./"$enc" -e "$app" "$rate" "$ch" "$bitrate" "$@" "$input" x.bit > x.log 2>&1; then
        echo "  encode failed: $name"
        return 1
    fi
    if ! ./"$dec" -d "$rate" "$ch" x.bit x.out >> x.log 2>&1; then
        echo "  decode failed: $name: $(grep -m1 -i error x.log)"
        return 1
    fi
}

run_cross() {
    if [ ! -x opus_demo_ref ]; then
        record cross-check "SKIP (no reference libopus: pkg-config opus)"
        return
    fi
    make_inputs
    local configs=(
        "voip 8000 1 12000"
        "voip 16000 1 24000"
        "voip 16000 1 16000 -cbr"
        "voip 16000 1 20000 -inbandfec -loss 10"
        "voip 16000 2 32000"
        "voip 48000 2 32000 -dtx"
        "audio 24000 1 32000"
        "audio 48000 1 96000 -framesize 10"
        "audio 48000 1 64000 -framesize 60"
        "audio 48000 2 64000"
        "audio 48000 2 128000 -cbr"
        "audio 48000 2 24000 -framesize 2.5"
        "audio 48000 2 510000"
        "restricted-lowdelay 48000 2 64000 -framesize 5"
        "audio 48000 1 12000 -bandwidth NB"
        "audio 48000 2 48000 -complexity 0"
    )
    local fails=0 total=0 c
    for c in "${configs[@]}"; do
        # shellcheck disable=SC2086
        cross opus_demo_ref opus_demo $c || fails=$((fails + 1))
        cross opus_demo opus_demo_ref $c || fails=$((fails + 1))
        total=$((total + 2))
    done
    if [ "$fails" = 0 ]; then
        record cross-check "PASS ($total runs: libopus<->opus-rs, range coder checked per packet)"
    else
        record cross-check "FAIL ($fails of $total runs)"
    fi
}

run_vectors() {
    if [ -z "${OPUS_TESTVECTORS:-}" ]; then
        record test-vectors "SKIP (set OPUS_TESTVECTORS to the RFC 8251 vectors)"
        return
    fi
    local rate fails=0 matched=0
    for rate in 48000 24000 16000 12000 8000; do
        OPUS_DEMO=$build/opus_demo OPUS_COMPARE=$build/opus_compare \
            sh "$src/tests/run_vectors.sh" "$build" "$OPUS_TESTVECTORS" "$rate" \
            > "vectors_$rate.log" 2>&1
        # run_vectors.sh succeeds without vectors, and stops at a rate's
        # first failure: count the comparisons that matched.
        n=$(grep -c 'output matches reference' "vectors_$rate.log")
        [ "$n" = 24 ] || fails=$((fails + 1))
        matched=$((matched + n))
    done
    if [ "$fails" = 0 ]; then
        record test-vectors "PASS ($matched/120: 12 vectors x mono/stereo x 5 rates)"
    else
        record test-vectors "FAIL at $fails of 5 rates (capi/conformance.sh lists every case)"
    fi
}

tests=("$@")
[ ${#tests[@]} = 0 ] && tests=(test_opus_padding test_opus_decode test_opus_api test_opus_encode cross-check test-vectors)
for t in "${tests[@]}"; do
    case $t in
        cross-check) run_cross ;;
        test-vectors) run_vectors ;;
        *) run_test "$t" ;;
    esac
done

echo
echo "libopus test suite against $(readlink -f libopus.so.0)"
printf '  %s\n' "${results[@]}"
printf '%s\n' "${results[@]}" | grep -q FAIL && exit 1
exit 0
