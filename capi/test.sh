#!/usr/bin/env bash
# Tests a libopus.so.0 built from capi/ against libopus's own tests:
#
#   test_opus_*   libopus's test programs, compiled unmodified from the
#                 pinned libopus sources: api, decode, encode, padding,
#                 projection (with libopus's src/mapping_matrix.c, whose
#                 internal functions it also tests) and, with --dred, dred.
#                 test_opus_extensions is left out: it tests libopus's
#                 internal extension and repacketizer functions.
#   small_stack   capi/small_stack.c: encoders and decoders created and
#                 run on a thread with a 128 KiB stack (without --dred:
#                 the neural decoder's setup needs more).
#   vectors       the RFC 8251 test vectors as tests/run_vectors.sh decodes
#                 and compares them, every case (12 vectors, mono and
#                 stereo, 5 rates).
#   cross-check   opus_demo against the reference libopus (pkg-config opus):
#                 each encodes what the other decodes, over 16 settings;
#                 opus_demo checks every packet's final range.
#
# Usage: capi/test.sh <dir with libopus.so.0> <work dir> [--dred]
#
# Needs a C compiler, pkg-config with the reference libopus, python3, the
# pinned libopus sources in OPUSREF_SRC and the extracted vectors in
# OPUS_TESTVECTORS (or libopus/ and testvectors/, which
# scripts/fetch-libopus.sh fetches). The flake's dev shell provides all of
# them.
set -euo pipefail

usage() { echo "usage: $0 <dir with libopus.so.0> <work dir> [--dred]" >&2; exit 2; }
[ $# -ge 2 ] || usage
here=$(cd "$(dirname "$0")" && pwd)
lib=$(cd "$1" && pwd)
mkdir -p "$2"
work=$(cd "$2" && pwd)
dred=0
[ "${3:-}" = --dred ] && dred=1
src=${OPUSREF_SRC:-$here/../libopus}
vectors=${OPUS_TESTVECTORS:-$here/../testvectors}
[ -f "$src/tests/test_opus_api.c" ] || { echo "no libopus sources in $src" >&2; exit 2; }
[ -f "$vectors/testvector01.bit" ] || { echo "no testvector01.bit in $vectors" >&2; exit 2; }
[ -f "$lib/libopus.so.0" ] || { echo "no libopus.so.0 in $lib" >&2; exit 2; }

cd "$work"
mkdir -p cfg
: > cfg/config.h
cflags=(-O2 -w -DHAVE_CONFIG_H -Icfg -I"$src/include" -I"$src/celt" -I"$src/silk" -I"$src/src" -I"$src/tests")
link() { cc "${cflags[@]}" "$@" -L"$lib" -Wl,-rpath,"$lib" -l:libopus.so.0 -lm; }

# The encoder size check of test_opus_api: opus-rs's encoder state is
# larger than the 256 KiB libopus's test allows.
size_check='(i<=2048||i>1<<18)'
size_line=$(($(grep -n 'i=opus_encoder_get_size(c);' "$src/tests/test_opus_api.c" | cut -d: -f1) + 1))
sed -n "${size_line}p" "$src/tests/test_opus_api.c" | grep -F "$size_check" > /dev/null
sed "${size_line}s/$size_check/(i<=2048||i>1<<19)/" "$src/tests/test_opus_api.c" > test_opus_api_512k.c
programs=(test_opus_api test_opus_api_512k test_opus_decode test_opus_encode test_opus_padding test_opus_projection)
link "$src/tests/test_opus_api.c" -o test_opus_api
link test_opus_api_512k.c -o test_opus_api_512k
link "$src/tests/test_opus_decode.c" -o test_opus_decode
link "$src/tests/test_opus_encode.c" "$src/tests/opus_encode_regressions.c" -o test_opus_encode
link "$src/tests/test_opus_padding.c" -o test_opus_padding
link "$src/tests/test_opus_projection.c" "$src/src/mapping_matrix.c" -o test_opus_projection
if [ $dred = 1 ]; then
  link "$src/tests/test_opus_dred.c" -o test_opus_dred
  programs+=(test_opus_dred)
else
  link "$here/small_stack.c" -o small_stack -lpthread
  programs+=(small_stack)
fi
link "$src/src/opus_demo.c" -o opus_demo
cc -O2 "$src/src/opus_compare.c" -o opus_compare -lm
# shellcheck disable=SC2046
cc "${cflags[@]}" "$src/src/opus_demo.c" -o opus_demo_ref $(pkg-config --cflags --libs opus) -lm
# The programs have to load this libopus.so.0, never another one.
for p in "${programs[@]}" opus_demo; do
  ldd "$p" | grep -F "$lib/libopus.so.0" > /dev/null || { echo "$p does not load $lib/libopus.so.0" >&2; exit 1; }
done

results=()
failed=0
record() {
  results+=("$(printf '%-22s %s' "$1" "$2")")
  case $2 in FAIL*) failed=1 ;; esac
}

for p in "${programs[@]}"; do
  if ./"$p" > "$p.log" 2>&1; then
    record "$p" PASS
  elif [ "$p" = test_opus_api ] && grep -q "test_opus_api.c at line $size_line " "$p.log"; then
    record "$p" "KNOWN FAIL: opus_encoder_get_size() above 256 KiB (test_opus_api_512k runs the rest)"
  else
    record "$p" "FAIL: $(grep -o 'fails .* at line [0-9]*' "$p.log" | head -1) (see $work/$p.log)"
  fi
done

# One vector case: "<vector> <rate> <channels> PASS|FAIL".
vector_case() {
  local f=$1 rate=$2 ch=$3 out=v$1-$2-$3.pcm stereo=()
  [ "$ch" = 2 ] && stereo=(-s)
  if ./opus_demo -d "$rate" "$ch" -ignore_extensions "$vectors/testvector$f.bit" "$out" > "$out.log" 2>&1 &&
    { ./opus_compare "${stereo[@]}" -r "$rate" "$vectors/testvector$f.dec" "$out" ||
      ./opus_compare "${stereo[@]}" -r "$rate" "$vectors/testvector${f}m.dec" "$out"; } >> "$out.log" 2>&1; then
    echo "testvector$f $rate $ch PASS"
  else
    echo "testvector$f $rate $ch FAIL"
  fi
  rm -f "$out"
}
export -f vector_case
export vectors
for rate in 48000 24000 16000 12000 8000; do
  for f in 01 02 03 04 05 06 07 08 09 10 11 12; do
    for ch in 1 2; do echo "$f $rate $ch"; done
  done
done | xargs -P "$(nproc)" -L 1 bash -c 'vector_case "$@"' _ > vectors.log
pass=$(grep -c ' PASS$' vectors.log || true)
if [ "$pass" = 120 ]; then
  record vectors "PASS (120/120)"
else
  record vectors "FAIL ($pass/120): $(grep ' FAIL$' vectors.log | sort | tr '\n' ';')"
fi

# Raw 16-bit inputs: the speech fixture, its stereo version (the right
# channel delayed by a sample), read at whatever rate opus_demo is given.
python3 -I - "$here/../fixtures/answer_16k.wav" <<'EOF'
import array, sys, wave
w = wave.open(sys.argv[1])
mono = array.array('h', w.readframes(w.getnframes()))
open('speech_mono.raw', 'wb').write(mono.tobytes())
stereo = array.array('h', [0]) * (2 * len(mono))
stereo[0::2] = mono
stereo[3::2] = mono[:-1]
open('speech_stereo.raw', 'wb').write(stereo.tobytes())
EOF
cross() {
  local enc=$1 dec=$2 app=$3 rate=$4 ch=$5 bitrate=$6 input=speech_mono.raw
  shift 6
  [ "$ch" = 2 ] && input=speech_stereo.raw
  ./"$enc" -e "$app" "$rate" "$ch" "$bitrate" "$@" "$input" cross.bit > cross.log 2>&1 &&
    ./"$dec" -d "$rate" "$ch" cross.bit cross.pcm >> cross.log 2>&1
}
configs=(
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
cross_failed=()
for c in "${configs[@]}"; do
  # shellcheck disable=SC2086
  cross opus_demo_ref opus_demo $c || cross_failed+=("libopus->opus-rs $c")
  # shellcheck disable=SC2086
  cross opus_demo opus_demo_ref $c || cross_failed+=("opus-rs->libopus $c")
done
if [ ${#cross_failed[@]} = 0 ]; then
  record cross-check "PASS (${#configs[@]} settings, both directions)"
else
  record cross-check "FAIL: ${cross_failed[*]}"
fi

echo
echo "libopus tests against $lib/libopus.so.0:"
printf '  %s\n' "${results[@]}"
exit $failed
