#!/usr/bin/env bash
# Runs every reproducer in manifest.tsv and manifest.d/*.tsv against the current build and prints
# whether its bug still reproduces. Build first: cargo +nightly fuzz build
# (and the aarch64 build, see ../DIFFERENTIAL.md, for @aarch64 entries).
# @dnn and @fixed entries need binaries built with --features dnn or fixed and
# OFZ_BUILD=dnn or fixed; such a build parses inputs differently or compares
# with another reference, so then only its own entries run.
# ../tests/regressions.rs replays the same entries in an ordinary cargo test.
set -uo pipefail
cd "$(dirname "$0")"
dir=$(cd .. && cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')
status=0
while IFS=$'\t' read -r id target env pattern desc; do
  [[ $id == \#* || -z $id ]] && continue
  if [[ $target == test:* ]]; then
    # A directed differential test in ../opusref/tests.
    out=$(cd ../opusref && cargo test --release --test "${target#test:}" 2>&1)
    if grep -qF -- "$pattern" <<<"$out"; then r=present; status=1; else r=absent; fi
    printf '%-4s %-8s %s\n' "$id" "$r" "$desc"
    continue
  fi
  [ -f "$id.bin" ] || { printf '%-4s %-8s %s\n' "$id" "no-input" "$desc"; continue; }
  case $target in *@dnn) build=dnn ;; *@fixed) build=fixed ;; *) build=default ;; esac
  if [ "$build" != "${OFZ_BUILD:-default}" ]; then
    printf '%-4s %-8s %s\n' "$id" "skip" "$desc"
    continue
  fi
  target=${target%@dnn}; target=${target%@fixed}
  case $target in
    *@aarch64) run=(qemu-aarch64 "$dir/aarch64-unknown-linux-gnu/release/${target%@aarch64}") ;;
    *) run=("$dir/x86_64-unknown-linux-gnu/release/$target") ;;
  esac
  [ -x "${run[-1]}" ] || { printf '%-4s %-8s %s\n' "$id" "no-build" "$desc"; continue; }
  [ "$env" = - ] && env=
  out=$(env $env "${run[@]}" "$id.bin" 2>&1)
  if grep -qF -- "$pattern" <<<"$out"; then r=present; status=1; else r=absent; fi
  printf '%-4s %-8s %s\n' "$id" "$r" "$desc"
done < <(cat manifest.tsv manifest.d/*.tsv 2>/dev/null)
exit $status
