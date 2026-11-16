#!/usr/bin/env bash
# Runs one differential target in survey mode: findings are grouped by
# signature and saved instead of stopping at the first crash.
#
#   fuzz/survey.sh <name> <target> <seconds> [ENV=VALUE...]
#
# Findings: fuzz/survey/<name>/findings.txt (+ one input file per signature).
# Reproduce one: run the target binary on that file without OFZ_SURVEY.
set -euo pipefail
cd "$(dirname "$0")"
name=$1 target=$2 secs=$3
shift 3
bin=${OFZ_BIN:-$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')/x86_64-unknown-linux-gnu/release/$target}
mkdir -p "survey/$name" "corpus/$name" artifacts
cp "$bin" "survey/$name/$target"
echo $$ > "survey/$name/pid"
exec env "$@" OFZ_SURVEY="$PWD/survey/$name" timeout $((secs + 300)) "survey/$name/$target" \
  -max_total_time="$secs" -max_len=8192 -len_control=0 -artifact_prefix=artifacts/ "corpus/$name" \
  > "survey/$name.log" 2>&1
