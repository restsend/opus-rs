#!/usr/bin/env bash
# Fetches the libopus revision opus-rs tracks, with its neural model data,
# into libopus/ at the repository root: the default source of the reference
# build in fuzz/opusref and of capi/test.sh (OPUSREF_SRC overrides it); and
# the RFC 8251 test vectors into testvectors/ (OPUS_TESTVECTORS overrides
# it). Needs git, curl or wget, sha256sum or shasum, and tar; the vectors
# need curl and sha256sum.
set -euo pipefail

rev=f1afc4222aff7c5456d27581ec618ae64158fbe1
cd "$(dirname "$0")/.."
if [ ! -d libopus ]; then
  git init -q libopus
  git -C libopus fetch -q --depth 1 https://github.com/xiph/opus "$rev"
  git -C libopus checkout -q FETCH_HEAD
fi
if [ "$(git -C libopus rev-parse HEAD)" != "$rev" ]; then
  echo "libopus/ is not at $rev; remove it and run this script again" >&2
  exit 1
fi
cd libopus
# The model data archive autogen.sh names, checked against its sha256.
model=$(sed -n 's/^dnn\/download_model.sh "\([0-9a-f]*\)"$/\1/p' autogen.sh)
dnn/download_model.sh "$model"
cd ..
if [ ! -f testvectors/testvector01.bit ]; then
  curl -sSfL -o testvectors.tar.gz https://opus-codec.org/static/testvectors/opus_testvectors-rfc8251.tar.gz
  echo "6b26a22f9ba87b2b836906a9bb7afec5f8e54d49553b1200382520ee6fedfa55  testvectors.tar.gz" | sha256sum -c -
  mkdir -p testvectors
  tar -xzf testvectors.tar.gz -C testvectors --strip-components=1
  rm testvectors.tar.gz
fi
