# Weights of the libopus neural network models

Each `.bin` file holds the arrays of one model in libopus's weight blob
format (the format `OPUS_SET_DNN_BLOB` reads). `src/dnn/builtin.rs` embeds
them with `include_bytes!`.

| File | Model | libopus array list |
|---|---|---|
| `pitchdnn.bin` | PitchDNN (pitch estimator of the LPCNet features) | `pitchdnn_arrays` |
| `fargan.bin` | FARGAN (vocoder of deep PLC and DRED) | `fargan_arrays` |
| `plc.bin` | PLC model (feature prediction of deep PLC) | `plcmodel_arrays` |
| `dred_rdovae_enc.bin` | RDOVAE encoder (DRED) | `rdovaeenc_arrays` |
| `dred_rdovae_dec.bin` | RDOVAE decoder (DRED) | `rdovaedec_arrays` |
| `lace.bin` | LACE (SILK enhancer, decoder complexity 6) | `lacelayers_arrays` |
| `nolace.bin` | NoLACE (SILK enhancer, decoder complexity 7+) | `nolacelayers_arrays` |
| `bbwenet.bin` | BBWENet (SILK bandwidth extension) | `bbwenetlayers_arrays` |

## Provenance

- Source: the model data of libopus 1.6.1 and of main at f1afc4222a,
  `opus_data-a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e.tar.gz`
  (SHA-256 a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e,
  the archive `autogen.sh` downloads). The opus-1.6.1 release tarball ships
  its `dnn/*_data.c` files without the float debug copies of the int8 layers
  (the `#ifndef DISABLE_DEBUG_FLOAT` arrays); built with DISABLE_DEBUG_FLOAT,
  libopus's default, both give the same weights.
- Generation: the reference build of `fuzz/opusref` (feature `dnn`)
  compiles those data files as libopus does by default (DISABLE_DEBUG_FLOAT,
  so int8 layers carry only their int8 weights), and
  `fuzz/opusref/dnn_shim.c` writes each array list with the record writer
  of libopus's `dnn/write_lpcnet_weights.c`. Regenerate with
  `OPUSREF_WRITE_DNN_WEIGHTS=1 cargo test -p opusref --features dnn --test dnn_weights`
  (from `fuzz/`).
- Checks: `fuzz/opusref/tests/dnn_weights.rs` compares the embedded blobs
  byte for byte with the arrays compiled into the reference, and the unit
  test in `src/dnn/builtin.rs` compares their SHA-256 with `SHA256SUMS`.
  The blobs are also identical to those of opus1 (commit 13eee3e,
  `dnn-weights/`), which were written by a separate generator.

The weights are part of libopus and covered by its BSD licence (see
`COPYING` at the repository root).
