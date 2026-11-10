# PR Review — 34 open PRs (#77–#114), focus: bit-exactness & performance regressions

Date: 2026-11-10 · Reviewer pass: static diff review vs libopus 1.6.1 C (opusic-sys 0.7.3) + local execution.

## What was verified empirically

| Check | Result |
|---|---|
| `cargo test --release` on `main` (75bd9e9) | PASS |
| `cargo test --release` on pr/113 (stack #77–#113) | PASS |
| `cargo test` (debug, overflow checks) on pr/113 | PASS |
| `cargo test --release` on pr/114 (#114 rebase) | PASS (1 load-induced flake, see S) |
| **Merge pr/113 → local main** + cherry-pick #114, full suite | **PASS after test fix** (see E) |
| Criterion benches main vs pr/113 vs pr/114 | see Performance |
| Rust-vs-C encode timing harness (release) | see Performance |
| `cargo clippy --all-targets` on stack head | 6 new minor warnings (repo bar is zero) |
| no_std build (`scripts/check_no_std.sh`) | not run (embedded targets absent); new `analysis.rs`/`mlp.rs` contain no `std::` refs, custom `logd`/`fast_atan2f`, `f64::sqrt` is `core` — looks clean, verify in CI |

## Findings to fix before merge (bit-exactness)

### A. [MAJOR] TF analysis missing the tonality gate — in stack head (`src/celt.rs:2910`, introduced by the series, cf. #96)
C (`celt/celt_encoder.c:2242`):
`enable_tf_analysis = effectiveBytes>=15*C && !hybrid && complexity>=2 && !lfe && toneishness < QCONST32(.98f, 29)`
Rust: `effective_bytes >= 15 * channels && !hybrid && self.complexity >= 2` — **no `toneishness < 0.98` term**.
On strongly tonal frames C skips `tf_analysis` and forces `tf_res = isTransient, tf_select = 0`; the port runs the full analysis → different bits on exactly the tonal-music class, plus extra per-frame compute (visible in the 440 Hz sine bench, +12–22 %).

### B. [MAJOR] #109: analysis always fed `lsb_depth = 24` (`src/lib.rs` `run_analysis`)
C's int16 API passes `lsb_depth = 16` into `run_analysis` (`opus_encoder.c` int16 entry); only the float API passes 24. The analysis noise floor `5.7e-4/(1<<(lsb_depth-8))` (`analysis.c:779`) is therefore **256× lower** in the port for `encode_i16` input → quiet 16-bit signals get wrong `detected_bandwidth`/bandwidth decisions. Fix: 16 for `PcmInput::I16`, 24 for `F32`.

### C. [MAJOR] #102/#111: `is_silence` not computed below complexity 7 / 16 kHz (`run_analysis` else-branch returns `(zeroed, false)`)
C computes `is_silence = is_digital_silence(...)` **unconditionally** (`opus_encoder.c:1246`); the complexity/Fs gate only wraps the analysis run. Consequences in the port: digital silence at complexity < 7 or Fs < 16 kHz (8/12 kHz VoIP!) never takes libopus's ms-based generalized DTX (`decide_dtx_mode`), falls into SILK-VAD DTX with a different cadence/ToC; also `peak_signal_energy` is never updated when the analysis is off/invalid (C updates on `!valid || prob > THR`, `opus_encoder.c:1312`).

### D. [MAJOR] #100: multiframe packet framing diverges byte-wise from C's repacketizer
Rust `encode_multiframe` always emits code 2 for nb==2 and code 3 **with the V flag + per-frame size bytes**. C (`repacketizer.c:194–201, 243–252`) emits **code 1** when count==2 with equal lengths, and code 3 **non-VBR** (`count` byte, no sizes) when all frames are equal. Under CBR all sub-frames are padded equal → **every CBR 40/60/80/100/120 ms packet differs byte-wise from libopus** (valid, decodes identically — invisible to decode-level tests; the in-code comment acknowledges the divergence). The all-DTX path in the same PR gets this right (code 1 / non-VBR code 3), so the convention exists.

## Merge mechanics

### E. Stack does not merge onto local main as-is; #71 fix supersession verified
- Local `main` has 2 unpushed commits (19e4cfc #71 fix + release notes) that the PRs predate.
- `git merge pr/113` → conflicts in `src/lib.rs` (3) and `README.md`; pr/77 and pr/114 alone also conflict.
- Resolution: take the stack side — verified it **supersedes** 19e4cfc: the stack's `decide_fec` is fed `Some(mode)` (decided mode) with the mode-aware `compute_equiv_rate`, and the SILK↔Hybrid fixup runs after the FEC bandwidth walk (#77 + #100 per-packet decision).
- After merge, `tests/issue_71_hybrid_fec_div_zero.rs` fails (24 kHz/12 kb/s: cfg 0 vs cfg 12) — **not an encoder regression**: #113 adds auto-bandwidth to the Rust encoder while that test still relies on the old "derive bandwidth from API rate" behavior and forces the C side to SWB/FB. Fix: mirror `enc.set_bandwidth(Superwideband/Fullband)` in `rs_packets` (one line, verified — suite then fully green, including all 5 #71 tests + cherry-picked #114).
- #114's branch is a rebase that drops #84–#88 and #95–#113 from its history; the commit itself cherry-picks cleanly onto the merged stack (verified) — merge the series first, then #114.

## Per-PR verdicts

| PR | Verdict | Notes |
|---|---|---|
| #77 | superseded-by-stack | Own commit = series base; do **not** merge alone (conflicts with 19e4cfc; missing mode-aware `compute_equiv_rate` at first call site). Its test has no C-oracle comparison. |
| #78 fuzz decoder | LGTM | Real stateful sequence fuzzer; [MINOR] no differential decode vs C oracle. |
| #79 SILK down-FIR | LGTM | Matches `resampler.c:101–158`; [NIT] test asserts SNR>40 dB where bit-exact equality is achievable. |
| #80 two-bin inversion | LGTM | Matches `bands.c:1559–1568` (flip all N, skip only merge). |
| #81 multistream layout | LGTM | Matches `opus_multistream.c:41–54`; [NIT] `channels == 0` still accepted (C rejects <1). |
| #82 single-bin folding | LGTM | Float `SHR32` is a no-op (`arch.h:313`) — storing unscaled is exactly right. |
| #83 2-byte CBR padding | LGTM | Byte-exact vs `opus_packet_pad_impl` for all sizes (agent compiled C to confirm). |
| #84 range-dec clamp | LGTM | Exact port of `ec_dec_uint` clamp; unreachable on valid streams. |
| #85 CELT last raw bits | LGTM | Removal restores `ec_enc_done` ownership; strictly less work. |
| #86 12 kHz CELT as WB | LGTM | Matches `opus_encoder.c:1680–1682` MB→WB fixup placement. |
| #87 0/1-byte conceal | LGTM | Matches `opus_decoder.c:315–333`; [MINOR] errors where C does `IMIN(frame_size, st->frame_size)`. |
| #88 PLC after redundancy | LGTM in stack | [MINOR, alone-only] `prev_mode` written before concealment (C writes at end) — healed by #106 later in the stack. Best test in series (1.6.1 FFI oracle). |
| #89 final_range() | LGTM in stack | [MINOR, alone-only] encoder missing `rangeFinal ^= redundant_rng` — completed by #97. Decoder side exact. Strong oracle tests. |
| #90 transient flag | LGTM | Exact port incl. `transient_got_disabled`; no `ec_enc_*` between decision and coding → stream-neutral. |
| #91 intensity clamp | LGTM | Restores `celt_encoder.c:2405` exactly; final-range oracle tests. |
| #92 redundancy flag gate | LGTM | Gate `tell+37 <= 8*(n-1)` matches `opus_encoder.c:2352`. |
| #93 SILK VBR prev mode | LGTM | Matches unconditional `prev_mode = st->mode`; completed by #97's `to_celt` rule. |
| #94 hybrid VBR share / SILK parity | LGTM in stack | Dozens of constants verified bit-exact. [MINOR] stereo `s_mid/s_side` state not reset on SILK rate switch (C re-inits encoder); [NIT] prefill-pass switch-room divisor. Switch-redundancy gap in this commit alone is completed by #97 (byte counts 77/69 match C at head). |
| #95 SILK 10 ms pulse tail | LGTM | Faithful zero-fill of the 12 kHz tail; excellent differential unit test. |
| #96 hybrid VBR split | **Fix A** | Everything else verified exact (hybrid rate table, HB gain, weak transients, spread, trim). [MINOR] spread uses `effective_bytes` where C uses `nbAvailableBytes` (only when CVBR reservoir binds). |
| #97 mode-switch redundancy | LGTM | `compute_redundancy_bytes`, signaling, prefill, CELT/SILK restarts all traced line-for-line. [MINOR] multiframe transition bookkeeping only structurally tested; error path can leave `forced_mode` set. |
| #98 VBR constraint + 510 kb/s | LGTM | Exact cap `min(1275>>(3-lm))`, constraint flag per C. |
| #99 2.5/5 ms everywhere | LGTM | Verbatim `opus_encoder.c:1533–1539`. |
| #100 multiframe packets | **Fix D** | Budget math otherwise verified against C. [MINOR] CBR target clamped at 1276 where C allows up to `min(7656, out)`; [MINOR] CBR bitrate snap missing from per-frame budget (1-byte budget differences at odd rates). |
| #101 80–120 ms frames | LGTM in stack | Split table/order matches C; inherits #100's framing. |
| #102 DTX | **Fix C** | SILK VAD plumbing and `decide_dtx_mode` are line-for-line exact; multiframe all-DTX packet form exact. |
| #103 fuzz encoder | LGTM | Good sequence coverage + `final_range` round-trip invariant. [MINOR] no PLC/decode_fec/resets; [MINOR] the range invariant on DTX packets is a Rust-internal convention (C decoder keeps previous range). |
| #104 PLC trace removal | LGTM | Pure removal, provably behavior-neutral. |
| #105 NEON pitch decimation | LGTM | Lane math now matches C exactly (`vld2q` stride-2); old code read wrong lanes; also faster. |
| #106 CELT switch bridge | LGTM | Faithful `pcm_transition` port; fixed 4 divergences of the old bridge. [MINOR] bridge sum `.clamp(-1,1)` where C float doesn't clamp (follows existing convention). |
| #107 CBR SILK rate | LGTM | Strongest PR of the series: whole-packet byte-hash tests vs fixed-point libopus incl. busted-packet padding (compiled-C-verified byte-exact). Removes a range-coder clone + duplicate LBRR encode (real perf win). |
| #108 SILK/Hybrid/CELT decision | LGTM | Term-for-term vs `opus_encode_native` incl. degenerate `mode_music` indexing; mode-sequence equality tests vs live libopus. |
| #109 signal analysis port | **Fix B** | Otherwise outstanding: MLP weights/tables verified element-by-element bit-identical, `fast_atan2f` 0/300k mismatches vs C, op order preserved, save/restore for multiframe analysis. [NIT] `run()` public with debug-panic on `lsb_depth ≥ 32`. |
| #110 CELT VBR target | LGTM | Every restored term matches C op-for-op (stereo saving MIN32 form, floor depth, pitch_change, temporal VBR, trim step integer form). |
| #111 DTX/VAD from analysis | Fix C (shared) | Analysis-enabled path exact incl. `DTX_ACTIVITY_THRESHOLD` strict `<`; see C for the no-analysis regime. |
| #112 speech estimate | LGTM | Exact `voice_ratio`/`voice_est` port incl. f64 promotion points; oracle tests on coded mode. |
| #113 bandwidth selection | LGTM | Decision chain, threshold tables, caps and API semantics faithful (`set_bandwidth` forces, `set_max_bandwidth` caps). [MINOR] tiny-budget TOC-only path advances `self.bandwidth` where C returns before the decision (state carry-over divergence after starved packets). |
| #114 NLSF weights | LGTM | Matches `process_NLSFs.c:67`/`NLSF_encode.c:67`; 902-packet byte-hash oracle vs fixed-point libopus. Cherry-picks cleanly onto merged stack. |

## Performance

End-to-end (criterion, 20 samples, same machine, sequential runs; ns/iter ratio vs main):

| bench | pr/113 (full stack) | pr/114 |
|---|---|---|
| opus_encode_silk 8k/20ms | 1.00 | 1.00 |
| opus_encode_silk 16k/20ms | 1.08 | 1.01 |
| opus_encode_i16 16k (3 cfgs) | 1.07 | 1.00–1.01 |
| silk_encode_real (3 cfgs) | 1.03–1.05 | 1.03–1.05 |
| opus_encode_celt 48k (3 cfgs, sine) | **1.15–1.22** | 1.00–1.01 |
| opus_real (real speech, 5 cfgs) | 1.015–1.028 | 1.009–1.022 |

Rust-vs-C direct timing (release, 1500 frames, min of 3):

| config | main | stack (pr/113) |
|---|---|---|
| 48 kHz Audio 20 ms 64 kb/s | **0.80× C** (53.2 vs 66.2 µs) | **0.95× C** (62.5 vs 66.0 µs) |
| 16 kHz Voip 20 ms 16 kb/s | 1.23× C (126.4 vs 103.1 µs) | 1.33× C (136.6 vs 102.8 µs) |

Reading:
- The added cost is the signal analysis (#109), pitch stage-2 d_srch (#94), tf-analysis terms — **work C also does**; Rust remains *faster than C* at 48 kHz Audio after the stack. Not a parity regression; it narrows the README's "12–28 % faster" claim.
- The 16 kHz VoIP gap vs C (1.23× → 1.33×) is pre-existing and slightly widened — worth its own investigation, separate from these PRs.
- The sine CELT bench (+12–22 %) is inflated by **Fix A** (tf_analysis running on tonal input where C skips it).
- No PR adds allocation; #107/#85/#105 strictly remove work. `no_std`/no-alloc preserved.

## Test-quality themes

1. **Strong additions**: final_range-parity per frame (#89–#93), whole-packet FNV byte-hash vs fixed-point libopus (#94, #107, #114), mode-sequence equality vs live C encoder (#108), ToC-config equality (#113), 902-packet hash (#114).
2. **Weak spots**: SNR-threshold gates (40/60/30/8 dB) where the fixed paths are deterministic and bit-exact equality is achievable (#79/#80/#82/#85/#87/#105/#106); several tests use the older-libopus `opus` crate instead of the 1.6.1 `opusic_sys` FFI; no full-packet byte comparison vs a **float** C encoder for the CELT/hybrid/multiframe paths — which is exactly how Fix D stayed invisible.
3. `tests/high_bitrate_interop_test.rs` shells out to ffmpeg without validating subprocess output; it flaked once under 3-way parallel `cargo test` load (re-ran clean 3×). Add output-length/completeness validation or a retry.
4. Clippy: 6 new minor warnings on the stack head (repo bar is zero).

## Recommended merge plan

1. Fix A (celt.rs one-liner: add `toneishness < 0.98`), Fix B (lsb_depth by input type), Fix C (compute `is_digital_silence` before the complexity/Fs gate + peak-energy condition), Fix D (code 1 for equal-pair, non-VBR code 3 for equal multi) — each is small and independent of merge order.
2. Merge the series via #113 (widest branch) or top-down; resolve `src/lib.rs`/`README.md` by taking the stack side (supersedes unpushed 19e4cfc — verified).
3. Update `tests/issue_71_hybrid_fec_div_zero.rs::rs_packets` to force `set_bandwidth` mirroring the C side (one line; without it the merge fails the #71 oracle tests).
4. Cherry-pick #114 onto the merged result (branch conflicts; commit applies cleanly).
5. Close #77 as superseded (its commit is the series base); merge #78–#83 independently — all LGTM.
6. Follow-ups: 16 kHz VoIP perf gap; differential fuzzing vs the C oracle (#78/#103); migrate SNR-gated tests to sample-exact; opusic-sys-only oracle policy.
