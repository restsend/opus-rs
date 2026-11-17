# Regression guards

The chain since upstream `main` (change `rskortqy`) has 215 commits. This
file lists every one of them that changes behaviour, with the test or
reproducer that fails without it. Commits that add features (the neural
network ports, guarded by their bit-exact `opusref/tests/dnn_*.rs`
comparisons), refactor or optimise without a behaviour change, or touch only
the fuzz harness, docs or formatting are not listed.

`scripts/test-all.sh` runs every guard named here: the opus-rs suite, the
opusref tests (float, `--features dnn`, `--features fixed --test
silk_fixed`), the parity example and the regression replay
(`fuzz/tests/regressions.rs`, see `DIFFERENTIAL.md`).

## Classes

The class is the state before this audit:

- **G** guarded: a test in `cargo test` or the opusref tests fails without
  the fix.
- **R** reproducer only: only a `.bin` in `regressions/` showed the bug, in
  `check.sh`. The replay now runs these inputs in an ordinary `cargo test`.
- **F** fuzz-findable: no saved input, but a fuzz oracle covers the code.
- **U** unguarded: nothing failed without the fix. "new:" names the test this
  chain adds; otherwise the reason is given.
- **X** the fix was to a target-specific SIMD kernel that is gone; its
  reproducer stays in the replay.
- **N** no behaviour change.

The "check" column says how the guard was confirmed. "reverted" means the
fix's source diff was reverted on `lmyzrvyq` (the first commit of this test
chain) and the suites ran in this order until one failed: opusref (float),
replay, the opus-rs suite, and for SILK or neural changes the fixed or dnn
opusref tests. "commit message" means the fix's commit names the test and
the revert no longer applies, so the claim was not re-run. Fixes that no
longer revert cleanly were checked by reverting the relevant lines by hand
where noted.

## Counts

150 fix commits: 93 guarded (G), 25 reproducer only (R), 1 fuzz-findable (F), 24 unguarded (U), 3 SIMD kernel removed (X), 4 no behaviour change (N).

## Fixes

| # | change | behaviour | class | guard | check |
| --- | --- | --- | --- | --- | --- |
| 4 | `knnmrzrq` | fix: clamp the bitrate to 750 kb/s per channel, like libopus | U | new: opusref encoder_packet::bitrate_is_capped_at_750_kbps_per_channel (regression 09's input no longer reaches the cap) | new test fails with the 750 kb/s cap of encode_native raised (replay 09 stays absent) |
| 5 | `yytzpzvn` | fix: limit before shifting in silk_lshift_sat32, like libopus | R | regression 03 (diff_kernels silk_lshift_sat32 exact); silk_fixed loud part (via #170) | reverted: fails replay (regression entry) |
| 6 | `zvslzxvn` | fix: use libopus's comb-filter tap constants | G | opusref celt_parity and celt_encoder_parity (the pre-filter runs in every frame); regression 05 (diff_kernels comb_filter exact) | reverted: fails opusref (default_bitrate_is_auto, frames_code_at_most_1275_bytes, free_space_stays_between_range_and_raw_bits, ...) |
| 7 | `zwsqsuqy` | fix: drop the band-0 dynalloc boost above 320 bytes | G | opusref celt_encoder_parity::frames_code_at_most_1275_bytes; regression 08 | reverted: fails opusref (frames_code_at_most_1275_bytes) |
| 8 | `pzwwwmtp` | fix: scale before adding in the AVX2 haar1, like libopus | X | regression 10; the AVX2 haar1 is gone (#39), diff_kernels haar1 exact | commit message |
| 9 | `vqssqnuv` | fix: index the range coder and rate tables with bounds checks | R | regressions 01a-01d (api_args); #![forbid(unsafe_code)] (#46) keeps get_unchecked out. SECURITY | commit message |
| 10 | `wyyslwln` | fix: check frame_size * channels for overflow in decode_fec and loss | R | regressions 17a, 17b (overflow checks of the replay's test profile; reverting the decode-path checked_mul makes 17a present). SECURITY | 17a present with the decode-path checked_mul reverted |
| 11 | `vsqtozxz` | fix: compute silk_sum_sqr_shift as libopus does on every target | X | regressions 15a, 15b@aarch64; one scalar silk_sum_sqr_shift | commit message |
| 12 | `ooskpllp` | fix: drop the NEON celt_fir | X | regression 16@aarch64; NEON celt_fir gone | commit message |
| 13 | `vxtnlutq` | fix: stop clamping float decoder output to [-1, 1] | R | regression 04 | commit message |
| 15 | `mqylzmlv` | fix: report the CELT concealment's final range from Hybrid decode_fec | R | regression 14 | commit message |
| 16 | `ktynvrnv` | fix: let decode_fec check the mode of the last accepted packet | G | opusref fec_after_dtx (13) | commit message |
| 18 | `zzywtuzx` | fix: fade out the CELT layer on Hybrid-to-SILK switches | R | regression 02 | commit message |
| 19 | `swymkprl` | fix: DC-reject and round SILK's input outside VoIP, like libopus | R | regression 06 (reverting the 16-bit rounding makes 06 present; no opusref test fails) | 06 present with the 16-bit rounding truncated |
| 21 | `pypsmsnw` | fix: follow libopus for silent VBR frames and the pre-filter gate | U | no reproducer for the silent-VBR budget (commit message); the parity example's silence+voice signals run that path | commit message |
| 23 | `ksppoyny` | fix: give spreading_decision libopus's weights and arguments | G | opusref celt_parity, parity example (spreading decides every frame); regression 07 | commit message |
| 24 | `lykrpowr` | fix: follow libopus's operation order in the comb-filter overlap | G | opusref celt_parity and celt_encoder_parity | reverted: fails opusref (default_bitrate_is_auto, free_space_stays_between_range_and_raw_bits, pulses_resynthesise_with_reciprocal_sqrt, ...) |
| 25 | `lyzvyqnl` | fix: align silk_lin2log, silk_limit_32 and silk_rshift_round with libopus | F | diff_kernels silk macro oracles; no saved input | commit message |
| 47 | `rlynqznt` | fix: compute exp_rotation's rotation as libopus's float build does | G | diff_kernels exp_rotation exact (fuzz-findable); opusref celt_parity | reverted: fails opusref (pulses_resynthesise_with_reciprocal_sqrt) |
| 48 | `qypmsnyl` | fix: conceal losses with libopus's mode and frame length | R | regressions 14b, 14c, 14d | commit message |
| 52 | `oqwzoyuu` | fix: downsample decoded SILK with libopus's down-FIR | R | regression d1 | reverted: fails replay (regression entry) |
| 53 | `lkvmutwl` | fix: keep the N=1 band's folding source unscaled, as the float build does | R | regression d2 | reverted: fails replay (regression entry) |
| 54 | `kktkmstm` | fix: invert the side of an N=2 stereo band as libopus does | R | regression d3 | reverted: fails replay (regression entry) |
| 55 | `rlolpuxl` | fix: bridge mode transitions with concealment in the previous mode | R | regression d4 | commit message |
| 56 | `vpksznrr` | fix: fade into a 2.5 ms frame from the transition bridge | R | regression d5 | commit message |
| 57 | `yuynryps` | fix: fade out Hybrid's CELT layer in decode_fec of a SILK packet | R | regression d6 | commit message |
| 58 | `spzqxsmn` | fix: conceal a Hybrid CELT layer of at most one byte | G | opusref hybrid_lost_celt_layer (d7) | commit message |
| 59 | `kqusxquq` | fix: conceal in frames of at most the last packet's frame size | R | regression d8 | commit message |
| 60 | `rtlzqmmk` | fix: reject the packets opus_packet_parse_impl rejects | G | opusref packet_limits (d9). SECURITY (packet validation) | commit message |
| 62 | `xpqrzplx` | fix: let each frame of a packet follow the frame before it | R | regression d10 | commit message |
| 63 | `nuyuoxnp` | fix: conceal decode_fec's part of a ToC-only packet in its frame size | R | regression d11 | commit message |
| 64 | `xkpkmyyv` | fix: reset SILK and bridge out of CELT in decode_fec too | R | regression d12 | commit message |
| 65 | `pqlrpqnk` | fix: reject a ToC-only packet of code 2 | G | opusref toc_only_code2 (d14) | commit message |
| 66 | `lylvqkvy` | fix: decode each frame through a port of opus_decode_frame | G | opusref tiny_frames (d13) | commit message |
| 67 | `orqruvtm` | fix: add VERY_SMALL last when CELT accumulates onto SILK | U | rounding of x + m + 1e-30 differs only for x below about 1e-22 in magnitude; no stream reaches it (decoder_exact's Hybrid streams pass with the old order) | decoder_exact passes with the old addition order |
| 68 | `xttyklns` | fix: keep CELT's postfilter on frames that start at band 17 | G | opusref hybrid_postfilter_carry (d15) | reverted: fails opusref (hybrid_frame_keeps_the_postfilter_of_a_redundant_frame) |
| 69 | `rqxxwsyz` | fix: ignore the intensity inversion in a mono decoder | G | opusref mono_inv (d16) | reverted: fails opusref (mono_decoder_ignores_intensity_inversion) |
| 70 | `nrwnsppp` | fix: run the periodic concealment's synthesis filter as celt_iir does | U | new: opusref decoder_exact::celt_concealment_filters_like_celt_iir | new test fails with the old synthesis loop |
| 71 | `qpvzupxq` | fix: fold a concealed overlap with the current postfilter only | G | opusref celt_fold_after_loss (d17) | reverted: fails opusref (frame_after_a_loss_folds_without_crossfade) |
| 72 | `pmukqupt` | fix: leave CELT's output unsaturated, as the float build does | G | opusref celt_no_saturation (d18) | reverted: fails opusref (extreme_celt_frame_is_not_saturated) |
| 73 | `puootvry` | fix: decode a uint past its range as the largest value | G | opusref dec_uint_overflow (d19) | reverted: fails opusref (out_of_range_uint_decodes_as_the_largest) |
| 74 | `ylmnrvls` | fix: compute CELT's 2^x as the float build's celt_exp2 | U | new: opusref decoder_exact::band_gains_use_the_float_celt_exp2 | reverted: fails decoder_exact (new) |
| 75 | `opokrupk` | fix: use libopus's stability limit in silk_lpc_inverse_pred_gain | R | regression d20 | reverted: replay d20 present (check.sh agrees) |
| 76 | `txzlpsrm` | fix: saturate the coefficient update in silk_lpc_inverse_pred_gain | U | the saturating and wrapping subtractions differ only outside the int32 range, which no filter from random coefficients or silk_NLSF2A reaches; opusref inv_pred_gain passes with it reverted | reverted: every suite passes |
| 78 | `kquqtrpq` | fix: DC-reject the RestrictedLowDelay input | G | opusref celt_parity::rld_input_is_dc_rejected | reverted: fails opusref (rld_input_is_dc_rejected, transient_analysis_uses_libopus_inv_table) |
| 79 | `vrnopyyt` | fix: use libopus's static MDCT and FFT twiddles and window | G | opusref celt_parity::static_mode_tables_are_libopus_literals, fft_matches_libopus | reverted: fails opusref (default_bitrate_is_auto, fft_matches_libopus, frames_code_at_most_1275_bytes, ...) |
| 80 | `ymkksmqr` | fix: follow libopus's radix-3 and radix-5 FFT butterflies | G | opusref celt_parity::fft_matches_libopus | reverted: fails opusref (default_bitrate_is_auto, fft_matches_libopus, intensity_stereo_weights_follow_libopus, ...) |
| 81 | `pqtkuywz` | fix: leave the CELT packet's free space to ec_enc_done | G | opusref celt_parity::free_space_stays_between_range_and_raw_bits | reverted: fails opusref (frames_code_at_most_1275_bytes, free_space_stays_between_range_and_raw_bits, input_beyond_65536_is_clipped, ...) |
| 82 | `uvllsmll` | fix: use libopus's transient_analysis inv_table | G | opusref celt_parity::transient_analysis_uses_libopus_inv_table | commit message |
| 83 | `vwmtltmz` | fix: compute celt_log2 and celt_exp2 in double precision | G | opusref celt_parity::band_log_energy_matches_libopus | commit message |
| 84 | `zxsutnwl` | fix: run dynalloc and tf analysis before coarse energy | G | opusref celt_parity::dynalloc_reads_the_previous_frame_energy | commit message |
| 85 | `wnzrxzrt` | fix: bias stable bands' coarse energy by the last frame's error | G | opusref celt_parity::coarse_energy_follows_the_previous_error | commit message |
| 86 | `mlxkuxwp` | fix: take the split angle's Q14 part from a Q30 angle | G | opusref celt_parity::split_angle_is_truncated_from_q30 | reverted: fails opusref (frames_code_at_most_1275_bytes, intensity_stereo_weights_follow_libopus, small_split_angles_match_libopus, ...) |
| 87 | `onuwvonq` | fix: start stereo_itheta's energies at zero | G | opusref celt_parity::silent_split_has_zero_angle | reverted: fails opusref (silent_split_has_zero_angle, small_split_angles_match_libopus) |
| 88 | `mqpwosnt` | fix: evaluate celt_atan_norm in libopus's operation order | G | opusref celt_parity::split_angles_match_libopus | reverted: fails opusref (small_split_angles_match_libopus, split_angles_match_libopus) |
| 89 | `vxovoxlr` | fix: use libopus's celt_atan_norm coefficients | G | opusref celt_parity::split_angles_match_libopus | reverted: fails opusref (split_angles_match_libopus) |
| 90 | `rsuoxxot` | fix: count only budget changes in the coarse energy's badness | G | opusref celt_parity::intra_decision_counts_only_budget_changes | reverted: fails opusref (intra_decision_counts_only_budget_changes, low_rate_stereo_codes_a_mono_stream, stereo_angle_uses_rate_distortion_choice) |
| 92 | `wlxqukuv` | fix: limit the VBR target by dynalloc's maxDepth | G | opusref celt_parity::vbr_target_uses_dynalloc_max_depth | commit message |
| 93 | `ktxzqvum` | fix: update the VBR drift with the float build's alpha | G | opusref celt_parity::vbr_drift_uses_the_float_alpha | commit message |
| 94 | `mplkrmml` | fix: give the 970th frame the VBR alpha 1/990 | U | the 970th frame's alpha changes the drift only for a large miss; 1000-frame runs stay identical (commit message); needs a hook to set vbr_count | reverted: every suite passes |
| 95 | `kmtvntyr` | fix: run the pitch pre-filter on stereo frames | G | opusref celt_parity::stereo_frames_use_the_pre_filter | reverted: fails opusref (default_bitrate_is_auto, dynalloc_cap_follows_libopus, frames_code_at_most_1275_bytes, ...) |
| 96 | `ztkywpxs` | fix: resynthesise the encoder's partitions when asked | G | opusref celt_parity::stereo_angle_uses_rate_distortion_choice (with #97) | reverted: fails opusref (default_bitrate_is_auto, dynalloc_cap_follows_libopus, frames_code_at_most_1275_bytes, ...) |
| 97 | `upywvytw` | fix: choose the stereo angle's rounding by rate-distortion | G | opusref celt_parity::stereo_angle_uses_rate_distortion_choice | reverted: fails opusref (default_bitrate_is_auto, frames_code_at_most_1275_bytes, low_rate_stereo_fades_towards_mono, ...) |
| 98 | `stqtpssw` | fix: fade low-rate stereo towards mono | G | opusref celt_parity::low_rate_stereo_fades_towards_mono | commit message |
| 99 | `kztptqrp` | fix: scale resynthesised pulses by the reciprocal square root | G | opusref celt_parity::pulses_resynthesise_with_reciprocal_sqrt | reverted: fails opusref (default_bitrate_is_auto, pulses_resynthesise_with_reciprocal_sqrt) |
| 100 | `xvmpwqzs` | fix: renormalise folded bands with the reciprocal square root | G | opusref celt_parity::renormalise_uses_reciprocal_sqrt | reverted: fails opusref (renormalise_adds_epsilon_after_the_sum, renormalise_uses_reciprocal_sqrt) |
| 101 | `qrwmqqvl` | fix: add renormalise_vector's epsilon after the sum of squares | G | opusref celt_parity::renormalise_adds_epsilon_after_the_sum | reverted: fails opusref (renormalise_adds_epsilon_after_the_sum) |
| 102 | `mknszrmn` | fix: cap dynalloc boosts like libopus | G | opusref celt_parity::dynalloc_cap_follows_libopus | commit message |
| 103 | `kzzmowyu` | fix: share dynalloc's last rising band between channels | U | no input found: the commit's search, and 96 CELT-level stereo configurations at 8/16/48 kHz (20 frames each) stay identical with it reverted | reverted: every suite passes |
| 104 | `ospkzroy` | fix: give compute_vbr dynalloc's boost total | U | no input found: 30-frame runs of 1050 configurations code the same packets (commit message) | commit message |
| 105 | `yumzukul` | fix: leave dynalloc's boosts unclamped by the band caps | U | the follower's limit of 4 keeps boosts below the 48 kHz caps; no input found (commit message, and the 96-configuration search) | commit message (revert conflicts) |
| 106 | `mnktwpwu` | fix: seed the encoder's resynthesis with the last final range | G | opusref celt_parity::resynthesis_noise_seeds_from_the_last_range | commit message |
| 107 | `psrrzqkv` | fix: detect digital silence in every frame of a multi-frame packet | U | the CELT-only activity rule of encode_frame (peak energy against the frame's energy) already gives silent frames activity 0, so without this the packets do not change; regression n5's input no longer differs | replay and opusref pass with the frame silence flag limited to analysed frames |
| 108 | `wsyyrrom` | fix: pad CBR packets to their full size above 1276 bytes | U | new: opusref encoder_packet::cbr_packets_pad_beyond_1276_bytes (regression n7's input no longer reaches it) | new test fails with the padding limited to 1276 bytes (replay n7 stays absent) |
| 109 | `tuxruokw` | fix: send two-byte CBR PLC packets without a padding flag | R | regression n10 (reverting the flag makes n10 present; no opusref test fails) | n10 present with the padding flag always set; opusref passes |
| 110 | `qyuqtxmx` | fix: use the float build's Hybrid high-band gain | G | unit test silk_rate_tests::high_band_gain_follows_the_celt_share | reverted without its test hunk: fails unit test silk_rate_tests::high_band_gain_follows_the_celt_share |
| 111 | `kmwyvyuu` | fix: patch the transient decision on rising band energies | G | opusref celt_parity::rising_band_energy_patches_the_transient_flag | commit message |
| 113 | `rtqsvqmv` | fix: code low-rate stereo as a mono stream | G | opusref celt_parity::low_rate_stereo_codes_a_mono_stream | commit message |
| 114 | `zwynvxtm` | fix: keep the intensity band the allocation codes | U | no input found: the commit's 48 kHz runs and the 96-configuration search including 8 kHz stereo | reverted: every suite passes |
| 115 | `vtsukqwt` | fix: scale stereo_analysis's M/S sum by 0.707107 | U | regression e1's input no longer reaches it (it predates quqvsrlu); 216 CELT-level stereo configurations near the dual/joint boundary stay identical with it reverted. A direct test needs the stereo_analysis hook (see report) | reverted: every suite passes (e1 absent) |
| 116 | `ktvzmpxx` | fix: start stereo_analysis's sums at EPSILON | U | no input found: only near-silent low bands; the 96-configuration search stays identical. A direct test needs the stereo_analysis hook | commit message |
| 117 | `rlkluqnr` | fix: start the encoder's pre-filter period at 0 | R | regression e2; tests/issue_38_48k_delay.rs | reverted: fails replay (regression entry) |
| 119 | `qqmuslrz` | fix: spread normally at complexity 0 when the spread symbol does not fit | U | the spread symbol misses only with fewer than 4 bits left, which leaves no pulses to rotate; 192 complexity-0 CELT configurations with 2-13 bytes stay identical with it reverted | reverted: celt-level search identical |
| 120 | `rmuwsyxw` | fix: code 2.5 ms stereo frames as M/S | G | opusref celt_parity::short_stereo_frames_use_mid_side | reverted: fails opusref (short_stereo_frames_use_mid_side) |
| 121 | `prtynpoz` | fix: bound a negative VBR byte target by min_allowed | U | a negative VBR target needs a low-rate Hybrid frame; 48 CELT-level Hybrid configurations (VBR, 1-12 kb/s) stay identical with it reverted | reverted: celt-level search identical |
| 122 | `omspokzy` | fix: give the coarse energy nbAvailableBytes | U | new: opusref celt_encoder_parity::coarse_energy_counts_the_bytes_left_in_the_range_coder (CELT after bits already in the range coder; in Opus Hybrid frames the 4 coded bands leave only the intra test at complexity below 4) | reverted: the new test fails; it passes before and after |
| 124 | `rulzrrqt` | fix: initialise the CELT encoder through its reset | R | regression e3 | reverted: fails replay (regression entry) |
| 125 | `nwvmxxqy` | fix: restart bands outside start..end from zero energy | U | new: opusref celt_encoder_parity::bands_beyond_the_end_band_restart_from_zero_energy (regression e4's input no longer differs) | reverted: the new test fails from frame 9 on; every other suite passes (e4 absent) |
| 126 | `suutuxry` | fix: clear the LPC coefficients before the recursion | U | new: tests/celt_lpc_coefficients.rs (both tests) | reverted: both new tests fail |
| 141 | `ktrsyvvq` | fix: write decode_core's loss-recovery tap and lag into the decoder control | G | opusref dnn_osce (--features dnn) | reverted: fails opusref --features dnn (osce_bwe_matches_libopus, osce_matches_libopus) |
| 151 | `llyuszwl` | fix: make the osce feature turn on deep-plc | U | Cargo feature wiring; the script builds and tests with deep-plc,dred,osce together | commit message |
| 157 | `tssulprx` | multistream: reject the layouts libopus rejects | G | unit test multistream_decoder_rejects_invalid_layouts; regression m1. SECURITY (layout validation) | reverted: replay m1 present (its unit test is part of the reverted diff) |
| 158 | `srkknzpy` | multistream: limit frame_size to 120 ms | G | opusref multistream::concealment_is_limited_to_120_ms (m2) | reverted: fails opusref (concealment_is_limited_to_120_ms) |
| 159 | `ttkszqty` | multistream: validate the whole packet before decoding | G | opusref multistream *_is_invalid (m4); regression m3. SECURITY (packet validation) | commit message |
| 160 | `uxmsxwmp` | multistream: reject self-delimited code-3 packets over 120 ms | G | unit test split_code3_over_120_ms_is_error. SECURITY (packet validation) | reverted without its test hunk: fails unit test split_code3_over_120_ms_is_error |
| 161 | `ywxwzonp` | multistream: conceal every stream on an empty packet | G | opusref multistream::an_empty_packet_conceals_every_stream (m5); regression m6 | commit message |
| 162 | `xwxnzxpk` | multistream: add decode_fec and final_range | G | opusref multistream::fec_recovers_every_stream (m7) | commit message |
| 163 | `zwwpwywv` | opus decoder: zero a CELT-only frame skipped for the bandwidth extension | G | opusref dnn_bwe_celt (--features dnn) | reverted: fails opusref --features dnn (celt_frames_after_bbwe_are_zeros) |
| 164 | `kxrsxztx` | dnn: write the Burg backward index as n + 1 - k | R | regression m8@dnn (replay with --features dnn). SECURITY (overflow panic) | reverted: fails replay --features dnn (regression entry) |
| 165 | `owykwowr` | celt plc: remove a temporary trace | N | removes a trace; no behaviour | commit message |
| 167 | `usyvzzyk` | silk: run the variable high-pass smoother once per frame | G | opusref silk_fixed (--features fixed) | reverted: fails opusref --features fixed (stereo_to_mono_switches_match_fixed_point_libopus) |
| 168 | `tpqmxonr` | silk: apply harmonic noise shaping in unvoiced NSQ frames with a lag | G | opusref silk_fixed (--features fixed) | reverted: fails opusref --features fixed (silk_packets_match_fixed_point_libopus, stereo_to_mono_switches_match_fixed_point_libopus) |
| 169 | `ymnzxwzv` | silk: accept stage-2 pitch candidates with libopus's lag range test | G | opusref silk_fixed (--features fixed) | reverted: fails opusref --features fixed (silk_packets_match_fixed_point_libopus) |
| 170 | `vzpooslt` | silk: saturate the initial noise-shaping gains like silk_LSHIFT_SAT32 | G | opusref silk_fixed (--features fixed) | reverted: fails opusref --features fixed (silk_packets_match_fixed_point_libopus) |
| 171 | `rsvymwwl` | silk: give the stereo prefill libopus's frame target rate | G | unit test stereo_prefill_targets_a_10_ms_frame_less_the_reservoir | commit message |
| 172 | `osklpyrs` | silk: restore the mid's low-pass state into the side at a stereo prefill | G | unit test stereo_prefill_gives_the_side_the_mids_low_pass_state | commit message |
| 173 | `tsylkouv` | silk: reject a partial packet in the mono encoder | G | unit test encode_rejects_a_partial_packet | commit message |
| 174 | `rlpulxxp` | silk: take libopus's toMono input in the stereo encoder | G | unit test to_mono_collapses_the_stereo_width | commit message (revert conflicts) |
| 175 | `kksqyrwn` | bands: take amp2Log2 of the band energy without a floor | R | regression b1 | commit message |
| 176 | `zusmrroz` | celt: compute bandLogE below the start band in Hybrid frames | G | opusref celt_encoder_parity::hybrid_dynalloc_sees_the_low_band_energies | reverted: fails opusref (band_log_energy_matches_libopus, hybrid_dynalloc_sees_the_low_band_energies) |
| 177 | `mqrznpvp` | celt: size the frame from whole bytes, at most 1275, at the CBR rate | G | opusref celt_encoder_parity::cbr_frames_take_the_rate_bytes, frames_code_at_most_1275_bytes | commit message |
| 178 | `qsutxylo` | celt: sum plain squares in transient_analysis | G | opusref celt_encoder_parity::quiet_onsets_weigh_the_epsilon_like_libopus | reverted: fails opusref (quiet_onsets_weigh_the_epsilon_like_libopus) |
| 179 | `omomzwxz` | bands: weigh intensity stereo channels as libopus does | G | opusref celt_encoder_parity::intensity_stereo_weights_follow_libopus | reverted: fails opusref (intensity_stereo_weights_follow_libopus) |
| 180 | `skzslxqw` | celt: clip input beyond 65536 before the pre-emphasis | G | opusref celt_encoder_parity::input_beyond_65536_is_clipped | commit message |
| 181 | `ztzylsks` | celt: test tone_lpc's singularity on the product r00*r11 | G | unit test celt::tests::tone_lpc_tests_singularity_on_the_product | commit message |
| 182 | `wtsyzloq` | celt: limit tone_lpc's coefficients where libopus does | G | unit test celt::tests::tone_lpc_keeps_quotients_below_two | commit message |
| 183 | `swlqrrpo` | celt: round dynalloc's tone bin in double precision | G | unit test celt::tests::dynalloc_tone_bin_rounds_in_double | commit message |
| 184 | `xlzptztl` | celt: compute a pure tone's comb-filter period in double | G | unit test celt::tests::tone_period_rounds_in_double | commit message |
| 185 | `poznksup` | celt: subtract tf_estimate's 0.139 in double | G | unit test celt::tests::tf_estimate_subtracts_in_double | commit message |
| 186 | `lxpsktqm` | pitch: compute the pitch gain without a sign guard | G | opusref pitch_parity::pitch_search_and_doubling_match_libopus | reverted: fails opusref (doubling_keeps_the_running_energy, pitch_search_and_doubling_match_libopus) |
| 187 | `pksnoxzk` | pitch: keep remove_doubling's running energy unclamped | G | opusref pitch_parity::doubling_keeps_the_running_energy | reverted: fails opusref (doubling_keeps_the_running_energy) |
| 188 | `qzszwppn` | pitch: leave fine-search lags outside the candidates at 0 | G | opusref pitch_parity (both tests) | reverted: fails opusref (concealment_pitch_search_matches_libopus, doubling_keeps_the_running_energy, pitch_search_and_doubling_match_libopus) |
| 189 | `osvopnyo` | bands: add the 1e-27 floor after the band's sum of squares | G | opusref celt_parity::band_energies_add_the_floor_after_the_sum | reverted: fails opusref (band_energies_add_the_floor_after_the_sum) |
| 190 | `uomrxnvk` | celt: start a new CELT encoder at complexity 5 | G | opusref celt_encoder_parity::default_complexity_is_5 | reverted: fails opusref (default_complexity_is_5) |
| 191 | `oxzozqrq` | celt: decide a silent frame's VBR shrink from the encoder's rate | G | unit test celt::tests::silent_frames_shrink_under_vbr_only | commit message |
| 192 | `qskxmozz` | celt: drop the encoder's resynthesis into an unread buffer | N | removes an unread resynthesis; no packet changes | commit message |
| 193 | `vkskwvpv` | celt: ignore out-of-range CELT settings as libopus's ctls do | G | unit test celt::tests::out_of_range_settings_are_ignored | commit message |
| 194 | `uknwvsto` | encoder: split the packet and frame levels as opus_encode_native does | N | restructuring; a 4320-configuration sweep hashes the same (commit message) | commit message |
| 195 | `wxmnuorq` | encoder: code each packet at libopus's packet bitrate | G | opusref encoder_packet (a1, a2) | commit message |
| 196 | `xtllsypt` | encoder: default to libopus's automatic bitrate, add BITRATE_AUTO/MAX | G | opusref encoder_packet::default_bitrate_is_auto (a3), bitrate_max_fills_the_buffer | commit message |
| 197 | `xkknkkys` | encoder: floor the small-budget CELT threshold to whole bytes | G | opusref encoder_packet (a4) | reverted: fails opusref (small_buffer_threshold_floors_to_bytes) |
| 198 | `vwktowxz` | encoder: start at Hybrid fullband like libopus | G | opusref encoder_packet::initial_bandwidth_is_fullband (a5) | reverted: fails opusref (initial_bandwidth_is_fullband, starved_multiframe_packets_are_toc_only, toc_only_packets_keep_the_previous_config) |
| 199 | `upwkrkkr` | encoder: send ToC-only packets before the packet's decisions | G | opusref encoder_packet (a6, a7, a8) | commit message |
| 200 | `mvvvkuuk` | encoder: decide the stereo-to-mono delay per packet before the bandwidth | G | opusref encoder_packet (a9) | commit message |
| 201 | `mnqpuvqr` | encoder: code stereo input in one SILK channel when the stream is mono | G | opusref encoder_packet (a10); tests/issue_73_silk_vbr_rate.rs, hybrid_cbr_silk_rate.rs hashes | commit message |
| 202 | `xxzxmtkx` | encoder: fade Hybrid CELT input to mono while SILK delays the switch | G | unit test silk_rate_tests::hybrid_fades_to_mono_while_silk_delays_the_switch | reverted without its test hunk: fails its unit test |
| 203 | `oxonrrxu` | encoder: prefill SILK in every frame of a multiframe packet after CELT | G | tests/hybrid_cbr_silk_rate.rs hybrid_celt_cbr_switches_keep_silk_like_libopus | reverted: fails suite (hybrid_celt_cbr_switches_keep_silk_like_libopus) |
| 204 | `trutqovk` | encoder: keep the maximum bandwidth when set_max_bandwidth gets Auto | G | opusref encoder_packet::max_bandwidth_ignores_auto (a12) | reverted: fails opusref (max_bandwidth_ignores_auto) |
| 205 | `rvpypqxy` | encoder: size the fallback packet of a busted frame like libopus | U | no reproducer: busted frames with redundancy did not occur in the commit's sweeps | commit message |
| 206 | `wmompwrv` | encoder: set SILK up for every packet as libopus does | G | tests/issue_73_silk_vbr_rate.rs silk_follows_a_change_of_complexity | reverted: fails suite (silk_follows_a_change_of_complexity) |
| 207 | `vtovpluk` | encoder: tell CELT the depth of the input | G | opusref encoder_packet::celt_codes_16_bit_input_at_16_bits (a13) | reverted: fails opusref (celt_codes_16_bit_input_at_16_bits) |
| 208 | `uupktxwr` | encoder: round float VoIP input to 16 bits as FLOAT2INT16 does | G | tests/issue_73_silk_vbr_rate.rs float_voip_input_rounds_like_libopus | reverted: fails suite (float_voip_input_rounds_like_libopus, silk_concealment_extrapolates_the_signal) |
| 209 | `ylptovnr` | encoder: control SILK without a bandwidth switch after a prefill | U | no reproducer: rate-switching streams match fixed-point libopus with and without it (commit message) | commit message |
| 210 | `mxrromnu` | encoder: compare CELT-only activity in float as the float build does | U | a single f32 value per frame energy decides it; no reproducer (commit message) | commit message |
| 211 | `ylqzmlwm` | encoder: hand SILK the toMono flag while a switch to mono waits | G | opusref silk_fixed::stereo_to_mono_switches_match_fixed_point_libopus (--features fixed) | reverted: fails opusref --features fixed (stereo_to_mono_switches_match_fixed_point_libopus) |
| 213 | `sztrnrtw` | silk: zero the padding pulses of a 10 ms frame at 12 kHz | G | opusref silk_frame_switch_range (s1) | reverted: fails opusref (ten_ms_silk_frames_after_longer_frames_keep_the_range) |
| 214 | `ppnxtost` | decoder: reject an out-of-range complexity with error::BAD_ARG | R | regression x1 (diff_multistream `ctl`) | x1 present before the fix |
| 215 | `qoyyxyzs` | dred: report the whole concealment as the last packet duration | G | opusref dnn_dred_codec::dred_decode_i16_matches_libopus (--features dnn, x2) | fails before the fix |
| 216 | `qvrtskxs` | range_coder: Laplace coding in libopus's unsigned arithmetic | N | for the arguments 217 accepts the arithmetic is unchanged; it differed only above fs 32736, which 217 refuses | its own test failed before it, until 217 |
| 217 | `nnwwmsyx` | range_coder: refuse Laplace arguments libopus's coder can loop on | G | opusref laplace::laplace_refuses_arguments_libopus_loops_on (x4) | reverted: fails opusref (laplace_refuses_arguments_libopus_loops_on) |

## Reproducers that no longer reach their bug

`diff_encoder`'s input layout changed in `quqvsrlu` ("vary frame duration,
output budget and entry point per segment"), after these inputs were
saved. With their fix reverted on `lmyzrvyq`, these entries stay absent;
directed tests replace them where one could be found: 09
(`encoder_packet::bitrate_is_capped_at_750_kbps_per_channel`), n7
(`encoder_packet::cbr_packets_pad_beyond_1276_bytes`) and e4
(`celt_encoder_parity::bands_beyond_the_end_band_restart_from_zero_energy`).
e1 has no replacement (see #115), and n5's fix no longer changes packets
(see #107). Entries 06, n10, e2 and e3, saved before that change too,
still show their bug when their fix is reverted. n3, n8 and n9 were recorded as divergences without a named fix.

## Reproducers that need another build

- m8 (`diff_decoder@dnn`): `cargo test --features dnn --test regressions`.
- n2, n4 (`diff_encoder@fixed`): the VoIP SILK input of `silk-fixed`
  follows fixed-point libopus by design; absent against the fixed-point
  reference, `cargo test --features fixed --test regressions` (the fuzz
  crate's `fixed` turns `silk-fixed` on).
- 12, 15b, 16 (`@aarch64`): NEON kernels that are gone; they run only on
  aarch64 (check.sh with the qemu build, see `DIFFERENTIAL.md`).
- n1 (`stack-buffer-overflow`): needs AddressSanitizer, so only check.sh
  can show it; the AVX2 kernel it read past is gone.

