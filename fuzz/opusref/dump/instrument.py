#!/usr/bin/env python3
"""Copies a libopus source tree and adds OFZ_DUMP dumps to celt_encoder.c,
at the points where opus-rs's CeltEncoder dumps the same values (dumpf!/dumpi!
in src/celt.rs). Usage: instrument.py <libopus source> <new directory>;
see fuzz/DIFFERENTIAL.md."""
import os, shutil, sys

src, out = sys.argv[1], sys.argv[2]
if os.path.exists(out):
    shutil.rmtree(out)
shutil.copytree(src, out, symlinks=False, ignore=shutil.ignore_patterns('.git', '*.o', '*.lo'))
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ofz_dump.h'), os.path.join(out, 'celt'))
path = os.path.join(out, 'celt', 'celt_encoder.c')
text = open(path).read()

# (line in celt_encoder.c, dump code, 'after' or 'before' the statement
# that line starts)
POINTS = [
    ('#include "celt_lpc.h"', '#include "ofz_dump.h"', 'after'),
    ('   sample_max=MAX32(st->overlap_max, celt_maxabs_res(pcm, C*(N-overlap)/st->upsample));',
     'ofz_dumpf("pcm", pcm, CC*N/st->upsample);', 'before'),
    ('   tone_freq = tone_detect(in, CC, N+overlap, &toneishness, mode->Fs);',
     'ofz_dumpf("pre", in, CC*(N+overlap)); ofz_dump1f("tone_freq", tone_freq); ofz_dump1f("toneishness", toneishness);', 'after'),
    ('   toneishness = MIN32(toneishness, QCONST32(1.f, 29)-SHL32(tf_estimate, 15));',
     'ofz_dump1i("isTransient", isTransient); ofz_dump1f("tf_estimate", tf_estimate); ofz_dump1i("tf_chan", tf_chan);', 'before'),
    ('      pf_on = run_prefilter(st, in, prefilter_mem, CC, N, prefilter_tapset, &pitch_index, &gain1, &qg, enabled, st->complexity, tf_estimate, nbAvailableBytes, &st->analysis, tone_freq, toneishness ARG_QEXT(qext_scale));',
     'ofz_dump1i("pf_on", pf_on); ofz_dump1i("pitch_index", pitch_index); ofz_dump1f("gain1", gain1); ofz_dumpf("inpf", in, CC*(N+overlap));', 'after'),
    ('   ALLOC(X, C*N, celt_norm);         /**< Interleaved normalised MDCTs */',
     'ofz_dumpf("freq", freq, C*N); ofz_dumpf("bandE", bandE, C*nbEBands); ofz_dumpf("bandLogE", bandLogE, C*nbEBands); ofz_dumpf("bandLogE2", bandLogE2, C*nbEBands); ofz_dump1i("isTransient2", isTransient);', 'before'),
    ('   enable_tf_analysis = effectiveBytes>=15*C && !hybrid && st->complexity>=2 && !st->lfe && toneishness < QCONST32(.98f, 29);',
     'ofz_dumpf("X", X, C*N);', 'before'),
    ('   ALLOC(tf_res, nbEBands, int);',
     'ofz_dumpi("offsets", offsets, end); ofz_dumpi("importance", importance, end); ofz_dumpi("spread_weight", spread_weight, end); ofz_dump1f("maxDepth", maxDepth); ofz_dump1i("tot_boost", tot_boost);', 'after'),
    ('   ALLOC(error, C*nbEBands, celt_glog);',
     'ofz_dumpi("tf_res", tf_res, end); ofz_dump1i("tf_select", tf_select);', 'after'),
    ('   quant_coarse_energy(mode, start, end, effEnd, bandLogE,',
     'ofz_dumpf("bandLogE_q", bandLogE, C*nbEBands);', 'before'),
    ('   tf_encode(start, end, isTransient, tf_res, LM, tf_select, enc);',
     'ofz_dumpf("error", error, C*nbEBands); ofz_dumpf("oldBandE", oldBandE, C*nbEBands);', 'before'),
    ('   /* For LFE, everything interesting is in the first band */',
     'ofz_dump1i("spread", st->spread_decision);', 'before'),
    ('   min_allowed = ((tell+total_boost+(1<<(BITRES+3))-1)>>(BITRES+3)) + 2;',
     'ofz_dump1i("alloc_trim", alloc_trim); ofz_dumpi("boosts", offsets, end); ofz_dump1i("intensity", st->intensity);', 'before'),
    ('   ALLOC(fine_quant, nbEBands, int);',
     'ofz_dump1i("nbCompressedBytes", nbCompressedBytes);', 'before'),
    ('   quant_fine_energy(mode, start, end, oldBandE, error, NULL, fine_quant, enc, C);',
     'ofz_dumpi("pulses", pulses, end); ofz_dumpi("fine_quant", fine_quant, end); ofz_dumpi("fine_priority", fine_priority, end); ofz_dump1i("codedBands", codedBands); ofz_dump1i("balance", balance);', 'before'),
    ('   if (anti_collapse_rsv > 0)\n   {\n      anti_collapse_on',
     'ofz_dumpf("Xq", X, C*N);', 'before'),
    ('        target = compute_vbr(mode, &st->analysis, base_target, LM, equiv_rate,',
     'ofz_dumpi("vbr_in", (int[]){base_target, LM, equiv_rate, st->lastCodedBands, C, st->intensity, st->constrained_vbr, tot_boost, pitch_change}, 9); ofz_dumpf("vbr_inf", (float[]){st->stereo_saving, tf_estimate, maxDepth, temporal_vbr}, 4);', 'before'),
    ('      quant_energy_finalise(mode, start, end, oldBandE, error, fine_quant, fine_priority, nbCompressedBytes*8-ec_tell(enc), enc, C);',
     'ofz_dumpf("error_fine", error, C*nbEBands); ofz_dump1i("bits_left", nbCompressedBytes*8-ec_tell(enc));', 'before'),
]
for line, code, where in POINTS:
    assert text.count(line) == 1, line
    i = text.index(line)
    if where == 'after':
        j = text.index('\n', text.index(';' if line.rstrip().endswith(';') else line[-1], i + len(line) - 1))
        text = text[:j + 1] + code + '\n' + text[j + 1:]
    else:
        text = text[:i] + code + '\n' + text[i:]
open(path, 'w').write(text)

# bands.c: the stereo angle's rate-distortion choice.
path = os.path.join(out, 'celt', 'bands.c')
text = open(path).read()
for line, code, where in [
    ('#include "quant_bands.h"', '#include "ofz_dump.h"', 'after'),
    ('               if (dist0 >= dist1) {',
     'ofz_dump1i("rdo_band", i); ofz_dumpf("rdo_dist", (float[]){dist0, dist1}, 2); ofz_dumpf("rdo_x", X, N); ofz_dumpf("rdo_xsave", X_save, N); ofz_dumpf("rdo_xdown", X_save2, N); ofz_dumpf("rdo_ydown", Y_save2, N); ofz_dumpf("rdo_ysave", Y_save, N);', 'before'),
]:
    assert text.count(line) == 1, line
    i = text.index(line)
    if where == 'after':
        j = text.index('\n', i)
        text = text[:j + 1] + code + '\n' + text[j + 1:]
    else:
        text = text[:i] + code + '\n' + text[i:]
open(path, 'w').write(text)
