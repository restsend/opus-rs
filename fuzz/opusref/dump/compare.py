#!/usr/bin/env python3
"""Compares the OFZ_DUMP output of opus-rs (`R` lines) and the instrumented
reference (`D` lines) for one frame: compare.py <dump> [frame]."""
import struct, sys

dumps = {'R': {}, 'D': {}}
order = []
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) < 4 or p[0] not in dumps:
        continue
    side, tag, ty, n = p[0], p[1], p[2], int(p[3])
    vals = p[4:4 + n]
    vals = [struct.unpack('>f', bytes.fromhex(v))[0] for v in vals] if ty == 'f' else [int(v) for v in vals]
    dumps[side].setdefault(tag, []).append(vals)
    if tag not in order:
        order.append(tag)
frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for t in order:
    a, b = dumps['R'].get(t, []), dumps['D'].get(t, [])
    if frame >= len(a) or frame >= len(b):
        print(f'{t:18s} missing (opus-rs {len(a)}, libopus {len(b)} frames)')
        continue
    x, y = a[frame], b[frame]
    m = min(len(x), len(y))
    note = f' (length {len(x)} vs {len(y)})' if len(x) != len(y) else ''
    diffs = [i for i in range(m) if x[i] != y[i]]
    if not diffs:
        print(f'{t:18s} identical ({m}){note}')
    elif isinstance(x[0], float):
        rel = max(abs(x[j] - y[j]) / max(abs(y[j]), 1e-30) for j in diffs)
        i = diffs[0]
        print(f'{t:18s} {len(diffs)}/{m} differ, first [{i}] opus-rs {x[i]!r} libopus {y[i]!r}, max rel {rel:.3g}{note}')
    else:
        i = diffs[0]
        print(f'{t:18s} {len(diffs)}/{m} differ, first [{i}] opus-rs {x[i]} libopus {y[i]}: {x[:24]} vs {y[:24]}{note}')
