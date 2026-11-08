#!/usr/bin/env python3
"""Metal plans a long prefill as df27abb4 did: the line's slope, the piece rule and chunkattn's
row search, with CUDA's rules kept behind BACKEND. No model, no GPU."""
import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chunkattn
import stream_fast as sf

MB = 10**6
REAL = sf.BACKEND
# A cold history as an M-series Mac measured it: the 205-position probe and three long passes.
SEEN = {205: 81.2 * MB, 7907: 1402.7 * MB, 12288: 2136.9 * MB, 22528: 3782.5 * MB}
ROWS = {205: 314.5e3, 7907: 150.4e3, 12288: 150.2e3, 22528: 148.3e3}


def make(calib=None):
    proj = types.SimpleNamespace(bits=4, group_size=64, mode="affine",
                                 _packed={"n_experts": 8, "stride": 64, "parts": []})
    glu = types.SimpleNamespace(_projs=[proj], _exl3=None)
    b = sf.MemBudget([glu], 6, 0.5, 2, 1, 1, None, None, calib=calib)
    if calib:
        return b
    b.layer_major = b.lm_ok = True
    b._lm_seen, b._lm_rows = dict(SEEN), dict(ROWS)
    b.rate, b.state, b.min_w = 0.4 * MB, 115.6 * MB, 205
    b._measure = lambda: (0, 0, 0)
    return b


def df27abb4_line(seen, rows):
    """The line exactly as df27abb4 computed it."""
    rate = rows[max(rows)]
    wide = [w for w in seen if w >= 2 * sf.LM_W]
    if wide:
        hi = max(wide)
        lo = max((w for w in wide if 1.25 * w <= hi), default=None)
        if lo:
            rate = max(rate, (seen[hi] - seen[lo]) / (hi - lo))
    a = max(g - rate * w for w, g in seen.items())
    top = max(seen)
    if top < sf.LM_W:
        a = max(a, (seen[top] - rate * top) * sf.LM_W / top)
    return a, rate


def plan(b, n):
    sf._eval_caches, sf._copy, sf._keys = (lambda c: None), (lambda c: 0.0), (lambda c: 0)
    return b.plan(n, True, [])


def df27abb4_plan(b, n):
    """The first piece of a fresh pass of n positions as df27abb4 cut it, for a need that does not fit."""
    a, rate = df27abb4_line(b._lm_seen, b._lm_rows)
    room = b.peak - b.cache - b.ring_bytes
    fits = int((room - b.state - a) / rate)
    if fits >= sf.LM_W:
        return min(n, fits // sf.LM_W * sf.LM_W)
    k = -(-n // max(fits, 1))
    return max(b.min_w, -(-n // k))


try:
    sf.BACKEND = "metal"
    b = make()
    assert b._lm_line() == df27abb4_line(SEEN, ROWS), b._lm_line()
    for n in (21619, 42724):
        assert plan(b, n) == df27abb4_plan(b, n), (n, plan(b, n))
    sf.BACKEND = "cuda"
    assert b._lm_line() != df27abb4_line(SEEN, ROWS), "CUDA's line is the metal one"
    sf.set_prefill_attention((48.0, 15.0, 24576.0, 841.0, 128))
    assert plan(b, 42724) != df27abb4_plan(b, 42724), "CUDA's plan is the metal one"
    # A saved width under min_w stays on Metal, as it did in df27abb4.
    saved = {"v": sf.CALIB_VERSION, "rate": 1.0, "state": 1.0, "keep": 0.0, "keep_w": 0,
             "host_mb": 1, "host_n": 1, "lm_seen": {"10": 1.0, "7907": 2.0},
             "lm_rows": {"10": 1.0, "7907": 2.0}}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.json")
        open(path, "w").write(json.dumps(saved))
        sf.BACKEND = "metal"
        assert set(make(path)._lm_seen) == {10, 7907}
finally:
    sf.set_prefill_attention(None)
    sf.BACKEND = REAL
# The slope probe's shapes feed chunkattn's chunk rows, so Metal keeps df27abb4's 512 rows over 512 keys.
assert (chunkattn.SAMPLE_T, chunkattn.SAMPLE_N) == ((64, 4096) if REAL == "cuda" else (512, 512))
want = (5, 8, 16, 32, 64) + ((128, 256) if REAL == "cuda" else ())
assert chunkattn.ROW_CANDIDATES == want, chunkattn.ROW_CANDIDATES
print("ok")
