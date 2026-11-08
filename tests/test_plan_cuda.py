#!/usr/bin/env python3
"""CUDA's prefill plan replayed offline on the PC's measured line (two recorded plans): two fresh passes a few hundred positions apart must not set the slope. No model, no GPU."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stream_fast as sf

MB = 10**6
REAL = sf.BACKEND
# The PC's install line for chunkattn and the indexer.
PC_ATTN = (48.0, 14.501, 24576.0, 841.73, 128)
# plan_main.json (d6907574): the golden call order's fresh passes, then the 16k and 32k prompts.
SMALL = {16: 171.2, 31: 80.6, 36: 51.1, 37: 51.1, 39: 93.0, 265: 181.2}
MAIN = {**SMALL, 15637: 2968.2, 32558: 6023.7}
# plan_main's 32k prompt and plan_e15's (e15c9340, the same CUDA line), 209 positions apart.
NEAR = {**SMALL, 32558: 6023.7, 32767: 5714.1}
# plan_main's state at its 'system' request, where the 16/39 pair grew less at the wider width.
SYSTEM = {16: 171.2, 31: 80.6, 36: 51.1, 37: 51.1, 39: 93.0}


def make(seen):
    proj = types.SimpleNamespace(bits=4, group_size=64, mode="affine",
                                 _packed={"n_experts": 512, "stride": 64, "parts": []})
    glu = types.SimpleNamespace(_projs=[proj], _exl3=None)
    sf.BACKEND = "metal"            # the toy expert has no parts for CUDA's slot count
    b = sf.MemBudget([glu], 16, 0.05, 10, 1, 1, 10 * 2560, None)
    sf.BACKEND = "cuda"
    b.layer_major = b.lm_ok = True
    b._lm_seen = {w: g * MB for w, g in seen.items()}
    b._lm_fresh = dict(b._lm_seen)
    b._lm_rows = {w: g * MB / w for w, g in seen.items()}
    b.rate, b.state, b.min_w, b.ring_bytes = 10702.8e3, 115.6 * MB, 205, 37 * MB
    return b


def caches(keys):
    """Flash-Next's 12 attention layers, 2048 bytes a position each, on mlx_lm's 256-position steps."""
    return [types.SimpleNamespace(offset=keys, nbytes=-(-keys // 256) * 256 * 2048, step=256)
            for _ in range(12)] if keys else []


def pieces(b, n, fresh, keys, held_mb, host_mb):
    """The passes a prefill of n is cut into, held growing by the PC's 28.7 KB of K/V and indexer keys a position."""
    out = []
    while sum(out) < n:
        a = sum(out)
        held, host = (held_mb + (a + keys) * 28.7e-3) * MB, host_mb * MB
        b._measure = lambda h=held, o=host: (h, o, 0)
        out.append(b.plan(n - a, fresh and not a, caches(keys + a)))
    return out


try:
    sf.set_prefill_attention(PC_ATTN)
    # A 3,000-token tail at 60,000 keys plans one piece, as 0f76c2dd did, where the near pair read a slope of 0 and cut 128-position pieces.
    b = make(NEAR)
    assert b._lm_line()[1] > 0.1 * MB, b._lm_line()
    got = pieces(b, 3000, False, 60000, 2930.0, 3597.1)
    assert got == [3000], got
    # A tail as long as the agent bench's (10,825 at 100,000 keys) is one piece too.
    got = pieces(b, 10825, False, 100000, 2930.0, 3597.1)
    assert got == [10825], got
    # The small pair that grew less at its wider width keeps a slope, capped at four expert rows a position.
    assert sf.MemBudget._lm_line(make(SYSTEM))[1] == sf.PREFILL_ROWS * 10 * 2560 * 2
    # plan_main's own history plans as d6907574 did: the 16k prompt in one pass, the 3,000 tail in one.
    assert pieces(make(SMALL), 15637, True, 0, 2939.6, 1872.9) == [15637]
    assert pieces(make(MAIN), 3000, False, 16384, 2914.0, 2674.4) == [3000]
    # The 64k and 128k needles plan the pieces d6907574 plans from the same line, the ones its runs answered in.
    got = pieces(make(MAIN), 65538, True, 0, 2935.4, 1567.4)
    assert got == [34816, 30722], got
    got = pieces(make(MAIN), 131072, True, 0, 2935.4, 1567.4)
    assert got == [44032, 43520, 22528, 20992], got
    got = pieces(make(SMALL), 131072, True, 0, 2935.4, 1567.4)
    assert got == [44032, 29696, 28672, 28672], got
finally:
    sf.set_prefill_attention(None)
    sf.BACKEND = REAL
print("ok")
