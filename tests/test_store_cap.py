#!/usr/bin/env python3
"""ExpertStore.gather_slots / build_extra at Flash-Next's expert shapes, and MemBudget._quotas keeping a store
under its slot cap. No model, no GPU work."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import stream_fast as sf


def proj(n, k, n_exp=512):
    parts = [{"name": "weight", "dtype": "U32", "shape": [n, k // 8]},
             {"name": "scales", "dtype": "BF16", "shape": [n, k // 32]},
             {"name": "biases", "dtype": "BF16", "shape": [n, k // 32]}]
    stride = sum(int(np.prod(p["shape"])) * (4 if p["dtype"] == "U32" else 2) for p in parts)
    return types.SimpleNamespace(bits=4, group_size=32, mode="affine",
                                 _packed={"n_experts": n_exp, "stride": stride, "parts": parts})


def glu():
    return types.SimpleNamespace(_projs=[proj(640, 2560), proj(640, 2560), proj(2560, 640)])


backend = sf.BACKEND
try:
    sf.BACKEND = "cuda"
    # 640 x 2560 4-bit weights a slot: 2^32 / 1638400 = 2621.44.
    assert sf.ExpertStore.gather_slots(glu()) == 2621
    # Largest first: the third weight moves beside the other two, 4 x 819200 against a 3072000-byte slot.
    assert sf.ExpertStore.build_extra(glu()) == 204800
    gs = [glu() for _ in range(4)]
    b = sf.MemBudget(gs, 6, 0.0, 10, 1, 1, None, None)
    assert b.slot_cap == {gs[0]._skey: 2621 - 20} and abs(b.build_frac - 204800 / 3072000) < 1e-12
    b.slot_cap = {gs[0]._skey: 50}
    q = b._quotas(1e12)                 # room for every expert of every layer, the cap for 50
    assert sum(q.values()) == 50, q
    gs[1]._ghost_hist[:20] = 5          # the ghosts' greedy too
    q = b._quotas(1e12)
    assert sum(q.values()) == 50 and q[id(gs[1])] >= 20, q
    sf.BACKEND = "metal"
    assert sf.ExpertStore.gather_slots(glu()) == 1 << 62 and sf.ExpertStore.build_extra(glu()) == 0
    b = sf.MemBudget([glu() for _ in range(4)], 6, 0.0, 10, 1, 1, None, None)
    assert b.build_frac == 0 and sum(b._quotas(1e12).values()) == 4 * 512
finally:
    sf.BACKEND = backend
print("test_store_cap: ok")
