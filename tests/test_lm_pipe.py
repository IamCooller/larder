#!/usr/bin/env python3
"""MOE_LM_PIPE's whole-array unit building (stream_fast._lmr_groups, _miss_rounds) against the piece-by-piece
originals: the same groups and rounds, rows and experts, in the same order, over random runs and edge cases; and the
cache snapshot a failed run is put back from (_lmr_snap, _lmr_restore).

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_lm_pipe.py
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stream_fast as sf    # noqa: E402
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache    # noqa: E402


def same(a, b):
    return len(a) == len(b) and all(len(x) == len(y) and all(np.array_equal(p, q) and p.dtype == q.dtype
                                                             for p, q in zip(x[:2], y[:2])) for x, y in zip(a, b))


def case(rng, n_pos, k, n_exp, n_pieces, half, miss):
    ids = np.stack([rng.choice(n_exp, k, replace=False) for _ in range(n_pos)]).reshape(-1)
    rows = np.argsort(ids, kind="stable")
    uniq, inv = np.unique(ids, return_inverse=True)
    pos = inv[rows]
    bounds = np.sort(rng.choice(np.arange(1, n_pos), n_pieces - 1, replace=False)) if n_pieces > 1 else []
    starts = [0, *bounds]
    piece = np.searchsorted(starts, rows // k, "right") - 1
    g_ref = sf._lmr_groups_ref(pos, piece, n_pieces, half)
    assert same(g_ref, sf._lmr_groups(pos, piece, n_pieces, half)), "groups differ"
    on = rng.random(len(pos)) < 0.8
    cmiss = np.unique(rng.choice(len(uniq), max(0, int(miss * len(uniq))), replace=False)) if miss else pos[:0]
    for per in (half, 2 * half):
        assert same(sf._miss_rounds_ref(on, pos, cmiss, per), sf._miss_rounds(on, pos, cmiss, per)), "rounds differ"


rng = np.random.default_rng(1)
n = 0
for n_pos, k, n_exp, n_pieces, half, miss in [(1024, 8, 256, 8, 16, 0.1), (1024, 8, 256, 1, 16, 0.0),
                                              (300, 8, 256, 3, 16, 0.5), (40, 10, 512, 4, 8, 0.2),
                                              (2, 8, 256, 2, 16, 1.0), (5000, 8, 256, 8, 16, 0.02)]:
    for _ in range(5):
        case(rng, n_pos, k, n_exp, n_pieces, half, miss)
        n += 1


def kv(c, t):
    c.update_and_fetch(mx.random.normal((1, 2, t, 8)), mx.random.normal((1, 2, t, 8)))


# A fresh pair and a pair that already holds positions, each run over and put back.
for held in (0, 300):
    a, b = KVCache(), ArraysCache(size=2)
    if held:
        kv(a, held)
        b[0], b[1] = mx.random.normal((1, 3, 4)), mx.random.normal((1, 2, 4, 4))
        mx.eval(a.keys, a.values, b.cache)
    keys0, vals0, off0, st0 = a.keys, a.values, a.offset, list(b.cache)
    want = [np.array(x) for x in (a.state if held else ()) + tuple(st0 if held else ())]
    snap = sf._lmr_snap([a, b])
    assert snap is not None
    assert sf._lmr_held(snap) == (a.keys.nbytes + a.values.nbytes + sum(x.nbytes for x in st0) if held else 0)
    for _ in range(3):
        kv(a, 1024)
        b[0], b[1] = mx.random.normal((1, 3, 4)), mx.random.normal((1, 2, 4, 4))
    mx.eval(a.keys, b.cache)
    sf._lmr_restore([a, b], snap)
    assert a.offset == off0 and a.keys is keys0 and a.values is vals0, "KV cache not put back"
    assert all(x is y for x, y in zip(b.cache, st0)), "recurrent cache not put back"
    got = [np.array(x) for x in (a.state if held else ()) + tuple(b.cache if held else ())]
    assert all(np.array_equal(x, y) for x, y in zip(got, want)), "values changed"
    kv(a, 5)                # the cache takes the next piece from the restored offset
    assert a.offset == off0 + 5
assert sf._lmr_snap([KVCache(), RotatingKVCache(max_size=64)]) is None, "an unknown cache type must not be retried"
assert sf._lmr_held(None) == 0


class Bud:
    """MemBudget's fields _narrow reads: a need of 1 MB a position, None before anything is measured."""
    min_w = 128

    def __init__(self, measured=True):
        self.measured = measured

    def _need(self, w, fresh, prefill, copy, keys):
        return w * 1e6 if self.measured else None


real = sf.mx.device_info
try:
    for free, want in ((2e9, 1024), (600e6, 512), (300e6, 256), (50e6, 128)):
        sf.mx.device_info = lambda f=free: {**real(), "free_memory": f}
        got = sf.MemBudget._narrow(Bud(), 1024, False, [])
        assert got == want, f"{free / 1e6:.0f} MB free: piece {got}, want {want}"
    assert sf.MemBudget._narrow(Bud(False), 1024, True, []) == 1024, "an unknown need keeps the width"
finally:
    sf.mx.device_info = real
print(f"ok: {n} runs, groups and rounds equal; snapshots put back; pieces narrowed to the room")
