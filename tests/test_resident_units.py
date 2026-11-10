#!/usr/bin/env python3
"""MOE_RESIDENT's GGUF prefill units (stream_fast._call_resident) without a model or
a GPU of CUDA's kind: a toy store (quota rows, a 16-row ring with stale rows, a tier in host memory, an experts.bin)
and one layer's sorted rows, through StreamingSwitchGLU._call_grouped.

  1. MOE_RESIDENT_OVERLAP, _ONECALL and both give main's grouped output bit for bit (the decode chain for most groups,
     the dense path for a heavy one, misses from the tier and from the file), over passes of 64, 33 and 5 positions;
  2. the chain rows match a gemv over a buffer of every expert bit for bit (the right bytes in the ring rows);
  3. the copies go where the switches say: OVERLAP alternates the ring halves with at most a half a copy, ONECALL alone
     fills the whole ring a round, and ONECALL makes fewer copies than the groups;
  4. MemBudget._fit on CUDA drops every store due for a rebuild before it builds any, so a build that fails (another
     program took the freed VRAM) still leaves the big store given back;
  5. MOE_RESIDENT_LM: one call over several pieces' rows (the pieces' bounds in `_lmr`) gives every row the bits its
     own piece's call gives, the heavy group dense in the wide pieces and on the chain in the narrow one, with fewer
     copies than the pieces make apart.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_resident_units.py
"""
import ctypes
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx       # noqa: E402

import gguf_experts as ge   # noqa: E402
import stream_fast as sf    # noqa: E402

E, HID, INTER, K = 48, 256, 128, 4
POOL, HALF = 20, 8
TYPES = ("Q4_0", "Q4_0", "Q4_0")
SHAPES = ((INTER, HID), (INTER, HID), (HID, INTER))


class Tier:
    """The native agent's tier: tier_find gives each held expert's host address, 0 for the others."""
    def __init__(self, recs, held):
        self.buf = np.concatenate([recs[e] for e in held]) if held else np.zeros(1, np.uint8)
        self.at = {e: self.buf.ctypes.data + i * len(recs[e]) for i, e in enumerate(held)}
        self.tier_slot, self.agent = True, self

    def tier_find(self, idx, e):
        return np.array([self.at.get(int(v), 0) for v in e], np.uint64)


class Stage:
    def copy_in(self, copies):
        for d, s, n in copies:
            ctypes.memmove(d, s, n)


class Store:
    scratch = sf.ExpertStore.scratch

    def __init__(self, bufs):
        self.pool, self.slots, self.bufs = POOL, POOL + 2 * HALF, bufs
        self.owner = [None] * self.slots


class Glu:
    _call_grouped = sf.StreamingSwitchGLU._call_grouped
    _call_resident = sf.StreamingSwitchGLU._call_resident
    _take_group = sf.StreamingSwitchGLU._take_group
    _fetch = sf.StreamingSwitchGLU._fetch
    _experts = sf.StreamingSwitchGLU._experts
    _own_kernels = sf.StreamingSwitchGLU._own_kernels


def setup(d, rng):
    w = [ge.random_experts(rng, E, r, c, t) for (r, c), t in zip(SHAPES, TYPES)]
    stride = [a[0].nbytes for a in w]
    off = np.cumsum([0] + [a.nbytes for a in w])[:-1]
    (d / "experts.bin").write_bytes(b"".join(a.tobytes() for a in w))
    fd = os.open(d / "experts.bin", os.O_RDONLY)
    slots = POOL + 2 * HALF
    bufs, views = [], []
    for a in w:
        b = mx.zeros((slots,) + a.shape[1:], dtype=mx.uint8)
        mx.eval(b)
        bufs.append((b,))
        views.append(np.array(b, copy=False))
    order = rng.permutation(E)
    quota, ring = order[:POOL], order[POOL:POOL + 6]
    slot = np.full(E, -1, np.int64)
    slot[quota] = np.arange(POOL)
    slot[ring] = POOL + rng.choice(2 * HALF, len(ring), replace=False)       # stale ring members: fetched again
    for j, a in enumerate(w):
        views[j][slot[quota]] = a[quota]
    g = Glu()
    g._exl3, g._gguf, g._native_idx, g._budget = None, TYPES, 0, None
    g._projs = [type("P", (), {"_packed": {"n_experts": E}})()]
    g._slot_np = slot
    g._akey = sf._ArenaKey()
    sf._ARENAS[g._akey] = [(bufs[j], [[views[j][s]] for s in range(slots)], fd, int(off[j]), stride[j])
                           for j in range(3)]
    recs = {e: np.concatenate([a[e].ravel() for a in w]) for e in range(E)}
    absent = [e for e in range(E) if slot[e] < 0 or slot[e] >= POOL]
    tier = Tier(recs, absent[::2])                                            # half from the tier, half from the file
    return g, Store(bufs), views, w, tier


def routing(rng, t):
    """t positions of K distinct experts: 2 of the heavy 0..7 (a dense-path group) and K - 2 of the rest."""
    idx = np.stack([np.concatenate([rng.choice(8, 2, replace=False), 8 + rng.choice(E - 8, K - 2, replace=False)])
                    for _ in range(t)])
    flat = idx.ravel()
    order = np.argsort(flat, kind="stable")
    unique_ids, pos = np.unique(flat[order], return_inverse=True)
    return idx, unique_ids, pos, order


def run(g, store, views, x, uid, pos, order, overlap, onecall, log):
    for v in views:
        v[POOL:] = 0xA5                         # whatever the ring held: every miss must be copied again
    sf._OVERLAP, sf._ONECALL = overlap, onecall
    log.clear()
    y = g._call_grouped(x, uid, pos, mx.array(order.astype(np.uint32)), None, len(order), store)
    mx.eval(y)
    return np.array(y.astype(mx.float32))


def check_lmr(g, store, views, rng, log):
    """Four pieces (33, 64, 5 and 33 positions) apart, then as one MOE_RESIDENT_LM call; each row must match."""
    sizes = (33, 64, 5, 33)
    xs, idxs = [], []
    for t in sizes:
        idxs.append(routing(rng, t)[0])
        xs.append(mx.array(rng.standard_normal((t, 1, HID)).astype(np.float32)).astype(mx.bfloat16))
    apart, n_apart = [], 0
    for x, idx in zip(xs, idxs):
        flat = idx.ravel()
        order = np.argsort(flat, kind="stable")
        uid, pos = np.unique(flat[order], return_inverse=True)
        apart.append(run(g, store, views, x, uid, pos, order, True, True, log))
        n_apart += len(log)
    flat = np.concatenate(idxs).ravel()
    order = np.argsort(flat, kind="stable")
    uid, pos = np.unique(flat[order], return_inverse=True)
    at = np.cumsum((0,) + sizes)
    g._lmr = [(int(a), int(b)) for a, b in zip(at, at[1:])]
    try:
        y = run(g, store, views, mx.concatenate(xs), uid, pos, order, True, True, log)
    finally:
        del g._lmr
    ref = np.concatenate(apart)
    bad = int((y.view(np.uint32) != ref.view(np.uint32)).sum())
    assert bad == 0, f"layer-major: {bad} of {y.size} values differ from the pieces' own calls"
    assert len(log) < n_apart, (len(log), n_apart)
    print(f"  layer-major: pieces {sizes} in one call bit-equal to each piece's own, {len(log)} copies against "
          f"{n_apart} apart")


def check_fit_order():
    """A store build that runs out of memory: the big store must already be gone."""
    log = []

    class Old:
        def __init__(self, name, pool):
            self.name, self.pool, self.nbytes = name, pool, pool

    class New:
        fail = True

        def __init__(self, gs, size, halves, per_half):
            if New.fail:
                New.fail = False
                log.append("build fails")
                raise MemoryError("the other program took the freed VRAM")
            self.pool, self.nbytes, self.swing, self.per_half = size, size, 0, per_half
            self.kept = self.written = None
            log.append(f"build {size}")

        def release(self):
            pass

    class G:
        def __init__(self, name):
            self.name, self._store = name, None

        def clear_arena(self):
            log.append(f"drop {self.name}")

    class Budget:
        _fit = sf.MemBudget._fit

        def _measure(self):
            return 0, 0, 0

        def _quotas(self, pool):
            return {id(g): q for g, q in zip(self.gs, (60, 400))}

        def _settle(self, host):
            log.append("settle")

    g1, g2 = G("small"), G("big")
    b = Budget()
    b.gs, b.groups, b.stores = [g1, g2], {"q40": [g1], "q41": [g2]}, {"q40": Old("small", 100), "q41": Old("big", 700)}
    b.peak, b.cache, b.build_swing, b.build_frac, b.ring_bytes = 1000, 0, 0, 0.0, 0
    b._short, b.prefill, b.halves, b.per_half, b.rebuilds, b._row_elems = True, False, 2, 8, 0, None
    real = sf.BACKEND, sf.ExpertStore
    sf.BACKEND, sf.ExpertStore = "cuda", New
    try:
        b._fit(False, 900)
    except MemoryError:
        pass
    finally:
        sf.BACKEND, sf.ExpertStore = real
    assert log[:4] == ["drop small", "drop big", "settle", "build fails"] and not b.stores, (log, b.stores)
    print(f"  fit order: {log}, the big store given back before the failed build")


def main():
    check_fit_order()
    rng = np.random.default_rng(7)
    ge._COST = [0.0, 1.0, 0.0, 6.0, 0.0]        # dense pays above 6 rows an expert: the heavy group only
    sf._RESIDENT, sf._INPLACE = True, True
    log = []
    fetch = sf.StreamingSwitchGLU._fetch

    def logged(self, e, store, base=0):
        log.append((len(e), base))
        return fetch(self, e, store, base)
    Glu._fetch = logged
    dense, n_dense = ge.dense, [0]

    def counted(*a, **k):
        n_dense[0] += 1
        return dense(*a, **k)
    ge.dense = counted
    with tempfile.TemporaryDirectory() as t:
        g, store, views, w, tier = setup(Path(t), rng)
        sf._NATIVE, sf._CUDA_STAGE = tier, Stage()
        full = tuple(mx.array(a) for a in w)
        for n_pos in (64, 33, 5):
            idx, uid, pos, order = routing(rng, n_pos)
            x = mx.array(rng.standard_normal((n_pos, 1, HID)).astype(np.float32)).astype(mx.bfloat16)
            n_dense[0] = 0
            main_y = run(g, store, views, x, uid, pos, order, False, False, log)
            n_main, heavy = len(log), n_dense[0] > 0
            assert heavy == (n_pos * 2 > 6 * 8), "the heavy group's path"
            got = {}
            for name, ov, oc in (("overlap", True, False), ("onecall", False, True), ("both", True, True)):
                y = run(g, store, views, x, uid, pos, order, ov, oc, log)
                bad = int((y.view(np.uint32) != main_y.view(np.uint32)).sum())
                assert bad == 0, f"{n_pos} positions, {name}: {bad} of {y.size} values differ from main's"
                got[name] = list(log)
            # Chain rows against a gemv over every expert at its own index.
            flat = idx.ravel()
            chain = np.flatnonzero(flat >= 8)
            ref = ge.gemv(x.reshape(-1, HID)[mx.array(chain // K)], full, mx.array(flat[chain].astype(np.uint32)), 1,
                          TYPES, mx.bfloat16)
            ref = np.array(ref.astype(mx.float32))
            assert (ref.view(np.uint32) == main_y.reshape(-1, HID)[chain].view(np.uint32)).all(), "chain rows differ"
            for n, b in got["overlap"] + got["both"]:
                assert n <= HALF and b in (0, HALF), (n, b)
            for name in ("overlap", "both"):
                bases = [b for _, b in got[name]]
                assert bases == [i % 2 * HALF for i in range(len(bases))], (name, bases)
            assert all(b == 0 and n <= 2 * HALF for n, b in got["onecall"]), got["onecall"]
            assert len(got["onecall"]) < n_main or n_main <= 1, (got["onecall"], n_main)
            print(f"  {n_pos} positions: {len(flat)} rows (the heavy group's {int((flat < 8).sum())} on the {'dense path' if heavy else 'chain'}), main {n_main} copies, "
                  f"overlap {len(got['overlap'])}, onecall {len(got['onecall'])} (sizes "
                  f"{[n for n, _ in got['onecall']]}), both {len(got['both'])}: bit-equal to main")
        check_lmr(g, store, views, rng, log)
        sf._NATIVE, sf._CUDA_STAGE = None, None
    print("test_resident_units: ok")


if __name__ == "__main__":
    main()
