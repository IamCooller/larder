#!/usr/bin/env python3
"""MOE_STORE_KEEP on test_store.py's toy packed MoE, budget mode, cache_bias 0.

Three turns on one cache: a fresh 40-position prefill and 16 tokens, a 40-position prefill on that cache (39 fed:
a prefill, the store released and used as its scratch) and 12 tokens, a 6-position turn (rounds) and 8 tokens.
Before the second turn the budget is cut to the store + twice the larger of the prefill need and the standing room,
with a build transient of 3x the pool (CUDA has one): a built pool would be smaller, the store as it is fits.

  keep on:  the store is the same object through turns 2 and 3 (not rebuilt for a transient no build pays), the
            quota members whose rows turn 2's prefill did not write are published again (STATS store_kept > 0),
            and after every turn each quota member's slot holds its expert's record of experts.bin, no two
            members share a slot (the holes the put-back leaves are filled lowest first, as the agent does).
  keep off: main's store (rebuilt in turn 2, nothing put back), the same slot check.
  seed:     keep on + MOE_STORE_SEED: after each prefill the empty quota rows take the prompt's most routed
            experts, read then; the same slot check.
  All: the tokens equal stock mlx_lm's. The put-back cuts the quotas again (_quotas), each layer keeping the
  rows of its newest survivors: the slot check covers the rows that moved between layers.

    python3 tests/test_store_keep.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

import stream_fast
import test_store as T

_rng = np.random.default_rng(5)
TURNS = [(_rng.integers(0, 200, 40).tolist(), 16), (_rng.integers(0, 200, 40).tolist(), 12),
         (_rng.integers(0, 200, 6).tolist(), 8)]


def bad_slots(glu):
    """Quota members whose slot does not hold their record, and slots held twice."""
    slots = list(glu._lru.values())
    bad = len(slots) - len(set(slots))
    for e, sl in glu._lru.items():
        for _, iov, fd, off, stride in stream_fast._ARENAS[glu._akey]:
            if b"".join(bytes(v) for v in iov[sl]) != os.pread(fd, stride, off + e * stride):
                bad += 1
    return bad


def run(out, keep, seed=False):
    stream_fast._KEEP, stream_fast._SEED = keep, seed
    model, bud = T.engine(out, peak_gb=T.budget_gb())
    cache, toks, kept, same, holes = make_prompt_cache(model), [], [], [], []
    real = bud._put_back

    def put_back():         # layers whose put-back quota has holes, before any claim fills them
        n = real()
        holes.append(sum(sorted(g._lru.values()) != g._qslots[:len(g._lru)] for g in bud.all))
        return n
    bud._put_back = put_back
    for i, (p, n) in enumerate(TURNS):
        if i == 1:
            # The store as it is fits beside turn 2's prefill and the standing room with room to spare;
            # a pool sized for a build (a transient of 3x the pool) does not.
            held, host, have = bud._measure()
            m = max(bud._need(len(p) - 1, False, True), bud._need(bud.min_w - 1, False, False))
            bud.peak = held + host + bud.cache + have + 2 * m
            bud.build_frac = 3.0
        k0, st0 = stream_fast.STATS["store_kept"] + stream_fast.STATS["store_seeded"], dict(bud.stores)
        toks.append([int(t) for t, _ in generate_step(mx.array(p, dtype=mx.int32), model, max_tokens=n,
                                                       prompt_cache=cache)])
        mx.synchronize()
        kept.append(stream_fast.STATS["store_kept"] + stream_fast.STATS["store_seeded"] - k0)
        same.append(all(bud.stores.get(k) is s for k, s in st0.items()))
        assert sum(bad_slots(g) for g in bud.all) == 0, (keep, i)
    holes = max(holes, default=0)
    del model, bud, real
    T.settle()
    return toks, kept, same, holes


def main():
    tmp = Path(tempfile.mkdtemp())
    out = tmp / "toy"
    try:
        T.build(out)
        stock = T.stock_model(out)
        cache = make_prompt_cache(stock)
        ref = [[int(t) for t, _ in generate_step(mx.array(p, dtype=mx.int32), stock, max_tokens=n,
                                                 prompt_cache=cache)] for p, n in TURNS]
        del stock, cache
        T.settle()
        on, kept, same, holes = run(out, True)
        off, kept0, same0, _ = run(out, False)
        s0 = stream_fast.STATS["store_seeded"]
        seeded = run(out, True, seed=True)[0]
        s1 = stream_fast.STATS["store_seeded"] - s0
        assert on == ref and off == ref and seeded == ref, (on, off, seeded, ref)
        assert s1 > 0, s1
        assert kept[1] > 0 and same[1] and same[2] and holes > 0, (kept, same, holes)
        assert kept0 == [0, 0, 0] and not same0[1], (kept0, same0)
        print(f"store_keep: turn 2's prefill kept the store and put {kept[1]} quota members back "
              f"({holes} layers with holes, refilled lowest first); every member's slot == experts.bin, "
              f"tokens == stock, keep on and off; off rebuilt it; MOE_STORE_SEED read {s1} prompt experts "
              f"into empty quota rows, tokens == stock")
        print("test_store_keep: ok")
    finally:
        stream_fast._KEEP, stream_fast._SEED = True, False
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
