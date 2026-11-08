#!/usr/bin/env python3
"""MOE_NATIVE=1 against the Python path on test_store's toy packed MoE
(qwen3_next, 4 MoE layers, 32 experts, top-4, a 6-slot arena):

  1. four turns of greedy decode -- a 40-id prefill, then a 5-id prompt and
     29/39-id passes that go through Python between the native decode
     passes, an arena parked and read back before turn 2 and every arena
     rebuilt before turn 3 -- give the same tokens and bitwise the same
     logprobs, at cache_bias 0 and 0.5;
  2. after the run the arena LRU (order and slots) and `_seen` are the same;
  3. the native passes did run, and reported no agent error;
  4. budget mode (--mem-peak-gb's shared stores), 8 MoE layers in two
     stores, quotas cut by hand from 0 to 6 rows so the ring halves carry
     most misses and layers of one half evict each other's rows: the same
     four turns give the same tokens and logprobs at cache_bias 0 and 0.5,
     and the quota LRU, `_ring`, `_seen`, the ghost LRU and the ring halves
     (owner, next) end the same;
  5. CUDA: the agent's victim tier served misses in
     every native arm, unless MOE_PIPE_TIER_GB=0 turned it off;
  6. Metal, MOE_NATIVE_MTP: passes of 1-5
     positions after a prefill, 2-4 through the agent and 5 (wider than the
     arena's slots hold) through Python, on a 16-slot arena and on the budget
     stores, at cache_bias 0 and 0.5: every pass's logits bitwise and the
     layer state after them equal the Python path's;
  7. CUDA with MOE_PIPE_STAGING=1 in the environment: every native arm read
     guess records and copied misses from them, and 1-4 still hold.

    python3 tests/test_native.py      # Metal: needs native/build (native/CMakeLists.txt);
                                      # CUDA: native/moe_pipe_cuda, g++ builds it at first use
    MOE_PIPE_STAGING=1 python3 tests/test_native.py      # CUDA, staging on
"""
import gc
import os
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
import test_store as ts


def between(model, i):
    if i == 1:
        stream_fast.park_arenas(model)
        stream_fast.unpark_arenas(model)
    elif i == 2:
        for g in stream_fast.glus_of(model):
            g.resize(g._cap + 1)
            g.resize(g._cap - 1)


def tier_hits():
    return stream_fast.STATS["tier_hits"]


def check_tier(native, hits, what):
    """CUDA, the tier on: the native arm took misses from it."""
    if native and stream_fast.BACKEND == "cuda" and os.environ.get("MOE_PIPE_TIER_GB") != "0":
        assert hits > 0, f"{what}: no miss came from the victim tier"
        print(f"{what}: {hits} misses from the victim tier, "
              f"{stream_fast.STATS['dio_fell']} records read buffered after O_DIRECT")


def staged():
    return stream_fast.STATS["stage_rec"], stream_fast.STATS["stage_prom"]


def check_stage(native, st0, what):
    """CUDA, MOE_PIPE_STAGING=1: the native arm read guess records and a claim used some of them."""
    if native and stream_fast._PIPE_STAGING:
        rec, prom = (a - b for a, b in zip(staged(), st0))
        assert rec > 0 and prom > 0, f"{what}: staging read {rec} guess records, a claim used {prom}"
        print(f"{what}: staging read {rec} guess records, {prom} copied into slots")


def run(out, native, bias):
    model = ts.stock_model(out)
    stream_fast.install(model, out, resident=ts.RES,
                        cache_bias=bias, defer=False, native=native)
    park0, hits0, st0 = stream_fast.TIMERS["native_park"], tier_hits(), staged()
    cache, toks, lps = make_prompt_cache(model), [], []
    for i, (p, n) in enumerate(zip(ts.PROMPTS, ts.GEN)):
        between(model, i)
        for t, lp in generate_step(mx.array(p, dtype=mx.int32), model,
                                   max_tokens=n, prompt_cache=cache):
            toks.append(int(t))
            lps.append(np.array(lp.astype(mx.float32)))
    stream_fast.native_sync()
    state = [(list(g._lru.items()), g._seen.copy()) for g in stream_fast.glus_of(model)]
    ran = stream_fast.TIMERS["native_park"] > park0
    if native:
        assert stream_fast._NATIVE is not None and not stream_fast._NATIVE.agent.error
    check_tier(native, tier_hits() - hits0, f"bias {bias}")
    check_stage(native, st0, f"bias {bias}")
    return toks, lps, state, ran


CFG8 = {**ts.CFG, "num_hidden_layers": 8}  # 4-bit layers 0, 3-7 share a store, 2-bit 1-2
QUOTA = [0, 2, 3, 6, 1, 4, 2, 0]             # rows per layer; top_k is 4


def run_budget(out, native, bias):
    model = ts.stock_model(out, CFG8)
    stream_fast.install(model, out, resident=ts.RES,
                        cache_bias=bias, defer=False, native=native)
    bud = stream_fast.enable_budget(model, ts.budget_gb())
    glus = stream_fast.glus_of(model)
    quota = {id(g): q for g, q in zip(glus, QUOTA)}
    bud._quotas = lambda pool: quota
    park0, hits0, st0 = stream_fast.TIMERS["native_park"], tier_hits(), staged()
    # A store being replaced must be gone when its successor is built: kept
    # alive (the agent's ring list did), the budget counted it as held.
    real_init, stale = stream_fast.ExpertStore.__init__, []

    def init(st, *a, **k):
        gc.collect()
        live = sum(isinstance(o, stream_fast.ExpertStore) for o in gc.get_objects())
        if live > len(bud.stores) + 1:          # + the one being built
            stale.append(live - len(bud.stores) - 1)
        real_init(st, *a, **k)
    stream_fast.ExpertStore.__init__ = init
    cache, toks, lps = make_prompt_cache(model), [], []
    try:
        for i, (p, n) in enumerate(zip(ts.PROMPTS, ts.GEN)):
            if i == 3:
                # Smaller quotas: turn 4's prefill rebuilds both stores
                # after native passes.
                quota.update({id(g): max(0, q - 1) for g, q in zip(glus, QUOTA)})
                rebuilds, kept = bud.rebuilds, stream_fast.STATS["store_kept"]
            for t, lp in generate_step(mx.array(p, dtype=mx.int32), model,
                                       max_tokens=n, prompt_cache=cache):
                toks.append(int(t))
                lps.append(np.array(lp.astype(mx.float32)))
    finally:
        stream_fast.ExpertStore.__init__ = real_init
    # MOE_STORE_KEEP: the stores stay and the put-back cuts the smaller quotas instead; either
    # way every layer's arena changed under the agent.
    assert bud.rebuilds > rebuilds or stream_fast.STATS["store_kept"] > kept, \
        "turn 4 neither rebuilt the stores nor cut their quotas again"
    assert not stale, f"replaced stores still alive at a rebuild: {stale}"
    stream_fast.native_sync()
    at = {id(g): i for i, g in enumerate(glus)}
    state = [(list(g._lru.items()), sorted(g._ring.items()), g._seen.copy(),
              g._ghost_hist.copy(), g._ghost_last.copy(), g._ghost_clock) for g in glus]
    rings = [(list(st.next), [None if o is None else (at[id(o[0])], o[1]) for o in st.owner])
             for st in bud.stores.values()]
    ran = stream_fast.TIMERS["native_park"] > park0
    if native:
        assert stream_fast._NATIVE is not None and not stream_fast._NATIVE.agent.error
    check_tier(native, tier_hits() - hits0, f"budget, bias {bias}")
    check_stage(native, st0, f"budget, bias {bias}")
    del model, bud, glus, cache
    ts.settle()
    return toks, lps, state, rings, ran


# Verify-shaped passes after a prefill; two 1-row passes first, since the prefill leaves no arena.
ROWS = [1, 1, 2, 1, 3, 4, 2, 5, 1, 4, 3, 1, 2, 4, 1]


def run_rows(out, native, bias, budget):
    """ROWS on the Python path or through the agent: each pass's logits, the layer state after them (as run and run_budget keep it) and the passes of 2+ rows the agent took, on an arena (or ring halves, t_max 4) that holds a 4-row union."""
    stream_fast._NATIVE_ROWS = True
    cfg = CFG8 if budget else ts.CFG
    model = ts.stock_model(out, cfg)
    stream_fast.install(model, out, resident=4 * cfg["num_experts_per_tok"],
                        cache_bias=bias, defer=False, native=native)
    glus = stream_fast.glus_of(model)
    if budget:
        bud = stream_fast.enable_budget(model, ts.budget_gb(), t_max=4)
        quota = {id(g): q for g, q in zip(glus, QUOTA)}
        bud._quotas = lambda pool: quota
    n0 = stream_fast.STATS["native_rows"]
    cache, rng, lps = make_prompt_cache(model), np.random.default_rng(5), []
    y = model(mx.array([ts.PROMPTS[0]], dtype=mx.int32), cache=cache)
    for T in ROWS:
        ids = [int(mx.argmax(y[0, -1]))] + rng.integers(0, 200, T - 1).tolist()
        y = model(mx.array([ids], dtype=mx.int32), cache=cache)
        lps.append(np.array(y.astype(mx.float32)))
    stream_fast.native_sync()
    if budget:
        at = {id(g): i for i, g in enumerate(glus)}
        state = [(list(g._lru.items()), sorted(g._ring.items()), g._seen.copy(),
                  g._ghost_hist.copy(), g._ghost_last.copy(), g._ghost_clock) for g in glus]
        state.append([(list(st.next), [None if o is None else (at[id(o[0])], o[1]) for o in st.owner])
                      for st in bud.stores.values()])
    else:
        state = [(list(g._lru.items()), g._seen.copy()) for g in glus]
    if native:
        assert stream_fast._NATIVE is not None and not stream_fast._NATIVE.agent.error
    rows = stream_fast.STATS["native_rows"] - n0
    stream_fast._NATIVE_ROWS = False
    del model, glus, cache
    if budget:
        del bud
    ts.settle()
    return lps, state, rows


def same(x, y):
    if isinstance(x, np.ndarray):
        return np.array_equal(x, y)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    return x == y


def main():
    if stream_fast.BACKEND != "cuda":
        sys.path.append(str(Path(__file__).resolve().parents[1] / "native" / "build"))
        try:
            import moe_native  # noqa: F401
        except ImportError:
            print("test_native: skipped (needs moe_native built by native/build.sh)")
            return
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "toy"
        ts.build(out)
        out8 = Path(d) / "toy8"
        ts.build(out8, CFG8)
        for bias in (0.0, 0.5):
            a = run(out, False, bias)
            b = run(out, True, bias)
            assert not a[3] and b[3], "the native arm must run native passes, the other none"
            assert a[0] == b[0], f"bias {bias}: tokens differ\n{a[0]}\n{b[0]}"
            bad = [i for i, (x, y) in enumerate(zip(a[1], b[1])) if not np.array_equal(x, y)]
            assert not bad, f"bias {bias}: logprobs differ at steps {bad}"
            for l, ((la, sa), (lb, sb)) in enumerate(zip(a[2], b[2])):
                assert la == lb, f"bias {bias}: layer {l} LRU {la} != {lb}"
                assert np.array_equal(sa, sb), f"bias {bias}: layer {l} _seen differs"
            print(f"bias {bias}: {len(a[0])} tokens, logprobs and LRU identical")
        for bias in (0.0, 0.5):
            a = run_budget(out8, False, bias)
            b = run_budget(out8, True, bias)
            assert not a[4] and b[4], "the native arm must run native passes, the other none"
            assert a[0] == b[0], f"budget, bias {bias}: tokens differ\n{a[0]}\n{b[0]}"
            bad = [i for i, (x, y) in enumerate(zip(a[1], b[1])) if not np.array_equal(x, y)]
            assert not bad, f"budget, bias {bias}: logprobs differ at steps {bad}"
            names = ("quota LRU", "_ring", "_seen", "ghost hist", "ghost last", "ghost clock")
            for l, (sa, sb) in enumerate(zip(a[2], b[2])):
                for name, x, y in zip(names, sa, sb):
                    assert same(x, y), f"budget, bias {bias}: layer {l} {name}: {x} != {y}"
            assert same(a[3], b[3]), f"budget, bias {bias}: ring halves {a[3]} != {b[3]}"
            used = sum(o is not None for _, owners in b[3] for o in owners)
            assert used, "no ring row was ever used"
            print(f"budget, bias {bias}: {len(a[0])} tokens, logprobs, quota LRU, rings "
                  f"({used} rows held) and ghost LRU identical")
        if stream_fast.BACKEND == "metal":
            want = sum(1 < T <= 4 for T in ROWS)
            for budget in (False, True):
                for bias in (0.0, 0.5):
                    what = f"rows {'budget' if budget else 'arena'}, bias {bias}"
                    a = run_rows(out8 if budget else out, False, bias, budget)
                    b = run_rows(out8 if budget else out, True, bias, budget)
                    assert a[2] == 0 and b[2] == want, f"{what}: native T-row passes {a[2]} / {b[2]} of {want}"
                    bad = [i for i, (x, y) in enumerate(zip(a[0], b[0])) if not np.array_equal(x, y)]
                    assert not bad, f"{what}: logits differ at passes {bad} (rows {[ROWS[i] for i in bad]})"
                    assert same(a[1], b[1]), f"{what}: layer state differs after the passes"
                    print(f"{what}: {len(ROWS)} passes ({want} native with 2-4 rows), logits and "
                          f"layer state identical")
    print("ok")


if __name__ == "__main__":
    main()
