#!/usr/bin/env python3
"""Budget mode (--mem-peak-gb): the shared expert store against the per-layer
arena and against stock mlx_lm, on a toy packed MoE, no real weights, a few
MB on disk.

  1. gather_qmm over a big shared tensor at scattered slots == over a small
     arena (4-bit and 2-bit, T 1 and 2); the kernel line _rhs_line probes
     (sorted and unsorted part at exactly 4 pairs per tensor row on MLX
     0.32, checked here on other data); and the
     sorted-rhs kernel gives a row the same result in padded groups as in one
     call. What the prefill's exactness rests on.
  2. Prefill logits bitwise equal to stock mlx_lm's at 16/31/32/40 positions
     (both sides of the kernel threshold, 32 on this toy) with the store at
     pool 0 (the first prefill: probed, then run whole), a small pool and a
     full one, and the arena's too. After a prefill, a 5- and a
     29-position pass (rounds, each gathering only its own rows) equal
     stock's too.
  3. A tiny qwen3_next with two expert shapes (4-bit layers 0/3, 2-bit 1/2,
     so two stores), packed to experts.bin: at cache_bias 0 the budget engine
     emits stock mlx_lm's greedy tokens over four turns, a fresh
     prefill, a 5-token prompt wider than the layer's slots, a 29-position
     sorted pass under the threshold (a forced-rows-shaped pass: rounds, the
     pool kept, no refit), a 39-position pass over it (a prefill: released,
     grouped), sync and run-ahead alike. Stock, not the arena: the arena's
     split pass gathers its residents per row where stock's sorted call
     takes the rhs kernel, so on the last turn it can differ from both.
  4. The pool never exceeds its target at any fit, the store is released
     before every prefill and is that prefill's only scratch (nothing in
     _WIDE_SCRATCH, no _dest rows), and the process's MLX peak stays within
     a budget that holds ~3 slots a layer beside the standing room (turns
     2-3), then within one that holds turn 4's prefill whole.
  5. cache_bias 0.5: runs, and the router's mask is the quota members only;
     under run-ahead, with the budget forced over every few decode passes (a
     refit mid-decode), no repaired layer misses.
  6. A prefill's measured need leaves attention's transient out: with
     chunkattn unbounded it equals the need with chunkattn held to
     min_rows, where counting attention in would not. And at a budget MLX
     already fills, the first prefill's attention is still chunked.
  7. A prefill with the pool at the whole store runs the same expert groups
     as one at pool 0, and the peak stays within the budget.
  8. A chunked prefill (stream_generate's steps) longer than any measured,
     the pool at the whole store: the pool gives way as the KV grows, and
     the peak stays within the budget.
  9. A pass that keeps the pool (20 positions) leaves the store as it was; a
     wide prefill gives up only what it needs (pool > 0, smaller), and the
     pass after it grows the pool back. The peak stays within the budget.
 10. A prefill that does not fit even at pool 0 is cut at the stack into
     equal pieces, the rest planned again at each piece's entry: every
     piece a prefill, each within the room it was planned in, and the
     hidden states and cache equal stock mlx_lm's fed the same pieces (not
     one pass: a cut changes the numerics, reported, not asserted).
 11. A prefill sized from a too-low prediction (the rate cut to a tenth)
     raises the rate to what it grew, once: the need of that same pass is
     then what it grew, not that times a headroom raised by the same miss.
 12. trim_kv (before a front's save under a budget): the save copies
     nothing, and the trimmed cache holds and grows as the untrimmed one.
 13. A front restored from SSD is lazy until evaluated: the budget counts
     it as held before the pass that follows it, so warm turns after it
     keep the pool (keep measured without it), a prefill after it is
     planned beside it and stays within the budget, and a new conversation
     measures keep again.
 14. plan(): a need over by a hair is two balanced pieces, not [n - 1, 1];
     a cut is equal pieces, and a room that moves by less than it takes to
     change their count cuts the same.
 15. A pass with no cache is never cut (== stock's one pass), and nothing
     holds the budget between passes: a dropped engine frees its memory.
 16. A pass that keeps the pool runs attention in one stock call where the
     room left holds it, chunked only down to that room otherwise.
 17. What a budget measured outlives its process (enable_budget's calib):
     a fresh engine on the same file starts from that rate, state, keep and
     host, plans its first prefill without the probe and still raises them;
     a missing or unreadable file is the probe again.
 18. Run-ahead on auto: decode runs ahead only while every quota holds a
     step (MemBudget.carries): none deferred and no rollback at pool 0,
     all of them with the whole store a pool; stock's tokens both ways.
 19. Pipelined rounds (_call_rounds reads round r+1 while the GPU gathers
     round r): the rows of serial rounds and of stock's SwitchGLU on every
     layer, at pool 0 and with a quota; no expert read lands in a slot a
     gather in flight reads (RaceWatch), which does catch it when the claim
     is not told the busy slots.

    python3 tests/test_store.py
"""
import faulthandler
import gc
import json
import os
import resource
import shutil
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.generate import generate_step
from mlx_lm.models import qwen3_next as q3n
from mlx_lm.models.cache import make_prompt_cache

import stream_fast

faulthandler.dump_traceback_later(270, exit=True)

# test_embed_stream.py's proven toy shape, with 32 experts at top_k 4 so a
# prefill's union (~25) is far wider than a small quota + ring half.
CFG = dict(
    model_type="qwen3_next", hidden_size=64, num_hidden_layers=4,
    intermediate_size=64, num_attention_heads=2, linear_num_value_heads=2,
    linear_num_key_heads=1, linear_key_head_dim=32, linear_value_head_dim=32,
    linear_conv_kernel_dim=4, num_experts=32, num_experts_per_tok=4,
    decoder_sparse_step=1, shared_expert_intermediate_size=64,
    mlp_only_layers=[], moe_intermediate_size=64, rms_norm_eps=1e-6,
    vocab_size=200, num_key_value_heads=1, rope_theta=1e7,
    partial_rotary_factor=0.25, max_position_embeddings=4096, head_dim=32,
    norm_topk_prob=True, full_attention_interval=2, tie_word_embeddings=False,
)
TWO_BIT = (1, 2)            # Qwen3.6-edge's shape: 2-bit middle, 4-bit edges
RES = 6                     # the plain engine's arena: >= top_k, < any prefill union
_rng = np.random.default_rng(1)
PROMPTS = [_rng.integers(0, 200, 40).tolist(),     # fresh: a prefill
           [7, 99, 3, 150, 42, 8],                 # 5 fed: narrow, > slots
           _rng.integers(0, 200, 30).tolist(),     # 29 fed: sorted, under 4*32 pairs
           _rng.integers(0, 200, 40).tolist()]     # 39 fed: over it, a prefill
GEN = [12, 8, 8, 8]


def check_gather_exact():
    rng = np.random.default_rng(0)
    for bits in (4, 2):
        w = mx.array(rng.standard_normal((8, 64, 64)).astype(np.float32))
        wq, sc, bi = mx.quantize(w, group_size=32, bits=bits)
        slots = rng.permutation(40)[:8]           # scattered in a 40-slot tensor
        big = []
        for a in (wq, sc, bi):
            b = mx.zeros((40,) + a.shape[1:], dtype=a.dtype)
            b[mx.array(slots)] = a
            big.append(b)
        for T in (1, 2):
            x = mx.array(rng.standard_normal((1, T, 1, 1, 64)).astype(np.float32)).astype(mx.bfloat16)
            idx = rng.integers(0, 8, (1, T, 4))
            small_y = mx.gather_qmm(x, wq, sc, bi, rhs_indices=mx.array(idx), transpose=True,
                                    group_size=32, bits=bits)
            big_y = mx.gather_qmm(x, *big, rhs_indices=mx.array(slots[idx]), transpose=True,
                                  group_size=32, bits=bits)
            assert mx.array_equal(small_y, big_y), (bits, T)

    # The kernel rule, and the padded-group identity _call_grouped rests on.
    w = mx.quantize(mx.array(rng.standard_normal((64, 128, 256)).astype(np.float32) * 0.05),
                    group_size=32, bits=4)

    def g(x, ws, r, srt):
        return mx.gather_qmm(x, *ws, rhs_indices=mx.array(r), transpose=True,
                             group_size=32, bits=4, sorted_indices=srt)
    for E in (8, 16):
        ws = [a[:E] for a in w]
        for B in (4 * E - 1, 4 * E):
            r = np.sort(rng.integers(0, E, B)).astype(np.uint32)
            x = mx.array(rng.standard_normal((B, 1, 256)).astype(np.float32)).astype(mx.bfloat16)
            same = bool(mx.array_equal(g(x, ws, r, True), g(x, ws, r, False)))
            assert same == (not stream_fast._takes_rhs(B, E)), (E, B, same)
    B = 1200
    r = np.sort(rng.integers(0, 64, B)).astype(np.uint32)
    x = mx.array(rng.standard_normal((B, 1, 256)).astype(np.float32)).astype(mx.bfloat16)
    whole = g(x, w, r, True)
    for G in (4, 8, 32):
        cuts = np.searchsorted(r, np.arange(G, 64, G))
        ys = []
        for gi, (r0, r1) in enumerate(zip([0, *cuts], [*cuts, B])):
            rr, xr = r[r0:r1] - gi * G, x[r0:r1]
            pad = stream_fast._rhs_rows(G) - len(rr)
            if pad > 0:
                rr = np.concatenate([rr, np.full(pad, rr[-1], dtype=rr.dtype)])
                xr = mx.concatenate([xr, mx.zeros((pad, 1, 256), dtype=xr.dtype)])
            ys.append(g(xr, [a[gi * G:(gi + 1) * G] for a in w], rr, True)[:r1 - r0])
        assert mx.array_equal(mx.concatenate(ys), whole), G
    print("1. gather_qmm over a 40-slot store at scattered slots == over an "
          "8-slot arena; sorted/unsorted part at 4E (E 8, 16); padded groups "
          "of 4-32 experts == one sorted call")


def fresh_model(cfg=CFG, dtype=None):
    m = q3n.Model(q3n.ModelArgs.from_dict(cfg))
    if dtype is not None:
        m.set_dtype(dtype)
    base = m.quant_predicate

    def pred(path, mod):
        if not hasattr(mod, "to_quantized"):
            return False
        r = base(path, mod)
        if r is True and ".switch_mlp." in path and any(f"layers.{i}." in path for i in TWO_BIT):
            return {"group_size": 32, "bits": 2, "mode": "affine"}
        return r
    nn.quantize(m, group_size=32, bits=4, class_predicate=pred)
    return m


def build(out, cfg=CFG, dtype=None):
    mx.random.seed(0)
    model = fresh_model(cfg, dtype)
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    out.mkdir(parents=True)
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model.safetensors" for k in weights}}))
    (out / "config.json").write_text(json.dumps(
        {**cfg, "quantization": {"group_size": 32, "bits": 4, "mode": "affine"}}))
    import repack
    argv, sys.argv = sys.argv, ["repack.py", "--src", str(out)]
    try:
        repack.main()
    finally:
        sys.argv = argv


def stock_model(out, cfg=CFG, dtype=None):
    model = fresh_model(cfg, dtype)
    model.load_weights(str(out / "model.safetensors"))
    mx.eval(model.parameters())
    return model


def engine(out, peak_gb=None, defer=False, cache_bias=0.0, cfg=CFG, dtype=None):
    model = stock_model(out, cfg, dtype)
    stream_fast.install(model, out, resident=RES,
                        cache_bias=cache_bias, defer=defer)
    bud = stream_fast.enable_budget(model, peak_gb) if peak_gb else None
    return model, bud


def budget_gb():
    return mx.get_active_memory() / 1e9 + 1.0


def settle():
    # The engine's layers hold each other (_next_glu, _all_glus): only the
    # collector frees them, and freed during the next engine's first prefill
    # they would shrink the reserve it measures. Callers `del` theirs first.
    gc.collect()
    mx.clear_cache()


def turns(model, between=None):
    cache = make_prompt_cache(model)
    got = []
    for i, (p, n) in enumerate(zip(PROMPTS, GEN)):
        if between:
            between(i)
        got.append([int(t) for t, _ in generate_step(
            mx.array(p, dtype=mx.int32), model, max_tokens=n, prompt_cache=cache)])
    return got


def prefill_logits(model, p):
    y = model(mx.array(p, dtype=mx.int32)[None], cache=make_prompt_cache(model))
    mx.eval(y)
    return y


def after_logits(model, prompts):
    """Logits of each pass over one cache: a prefill, then narrower passes."""
    cache, ys = make_prompt_cache(model), []
    for p in prompts:
        ys.append(model(mx.array(p, dtype=mx.int32)[None], cache=cache))
        mx.eval(ys[-1])
    return ys


class PeakWatch:
    """The most MLX held while open. The budget resets MLX's peak at every
    pass, so the resets are wrapped to keep what each one would drop."""

    def __init__(self):
        self.top, self._real = 0, mx.reset_peak_memory

    def __enter__(self):
        mx.reset_peak_memory = self._reset
        self._real()
        return self

    def _reset(self):
        self.top = max(self.top, mx.get_peak_memory())
        self._real()

    def __exit__(self, *exc):
        if mx.reset_peak_memory == self._reset:
            self._reset()
            mx.reset_peak_memory = self._real


class Passes:
    """Every measured pass's need against the room it was given and against
    what it grew, from the budget's own tick() and close(). The plan gives
    each pass the room for its need, exactly; a pass that grows past its
    need raises the rate (a prefill) or keep (any other pass) to what it
    grew. Identical toy passes move by up to ~70% run to run (a layer's
    intermediates can still be counted when the next layer's land), so a
    peak is held to its budget plus the most any pass grew past its need --
    what the budget then learns, and by nothing else."""

    def __init__(self, bud):
        self.bud, self.short, self.over = bud, [], 0.0
        tick, close = bud.tick, bud.close

        def ticked(n, fresh, caches):
            tick(n, fresh, caches)
            need = bud._pass[3]
            if need is not None:
                self.short.append(bud._held0 + bud._host + bud._have + bud.cache
                                  + need - bud.peak)

        def closed():
            p = bud._pass
            if p is not None and p[3] is not None:
                grew = max(bud._top, mx.get_peak_memory()) - bud._have - bud._held0
                self.over = max(self.over, grew - p[3])
            close()
        bud.tick, bud.close = ticked, closed

    def pages(self):
        # MLX gives a buffer past a page whole pages: `held` (active minus
        # the stores' bytes) carries what their tensors round up to, and a
        # rebuilt store rounds differently.
        return resource.getpagesize() * sum(len(b) for st in self.bud.stores.values()
                                            for b in st.bufs)

    def promised(self):
        assert self.short and max(self.short) <= self.pages(), ("a need past its room", self.short)

    def check(self, peak):
        self.promised()
        # The same page rounding shows in the peak, all under a page per store buffer.
        assert peak <= self.bud.peak + self.over + self.pages(), (peak, self.bud.peak, self.over)


def feeder(model, seed):
    """Feed k random tokens into one cache, logits evaluated."""
    rng, cache = np.random.default_rng(seed), make_prompt_cache(model)

    def feed(k):
        mx.eval(model(mx.array(rng.integers(0, 200, k).tolist(), dtype=mx.int32)[None],
                      cache=cache))
    return feed


def pool_bytes(bud):
    return sum(st.pool * st.slot_bytes for st in bud.stores.values())


def check_prefill_logits(out):
    rng = np.random.default_rng(5)
    ps = {T: rng.integers(0, 200, T).tolist() for T in (40, 32, 31, 16)}
    ref = {T: prefill_logits(stock_model(out), p) for T, p in ps.items()}
    # After a prefill, a 5-position pass (unsorted) and a 29-position one
    # (sorted, under stock's line): both go in rounds over the quota + ring.
    seq = [ps[40], rng.integers(0, 200, 5).tolist(), rng.integers(0, 200, 29).tolist()]
    ref_seq = after_logits(stock_model(out), seq)
    model, _ = engine(out)
    for T, p in ps.items():
        assert mx.array_equal(prefill_logits(model, p), ref[T]), ("arena", T)
    del model
    settle()
    model, bud = engine(out, peak_gb=budget_gb())
    glus = stream_fast.glus_of(model)
    per_layer = sum(bud.slot_bytes[g._skey] for g in glus)
    real_rounds, rounds = stream_fast.StreamingSwitchGLU._call_rounds, [0]

    def counted(glu, *a):
        rounds[0] += 1
        return real_rounds(glu, *a)
    stream_fast.StreamingSwitchGLU._call_rounds = counted
    seen = []
    for slots in (None, 2, 64):
        if slots is not None:
            # A decode pass closes the last prefill; then a budget whose
            # pool keeps `slots` a layer through a 40-position prefill (64:
            # past the whole store, so no prefill shrinks it). The store's
            # exactness is what this checks, not the plan: nothing is cut
            # (check 10 compares cut prefills).
            model(mx.array([[1]]), cache=cache)
            bud.plan = lambda n, fresh, caches: n
            held, host, _ = bud._measure()
            bud.peak = (held + host + bud.cache + bud.ring_bytes
                        + bud._need(40, True, True) + slots * per_layer)
            bud.fit(rebuild=True)
        for T, p in ps.items():
            cache = make_prompt_cache(model)
            y = model(mx.array(p, dtype=mx.int32)[None], cache=cache)
            mx.eval(y)
            assert mx.array_equal(y, ref[T]), (slots, T, bud.report())
        seen.append([st.slots for st in bud.stores.values()])
        before = rounds[0]
        for i, (y, r) in enumerate(zip(after_logits(model, seq), ref_seq)):
            assert mx.array_equal(y, r), ("rounds", slots, i, bud.report())
        assert rounds[0] > before, ("no pass after the prefill went in rounds", slots)
    stream_fast.StreamingSwitchGLU._call_rounds = real_rounds
    assert bud.probes == 1, bud.probes
    print(f"2. prefill logits == stock at T 16/31/32/40: arena, and the store at "
          f"slots per shape {seen} (pool 0 with the first prefill probed, small, "
          f"full); so are the 5- and 29-position passes after a prefill (rounds)")
    del model, bud, cache
    settle()


class Watch:
    """Asserts the budget's own invariants while the engine runs."""

    def __init__(self, bud):
        self.bud, self.fits, self.grouped = bud, 0, 0
        fit, grouped = bud._fit, stream_fast.StreamingSwitchGLU._call_grouped

        def checked_fit(rebuild, need):
            fit(rebuild, need)
            last = bud.last
            assert last["pool_mb"] <= last["pool_target_mb"] + 1e-9, last
            self.fits += 1

        def checked_grouped(glu, x, unique_ids, pos, rows, out, n_rows, store=None):
            if store is not None:
                for g in store.glus:        # released: nothing published, no parked read
                    assert not g._lru and not g._ring and not g._defer, g._half
                    assert g._slot_np is None or (g._slot_np < 0).all()
                assert not stream_fast._WIDE_SCRATCH
                self.grouped += 1
            return grouped(glu, x, unique_ids, pos, rows, out, n_rows, store)
        bud._fit = checked_fit
        self._grouped = grouped
        stream_fast.StreamingSwitchGLU._call_grouped = checked_grouped

    def close(self):
        stream_fast.StreamingSwitchGLU._call_grouped = self._grouped


def run_budget(out, defer=False, cache_bias=0.0):
    model, bud = engine(out, peak_gb=budget_gb(), defer=defer, cache_bias=cache_bias)
    w, passes = Watch(bud), Passes(bud)
    glus = stream_fast.glus_of(model)
    rounds = [0]
    real_rounds = stream_fast.StreamingSwitchGLU._call_rounds

    def counted(glu, *a):
        rounds[0] += 1
        return real_rounds(glu, *a)
    stream_fast.StreamingSwitchGLU._call_rounds = counted
    # (grouped, rounds, rebuilds, rows read) as each turn starts
    at, over = [], []
    tight, wide = PeakWatch(), PeakWatch()
    budgets = []

    def between(i):
        at.append((w.grouped, rounds[0], bud.rebuilds, stream_fast.STATS["rows_block"]))
        if i == 1:
            # Turn 1 measured a prefill; now a budget with ~3 slots a layer
            # beside the standing room, so turn 2 shrinks the store, evicts
            # and overflows, and turns 2-3 keep the pool within it.
            held, host, have = bud._measure()
            want = 3 * sum(bud.slot_bytes[g._skey] for g in glus)
            steady = bud._need(bud.min_w - 1, False, False)
            bud.peak = held + host + bud.cache + bud.ring_bytes + steady + want
            assert sum(len(g._lru) for g in glus) > 0
            before = bud.last["pool_mb"]
            bud.fit(rebuild=False)
            assert bud.last["pool_mb"] < before, (bud.last, before)
            passes.over = 0.0
            tight.__enter__()
        if i == 3:
            tight.__exit__()
            budgets.append(bud.peak)
            assert sum(len(g._lru) for g in glus) > 0, "pool empty before the prefill"
            # Room for turn 4's 39-position prefill whole (the tokens are
            # compared with stock's one pass) at pool 0.
            held, host, have = bud._measure()
            need = bud._need(len(PROMPTS[3]) - 1, False, True)
            bud.peak = max(bud.peak, held + host + bud.cache + bud.ring_bytes + need)
            over.append(passes.over)
            passes.over = 0.0
            wide.__enter__()
    try:
        got = turns(model, between)
    finally:
        w.close()
        stream_fast.StreamingSwitchGLU._call_rounds = real_rounds
        tight.__exit__()
        wide.__exit__()
    at.append((w.grouped, rounds[0], bud.rebuilds, stream_fast.STATS["rows_block"]))
    g, r, rb, rows = zip(*at)
    assert g[1] > 0 and g[3] == g[1] and g[4] > g[3], ("grouped per turn", g)
    assert r[2] > r[1] and r[3] > r[2], ("rounds per turn", r)
    # The 29-position turn is served from the quotas and the ring: nothing
    # released, nothing rebuilt when it ends.
    assert rb[3] == rb[2], ("the sorted non-prefill turn rebuilt the store", rb)
    assert not stream_fast._WIDE_SCRATCH
    assert all(not p._bufs for g_ in glus for p in g_._projs), "_dest rows allocated"
    assert len(bud.stores) == 2, "one store per expert shape"
    turn2 = {"rows_read": rows[3] - rows[2], "rebuilds": rb[3] - rb[2]}
    # Turn 4 is the first prefill on a cache that already holds positions:
    # its attention sees more keys than any fresh one measured (stock SDPA
    # here, its transient not capped). Each window may exceed its budget by
    # what a pass in it grew past its need (Passes), and by nothing else.
    passes.promised()
    peaks = {"turns 2-3": (tight.top, budgets[0], max(0.0, over[0])),
             "turn 4": (wide.top, bud.peak, max(0.0, passes.over))}
    return model, bud, got, w, peaks, r[-1], turn2


def check_bias_refit(out):
    """Run-ahead + cache_bias 0.5 with the budget forced over every few
    decode passes: a refit that clears quotas mid-decode. Ticked inside the
    first streaming layer, between its router and its gather, this raised
    'a repaired layer still misses' (the repair re-ranked against an empty
    hold)."""
    model, bud = engine(out, peak_gb=budget_gb(), defer=True, cache_bias=0.5)
    real, n, forced, roomy = bud.tick, [0], [0], bud.peak

    def tick(pos, fresh, caches):
        if pos * bud.top_k < stream_fast.SORT_MIN and bud.rate is not None:
            n[0] += 1
            if n[0] % 6 == 0:
                # Over by a byte: the next check shrinks the store, quotas
                # that three passes of claims have filled included.
                assert sum(len(g._lru) for g in bud.all) > 0
                held, host, have = bud._measure()
                bud.peak = held + have + host + bud.cache - 1
                forced[0] += 1
            elif n[0] % 6 == 3:
                bud.peak = roomy
                bud.fit(rebuild=True)
        return real(pos, fresh, caches)
    bud.tick = tick
    before = stream_fast.STATS["defer_rollbacks"]
    toks = [int(t) for t, _ in generate_step(
        mx.array(PROMPTS[0], dtype=mx.int32), model, max_tokens=90,
        prompt_cache=make_prompt_cache(model))]
    rb = stream_fast.STATS["defer_rollbacks"] - before
    assert len(toks) == 90 and forced[0] >= 10 and bud.rebuilds >= 2 * forced[0], (
        len(toks), forced[0], bud.rebuilds)
    print(f"5. run-ahead, cache_bias 0.5, budget forced over {forced[0]} times "
          f"mid-decode: {len(toks)} tokens, {bud.rebuilds} rebuilds, {rb} "
          f"rollbacks repaired, no repaired layer missed")
    del model, bud, tick, real
    settle()


def check_second_prefill(tmp):
    """A prefill with the pool at the whole store runs the same expert groups
    as one at pool 0 and stays within the budget. The groups used to be half
    the whole store, so they grew with the pool, and so did each group's
    gate/up/activation rows: at 400 positions this toy's second prefill grew
    11.3 MB against a 5.6 MB reserve, 5.7 MB over the budget, and the pool
    fell 256 -> 64 slots. The same prompt each time, so routing is not what
    moves."""
    cfg = {**CFG, "hidden_size": 128, "moe_intermediate_size": 128,
           "num_experts": 64, "num_experts_per_tok": 8}
    out = tmp / "wide_q3n"
    build(out, cfg)
    model, bud = engine(out, peak_gb=budget_gb(), cfg=cfg)
    glus = stream_fast.glus_of(model)
    p = mx.array(np.random.default_rng(7).integers(0, 200, 400), dtype=mx.int32)[None]
    n_all = sum(g._projs[0]._packed["n_experts"] for g in glus)
    whole = sum(bud.slot_bytes[g._skey] * g._projs[0]._packed["n_experts"] for g in glus)
    real = stream_fast.StreamingSwitchLinear._batch_wide
    seen = []

    def spied(lin, unique_ids, scratch=None):
        seen.append(tuple(int(e) for e in unique_ids))
        return real(lin, unique_ids, scratch)

    def prefill(pool):
        # A budget whose pool is `pool` bytes through this prefill, run
        # whole: the groups are what this checks, not the plan (a store
        # rebuilt for the budget rounds to other pages, and a need over by
        # those was cut in two, the pool taking the half it no longer needed).
        bud.plan = lambda n, fresh, caches: n
        held, host, _ = bud._measure()
        bud.peak = (held + host + bud.cache + bud.ring_bytes
                    + bud._need(p.shape[1], True, True) + pool)
        bud.fit(rebuild=True)
        seen.clear()
        passes = Passes(bud)
        cache = make_prompt_cache(model)
        with PeakWatch() as w:
            mx.eval(model(p, cache=cache))
            during = sum(st.pool for st in bud.stores.values())
            model(mx.array([[1]]), cache=cache)         # closes the prefill
        settle()
        passes.check(w.top + bud._measure()[1] + bud.cache)
        del bud.tick, bud.close                         # Passes' wrappers
        return during, list(seen), w.top + bud._measure()[1] + bud.cache

    stream_fast.StreamingSwitchLinear._batch_wide = spied
    try:
        cache = make_prompt_cache(model)
        mx.eval(model(p, cache=cache))              # measured, the probe first
        model(mx.array([[1]]), cache=cache)
        del cache
        settle()
        pool0, groups0, peak0 = prefill(0)
        pool, groups, peak = prefill(whole)
    finally:
        stream_fast.StreamingSwitchLinear._batch_wide = real
    # ~0: the room read at the prefill's own entry is a little above the
    # one the budget was set from, and what is free goes to the pool.
    assert pool0 < n_all // 8 and pool == n_all, (pool0, pool, n_all)
    assert groups == groups0, "the prefill groups moved with the pool"
    print(f"7. prefill at pool {pool} slots (the whole store): the same "
          f"{len(groups) // 3} expert groups as at pool {pool0}; peak {peak / 1e6:.2f} MB "
          f"(budget {bud.peak / 1e6:.2f} MB)")
    del model, bud
    settle()


def check_chunked_prefill(tmp):
    """A chunked prefill (stream_generate's prefill_step_size) longer than
    any measured: 12 chunks of 64 after a 3-chunk prefill measured them, the
    pool at the whole store. Each chunk is sized from its own need, the
    room re-read at its entry, so the pool gives way as the KV grows and the
    peak stays within the budget.

    bf16, attention through chunkattn as the engine installs it: stock SDPA
    left alone keeps its [heads, T, N] scores, which grow with every chunk's
    keys (+128 KB a chunk on this fp32 toy, 0.33 MB over the budget by the
    12th), where the engine caps them at the chunk's own MoE rows."""
    import chunkattn
    cfg = {**CFG, "hidden_size": 128, "moe_intermediate_size": 256,
           "num_experts": 64, "num_experts_per_tok": 8,
           "num_attention_heads": 8, "num_key_value_heads": 8}
    out = tmp / "chunk_q3n"
    build(out, cfg, mx.bfloat16)
    model, bud = engine(out, peak_gb=budget_gb(), cfg=cfg, dtype=mx.bfloat16)
    shape = (8, 8, cfg["head_dim"])
    chunkattn.install(model, table={shape: (chunkattn._kv_slope(*shape, mx.bfloat16), 5)})
    glus = stream_fast.glus_of(model)
    rng = np.random.default_rng(4)

    def prefill(chunks):
        p = mx.array(rng.integers(0, 200, 64 * chunks + 1).tolist(), dtype=mx.int32)
        for _ in generate_step(p, model, max_tokens=1, prefill_step_size=64,
                               prompt_cache=make_prompt_cache(model)):
            pass

    prefill(3)
    settle()
    held, host, _ = bud._measure()
    whole = sum(bud.slot_bytes[g._skey] * g._projs[0]._packed["n_experts"] for g in glus)
    bud.peak = (held + host + bud.cache + bud.ring_bytes
                + bud._need(64, True, True) + whole)
    bud.fit(rebuild=True)
    pool, rebuilds = pool_bytes(bud), bud.rebuilds
    passes = Passes(bud)
    try:
        with PeakWatch() as w:
            prefill(12)
    finally:
        chunkattn.layout = {}
    peak = w.top + bud._measure()[1] + bud.cache
    assert bud.pieces == [64] * 12, bud.pieces
    assert bud.rebuilds > rebuilds, "the pool never gave way to the chunks' KV"
    passes.check(peak)
    print(f"8. 12-chunk prefill, pool {pool / 1e6:.2f} MB (the whole store) "
          f"when it began: {bud.rebuilds - rebuilds} rebuilds as its KV grew, "
          f"peak {peak / 1e6:.2f} MB (budget {bud.peak / 1e6:.2f} MB)")
    del model, bud
    settle()


def check_reserve_attention(tmp):
    """A bf16 toy with 16 attention heads and a 1000-token prompt, so its
    unfused attention transient (32 MB of scores) outweighs the rest's
    (~10 MB, which moves ~5-10% run to run), and chunkattn routing that shape
    (min_rows 5, exactness is not what this checks)."""
    import chunkattn
    cfg = {**CFG, "num_attention_heads": 16}
    out = tmp / "att_q3n"
    build(out, cfg, mx.bfloat16)
    p = np.random.default_rng(3).integers(0, 200, 1000).tolist()

    def first_need(row_elems, bracket=True):
        model, bud = engine(out, peak_gb=budget_gb(), cfg=cfg, dtype=mx.bfloat16)
        bud._row_elems = row_elems
        chunkattn.install(model, table={(16, 1, 32): (1.0, 5)})
        if not bracket:
            bud.attention = lambda start: None
        bud.plan = lambda n, fresh, caches: n               # one measured pass, no probe
        cache = make_prompt_cache(model)
        mx.eval(model(mx.array(p, dtype=mx.int32)[None], cache=cache))
        model(mx.array([[1]]), cache=cache)         # closes the prefill
        r = bud._need(len(p), True, True)           # what that one pass grew
        del model, bud, cache
        settle()
        return r

    def tight_first_prefill():
        # A budget of what MLX holds before the model is even built: the
        # room is negative from the start, and the cap must still be the
        # MoE-row ceiling, not unbounded.
        rows, real = [], chunkattn.chunked
        chunkattn.chunked = lambda *a: (rows.append(a[4]), real(*a))[1]
        try:
            model, bud = engine(out, peak_gb=mx.get_active_memory() / 1e9, cfg=cfg,
                                dtype=mx.bfloat16)
            chunkattn.install(model, table={(16, 1, 32): (1.0, 5)})
            mx.eval(model(mx.array(p, dtype=mx.int32)[None], cache=make_prompt_cache(model)))
        finally:
            chunkattn.chunked = real
        del model, bud
        settle()
        return rows
    try:
        free = first_need(None)                     # attention unbounded
        tight = first_need(1)                       # a ceiling of N bytes: min_rows
        counted = first_need(None, bracket=False)
        rows = tight_first_prefill()
    finally:
        chunkattn.layout = {}
        stream_fast.set_prefill_budget_gb(None)
    # Two engines' passes: 10.85 against 9.84 MB once in ~30 runs, where
    # counting attention in is ~4x.
    assert abs(free - tight) <= 0.25 * tight, (free, tight)
    assert counted > 2 * free, (counted, free)
    assert rows and max(rows) < len(p), ("attention unbounded under a tight budget", rows)
    print(f"6. a 1000-position prefill's need: {free / 1e6:.2f} MB with attention "
          f"free, {tight / 1e6:.2f} MB with it at min_rows; counting attention in "
          f"would have made it {counted / 1e6:.2f} MB. At a budget MLX already "
          f"fills, the first prefill's attention ran in {len(rows)} chunked "
          f"calls of at most {max(rows)} rows")


def check_pool_kept(out):
    """Time-sharing: the pool stands beside the room the widest pass that
    keeps it needs, and only a prefill takes from it, only what it needs."""
    model, bud = engine(out, peak_gb=budget_gb())
    feed = feeder(model, 9)
    feed(40)
    for _ in range(4):
        feed(1)
    # A standing pool that a 100-position prefill cuts to ~3 slots a layer.
    per_layer = sum(bud.slot_bytes[g._skey] for g in bud.all)
    held, host, _ = bud._measure()
    steady = bud._need(bud.min_w - 1, False, False)
    wide = bud._need(100, False, True)
    assert wide > steady
    bud.peak = held + host + bud.cache + bud.ring_bytes + wide + 3 * per_layer
    bud.fit(rebuild=False)
    passes = Passes(bud)
    with PeakWatch() as w:
        feed(1)
        standing, rebuilds = pool_bytes(bud), bud.rebuilds
        feed(20)                                    # keeps the pool: rounds
        assert (pool_bytes(bud), bud.rebuilds) == (standing, rebuilds), (
            "a 20-position pass moved the pool", standing, bud.last)
        assert sum(len(g._lru) for g in bud.all) > 0
        feed(100)                                   # a prefill, not the first
        during = pool_bytes(bud)
        feed(1)
        after = pool_bytes(bud)
    peak = w.top + bud._measure()[1] + bud.cache
    assert 0 < during < standing and after > during, (standing, during, after)
    passes.check(peak)
    print(f"9. a 20-position pass kept the {standing / 1e3:.0f} KB pool; a "
          f"100-position prefill gave up {(standing - during) / 1e3:.0f} KB of "
          f"it, {during / 1e3:.0f} KB left, back to {after / 1e3:.0f} KB after; "
          f"peak {peak / 1e6:.2f} MB (budget {bud.peak / 1e6:.2f} MB)")
    del model, bud
    settle()


def check_cut_prefill(out):
    """A fresh 200-position prefill with room for ~80 at pool 0: cut at the
    stack into equal pieces, the rest planned again at each piece's entry
    in the room the pieces before left. Every piece is a prefill (no
    remainder narrower than min_w), fits the room it was planned in
    (Passes), and the hidden states and cache equal stock mlx_lm's fed the
    same pieces."""
    model, bud = engine(out, peak_gb=budget_gb())
    feed = feeder(model, 11)
    feed(40)
    feed(1)
    assert bud.probes == 1
    real_tick, widths = bud.tick, []

    def tick(n, fresh, caches):
        widths.append(n)
        return real_tick(n, fresh, caches)
    p = mx.array(np.random.default_rng(12).integers(0, 200, 200).tolist(), dtype=mx.int32)[None]
    settle()
    held, host, _ = bud._measure()
    bud.peak = held + host + bud.cache + bud.ring_bytes + bud._need(80, True, True)
    bud.tick = tick
    passes = Passes(bud)
    cache = make_prompt_cache(model)
    with PeakWatch() as w:
        h = model.model(p, cache=cache)             # the stack: hidden states
        mx.eval(h)
        pieces = list(widths)
        model(mx.array([[1]]), cache=cache)
    peak = w.top + bud._measure()[1] + bud.cache
    assert len(pieces) > 1 and sum(pieces) == 200 and min(pieces) >= bud.min_w, pieces
    assert bud.pieces == pieces and bud.probes == 1, (pieces, bud.pieces, bud.probes)
    passes.check(peak)
    stock = stock_model(out)
    ref_cache, ys, a = make_prompt_cache(stock), [], 0
    for k in pieces:
        ys.append(stock.model(p[:, a:a + k], cache=ref_cache))
        a += k
    stock(mx.array([[1]]), cache=ref_cache)
    assert mx.array_equal(h, mx.concatenate(ys, axis=1)), "a piece differs from stock's"
    for c, r in zip(cache, ref_cache):
        for x, y in zip(c.state, r.state):
            assert mx.array_equal(x, y), "the cache differs from stock's"
    one = mx.array_equal(h, stock.model(p, cache=make_prompt_cache(stock))).item()
    print(f"10. a 200-position prefill with room for ~80: cut into {pieces} (min_w "
          f"{bud.min_w}), each within its room, peak {peak / 1e6:.2f} MB (budget "
          f"{bud.peak / 1e6:.2f} MB); hidden states and cache == stock fed the same "
          f"pieces (== stock in one pass: {one})")
    del model, bud, stock, cache
    settle()


def check_rate_once(out):
    """A prediction that turns out too low: the per-position rate cut to a
    tenth, as a bad measurement would leave it. The prefill sized from it
    grows past its need, and the rate rises to what it grew, once: the
    need of that same pass is then what it grew, and the next prefill is
    sized from the rate alone. A headroom raised by the same miss used to
    size it again on top, and never forget it (the next one ran as
    [32, 32, 32], ~3x over what it grew)."""
    model, bud = engine(out, peak_gb=budget_gb())
    feed = feeder(model, 13)
    feed(40)
    feed(1)
    real_close, seen = bud.close, []

    def close():
        p = bud._pass
        if p is not None and p[2]:                  # a prefill: n, need, copy, grew
            seen.append((p[0], p[3], p[5],
                         max(bud._top, mx.get_peak_memory()) - bud._have - bud._held0))
        real_close()
    bud.close = close
    bud.rate /= 10
    feed(100)
    feed(1)
    n, sized, copy, grew = seen[-1]
    assert grew > sized, (sized, grew)
    again = bud._need(n, False, True, copy)
    assert abs(again - grew) <= 1e-6 * grew, (again, grew)
    feed(100)
    feed(1)
    n2, sized2, copy2, grew2 = seen[-1]
    assert abs((sized2 - copy2) - (grew - copy)) <= 1e-6 * grew, (sized2, copy2, grew, copy)
    print(f"11. a prefill sized from a tenth of the measured rate grew {grew / sized:.2f}x "
          f"its need; the rate rose to it once: the next was sized {sized2 / 1e6:.2f} MB "
          f"and grew {grew2 / 1e6:.2f} MB ({sized2 / grew2:.2f}x)")
    del model, bud
    settle()


def check_trim_kv(tmp):
    """stream_fast.trim_kv, run before a front's save under a budget: the
    save then copies nothing, and the trimmed cache holds and grows exactly
    as the untrimmed one."""
    from mlx_lm.models.cache import ArraysCache, KVCache, save_prompt_cache

    def cache():
        rng, cs = np.random.default_rng(14), []
        for _ in range(4):
            c = KVCache()
            k = mx.array(rng.standard_normal((1, 2, 300, 64)).astype(np.float32))
            c.update_and_fetch(k, k * 2)
            mx.eval(c.keys, c.values)
            cs.append(c)
        a = ArraysCache(2)
        a.cache = [mx.ones((1, 3, 64)), mx.ones((1, 2, 8, 8))]
        mx.eval(a.cache)
        return cs + [a]

    def save_grew(cs):
        mx.synchronize()
        a0 = mx.get_active_memory()
        mx.reset_peak_memory()
        save_prompt_cache(str(tmp / "trim.safetensors"), cs)
        mx.synchronize()
        # The peak reads 0 when nothing was allocated since the reset.
        return max(0, mx.get_peak_memory() - a0)

    whole, trimmed = cache(), cache()
    one = whole[0].nbytes
    untrimmed = save_grew(whole)
    stream_fast.trim_kv(trimmed)
    for c, d in zip(whole[:4], trimmed[:4]):
        assert d.offset == c.offset == d.keys.shape[2] < c.keys.shape[2]
        assert all(mx.array_equal(x, y) for x, y in zip(c.state, d.state))
    grew = save_grew(trimmed)
    # 4 caches of 300 positions in 512: 2.3 of one untrimmed, none trimmed.
    assert untrimmed > 2 * one and grew < one / 10, (untrimmed, grew, one)
    k = mx.array(np.random.default_rng(15).standard_normal((1, 2, 10, 64)).astype(np.float32))
    for c, d in zip(whole[:4], trimmed[:4]):
        assert all(mx.array_equal(x, y)
                   for x, y in zip(c.update_and_fetch(k, k), d.update_and_fetch(k, k)))
    print(f"12. trim_kv: a save grew MLX {untrimmed / 1e6:.2f} MB untrimmed, "
          f"{grew / 1e6:.2f} MB trimmed; same values, and 10 more positions "
          f"append the same")


def check_restored_front(tmp, out):
    """A front restored from SSD (pcache.load_front, load_prompt_cache) is
    lazy: nothing of it is in MLX until something evaluates it. Read
    around it, the budget took it for the growth of the pass that loaded
    it: a warm turn after it set keep to about the front, and the pool
    was gone for good (the 16 GB machine: keep 172 MB, pool 0 from the second request),
    and a prefill after it was planned in a room that still counted the
    front as free. Now both passes begin with it held. And a new
    conversation measures keep again, from a pass as wide as the one
    it had, not from a decode token."""
    from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache
    model, bud = engine(out, peak_gb=budget_gb())
    rng = np.random.default_rng(21)

    def fed(cache, k):
        mx.eval(model(mx.array(rng.integers(0, 200, k).tolist(), dtype=mx.int32)[None],
                      cache=cache))
    cache = make_prompt_cache(model)
    fed(cache, 2000)                                # the front
    fed(cache, 1)
    stream_fast.trim_kv(cache)
    path = str(tmp / "front.safetensors")
    save_prompt_cache(path, cache)
    front = sum(c.nbytes for c in cache)
    for _ in range(3):
        fed(cache, 1)                               # the first request's decode
    per_layer = sum(bud.slot_bytes[g._skey] for g in bud.all)
    held, host, _ = bud._measure()
    bud.peak = (held + host + bud.cache + bud.ring_bytes
                + bud._need(bud.min_w - 1, False, False) + 6 * per_layer)
    bud.fit(rebuild=True)
    pool, rebuilds = pool_bytes(bud), bud.rebuilds
    assert pool > 0

    def restore():
        # StreamingLLM.reset, then _reuse_front
        nonlocal cache
        bud.reset()
        del cache
        settle()
        a0, held = mx.get_active_memory(), bud._measure()[0]
        cache = load_prompt_cache(path)
        assert mx.get_active_memory() - a0 < front / 10, "the restore was not lazy"
        return held

    restore()
    passes = Passes(bud)
    with PeakWatch() as w:
        fed(cache, 20)                              # a warm turn: keeps the pool
        for _ in range(3):
            fed(cache, 1)
    peak = w.top + bud._measure()[1] + bud.cache
    assert (pool_bytes(bud), bud.rebuilds) == (pool, rebuilds), (
        "a warm turn after a restored front moved the pool", pool, bud.last)
    assert bud.keep < front / 4, (bud.keep, front)
    passes.check(peak)
    warm = bud.keep
    del bud.tick, bud.close

    # A conversation's keep, handed on as a lazy restore used to leave it.
    bud.keep = front
    restore()
    fed(cache, 1)
    fed(cache, 1)                                   # closes the decode token
    assert bud.keep == front, "a decode token stood in for a warm turn"
    fed(cache, 20)
    fed(cache, 1)
    assert bud.keep < front / 4, (bud.keep, front)

    # A prefill after it, in exactly the room it needs beside the front.
    base = restore()
    held, host, _ = bud._measure()
    bud.peak = (held + front + host + bud.cache + bud.ring_bytes
                + bud._need(200, False, True, stream_fast._copy(cache)))
    real_tick, first = bud.tick, []

    def tick(n, fresh, caches):
        real_tick(n, fresh, caches)
        first.append(bud._held0)
    bud.tick = tick
    passes = Passes(bud)
    with PeakWatch() as w:
        fed(cache, 200)
        pieces = list(bud.pieces)
        fed(cache, 1)
    peak = w.top + bud._measure()[1] + bud.cache
    assert first[0] >= base + front, ("the front was not held when the prefill began",
                                      first[0] - base, front)
    passes.check(peak)
    print(f"13. a front restored from SSD ({front / 1e6:.2f} MB, lazy until evaluated): "
          f"a warm turn after it kept the {pool / 1e3:.0f} KB pool, keep "
          f"{warm / 1e3:.0f} KB; a new conversation's keep came from its warm turn, "
          f"not a decode token; a 200-position prefill after it began with the front "
          f"held, ran as {pieces}, peak {peak / 1e6:.2f} MB (budget {bud.peak / 1e6:.2f} MB)")
    del model, bud, cache
    settle()


def check_plan(out):
    """plan() with the room set by hand (_measure stubbed, no KV between the
    pieces): a need over by a hair is two balanced pieces, not [n - 1, 1];
    a cut is equal pieces; and a room that moves by less than it takes to
    change their count cuts the same (the widest that fits followed it
    position by position)."""
    model, bud = engine(out, peak_gb=budget_gb())
    feed = feeder(model, 31)
    feed(40)
    feed(1)
    real = bud._measure
    bud._measure = lambda: (0.0, 0.0, 0.0)

    def cut(n, room):
        bud.peak = bud.cache + bud.ring_bytes + room
        widths, a = [], 0
        while a < n:
            widths.append(bud.plan(n - a, a == 0, []))
            a += widths[-1]
        return widths
    try:
        hair = cut(200, bud._need(200, True, True) - 1)
        room = bud._need(80, True, True)
        cuts = [cut(200, room + d * bud.rate) for d in (-6, 0, 6)]
    finally:
        bud._measure = real
    assert hair == [100, 100], hair
    assert cuts[0] == cuts[1] == cuts[2] == [67, 67, 66], cuts
    print(f"14. plan(): a need over by a hair cut as {hair}; room for 80 +- 6 "
          f"positions cuts 200 as {cuts[1]} each time")
    del model, bud
    settle()


def check_no_cache_and_release(out):
    """A pass with no cache (a caller scoring a whole text) is never cut: a
    piece could not attend to the positions before it. And nothing holds
    the budget between passes: kept in stream_fast._MEASURING, it held a
    dropped engine and its model, until the next pass of another."""
    settle()
    base = mx.get_active_memory()
    model, bud = engine(out, peak_gb=budget_gb())
    feed = feeder(model, 11)
    feed(40)
    feed(1)
    assert stream_fast._MEASURING is None, "the budget is held between passes"
    settle()
    held, host, _ = bud._measure()
    bud.peak = held + host + bud.cache + bud.ring_bytes + bud._need(80, True, True)
    real_tick, widths = bud.tick, []

    def tick(n, fresh, caches):
        widths.append(n)
        return real_tick(n, fresh, caches)
    bud.tick = tick
    p = mx.array(np.random.default_rng(12).integers(0, 200, 200).tolist(), dtype=mx.int32)[None]
    h = model.model(p)
    mx.eval(h)
    stock = stock_model(out)
    assert widths == [200] and mx.array_equal(h, stock.model(p)), widths
    live = mx.get_active_memory()
    del model, bud, feed, tick, real_tick, stock, h
    settle()
    left = mx.get_active_memory() - base
    assert left < (live - base) / 10, (left, live - base)
    print(f"15. a 200-position pass with no cache ran whole (== stock); a dropped "
          f"engine left {left / 1e3:.0f} KB of its {(live - base) / 1e6:.2f} MB")


def check_warm_attention(tmp):
    """A pass that keeps the pool (20 positions over ~1000 keys) runs
    attention in one stock call where the room left holds its transient --
    held to its own MoE rows, it was chunks of min_rows, never below
    those rows where the room is gone, and within a budget with no pool."""
    import chunkattn
    cfg = {**CFG, "hidden_size": 128, "moe_intermediate_size": 256,
           "num_experts": 64, "num_experts_per_tok": 8,
           "num_attention_heads": 8, "num_key_value_heads": 8}
    out = tmp / "warm_q3n"
    build(out, cfg, mx.bfloat16)
    model, bud = engine(out, peak_gb=budget_gb(), cfg=cfg, dtype=mx.bfloat16)
    shape = (8, 8, cfg["head_dim"])
    per_key = chunkattn._kv_slope(*shape, mx.bfloat16)
    chunkattn.install(model, table={shape: (per_key, 5)})
    rows, real = [], chunkattn.chunked
    chunkattn.chunked = lambda *a: (rows.append(a[4]), real(*a))[1]
    rng = np.random.default_rng(6)
    cache = make_prompt_cache(model)

    def warm():
        # 20 positions, and a decode token to close them
        rows.clear()
        for k in (20, 1):
            mx.eval(model(mx.array(rng.integers(0, 200, k).tolist(), dtype=mx.int32)[None],
                          cache=cache))
        return list(rows), cache[3].offset - 1
    try:
        mx.eval(model(mx.array(rng.integers(0, 200, 1000).tolist(), dtype=mx.int32)[None],
                      cache=cache))
        roomy, _ = warm()
        room = bud.room
        bud.room = lambda: 0.0
        gone, n_keys = warm()
        bud.room = room
        held, host, _ = bud._measure()
        bud.peak = (held + host + bud.cache + bud.ring_bytes
                    + bud._need(bud.min_w - 1, False, False))
        bud.fit(rebuild=False)
        passes = Passes(bud)
        with PeakWatch() as w:
            tight, _ = warm()
        peak = w.top + bud._measure()[1] + bud.cache
    finally:
        chunkattn.chunked = real
        chunkattn.layout = {}
    own = max(5, int(20 * stream_fast.moe_row_elems(model) * 2 // (n_keys * per_key)) or 1)
    assert not roomy, ("a warm turn's attention was chunked with room to spare", roomy)
    assert gone and set(gone) == {own}, (gone, own)
    passes.check(peak)
    print(f"16. a 20-position pass over {n_keys} keys: one stock call in a roomy budget, "
          f"chunks of {own} rows (its own MoE rows) with no room left; with no pool "
          f"{'one stock call' if not tight else f'chunks of {min(tight)} rows'}, peak "
          f"{peak / 1e6:.2f} MB (budget {bud.peak / 1e6:.2f} MB)")
    del model, bud, cache
    settle()


def check_calibration(tmp, out):
    """What a budget measured outlives its process (enable_budget's calib,
    the file moe_stream.py keys per setup): a fresh engine on the same file
    starts from that rate, state, keep and host, so its first prefill is
    planned without the probe, on a restarted 2 GB server, whose front
    comes back from SSD and whose turns are no prefill, nothing would
    measure a rate at all and the pool would stay 0, and it still raises
    them. A missing or unreadable file is the probe, as before."""
    path = tmp / "calib" / "membudget.json"

    def fresh():
        model = stock_model(out)
        stream_fast.install(model, out, resident=RES, cache_bias=0.0)
        return model, stream_fast.enable_budget(model, budget_gb(), calib=path)

    def prefill(model):
        # A fresh 40-position prefill, then a decode token to close it.
        cache = make_prompt_cache(model)
        for p in (PROMPTS[0], [5]):
            mx.eval(model(mx.array(p, dtype=mx.int32)[None], cache=cache))

    model, bud = fresh()
    assert bud.rate is None and bud.state is None
    bud._host_top = 123.4e6             # this test's host is 0 (main); a real one is not
    prefill(model)
    assert bud.probes == 1, bud.probes
    saved = json.loads(path.read_text())
    assert saved == bud._calibration() and saved["rate"] > 0, saved
    assert saved["host_mb"] == 124, saved
    del model, bud
    settle()

    model, bud = fresh()
    assert bud._calibration() == saved and bud._host_now() == 124e6, (bud._calibration(), saved)
    bud._host_top = 123.4e6             # its own host replaces the file's once it has run as many passes
    bud.rate = saved["rate"] / 2        # measured low: this run's prefill raises it
    prefill(model)
    assert bud.probes == 0, "a budget started from the file still probed"
    assert bud.rate > saved["rate"] / 2 and json.loads(path.read_text()) == bud._calibration()
    del model, bud
    settle()

    path.write_text("{not json")
    model, bud = fresh()
    prefill(model)
    assert bud.probes == 1, "an unreadable file skipped the probe"
    print(f"17. calibration: rate {saved['rate'] / 1e3:.1f} KB/position, state "
          f"{saved['state'] / 1e3:.0f} KB, keep {saved['keep'] / 1e3:.0f} KB, host "
          f"{saved['host_mb']} MB kept across engines; the second planned its prefill "
          f"with no probe and raised a rate set low; an unreadable file probed again")
    del model, bud
    settle()


def check_carries(out):
    """Run-ahead on auto (the server's default), budget mode: decode runs
    ahead only while every quota can hold a step (MemBudget.carries). With
    the pool cut to 0 right after the prefill no decode pass is deferred and
    nothing rolls back; with the whole store a pool every one is. Stock's
    tokens either way."""
    p, n = PROMPTS[0], 10
    ref = [int(t) for t, _ in generate_step(mx.array(p, dtype=mx.int32), stock_model(out),
                                            max_tokens=n)]
    got = {}
    for cut in (False, True):
        model, bud = engine(out, peak_gb=budget_gb(), defer="auto")
        real = bud.tick

        def tick(pos, fresh, caches):
            if cut and pos * bud.top_k < stream_fast.SORT_MIN and bud.rate is not None:
                # A byte past the ring and the standing need: the refit
                # after the prefill leaves every quota empty.
                held, host, _ = bud._measure()
                bud.peak = (held + host + bud.cache + bud.ring_bytes
                            + bud._need(bud.min_w - 1, False, False) + 1)
            return real(pos, fresh, caches)
        bud.tick = tick
        s0 = {k: stream_fast.STATS[k] for k in ("defer_tokens", "defer_rollbacks")}
        toks = [int(t) for t, _ in generate_step(mx.array(p, dtype=mx.int32), model,
                                                 max_tokens=n,
                                                 prompt_cache=make_prompt_cache(model))]
        got[cut] = {k: stream_fast.STATS[k] - v for k, v in s0.items()}
        q = [len(g._qslots) for g in bud.all]
        assert toks == ref, (cut, toks, ref)
        assert (max(q) == 0) == cut and bud.carries(1) != cut, (cut, q)
        del model, bud, tick, real
        settle()
    assert got[True] == {"defer_tokens": 0, "defer_rollbacks": 0}, got
    assert got[False]["defer_tokens"] >= n - 1, got
    print(f"18. run-ahead where every quota holds a step: {got[False]['defer_tokens']} "
          f"deferred decode passes with the whole store a pool, none (and no "
          f"rollback) at pool 0; stock's {n} tokens both ways")


class RaceWatch:
    """Every expert read into a store slot against the gathers the GPU may
    still run from that slot: built and not handed over yet, or handed over
    by an async_eval nobody has waited for. MLX's GPU stream runs its
    command buffers in order, so eval(a) of an array an async_eval handed
    over waits for that submission and every one before it; any other eval,
    or a synchronize, waits for everything. Layer by layer only: there every
    gather's rhs is a host array (run-ahead's are lazy)."""

    def __init__(self, bud):
        self.bud, self.lock = bud, threading.Lock()
        self.built, self.flight, self.n, self.writes, self.bad = [], [], 0, 0, []
        self.real = (mx.gather_qmm, mx.eval, mx.async_eval, mx.synchronize,
                     stream_fast._do_read)

    def _slot(self, w=None, iov=None):
        for st in self.bud.stores.values():
            if w is not None and any(b[0] is w for b in st.bufs):
                return st, None
            for per in st.iov if iov is not None else ():
                for s, lst in enumerate(per):
                    if lst is iov:
                        return st, s
        return None, None

    def __enter__(self):
        gather, ev, aev, sync, read = self.real

        def gather_qmm(x, w, *a, **k):
            st, _ = self._slot(w=w)
            if st is not None:
                with self.lock:
                    self.built.append((st, set(np.asarray(k["rhs_indices"]).reshape(-1).tolist())))
            return gather(x, w, *a, **k)

        def async_eval(*arrs):
            with self.lock:
                self.n += 1
                self.flight.append((self.n, self.built, arrs))
                self.built = []
            aev(*arrs)

        def eval_(*arrs):
            with self.lock:
                k = next((n for n, _, held in self.flight
                          if len(arrs) == 1 and any(a is arrs[0] for a in held)), None)
            ev(*arrs)
            with self.lock:
                self.flight = [f for f in self.flight if k is not None and f[0] > k]
                self.built = self.built if k is not None else []

        def synchronize(*a):
            sync(*a)
            with self.lock:
                self.flight, self.built = [], []

        def do_read(job):
            st, s = self._slot(iov=job[1])
            if st is not None:
                with self.lock:
                    self.writes += 1
                    reading = self.built + [g for f in self.flight for g in f[1]]
                    if any(g_st is st and s in slots for g_st, slots in reading):
                        self.bad.append(s)
            return read(job)
        mx.gather_qmm, mx.eval, mx.async_eval, mx.synchronize = (
            gather_qmm, eval_, async_eval, synchronize)
        stream_fast._do_read = do_read
        return self

    def __exit__(self, *exc):
        (mx.gather_qmm, mx.eval, mx.async_eval, mx.synchronize,
         stream_fast._do_read) = self.real


def pool_slots(bud, slots):
    """A budget for `slots` a layer beside the standing room (0: the ring
    alone, the 2 GB setup), re-set at every decode-shaped pass's tick; the
    prefill's is left as it was. The pool grows to it after a prefill and
    shrinks to it at once."""
    if not hasattr(bud, "_slots"):
        real, per_layer = bud.tick, sum(bud.slot_bytes[g._skey] for g in bud.all)

        def tick(pos, fresh, caches):
            if pos * bud.top_k < stream_fast.SORT_MIN and bud.rate is not None:
                stream_fast._eval_caches(caches)       # held as the tick reads it
                held, host, _ = bud._measure()
                bud.peak = (held + host + bud.cache + bud.ring_bytes + 1
                            + bud._need(bud.min_w - 1, False, False, stream_fast._copy(caches))
                            + bud._slots * per_layer)
            return real(pos, fresh, caches)
        bud.tick = tick
    bud._slots = slots


def check_top_k_slots(tmp):
    """A block routing more experts than 8 keeps a usable arena: the slot
    width is the block's top_k, not the 8 every layer used to default to."""
    cfg = dict(CFG, num_experts_per_tok=10)
    out = tmp / "topk10"
    build(out, cfg)
    model = stock_model(out, cfg)
    stream_fast.install(model, out, resident=10)
    glus = stream_fast.glus_of(model)
    assert glus, "the top_k-10 toy streamed no MoE layer"
    widths = [g._projs[0]._slots for g in glus]
    assert all(w == 10 for w in widths), widths
    # The arena must not shrink under one token's ten experts: a cap of 8
    # sends every layer to the read path while still holding its slots.
    glu = glus[0]
    glu.resize(2)
    assert glu._cap == 10, glu._cap
    del model, glus, glu
    settle()
    model = stock_model(out, cfg)
    stream_fast.install(model, out, resident=12)
    ref = turns(stock_model(out, cfg))
    got = turns(model)
    assert got == ref, "top_k 10 changed the tokens at cache_bias 0"
    assert all(g._lru for g in stream_fast.glus_of(model)), \
        "the arena path never ran at top_k 10"
    print(f"top_k 10: slots {widths[0]}, the arena held a token's experts "
          f"and the reply equals stock's")
    del model, ref, got
    settle()


def check_pipelined_rounds(out):
    """_call_rounds off run-ahead reads round r+1 while the GPU gathers round
    r. On a 15-position unsorted pass (60 pairs, ~7 rounds at pool 0): the
    same rows as the serial rounds and as stock's SwitchGLU, at pool 0 and
    with a quota of 3; no read lands in a slot a gather still in flight
    reads, and with the busy slots ignored in the claim, one does (the
    watch can see it)."""
    stock = stock_model(out)
    rng = np.random.default_rng(21)
    real_place = stream_fast.StreamingSwitchGLU._place_store

    def rounds(glu, li):
        idx = np.stack([rng.permutation(32)[:4] for _ in range(15)])[None].astype(np.int32)
        x_in = mx.array(rng.standard_normal((1, 15, 64)).astype(np.float32))
        unique_ids, _, pos = stream_fast._decode_indices(mx.array(idx), memo=False)
        x = mx.expand_dims(x_in, (-2, -3))
        with RaceWatch(bud) as w:
            y = glu._call_rounds(x, unique_ids, pos).squeeze(-2)
            mx.eval(y)
        return y, w, (x, unique_ids, pos), stock.layers[li].mlp.switch_mlp(x_in, mx.array(idx))
    seen = {}
    for slots in (3, 0):
        model, bud = engine(out, peak_gb=budget_gb())
        feed = feeder(model, 21)
        pool_slots(bud, slots)
        feed(40)
        for _ in range(3):
            feed(1)
        for li, glu in enumerate(bud.all):
            pipe, w, args, ref = rounds(glu, li)
            stream_fast._DEFER_ACTIVE = True
            try:
                serial = glu._call_rounds(*args).squeeze(-2)
                mx.eval(serial)
            finally:
                stream_fast._DEFER_ACTIVE = False
            assert mx.array_equal(pipe, serial) and mx.array_equal(pipe, ref), (slots, li)
            assert w.writes > 0 and not w.bad, (slots, li, w.bad)
            assert w.n > 0, "no round was handed to the GPU ahead of the next read"
            seen[slots, li] = (w.n + 1, len(glu._qslots))
        if slots:
            # The watch itself: the claim told nothing of the round on the GPU.
            stream_fast.StreamingSwitchGLU._place_store = (
                lambda g, ids, count, half=None, busy=(): real_place(g, ids, count, half))
            try:
                li, glu = max(enumerate(bud.all), key=lambda t: len(t[1]._qslots))
                _, caught, _, _ = rounds(glu, li)
            finally:
                stream_fast.StreamingSwitchGLU._place_store = real_place
            assert caught.bad, ("the watch saw no read into a busy slot", seen)
        del model, bud, feed, glu
        settle()
    assert any(q for (s, _), (_, q) in seen.items() if s) and not any(
        q for (s, _), (_, q) in seen.items() if not s), seen
    print(f"19. pipelined rounds, 15 positions x 4 experts: == serial rounds == stock "
          f"SwitchGLU on every layer at quotas of ~3 and 0 ((quota slots, layer): "
          f"(rounds, quota) {seen}); no read into a slot a gather in flight reads, and "
          f"the watch caught {len(caught.bad)} with the busy slots hidden from the claim")
    del stock
    settle()


def main():
    mx.set_cache_limit(0)
    # Host = 0: the toy's own process (Python, the test harness) is not what
    # is under test, and its footprint moves by more than the whole toy.
    stream_fast.phys_footprint = lambda: mx.get_active_memory() + mx.get_cache_memory()
    check_gather_exact()
    tmp = Path(tempfile.mkdtemp(prefix="store_"))
    try:
        out = tmp / "tiny_q3n"
        build(out)
        # An MLX whose kernel switch is not of _rhs_line's form: budget mode
        # refuses rather than run groups on a guessed kernel.
        line = stream_fast._rhs_line()
        stream_fast._RHS_LINE = line[:2] + (False,)
        try:
            engine(out, peak_gb=budget_gb())
        except RuntimeError as e:
            assert "kernel switch" in str(e), e
        else:
            raise AssertionError("budget mode ran on an unknown kernel line")
        finally:
            stream_fast._RHS_LINE = line
        settle()
        check_prefill_logits(out)
        ref = turns(stock_model(out))
        for defer in (False, True):
            name = "run-ahead" if defer else "sync"
            # Run-ahead with every pass sorted by slot (the T >= 2 verify path
            # at top-8, here from T 1): the row order then follows the slot
            # numbers, which the store assigns differently from the arena.
            stream_fast.DEFER_SORT_MIN = 4 if defer else 16
            model, bud, got, w, peaks, nr, turn2 = run_budget(out, defer=defer)
            print(f"3. {name}: stock {ref}\n   {' ' * len(name)}  store {got}")
            assert got == ref, "budget mode changed the tokens at cache_bias 0"
            rep = bud.report()
            q = [len(g._qslots) for g in stream_fast.glus_of(model)]
            print(f"4. {name}: {w.fits} fits, pool <= target at each; "
                  f"{w.grouped} prefill layers ran on a released store, {nr} "
                  f"rounds calls; 29-position turn: {turn2['rows_read']} rows "
                  f"read, {turn2['rebuilds']} rebuilds; quotas {q}, store "
                  f"{rep['store_mb'] * 1e3:.0f} KB, rate {rep['prefill_kb_per_pos']} "
                  f"KB/position, state {rep['state_mb'] * 1e3:.0f} KB, keep "
                  f"{rep['keep_mb'] * 1e3:.0f} KB; "
                  + "; ".join(f"peak {k} {p / 1e6:.2f} MB <= budget {b / 1e6:.2f} MB"
                              + (f" + {t / 1e3:.0f} KB over a need" if t else "")
                              for k, (p, b, t) in peaks.items()))
            for k, (p, b, t) in peaks.items():
                assert p <= b + t, (k, p, b, t)
            del model, bud, w
            settle()
        stream_fast.DEFER_SORT_MIN = 16

        model, bud, got, w, peaks, nr, _ = run_budget(out, cache_bias=0.5)
        for layer in model.layers:
            moe = stream_fast._moe_of(layer)
            if isinstance(moe, stream_fast.CacheBiasedMoE):
                glu = moe.block.switch_mlp
                assert moe._keys() == list(glu._lru), "bias mask is not the quota"
        print(f"5. cache_bias 0.5: {sum(len(t) for t in got)} tokens, the mask is "
              f"the quota members, {w.fits} fits within budget")
        del model, bud, w, moe, glu
        settle()
        check_bias_refit(out)
        check_reserve_attention(tmp)
        check_second_prefill(tmp)
        check_chunked_prefill(tmp)
        check_pool_kept(out)
        check_cut_prefill(out)
        check_rate_once(out)
        check_trim_kv(tmp)
        check_restored_front(tmp, out)
        check_plan(out)
        check_no_cache_and_release(out)
        check_warm_attention(tmp)
        check_calibration(tmp, out)
        check_carries(out)
        check_top_k_slots(tmp)
        check_pipelined_rounds(out)
        print("test_store: ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
