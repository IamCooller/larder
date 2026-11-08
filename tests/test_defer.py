#!/usr/bin/env python3
"""The run-ahead decode wrappers, checked without a model or a GPU job of any size.

_defer_layer_call is the part of --defer that can be wrong in ways no speed
number shows: a rollback that restores one cache but not the other, a layer
recomputed on the stale hidden state, a flag read before the next layer is
queued, a check skipped on the last layer. Fake layers with a scripted miss
schedule pin every one of those down, through the SAME install_defer that the
real models get -- so the wrapper's model-agnostic plumbing (finding the
layer list, the caches among the arguments, the streaming layers) is under
test too.

    python test_defer.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, _BaseCache

import stream_fast

N = 6
STALE = 1000.0          # a layer computed on a stale arena adds this, so garbage is visible


class FakeKV(_BaseCache):
    """The attention cache's contract as the snapshot sees it: trimmable."""
    def __init__(self):
        self.offset = 0
        self.indexer = FakeIndexer()      # an owned sub-cache, as qwen4_exp has

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.offset -= n
        self.indexer.n -= n
        return n


class FakeIndexer(_BaseCache):
    def __init__(self):
        self.n = 0


class FakeStateCache(_BaseCache):
    """Neither ArraysCache nor trimmable: the generic state fallback."""
    def __init__(self):
        self.value = 0

    def is_trimmable(self):
        return False

    @property
    def state(self):
        return self.value

    @state.setter
    def state(self, v):
        self.value = v


class FakeHook:
    def __init__(self):
        self.refreshed = 0

    def refresh(self):
        self.refreshed += 1


class FakeNext:
    """The arena the depth prefetch targets: only warm() is reached from here."""
    def __init__(self, i):
        self.i, self.warmed = i, []

    def warm(self, ids):
        FakeLayer.calls.append(("warm", self.i))
        self.warmed.append([int(e) for e in ids])


def fake_glu(i, miss_first, misses=1):
    g = stream_fast.StreamingSwitchGLU.__new__(stream_fast.StreamingSwitchGLU)
    nn.Module.__init__(g)
    # _pending/_defer: the real ones, so the layer boundary's commit of a
    # depth prefetch runs the actual code path.
    g.__dict__.update(i=i, repaired=not miss_first, hook=FakeHook(), _next_glu=None,
                      _pending=[], _defer=[],
                      misses_left=misses if miss_first else 0)
    return g


class FakeMoE(nn.Module):
    """The MoE half: STALE while the layer's arena is unrepaired, and the
    record a streaming GLU leaves behind for the deferred check."""
    def __init__(self, i, glu):
        super().__init__()
        self.i = i
        self.switch_mlp = glu
        self.top_k = 8            # install_defer reads it: passes up to 63 // 8 = 7 wide

    def __call__(self, h):
        FakeLayer.calls.append(("moe", self.i))
        glu = self.switch_mlp
        if not stream_fast._DEFER_ACTIVE:
            # The ordinary path reads a missing expert in before its gather:
            # never stale, one miss for DeferPolicy's signal.
            if not glu.repaired:
                stream_fast.STATS["miss_layers"] += 1
                glu.repaired = True
            return mx.array(0.0)
        stale = not glu.repaired
        idx = mx.array([self.i] * 8, dtype=mx.int32)
        ok = mx.array(not stale)
        # With the lookahead on, the layer also predicts what the layer
        # look_depth ahead will want. It rides the record and is read on this
        # layer's check, one layer later.
        top = mx.array([100 + self.i], dtype=mx.int32) if glu._next_glu is not None else None
        stream_fast._DEFER_REC.append((glu, idx, ok, idx, 0, [], top))
        return mx.array(STALE if stale else 0.0)


class FakeLayer(nn.Module):
    """h -> h + 1 per layer, plus STALE while the layer's arena is unrepaired.
    Records itself in _DEFER_REC exactly as a streaming GLU would. Takes the
    caches at qwen4_exp's positions (cache and its indexer, plus an extra one)
    to exercise _caches_in."""
    calls = []

    def __init__(self, i, glu):
        super().__init__()
        self.i = i
        self.mlp = FakeMoE(i, glu)

    def __call__(self, h, mask, cache, idx_cache, extra):
        assert mask is None and idx_cache is cache.indexer if isinstance(cache, FakeKV) else True
        FakeLayer.calls.append(("layer", self.i))
        out = h + 1.0 + self.mlp(h)
        if isinstance(cache, ArraysCache):
            # the state the layer reads must be the PRE-layer state on every
            # run, first pass and re-runs alike: a rollback that restored an
            # aliased snapshot would show the previous run's write here
            assert float(cache[0][0, 0, 0]) == -1.0 or float(cache[0][0, 0, 0]) == float(h[0, 0, 0]), \
                f"layer {self.i} ran on a post-run cache state"
            cache[0] = out
        else:
            cache.offset += h.shape[1]
            cache.indexer.n += h.shape[1]
        extra.value += 1
        return out


class FakeAttn(nn.Module):
    """The dense half: +1, and the only half that writes the cache."""
    def __init__(self, i):
        super().__init__()
        self.i = i

    def __call__(self, h, mask, cache, idx_cache, extra):
        FakeLayer.calls.append(("dense", self.i))
        if isinstance(cache, ArraysCache):
            cache[0] = h + 1.0
        else:
            cache.offset += h.shape[1]
            cache.indexer.n += h.shape[1]
        extra.value += 1
        return mx.array(1.0)


class Qwen3NextDecoderLayer(nn.Module):
    """The same layer under the name (and shape) stream_fast._halves knows, so
    a repair here takes the half path. Behaviour is FakeLayer's -- +1 per
    layer, +STALE while the arena is unrepaired -- but split across the two
    halves, each recording itself: a repair that re-ran the dense half, or
    restored a cache the dense half had already written correctly, shows up.
    """
    def __init__(self, i, glu):
        super().__init__()
        self.i = i
        self.is_linear = True
        self.linear_attn = FakeAttn(i)
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = FakeMoE(i, glu)

    def __call__(self, h, mask, cache, idx_cache, extra):
        FakeLayer.calls.append(("layer", self.i))
        d = h + self.linear_attn(self.input_layernorm(h), mask, cache, idx_cache, extra)
        return d + self.mlp(self.post_attention_layernorm(d))


class Qwen2MoeDecoderLayer:
    """Bare object standing in for mlx_lm's real class (qwen2_moe.py:148):
    only the attributes stream_fast._halves reads, and critically NO
    `is_linear` at all -- only the linear-attention Next/3.6 layers have
    that. _halves used to read `layer.is_linear` unconditionally, which is
    exactly the AttributeError that kept this class out of _HALF_LAYERS."""
    def __init__(self):
        self.self_attn = object()
        self.input_layernorm = object()
        self.post_attention_layernorm = object()
        self.mlp = object()


def test_qwen2_moe_halves():
    assert stream_fast._halves(Qwen2MoeDecoderLayer()) is not None, \
        "Qwen2MoeDecoderLayer belongs in _HALF_LAYERS and must not need is_linear"
    print("test_qwen2_moe_halves: ok")


def test_defer_policy():
    """DeferPolicy's state machine, driven with a fake clock -- no model, no
    install_defer. Checks the seeded floor never probes, a wide-open bracket
    opens exactly one probe and resolves it to the measured side, a losing
    flip reverts, and a draw (within NOISE) moves the mode back without
    touching the bracket.
    """
    saved = (stream_fast._DEFER, stream_fast._AUTO, stream_fast._GOV, time.perf_counter)
    clock = [0.0]
    time.perf_counter = lambda: clock[0]
    stream_fast._GOV = None
    W = stream_fast.DeferPolicy.WINDOW
    try:
        # 1. Below the seeded floor (lo=1): never probes, however long it runs.
        stream_fast._DEFER = True
        p = stream_fast.DeferPolicy()
        for _ in range(200):
            clock[0] += 0.010
            p.step(0)
        assert stream_fast._DEFER is True and p.probe is None and not p.history

        # 2. A decisive probe: defer (already running) measures 2x faster
        # than sync at m=8, so the bracket's floor rises to (at least) 8 and
        # the flip taken to time the sync half reverts.
        p.m = 8.0
        clock[0] += 0.010
        p.step(8)                       # crosses lo=1 with hi still inf: probes
        assert p.probe is not None and p.probe.phase == 0
        m_probe = p.probe.m_probe
        assert m_probe == 8.0, m_probe
        for _ in range(W - 1):
            clock[0] += 0.010            # 10 ms/token under defer
            p.step(8)
        assert p.probe.phase == 0 and stream_fast._DEFER is True
        clock[0] += 0.010
        p.step(8)                       # Wth defer sample: flips to sync
        assert p.probe is not None and p.probe.phase == 1 and stream_fast._DEFER is False
        assert p.history[-1][1] is False, p.history
        for _ in range(W - 1):
            clock[0] += 0.020            # 20 ms/token under sync: half the rate
            p.step(8)
        clock[0] += 0.020
        p.step(8)                       # Wth sync sample: resolves the probe
        assert p.probe is None
        assert stream_fast._DEFER is True, "defer measured 2x faster, must win back"
        assert p.lo >= m_probe - 1e-9 and p.hi == float("inf"), (p.lo, p.hi)
        assert [h[1] for h in p.history] == [False, True], p.history

        # 3. A draw (within the 3% NOISE band): the mode reverts just like a
        # loss, but the bracket itself is left exactly where it was.
        stream_fast._DEFER = True
        q = stream_fast.DeferPolicy()
        q.m = 5.0
        clock[0] += 0.010
        q.step(5)
        assert q.probe is not None
        for _ in range(W + 1):           # the first has no pace to be read against
            clock[0] += 0.010
            q.step(5)
        for _ in range(W - 1):
            clock[0] += 0.0102           # 1% slower: inside the 3% band
            q.step(5)
        clock[0] += 0.0102
        q.step(5)
        assert q.probe is None
        assert stream_fast._DEFER is True
        assert [h[1] for h in q.history] == [False, True], q.history
        assert q.lo == 1.0 and q.hi == float("inf"), (q.lo, q.hi)
    finally:
        stream_fast._DEFER, stream_fast._AUTO, stream_fast._GOV, time.perf_counter = saved
    print("test_defer_policy: ok")


class GovStub:
    """Just the bits of ArenaGovernor that DeferPolicy.step reads: `cap` (to
    notice a resize), `_skip` (a fresh-resize guard window) and `_rate`
    (cleared on probe resolution, see DeferPolicy._resolve_probe)."""
    def __init__(self, cap):
        self.cap = cap
        self._skip = False
        self._rate = {48: 30.0, 96: 10.0}


def test_defer_policy_probe_survives_resize():
    """A probe spans 2*WINDOW=128 tokens; ArenaGovernor's own speed lane
    resizes every period=64 -- on warm-up the two collide routinely, not as
    an edge case. A resize mid-probe must throw the open probe away and
    revert to the mode running before it opened, rather than let a window
    split across two cap sizes resolve the bracket on a meaningless number.
    """
    saved = (stream_fast._DEFER, stream_fast._AUTO, stream_fast._GOV, time.perf_counter)
    clock = [0.0]
    time.perf_counter = lambda: clock[0]
    gov = GovStub(48)
    stream_fast._GOV = gov
    W = stream_fast.DeferPolicy.WINDOW
    try:
        stream_fast._DEFER = True
        p = stream_fast.DeferPolicy()
        p._cap_seen = gov.cap      # so step 1 isn't itself seen as a resize
        p.m = 8.0
        clock[0] += 0.010
        p.step(8)                  # crosses lo=1 with hi=inf: opens a probe
        assert p.probe is not None and p.probe.mode_a is True
        for _ in range(W // 2):    # partway through phase 0, still open
            clock[0] += 0.010
            p.step(8)
        assert p.probe is not None
        gov.cap = 96                # the arena resizes under the open probe
        clock[0] += 0.010
        p.step(8)
        assert p.probe is None, "a cap change must throw the open probe away"
        assert stream_fast._DEFER is True, "must revert to the pre-probe mode"
        assert p._cooldown == stream_fast.DeferPolicy.COOLDOWN - 1, p._cooldown
        assert p.lo == 1.0 and p.hi == float("inf"), \
            "the bracket must be untouched by a probe that never resolved"
    finally:
        stream_fast._DEFER, stream_fast._AUTO, stream_fast._GOV, time.perf_counter = saved
    print("test_defer_policy_probe_survives_resize: ok")


class FakeStack(nn.Module):
    """A decoder loop of the model's own shape: embed, layers, a tail."""
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.extras = [FakeStateCache() for _ in layers]

    def __call__(self, inputs, cache=None):
        h = mx.zeros((1, inputs.shape[-1], 4))
        for l, c, e in zip(self.layers, cache, self.extras):
            h = l(h, None, c, getattr(c, "indexer", None), e)
        return h * 1.0            # the tail: must never see a rolled-back pass


class FakeModel(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.model = stack

    @property
    def layers(self):
        return self.model.layers


def fake_check(resolves):
    """_defer_check's stand-in: a layer misses while its arena is unrepaired."""
    def check(glu, idx, ok, indices, claimed, gen, keys):
        assert gen == 0 and keys == []
        mx.eval(idx, ok)
        hit = bool(ok.item())
        resolves.append((glu.i, claimed))
        if claimed:
            assert hit, "a repaired layer must be a hit"
            return False
        if not hit:
            glu.misses_left -= 1
            glu.repaired = glu.misses_left <= 0
            glu._defer_hold = stream_fast._Hold(["pre-claim keys"], glu.hook)
        return not hit
    return check


def run(miss_layers, tokens=1, width=1, look=False, half=False):
    FakeLayer.calls = []
    stream_fast.STATS["defer_now_misses"] = 0
    stream_fast.STATS["repair_half"] = 0
    glus = [fake_glu(i, i in miss_layers) for i in range(N)]
    nexts = []
    if look:
        # look_depth 2, the shallowest depth the deferred check can serve.
        nexts = [FakeNext(i) for i in range(N)]
        for i in range(N - 2):
            glus[i]._next_glu = nexts[i + 2]
    cls = Qwen3NextDecoderLayer if half else FakeLayer
    layers = [cls(i, glus[i]) for i in range(N)]
    stack = FakeStack(layers)
    model = FakeModel(stack)
    cache = [ArraysCache(1) if i % 2 == 0 else FakeKV() for i in range(N)]
    for c in cache:
        if isinstance(c, ArraysCache):
            c[0] = mx.zeros((1, 1, 4)) - 1.0

    resolves = []
    fake_resolve = fake_check(resolves)
    real_async = mx.async_eval

    def logged_async(*arrays):
        FakeLayer.calls.append(("submit", len(arrays)))
        return real_async(*arrays)

    # Only the CHECK is faked: _defer_resolve itself -- the wrapper that turns
    # the prediction into a prefetch on the sync the check pays -- is the real
    # one, so the lookahead path is under test.
    saved = stream_fast._defer_check, mx.async_eval, stream_fast._DEFER, stream_fast._GOV
    stream_fast._defer_check, mx.async_eval = fake_resolve, logged_async
    stream_fast._DEFER, stream_fast._GOV = True, None
    try:
        stream_fast.install_defer(model)
        for t in range(tokens):
            if t:
                for i, g in enumerate(glus):     # the scheduled layers miss again on the next token
                    if i in miss_layers:
                        g.repaired, g.misses_left = False, 1
                for c in cache:
                    if isinstance(c, ArraysCache):
                        c[0] = mx.zeros((1, 1, 4)) - 1.0
            out = model.model(mx.zeros((1, width), dtype=mx.int32), cache)
    finally:
        stream_fast._defer_check, mx.async_eval, stream_fast._DEFER, stream_fast._GOV = saved
    assert stream_fast._DEFER_ACTIVE is False
    assert all(g._defer_hold is None for g in glus), [g.i for g in glus if g._defer_hold]
    assert all(g.hook.refreshed == (tokens if i in miss_layers else 0) for i, g in enumerate(glus)), \
        [(g.i, g.hook.refreshed) for g in glus]
    # every cache advanced exactly once, the extras (state fallback) too
    for c in cache:
        if isinstance(c, FakeKV):
            assert c.offset == tokens * width and c.indexer.n == tokens * width, (c.offset, c.indexer.n)
    assert all(e.value == tokens for e in stack.extras), [e.value for e in stack.extras]
    if look:
        return out, cache, FakeLayer.calls, resolves, nexts
    return out, cache, FakeLayer.calls, resolves


def decode(layout, mode, tokens, miss=(1, 2), cost=None):
    """`tokens` one-position passes under --defer-mode `mode`, the first on
    empty caches (the prompt). `layout`: "hybrid" (layer 0 an ArraysCache,
    as Qwen3.6's GatedDeltaNet), "kv" (attention only) or "arrays" (no
    offset anywhere). The `miss` layers miss on every pass; a pass costs
    `cost[run ahead?]` on a fake clock. Returns every pass's output, the
    policy and whether the last pass ran ahead."""
    glus = [fake_glu(i, i in miss) for i in range(N)]
    model = FakeModel(FakeStack([FakeLayer(i, glus[i]) for i in range(N)]))
    arrays = {"hybrid": lambda i: i % 2 == 0, "kv": lambda i: False,
              "arrays": lambda i: True}[layout]
    cache = [ArraysCache(1) if arrays(i) else FakeKV() for i in range(N)]
    policy = stream_fast.DeferPolicy() if mode == "auto" else None
    if policy is not None:
        policy.WINDOW = policy.COOLDOWN = 8
    clock, outs = [0.0], []
    saved = (stream_fast._defer_check, stream_fast._DEFER, stream_fast._AUTO,
             stream_fast._GOV, time.perf_counter)
    stream_fast._defer_check = fake_check([])
    stream_fast._DEFER, stream_fast._AUTO, stream_fast._GOV = mode != "never", policy, None
    time.perf_counter = lambda: clock[0]
    try:
        stream_fast.install_defer(model)
        for _ in range(tokens):
            for i in miss:
                glus[i].repaired, glus[i].misses_left = False, 1
            for c in cache:
                if isinstance(c, ArraysCache):
                    c[0] = mx.zeros((1, 1, 4)) - 1.0
            clock[0] += cost[stream_fast._DEFER] if cost else 0.001
            out = model.model(mx.zeros((1, 1), dtype=mx.int32), cache)
            outs.append(float(out[0, 0, 0]))
        ahead = stream_fast._DEFER
    finally:
        (stream_fast._defer_check, stream_fast._DEFER, stream_fast._AUTO,
         stream_fast._GOV, time.perf_counter) = saved
    assert all(c.offset == tokens for c in cache if isinstance(c, FakeKV))
    return outs, policy, ahead


def test_auto_hybrid():
    """--defer-mode auto where layer 0's cache has no offset (Qwen3.6). Every
    pass but the prompt is sampled, and the policy picks whichever mode its
    probe measures faster. An attention-only stack is sampled as before, a
    stack with no offset anywhere still never is, and every mode computes
    the same outputs."""
    fast_sync = {True: 0.002, False: 0.001}      # seconds a pass: ahead, sync
    outs, p, ahead = decode("hybrid", "auto", 40, cost=fast_sync)
    assert p._n_tok == 39, p._n_tok
    assert ahead is False and p.hi < float("inf"), (p.lo, p.hi, p.history)
    _, p, ahead = decode("hybrid", "auto", 40, cost={True: 0.001, False: 0.002})
    assert ahead is True and p.lo > 1.0 and p.hi == float("inf"), (p.lo, p.hi, p.history)
    kv, p, ahead = decode("kv", "auto", 40, cost=fast_sync)
    assert p._n_tok == 39 and ahead is False and kv == outs, (p._n_tok, p.history)
    _, p, ahead = decode("arrays", "auto", 40, cost=fast_sync)
    assert p._n_tok == 0 and ahead is True, p._n_tok
    assert outs == [float(N)] * 40, outs
    for mode in ("always", "never"):
        o, _, _ = decode("hybrid", mode, 40)
        assert o == outs, (mode, o)
    print("test_auto_hybrid: ok")


def test_defer_policy_gap():
    """The gap filter reads the recent pace, not a fixed 0.25 s. Passes of
    0.4 s -- forced passes at 2 GB take 0.3-0.7 s -- time a probe, where
    the fixed filter dropped every one and the probe never resolved; a turn
    boundary among them is dropped; past GAP_CAP nothing counts, however
    slow the recent pace."""
    saved = (stream_fast._DEFER, stream_fast._AUTO, stream_fast._GOV, time.perf_counter)
    clock = [0.0]
    time.perf_counter = lambda: clock[0]
    stream_fast._GOV = None
    W = stream_fast.DeferPolicy.WINDOW
    try:
        stream_fast._DEFER = True
        p = stream_fast.DeferPolicy()
        for _ in range(4):              # below the floor: no probe, a pace
            clock[0] += 0.4
            p.step(0)
        p.m = 8.0
        clock[0] += 0.4
        p.step(8)                       # opens the probe
        for _ in range(W):
            clock[0] += 0.4             # 0.4 s a pass ahead
            p.step(8)
        assert p.probe is not None and p.probe.phase == 1 and stream_fast._DEFER is False
        clock[0] += 5.0                 # a turn boundary
        p.step(8)
        assert p.probe.n == 0 and p.probe.elapsed == 0.0, (p.probe.n, p.probe.elapsed)
        for _ in range(W):
            clock[0] += 0.2             # sync, twice as fast
            p.step(8)
        assert p.probe is None and stream_fast._DEFER is False and p.hi == 8.0, (p.lo, p.hi)

        q = stream_fast.DeferPolicy()   # every pass a pause: the cap still drops them
        q.m = 8.0
        for _ in range(W + 2):
            clock[0] += 20.0
            q.step(8)
        assert q.probe is not None and q.probe.n == 0, q.probe.n
    finally:
        stream_fast._DEFER, stream_fast._AUTO, stream_fast._GOV, time.perf_counter = saved
    print("test_defer_policy_gap: ok")


def main():
    out, cache, calls, resolves = run(set())
    assert float(out[0, 0, 0]) == N, float(out[0, 0, 0])
    seq = [c[1] for c in calls if c[0] == "layer"]
    assert seq == list(range(N)), seq
    submits = [c for c in calls if c[0] == "submit"]
    assert len(submits) == N and all(c[1] == 3 for c in submits), calls
    # layer i's flag is resolved only AFTER layer i+1 was submitted
    order = [c for c in calls if c[0] == "submit"]
    assert [r[0] for r in resolves] == list(range(N)) and not any(r[1] for r in resolves)

    out, cache, calls, resolves = run({2})
    assert float(out[0, 0, 0]) == N, float(out[0, 0, 0])
    seq = [c[1] for c in calls if c[0] == "layer"]
    assert seq == [0, 1, 2, 3, 2, 3, 4, 5], seq
    assert float(cache[2][0][0, 0, 0]) == 3.0, float(cache[2][0][0, 0, 0])
    assert (2, True) in resolves and (2, False) in resolves, resolves
    assert [r for r in resolves if r[0] == 3] == [(3, False)], resolves

    out, cache, calls, resolves = run({N - 1})
    assert float(out[0, 0, 0]) == N, float(out[0, 0, 0])
    seq = [c[1] for c in calls if c[0] == "layer"]
    assert seq == [0, 1, 2, 3, 4, 5, 5], seq
    assert resolves[-1] == (N - 1, True), resolves

    out, cache, calls, resolves = run({1, 2})
    assert float(out[0, 0, 0]) == N, float(out[0, 0, 0])
    seq = [c[1] for c in calls if c[0] == "layer"]
    assert seq == [0, 1, 2, 1, 2, 3, 2, 3, 4, 5], seq
    assert float(cache[2][0][0, 0, 0]) == 3.0, float(cache[2][0][0, 0, 0])

    # every layer missing, two tokens on the same caches: each rollback must
    # restore the PRE-layer state, never the one its own re-run wrote
    out, cache, calls, resolves = run(set(range(N)), tokens=2)
    assert float(out[0, 0, 0]) == N, float(out[0, 0, 0])

    assert stream_fast.STATS["defer_rollbacks"] >= 4, stream_fast.STATS["defer_rollbacks"]
    assert stream_fast.STATS["defer_tokens"] == 6

    # Policies: "now" resolves every layer at once (no run-ahead, so a miss
    # redoes ONE layer, not two) and "auto" learns to do so on a layer that
    # keeps missing; all three produce the same output.
    # A speculative pass, 3 positions wide: same repair order, every trimmable
    # cache gives back all 3 positions on the rollback (the width check above).
    out, cache, calls, resolves = run({1, 2}, width=3)
    assert float(out[0, 2, 0]) == N, float(out[0, 2, 0])
    seq = [c[1] for c in calls if c[0] == "layer"]
    assert seq == [0, 1, 2, 1, 2, 3, 2, 3, 4, 5], seq
    assert stream_fast._DEFER_MAX_T == 7, stream_fast._DEFER_MAX_T
    # Wider than the GLU can take without sorting: the ordinary path, no loop.
    out, cache, calls, resolves = run(set(), width=8)
    assert not [c for c in calls if c[0] == "submit"], calls
    assert resolves == [], resolves

    # Depth lookahead INSIDE the run-ahead path. Layer i's prediction is
    # issued on i's check -- after layer i+1 has been submitted and before
    # layer i+2, its target, is built -- and never twice for one prediction.
    out, cache, calls, resolves, nexts = run(set(), look=True)
    assert float(out[0, 0, 0]) == N, float(out[0, 0, 0])
    seq = [c for c in calls if c[0] in ("layer", "warm")]
    assert seq == [("layer", 0), ("layer", 1), ("warm", 2), ("layer", 2),
                   ("warm", 3), ("layer", 3), ("warm", 4), ("layer", 4),
                   ("warm", 5), ("layer", 5)], seq
    assert [n.warmed for n in nexts] == [[], [], [[100]], [[101]], [[102]], [[103]]], \
        [n.warmed for n in nexts]
    # Everything the check reads is submitted with the layer: routing, flag and
    # the prediction on the layers that make one.
    assert [c[1] for c in calls if c[0] == "submit"] == [4, 4, 4, 4, 3, 3], calls

    # A rollback re-runs the layer, and the re-run predicts nothing: its first
    # pass already did, from the same input.
    out, cache, calls, resolves, nexts = run({2}, look=True)
    assert float(out[0, 0, 0]) == N, float(out[0, 0, 0])
    assert [len(n.warmed) for n in nexts] == [0, 0, 1, 1, 1, 1], [n.warmed for n in nexts]

    # The MoE-only repair: a layer stream_fast._halves knows redoes just
    # `h + mlp(norm2(h))` on a rollback. Its dense half ran on an input that
    # was already checked and wrote the cache, so neither is touched again.
    out, cache, calls, resolves = run({2}, half=True)
    assert float(out[0, 0, 0]) == N, float(out[0, 0, 0])
    seq = [c for c in calls if c[0] in ("dense", "moe")]
    assert seq == [("dense", 0), ("moe", 0), ("dense", 1), ("moe", 1),
                   ("dense", 2), ("moe", 2), ("dense", 3), ("moe", 3),
                   ("moe", 2),                       # <- the repair: MoE only
                   ("dense", 3), ("moe", 3),         # the layer behind it, whole
                   ("dense", 4), ("moe", 4), ("dense", 5), ("moe", 5)], seq
    assert seq.count(("dense", 2)) == 1, seq         # the dense half was not re-run
    assert stream_fast.STATS["repair_half"] == 1, stream_fast.STATS["repair_half"]
    # ... and the cache the dense half wrote is still what it wrote (input 2, +1)
    assert float(cache[2][0][0, 0, 0]) == 3.0, float(cache[2][0][0, 0, 0])
    assert (2, True) in resolves and (2, False) in resolves, resolves

    # The last layer checks itself, so its repair is the same one on the
    # resolve-now path.
    out, cache, calls, resolves = run({N - 1}, half=True)
    assert float(out[0, 0, 0]) == N, float(out[0, 0, 0])
    seq = [c for c in calls if c[0] in ("dense", "moe")]
    assert seq[-3:] == [("dense", N - 1), ("moe", N - 1), ("moe", N - 1)], seq
    assert stream_fast.STATS["repair_half"] == 1, stream_fast.STATS["repair_half"]

    # Two layers in a row, three positions wide, two tokens on one cache: the
    # half path has to leave every cache exactly where the full one did.
    out, cache, calls, resolves = run({1, 2}, width=3, tokens=2, half=True)
    assert float(out[0, 2, 0]) == N, float(out[0, 2, 0])
    assert stream_fast.STATS["repair_half"] == 4, stream_fast.STATS["repair_half"]

    # The A/B switch: same layers, today's whole-layer repair.
    stream_fast.DEFER_FULL_REPAIR = True
    try:
        out, cache, calls, resolves = run({2}, half=True)
    finally:
        stream_fast.DEFER_FULL_REPAIR = False
    assert float(out[0, 0, 0]) == N, float(out[0, 0, 0])
    assert stream_fast.STATS["repair_half"] == 0, stream_fast.STATS["repair_half"]
    assert [c for c in calls if c[0] in ("dense", "moe")].count(("dense", 2)) == 2, calls
    assert float(cache[2][0][0, 0, 0]) == 3.0, float(cache[2][0][0, 0, 0])


    test_qwen2_moe_halves()
    test_defer_policy()
    test_defer_policy_probe_survives_resize()
    test_defer_policy_gap()
    test_auto_hybrid()
    print("test_defer: ok")


if __name__ == "__main__":
    main()
