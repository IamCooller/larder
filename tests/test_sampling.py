#!/usr/bin/env python3
"""sampler.py and the sampled speculative round, on toy distributions and toy models. No checkpoint.

The round is checked against the target's exact joint distribution of the first L tokens (every
prefix enumerated), by Pearson chi-square at p = 0.001, with a broken rule run as a power check.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_sampling.py
"""
import math
import os
import sys
import threading
import time
import types
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx

import sampler
import spec
import spec_draft
from sampler import Sampling
import support
from support import tokenizer_dir, toy_llm

IDS = [3, 9, 27, 1, 5, 11, 23, 17]


def crit(df, z=3.09):
    """Chi-square's upper 0.1% point (Wilson-Hilferty)."""
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def gof(counts, probs):
    """Pearson chi-square of counts against probs, cells expecting < 5 pooled; (stat, df, outside)."""
    n = sum(counts.values())
    stat, df, po, pe = 0.0, -1, 0, 0.0
    for k, p in probs.items():
        o, e = counts.get(k, 0), n * p
        if e < 5:
            po, pe = po + o, pe + e
        else:
            stat, df = stat + (o - e) ** 2 / e, df + 1
    if pe:
        stat, df = stat + (po - pe) ** 2 / pe, df + 1
    return stat, df, sum(v for k, v in counts.items() if k not in probs)


def test_process():
    lg = mx.log(mx.array([0.4, 0.3, 0.15, 0.1, 0.05]))

    def probs(**kw):
        return [round(x, 4) for x in mx.exp(Sampling(**kw).logprobs(lg)).tolist()]
    assert probs(temperature=1.0) == [0.4, 0.3, 0.15, 0.1, 0.05]
    assert probs(temperature=1.0, top_k=3) == [round(x / 0.85, 4) for x in (0.4, 0.3, 0.15)] + [0, 0]
    assert probs(temperature=1.0, top_p=0.7) == [round(x / 0.7, 4) for x in (0.4, 0.3)] + [0] * 3
    assert probs(temperature=1.0, top_p=0.75) == [round(x / 0.85, 4) for x in (0.4, 0.3, 0.15)] + [0, 0]
    assert probs(temperature=1.0, top_k=3, top_p=0.9) == probs(temperature=1.0, top_k=3)
    assert probs(temperature=1.0, min_p=0.5) == [round(x / 0.7, 4) for x in (0.4, 0.3)] + [0] * 3
    sq = [x * x for x in (0.4, 0.3, 0.15, 0.1, 0.05)]
    assert probs(temperature=0.5) == [round(x / sum(sq), 4) for x in sq]
    assert Sampling.from_request({}) is None
    assert Sampling.from_request({"temperature": 0, "top_p": 0.5}) is None
    assert Sampling.from_request({"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 3}) == \
        Sampling(0.6, 20, 0.95, 0.0, 3)
    for bad in ({"temperature": -1}, {"temperature": 1, "top_p": 1.5}, {"temperature": "x"},
                {"temperature": 1, "min_p": 1}, {"temperature": 1, "top_k": -2}):
        try:
            Sampling.from_request(bad)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"accepted {bad}")
    print("  process: temperature, top_k, top_p, min_p and the request fields as specified")


def test_verify_rows(n=40000):
    """Each row of verify() is an independent one-draft round: x ~ q, then stand or be replaced."""
    p = mx.log(mx.array([0.5, 0.3, 0.15, 0.05]))
    for name, q in (("q = p", p), ("bad q", mx.log(mx.array([0.05, 0.05, 0.1, 0.8]))),
                    ("one-hot q at p's rarest", mx.log(mx.array([0.0, 0.0, 0.0, 1.0])))):
        x = mx.random.categorical(mx.broadcast_to(q, (n, 4)))
        stands, fix, _ = sampler.verify(mx.broadcast_to(p, (n + 1, 4)), mx.broadcast_to(q, (n, 4)), x)
        out = mx.where(stands, x, fix)
        got = Counter(out.tolist())
        stat, df, outside = gof(got, dict(enumerate(mx.exp(p).tolist())))
        acc = float(stands.astype(mx.float32).mean())
        want = float(mx.minimum(mx.exp(p), mx.exp(q)).sum())     # 1 - TV(p, q)
        assert outside == 0 and stat < crit(df), (name, stat, df)
        assert abs(acc - want) < 4 * math.sqrt(max(want * (1 - want), 0) / n) + 1e-6, (name, acc, want)
        print(f"  verify, {name:24s} chi2 {stat:5.2f} (df {df}, bar {crit(df):.1f}), "
              f"stood {acc:.4f} vs 1-TV {want:.4f}")


class ToyHead:
    """A stand-in MTP head: the next hidden from the target's hidden and the token's embedding."""

    def __init__(self, hidden, seed):
        k1, k2 = mx.random.split(mx.random.key(seed))
        self.w1 = mx.random.normal((hidden, hidden), key=k1) / math.sqrt(hidden)
        self.w2 = mx.random.normal((hidden, hidden), key=k2) / math.sqrt(hidden)
        self.scale = math.sqrt(hidden)

    def __call__(self, hidden, embed, cache=None):
        return mx.tanh(hidden @ self.w1 + embed @ self.w2) * self.scale


class BadDraft(spec.Speculator):
    """q fixed and far from p: 60% of its mass on ids 0-3, whatever the context."""

    def _draft_ids(self, h):
        V = self.embed.weight.shape[0]
        q = mx.log(mx.where(mx.arange(V) < 4, 0.6 / 4, 0.4 / (V - 4)))[None]
        self._q.append(q)
        return mx.random.categorical(q)


class OracleDraft(spec.Speculator):
    """Drafts from the target itself, re-run on the context, so most stand and every depth and the
    bonus run: q = drafter's view of its logits, or its argmax as a certain (one-hot) draft."""
    drafter = None

    def __init__(self, model, gamma=4):
        super().__init__(model, None, gamma)
        self._hist = []

    def generate(self, ids, max_tokens, eos=(), stats=None, cache=None,
                 prefill_step=0):
        self._hist = list(ids)
        yield from super().generate(ids, max_tokens, eos, stats, cache,
                                    prefill_step)

    def _draft(self, pairs, hcache, st):
        self._hist.extend(p[1] for p in pairs)
        out = []
        for _ in range(self.gamma):
            lg = self.model(mx.array([self._hist + out]))[:, -1]
            if self.drafter is None:
                out.append(int(mx.argmax(lg)))
            else:
                self._q.append(self.drafter.logprobs(lg))
                out.append(int(mx.random.categorical(self._q[-1])))
        return out


def exact(m, S, L):
    """The target's exact joint distribution of its first L sampled tokens after IDS."""
    dist = {(): 1.0}
    for _ in range(L):
        nxt = {}
        for pre, pr in dist.items():
            pv = mx.exp(S.logprobs(m(mx.array([IDS + list(pre)]))[0, -1])).tolist()
            nxt.update({pre + (t,): pr * v for t, v in enumerate(pv) if v > 0})
        dist = nxt
    return dist


def run(sp, S, L, n):
    """n replies of L tokens; Stats are per reply (generate stops on their token count), summed."""
    sp.sampling = S
    got, tot = Counter(), spec.Stats()
    for _ in range(n):
        st = spec.Stats()
        got[tuple(sp.generate(IDS, L, stats=st))] += 1
        tot.rounds, tot.tokens = tot.rounds + st.rounds, tot.tokens + st.tokens - 1
        for d, (s, h) in enumerate(zip(st.seen, st.hit)):
            tot.note(d, False)
            tot.seen[d] += s - 1
            tot.hit[d] += h
    return got, tot


def test_round(n=2000, L=4):
    m = spec_draft._toy(32, 32, 2)
    S = Sampling(1.0, top_k=3, top_p=0.9)
    ref = exact(m, S, L)
    rows = []

    def check(name, got, st=None, fail=False):
        stat, df, outside = gof(got, ref)
        ok = outside == 0 and stat < crit(df)
        assert ok != fail, (name, stat, df, outside)
        rows.append(name)
        print(f"  {name:34s} chi2 {stat:7.1f} (df {df}, bar {crit(df):.1f}), outside the support "
              f"{outside}" + (f", per draft stood {[round(x, 2) for x in st.per_depth]}, "
                              f"{st.per_round:.2f} tokens/round" if st else ""))
    plain = Counter(tuple(t for t, _ in spec_draft.first_early(mx.array(IDS), m, max_tokens=L,
                                                                 sampler=S)) for _ in range(n))
    check("plain (first_early + sampler)", plain)
    head = ToyHead(32, seed=1)
    check("toy head (q unlike p), gamma 2", *run(spec.Speculator(m, head, gamma=2), S, L, n))
    check("bad q, gamma 3", *run(BadDraft(m, head, gamma=3), S, L, n))
    near, far = Sampling(1.3, top_k=4), Sampling(2.0, top_k=6)
    for g, q in ((1, near), (2, near), (4, near), (2, far)):
        sp = OracleDraft(m, gamma=g)
        sp.drafter = q
        check(f"q {'near' if q is near else 'far from'} p, gamma {g}", *run(sp, S, L, n))
    check("certain drafts (one-hot q), gamma 2", *run(OracleDraft(m, gamma=2), S, L, n))
    # Power: two broken rules on the far draft (the last `sp`), a gross one and a subtle one.
    real = sampler.verify
    try:
        sampler.verify = lambda p, q, d: (mx.ones(d.shape, mx.bool_),) + real(p, q, d)[1:]
        check("BROKEN: every draft stands", *run(sp, S, L, n), fail=True)
        sampler.verify = lambda p, q, d: real(p, q, d)[:1] + real(p, p[:-1], d)[1:]
        check("BROKEN: fix ~ p, not the residual", *run(sp, S, L, n), fail=True)
    finally:
        sampler.verify = real


def test_greedy_untouched(L=12):
    """sampling None runs the greedy round: the same tokens as the toy's own greedy decode."""
    m = spec_draft._toy(32, 32, 2)
    ref = [t for t, _ in spec_draft.first_early(mx.array(IDS), m, max_tokens=L)]
    for g in (1, 3):
        sp = spec.Speculator(m, ToyHead(32, seed=1), gamma=g)
        sp.sampling = Sampling(1.0)
        list(sp.generate(IDS, L))          # a sampled turn first must leave nothing behind
        sp.sampling = None
        assert list(sp.generate(IDS, L)) == ref, g
    print(f"  greedy after a sampled turn: {L} tokens = plain greedy at gamma 1 and 3")


def test_engine():
    """The seam with moe_stream on the toy engine: sampled MTP and plain turns keep the cache
    contract, a seed repeats a reply, and greedy turns afterwards are the greedy turns of an
    engine that never sampled."""
    if not support.has_tokenizer():
        print("engine seam: SKIPPED (no tokenizer: set MOE_TOY_TOKENIZER)")
        return
    import moe_stream
    from mtp import MTPHead
    Q = ["Say the word apple.", "Now say the word banana."]
    S = Sampling(0.8, top_k=20, top_p=0.95, seed=5)
    head = MTPHead(toy_llm("slice").model.args)
    mx.eval(head.parameters())

    def build(arm):
        llm = toy_llm("slice", max_tokens=16)
        llm._spec = spec.Speculator(llm.model, head, gamma=2)
        llm.decode_policy_free = types.SimpleNamespace(pick=lambda: arm, record=lambda *x: None)
        return llm
    for arm in ("mtp", "plain"):
        ref = build(arm)
        want = [ref.ask(Q[0]), ref.ask(Q[1])]
        llm = build(arm)
        a = llm.ask(Q[0], sampling=S)
        st = llm.stats()
        assert (st.get("mtp_rounds", 0) > 0) == (arm == "mtp"), st
        b = llm.ask(Q[1], sampling=S)
        assert llm.stats()["cached_tokens"] > 0, "a sampled turn broke the increment"
        for c in llm._cache:
            if isinstance(c, moe_stream.KVCache):
                assert c.offset - c.gap == len(llm._seen), (c.offset, c.gap, len(llm._seen))
        llm.reset()
        assert llm.ask(Q[0], sampling=S) == a, "the same seed gave another reply"
        llm.reset()
        assert [llm.ask(Q[0]), llm.ask(Q[1])] == want, arm
        print(f"  engine, {arm:5s}: sampled {len(a)}+{len(b)} chars, cache exact, seed repeats, "
              f"greedy afterwards = an engine that never sampled")


def test_batch():
    """server.py's batch rows (dense targets): sampled rows beside greedy ones, which stay their
    solo replies; on the CPU, where test_batching asserts solo replies token for token."""
    if not support.has_tokenizer():
        print("batch rows: SKIPPED (no tokenizer: set MOE_TOY_TOKENIZER)")
        return
    import moe_stream
    import test_batching as tb
    from mlx_lm.utils import load_tokenizer
    from test_reuse import cpu
    cpu()
    gen = sys.modules["mlx_lm.generate"]
    spec_draft.generation_stream = gen.generation_stream
    spec_draft.wired_limit = gen.wired_limit
    tok = load_tokenizer(tokenizer_dir())
    target, _ = tb.toys(max([tok.vocab_size] + list(tok.get_added_vocab().values())) + 1)
    ref = tb.engine(target, tok)
    want = [ref.ask(messages=tb.MSGS[i], max_tokens=tb.ASKS[i][1]) for i in (0, 2)]
    h, seen = tb.host(tb.engine(target, tok))
    S = Sampling(1.0, top_k=5)
    gate = threading.Event()
    h.executor.submit(gate.wait)
    try:
        pieces = [h.generate(tb.MSGS[i], tb.ASKS[i][1], stream=True,
                             sampling=S if i % 2 else None) for i in range(4)]
    finally:
        gate.set()
    got = ["".join(x) for x in pieces]
    assert [got[0], got[2]] == want and seen["width"] == 4, (got, want, seen)
    print(f"  batch: 2 sampled rows ({len(got[1])}, {len(got[3])} chars) beside 2 greedy ones, "
          f"width {seen['width']}; the greedy replies are their solo ones")


if __name__ == "__main__":
    t0 = time.time()
    mx.random.seed(1234)
    test_process()
    test_verify_rows()
    test_round()
    test_greedy_untouched()
    test_engine()
    test_batch()          # last: it moves the process to the CPU
    print(f"ok ({time.time() - t0:.0f} s)")
