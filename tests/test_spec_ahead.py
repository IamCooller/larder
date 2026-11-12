#!/usr/bin/env python3
"""MOE_SPEC_AHEAD's loop on a toy model, no checkpoint: the round built ahead of its verdict changes nothing.

    python3 tests/test_spec_ahead.py

Scripted drafts keyed by position (so both loops propose the same drafts), every decoder layer a hand-off
(an mx.async_eval and spec's check, as stream_fast._call_native's), the verdict read at hand-off 1, 2, 3 or
after the whole pass, and with a stand-in for the mapped block found earlier at some hand-offs. Asserted against the same speculator with `ahead` off, the loop main runs: the
tokens, the round counts, and every cache array bit for bit, through rejections, eos, max_tokens and a
consumer that hangs up mid-turn.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache, make_prompt_cache

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import spec
from support import check_cache, greedy, toy_model

HOOK = [None]


def watch(fn):
    HOOK[0] = fn


def handoffs():
    """Each decoder layer of the toy hands its output to the GPU and calls the watched check after it."""
    import mlx_lm.models.qwen3_5 as q35
    orig = q35.DecoderLayer.__call__

    def call(self, x, mask=None, cache=None):
        out = orig(self, x, mask, cache)
        if HOOK[0] is not None:
            mx.async_eval(out)
            HOOK[0]()
        return out
    q35.DecoderLayer.__call__ = call


class Keyed(spec.Speculator):
    """Draft j of a round is ref[st.tokens + j] where hit(that position) says so, else a wrong id, as a lazy array."""

    def __init__(self, model, ref, hit, vocab, gamma):
        super().__init__(model, None, gamma=gamma)
        self.ref, self.hit, self.vocab = ref, hit, vocab

    def _draft(self, pairs, hcache, st):
        # The pairs' tokens are evaluated as a real head would read them, so a lazy id of a round built ahead runs.
        mx.eval([p[1] for p in pairs if isinstance(p[1], mx.array)])
        out = []
        for j in range(self.gamma):
            n = st.tokens + j
            want = self.ref[n] if n < len(self.ref) else 0
            out.append(want if self.hit(n) else (want + 1) % self.vocab)
        return mx.array(out, dtype=mx.int32)


class Board:
    """A stand-in for moe_pipe_cuda's mapped block: post keeps the round's arrays, peek hands them out every other look
    (so a verdict is found before, at and after the hand-off that would wait for it)."""

    def __init__(self, late=False):
        # late: the GPU never gets there before the build ends, so the loop reads the arrays itself.
        self.rounds, self.n, self.looks, self.late = {}, 0, 0, late

    def post(self, am, drafts):
        self.n += 1
        self.rounds[self.n] = (am, drafts)
        return am, self.n

    def peek(self, n, count):
        # The block holds only the newest round's ids: a look for an older round would read another round's.
        assert n == self.n, f"peek({n}) after post handed out {self.n}"
        self.looks += 1
        if self.late or self.looks % 2:
            return None
        am, dr = self.rounds[n]
        mx.eval(am, dr)
        return (am.tolist() + dr.tolist())[:count]


def run(m, ref, hit, ids, n, ahead, gamma=1, eos=(), close=None, vocab=128, board=False):
    # gamma 0: the loop with no drafter (ForcedSpeculator), whose rounds are one row each.
    sp = Keyed(m, ref, hit, vocab, gamma) if gamma else spec.ForcedSpeculator(m)
    sp.ahead, sp.watch = ahead, watch
    if board:
        b = Board(late=board == "late")
        sp.post, sp.peek = b.post, b.peek
    cache, st, got = make_prompt_cache(m), spec.Stats(), []
    gen = sp.generate(ids, n, eos, st, cache)
    for t in gen:
        got.append(t)
        if close is not None and len(got) == close:
            gen.close()
            break
    assert HOOK[0] is None, "the hand-off check was left on"
    return got, st, cache


def same_cache(a, b, why):
    for x, y in zip(a, b):
        if isinstance(x, KVCache):
            assert x.offset == y.offset, (why, x.offset, y.offset)
            pairs = [(x.keys[..., :x.offset, :], y.keys[..., :y.offset, :]),
                     (x.values[..., :x.offset, :], y.values[..., :y.offset, :])]
        else:
            pairs = list(zip(x.cache, y.cache))
        for p, q in pairs:
            assert (p is None) == (q is None), why
            if p is not None:
                assert np.array_equal(np.array(p), np.array(q)), f"{why}: a cache array differs"


def test_ahead():
    handoffs()
    vocab = 128
    m = toy_model(seed=0, vocab=vocab)
    ids = [3, 9, 27, 81, 5, 11, 23, 47]
    ref = list(greedy(m, ids, 40))
    scripts = {"all": lambda n: True, "none": lambda n: False, "every 3rd wrong": lambda n: n % 3 != 1,
               "mixed": lambda n: (n * 7) % 5 not in (1, 3)}
    cases, used, cut, dropped = 0, 0, 0, 0
    for gamma in (0, 1, 2, 3):
        for name, hit in (scripts.items() if gamma else [("no drafts", None)]):
            for ahead, board in ((1, False), (2, False), (3, False), (99, False), (2, True), (3, True), (99, True),
                                 (1, "late"), (99, "late")):
                for n, eos, close in ((14, (), None), (5, (), None), (14, (ref[9],), None), (14, (), 7)):
                    want = run(m, ref, hit, ids, n, 0, gamma, eos, close, vocab)
                    got = run(m, ref, hit, ids, n, ahead, gamma, eos, close, vocab, board)
                    why = f"gamma {gamma}, {name}, ahead {ahead}, board {board}, max {n}, eos {eos}, close {close}"
                    assert got[0] == want[0], (why, got[0], want[0])
                    for k in ("rounds", "accepted", "tokens", "rebuilds", "replays", "seen", "hit"):
                        assert getattr(got[1], k) == getattr(want[1], k), (why, k)
                    same_cache(got[2], want[2], why)
                    if close is None:
                        check_cache(m, got[2], ids + got[0], why)
                    cases += 1
                    used += got[1].ahead_used
                    cut += got[1].ahead_cut
                    dropped += got[1].ahead_dropped - got[1].ahead_cut
    # Every branch ran: a round built ahead taken as the next, one cut at a hand-off, one dropped after its whole build.
    assert used and cut and dropped, (used, cut, dropped)
    print(f"  {cases} runs: tokens, rounds and every cache array equal to the loop without ahead; rounds built ahead "
          f"{used} taken, {cut} cut at a hand-off, {dropped} dropped whole  OK")


if __name__ == "__main__":
    test_ahead()
    print("test_spec_ahead: ok")
