#!/usr/bin/env python3
"""Prefill fills the cache and must not materialize every position's logits.

Evaluating the model's output over a whole prompt allocates L x vocab: on this
model (vocab 248320, bf16) a 5.5k-token tool front is 2.7 GB of logits nobody
reads, and it was the bulk of the 4.5 -> 8.7 GB spike measured on 2026-09-08.
mlx_lm's own prefill evaluates only the cache state for the same reason.

The real `_prefill` and `_sync_draft_cache` run on an engine built with
__new__, against a stand-in model whose logits are as wide as the real ones.

    python tests/test_prefill_logits.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

import moe_stream

VOCAB = 248_320
PROMPT = 512            # 254 MB of bf16 logits if they are ever evaluated
BUDGET = 32 * 2**20     # what prefill may allocate beyond the cache


class FakeCache:
    def __init__(self):
        self.state = mx.zeros((1,))


class FakeModel:
    def __call__(self, x, cache):
        n = x.shape[1]
        for c in cache:
            c.state = c.state + n
        return mx.ones((1, n, VOCAB), dtype=mx.bfloat16) * cache[0].state[0]


def peak_while(fn):
    mx.eval(mx.zeros((1,)))
    mx.reset_peak_memory()
    before = mx.get_active_memory()
    fn()
    return mx.get_peak_memory() - before


def main():
    engine = moe_stream.StreamingLLM.__new__(moe_stream.StreamingLLM)
    engine.model, engine._cache, engine.prefill_step = FakeModel(), [FakeCache()], None

    grew = peak_while(lambda: engine._prefill(list(range(PROMPT))))
    assert engine._cache[0].state.item() == PROMPT, "prefill must still fill the cache"
    assert grew < BUDGET, f"target prefill allocated {grew / 2**20:.0f} MB: logits were evaluated"

    engine._draft_model, engine._draft_cache, engine._draft_seen = FakeModel(), [FakeCache()], []
    grew = peak_while(lambda: engine._sync_draft_cache(list(range(PROMPT))))
    assert engine._draft_cache[0].state.item() == PROMPT
    assert grew < BUDGET, f"draft prefill allocated {grew / 2**20:.0f} MB: logits were evaluated"

    print("ok: prefill fills the cache without evaluating logits")


if __name__ == "__main__":
    main()
