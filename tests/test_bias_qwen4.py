#!/usr/bin/env python3
"""CacheBiasedMoE reproduces the qwen4_exp SparseMoeBlock at lambda 0 and only
moves the SELECTION at lambda > 0.

Qwen3.8-Flash-Next routes by top-k of the raw fp32 logits and softmaxes the
k selected -- a different convention from Qwen3.5/Next, which the wrapper
had duplicated. Needs mlx_lm with qwen4_exp (the open PR branch); skips
otherwise. Tiny random block, no checkpoint.

    python test_bias_qwen4.py
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np

try:
    from mlx_lm.models import qwen4_exp
except ImportError:
    print("test_bias_qwen4: skipped (mlx_lm without qwen4_exp)")
    sys.exit(0)

import stream_fast


def main():
    mx.random.seed(0)
    args = qwen4_exp.TextArgs.from_dict(dict(
        hidden_size=64, moe_intermediate_size=32, shared_expert_intermediate_size=32,
        num_hidden_layers=2, num_experts=16, num_experts_per_tok=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=100,
        layer_types=["linear_attention", "full_attention"]))
    block = qwen4_exp.SparseMoeBlock(args)
    mx.eval(block.parameters())
    # the wrapper reads residency off the GLU; give the plain GLU an LRU
    block.switch_mlp._lru = collections.OrderedDict()
    w = stream_fast.CacheBiasedMoE(block, 0.0)
    assert w._logit_routed
    x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
    a, c = block(x), w(x)
    mx.eval(a, c)
    gap = float(mx.max(mx.abs(a.astype(mx.float32) - c.astype(mx.float32))))
    assert gap == 0.0, f"lambda 0 diverges from the block by {gap}"
    assert c.dtype == a.dtype

    # lambda > 0 with experts 0..3 "resident": the selection must move toward
    # them, the weights must still be the block's softmax over the selected.
    block.switch_mlp._lru.update((e, e) for e in range(4))
    w._lam = 5.0
    # The route log lives in StreamingSwitchGLU now; this plain GLU records
    # the selection it is handed itself.
    seen = []

    class _Rec(type(block.switch_mlp)):
        def __call__(self, x, indices):
            seen.append(np.asarray(indices).reshape(-1, indices.shape[-1]))
            return super().__call__(x, indices)
    block.switch_mlp.__class__ = _Rec
    y = w(x); mx.eval(y)
    chosen = seen[-1]
    resident_share = (chosen < 4).mean()
    assert resident_share > 0.5, f"bias did not pull the selection toward residents: {chosen}"
    logits = block.gate(x.astype(mx.float32))
    sel = mx.take_along_axis(logits, mx.array(chosen).reshape(1, 3, 4), axis=-1)
    wts = mx.softmax(sel, axis=-1, precise=True)
    assert abs(float(wts.sum(axis=-1).mean()) - 1.0) < 1e-5
    print("test_bias_qwen4: ok")


if __name__ == "__main__":
    main()
