#!/usr/bin/env python3
"""MOE_ROUTE_K (route_k.py) on a tiny random qwen4_exp block, no checkpoint: unset and k1 = k2 = k are bit for bit
today's block, k1 < k matches a numpy reference in the selection and the weights, the cache-bias wrapper weighs
its biased selection by the unbiased top-k2 mass, and a mutant of the weights is caught.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_route_k.py
"""
import collections
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np

try:
    from mlx_lm.models import qwen4_exp
except ImportError:
    print("test_route_k: skipped (mlx_lm without qwen4_exp)")
    sys.exit(0)

import route_k
import stream_fast

E, K, H = 16, 4, 64
STOCK = qwen4_exp.SparseMoeBlock.__call__


def block(seed):
    mx.random.seed(seed)
    args = qwen4_exp.TextArgs.from_dict(dict(
        hidden_size=H, moe_intermediate_size=32, shared_expert_intermediate_size=32,
        num_hidden_layers=2, num_experts=E, num_experts_per_tok=K,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=100,
        layer_types=["linear_attention", "full_attention"]))
    b = qwen4_exp.SparseMoeBlock(args)
    mx.eval(b.parameters())
    seen = []

    class Rec(type(b.switch_mlp)):
        def __call__(self, x, indices):
            seen.append(np.array(indices))
            return super().__call__(x, indices)
    b.switch_mlp.__class__ = Rec
    b.switch_mlp._lru = collections.OrderedDict()
    return b, seen


def model_of(*blocks):
    return types.SimpleNamespace(layers=[types.SimpleNamespace(mlp=b) for b in blocks])


def bits(a):
    return np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a).tobytes()


def ref_weights(z, sel, k2):
    """numpy: p = softmax over all experts, weight_i = p_i / the top k2 probabilities' sum."""
    z = z.astype(np.float64)
    p = np.exp(z - z.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    mass = -np.sort(-p, axis=-1)[..., :k2].sum(-1, keepdims=True)
    return np.take_along_axis(p, sel, -1) / mass


def check(b, seen, x, k1, k2):
    """Largest gap of the block's output to the numpy reference, and the selection checked against numpy's top k1."""
    seen.clear()
    y = b(x)
    mx.eval(y)
    z = np.array(b.gate(x.astype(mx.float32)))
    got = seen[-1]
    want = np.argsort(-z, axis=-1, kind="stable")[..., :k1]
    assert got.shape[-1] == k1, got.shape
    assert (np.sort(got, -1) == np.sort(want, -1)).all(), (got, want)
    w = ref_weights(z, got.astype(np.int64), k2)
    ref = (b.switch_mlp(x, mx.array(got)) * mx.array(w.astype(np.float32))[..., None]).sum(axis=-2)
    ref = ref.astype(x.dtype) + mx.sigmoid(b.shared_expert_gate(x)) * b.shared_expert(x)
    return float(mx.max(mx.abs(y.astype(mx.float32) - ref.astype(mx.float32))))


def main():
    assert route_k.parse("") is None and route_k.parse("5:10") == (5, 10)
    for bad in ("10:5", "0:4", "5", "a:b"):
        try:
            route_k.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"parse({bad!r}) accepted")

    x = mx.random.normal((1, 3, H), key=mx.random.key(1)).astype(mx.bfloat16)
    b0, _ = block(0)
    y0 = b0(x)
    mx.eval(y0)

    # Unset and k1 = k2 = k: nothing marked, the class untouched, the output bit for bit.
    assert route_k.install(model_of(b0), stream_fast._moe_of, k=None) == 0
    assert route_k.install(model_of(b0), stream_fast._moe_of, k=(K, K)) == 0
    assert qwen4_exp.SparseMoeBlock.__call__ is STOCK and b0.top_k == K
    assert bits(b0(x)) == bits(y0)
    # The shared weights helper at the selection's width is the old line, op for op.
    lg = b0.gate(x.astype(mx.float32))
    idx = mx.argpartition(-lg, K - 1, axis=-1)[..., :K]
    old = mx.softmax(mx.take_along_axis(lg, idx, axis=-1), axis=-1, precise=True)
    assert bits(route_k.weights(lg, idx, K)) == bits(old)

    # k1 < k: selection and weights against numpy, k2 = k1 (plain renormalization) and k2 > k.
    for k1, k2 in ((2, 2), (2, 4), (2, 8), (3, 16)):
        b, seen = block(0)
        assert route_k.install(model_of(b), stream_fast._moe_of, k=(k1, k2)) == 1
        assert b.top_k == k1
        gap = check(b, seen, x, k1, k2)
        assert gap < 2e-2, f"k1 {k1} k2 {k2}: off the numpy reference by {gap}"
        w = route_k.weights(lg, mx.argpartition(-lg, k1 - 1, axis=-1)[..., :k1], k2)
        sel = np.array(mx.argpartition(-lg, k1 - 1, axis=-1)[..., :k1]).astype(np.int64)
        err = np.abs(np.array(w) - ref_weights(np.array(lg), sel, k2)).max()
        assert err < 1e-6, f"weights off numpy by {err}"
    # The patched class still runs an unmarked block as stock, bit for bit.
    assert qwen4_exp.SparseMoeBlock.__call__ is not STOCK
    assert bits(b0(x)) == bits(y0)
    # k2 = k1 is --top-k k1 exactly: the stock block at top_k k1.
    b, _ = block(0)
    route_k.install(model_of(b), stream_fast._moe_of, k=(2, 2))
    c, _ = block(0)
    c.top_k = 2
    assert bits(b(x)) == bits(STOCK(c, x))

    # Cache bias: the wrapper takes k1 from the block, moves the selection, and weighs by the unbiased top-k2 mass.
    b, seen = block(0)
    route_k.install(model_of(b), stream_fast._moe_of, k=(2, 8))
    w = stream_fast.CacheBiasedMoE(b, 0.0)
    assert w.top_k == 2 and w._logit_routed
    assert check(b, seen, x, 2, 8) < 2e-2
    seen.clear()
    ya = w(x)
    mx.eval(ya)
    assert np.abs(np.array(ya.astype(mx.float32)) - np.array(b(x).astype(mx.float32))).max() < 2e-2
    b.switch_mlp._lru.update((e, e) for e in range(4))
    w._lam = 5.0
    seen.clear()
    yb = w(x)
    mx.eval(yb)
    got = seen[0].astype(np.int64)
    assert (got < 4).mean() > 0.5, f"bias did not pull the selection: {got}"
    z = np.array(b.gate(x.astype(mx.float32)))
    wr = ref_weights(z, got, 8).astype(np.float32)
    ref = (b.switch_mlp(x, mx.array(got.astype(np.uint32))) * mx.array(wr)[..., None]).sum(axis=-2)
    ref = ref.astype(x.dtype) + mx.sigmoid(b.shared_expert_gate(x)) * b.shared_expert(x)
    gap = float(mx.max(mx.abs(yb.astype(mx.float32) - ref.astype(mx.float32))))
    assert gap < 2e-2, f"biased selection weighed off the unbiased reference by {gap}"

    # The native decode row's route kernel (Metal): k1 selected by the kernel, the weights replaced by the top-k2 mass.
    if mx.metal.is_available():
        global E
        E, e0 = 32, E
        b, _ = block(2)
        E = e0
        route_k.install(model_of(b), stream_fast._moe_of, k=(3, 10))
        w = stream_fast.CacheBiasedMoE(b, 0.0)
        x1 = x[:, :1]
        inds, sc = w._route_kernel(x1)
        z = np.array(b.gate(x1.astype(mx.float32)))
        got = np.array(inds).astype(np.int64)
        assert got.shape[-1] == 3
        assert set(got.ravel()) == set(np.argsort(-z.ravel())[:3].tolist())
        err = np.abs(np.array(sc) - ref_weights(z, got, 10)).max()
        assert err < 1e-6, f"route kernel weights off numpy by {err}"

    # A mutant (renormalize over the k1 selected, k2 ignored) must fail the numpy check.
    real = route_k.weights
    route_k.weights = lambda logits, inds, k2: real(logits, inds, inds.shape[-1])
    try:
        b, seen = block(0)
        route_k.install(model_of(b), stream_fast._moe_of, k=(2, 8))
        gap = check(b, seen, x, 2, 8)
    finally:
        route_k.weights = real
    assert gap > 5e-2, f"mutant not caught (gap {gap})"
    print(f"test_route_k: ok (mutant gap {gap:.3f})")


if __name__ == "__main__":
    main()
