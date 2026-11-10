#!/usr/bin/env python3
"""MOE_CUDA_BIAS_TAIL: route_cuda.bias_probs() and route_cuda.tail() give the bits of CacheBiasedMoE's MLX ops on CUDA.

bias_probs: softmax(precise) of [1, T, 256] bf16 logits, the cache bias toward the experts the slot table holds
(0 <= slot_of < lim), argpartition top-k, the budget's fallback to the honest top-k, take_along_axis and the
renormalisation; T 1-5, k 8 and 4, bias 0.5 / 2.0, budget off / 0.2 / 0.01, norm on and off, rows with tied logits
and rounded ones, tables from empty to half full with quota and ring slots. The combine: tail() against
(y * scores).sum(-2) + sigmoid(gate) * shared with the 35B's shapes (k 8, D 2048) on bias_probs' weights.
Skipped without MLX's CUDA backend.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_bias_tail_cuda.py [--rows 600]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mlx.core as mx

import stream_fast


def ops(logits, slot_of, lim, lam, tau, k, norm):
    """CacheBiasedMoE.__call__'s probability-routed lines with _cuda_route's table-mode mask."""
    gates = mx.softmax(logits, axis=-1, precise=True)
    s = np.array(slot_of)
    mask = mx.array(((s >= 0) & (s < lim)).astype(np.float32))
    spread = mx.max(gates, axis=-1, keepdims=True) - mx.min(gates, axis=-1, keepdims=True)
    rank = gates + (lam * spread) * mask
    inds = mx.argpartition(rank, kth=-k, axis=-1)[..., -k:]
    if tau > 0.0:
        honest = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        m_h = mx.take_along_axis(gates, honest, axis=-1).sum(axis=-1, keepdims=True)
        m_b = mx.take_along_axis(gates, inds, axis=-1).sum(axis=-1, keepdims=True)
        inds = mx.where(m_h - m_b > tau * m_h, honest, inds)
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if norm:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


def logits_of(rng, t, i):
    x = rng.normal(0, rng.uniform(0.5, 4.0), (t, 256)).astype(np.float32)
    if i % 5 == 0:
        j = rng.integers(0, 256, 6)
        x[:, j] = x[:, j[:1]]
    if i % 7 == 0:
        x = np.round(x * 4) / 4
    return mx.array(x).reshape(1, t, 256).astype(mx.bfloat16)


def table_of(rng):
    n = int(rng.integers(0, 128))
    slot = np.full(256, -1, np.int32)
    slot[rng.choice(256, n, replace=False)] = rng.choice(4200, n, replace=False).astype(np.int32)
    return mx.array(slot), 4000


def bits(a):
    return np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=600)
    a = ap.parse_args()
    if stream_fast.BACKEND != "cuda":
        print(f"SKIP: needs MLX's CUDA backend (this is {stream_fast.BACKEND})")
        return
    import route_cuda as rc

    rng = np.random.default_rng(1)
    n = bad = fell = 0
    for k in (8, 4):
        for lam, tau in ((0.5, 0.0), (0.5, 0.2), (0.5, 0.01), (2.0, 0.2)):
            for norm in (True, False):
                for i in range(a.rows // 16):
                    t = 1 + i % 5
                    lg = logits_of(rng, t, i)
                    slot_of, lim = table_of(rng)
                    want = ops(lg, slot_of, lim, lam, tau, k, norm)
                    got = rc.bias_probs(lg, slot_of, lam, tau, k, lim, norm)
                    mx.eval(want, got)
                    n += 1
                    ok = (got[0].dtype == mx.uint32 and got[1].dtype == mx.bfloat16
                          and np.array_equal(bits(want[0]).astype(np.int64), bits(got[0]).astype(np.int64))
                          and np.array_equal(bits(want[1]), bits(got[1])))
                    if not ok:
                        bad += 1
                        if bad <= 5:
                            print("differs", k, lam, tau, norm, t, bits(want[0])[:k], bits(got[0])[:k],
                                  bits(want[1])[:k], bits(got[1])[:k])
                    if tau > 0:
                        fell += not np.array_equal(np.array(ops(lg, slot_of, lim, lam, 0.0, k, norm)[0]),
                                                   np.array(want[0]))
        print(f"bias_probs k={k}: {n - bad}/{n} identical so far, {fell} rows fell back to honest", flush=True)
    assert bad == 0, f"bias_probs: {bad} of {n} differ"
    assert fell > 0, "the budget never fell back: the test does not reach it"

    # The combine on bias_probs' weights, the 35B's shapes (k 8, D 2048) and a narrow D.
    m = 0
    for d in (2048, 96):
        for i in range(a.rows // 20):
            lg = logits_of(rng, 1, i)
            slot_of, lim = table_of(rng)
            inds, scores = rc.bias_probs(lg, slot_of, 0.5, 0.2, 8, lim, True)
            y = mx.array(rng.normal(0, rng.uniform(0.1, 3.0), (1, 1, 8, d)).astype(np.float32)).astype(mx.bfloat16)
            shared = mx.array(rng.normal(0, 1.0, (1, 1, d)).astype(np.float32)).astype(mx.bfloat16)
            gate = mx.array(rng.normal(0, 3.0, (1, 1, 1)).astype(np.float32)).astype(mx.bfloat16)
            want = (y * scores[..., None]).sum(axis=-2) + mx.sigmoid(gate) * shared
            got = rc.tail(y, scores, shared, gate)
            mx.eval(want, got)
            m += 1
            assert np.array_equal(bits(want), bits(got)), f"tail differs at D {d}, row {i}"
    print(f"tail: {m}/{m} identical")
    print("ok: bias_probs and tail bit-equal to the MLX ops")


if __name__ == "__main__":
    main()
