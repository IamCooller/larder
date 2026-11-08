#!/usr/bin/env python3
"""fused_router.route_logits (MOE_ROUTE_KERNEL) against the MLX ops it
replaces (glue_logits, CacheBiasedMoE's order): the same experts in the same
order and bit-equal weights, on random fp32 logit rows at Flash-Next's shape
(512 experts, top-10) with and without the bias and the budget, plus rows
with ties; and the staging guess against route() on MLX's softmax. No model.

    python tests/test_route_kernel.py [--rows 3000]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np

import fused_router as fr


def rows(rng, n, e):
    """Logit rows like a router's: a few strong experts over a broad base,
    some with exact ties (repeated values) to test the order rule."""
    out = []
    for i in range(n):
        x = rng.normal(0, rng.uniform(0.5, 4.0), e).astype(np.float32)
        if i % 5 == 0:
            j = rng.integers(0, e, 6)
            x[j] = x[j[0]]
        if i % 7 == 0:
            x = np.round(x * 4) / 4
        out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    bad = 0
    for e, k in ((512, 10), (256, 8)):
        for lam, tau in ((0.0, 0.0), (0.5, 0.0), (0.5, 0.2), (2.0, 0.2)):
            n_ok = fell = 0
            for x in rows(rng, a.rows // 4, e):
                logits = mx.array(x).reshape(1, 1, e)
                m = np.zeros(e, np.float32)
                m[rng.choice(e, rng.integers(0, e // 3), replace=False)] = 1.0
                mask = mx.array(m)
                ri, rs = fr.glue_logits(logits, mask, lam, tau, k)
                ki, ks, kid = fr.route_logits(logits, mask, lam, tau if lam > 0 else 0.0, k)
                mx.eval(ri, rs, ki, ks, kid)
                same = (np.array_equal(np.array(ri), np.array(ki))
                        and np.array_equal(np.array(rs).view(np.uint32), np.array(ks).view(np.uint32))
                        and np.array_equal(np.array(ki).astype(np.int32), np.array(kid)))
                if not same:
                    bad += 1
                    if bad <= 3:
                        print("differs", e, k, lam, tau, np.array(ri).ravel(), np.array(ki).ravel(),
                              float(mx.max(mx.abs(rs - ks))))
                # The staging guess: route() on MLX's softmax, best first.
                if lam > 0:
                    gates = mx.softmax(logits, axis=-1, precise=True)
                    slot = mx.full((e,), -1, dtype=mx.int32)
                    want = fr.route(gates, mask, slot, lam, tau, k)[0][..., ::-1].astype(mx.int32)
                    prev = kid
                    got = fr.route_logits(logits, mask, lam, tau, k, prev=prev)
                    want = mx.concatenate([prev, want], axis=-1)
                    mx.eval(want, got)
                    if not np.array_equal(np.array(want), np.array(got)):
                        bad += 1
                        if bad <= 3:
                            print("guess differs", np.array(want).ravel(), np.array(got).ravel())
                        same = False
                n_ok += same
                # Coverage: rows where the budget sent the selection back to honest.
                if tau > 0 and lam > 0:
                    b_only, _ = fr.glue_logits(logits, mask, lam, 0.0, k)
                    h_only, _ = fr.glue_logits(logits, mask, 0.0, 0.0, k)
                    fell += (np.array_equal(np.array(ri), np.array(h_only))
                             and not np.array_equal(np.array(b_only), np.array(h_only)))
            print(f"E={e} k={k} lam={lam} tau={tau}: {n_ok}/{a.rows // 4} rows identical, "
                  f"{fell} fell back to honest", flush=True)
    # The probability-routed kernel (the 35B: 256 experts, top-8, bf16 logits,
    # norm_topk_prob; Qwen3-Next-80B: 512, top-10), 1..5 rows a pass.
    for e, k, lam, tau, norm in ((256, 8, 0.0, 0.0, True), (256, 8, 0.5, 0.0, True),
                                 (256, 8, 0.5, 0.2, True), (256, 8, 2.0, 0.2, False),
                                 (512, 10, 0.5, 0.2, True)):
        n_ok = n_rows = 0
        for i, x in enumerate(rows(rng, a.rows // 4, e)):
            t = 1 + i % 5
            xs = np.stack([x] + [rng.normal(0, 2, e).astype(np.float32) for _ in range(t - 1)])
            logits = mx.array(xs).reshape(1, t, e).astype(mx.bfloat16)
            m = np.zeros(e, np.float32)
            m[rng.choice(e, rng.integers(0, e // 3), replace=False)] = 1.0
            mask = mx.array(m)
            ri, rs = fr.glue_probs(logits, mask, lam, tau, k, norm)
            ki, ks, kid = fr.route_probs(logits, mask, lam, tau if lam > 0 else 0.0, k, norm)
            mx.eval(ri, rs, ki, ks, kid)
            same = (np.array_equal(np.array(ri), np.array(ki))
                    and np.array_equal(np.array(rs.view(mx.uint16)), np.array(ks.view(mx.uint16)))
                    and np.array_equal(np.array(ki).astype(np.int32), np.array(kid)))
            if not same:
                bad += 1
                if bad <= 3:
                    print("probs differ", e, k, lam, tau, norm, np.array(ri).ravel()[:k], np.array(ki).ravel()[:k])
            n_ok += same
            n_rows += 1
        print(f"probs E={e} k={k} lam={lam} tau={tau} norm={norm}: {n_ok}/{n_rows} passes identical "
              f"(1-5 rows each)", flush=True)
    assert bad == 0, f"{bad} rows differ"
    print("test_route_kernel: ok")


if __name__ == "__main__":
    main()
