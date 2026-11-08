#!/usr/bin/env python3
"""route_cuda (MOE_CUDA_SLOTS / MOE_CUDA_ROUTE): fused_router.route()'s cache-bias route as one CUDA kernel.

Everywhere: importing stream_fast imports route_cuda only on CUDA, and with both switches off not even there.
On CUDA: inds / ok / rhs bit-equal to CacheBiasedMoE's MLX ops (`ops` below: _select's order, then the slot
lookup _call_defer takes) on random router rows: E 512 k 10 and E 256 k 8, T 1-5, fp32 gates (Flash-Next: softmax
of fp32 logits) and bf16 (Qwen3.5 / 3.6), bias 0 / 0.5 / 2.0, budget off / 0.2 / 0.01, rows with tied logits;
tau set at each row's budget boundary (the float32 neighbours of (m_h - m_b) / m_h), where the order of the
two k-gate sums decides; the inds-only variant; and the table mode (the mask read from the slot table, 0 <=
slot_of < lim) against the ops on the mask that table implies.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_route_cuda.py [--rows 600]
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mlx.core as mx

import stream_fast


def ops(gates, mask, slot_of, lam, tau, k):
    """CacheBiasedMoE._select's ops in its order (no bias and no budget at lam 0), then the slot lookup."""
    if lam > 0.0:
        spread = mx.max(gates, axis=-1, keepdims=True) - mx.min(gates, axis=-1, keepdims=True)
        rank = gates + (lam * spread) * mask
    else:
        rank = gates
    inds = mx.argpartition(rank, kth=-k, axis=-1)[..., -k:]
    if tau > 0.0 and lam > 0.0:
        honest = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        m_h = mx.take_along_axis(gates, honest, axis=-1).sum(axis=-1, keepdims=True)
        m_b = mx.take_along_axis(gates, inds, axis=-1).sum(axis=-1, keepdims=True)
        inds = mx.where(m_h - m_b > tau * m_h, honest, inds)
    slots = mx.take(slot_of, inds)
    return inds, mx.all(slots >= 0), mx.maximum(slots, 0)


def gates_of(rng, t, e, dtype, i):
    """Router rows: a few strong experts over a broad base; every 5th row with exact ties, every 7th rounded."""
    x = rng.normal(0, rng.uniform(0.5, 4.0), (t, e)).astype(np.float32)
    if i % 5 == 0:
        j = rng.integers(0, e, 6)
        x[:, j] = x[:, j[:1]]
    if i % 7 == 0:
        x = np.round(x * 4) / 4
    return mx.softmax(mx.array(x).reshape(1, t, e).astype(dtype), axis=-1, precise=True)


def residency(rng, e):
    """A layer's residency: n experts held, some in quota slots (< lim) and some in ring slots (>= lim)."""
    n = int(rng.integers(0, e // 3))
    held = rng.choice(e, n, replace=False)
    lim = 4000
    slot = np.full(e, -1, np.int32)
    slot[held] = rng.choice(4200, n, replace=False).astype(np.int32)
    return slot, lim


def same(a, b):
    return all(np.array_equal(np.array(x).astype(np.int64).ravel(), np.array(y).astype(np.int64).ravel())
               for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=600)
    a = ap.parse_args()
    on = stream_fast.BACKEND == "cuda" and (os.environ.get("MOE_CUDA_ROUTE", "0") == "1"
                                            or os.environ.get("MOE_CUDA_SLOTS", "1") == "1"
                                            or os.environ.get("MOE_CUDA_COMBINE", "1") == "1")
    assert ("route_cuda" in sys.modules) == on, (stream_fast.BACKEND, on)
    if stream_fast.BACKEND != "cuda":
        print(f"backend {stream_fast.BACKEND}: route_cuda not imported")
        print("PASS")
        return
    env = dict(os.environ, MOE_CUDA_ROUTE="0", MOE_CUDA_SLOTS="0", MOE_CUDA_COMBINE="0")
    out = subprocess.run([sys.executable, "-c", "import sys, stream_fast; print('route_cuda' in sys.modules)"],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.stdout.strip().endswith("False"), out.stdout + out.stderr
    import route_cuda as rc

    rng = np.random.default_rng(0)
    bad = n = fell = edge = 0
    for dtype in (mx.float32, mx.bfloat16):
        for e, k in ((512, 10), (256, 8)):
            for lam, tau in ((0.0, 0.0), (0.0, 0.2), (0.5, 0.0), (0.5, 0.2), (0.5, 0.01), (2.0, 0.2)):
                for i in range(a.rows // 12):
                    t = 1 + i % 5
                    g = gates_of(rng, t, e, dtype, i)
                    slot, lim = residency(rng, e)
                    m = np.zeros(e, np.float32)
                    m[rng.choice(e, rng.integers(0, e // 3), replace=False)] = 1.0
                    mask, slot_of = mx.array(m), mx.array(slot)
                    # mask mode, then the inds-only variant, then the table mode on the mask it implies
                    want = ops(g, mask, slot_of, lam, tau, k)
                    got = rc.route(g, mask, slot_of, lam, tau, k)
                    only = rc.route(g, mask, slot_of, lam, tau, k, slots=False)
                    tmask = mx.array(((slot >= 0) & (slot < lim)).astype(np.float32))
                    want_t = ops(g, tmask, slot_of, lam, tau, k)
                    got_t = rc.route(g, None, slot_of, lam, tau, k, lim=lim)
                    mx.eval(want, got, only, want_t, got_t)
                    ok = (same(want, got) and same(want[:1], only) and same(want_t, got_t)
                          and got[0].dtype == mx.uint32 and got[2].dtype == mx.int32)
                    n += 1
                    if not ok:
                        bad += 1
                        if bad <= 5:
                            print("differs", dtype, e, k, lam, tau, t, np.array(want[0]).ravel()[:k],
                                  np.array(got[0]).ravel()[:k], np.array(want_t[0]).ravel()[:k],
                                  np.array(got_t[0]).ravel()[:k])
                    if tau > 0 and lam > 0:
                        b_only = ops(g, mask, slot_of, lam, 0.0, k)[0]
                        fell += not np.array_equal(np.array(b_only), np.array(want[0]))
            print(f"{dtype} E={e} k={k}: {n - bad}/{n} passes identical so far, {fell} fell back to honest",
                  flush=True)
        # The budget's boundary: tau at the float32 neighbours of row 0's (m_h - m_b) / m_h, where the sums'
        # order decides the comparison. T 1 and 3; the flip count says the boundary was hit.
        for i in range(a.rows // 6):
            t = 1 + 2 * (i % 2)
            e, k = 512, 10
            g = gates_of(rng, t, e, dtype, i)
            m = np.zeros(e, np.float32)
            m[rng.choice(e, rng.integers(5, e // 3), replace=False)] = 1.0
            mask, slot_of = mx.array(m), mx.array(np.arange(e, dtype=np.int32))
            lam = 0.5
            spread = mx.max(g, axis=-1, keepdims=True) - mx.min(g, axis=-1, keepdims=True)
            rank = g + (lam * spread) * mask
            bi = mx.argpartition(rank, kth=-k, axis=-1)[..., -k:]
            hi = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
            mh = mx.take_along_axis(g, hi, axis=-1).sum(axis=-1).astype(mx.float32)
            mb = mx.take_along_axis(g, bi, axis=-1).sum(axis=-1).astype(mx.float32)
            mh, mb = float(np.array(mh).ravel()[0]), float(np.array(mb).ravel()[0])
            if mh <= mb:
                continue
            c = np.float32((mh - mb) / mh)
            picks = []
            for tau in (np.nextafter(np.nextafter(c, 0), 0), np.nextafter(c, 0), c, np.nextafter(c, 1),
                        np.nextafter(np.nextafter(c, 1), 1)):
                tau = float(tau)
                want = ops(g, mask, slot_of, lam, tau, k)
                got = rc.route(g, mask, slot_of, lam, tau, k)
                mx.eval(want, got)
                n += 1
                if not same(want, got):
                    bad += 1
                    if bad <= 5:
                        print("boundary differs", dtype, t, tau, np.array(want[0]).ravel()[:k],
                              np.array(got[0]).ravel()[:k])
                picks.append(np.array(want[0]).tobytes())
            edge += len(set(picks)) > 1
        print(f"{dtype} boundary: {edge} rows flipped within 4 float32 steps of tau", flush=True)
    print(f"{n - bad}/{n} passes identical, {fell} fell back to honest, {edge} boundary flips")
    assert bad == 0, f"{bad} passes differ"

    # MOE_CUDA_SLOTS' table, made as _init_arena makes it: the host's in-place writes through the numpy view reach
    # the kernel's table mask and the uint32 view's take, write after write, with no new array in between.
    tab = mx.full((512,), -1, dtype=mx.int32)
    mx.eval(tab)
    tnp = np.array(tab, copy=False)
    u32 = tab.view(mx.uint32)
    mx.eval(u32)
    g = gates_of(rng, 1, 512, mx.float32, 1)
    for step in range(50):
        held = rng.choice(512, 60, replace=False)
        tnp[:] = -1
        tnp[held] = rng.choice(4200, 60, replace=False).astype(np.int32)
        want = ops(g, mx.array(((tnp >= 0) & (tnp < 4000)).astype(np.float32)), mx.array(tnp.copy()), 0.5, 0.2, 10)
        got = rc.route(g, None, tab, 0.5, 0.2, 10, lim=4000)
        rows = mx.take(u32, got[0])
        mx.eval(want, got, rows)
        sel = np.array(got[0]).ravel()
        assert same(want, got), step
        assert np.array_equal(np.array(rows).ravel(), tnp[sel].astype(np.uint32)), step
    print("table: 50 in-place rewrites seen by the kernel's mask and the uint32 take")
    check_tail(rc, rng, a.rows)
    print("PASS")


def check_tail(rc, rng, rows):
    """MOE_CUDA_COMBINE: probs() and tail() against mlx_lm's block ops bit for bit, on their own and as a whole
    256-expert block (qwen3_next's, 4-bit, shared expert and its gate) before and after install_tail."""
    bad = n = 0
    for i in range(rows):
        lg = rng.normal(0, rng.uniform(0.3, 6.0), (1, 1, 256)).astype(np.float32)
        if i % 5 == 0:
            j = rng.integers(0, 256, 12)
            lg[..., j] = lg[..., j[:1]]
        if i % 7 == 0:
            lg = np.round(lg * 2) / 2
        logits = mx.array(lg).astype(mx.bfloat16)
        for norm in (True, False):
            g = mx.softmax(logits, axis=-1, precise=True)
            inds = mx.argpartition(g, kth=-8, axis=-1)[..., -8:]
            sc = mx.take_along_axis(g, inds, axis=-1)
            if norm:
                sc = sc / sc.sum(axis=-1, keepdims=True)
            gi, gs = rc.probs(logits, 8, norm)
            y = mx.array(rng.normal(0, rng.uniform(0.01, 3.0), (1, 1, 8, 2048)).astype(np.float32)).astype(mx.bfloat16)
            sh = mx.array(rng.normal(0, 1.0, (1, 1, 2048)).astype(np.float32)).astype(mx.bfloat16)
            gt = mx.array(rng.normal(0, 4.0, (1, 1, 1)).astype(np.float32)).astype(mx.bfloat16)
            want = (y * sc[..., None]).sum(axis=-2) + mx.sigmoid(gt) * sh
            got = rc.tail(y, sc, sh, gt)
            mx.eval(inds, sc, gi, gs, want, got)
            n += 1
            ok = (np.array_equal(np.array(inds), np.array(gi)) and gi.dtype == mx.uint32
                  and np.array_equal(np.array(sc.view(mx.uint16)), np.array(gs.view(mx.uint16)))
                  and np.array_equal(np.array(want.view(mx.uint16)), np.array(got.view(mx.uint16))))
            if not ok:
                bad += 1
                if bad <= 5:
                    print("tail differs", i, norm, np.array(inds).ravel(), np.array(gi).ravel(),
                          np.array(sc.astype(mx.float32)).ravel(), np.array(gs.astype(mx.float32)).ravel(),
                          int(mx.sum(want != got).item()), "of 2048 combined values")
    print(f"probs + tail: {n - bad}/{n} rows identical (router inds, scores, combined row)", flush=True)
    assert bad == 0, f"{bad} rows differ"

    from mlx_lm.models import qwen3_next as q3n
    import mlx.nn as nn
    args = q3n.ModelArgs(model_type="qwen3_next", hidden_size=2048, num_hidden_layers=1, intermediate_size=512,
                         num_attention_heads=16, num_key_value_heads=2, head_dim=256, linear_num_value_heads=32,
                         linear_num_key_heads=16, linear_key_head_dim=128, linear_value_head_dim=128,
                         linear_conv_kernel_dim=4, num_experts=256, num_experts_per_tok=8, decoder_sparse_step=1,
                         shared_expert_intermediate_size=512, mlp_only_layers=[], moe_intermediate_size=128,
                         rms_norm_eps=1e-6, vocab_size=1000, rope_theta=1e7, partial_rotary_factor=0.25,
                         max_position_embeddings=4096, norm_topk_prob=True, full_attention_interval=4)
    mx.random.seed(3)
    blk = q3n.Qwen3NextSparseMoeBlock(args)
    blk.set_dtype(mx.bfloat16)
    nn.quantize(blk, group_size=64, bits=4,
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and not p.endswith("gate"))
    nn.quantize(blk, group_size=64, bits=8, class_predicate=lambda p, m: isinstance(m, nn.Linear))
    mx.eval(blk.parameters())
    xs = [mx.random.normal((1, 1, 2048)).astype(mx.bfloat16) * float(s) for s in rng.uniform(0.2, 3.0, 64)]
    xs.append(mx.random.normal((1, 5, 2048)).astype(mx.bfloat16))
    ref = [blk(x) for x in xs]
    mx.eval(ref)
    assert rc.install_tail([blk]) == 1 and rc.install_tail([blk]) == 0
    got = [blk(x) for x in xs]
    mx.eval(got)
    diff = sum(not np.array_equal(np.array(r.view(mx.uint16)), np.array(g.view(mx.uint16))) for r, g in zip(ref, got))
    print(f"block: {len(xs) - diff}/{len(xs)} rows identical through install_tail (64 decode rows, one 5-row pass)")
    assert diff == 0, f"{diff} block rows differ"


if __name__ == "__main__":
    main()
