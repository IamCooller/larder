#!/usr/bin/env python3
"""MOE_CUDA_ROWS: the kernels of a T-row verify pass give each row the bits of the one-row pass (MLX CUDA).

- route_cuda.qmv_rows: a 4-bit affine matmul of T bf16 rows against mx.quantized_matmul on each row alone, at the 35B's
  dense shapes (lm_head included), g64 and g32, T 2-5; the same launch twice gives the same bits.
- route_cuda.probs / tail on T rows against the same kernels row by row and against mlx_lm's ops (softmax precise,
  argpartition, take_along_axis, the renormalisation; the combine), tied and rounded logits, k 8 and 4.
- gdn_cuda.step_glue through qwen3_5's patched GatedDeltaNet (4-bit projections) on T rows against T one-row calls on a
  copy of the cache: outputs, conv state and recurrent state, the 35B's shapes and a narrow layer.
- route_cuda.rope_rows: T positions against mx.fast.rope on each row-contiguous position alone (MLX's rope_single).
- moe_pipe_cuda.sync_rows(sort=True): the job's rows sorted by slot (stable), and the Q4_0 experts through the run
  kernels in that order, unsorted, against the decode chain on the job's rows.
- moe_pipe_cuda.verdict / peek (MOE_SPEC_AHEAD): a round's argmax rows and draft ids read from the mapped block under the
  round's number, never before the GPU has run the kernel.
Skipped without MLX's CUDA backend.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_rows_cuda.py
"""
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("MOE_CUDA_ROWS", "5")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mlx.core as mx
import mlx.nn as nn

import stream_fast


def bits(a):
    a = mx.contiguous(a)
    return np.array(a.view({2: mx.uint16, 4: mx.uint32}[a.dtype.size])).ravel()


def same(a, b, what):
    mx.eval(a, b)
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
    d = int((bits(a) != bits(b)).sum())
    assert d == 0, f"{what}: {d} of {a.size} values differ"


def bf(rng, *shape, s=1.0):
    return (mx.random.normal(shape, key=mx.random.key(int(rng.integers(1 << 30)))) * s).astype(mx.bfloat16)


def test_qmv(rng):
    import route_cuda as rc
    n = 0
    for k, out, g in ((2048, 8192, 64), (2048, 4096, 64), (2048, 32, 64), (4096, 2048, 64), (2048, 512, 64),
                      (512, 2048, 64), (2048, 248320, 64), (2560, 640, 32), (640, 2560, 32)):
        wq, sc, bi = mx.quantize(bf(rng, out, k, s=0.05), group_size=g, bits=4)
        x = bf(rng, 1, 5, k, s=float(rng.uniform(0.3, 3.0)))
        assert rc.qmv_fits(x, wq, sc, g, 4, "affine", bi)
        for t in (2, 3, 4, 5):
            got = rc.qmv_rows(x[:, :t], wq, sc, bi, g)
            again = rc.qmv_rows(x[:, :t], wq, sc, bi, g)
            same(got, again, f"qmv_rows {k}->{out} T={t} twice")
            for i in range(t):
                want = mx.quantized_matmul(x[:, i:i + 1], wq, sc, bi, transpose=True, group_size=g, bits=4)
                same(got[:, i:i + 1], want, f"qmv_rows {k}->{out} g{g} T={t} row {i}")
                n += 1
        del wq, sc, bi
    print(f"qmv_rows: {n} rows bit-equal to the one-row qmv", flush=True)


def test_router(rng):
    import route_cuda as rc
    n = 0
    for trial in range(60):
        t = int(rng.integers(2, 6))
        k = (8, 4)[trial % 2]
        norm = trial % 3 != 0
        x = rng.normal(0, rng.uniform(0.5, 4.0), (t, 256)).astype(np.float32)
        if trial % 5 == 0:
            j = rng.integers(0, 256, 6)
            x[:, j] = x[:, j[:1]]
        if trial % 7 == 0:
            x = np.round(x * 4) / 4
        lg = mx.array(x).reshape(1, t, 256).astype(mx.bfloat16)
        inds, scores = rc.probs(lg, k, norm)
        y = bf(rng, 1, t, k, 2048)
        shared, gate = bf(rng, 1, t, 2048), bf(rng, 1, t, 1)
        out = rc.tail(y, scores, shared, gate)
        for i in range(t):
            i1, s1 = rc.probs(lg[:, i:i + 1], k, norm)
            same(inds[:, i:i + 1], i1, f"probs inds T={t} row {i}")
            same(scores[:, i:i + 1], s1, f"probs scores T={t} row {i}")
            g = mx.softmax(lg[:, i:i + 1], axis=-1, precise=True)
            wi = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
            ws = mx.take_along_axis(g, wi, axis=-1)
            if norm:
                ws = ws / ws.sum(axis=-1, keepdims=True)
            same(i1, wi.astype(mx.uint32), f"probs inds vs ops row {i}")
            same(s1, ws, f"probs scores vs ops row {i}")
            o1 = rc.tail(y[:, i:i + 1], s1, shared[:, i:i + 1], gate[:, i:i + 1])
            same(out[:, i:i + 1], o1, f"tail T={t} row {i}")
            n += 1
    print(f"probs / tail: {n} rows bit-equal to the one-row kernels and the ops", flush=True)


def test_gdn(rng):
    import gdn_cuda
    import mlx_lm.models.qwen3_5 as q35
    from mlx_lm.models.cache import ArraysCache

    gdn_cuda.install_glue(5)
    glue = q35.GatedDeltaNet.__call__
    assert glue is not q35.GatedDeltaNet._gdn_glue_orig
    n = 0
    for hid, hk, dk, hv, dv in ((2048, 16, 128, 32, 128), (256, 4, 64, 8, 64)):
        for trial in range(3):
            cfg = SimpleNamespace(hidden_size=hid, linear_num_value_heads=hv, linear_num_key_heads=hk,
                                  linear_key_head_dim=dk, linear_value_head_dim=dv, linear_conv_kernel_dim=4,
                                  rms_norm_eps=1e-6)
            layer = q35.GatedDeltaNet(cfg)
            layer.set_dtype(mx.bfloat16)
            # 4-bit projections as the pack has them: a bf16 Linear on T rows is a gemm, not T gemvs
            nn.quantize(layer, group_size=64, bits=4, class_predicate=lambda _, m: isinstance(m, nn.Linear))
            layer.A_log = mx.array(np.log(rng.uniform(0.01, 16.0, hv))).astype(mx.bfloat16)
            layer.dt_bias = mx.array(rng.normal(0, 1.0 + trial, hv)).astype(mx.bfloat16)
            layer.norm.weight = mx.array(rng.uniform(0.5, 2.0, dv)).astype(mx.bfloat16)
            layer.eval()
            scale = (0.3, 1.0, 5.0)[trial]
            c0 = ArraysCache(size=2)
            mx.eval(q35.GatedDeltaNet._gdn_glue_orig(layer, bf(rng, 1, 5, hid, s=scale), None, c0))
            for t in (2, 3, 4, 5):
                c1, c2 = ArraysCache(size=2), ArraysCache(size=2)
                c1[0], c1[1], c2[0], c2[1] = c0[0], c0[1], c0[0], c0[1]
                x = bf(rng, 1, t, hid, s=scale)
                got = glue(layer, x, None, c1)
                want = mx.concatenate([glue(layer, x[:, i:i + 1], None, c2) for i in range(t)], axis=1)
                same(got, want, f"GDN T={t} out ({hid}, trial {trial})")
                same(c1[0], c2[0], f"GDN T={t} conv state")
                same(c1[1], c2[1], f"GDN T={t} state")
                c0[0], c0[1] = c1[0], c1[1]
                n += t
    print(f"GatedDeltaNet glue: {n} rows of T-row passes bit-equal to one-row calls (outputs and states)", flush=True)


def test_rebuild(rng):
    """spec.Speculator's taped 4-row verify of a GatedDeltaNet layer, then _rebuild's rerun of its first n rows from the
    snapshot (n 1-3), against n one-row glue calls from the snapshot: conv and recurrent states bit-equal."""
    import gdn_cuda
    import mlx_lm.models.qwen3_5 as q35
    from mlx_lm.models.cache import ArraysCache

    gdn_cuda.install_glue(5)
    cfg = SimpleNamespace(hidden_size=2048, linear_num_value_heads=32, linear_num_key_heads=16, linear_key_head_dim=128,
                          linear_value_head_dim=128, linear_conv_kernel_dim=4, rms_norm_eps=1e-6)
    layer = q35.GatedDeltaNet(cfg)
    layer.set_dtype(mx.bfloat16)
    nn.quantize(layer, group_size=64, bits=4, class_predicate=lambda _, m: isinstance(m, nn.Linear))
    layer.A_log = mx.array(np.log(rng.uniform(0.01, 16.0, 32))).astype(mx.bfloat16)
    layer.dt_bias = mx.array(rng.normal(0, 1.0, 32)).astype(mx.bfloat16)
    layer.norm.weight = mx.array(rng.uniform(0.5, 2.0, 128)).astype(mx.bfloat16)
    layer.eval()
    names = [k for k in layer if k.startswith("in_proj")]
    n_ok = 0
    for trial in range(6):
        c0 = ArraysCache(size=2)
        mx.eval(q35.GatedDeltaNet._gdn_glue_orig(layer, bf(rng, 1, 5, 2048), None, c0))
        snap = list(c0.cache)
        x = bf(rng, 1, 4, 2048, s=float(rng.uniform(0.3, 3.0)))
        rec = {}
        for k in names:
            def proj(v, f=layer[k], k=k):
                rec["x"], rec[k] = v, f(v)
                return rec[k]
            layer.__dict__[k] = proj
        layer(x, None, c0)                       # the verify pass, taped
        for k in names:
            del layer.__dict__[k]
        mx.eval(c0.cache, [rec[k] for k in names])
        for n in (1, 2, 3):
            c1, c2 = ArraysCache(size=2), ArraysCache(size=2)
            c1.cache, c2.cache = list(snap), list(snap)
            layer.__dict__.update({k: (lambda _, y=rec[k][:, :n]: y) for k in names})
            layer(rec["x"][:, :n], None, c1)     # spec._rebuild's rerun
            for k in names:
                del layer.__dict__[k]
            for i in range(n):
                layer(x[:, i:i + 1], None, c2)
            same(c1[0], c2[0], f"rebuild n={n} conv state, trial {trial}")
            same(c1[1], c2[1], f"rebuild n={n} recurrent state, trial {trial}")
            n_ok += 1
    print(f"GDN rebuild after a 4-row verify: {n_ok} reruns bit-equal to one-row calls (conv and recurrent states)",
          flush=True)


def test_rope(rng):
    """route_cuda.rope_rows against mx.fast.rope on each position alone, row-contiguous as a decode row is (MLX's
    rope_single), queries and keys of the 35B (16 / 2 heads of 256, 64 rotated, base 1e7) at random offsets."""
    import route_cuda as rc
    n = 0
    for h in (16, 2):
        x = bf(rng, 1, 4, h, 256).transpose(0, 2, 1, 3)       # [1, H, T, D] as the attention makes it
        for off in rng.integers(0, 40000, 700).tolist():
            for t in (2, 3, 4):
                got = rc.rope_rows(x[:, :, :t], 64, 1e7, 1.0, off)
                ones = [mx.fast.rope(mx.contiguous(x[:, :, i:i + 1]), 64, traditional=False, base=1e7, scale=1.0,
                                     offset=off + i) for i in range(t)]
                for i in range(t):
                    same(got[:, :, i:i + 1], ones[i], f"rope_rows H={h} offset {off} T={t} row {i}")
                    n += 1
    print(f"rope_rows: {n} rows bit-equal to the one-position kernel", flush=True)


def test_sort(rng):
    """moe_pipe_cuda.sync_rows(sort=True) on a table holding every id (no agent: `done` raised ahead as warm_one does),
    then the experts through the run kernels in slot order against the decode chain on the unsorted rows."""
    sys.path.append(os.path.join(ROOT, "native"))
    import moe_pipe_cuda as nv
    import gguf_experts as ge
    types = ("Q4_0",) * 3
    w = tuple(mx.array(ge.random_experts(rng, 40, r, c, "Q4_0")) for r, c in ((512, 2048), (512, 2048), (2048, 512)))
    n = 0
    for trial in range(40):
        t = int(rng.integers(2, 5))
        ids_np = rng.integers(0, 256, (1, t, 8))
        if trial % 2:
            ids_np[0, 1:, :4] = ids_np[0, 0, :4]        # rows that share experts
        slot = np.full(256, -1, np.int32)
        held = np.unique(ids_np)
        slot[held] = rng.choice(40, len(held), replace=False)
        tab, ids = mx.array(slot), mx.array(ids_np.astype(np.uint32))
        nv._blk[nv.DONE] = nv._blk[nv.CLAIMED] = int(nv._blk[nv.POSTED]) + 100
        rhs, srt, src, inv = nv.sync_rows(ids, 0, tab, sort=True)
        mx.eval(rhs, srt, src, inv)
        nv._blk[nv.SENT] = nv._blk[nv.POSTED]
        r, s, sr, iv = (np.array(a).ravel() for a in (rhs, srt, src, inv))
        assert (r == slot[ids_np.ravel()]).all(), "sync_rows: rows"
        order = np.argsort(r, kind="stable")
        assert (s == r[order]).all() and (sr == order // 8).all() and (iv[order] == np.arange(r.size)).all(), \
            f"sync_rows sort: {r} -> {s} {sr} {iv}"
        x = bf(rng, 1, t, 2048, s=float(rng.uniform(0.3, 3.0)))
        want = ge.experts(x.reshape(1, t, 1, 1, 2048), [(a,) for a in w], rhs, types).squeeze(-2)
        ys = ge.gemv(mx.take(x.reshape(t, 2048), src, axis=0), w, srt, 1, types, x.dtype)
        same(mx.take(ys, inv, axis=0).reshape(1, t, 8, 2048), want, f"experts in slot order T={t} trial {trial}")
        n += t
    print(f"handshake sort + run kernels: {n} rows bit-equal to the decode chain", flush=True)


def test_verdict(rng):
    """MOE_SPEC_AHEAD: moe_pipe_cuda.verdict writes a round's argmax rows and draft ids into the mapped block, peek finds
    them under the round's number only once the GPU has run it (a kernel queued behind a busy one is not seen early)."""
    sys.path.append(os.path.join(ROOT, "native"))
    import moe_pipe_cuda as nv
    n = 0
    for trial in range(60):
        t, g = int(rng.integers(1, 6)), int(rng.integers(0, 5))
        g = min(g, t - 1)
        am = mx.argmax(bf(rng, t, 4096), axis=-1)
        dr = mx.array(rng.integers(0, 248320, g), dtype=mx.int32)
        if trial % 3 == 0:
            # a long matmul chain ahead of it on the stream, so the host looks while the GPU is still busy
            x = bf(rng, 2048, 2048)
            for _ in range(40):
                x = (x @ x) * 0.01
            am = mx.depends([am], [x])[0]
        mark, k = nv.verdict(am, dr)
        early = nv.peek(k, t + g)
        mx.async_eval(am, dr, mark)
        t0 = time.perf_counter()
        while (got := nv.peek(k, t + g)) is None:
            assert time.perf_counter() - t0 < 30, f"round {k} never reached the block"
        want = np.array(am).tolist() + np.array(dr).tolist()
        assert got == want, (trial, got, want)
        assert early is None, f"round {k} seen before it was handed to the GPU"
        assert int(np.array(mark)[0]) == k and nv.peek(k + 1, 1) is None
        n += 1
    print(f"verdict: {n} rounds read from the mapped block, each under its own number", flush=True)


def test_selfcheck(rng):
    """stream_fast._rows_check passes the real kernels on a small 4-bit model and names the one forced to part, and
    _rows_off then turns the T-row path and the rounds built ahead off."""
    sys.path.append(os.path.join(ROOT, "native"))
    import moe_pipe_cuda as nv
    import route_cuda as rc
    layers = [nn.Linear(k, n, bias=False) for n, k in ((512, 2048), (2048, 512), (1024, 1024))]
    model = nn.Sequential(*layers, nn.RoPE(64, base=1e7))
    model.set_dtype(mx.bfloat16)        # the shapes qmv_rows serves: bf16 rows, scales and biases
    nn.quantize(model, group_size=64, bits=4)
    mx.eval(model.parameters())
    assert stream_fast._rows_check(model, nv, 2) is None, "a real kernel parted from MLX's one-row call"
    real = rc.qmv_rows
    rc.qmv_rows = lambda *a: real(*a) + 1
    try:
        why = stream_fast._rows_check(model, nv, 2)
    finally:
        rc.qmv_rows = real
    assert why and why.startswith("qmv_rows"), why
    rows, ahead = stream_fast._CUDA_ROWS, stream_fast._SPEC_AHEAD
    stream_fast._rows_off(why)
    assert stream_fast._CUDA_ROWS == 1 and stream_fast._SPEC_AHEAD == 0 and rc.ROWS == 1
    stream_fast._CUDA_ROWS, stream_fast._SPEC_AHEAD, rc.ROWS = rows, ahead, rows
    print(f"self-check: real kernels pass, a forced mismatch ({why}) turns the T-row path off", flush=True)


def main():
    if stream_fast.BACKEND != "cuda":
        print(f"SKIP: needs MLX's CUDA backend (this is {stream_fast.BACKEND})")
        return
    rng = np.random.default_rng(11)
    test_qmv(rng)
    test_router(rng)
    test_gdn(rng)
    test_rebuild(rng)
    test_rope(rng)
    test_sort(rng)
    test_verdict(rng)
    test_selfcheck(rng)
    print("ok")


if __name__ == "__main__":
    main()
