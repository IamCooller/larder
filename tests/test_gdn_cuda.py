#!/usr/bin/env python3
"""gdn_cuda (MOE_CUDA_GDN): mlx_lm's gated delta recurrence as a CUDA kernel.

Everywhere: importing stream_fast installs it only on CUDA, and with MOE_CUDA_GDN=0 not even there; on
Metal and CPU gdn_cuda is never imported and mlx_lm's gated_delta_update is its own.
On CUDA: the kernel against a float64 reference of the same recurrence and against mlx_lm's
gated_delta_ops, for a gate a head and a vectorized gate, with a padding mask or none, GQA heads, Dk 64
and 128, T 1 / 7 / 37, from a non-zero state. Its state and y are no further from float64 than the ops
loop's own; a masked position gives y = 0 and leaves the state alone; a model module that imported the
name before the install (qwen3_next) is routed too.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_gdn_cuda.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mlx.core as mx
import mlx_lm.models.qwen3_next as q3n      # imported first: install() must route it as well
import mlx_lm.models.gated_delta as gd

import stream_fast


def ref(q, k, v, g, beta, state, mask):
    """The recurrence in float64 (masked steps skipped, their y 0)."""
    q, k, v, g, beta, s = (np.array(x.astype(mx.float32), dtype=np.float64) for x in (q, k, v, g, beta, state))
    B, T, Hk, Dk = q.shape
    r = v.shape[2] // Hk
    q, k = np.repeat(q, r, axis=2), np.repeat(k, r, axis=2)
    m = np.ones((B, T), bool) if mask is None else np.array(mask)
    ys = np.zeros(v.shape)
    for t in range(T):
        gt = g[:, t]
        sn = s * (gt[..., None, None] if gt.ndim == 2 else gt[..., None, :])
        kv = (sn * k[:, t, :, None, :]).sum(-1)
        sn = sn + k[:, t, :, None, :] * ((v[:, t] - kv) * beta[:, t, :, None])[..., None]
        y = (sn * q[:, t, :, None, :]).sum(-1)
        s = np.where(m[:, t, None, None, None], sn, s)
        ys[:, t] = np.where(m[:, t, None, None], y, 0.0)
    return ys, s


def case(B, T, Hk, Hv, Dk, Dv, vec, masked, seed):
    import gdn_cuda
    rng = np.random.default_rng(seed)
    bf = lambda *s: mx.array(rng.normal(0, 1, s).astype(np.float32)).astype(mx.bfloat16)
    q = mx.fast.rms_norm(bf(B, T, Hk, Dk), None, 1e-6) * Dk ** -1.0
    k = mx.fast.rms_norm(bf(B, T, Hk, Dk), None, 1e-6) * Dk ** -0.5
    v = bf(B, T, Hv, Dv)
    a = bf(B, T, Hv, Dk) if vec else bf(B, T, Hv)
    g = gd.compute_g(mx.array(rng.normal(0, 0.5, (Hv, Dk) if vec else Hv).astype(np.float32)), a,
                     mx.array(rng.normal(0, 0.5, (Hv, Dk) if vec else Hv).astype(np.float32)))
    beta = mx.sigmoid(bf(B, T, Hv))
    state = mx.array(rng.normal(0, 0.3, (B, Hv, Dv, Dk)).astype(np.float32))
    mask = mx.array(rng.random((B, T)) > 0.3) if masked else None
    y, s = gdn_cuda.recurrence(q, k, v, g, beta, state, mask)
    yo, so = gd.gated_delta_ops(q, k, v, g, beta, state, mask)
    mx.eval(y, s, yo, so)
    yr, sr = ref(q, k, v, g, beta, state, mask)
    assert y.dtype == mx.bfloat16 and s.dtype == mx.float32 and y.shape == (B, T, Hv, Dv)
    err = lambda a_, b_: float(np.abs(np.array(a_.astype(mx.float32), dtype=np.float64) - b_).max())
    m = np.ones((B, T), bool) if mask is None else np.array(mask)
    keep = m[:, :, None, None]
    e_s, e_so = err(s, sr), err(so, sr)
    e_y = float(np.abs(np.where(keep, np.array(y.astype(mx.float32)) - yr, 0)).max())
    e_yo = float(np.abs(np.where(keep, np.array(yo.astype(mx.float32)) - yr, 0)).max())
    tag = f"B{B} T{T} Hk{Hk} Hv{Hv} Dk{Dk} vec{int(vec)} mask{int(masked)}"
    print(f"  {tag}: state err {e_s:.2e} (ops {e_so:.2e}), y err {e_y:.2e} (ops {e_yo:.2e})")
    assert e_s <= 2 * e_so + 1e-6, (tag, e_s, e_so)
    assert e_y <= 1.5 * e_yo + 1e-6, (tag, e_y, e_yo)
    if masked:
        assert not np.array(y.astype(mx.float32))[~m].any(), "a masked position wrote y"


def main():
    import mlx_lm.models.qwen3_5 as q35      # imported after the install: takes the name from gated_delta
    on = stream_fast.BACKEND == "cuda" and os.environ.get("MOE_CUDA_GDN", "1") == "1"
    assert ("gdn_cuda" in sys.modules) == on, (stream_fast.BACKEND, sorted(m for m in sys.modules if "gdn" in m))
    for f in (gd.gated_delta_update, q3n.gated_delta_update, q35.gated_delta_update):
        assert bool(getattr(f, "_gdn_cuda", False)) == on, f
    if stream_fast.BACKEND != "cuda":
        print(f"backend {stream_fast.BACKEND}: gdn_cuda not imported, gated_delta_update is mlx_lm's")
        print("PASS")
        return
    # The switch: off, nothing is imported even on CUDA.
    env = dict(os.environ, MOE_CUDA_GDN="0")
    out = subprocess.run([sys.executable, "-c", "import sys, stream_fast; print('gdn_cuda' in sys.modules)"],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.stdout.strip().endswith("False"), out.stdout + out.stderr
    for i, (B, T, Hk, Hv, Dk, Dv, vec, masked) in enumerate([
            (1, 1, 2, 6, 128, 128, False, False), (1, 37, 2, 6, 128, 128, False, False),
            (2, 37, 2, 6, 128, 64, False, True), (1, 37, 4, 4, 64, 32, True, False),
            (2, 7, 2, 4, 128, 128, True, True), (1, 37, 16, 48, 128, 128, False, False)]):
        case(B, T, Hk, Hv, Dk, Dv, vec, masked, i)
    print("PASS")


if __name__ == "__main__":
    main()
