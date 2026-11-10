#!/usr/bin/env python3
"""MOE_CUDA_GDN_GLUE: qwen3_5's GatedDeltaNet decode row through gdn_cuda.step_glue() and norm_gate() gives the bits of
mlx_lm's own lines (silu, normalize_qk, mx.sigmoid, compute_g, the recurrence kernel, the gated RMSNorm): a bf16 layer
with the 35B's shapes (hidden 2048, 16 key heads and 32 value heads of 128, conv 4) and a narrow one (4 / 8 heads of
64), random weights, A_log and dt_bias at several scales, a 5-position prefill through the original call, then 8
decode positions where the original and the patched call each advance their own copy of the cache: every output, conv
state and recurrent state must be bit-equal. Skipped without MLX's CUDA backend.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_gdn_glue_cuda.py
"""
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import mlx.core as mx

import stream_fast


def bits(a):
    return np.array(a.view(mx.uint16) if a.dtype == mx.bfloat16 else a.view(mx.uint32)).ravel()


def main():
    if stream_fast.BACKEND != "cuda":
        print(f"SKIP: needs MLX's CUDA backend (this is {stream_fast.BACKEND})")
        return
    import gdn_cuda
    import mlx_lm.models.qwen3_5 as q35
    from mlx_lm.models.cache import ArraysCache

    gdn_cuda.install_glue()             # stream_fast installed it already under MOE_RESIDENT=1
    orig, glue = q35.GatedDeltaNet._gdn_glue_orig, q35.GatedDeltaNet.__call__
    assert glue is not orig
    rng = np.random.default_rng(3)
    n = 0
    for hid, hk, dk, hv, dv in ((2048, 16, 128, 32, 128), (256, 4, 64, 8, 64)):
        for trial in range(4):
            cfg = SimpleNamespace(hidden_size=hid, linear_num_value_heads=hv, linear_num_key_heads=hk,
                                  linear_key_head_dim=dk, linear_value_head_dim=dv, linear_conv_kernel_dim=4,
                                  rms_norm_eps=1e-6)
            layer = q35.GatedDeltaNet(cfg)
            layer.set_dtype(mx.bfloat16)
            layer.A_log = mx.array(np.log(rng.uniform(0.01, 16.0, hv))).astype(mx.bfloat16)
            layer.dt_bias = mx.array(rng.normal(0, 1.0 + trial, hv)).astype(mx.bfloat16)
            layer.norm.weight = mx.array(rng.uniform(0.5, 2.0, dv)).astype(mx.bfloat16)
            layer.eval()
            scale = (0.3, 1.0, 3.0, 10.0)[trial]
            c0, c1 = ArraysCache(size=2), ArraysCache(size=2)
            x = mx.array(rng.normal(0, scale, (1, 5, hid))).astype(mx.bfloat16)
            mx.eval(orig(layer, x, None, c0))
            c1[0], c1[1] = c0[0], c0[1]
            c1.advance(5)
            for step in range(8):
                x = mx.array(rng.normal(0, scale, (1, 1, hid))).astype(mx.bfloat16)
                want = orig(layer, x, None, c0)
                got = glue(layer, x, None, c1)
                mx.eval(want, got, c0[0], c0[1], c1[0], c1[1])
                for name, a, b in (("out", want, got), ("conv state", c0[0], c1[0]), ("state", c0[1], c1[1])):
                    assert a.dtype == b.dtype and a.shape == b.shape, (name, a.dtype, b.dtype, a.shape, b.shape)
                    d = int((bits(a) != bits(b)).sum())
                    assert d == 0, f"{name} differs in {d} of {a.size} values (shape {hid, hk, dk, hv, dv}, " \
                                   f"trial {trial}, step {step})"
                n += 1
        print(f"hidden {hid}, {hk} x {dk} keys, {hv} x {dv} values: {n} decode positions bit-equal so far", flush=True)
    print(f"ok: {n} decode positions bit-equal (outputs, conv states, recurrent states)")


if __name__ == "__main__":
    main()
