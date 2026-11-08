#!/usr/bin/env python3
"""foldattn.py: folded GQA attention for dense multi-row verify passes.

    python3 tests/test_foldattn.py      # toy only, random tensors and a toy qwen3

Three checks:

 1. numerics  -- fold vs stock mx.fast SDPA vs an fp32 reference on random
                 tensors, across head shapes, N (not a tile multiple, K/V a
                 slice of a longer buffer like a KVCache hands out), T, chunk
                 counts, bf16/fp16, batch 2, and scores scaled up to sink size.
                 The fold must never land further from fp32 than stock does
                 (beyond one output ulp); max|fold - stock| is reported.
 2. dispatch  -- a toy bf16 qwen3 with the fold installed: an empty table or a
                 T=1 step or an uncalibrated bucket never reaches the kernel
                 and stays bit-identical to stock; a routed (T, bucket) goes
                 through it on every layer with the same argmax.
 3. calibrate -- two tiny buckets: runs, and only routes with a chunk count
                 from CHUNKS.
"""

import math
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import foldattn                                                # noqa: E402
from mlx_lm.models import qwen3                                # noqa: E402
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache  # noqa: E402


def reference(q, k, v, scale):
    """fp32, GQA repeated, bottom-right causal."""
    T, n = q.shape[2], k.shape[2]
    rep = q.shape[1] // k.shape[1]
    q, k, v = (a.astype(mx.float32) for a in (q, k, v))
    k, v = mx.repeat(k, rep, axis=1), mx.repeat(v, rep, axis=1)
    s = (q * scale) @ k.swapaxes(-1, -2)
    s = mx.where(mx.arange(n)[None] - (n - T) > mx.arange(T)[:, None], -mx.inf, s)
    return mx.softmax(s, axis=-1) @ v


def check_numerics():
    mx.random.seed(0)
    worst_gap = worst_diff = 0.0
    cases = 0
    for hq, hkv, d in ((32, 8, 128), (16, 8, 128), (8, 2, 64)):
        for n in (37, 300, 4100):
            for T in (2, 3, 5, 8):
                for amp in (1, 6):
                    for dtype, B in ((mx.bfloat16, 1), (mx.float16, 2)):
                        scale = d ** -0.5
                        q = (mx.random.normal((B, hq, T, d)) * amp).astype(dtype)
                        kb = mx.random.normal((B, hkv, n + 200, d)).astype(dtype)
                        # a sink: one key every query of its head leans on
                        sink = mx.mean(q.reshape(B, hkv, -1, d), axis=2) * 3
                        kb[:, :, 0, :] = sink.astype(dtype)
                        vb = mx.random.normal((B, hkv, n + 200, d)).astype(dtype)
                        k, v = kb[:, :, :n], vb[:, :, :n]
                        ref = reference(q, k, v, scale)
                        stock = mx.fast.scaled_dot_product_attention(
                            q, k, v, scale=scale, mask="causal").astype(mx.float32)
                        peak = mx.abs(ref).max().item()
                        ulp = 2.0 ** (math.frexp(peak)[1] - (8 if dtype == mx.bfloat16 else 11))
                        s_err = mx.abs(stock - ref).max().item()
                        for chunks in (1, 4, 64):
                            out = foldattn.fold(q, k, v, scale, chunks).astype(mx.float32)
                            f_err = mx.abs(out - ref).max().item()
                            assert f_err <= s_err + ulp, (hq, hkv, d, n, T, amp, dtype,
                                                               chunks, f_err, s_err)
                            worst_gap = max(worst_gap, f_err - s_err)
                            worst_diff = max(worst_diff, mx.abs(out - stock).max().item())
                            cases += 1
    # K/V with D strided (not what a KVCache hands out): the scalar load path
    q = mx.random.normal((1, 32, 5, 128)).astype(mx.bfloat16)
    k, v = (mx.random.normal((1, 8, 128, 300)).astype(mx.bfloat16).swapaxes(-1, -2)
            for _ in range(2))
    ref = reference(q, k, v, 128 ** -0.5)
    stock = mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5, mask="causal")
    out = foldattn.fold(q, k, v, 128 ** -0.5, 8)
    assert mx.abs(out - ref).max().item() <= mx.abs(stock - ref).max().item() + 2 ** -10
    cases += 1
    print(f"numerics: {cases} cases, fold never further from fp32 than stock "
          f"by more than {worst_gap:.2e}; max|fold - stock| = {worst_diff:.2e}")


def _toy():
    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=256, num_hidden_layers=3,
        intermediate_size=512, num_attention_heads=8, rms_norm_eps=1e-6,
        vocab_size=512, num_key_value_heads=2, max_position_embeddings=4096,
        rope_theta=1e6, head_dim=64, tie_word_embeddings=True)
    mx.random.seed(1)
    model = qwen3.Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


def check_dispatch():
    model = _toy()
    prompt = mx.random.randint(0, 512, (1, 300))
    cache = make_prompt_cache(model)
    mx.eval(model(prompt, cache=cache))

    def logits(rows):
        out = model(mx.random.randint(0, 512, (1, rows), key=mx.random.key(rows)),
                    cache=cache)
        mx.eval(out)
        trim_prompt_cache(cache, rows)
        return out

    stock = {r: logits(r) for r in (1, 3, 5)}
    calls = []
    real = foldattn.fold
    foldattn.fold = lambda *a: calls.append(a[0].shape) or real(*a)
    try:
        foldattn.install(model, table={})
        foldattn.install(model, table={})
        assert not hasattr(qwen3.scaled_dot_product_attention.stock, "stock"), "wrapped twice"
        for r in (1, 3, 5):
            assert mx.array_equal(logits(r), stock[r]), r
        assert not calls

        # N = 300 + 3 lands in bucket 8; T=5 and T=1 are not routed
        foldattn.install(model, table={(8, 2, 64, 3, 8): 8, (8, 2, 64, 1, 8): 8,
                                       (8, 2, 64, 5, 9): 8})
        for r in (1, 5):
            assert mx.array_equal(logits(r), stock[r]), r
        assert not calls, calls
        out = logits(3)
        assert len(calls) == 3 and calls[0] == (1, 8, 3, 64), calls
        diff = mx.abs(out.astype(mx.float32) - stock[3].astype(mx.float32)).max().item()
        assert mx.array_equal(mx.argmax(out, -1), mx.argmax(stock[3], -1))
        print(f"dispatch: stock untouched off-route, 3/3 layers folded on route, "
              f"max|logit diff| {diff:.3g}, argmax equal")
    finally:
        foldattn.fold = real
        foldattn.install(model, table={})


def check_calibrate():
    table = foldattn.calibrate({(8, 2, 64)}, 2, rows=(2, 8), buckets=(5, 11),
                               reps=3, verbose=True)
    assert all(k[:3] == (8, 2, 64) and k[3] in (2, 8) and k[4] in (5, 11)
               and c in foldattn.CHUNKS for k, c in table.items()), table
    print(f"calibrate: {len(table)} of 4 (T, bucket) routed")


if __name__ == "__main__":
    check_numerics()
    check_dispatch()
    check_calibrate()
    print("ok")
