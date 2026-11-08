#!/usr/bin/env python3
"""stream_fast's CUDA QSA decode path (MOE_QSA_ATTN on CUDA): a one-row decode pass
attends to the indexer's picks only, gathered by _qsa_gather for MLX's own SDPA.

A toy qwen4_exp model with Flash-Next's attention shapes (24 query heads, 2 KV heads
x 256, indexer 4 x 128, budget 2048, blocks of 4), random weights: its attention
layer is run twice on the same inputs, once with the engine's caches (pooled
indexer, step buffer, fp32 pool, the picks path) and once with mlx_lm's stock
caches, through a 4100-position prefill in pieces and 24 decode tokens:

  1. install measures the switch point (into a temporary ~/.cache) and keeps it;
  2. on every decode pass the picks path runs (forced from the budget on), the
     sorted top-k blocks it gathers equal stock's top-k from stock's own lines,
     and the SDPA output is >= 99% bit-equal to stock's masked SDPA with every
     difference within one bf16 step at its head's scale;
  3. a pass below the switch and every multi-row pass stay stock's, bit for bit.

No checkpoint. CUDA only (skips elsewhere); run on the PC:

    python3 tests/test_qsa_cuda.py
"""
import math
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlx.core as mx

try:
    from mlx_lm.models import qwen4_exp as q4
except ImportError:
    print("test_qsa_cuda: skipped (mlx_lm without qwen4_exp)")
    sys.exit(0)

import stream_fast as sf

if sf.BACKEND != "cuda":
    print(f"test_qsa_cuda: skipped (backend {sf.BACKEND})")
    sys.exit(0)

HOME = tempfile.mkdtemp(prefix="qsa_cuda_home_")
Path.home = classmethod(lambda cls: Path(HOME))     # the calibration file goes here, not into ~/.cache


def toy_model():
    args = q4.ModelArgs.from_dict({"model_type": "qwen4_exp", "text_config": dict(
        hidden_size=512, num_hidden_layers=2, num_attention_heads=24, num_key_value_heads=2,
        head_dim=256, partial_rotary_factor=0.25, vocab_size=256, num_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
        linear_value_head_dim=32, hc_count=4, hc_lowrank=16, indexer_n_heads=4,
        indexer_kv_heads=1, indexer_head_dim=128, indexer_budget=2048,
        indexer_compress_ratio=4, ple_layer_ids=[], rope_theta=10_000_000.0,
        layer_types=["linear_attention", "full_attention"])})
    m = q4.Model(args)
    m.set_dtype(mx.bfloat16)
    mx.eval(m.parameters())
    return m


def stock_top(ix, x, rope, keys, offset):
    """QSAIndexer.__call__'s lines up to the top-k over the stock cache's keys (no padding), sorted."""
    B, S, _ = x.shape
    q = ix.index_qk_proj(x)[..., :ix.n_heads * ix.head_dim].reshape(B, S, ix.n_heads, ix.head_dim)
    cr = ix.compress_ratio
    nb = keys.shape[1] // cr
    p = keys[:, :nb * cr].reshape(B, nb, cr, ix.head_dim)
    p = ix.k_layernorm(p.astype(mx.float32).mean(axis=2).astype(keys.dtype))
    cos_k, sin_k = rope((mx.arange(nb) * cr)[None, :])
    p = q4._rope_partial(p, cos_k, sin_k)
    q_pos = q4._positions(offset, S)
    cos_q, sin_q = rope(q_pos)
    q = q4._rope_partial(ix.q_layernorm(q), cos_q[:, :, None, :], sin_q[:, :, None, :])
    s = mx.einsum("bshd,bnd->bsnh", q.astype(mx.float32), p.astype(mx.float32))
    s = mx.maximum(s, 0).sum(axis=-1) / math.sqrt(ix.head_dim)
    k = min(ix.block_topk, nb)
    return mx.sort(mx.argpartition(-s, k - 1, axis=-1)[..., :k].reshape(-1))


def main():
    m = toy_model()
    assert sf.install_qsa_pool(m) == 1 and sf._QSA_ATTN_ON, "the CUDA QSA attention did not install"
    attn, rope = m.layers[1].self_attn, m.model.rope
    shape = sf._qsa_shape(attn)
    measured = sf._QSA_SWITCH[shape]
    assert isinstance(measured, int) and measured > attn.indexer.token_budget, measured
    assert list((Path(HOME) / ".cache" / "moe-ssd-offload").glob("qsaattn-*.json"))
    print(f"  switch measured at install: picks from {measured} keys")

    # Spies: the SDPA outputs of both paths and the blocks the picks path gathers.
    sdpa, gather = mx.fast.scaled_dot_product_attention, sf._qsa_gather
    seen = {"sdpa": [], "tb": []}

    def sdpa_spy(*a, **k):
        out = sdpa(*a, **k)
        seen["sdpa"].append(out)
        return out

    def gather_spy(k, v, tb, cr):
        seen["tb"].append(tb)
        return gather(k, v, tb, cr)
    mx.fast.scaled_dot_product_attention, sf._qsa_gather = sdpa_spy, gather_spy

    eng, ref = q4._AttnCache(), q4._AttnCache()
    ref.indexer = sf._QSA_STOCK[1]()
    assert type(eng.indexer) is not type(ref.indexer)
    mx.random.seed(0)
    hidden = attn.q_proj.weight.shape[1]

    def both(x):
        """The layer on the engine's caches and on stock's: outputs, SDPA outputs, gathered blocks."""
        seen["sdpa"].clear()
        seen["tb"].clear()
        a = attn(x, rope, None if x.shape[1] == 1 else "causal", eng, eng.indexer)
        sa, tb = list(seen["sdpa"]), list(seen["tb"])
        seen["sdpa"].clear()
        b = attn(x, rope, None if x.shape[1] == 1 else "causal", ref, ref.indexer)
        sb = list(seen["sdpa"])
        mx.eval(a, b, sa, sb, tb)
        return a, b, sa, sb, tb

    # Prefill and one pass below the switch: stock's lines on both sides, bit for bit.
    sf._QSA_SWITCH[shape] = 2 ** 62
    for piece in (2048, 2000, 52):
        x = (mx.random.normal((1, piece, hidden)) * 0.5).astype(mx.bfloat16)
        a, b, _, _, tb = both(x)
        assert not tb and bool(mx.array_equal(a, b)), f"prefill piece {piece} differs"
    a, b, _, _, tb = both((mx.random.normal((1, 1, hidden)) * 0.5).astype(mx.bfloat16))
    assert not tb and bool(mx.array_equal(a, b)), "a decode pass below the switch is not stock's"

    # Decode on the picks from the budget on.
    sf._QSA_SWITCH[shape] = attn.indexer.token_budget + 1
    eq, n, worst_head, worst_own, over, over_abs, outs_eq = 0, 0, 0.0, 0.0, 0, 0.0, 0
    for i in range(24):
        x = (mx.random.normal((1, 1, hidden)) * 0.5).astype(mx.bfloat16)
        a, b, sa, sb, tb = both(x)
        assert len(tb) == 1 and len(sa) == len(sb) == 1, (len(tb), len(sa), len(sb))
        top = stock_top(attn.indexer, x, rope, ref.indexer.keys, ref.offset - 1)
        assert bool(mx.array_equal(tb[0].astype(mx.uint32), top.astype(mx.uint32))), f"top-k differs at step {i}"
        pa = np.array(sa[0].astype(mx.float32)).reshape(attn.n_heads, -1)
        pb = np.array(sb[0].astype(mx.float32)).reshape(attn.n_heads, -1)
        d = np.abs(pa - pb)
        ulp = lambda v: np.exp2(np.floor(np.log2(np.maximum(v, 1e-30))) - 7)
        worst_head = max(worst_head, float((d / ulp(np.abs(pb).max(axis=1, keepdims=True))).max()))
        own = d / ulp(np.maximum(np.abs(pa), np.abs(pb)))
        worst_own = max(worst_own, float(own.max()))
        over += int((own > 1).sum())
        if (own > 1).any():
            over_abs = max(over_abs, float(np.abs(pb)[own > 1].max()))
        eq += int((d == 0).sum())
        n += d.size
        outs_eq += bool(mx.array_equal(a, b))
    share = eq / n
    print(f"  decode: 24 passes on the picks, top-k equal to stock's on every one; SDPA output "
          f"{100 * share:.2f}% bit-equal, worst difference {worst_head:.2f} bf16 steps at its head's "
          f"scale ({worst_own:.1f} at its own, {over} elements over one, all |x| <= {over_abs:.2g}); "
          f"layer outputs bit-equal on {outs_eq} of 24")
    assert share >= 0.99, share
    assert worst_head <= 1.0, worst_head

    # A multi-row pass after the picks: stock's lines again (a 2-row verify pass).
    x = (mx.random.normal((1, 2, hidden)) * 0.5).astype(mx.bfloat16)
    a, b, _, _, tb = both(x)
    assert not tb, "a 2-row pass took the picks path"
    mx.fast.scaled_dot_product_attention, sf._qsa_gather = sdpa, gather
    print("test_qsa_cuda: ok")


if __name__ == "__main__":
    main()
