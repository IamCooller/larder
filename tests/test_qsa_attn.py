#!/usr/bin/env python3
"""stream_fast._qsa_sdpa and the compact decode attention (MOE_QSA_ATTN): MLX's
masked two-pass SDPA over QSA's picks, bit for bit, reading only the picks.

  1. Kernel at Flash-Next's shapes (24 query heads, 2 KV heads x 256, bf16) and a
     dense-GQA one (8 x 128 over 2): random K/V in a cache buffer wider than the
     context (a KVCache view), 512 sorted blocks of 4 (random, clustered, the
     newest, the oldest) plus the 0-3 tail positions, contexts 4096-70000 with
     every tail length: output equal to mx.fast.scaled_dot_product_attention
     with the indexer's keep mask, array_equal.
  2. A tiny random qwen4_exp model (budget 64, 8 x 64 over 2 heads) past 4096
     positions: prefill, greedy decode, a 2-row pass, a trim and refeed with the
     compact attention and without: logits bit-identical, and the compact path
     taken on every one-row pass (counted).
  3. MOE_QSA_STEP's buffer under the same model: logits bit-identical, keys equal.

Toy shapes, no checkpoint; needs MLX 0.32.2 (the kernel's validated version).

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_qsa_attn.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx

try:
    from mlx_lm.models import qwen4_exp as q4
except ImportError:
    print("test_qsa_attn: skipped (mlx_lm without qwen4_exp)")
    sys.exit(0)

import stream_fast as sf

if sf.BACKEND != "metal" or mx.__version__ not in sf._SDPA_MLX:
    print(f"test_qsa_attn: skipped (backend {sf.BACKEND}, MLX {mx.__version__})")
    sys.exit(0)


def keep_mask(tb, n, cr=4):
    """The indexer's keep mask of one decode row: the blocks tb and the tail."""
    nb = n // cr
    kb = mx.put_along_axis(mx.zeros((nb + 1,), mx.bool_), tb, mx.array(True), axis=0)[:nb]
    keep = mx.repeat(kb, cr)
    tail = mx.ones((n - nb * cr,), mx.bool_)
    return mx.concatenate([keep, tail]).reshape(1, 1, 1, n)


def kernel_case(H, Hk, D, n, kind, seed):
    mx.random.seed(seed)
    cap = (n + 255) // 256 * 256 + 256
    kb = (mx.random.normal((1, Hk, cap, D)) * 2).astype(mx.bfloat16)
    vb = mx.random.normal((1, Hk, cap, D)).astype(mx.bfloat16)
    k, v = kb[..., :n, :], vb[..., :n, :]
    q = (mx.random.normal((1, 1, H, D)) * 2).astype(mx.bfloat16).transpose(0, 2, 1, 3)
    nb = n // 4
    if kind == "random":
        tb = mx.random.permutation(nb)[:512]
    elif kind == "cluster":
        s = int(mx.random.randint(0, nb - 600, (1,)).item())
        tb = mx.arange(s, s + 512)
    elif kind == "newest":
        tb = mx.arange(nb - 512, nb)
    else:
        tb = mx.arange(512)
    tb = mx.sort(tb.astype(mx.int32))
    scale = D ** -0.5
    parts = sf._sdpa_parts(H, Hk, n)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=keep_mask(tb, n))
    got = sf._qsa_sdpa(q, k, v, tb, nb * 4, n, scale, 4, parts)
    mx.eval(ref, got)
    ok = ref.shape == got.shape and bool(mx.array_equal(ref, got))
    if not ok:
        d = float(mx.max(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))))
        raise AssertionError(f"H {H} Hk {Hk} D {D} n {n} {kind}: max diff {d}")


def test_kernel():
    cases = 0
    for H, Hk, D in ((24, 2, 256), (8, 2, 128)):
        for n in (4096, 4097, 4098, 4099, 8191, 16384 + 2, 32768 + 3, 65536 + 1, 70001):
            for kind in ("random", "cluster", "newest", "oldest"):
                kernel_case(H, Hk, D, n, kind, cases)
                cases += 1
    print(f"  kernel: {cases} cases bit-identical to the masked SDPA")


def tiny_model():
    args = q4.ModelArgs.from_dict({"model_type": "qwen4_exp", "text_config": dict(
        hidden_size=128, num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
        head_dim=64, vocab_size=256, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, hc_count=4, hc_lowrank=16, indexer_n_heads=2,
        indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=64,
        indexer_compress_ratio=4, ple_layer_ids=[], rope_theta=10_000_000.0,
        layer_types=["linear_attention", "full_attention"] * 2)})
    m = q4.Model(args)
    m.set_dtype(mx.bfloat16)
    mx.eval(m.parameters())
    return m


def run(model, ids, compact, step):
    """Logits of a fixed script: a 4100-position prefill in pieces, 12 greedy
    tokens, a 2-row pass, a trim of 3 and 6 more tokens."""
    sf._QSA_STEP = step
    q4.Attention.__call__ = ATTN[compact]
    cache = model.make_cache()
    outs = []
    pos = 0
    for piece in (2048, 2000, 52):
        lg = model(ids[None, pos:pos + piece], cache=cache)
        mx.eval(lg)
        pos += piece
    outs.append(lg[:, -1])
    tok = mx.argmax(lg[:, -1], axis=-1)
    for i in range(12):
        lg = model(tok[None], cache=cache)
        outs.append(lg[:, -1])
        tok = mx.argmax(lg[:, -1], axis=-1)
        mx.eval(tok)
    lg = model(mx.concatenate([tok, tok])[None], cache=cache)
    outs.append(lg)
    for c in cache:
        if hasattr(c, "trim"):
            c.trim(3)
    for i in range(6):
        lg = model(tok[None], cache=cache)
        outs.append(lg[:, -1])
        tok = mx.argmax(lg[:, -1], axis=-1)
        mx.eval(tok)
    keys = [c.indexer.keys for c in cache if hasattr(c, "indexer")]
    mx.eval(outs, keys)
    return outs, keys


CALLS = {"n": 0}
ATTN = {}


def test_model():
    stock_attn = q4.Attention.__call__
    m = tiny_model()
    sf._QSA_ATTN = True
    assert sf.install_qsa_pool(m) == 2 and sf._QSA_ATTN_ON
    compact = q4.Attention.__call__

    def counted(self, x, rope, mask, cache, idx_cache):
        if x.shape[1] == 1:
            CALLS["n"] += 1
        return compact(self, x, rope, mask, cache, idx_cache)
    real = sf._qsa_sdpa
    used = {"n": 0}

    def spy(*a, **k):
        used["n"] += 1
        return real(*a, **k)
    sf._qsa_sdpa = spy
    ATTN[False], ATTN[True] = stock_attn, counted
    mx.random.seed(7)
    ids = mx.random.randint(0, 256, (4100,))
    base, kb = run(m, ids, False, False)
    for compact_on, step in ((True, False), (False, True), (True, True)):
        CALLS["n"] = used["n"] = 0
        got, kg = run(m, ids, compact_on, step)
        for i, (a, b) in enumerate(zip(base, got)):
            assert a.shape == b.shape and bool(mx.array_equal(a, b)), \
                f"compact {compact_on} step {step}: logits differ at step {i}"
        for a, b in zip(kb, kg):
            assert bool(mx.array_equal(a, b)), f"step {step}: indexer keys differ"
        if compact_on:
            # 2 attention layers x the 18 one-row passes past 4096 positions
            assert used["n"] == CALLS["n"] == 36, (used["n"], CALLS["n"])
    sf._qsa_sdpa = real
    q4.Attention.__call__ = compact
    print("  model: logits bit-identical (compact attention, step buffer, both), "
          "compact path on every one-row pass")


if __name__ == "__main__":
    test_kernel()
    test_model()
    print("test_qsa_attn: ok")
