#!/usr/bin/env python3
"""stream_fast.install_qsa_pool: qwen4_exp's QSA indexer keeps its pooled block
keys between calls. Against stock's indexer cache, which re-pools every call:

  1. Indexer at Flash-Next's shapes (hidden 2560, 4 x 128 heads, ratio 4), a
     budget of 64 and of 2048: random sequences of prefill pieces, decode
     tokens, multi-row verify passes and their rollback, cache trims, a stopped
     prefill pass (stream_fast._prefill_stop), a shadow's cut copy, a front
     saved and read back (pcache), a `keys` write, two conversations switched.
     After every step the masks are bit-identical, the keys equal, and the kept
     pool equals a fresh re-pool of the keys (stock's lines) block for block.
  2. A tiny random qwen4_exp model (make_cache, prefill in pieces, greedy decode,
     a 3-row pass rolled back, a trim and refeed): logits bit-identical.

No checkpoint. Needs mlx_lm with qwen4_exp (skips otherwise), e.g.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_qsa_pool.py
"""
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn

try:
    from mlx_lm.models import qwen4_exp as q4
except ImportError:
    print("test_qsa_pool: skipped (mlx_lm without qwen4_exp)")
    sys.exit(0)

import pcache
import stream_fast
from pathlib import Path

# CUDA's install measures its attention switch: into a temporary home, not ~/.cache.
_HOME = tempfile.mkdtemp(prefix="qsa_pool_home_")
Path.home = classmethod(lambda cls: Path(_HOME))


def tiny_model(budget):
    args = q4.ModelArgs.from_dict({"model_type": "qwen4_exp", "text_config": dict(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=256, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, hc_count=4, hc_lowrank=16, indexer_n_heads=2,
        indexer_kv_heads=1, indexer_head_dim=32, indexer_budget=budget,
        indexer_compress_ratio=4, ple_layer_ids=[], rope_theta=10_000_000.0,
        layer_types=["linear_attention", "full_attention"] * 2)})
    m = q4.Model(args)
    m.set_dtype(mx.bfloat16)
    mx.eval(m.parameters())
    return m


def ref_pool(ix, rope, raw):
    """QSAIndexer.__call__'s pooling of every complete block (no padding)."""
    cr = ix.compress_ratio
    n = raw.shape[1] // cr
    p = raw[:, :n * cr].reshape(raw.shape[0], n, cr, ix.head_dim)
    p = ix.k_layernorm(p.astype(mx.float32).mean(axis=2).astype(raw.dtype))
    cos_k, sin_k = rope((mx.arange(n) * cr)[None, :])
    return q4._rope_partial(p, cos_k, sin_k)


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b))


class Twin:
    """One conversation's attention cache twice: the pooled indexer and stock's."""

    def __init__(self, pooled=None, stock=None):
        self.p = pooled if pooled is not None else q4._AttnCache()
        self.s = stock if stock is not None else self.stock_cache()

    @staticmethod
    def stock_cache():
        c = q4._AttnCache()
        c.indexer = stream_fast._QSA_STOCK[1]()
        return c

    def both(self):
        return (self.p, self.s)


def indexer_run(budget, steps, seed, max_piece, first=None):
    rng = random.Random(seed)
    mx.random.seed(seed)
    args = q4.TextArgs.from_dict(dict(hidden_size=2560, head_dim=256, partial_rotary_factor=0.25,
                                      indexer_n_heads=4, indexer_kv_heads=1,
                                      indexer_head_dim=128, indexer_budget=budget,
                                      indexer_compress_ratio=4))
    ix = q4.QSAIndexer(args)
    ix.set_dtype(mx.bfloat16)
    mx.eval(ix.parameters())
    rope = q4.RotaryEmbedding(int(args.head_dim * args.partial_rotary_factor), 10_000_000.0)
    convs = [Twin(), Twin()]
    cur = 0
    counts = {"calls": 0, "pool_hit": 0}
    fronts = tempfile.mkdtemp(prefix="qsa_pool_")

    def feed(t, S):
        """S positions through the indexer of both caches, then their KV."""
        x = (mx.random.normal((1, S, args.hidden_size)) * 0.5).astype(mx.bfloat16)
        had = t.p.indexer.pool
        masks = [ix(x, rope, c.indexer, c.offset) for c in t.both()]
        assert same(*masks), f"mask differs at offset {t.s.offset}, S {S}"
        counts["calls"] += 1
        if masks[0] is not None and had is not None and had[1].shape[1]:
            counts["pool_hit"] += 1
        kv = mx.zeros((1, 1, S, 4), dtype=mx.bfloat16)
        for c in t.both():
            c.update_and_fetch(kv, kv)

    def check(t, what):
        p, s = t.both()
        assert p.offset == s.offset, (what, p.offset, s.offset)
        assert same(p.indexer.keys, s.indexer.keys), f"{what}: keys differ"
        n = 0 if p.indexer.keys is None else p.indexer.keys.shape[1]
        assert n == p.offset, f"{what}: indexer holds {n} keys, KV {p.offset}"
        pool = p.indexer.pool
        # An empty pool has nothing to compare, and MLX 0.32.4's CUDA array_equal fails on empty arrays.
        if pool is not None and pool[1].shape[1]:
            m = pool[1].shape[1]
            assert m <= n // pool[0], f"{what}: pool of {m} blocks over {n} keys"
            # The fp32 pool (stream_fast._QSA_POOL32) holds the same bf16 values widened.
            ref = ref_pool(ix, rope, p.indexer.keys)[:, :m].astype(pool[1].dtype)
            assert same(pool[1], ref), f"{what}: kept pool differs from a re-pool"

    ops = ["prefill", "decode", "decode", "verify", "trim", "stop", "shadow", "front",
           "assign", "switch"]
    for i in range(steps):
        t = convs[cur]
        op = rng.choice(ops) if t.p.offset else "prefill"
        if op == "prefill":
            feed(t, first if first and not t.p.offset else rng.randint(2, max_piece))
        elif op == "decode":
            for _ in range(rng.randint(1, 12)):
                feed(t, 1)
                check(t, op)
        elif op == "verify":
            T = rng.randint(2, 6)
            feed(t, T)
            check(t, "verify pass")
            r = rng.randint(0, T)               # rows rejected
            for c in t.both():
                assert c.trim(r) == r
        elif op == "trim":
            n = rng.randint(1, min(t.p.offset, max_piece))
            for c in t.both():
                assert c.trim(n) == n
        elif op == "stop":
            # A prefill pass stopped inside the layer: the KV offset goes back
            # to the layer's start, and the indexer keys with it.
            s0 = t.p.offset
            feed(t, rng.randint(2, max_piece))
            stream_fast.PREFILL_STOP = lambda: True
            stream_fast._PREFILL_AT = (1, None, [(c, s0) for c in t.both()])
            try:
                stream_fast._prefill_stop()
                raise AssertionError("the stop did not raise")
            except stream_fast.PrefillStopped:
                pass
            finally:
                stream_fast.PREFILL_STOP = None
        elif op == "shadow":
            # moe_stream.shadow's copy of a kept cache cut at r.
            r = rng.randint(1, t.p.offset)
            new = Twin()
            for src, dst in ((t.p, new.p), (t.s, new.s)):
                kv = (src.keys[..., :r, :], src.values[..., :r, :])
                # Newer mlx_lm's KVCache.state carries the offset as a third part.
                kv += (r,) if len(src.state) == 4 else ()
                dst.state = kv + (src.indexer.keys[:, :r],)
            assert new.p.indexer.pool is None
            convs[cur] = t = new
        elif op == "front":
            # Saved and read back as moe_stream's stored fronts are.
            path = os.path.join(fronts, f"f{i}.safetensors")
            pcache.save_prompt_cache(path, [t.p], {})
            back = pcache.load_prompt_cache(path)[0]
            assert type(back.indexer) is q4._IndexerCache and back.indexer.pool is None
            stock = Twin.stock_cache()
            stock.state = t.s.state
            convs[cur] = t = Twin(back, stock)
        elif op == "assign":
            had = t.p.indexer.keys
            t.p.indexer.keys = had          # a write that is not update/trim
            assert t.p.indexer.pool is None
        elif op == "switch":
            cur = 1 - cur
            t = convs[cur]
            if not t.p.offset:
                feed(t, rng.randint(2, max_piece))
        check(t, op)
        # the next call must re-pool to exactly what stock pools
        if t.p.offset:
            feed(t, 1)
            check(t, op + " + decode")
    return counts


def model_run(budget, seed):
    """Greedy decode of a tiny random model, pooled caches against stock's."""
    # CUDA's decode over the picks is not bit-equal (tests/test_qsa_cuda.py), so this indexer test keeps stock's attention.
    for shape in stream_fast._QSA_SWITCH:
        stream_fast._QSA_SWITCH[shape] = 2 ** 62
    mx.random.seed(seed)
    model = tiny_model(budget)
    pooled = model.make_cache()
    stock = model.make_cache()
    for c in stock:
        if hasattr(c, "indexer"):
            c.indexer = stream_fast._QSA_STOCK[1]()
    assert any(type(getattr(c, "indexer", None)) is q4._IndexerCache for c in pooled)

    def step(ids):
        a = model(mx.array([ids]), cache=pooled)
        b = model(mx.array([ids]), cache=stock)
        assert same(a, b), f"logits differ at offset {pooled[1].offset}"
        return int(mx.argmax(a[0, -1]).item())

    prompt = [int(t) for t in mx.random.randint(0, 256, (150,)).tolist()]
    for a, b in ((0, 37), (37, 101), (101, 150)):
        tok = step(prompt[a:b])
    out = [tok]
    for i in range(40):
        if i == 15:
            # a 3-row pass, two rows rolled back
            step([out[-1], 7, 9])
            for c in pooled + stock:
                if hasattr(c, "trim") and c.is_trimmable():
                    c.trim(2)
            continue
        if i == 25:
            for c in pooled + stock:
                if hasattr(c, "indexer"):
                    c.trim(20)
            out.append(step(prompt[:1] * 20))
            continue
        out.append(step([out[-1]]))
    return out


def main():
    n = stream_fast.install_qsa_pool(tiny_model(32))
    assert n == 2, f"install covered {n} indexers, want 2 (MOE_QSA_POOL=0?)"
    assert stream_fast.install_qsa_pool(tiny_model(32)) == 2     # idempotent
    default = stream_fast._QSA_POOL32
    # The fp32 pool is CUDA's default and plain MLX ops, so both settings run on any backend.
    for pool32 in (default, not default):
        stream_fast._QSA_POOL32 = pool32
        for budget, steps, seed, piece, first in ((64, 300, 0, 120, None), (64, 300, 1, 40, None),
                                                  (2048, 60, 2, 300, 2300)):
            c = indexer_run(budget, steps, seed, piece, first)
            assert c["pool_hit"] > c["calls"] // 4, c
            print(f"  indexer, budget {budget}, seed {seed}, fp32 pool {pool32}: {c['calls']} "
                  f"calls, {c['pool_hit']} on a kept pool, masks/keys/pool equal to stock's")
        out = model_run(32, 3)
        print(f"  tiny model, fp32 pool {pool32}: {len(out)} greedy tokens, logits bit-identical")
    stream_fast._QSA_POOL32 = default
    print("test_qsa_pool: OK")


if __name__ == "__main__":
    main()
