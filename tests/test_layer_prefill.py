#!/usr/bin/env python3
"""Layer-major prefill (stream_fast._lm_layer_call) under a budget store, on a
toy packed MoE, no real weights, a few MB on disk:

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_layer_prefill.py

1. One layer-major pass over a 300-position prompt (LM_W 64: five chunks,
   the last 44 wide) == stock mlx_lm fed the same 64-position pieces:
   logits and every cache array bit-equal. A two-shape Qwen3-Next (4-bit and
   2-bit layers: two stores, a reader each), MOE_LM_PREFILL=1; cache bias 0
   and 0.5 (the store is released before the pass routes, so the bias ranks
   nothing); and with each reader held to two regions (one group ahead).
2. Every expert of every layer is read at most once a pass, and exactly once
   when every layer is read whole (the two-region arm, MOE_READPATH=whole).
3. MOE_LM_PREFILL unset: this toy (not Flash-Next) stays on today's path.
4. The budget plans a layer-major prefill from a line: per position what
   its widest pass held at once (read in the pass), or the slope of two
   passes two chunks wide if more; the fixed part over every pass measured,
   a chunk's transient scaled up from the widest narrower pass until one is measured.
5. StreamingPLE: rows read ahead for a pass (deduplicated, on a thread) and
   rows read on demand (deduplicated) == one read per row, bit for bit.
6. chunkattn on a boolean (sparse) mask: chunks == one stock call.
7. A layer-major pass the budget has to cut runs in whole LM_W chunks a
   piece, and stays bit-equal to stock's 64-position pieces.
8. Pieces off that grid (non-fresh passes included) == stock fed the same
   chunks: a split moves the arithmetic only as mlx_lm's chunks would.
"""
import faulthandler
import gc
import json
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_next as q3n
from mlx_lm.models.cache import ArraysCache, make_prompt_cache

import stream_fast

faulthandler.dump_traceback_later(270, exit=True)

# test_store.py's toy: 32 experts, top 4, 2-bit middle layers (two shapes).
CFG = dict(
    model_type="qwen3_next", hidden_size=64, num_hidden_layers=4,
    intermediate_size=64, num_attention_heads=2, linear_num_value_heads=2,
    linear_num_key_heads=1, linear_key_head_dim=32, linear_value_head_dim=32,
    linear_conv_kernel_dim=4, num_experts=32, num_experts_per_tok=4,
    decoder_sparse_step=1, shared_expert_intermediate_size=64,
    mlp_only_layers=[], moe_intermediate_size=64, rms_norm_eps=1e-6,
    vocab_size=200, num_key_value_heads=1, rope_theta=1e7,
    partial_rotary_factor=0.25, max_position_embeddings=4096, head_dim=32,
    norm_topk_prob=True, full_attention_interval=2, tie_word_embeddings=False,
)
TWO_BIT = (1, 2)
W = 64
PROMPT = np.random.default_rng(5).integers(0, 200, 300).tolist()


def fresh_model():
    m = q3n.Model(q3n.ModelArgs.from_dict(CFG))
    base = m.quant_predicate

    def pred(path, mod):
        if not hasattr(mod, "to_quantized"):
            return False
        r = base(path, mod)
        if r is True and ".switch_mlp." in path and any(f"layers.{i}." in path for i in TWO_BIT):
            return {"group_size": 32, "bits": 2, "mode": "affine"}
        return r
    nn.quantize(m, group_size=32, bits=4, class_predicate=pred)
    return m


def build(out):
    mx.random.seed(0)
    model = fresh_model()
    mx.eval(model.parameters())
    weights = dict(tree_flatten(model.parameters()))
    out.mkdir(parents=True)
    mx.save_safetensors(str(out / "model.safetensors"), weights)
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {k: "model.safetensors" for k in weights}}))
    (out / "config.json").write_text(json.dumps(
        {**CFG, "quantization": {"group_size": 32, "bits": 4, "mode": "affine"}}))
    import repack
    argv, sys.argv = sys.argv, ["repack.py", "--src", str(out)]
    try:
        repack.main()
    finally:
        sys.argv = argv


def stock(out):
    model = fresh_model()
    model.load_weights(str(out / "model.safetensors"))
    mx.eval(model.parameters())
    return model


def engine(out, lm_env, cache_bias=0.0):
    stream_fast._LM_ENV = lm_env
    model = stock(out)
    stream_fast.install(model, out, resident=6, cache_bias=cache_bias)
    bud = stream_fast.enable_budget(model, mx.get_active_memory() / 1e9 + 1.0)
    return model, bud


def arrays(cache):
    got = []
    for c in cache:
        if isinstance(c, ArraysCache):
            got += [a for a in c.cache if a is not None]
        else:
            got += [c.keys[..., :c.offset, :], c.values[..., :c.offset, :]]
    return got


def reference(out):
    model = stock(out)
    cache = make_prompt_cache(model)
    logits = mx.concatenate([model(mx.array(PROMPT[a:a + W])[None], cache=cache)
                             for a in range(0, len(PROMPT), W)], axis=1)
    mx.eval(logits, arrays(cache))
    return logits, arrays(cache)


def run(model):
    cache = make_prompt_cache(model)
    logits = model(mx.array(PROMPT)[None], cache=cache)
    mx.eval(logits, arrays(cache))
    return logits, arrays(cache)


def same(a, b):
    return all(x.shape == y.shape and mx.array_equal(x, y).item() for x, y in zip(a, b)) \
        and len(a) == len(b)


def check_exact(out, ref):
    stream_fast.LM_W = W
    region_bytes = None
    for cache_bias, two in ((0.0, False), (0.5, False), (0.0, True)):
        real = stream_fast._LMReader
        starts = []
        if two:
            def held(store, glus, *rest):
                # The reader sees a store of two regions; the gathers read the
                # real one (_LM_READ is keyed by it).
                view = types.SimpleNamespace(per_half=store.per_half, iov=store.iov,
                                             views=store.views, slots=2 * store.per_half)
                return real(view, glus, *rest)
            stream_fast._LMReader = held
        start = stream_fast._lm_start

        def counted(*a):
            starts.append(1)
            return start(*a)
        stream_fast._lm_start = counted
        # Two regions: every layer read whole ahead (MOE_READPATH=whole);
        # otherwise the measured choice (test_union_lm covers the union reads).
        stream_fast._RP_MODE = "whole" if two else "auto"
        b0 = stream_fast.STATS["lm_read_b"]
        try:
            model, bud = engine(out, "1", cache_bias)
            assert bud.layer_major
            logits, cache = run(model)
        finally:
            stream_fast._LMReader = real
            stream_fast._lm_start = start
            stream_fast._RP_MODE = "auto"
        read = stream_fast.STATS["lm_read_b"] - b0
        assert len(starts) >= 1 and bud.pieces == [len(PROMPT)], (starts, bud.pieces)
        assert mx.array_equal(logits, ref[0]).item(), f"logits differ (bias {cache_bias}, two {two})"
        assert same(cache, ref[1]), f"caches differ (bias {cache_bias}, two {two})"
        glus = stream_fast.glus_of(model)
        region_bytes = sum(sum(p._packed["stride"] for p in g._projs)
                           * g._projs[0]._packed["n_experts"] for g in glus)
        assert read == len(starts) * region_bytes if two else read <= len(starts) * region_bytes, \
            (read, len(starts), region_bytes)
        print(f"   bias {cache_bias}{', two regions, whole' if two else ''}: {len(starts)} "
              f"layer-major pass(es) (probe included), logits and {len(cache)} cache "
              f"arrays == stock's 64-position pieces; {read / 1e6:.2f} MB read "
              f"(every expert once a pass: {len(starts) * region_bytes / 1e6:.2f} MB)")
        del model, bud
        gc.collect()
        mx.clear_cache()
    print("1-2. one layer-major pass == stock fed 64-position pieces, bit for bit; "
          "each expert read at most once a pass")


def check_cut_on_grid(out, ref):
    """A layer-major pass the budget cuts: whole LM_W chunks a piece, so the
    pieces keep the chunk grid, and stock's 64-position arithmetic."""
    stream_fast.LM_W = W
    model, bud = engine(out, "1")
    run(model)                                  # rate and state measured
    assert bud._lm_rows and min(bud._lm_rows.values()) > 0, bud._lm_rows
    held, host, _ = bud._measure()
    room = bud.peak - bud.cache - host - held - bud.ring_bytes
    r = (room - bud.state) / 150                # room for ~150 positions a pass
    pinned = {100: 100 * r, 200: 200 * r}
    real = bud.close

    def close():
        real()
        bud._lm_seen, bud._lm_rows = dict(pinned), {200: r}
    bud.close, bud._lm_seen, bud._lm_rows = close, dict(pinned), {200: r}
    logits, cache = run(model)
    p = bud.pieces
    assert len(p) > 1 and all(w % W == 0 for w in p[:-1]) and sum(p) == len(PROMPT), p
    assert mx.array_equal(logits, ref[0]).item() and same(cache, ref[1])
    del model, bud
    gc.collect()
    print(f"7. a layer-major prefill cut by the budget ran as {p}: whole chunks, "
          f"== stock's 64-position pieces bit for bit")


def check_off_grid(out):
    """Pieces off the chunk grid (a budget under one chunk a pass): each
    piece is its own chunks, == stock fed those chunks: the split changes
    the arithmetic exactly as mlx_lm's chunking would."""
    stream_fast.LM_W = W
    model, bud = engine(out, "1")
    bud.plan = lambda n, fresh, caches: min(n, 100)
    logits, cache = run(model)
    assert bud.pieces == [100, 100, 100], bud.pieces
    smodel = stock(out)
    scache = make_prompt_cache(smodel)
    sl = mx.concatenate([smodel(mx.array(PROMPT[a:b])[None], cache=scache)
                         for p0 in range(0, 300, 100)
                         for a, b in ((p0, p0 + 64), (p0 + 64, p0 + 100))], axis=1)
    mx.eval(sl)
    assert mx.array_equal(logits, sl).item() and same(cache, arrays(scache))
    del model, bud, smodel
    gc.collect()
    print("8. pieces of 100 (off the 64 grid): == stock fed each piece's chunks [64, 36]")


def check_default_off(out):
    model, bud = engine(out, None)
    assert not bud.layer_major
    del model, bud
    gc.collect()
    print("3. MOE_LM_PREFILL unset: Qwen3-Next stays on today's prefill")


def check_line(out):
    stream_fast.LM_W = W
    model, bud = engine(out, "1")
    # Per position: what the widest pass held at once. A one-chunk pass's
    # steeper slope (50 -> 200 here, its chunk grows with it) is not fitted;
    # the fixed part covers every pass.
    bud._lm_rows = {400: 1e3}
    bud._lm_seen = {50: 0.5e6, 200: 1e6 + 200 * 1e3, 300: 1e6 + 300 * 1e3, 400: 1e6 + 400 * 1e3}
    assert bud._lm_line() == (1e6, 1e3)
    bud._lm_seen[50] = 2e6                      # one above the line raises its fixed part
    assert bud._lm_line() == (2e6 - 50 * 1e3, 1e3)
    # Two passes two chunks wide (>= 128) and 25% apart that grow faster raise the slope.
    bud._lm_seen = {200: 1e6 + 200 * 1e3, 400: 1e6 + 400 * 2e3}
    assert bud._lm_line() == (0.6e6, 3e3)
    # Nothing a whole chunk wide measured yet (the probe alone): its growth
    # beyond its rows scaled up to a whole chunk.
    bud._lm_rows, bud._lm_seen = {16: 2e3}, {16: 1e5}
    assert bud._lm_line() == ((1e5 - 16 * 2e3) * W / 16, 2e3)
    # The widest such pass is scaled, not a narrower one's fixed bytes (8 -> W is 8x).
    bud._lm_rows, bud._lm_seen = {8: 3e3, 32: 1e3}, {8: 1e5, 32: 1.2e5}
    assert bud._lm_line() == ((1.2e5 - 32 * 1e3) * W / 32, 1e3)
    bud.rate, bud.state, bud.lm_ok = 1.0, 10.0, True
    bud._lm_rows, bud._lm_seen = {400: 1e3}, {100: 1e6 + 100 * 1e3, 400: 1e6 + 400 * 1e3}
    assert bud._need(1000, True, True) == 1e6 + 1000 * 1e3 + 10.0
    # Below one chunk: bounded by a width measured at least as wide, never by a narrower one.
    bud._lm_rows, bud._lm_seen = {50: 1e3}, {10: 20e3, 50: 1e6 + 50 * 1e3}
    assert bud._need(40, True, True) == 1e6 + 40 * 1e3 + 10.0
    assert bud._need(5, True, True) == 10e3 + 5 * 1e3 + 10.0
    del model, bud
    gc.collect()
    print("4. a layer-major prefill's need: rows a position held, the fixed part over every pass;"
          " below a chunk, the narrowest width measured at least as wide")


def check_ple(tmp):
    rows, shards, dim = 1000, 4, 64
    w = mx.random.normal((rows * shards, dim))
    wq, sc, bi = mx.quantize(w, group_size=32, bits=4)
    sc, bi = sc.astype(mx.float16), bi.astype(mx.float16)
    mx.eval(wq, sc, bi)
    parts = [np.array(a, copy=False).view(np.uint8).reshape(rows * shards, -1) for a in (wq, sc, bi)]
    blob = np.concatenate(parts, axis=1)
    stride = blob.shape[1]
    path = tmp / "ple.bin"
    path.write_bytes(blob.tobytes())
    meta = [{"name": "weight", "bytes": parts[0].shape[1], "shape": [wq.shape[1]], "dtype": "U32"},
            {"name": "scales", "bytes": parts[1].shape[1], "shape": [sc.shape[1]], "dtype": "F16"},
            {"name": "biases", "bytes": parts[2].shape[1], "shape": [bi.shape[1]], "dtype": "F16"}]
    entries = {k: {"offset": k * rows * stride, "stride": stride, "parts": meta}
               for k in range(shards)}
    fd = os.open(str(path), os.O_RDONLY)
    ple = stream_fast.StreamingPLE(fd, entries, 32, 4, "affine", dim, rows)
    ref_rows = mx.dequantize(wq, scales=sc, biases=bi, group_size=32, bits=4).astype(mx.float32)
    rng = np.random.default_rng(0)
    gid = mx.array(rng.integers(0, rows * shards, (1, 50, 16)) % 300, dtype=mx.int64)  # repeats
    want = ref_rows[gid.reshape(-1)].reshape(1, 50, 16, dim)
    on_demand = ple(gid)
    ple.read_ahead(gid)
    ahead = ple(gid[:, 10:30])                  # a chunk of the pass, from the rows read ahead
    ple.ahead = None
    miss = ple(gid + 1)                         # nothing read ahead: on demand again
    want_miss = ref_rows[(gid + 1).reshape(-1)].reshape(1, 50, 16, dim)
    for got, ref in ((on_demand, want), (ahead, want[:, 10:30]), (miss, want_miss)):
        assert mx.array_equal(got, ref).item()
    os.close(fd)
    # A read ahead that fails must raise when its rows are taken, not serve
    # the ones the thread never wrote: this file holds one shard, and the ids
    # below land in the second.
    short = tmp / "ple_short.bin"
    short.write_bytes(b"\0" * (rows * stride))
    fd = os.open(str(short), os.O_RDONLY)
    ple = stream_fast.StreamingPLE(fd, entries, 32, 4, "affine", dim, rows)
    gid = mx.array(rng.integers(rows, rows * shards, (1, 4, 2)), dtype=mx.int64)
    ple.read_ahead(gid)
    try:
        ple(gid)
    except AssertionError as e:
        assert "short read" in str(e), e
    else:
        raise AssertionError("a read ahead that failed served its rows anyway")
    os.close(fd)
    print(f"5. PLE rows: on demand, read ahead ({len(np.unique(np.asarray(gid)))} distinct of "
          f"{gid.size}) and after a miss == the table's rows, bit for bit")


def check_sparse_attention():
    """chunkattn on a boolean mask (Flash-Next's sparse attention past 2048
    keys): query chunks of the calibrated width, each with its rows of the
    mask and its keys cut as the causal ones are, == one stock call."""
    import chunkattn
    hq, hkv, d = 24, 2, 256                     # Flash-Next's attention shape
    found = chunkattn.calibrate({(hq, hkv, d)})
    if (hq, hkv, d) not in found:
        print("6. sparse-mask chunks: skipped, stock fuses this shape here")
        return
    _, rows, floor = found[(hq, hkv, d)]
    T, N = 600, 5000
    rng = np.random.default_rng(2)
    q, k, v = (mx.array(rng.standard_normal(s).astype(np.float32)).astype(mx.bfloat16)
               for s in ((1, hq, T, d), (1, hkv, N, d), (1, hkv, N, d)))
    causal = np.arange(N - T, N)[:, None] >= np.arange(N)[None, :]
    keep = rng.random((T, N)) < 0.3
    keep[np.arange(T), np.arange(N - T, N)] = True          # a row always sees itself
    mask = mx.array((causal & keep)[None, None])
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=d ** -0.5, mask=mask)
    for w in (rows, 2 * rows, 100):
        got = chunkattn.chunked(q, k, v, d ** -0.5, w, rows, floor, mask=mask)
        assert mx.array_equal(got, ref).item(), w
    print(f"6. sparse-mask attention in chunks of {rows}/{2 * rows}/100 rows (keys cut "
          f"past {floor or 'none'}) == one stock call")


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        out = tmp / "q3n"
        build(out)
        ref = reference(out)
        check_exact(out, ref)
        check_cut_on_grid(out, ref)
        check_off_grid(out)
        check_default_off(out)
        check_line(out)
        check_ple(tmp)
    # Last: its calibration's big transients leave the process's footprint
    # high, which the toy budgets above would read as host memory.
    check_sparse_attention()
    print("test_layer_prefill: all passed")


if __name__ == "__main__":
    main()
