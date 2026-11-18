#!/usr/bin/env python3
"""MOE_PREFILL_LAST_KV (stream_fast.LAST_KV): in a pass moe_stream._prefill makes, whose outputs nothing reads, the last
layer runs only the half that writes its cache. On test_layer_prefill's toy packed Qwen3-Next, no weights:

1. A 300-id prompt prefilled in five passes: every cache array is bit-equal with the lever on and off, for a model
   whose last layer is full attention and for one whose last layer is a GatedDeltaNet.
2. With the lever on, the last layer's experts are neither read ahead nor read in those passes; off, they are.
3. A consumer pass (the model called outside _prefill and its logits read, as the generate step does) keeps the
   whole last layer with the lever on: its logits and the cache after it are bit-equal to the lever off, and the
   last layer's experts are read. The same pass forced cache-only gives different logits, so the check has teeth.
4. chunkattn hands a cache-only last layer's attention back unevaluated, without chunking it.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_prefill_last_kv.py
"""
import faulthandler
import gc
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

import chunkattn
import moe_stream
import stream_fast
import test_layer_prefill as tlp

faulthandler.dump_traceback_later(270, exit=True)

STEP = 64           # 300 ids in passes of 64: five passes, the last 44 wide
TAIL = list(range(7, 27))   # the consumer pass: 20 ids, wider than a decode pass so it is a prefill-shaped one


def bits_equal(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    v = mx.uint16 if a.dtype.size == 2 else mx.uint32
    return mx.array_equal(a.view(v), b.view(v)).item()


def run(out, lever):
    """Prefill the prompt through moe_stream's own _prefill, then one consumer pass; what each read and left."""
    stream_fast.LAST_KV = lever
    model = tlp.stock(out)
    stream_fast.install(model, out, resident=6, cache_bias=0.0)
    last = stream_fast._glu_of(model.layers[-1])
    reads, whole = [], stream_fast._whole_layer
    read_layer = stream_fast._read_layer

    def whole_layer(glu, ids, T):
        reads.append(glu)
        return whole(glu, ids, T)

    def read(glu, i, ids=None):
        reads.append(glu)
        return read_layer(glu, i, ids)
    stream_fast._whole_layer, stream_fast._read_layer = whole_layer, read
    try:
        llm = types.SimpleNamespace(model=model, _cache=make_prompt_cache(model), prefill_step=None, _convs=(),
                                    _pieces=(), _parked=0, _dropped=0)
        moe_stream.StreamingLLM._prefill(llm, tlp.PROMPT, STEP)
        assert llm._pieces == (64, 64, 64, 64, 44), llm._pieces
        pre = tlp.arrays(llm._cache)
        mx.eval(pre)
        pre_reads, reads[:] = sum(g is last for g in reads), []
        logits = model(mx.array(TAIL)[None], cache=llm._cache)
        mx.eval(logits, tlp.arrays(llm._cache))
        post_reads = sum(g is last for g in reads)
        got = pre, logits, tlp.arrays(llm._cache), pre_reads, post_reads
        # The consumer pass again on the prefilled cache, forced cache-only: a lever applied where outputs are read.
        cache = make_prompt_cache(model)
        moe_stream.StreamingLLM._prefill(types.SimpleNamespace(**{**vars(llm), "_cache": cache}), tlp.PROMPT, STEP)
        stream_fast.CACHE_ONLY = lever
        try:
            forced = model(mx.array(TAIL)[None], cache=cache)
            mx.eval(forced)
        finally:
            stream_fast.CACHE_ONLY = False
    finally:
        stream_fast._whole_layer, stream_fast._read_layer = whole, read_layer
    del model
    gc.collect()
    mx.clear_cache()
    return got + (forced,)


def check(out, what):
    off, on = run(out, False), run(out, True)
    assert len(off[0]) == len(on[0]) and all(bits_equal(a, b) for a, b in zip(off[0], on[0])), \
        f"{what}: cache arrays after the prefill differ"
    assert off[3] > 0 and on[3] == 0, f"{what}: last layer's expert reads in the prefill, off {off[3]} on {on[3]}"
    assert bits_equal(off[1], on[1]), f"{what}: the consumer pass's logits differ"
    assert all(bits_equal(a, b) for a, b in zip(off[2], on[2])), f"{what}: cache arrays after the consumer pass differ"
    assert on[4] > 0 and on[4] == off[4], f"{what}: the consumer pass read the last layer {on[4]} times, off {off[4]}"
    assert bits_equal(off[5], off[1]), f"{what}: lever off, a forced pass changed the logits"
    assert not bits_equal(on[5], on[1]), f"{what}: a cache-only consumer pass left its logits unchanged"
    print(f"   {what}: {len(on[0])} cache arrays bit-equal after 5 passes; last layer's experts read "
          f"{off[3]} times off, {on[3]} on; consumer pass bit-equal, last layer read {on[4]} times")


def check_chunkattn():
    """The wrapper returns stock's lazy call for a cache-only last layer and chunks it otherwise."""
    stock = mx.fast.scaled_dot_product_attention
    calls, chunked = [], chunkattn.chunked
    f = chunkattn._wrap(lambda *a, **k: stock(*a[:3], scale=k["scale"], mask=k["mask"]))
    layout, cap = chunkattn.layout, stream_fast.prefill_cap_bytes
    chunkattn.layout = {(4, 2, 32): (1.0, 8)}
    stream_fast.prefill_cap_bytes = lambda n, size: 16.0 * n       # 16 rows a chunk
    chunkattn.chunked = lambda *a, **k: calls.append(a[0].shape) or chunked(*a, **k)
    q, k = (mx.random.normal((1, h, 64, 32)).astype(mx.bfloat16) for h in (4, 2))
    try:
        f(q, k, k, None, 0.2, "causal")
        assert calls, "chunkattn did not chunk the toy shape"
        calls.clear()
        stream_fast._KV_ONLY = True
        lazy = f(q, k, k, None, 0.2, "causal")
        assert not calls, "a cache-only last layer's attention was chunked (evaluated)"
        assert bits_equal(lazy, stock(q, k, k, scale=0.2, mask="causal")), "the stand-in is not stock's call"
    finally:
        stream_fast._KV_ONLY = False
        chunkattn.layout, stream_fast.prefill_cap_bytes, chunkattn.chunked = layout, cap, chunked
    print("   chunkattn: a cache-only last layer's attention goes back as stock's lazy call, unchunked")


def main():
    lever, group = stream_fast.LAST_KV, stream_fast.WIDE_GROUP
    cfg = tlp.CFG
    stream_fast.WIDE_GROUP = 8      # the toy's 32 experts: a prefill layer takes the grouped path, as the 35B's 256 do
    try:
        with tempfile.TemporaryDirectory() as d:
            for interval, what in ((2, "last layer full attention"), (3, "last layer GatedDeltaNet")):
                tlp.CFG = {**cfg, "full_attention_interval": interval}
                out = Path(d) / f"q3n{interval}"
                tlp.build(out)
                check(out, what)
        check_chunkattn()
    finally:
        tlp.CFG, stream_fast.LAST_KV, stream_fast.WIDE_GROUP = cfg, lever, group
    print("test_prefill_last_kv: all passed")


if __name__ == "__main__":
    main()
