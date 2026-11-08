#!/usr/bin/env python3
"""Prefill read path (stream_fast._whole_layer): a layer of a plain-path
prefill finds its whole expert region read ahead, or reads only the rows it
gathers once it has routed. Both fill the same set at the experts' own
index and _from_whole cuts the same buffers from it, so on a toy packed MoE
(test_layer_prefill's) the logits of every pass, the cache arrays and every
layer's arena (LRU order and slot table) are bit-equal whichever path the
layers take, and whether the arena's rows are copied into the set or read
from experts.bin: the reference is main's path (whole layers, every row
read), against forced whole, forced union and the measured choice, all with
the copies. One cache: a prefill, decode tokens that fill the arena, then
wider and narrower passes; with groups wider than the union (the split
path: residents from the arena, misses through the set) and narrower
(several groups a layer). Every arm asks the SSD for fewer bytes.

With a sidecar (mirror_experts.py) of every expert, then of half, the
layer reads send the chunks it holds whole to it: the same logits, caches
and arenas, the sidecar read part of the bytes, and a chunk goes there only
if the sidecar holds every record of it (checked against every chunk).

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_readpath.py
"""
import faulthandler
import gc
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

import stream_fast
from test_layer_prefill import arrays, build, same, stock

faulthandler.dump_traceback_later(270, exit=True)

PASSES = [40, 1, 1, 1, 16, 1, 1, 24, 1, 40, 16]     # widths fed in turn on one cache


def run(out, mode, group, copy=True):
    stream_fast._RP_MODE, stream_fast.WIDE_GROUP = mode, group
    arena_of = stream_fast._arena_of
    if not copy:                        # every row from experts.bin: main's path
        stream_fast._arena_of = lambda glu: None
    model = stock(out)
    stream_fast.install(model, out, resident=6, cache_bias=0.0)
    if stream_fast._STRIPE is not None:
        stream_fast._STRIPE.bw_install = (1.0, 1.0)     # both drives get chunks
        check_ext_off(stream_fast._STRIPE, model)
    cache, rng, ys = make_prompt_cache(model), np.random.default_rng(3), []
    s0 = dict(stream_fast.STATS)
    try:
        for n in PASSES:
            ys.append(model(mx.array(rng.integers(0, 200, n))[None], cache=cache))
            mx.eval(ys[-1])
    finally:
        stream_fast._arena_of = arena_of
    arena = [(list(g._lru.items()), g._slot_np.copy()) for g in stream_fast.glus_of(model)]
    st = {k: stream_fast.STATS[k] - s0[k]
          for k in ("rp_union", "rp_whole", "rp_b", "rp_copied", "wide_split", "rp_ext_b")}
    got = ys, arrays(cache), arena, st
    del model, cache
    gc.collect()
    mx.clear_cache()
    return got


def check_ext_off(s, model):
    """ext_off against brute force: every run of 1..PREFILL_CHUNK experts of
    every tensor goes to the sidecar exactly when it holds every record."""
    hits = 0
    for glu in stream_fast.glus_of(model):
        for g in glu._projs:
            pk = g._packed
            for e0 in range(pk["n_experts"]):
                for n in range(1, min(stream_fast.PREFILL_CHUNK, pk["n_experts"] - e0) + 1):
                    off = pk["offset"] + e0 * pk["stride"]
                    recs = [s.at.get(off + i * pk["stride"]) for i in range(n)]
                    want = recs[0][0] if all(recs) else None
                    assert s.ext_off((pk["fd"], None, off, n * pk["stride"])) == want, (e0, n)
                    hits += want is not None
    assert hits


def check_stripe(d, out, refs):
    import mirror_experts
    for frac in (1.0, 0.5):
        drive = Path(d) / f"drive{frac}"
        mirror_experts.main(["--model", str(out), "--drive", str(drive), "--seconds", "0.05",
                             "--fraction", str(frac), "--min-margin-gb=-1e9",
                             "--max-swap-mb", "1e9"])
        os.environ["MOE_STRIPE"] = str(drive / out.name / mirror_experts.META)
        try:
            for group, mode in ((32, "whole"), (32, "auto"), (8, "union")):
                ref = refs[group]
                ys, cache, arena, st = run(out, mode, group)
                what = f"sidecar {frac}, {mode}, group {group}"
                assert same(ys, ref[0]), f"logits differ ({what})"
                assert same(cache, ref[1]), f"cache arrays differ ({what})"
                for (lru, slots), (lru0, slots0) in zip(arena, ref[2]):
                    assert lru == lru0 and np.array_equal(slots, slots0), f"arena differs ({what})"
                if frac == 1.0:
                    assert 0 < st["rp_ext_b"] < st["rp_b"], st
                print(f"   {what}: == main's path; {st['rp_ext_b'] / 1e6:.2f} of "
                      f"{st['rp_b'] / 1e6:.2f} MB from the sidecar")
        finally:
            os.environ.pop("MOE_STRIPE")
            stream_fast.open_stripe(out, None)      # closes it


def main():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "q3n"
        build(out)
        refs = {}
        for group in (32, 8):
            ref = refs[group] = run(out, "whole", group, copy=False)
            assert ref[3]["rp_whole"] > 0 and ref[3]["rp_copied"] == 0, ref[3]
            for mode in ("whole", "union", "auto"):
                ys, cache, arena, st = run(out, mode, group)
                assert same(ys, ref[0]), f"logits differ ({mode}, group {group})"
                assert same(cache, ref[1]), f"cache arrays differ ({mode}, group {group})"
                for (lru, slots), (lru0, slots0) in zip(arena, ref[2]):
                    assert lru == lru0 and np.array_equal(slots, slots0), \
                        f"arena differs ({mode}, group {group})"
                assert st["rp_b"] < ref[3]["rp_b"], (st, ref[3])
                # A split pass's union is its misses: nothing to copy there.
                assert st["rp_copied"] > 0 or (mode == "union" and group == 32), st
                if mode == "union":
                    assert st["rp_whole"] == 0, st
                if group == 32:
                    assert st["wide_split"] > 0, st
                print(f"   group {group}, {mode}: {len(ys)} passes' logits, {len(cache)} cache "
                      f"arrays, {len(arena)} arenas == main's path; {st['rp_union']} union / "
                      f"{st['rp_whole']} read-ahead layers, {st['rp_copied']} rows copied, "
                      f"{st['rp_b'] / 1e6:.2f} MB read (main's path: {ref[3]['rp_b'] / 1e6:.2f} MB)")
        check_stripe(d, out, refs)
    stream_fast._RP_MODE, stream_fast.WIDE_GROUP = "auto", 128
    print("test_readpath: all passed")


if __name__ == "__main__":
    main()
