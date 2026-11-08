#!/usr/bin/env python3
"""Layer-major prefill read path (stream_fast._LMReader): a layer is read
whole ahead of the GPU, or, once it has routed, only its union, the store's
residents at the pass's start copied instead of read. Either way a group's
buffer holds its experts in order and the gathers see the same rows, so on
test_layer_prefill's toy Qwen3-Next (128 experts here, two expert shapes,
MOE_LM_PREFILL=1) every pass's logits and the cache arrays are bit-equal
whichever way the layers are read. The reference is main's path
(MOE_READPATH=whole: every layer whole, nothing copied), against forced
union reads and the measured choice, at cache bias 0 and 0.5. Passes: a
fresh 20-id prefill (layer-major, the store empty), decode tokens that fill
the store, a second conversation's fresh 24-id prefill (layer-major, with
residents to copy), decode, then 130 ids on that cache (layer-major and wide
enough for stock's sorted kernel). Union reads ask the drives for fewer
bytes and copy residents.

With a sidecar (mirror_experts.py) of every expert, then of half, union
and whole reads are the same bits and part of their bytes go to the sidecar,
also with every record landing 0-2 ms late (out of order) and with the
pipeline off (MOE_LM_PIPE=0: a barrier per group).

Mutants: a union read that leaves its last expert unread, a copy from
another slot, and a group handed to the GPU before its records landed are
all caught.

    perl -e 'alarm shift; exec @ARGV' 280 python3 tests/test_union_lm.py
"""
import contextlib
import faulthandler
import gc
import os
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

import stream_fast
import test_layer_prefill as tlp

faulthandler.dump_traceback_later(270, exit=True)

tlp.CFG = dict(tlp.CFG, num_experts=128)
KEYS = ("lm_read_b", "lm_whole", "lm_union", "lm_copied", "stripe_ext_b")


def run(out, mode, cache_bias=0.0):
    stream_fast._RP_MODE = mode
    stream_fast.LM_W = 64
    model, bud = tlp.engine(out, "1", cache_bias)
    bud.plan = lambda n, fresh, caches: n           # no cut: the same pieces in every arm
    rng, ys = np.random.default_rng(7), []
    s0 = {k: stream_fast.STATS[k] for k in KEYS}

    def feed(cache, n):
        ys.append(model(mx.array(rng.integers(0, 200, n))[None], cache=cache))
        mx.eval(ys[-1])

    try:
        a = make_prompt_cache(model)
        feed(a, 20)
        for _ in range(6):
            feed(a, 1)
        b = make_prompt_cache(model)
        feed(b, 24)
        for _ in range(4):
            feed(b, 1)
        feed(b, 130)
        feed(b, 1)
    finally:
        stream_fast._RP_MODE = "auto"
    st = {k: stream_fast.STATS[k] - s0[k] for k in KEYS}
    got = ys, tlp.arrays(a) + tlp.arrays(b), st
    del model, bud, a, b
    gc.collect()
    mx.clear_cache()
    return got


def check(ref, got, what):
    assert tlp.same(got[0], ref[0]), f"logits differ ({what})"
    assert tlp.same(got[1], ref[1]), f"cache arrays differ ({what})"


@contextlib.contextmanager
def late():
    """Every record read 0-2 ms late, so they land out of order."""
    rd, ext = stream_fast._do_read, stream_fast._Stripe.read
    rng = random.Random(1)

    def do_read(job):
        time.sleep(rng.random() * 2e-3)
        return rd(job)

    def ext_read(self, job, eoff):
        time.sleep(rng.random() * 2e-3)
        return ext(self, job, eoff)
    stream_fast._do_read, stream_fast._Stripe.read = do_read, ext_read
    try:
        yield
    finally:
        stream_fast._do_read, stream_fast._Stripe.read = rd, ext


def mutant(out, ref, name, patch):
    real = {k: getattr(stream_fast._LMReader, k) for k in patch}
    for k, f in patch.items():
        setattr(stream_fast._LMReader, k, f)
    try:
        with late():
            got = run(out, "union")
    finally:
        for k, f in real.items():
            setattr(stream_fast._LMReader, k, f)
    assert not (tlp.same(got[0], ref[0]) and tlp.same(got[1], ref[1])), f"mutant not caught: {name}"
    print(f"   mutant caught: {name}")


def rate_seed():
    """stream_fast._lm_ema: a path's first sample (its cold first read) only starts its rate, the
    second replaces it, later ones are averaged in."""
    rp = stream_fast._LMRP
    keep = rp["rate_u"], rp["n"]
    rp["rate_u"], rp["n"] = None, {}
    try:
        for v in (0.08, 2.26, 3.0):
            stream_fast._lm_ema("rate_u", v)
        assert abs(rp["rate_u"] - (2.26 + 0.3 * (3.0 - 2.26))) < 1e-12, rp["rate_u"]
    finally:
        rp["rate_u"], rp["n"] = keep
    print("   rate seed: the first sample is replaced by the second")


def main():
    rate_seed()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "q3n"
        tlp.build(out)
        for bias in (0.0, 0.5):
            # MOE_STORE_KEEP puts back the residents a prefill did not overwrite: whole reads overwrite
            # them all, union reads few, so at bias 0.5 the decode after routes on another residency.
            stream_fast._KEEP = bias == 0.0
            ref = run(out, "whole", bias)
            assert ref[2]["lm_union"] == 0 and ref[2]["lm_copied"] == 0, ref[2]
            for mode in ("union", "auto"):
                got = run(out, mode, bias)
                check(ref, got, f"{mode}, bias {bias}")
                st = got[2]
                assert st["lm_read_b"] < ref[2]["lm_read_b"] and st["lm_union"] > 0, (st, ref[2])
                assert st["lm_copied"] > 0, st
                if mode == "union":
                    assert st["lm_whole"] == 0, st
                print(f"   bias {bias}, {mode}: {len(got[0])} passes' logits, {len(got[1])} cache "
                      f"arrays == main's path; {st['lm_union']} union / {st['lm_whole']} whole "
                      f"layers, {st['lm_copied']} rows copied, {st['lm_read_b'] / 1e6:.2f} MB read "
                      f"(main's path {ref[2]['lm_read_b'] / 1e6:.2f} MB)")
        stream_fast._KEEP = True
        ref = run(out, "whole")

        def skip_last(self, glu, ids, r0, whole):
            return real_read(self, glu, ids[:-1], r0, whole) if not whole and len(ids) > 1 \
                else real_read(self, glu, ids, r0, whole)

        def wrong_slot(self, glu, e):
            sl = real_res(self, glu, e)
            others = [s for s in self.res.get(id(glu), {}).values() if s != sl]
            return others[0] if sl is not None and others else sl
        def early(self, key, r0, jobs):
            real_put(self, key, r0, jobs)
            with self.cv:
                self.done[key] = r0
                self.cv.notify_all()
        real_read, real_res = stream_fast._LMReader._read, stream_fast._LMReader._resident
        real_put = stream_fast._LMReader._put
        mutant(out, ref, "a union read leaves its last expert unread", {"_read": skip_last})
        mutant(out, ref, "a resident copied from another slot", {"_resident": wrong_slot})
        mutant(out, ref, "a group handed over before its records landed", {"_put": early})

        import mirror_experts
        for frac in (1.0, 0.5):
            drive = Path(d) / f"drive{frac}"
            mirror_experts.main(["--model", str(out), "--drive", str(drive), "--seconds", "0.05",
                                 "--fraction", str(frac), "--min-margin-gb=-1e9",
                                 "--max-swap-mb", "1e9"])
            os.environ["MOE_STRIPE"] = str(drive / out.name / mirror_experts.META)
            try:
                for mode, pipe, slow in (("union", True, False), ("whole", True, False),
                                         ("union", True, True), ("whole", True, True),
                                         ("union", False, False), ("whole", False, False)):
                    stream_fast._LM_PIPE = pipe
                    with late() if slow else contextlib.nullcontext():
                        got = run(out, mode)
                    what = f"sidecar {frac}, {mode}{', late' if slow else ''}{'' if pipe else ', pipe off'}"
                    check(ref, got, what)
                    assert got[2]["stripe_ext_b"] > 0, got[2]
                    print(f"   {what}: == main's path; {got[2]['stripe_ext_b'] / 1e6:.2f} of "
                          f"{got[2]['lm_read_b'] / 1e6:.2f} MB from the sidecar")
            finally:
                stream_fast._LM_PIPE = True
                os.environ.pop("MOE_STRIPE")
                stream_fast.open_stripe(out, None)      # closes it
    print("test_union_lm: all passed")


if __name__ == "__main__":
    main()
